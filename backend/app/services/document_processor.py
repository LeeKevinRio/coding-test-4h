"""
Document Processing Service

Extracts text, images, and tables from PDF documents.
Supports multiple backends:
- Docling (if available) - best quality
- PyPDF2 + pdfplumber (fallback) - works without Docling
"""
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from app.models.document import Document, DocumentChunk, DocumentImage, DocumentTable
from app.services.vector_store import VectorStore
import os
import time
from datetime import datetime

# Try to import PDF processing libraries
DOCLING_AVAILABLE = False
PYPDF_AVAILABLE = False
PDFPLUMBER_AVAILABLE = False

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
    print("Using Docling for PDF processing")
except ImportError:
    pass

try:
    import PyPDF2
    PYPDF_AVAILABLE = True
except ImportError:
    pass

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    pass

if not DOCLING_AVAILABLE and not PYPDF_AVAILABLE and not PDFPLUMBER_AVAILABLE:
    print("Warning: No PDF library available. Install with: pip install PyPDF2 pdfplumber")


class DocumentProcessor:
    """
    Process PDF documents and extract multimodal content.
    
    Uses Docling if available, falls back to PyPDF2/pdfplumber.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.vector_store = VectorStore(db)
        self.converter = None
        
        if DOCLING_AVAILABLE:
            try:
                self.converter = DocumentConverter(allowed_formats=[InputFormat.PDF])
            except Exception as e:
                print(f"Docling init error: {e}")
    
    async def process_document(self, file_path: str, document_id: int) -> Dict[str, Any]:
        """
        Process a PDF document and extract content.
        
        Args:
            file_path: Path to the PDF file
            document_id: Database ID of the document
            
        Returns:
            Processing result with counts and status
        """
        start_time = time.time()
        
        try:
            # Update status to processing
            await self._update_document_status(document_id, "processing")
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"PDF file not found: {file_path}")
            
            # Extract content based on available library
            if DOCLING_AVAILABLE and self.converter:
                result = await self._process_with_docling(file_path, document_id)
            elif PDFPLUMBER_AVAILABLE:
                result = await self._process_with_pdfplumber(file_path, document_id)
            elif PYPDF_AVAILABLE:
                result = await self._process_with_pypdf(file_path, document_id)
            else:
                raise RuntimeError("No PDF processing library available")
            
            # Update status to completed
            await self._update_document_status(document_id, "completed")
            
            processing_time = time.time() - start_time
            
            return {
                "status": "success",
                "text_chunks": result.get("chunks", 0),
                "images": result.get("images", 0),
                "tables": result.get("tables", 0),
                "processing_time": round(processing_time, 2)
            }
            
        except Exception as e:
            print(f"Error processing document: {e}")
            import traceback
            traceback.print_exc()
            
            await self._update_document_status(document_id, "error", str(e))
            
            return {
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def _process_with_pdfplumber(self, file_path: str, document_id: int) -> Dict[str, int]:
        """Process PDF using pdfplumber (good table extraction)."""
        import pdfplumber
        
        chunks_count = 0
        tables_count = 0
        all_chunks = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract text
                text = page.extract_text() or ""
                if text.strip():
                    page_chunks = self._chunk_text(text, page_num)
                    all_chunks.extend(page_chunks)
                
                # Extract tables
                tables = page.extract_tables()
                for tbl_idx, table in enumerate(tables):
                    if table:
                        await self._save_table(
                            table_data=table,
                            document_id=document_id,
                            page_number=page_num,
                            table_index=tbl_idx
                        )
                        tables_count += 1
        
        # Save chunks with embeddings
        if all_chunks:
            chunks_count = await self.vector_store.store_text_chunks(all_chunks, document_id)
        
        return {"chunks": chunks_count, "images": 0, "tables": tables_count}
    
    async def _process_with_pypdf(self, file_path: str, document_id: int) -> Dict[str, int]:
        """Process PDF using PyPDF2 (basic text extraction)."""
        import PyPDF2
        
        chunks_count = 0
        all_chunks = []
        
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    page_chunks = self._chunk_text(text, page_num)
                    all_chunks.extend(page_chunks)
        
        # Save chunks with embeddings
        if all_chunks:
            chunks_count = await self.vector_store.store_text_chunks(all_chunks, document_id)
        
        return {"chunks": chunks_count, "images": 0, "tables": 0}
    
    async def _process_with_docling(self, file_path: str, document_id: int) -> Dict[str, int]:
        """Process PDF using Docling (best quality)."""
        result = self.converter.convert(file_path)
        doc = result.document
        
        chunks_count = 0
        images_count = 0
        tables_count = 0
        all_chunks = []
        
        # Extract text by pages
        for page_num, page in enumerate(doc.pages, start=1):
            page_text = self._extract_docling_page_text(page)
            if page_text.strip():
                page_chunks = self._chunk_text(page_text, page_num)
                all_chunks.extend(page_chunks)
        
        # Save chunks
        if all_chunks:
            chunks_count = await self.vector_store.store_text_chunks(all_chunks, document_id)
        
        return {"chunks": chunks_count, "images": images_count, "tables": tables_count}
    
    def _extract_docling_page_text(self, page) -> str:
        """Extract text from a Docling page object."""
        texts = []
        if hasattr(page, 'text_blocks'):
            for block in page.text_blocks:
                if hasattr(block, 'text'):
                    texts.append(block.text)
        elif hasattr(page, 'text'):
            texts.append(page.text)
        return "\n".join(texts)
    
    def _chunk_text(
        self,
        text: str,
        page_number: int,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            page_number: Page number for metadata
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        
        # Clean text
        text = text.strip()
        if not text:
            return chunks
        
        # Split into sentences (rough approximation)
        sentences = text.replace('\n', ' ').split('. ')
        
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Add period back if not present
            if not sentence.endswith('.'):
                sentence += '.'
            
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "page_number": page_number,
                        "metadata": {"page": page_number}
                    })
                current_chunk = sentence
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append({
                "content": current_chunk.strip(),
                "page_number": page_number,
                "metadata": {"page": page_number}
            })
        
        return chunks
    
    async def _save_table(
        self,
        table_data: List[List[Any]],
        document_id: int,
        page_number: int,
        table_index: int = 0
    ) -> Optional[DocumentTable]:
        """Save extracted table to database."""
        try:
            # Convert table to structured format
            if not table_data or len(table_data) < 1:
                return None
            
            headers = table_data[0] if table_data else []
            rows = table_data[1:] if len(table_data) > 1 else []
            
            table = DocumentTable(
                document_id=document_id,
                page_number=page_number,
                table_metadata={
                    "headers": headers,
                    "rows": rows,
                    "table_index": table_index
                },
                data={"headers": headers, "rows": rows},
                rows=len(rows),
                columns=len(headers)
            )
            
            self.db.add(table)
            self.db.commit()
            
            return table
            
        except Exception as e:
            print(f"Error saving table: {e}")
            self.db.rollback()
            return None
    
    async def _update_document_status(
        self,
        document_id: int,
        status: str,
        error_message: str = None
    ):
        """Update document processing status."""
        try:
            doc = self.db.query(Document).filter(Document.id == document_id).first()
            if doc:
                doc.status = status
                if error_message:
                    doc.error_message = error_message
                doc.updated_at = datetime.utcnow()
                self.db.commit()
        except Exception as e:
            print(f"Error updating status: {e}")
            self.db.rollback()
    
    async def get_document_content(self, document_id: int) -> Dict[str, Any]:
        """Get all extracted content for a document."""
        try:
            chunks = self.db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).all()
            
            images = self.db.query(DocumentImage).filter(
                DocumentImage.document_id == document_id
            ).all()
            
            tables = self.db.query(DocumentTable).filter(
                DocumentTable.document_id == document_id
            ).all()
            
            return {
                "chunks": [
                    {
                        "id": c.id,
                        "content": c.content,
                        "page_number": c.page_number
                    }
                    for c in chunks
                ],
                "images": [
                    {
                        "id": i.id,
                        "file_path": i.file_path,
                        "page_number": i.page_number,
                        "caption": i.caption
                    }
                    for i in images
                ],
                "tables": [
                    {
                        "id": t.id,
                        "page_number": t.page_number,
                        "data": t.data
                    }
                    for t in tables
                ]
            }
        except Exception as e:
            print(f"Error getting content: {e}")
            return {"chunks": [], "images": [], "tables": []}

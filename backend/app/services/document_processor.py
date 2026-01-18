"""
Document processing service using Docling

TODO: Implement this service to:
1. Parse PDF documents using Docling
2. Extract text, images, and tables
3. Store extracted content in database
4. Generate embeddings for text chunks
"""
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from app.models.document import Document, DocumentChunk, DocumentImage, DocumentTable
from app.services.vector_store import VectorStore
import os
import hashlib
import json
import time
from datetime import datetime

# Docling imports
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    print("Warning: Docling not installed. Install with: pip install docling")


class DocumentProcessor:
    """
    Process PDF documents and extract multimodal content.
    
    This is a SKELETON implementation. You need to implement the core logic.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.vector_store = VectorStore(db)
        
        # Initialize Docling converter with optimal settings
        if DOCLING_AVAILABLE:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            pipeline_options.generate_picture_images = True
            
            self.converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF],
                pipeline_options=pipeline_options
            )
        else:
            self.converter = None
        
        # Chunking configuration
        self.chunk_size = 512  # tokens approximately
        self.chunk_overlap = 50  # overlap for context continuity
        
        # Storage paths
        self.upload_dir = os.getenv("UPLOAD_DIR", "/app/uploads")
        self.image_dir = os.path.join(self.upload_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)
    
    async def process_document(self, file_path: str, document_id: int) -> Dict[str, Any]:
        """
        Process a PDF document using Docling.
        
        Implementation steps:
        1. Update document status to 'processing'
        2. Use Docling to parse the PDF
        3. Extract and save text chunks
        4. Extract and save images
        5. Extract and save tables
        6. Generate embeddings for text chunks
        7. Update document status to 'completed'
        8. Handle errors appropriately
        
        Args:
            file_path: Path to the uploaded PDF file
            document_id: Database ID of the document
            
        Returns:
            {
                "status": "success" or "error",
                "text_chunks": <count>,
                "images": <count>,
                "tables": <count>,
                "processing_time": <seconds>
            }
        """
        start_time = time.time()
        
        try:
            # Step 1: Update status to processing
            await self._update_document_status(document_id, "processing")
            
            if not DOCLING_AVAILABLE:
                raise RuntimeError("Docling is not installed")
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"PDF file not found: {file_path}")
            
            # Step 2: Parse PDF using Docling
            result = self.converter.convert(file_path)
            doc = result.document
            
            # Initialize counters
            text_chunks_count = 0
            images_count = 0
            tables_count = 0
            
            # Step 3: Extract and chunk text by pages
            all_chunks = []
            for page_num, page in enumerate(doc.pages, start=1):
                page_text = self._extract_page_text(page)
                if page_text.strip():
                    chunks = self._chunk_text(page_text, document_id, page_num)
                    all_chunks.extend(chunks)
            
            # Step 4: Save text chunks with embeddings
            if all_chunks:
                text_chunks_count = await self._save_text_chunks(all_chunks, document_id)
            
            # Step 5: Extract and save images
            for page_num, page in enumerate(doc.pages, start=1):
                if hasattr(page, 'images') and page.images:
                    for img_idx, image in enumerate(page.images):
                        saved_image = await self._save_image(
                            image_data=image.image if hasattr(image, 'image') else None,
                            document_id=document_id,
                            page_number=page_num,
                            metadata=self._extract_image_metadata(image, img_idx)
                        )
                        if saved_image:
                            images_count += 1
            
            # Step 6: Extract and save tables
            for page_num, page in enumerate(doc.pages, start=1):
                if hasattr(page, 'tables') and page.tables:
                    for tbl_idx, table in enumerate(page.tables):
                        saved_table = await self._save_table(
                            table_data=self._extract_table_data(table),
                            document_id=document_id,
                            page_number=page_num,
                            metadata=self._extract_table_metadata(table, tbl_idx)
                        )
                        if saved_table:
                            tables_count += 1
            
            # Step 7: Update document with final counts
            processing_time = time.time() - start_time
            await self._update_document_status(
                document_id, 
                "completed",
                total_pages=len(doc.pages),
                text_chunks_count=text_chunks_count,
                images_count=images_count,
                tables_count=tables_count
            )
            
            return {
                "status": "success",
                "text_chunks": text_chunks_count,
                "images": images_count,
                "tables": tables_count,
                "processing_time": processing_time
            }
            
        except Exception as e:
            # Step 8: Handle errors
            processing_time = time.time() - start_time
            error_msg = str(e)
            await self._update_document_status(document_id, "error", error_message=error_msg)
            
            return {
                "status": "error",
                "error": error_msg,
                "text_chunks": 0,
                "images": 0,
                "tables": 0,
                "processing_time": processing_time
            }
    
    def _extract_page_text(self, page) -> str:
        """Extract text content from a Docling page object."""
        text_parts = []
        
        # Try different ways to access text based on Docling's structure
        if hasattr(page, 'text'):
            text_parts.append(page.text)
        elif hasattr(page, 'body'):
            if hasattr(page.body, 'text'):
                text_parts.append(page.body.text)
            elif isinstance(page.body, list):
                for item in page.body:
                    if hasattr(item, 'text'):
                        text_parts.append(item.text)
        
        # Also check for paragraphs
        if hasattr(page, 'paragraphs'):
            for para in page.paragraphs:
                if hasattr(para, 'text'):
                    text_parts.append(para.text)
        
        return "\n".join(text_parts)
    
    def _chunk_text(self, text: str, document_id: int, page_number: int) -> List[Dict[str, Any]]:
        """
        Split text into chunks for vector storage.
        
        TODO: Implement text chunking strategy
        - Split by sentences or paragraphs
        - Maintain context with overlap
        - Keep metadata (page number, position, etc.)
        
        Returns:
            List of chunk dictionaries with content and metadata
        """
        chunks = []
        
        # Strategy: Split by paragraphs first, then by sentences if too long
        paragraphs = text.split('\n\n')
        
        chunk_index = 0
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If paragraph fits in current chunk
            if len(current_chunk) + len(para) + 1 <= self.chunk_size * 4:  # ~4 chars per token
                current_chunk += (" " if current_chunk else "") + para
            else:
                # Save current chunk if not empty
                if current_chunk:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "document_id": document_id,
                        "page_number": page_number,
                        "chunk_index": chunk_index,
                        "metadata": {
                            "char_count": len(current_chunk),
                            "approx_tokens": len(current_chunk) // 4
                        }
                    })
                    chunk_index += 1
                
                # Start new chunk with overlap
                # Get last few sentences for overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + para
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append({
                "content": current_chunk.strip(),
                "document_id": document_id,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "metadata": {
                    "char_count": len(current_chunk),
                    "approx_tokens": len(current_chunk) // 4
                }
            })
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get the last portion of text for overlap."""
        if not text:
            return ""
        
        # Get approximately chunk_overlap tokens worth of text
        overlap_chars = self.chunk_overlap * 4
        if len(text) <= overlap_chars:
            return text + " "
        
        # Try to split at sentence boundary
        last_part = text[-overlap_chars:]
        sentence_end = last_part.find('. ')
        if sentence_end != -1:
            return last_part[sentence_end + 2:] + " "
        
        return last_part + " "
    
    async def _save_text_chunks(self, chunks: List[Dict[str, Any]], document_id: int) -> int:
        """
        Save text chunks to database with embeddings.
        
        TODO: Implement chunk storage
        - Generate embeddings
        - Store in database
        - Link related images/tables in metadata
        """
        saved_count = 0
        
        for chunk in chunks:
            try:
                # Use vector store to save chunk with embedding
                saved_chunk = await self.vector_store.store_chunk(
                    content=chunk["content"],
                    document_id=chunk["document_id"],
                    page_number=chunk["page_number"],
                    chunk_index=chunk["chunk_index"],
                    metadata=chunk.get("metadata", {})
                )
                if saved_chunk:
                    saved_count += 1
            except Exception as e:
                print(f"Error saving chunk: {e}")
                continue
        
        return saved_count
    
    async def _save_image(
        self,
        image_data: bytes,
        document_id: int,
        page_number: int,
        metadata: Dict[str, Any]
    ) -> DocumentImage:
        """
        Save an extracted image.
        
        TODO: Implement image saving
        - Save image file to disk
        - Create DocumentImage record
        - Extract caption if available
        """
        if image_data is None:
            return None
        
        try:
            # Generate unique filename
            image_hash = hashlib.md5(image_data).hexdigest()[:12]
            filename = f"doc_{document_id}_page_{page_number}_{image_hash}.png"
            file_path = os.path.join(self.image_dir, filename)
            
            # Save image file
            with open(file_path, 'wb') as f:
                f.write(image_data)
            
            # Create database record
            doc_image = DocumentImage(
                document_id=document_id,
                file_path=file_path,
                page_number=page_number,
                caption=metadata.get("caption", ""),
                width=metadata.get("width", 0),
                height=metadata.get("height", 0),
                image_metadata=metadata
            )
            
            self.db.add(doc_image)
            self.db.commit()
            self.db.refresh(doc_image)
            
            return doc_image
            
        except Exception as e:
            print(f"Error saving image: {e}")
            self.db.rollback()
            return None
    
    def _extract_image_metadata(self, image, index: int) -> Dict[str, Any]:
        """Extract metadata from a Docling image object."""
        metadata = {"index": index}
        
        if hasattr(image, 'bbox'):
            metadata["bbox"] = list(image.bbox) if hasattr(image.bbox, '__iter__') else None
        if hasattr(image, 'caption'):
            metadata["caption"] = image.caption
        if hasattr(image, 'width'):
            metadata["width"] = image.width
        if hasattr(image, 'height'):
            metadata["height"] = image.height
        
        return metadata
    
    async def _save_table(
        self,
        table_data: Any,
        document_id: int,
        page_number: int,
        metadata: Dict[str, Any]
    ) -> DocumentTable:
        """
        Save an extracted table.
        
        TODO: Implement table saving
        - Render table as image
        - Store structured data as JSON
        - Create DocumentTable record
        - Extract caption if available
        """
        try:
            # Generate table image path
            table_hash = hashlib.md5(json.dumps(table_data, default=str).encode()).hexdigest()[:12]
            image_filename = f"table_doc_{document_id}_page_{page_number}_{table_hash}.png"
            image_path = os.path.join(self.image_dir, image_filename)
            
            # Create database record
            doc_table = DocumentTable(
                document_id=document_id,
                image_path=image_path,
                data=table_data,
                page_number=page_number,
                caption=metadata.get("caption", ""),
                rows=metadata.get("rows", 0),
                columns=metadata.get("columns", 0),
                table_metadata=metadata
            )
            
            self.db.add(doc_table)
            self.db.commit()
            self.db.refresh(doc_table)
            
            return doc_table
            
        except Exception as e:
            print(f"Error saving table: {e}")
            self.db.rollback()
            return None
    
    def _extract_table_data(self, table) -> Dict[str, Any]:
        """Extract structured data from a Docling table object."""
        data = {"rows": [], "headers": []}
        
        if hasattr(table, 'data'):
            if isinstance(table.data, list):
                data["rows"] = table.data
            elif hasattr(table.data, 'rows'):
                data["rows"] = list(table.data.rows)
        
        if hasattr(table, 'headers'):
            data["headers"] = list(table.headers)
        
        if hasattr(table, 'to_dataframe'):
            try:
                df = table.to_dataframe()
                data["headers"] = df.columns.tolist()
                data["rows"] = df.values.tolist()
            except:
                pass
        
        return data
    
    def _extract_table_metadata(self, table, index: int) -> Dict[str, Any]:
        """Extract metadata from a Docling table object."""
        metadata = {"index": index}
        
        if hasattr(table, 'bbox'):
            metadata["bbox"] = list(table.bbox) if hasattr(table.bbox, '__iter__') else None
        if hasattr(table, 'caption'):
            metadata["caption"] = table.caption
        if hasattr(table, 'num_rows'):
            metadata["rows"] = table.num_rows
        if hasattr(table, 'num_cols'):
            metadata["columns"] = table.num_cols
        
        return metadata
    
    async def _update_document_status(
        self,
        document_id: int,
        status: str,
        error_message: str = None,
        total_pages: int = None,
        text_chunks_count: int = None,
        images_count: int = None,
        tables_count: int = None
    ):
        """
        Update document processing status.
        
        This is implemented as an example.
        """
        document = self.db.query(Document).filter(Document.id == document_id).first()
        if document:
            document.processing_status = status
            if error_message:
                document.error_message = error_message
            if total_pages is not None:
                document.total_pages = total_pages
            if text_chunks_count is not None:
                document.text_chunks_count = text_chunks_count
            if images_count is not None:
                document.images_count = images_count
            if tables_count is not None:
                document.tables_count = tables_count
            
            self.db.commit()

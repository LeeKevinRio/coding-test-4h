"""
Vector Store Service with pgvector Integration

This module handles:
1. Text embedding generation (multiple backends supported)
2. Vector storage in PostgreSQL with pgvector
3. Similarity search for RAG retrieval
4. Related content linking (images, tables)

Embedding backends supported:
- sentence-transformers (free, local, default)
- OpenAI text-embedding-3-small (requires API key)
"""
from typing import List, Dict, Any, Optional
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.models.document import DocumentChunk, DocumentImage, DocumentTable
from app.core.config import settings
import json


class VectorStore:
    """
    Vector store for document embeddings and similarity search.
    
    This class provides:
    - Multi-backend embedding support (sentence-transformers, OpenAI)
    - Efficient batch storage with pgvector
    - Cosine similarity search
    - Related content retrieval (images/tables linked to chunks)
    """
    
    def __init__(self, db: Session):
        """Initialize vector store with database session."""
        self.db = db
        self.embedding_model = None
        self.embedding_provider = None
        self.embedding_dim = 384  # Default for sentence-transformers
        
        self._ensure_extension()
        self._initialize_embedding_model()
    
    def _ensure_extension(self):
        """Ensure pgvector extension is enabled in PostgreSQL."""
        try:
            self.db.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            self.db.commit()
        except Exception as e:
            print(f"pgvector extension already exists or error: {e}")
            self.db.rollback()
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model with fallback chain."""
        # Try sentence-transformers first (free, local)
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_provider = "sentence-transformers"
            self.embedding_dim = 384
            print("Using sentence-transformers for embeddings")
            return
        except ImportError:
            pass
        except Exception as e:
            print(f"sentence-transformers error: {e}")
        
        # Try OpenAI
        try:
            import openai
            import os
            api_key = os.getenv("OPENAI_API_KEY") or getattr(settings, 'OPENAI_API_KEY', None)
            if api_key and api_key.strip() and not api_key.startswith('sk-your'):
                self.embedding_model = openai.OpenAI(api_key=api_key)
                self.embedding_provider = "openai"
                self.embedding_dim = 1536
                print("Using OpenAI for embeddings")
                return
        except ImportError:
            pass
        except Exception as e:
            print(f"OpenAI embedding error: {e}")
        
        print("Warning: No embedding model available!")
        self.embedding_model = None
        self.embedding_provider = None
    
    def _generate_embedding(self, text_input: str) -> Optional[List[float]]:
        """Generate embedding vector for text."""
        if self.embedding_model is None:
            return None
        
        try:
            if self.embedding_provider == "sentence-transformers":
                embedding = self.embedding_model.encode(text_input)
                return embedding.tolist()
            
            elif self.embedding_provider == "openai":
                response = self.embedding_model.embeddings.create(
                    model="text-embedding-3-small",
                    input=text_input
                )
                return response.data[0].embedding
                
        except Exception as e:
            print(f"Embedding error: {e}")
            return None
    
    async def store_text_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_id: int
    ) -> int:
        """Store text chunks with embeddings in the vector store."""
        stored_count = 0
        
        for chunk_index, chunk in enumerate(chunks):
            try:
                content = chunk.get("content", "")
                page_number = chunk.get("page_number", 1)
                chunk_metadata = chunk.get("metadata", {})
                
                # Generate embedding
                embedding = self._generate_embedding(content)
                if embedding is None:
                    print(f"Failed to generate embedding for chunk {chunk_index}")
                    continue
                
                # Create DocumentChunk record
                doc_chunk = DocumentChunk(
                    document_id=document_id,
                    content=content,
                    embedding=embedding,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    chunk_metadata=chunk_metadata if isinstance(chunk_metadata, dict) else {}
                )
                
                self.db.add(doc_chunk)
                stored_count += 1
                
            except Exception as e:
                print(f"Error storing chunk {chunk_index}: {e}")
                continue
        
        try:
            self.db.commit()
        except Exception as e:
            print(f"Error committing chunks: {e}")
            self.db.rollback()
            return 0
        
        return stored_count
    
    async def search_similar(
        self,
        query: str,
        document_id: Optional[int] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar text chunks using vector similarity."""
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        if query_embedding is None:
            print("Failed to generate query embedding")
            return []
        
        try:
            # Format embedding as PostgreSQL array literal
            embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
            
            # Use raw SQL with proper escaping for pgvector
            if document_id is not None:
                sql = f"""
                    SELECT
                        id,
                        content,
                        page_number,
                        chunk_metadata,
                        chunk_index,
                        1 - (embedding <=> '{embedding_str}'::vector) as similarity
                    FROM document_chunks
                    WHERE document_id = {document_id}
                    ORDER BY embedding <=> '{embedding_str}'::vector
                    LIMIT {k}
                """
            else:
                sql = f"""
                    SELECT
                        id,
                        content,
                        page_number,
                        chunk_metadata,
                        chunk_index,
                        document_id,
                        1 - (embedding <=> '{embedding_str}'::vector) as similarity
                    FROM document_chunks
                    ORDER BY embedding <=> '{embedding_str}'::vector
                    LIMIT {k}
                """
            
            result = self.db.execute(text(sql))
            
            # Format results
            results = []
            for row in result:
                chunk_metadata = row.chunk_metadata if row.chunk_metadata else {}
                
                results.append({
                    "id": row.id,
                    "content": row.content,
                    "page_number": row.page_number,
                    "chunk_index": row.chunk_index,
                    "score": float(row.similarity) if row.similarity else 0.0,
                    "metadata": chunk_metadata
                })
            
            return results
            
        except Exception as e:
            print(f"Search error: {e}")
            import traceback
            traceback.print_exc()
            self.db.rollback()
            return []
    
    async def get_related_content(
        self,
        chunk_ids: List[int],
        document_id: Optional[int] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get images and tables related to the given chunks."""
        images = []
        tables = []
        
        try:
            # Get chunks to find their pages
            chunks = self.db.query(DocumentChunk).filter(
                DocumentChunk.id.in_(chunk_ids)
            ).all()
            
            # Collect page numbers
            page_numbers = set()
            for chunk in chunks:
                if chunk.page_number:
                    page_numbers.add(chunk.page_number)
            
            # Query images by page proximity
            if page_numbers:
                image_query = self.db.query(DocumentImage).filter(
                    DocumentImage.page_number.in_(page_numbers)
                )
                if document_id:
                    image_query = image_query.filter(
                        DocumentImage.document_id == document_id
                    )
                
                for img in image_query.limit(10).all():
                    images.append({
                        "id": img.id,
                        "file_path": img.file_path,
                        "page_number": img.page_number,
                        "caption": img.caption,
                        "width": img.width,
                        "height": img.height
                    })
            
            # Query tables by page proximity
            if page_numbers:
                table_query = self.db.query(DocumentTable).filter(
                    DocumentTable.page_number.in_(page_numbers)
                )
                if document_id:
                    table_query = table_query.filter(
                        DocumentTable.document_id == document_id
                    )
                
                for tbl in table_query.limit(10).all():
                    tables.append({
                        "id": tbl.id,
                        "image_path": tbl.image_path,
                        "page_number": tbl.page_number,
                        "caption": tbl.caption,
                        "data": tbl.data,
                        "rows": tbl.rows,
                        "columns": tbl.columns
                    })
                    
        except Exception as e:
            print(f"Error getting related content: {e}")
        
        return {"images": images, "tables": tables}
    
    async def delete_document_vectors(self, document_id: int) -> bool:
        """Delete all vectors for a document."""
        try:
            self.db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).delete()
            self.db.commit()
            return True
        except Exception as e:
            print(f"Error deleting vectors: {e}")
            self.db.rollback()
            return False
    
    async def get_document_stats(self, document_id: int) -> Dict[str, int]:
        """Get statistics about stored vectors for a document."""
        try:
            chunk_count = self.db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).count()
            
            image_count = self.db.query(DocumentImage).filter(
                DocumentImage.document_id == document_id
            ).count()
            
            table_count = self.db.query(DocumentTable).filter(
                DocumentTable.document_id == document_id
            ).count()
            
            return {
                "chunks": chunk_count,
                "images": image_count,
                "tables": table_count
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"chunks": 0, "images": 0, "tables": 0}

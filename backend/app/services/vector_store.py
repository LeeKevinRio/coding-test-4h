"""
Vector store service using pgvector.

TODO: Implement this service to:
1. Generate embeddings for text chunks
2. Store embeddings in PostgreSQL with pgvector
3. Perform similarity search
4. Link related images and tables
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
    
    This is a SKELETON implementation. You need to implement the core logic.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.embeddings_model = None  # TODO: Initialize embedding model
        self._ensure_extension()
        self._initialize_embedding_model()
    
    def _ensure_extension(self):
        """
        Ensure pgvector extension is enabled.
        
        This is implemented as an example.
        """
        try:
            self.db.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            self.db.commit()
        except Exception as e:
            print(f"pgvector extension already exists or error: {e}")
            self.db.rollback()
    
    def _initialize_embedding_model(self):
        """
        Initialize the embedding model.
        
        Supports multiple backends:
        - OpenAI (requires API key)
        - HuggingFace sentence-transformers (local, free)
        - Ollama (local, free)
        """
        # Try HuggingFace sentence-transformers first (free, local)
        try:
            from sentence_transformers import SentenceTransformer
            # all-MiniLM-L6-v2 is a good balance of speed and quality
            # Produces 384-dimensional embeddings
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = 384
            self.embedding_provider = "sentence-transformers"
            print("Using sentence-transformers for embeddings")
            return
        except ImportError:
            pass
        
        # Try OpenAI if API key is available
        try:
            import openai
            if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
                openai.api_key = settings.OPENAI_API_KEY
                self.embeddings_model = "openai"
                self.embedding_dim = 1536
                self.embedding_provider = "openai"
                print("Using OpenAI for embeddings")
                return
        except ImportError:
            pass
        
        # Try Ollama as fallback (local, free)
        try:
            import ollama
            # Test if Ollama is running
            ollama.embeddings(model='nomic-embed-text', prompt='test')
            self.embeddings_model = "ollama"
            self.embedding_dim = 768  # nomic-embed-text dimension
            self.embedding_provider = "ollama"
            print("Using Ollama for embeddings")
            return
        except:
            pass
        
        print("Warning: No embedding model available. Install sentence-transformers: pip install sentence-transformers")
        self.embeddings_model = None
        self.embedding_dim = 384
        self.embedding_provider = None
    
    async def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.
        
        TODO: Implement embedding generation
        - Use OpenAI embeddings API or
        - Use HuggingFace sentence-transformers
        - Return numpy array of embeddings
        
        Example with OpenAI:
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.embeddings.create(
            model=settings.OPENAI_EMBEDDING_MODEL,
            input=text
        )
        return np.array(response.data[0].embedding)
        
        Example with HuggingFace:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model.encode(text)
        """
        if self.embeddings_model is None:
            # Return zero vector as fallback
            return np.zeros(self.embedding_dim)
        
        if self.embedding_provider == "sentence-transformers":
            # HuggingFace sentence-transformers
            embedding = self.embeddings_model.encode(text)
            return np.array(embedding)
        
        elif self.embedding_provider == "openai":
            # OpenAI embeddings
            import openai
            client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
            response = client.embeddings.create(
                model=getattr(settings, 'OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
                input=text
            )
            return np.array(response.data[0].embedding)
        
        elif self.embedding_provider == "ollama":
            # Ollama embeddings
            import ollama
            response = ollama.embeddings(
                model='nomic-embed-text',
                prompt=text
            )
            return np.array(response['embedding'])
        
        return np.zeros(self.embedding_dim)
    
    async def store_chunk(
        self,
        content: str,
        document_id: int,
        page_number: int,
        chunk_index: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentChunk:
        """
        Store a text chunk with its embedding.
        
        TODO: Implement chunk storage
        1. Generate embedding for content
        2. Create DocumentChunk record
        3. Store in database with embedding
        4. Include metadata (related_images, related_tables, etc.)
        
        Args:
            content: Text content
            document_id: Document ID
            page_number: Page number
            chunk_index: Index of chunk in document
            metadata: Additional metadata (related_images, related_tables, etc.)
            
        Returns:
            Created DocumentChunk
        """
        try:
            # Step 1: Generate embedding
            embedding = await self.generate_embedding(content)
            
            # Step 2: Prepare metadata with related content links
            chunk_metadata = metadata or {}
            
            # Find related images and tables on the same page
            related_images = self._find_related_images(document_id, page_number)
            related_tables = self._find_related_tables(document_id, page_number)
            
            chunk_metadata["related_images"] = related_images
            chunk_metadata["related_tables"] = related_tables
            
            # Step 3: Create DocumentChunk record
            chunk = DocumentChunk(
                document_id=document_id,
                content=content,
                embedding=embedding.tolist(),
                page_number=page_number,
                chunk_index=chunk_index,
                chunk_metadata=chunk_metadata
            )
            
            # Step 4: Store in database
            self.db.add(chunk)
            self.db.commit()
            self.db.refresh(chunk)
            
            return chunk
            
        except Exception as e:
            print(f"Error storing chunk: {e}")
            self.db.rollback()
            return None
    
    def _find_related_images(self, document_id: int, page_number: int) -> List[int]:
        """Find image IDs related to this page."""
        images = self.db.query(DocumentImage).filter(
            DocumentImage.document_id == document_id,
            DocumentImage.page_number == page_number
        ).all()
        return [img.id for img in images]
    
    def _find_related_tables(self, document_id: int, page_number: int) -> List[int]:
        """Find table IDs related to this page."""
        tables = self.db.query(DocumentTable).filter(
            DocumentTable.document_id == document_id,
            DocumentTable.page_number == page_number
        ).all()
        return [tbl.id for tbl in tables]
    
    async def similarity_search(
        self,
        query: str,
        document_id: Optional[int] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity.
        
        TODO: Implement similarity search
        1. Generate embedding for query
        2. Use pgvector's cosine similarity (<-> operator)
        3. Filter by document_id if provided
        4. Return top k results with scores
        5. Include related images and tables in results
        
        Example SQL query:
        SELECT
            id,
            content,
            page_number,
            metadata,
            1 - (embedding <-> :query_embedding) as similarity
        FROM document_chunks
        WHERE document_id = :document_id  -- optional filter
        ORDER BY embedding <-> :query_embedding
        LIMIT :k
        
        Args:
            query: Search query text
            document_id: Optional document ID to filter
            k: Number of results to return
            
        Returns:
            [
                {
                    "content": "...",
                    "score": 0.95,
                    "page_number": 3,
                    "metadata": {...},
                    "related_images": [...],
                    "related_tables": [...]
                }
            ]
        """
        try:
            # Step 1: Generate query embedding
            query_embedding = await self.generate_embedding(query)
            
            # Step 2: Build similarity search query
            embedding_str = "[" + ",".join(map(str, query_embedding.tolist())) + "]"
            
            if document_id is not None:
                # Filter by document
                sql = text("""
                    SELECT
                        id,
                        content,
                        page_number,
                        chunk_metadata,
                        chunk_index,
                        1 - (embedding <-> :query_embedding::vector) as similarity
                    FROM document_chunks
                    WHERE document_id = :document_id
                    ORDER BY embedding <-> :query_embedding::vector
                    LIMIT :k
                """)
                result = self.db.execute(sql, {
                    "query_embedding": embedding_str,
                    "document_id": document_id,
                    "k": k
                })
            else:
                # Search across all documents
                sql = text("""
                    SELECT
                        id,
                        content,
                        page_number,
                        chunk_metadata,
                        chunk_index,
                        document_id,
                        1 - (embedding <-> :query_embedding::vector) as similarity
                    FROM document_chunks
                    ORDER BY embedding <-> :query_embedding::vector
                    LIMIT :k
                """)
                result = self.db.execute(sql, {
                    "query_embedding": embedding_str,
                    "k": k
                })
            
            # Step 3: Format results
            results = []
            for row in result:
                chunk_metadata = row.chunk_metadata if row.chunk_metadata else {}
                
                # Get related images and tables
                related_images = await self._get_related_images(chunk_metadata.get("related_images", []))
                related_tables = await self._get_related_tables(chunk_metadata.get("related_tables", []))
                
                results.append({
                    "id": row.id,
                    "content": row.content,
                    "score": float(row.similarity),
                    "page_number": row.page_number,
                    "chunk_index": row.chunk_index,
                    "metadata": chunk_metadata,
                    "related_images": related_images,
                    "related_tables": related_tables
                })
            
            return results
            
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    async def _get_related_images(self, image_ids: List[int]) -> List[Dict[str, Any]]:
        """Get full image data for related images."""
        if not image_ids:
            return []
        
        images = self.db.query(DocumentImage).filter(
            DocumentImage.id.in_(image_ids)
        ).all()
        
        return [
            {
                "id": img.id,
                "file_path": img.file_path,
                "caption": img.caption,
                "page_number": img.page_number,
                "width": img.width,
                "height": img.height
            }
            for img in images
        ]
    
    async def _get_related_tables(self, table_ids: List[int]) -> List[Dict[str, Any]]:
        """Get full table data for related tables."""
        if not table_ids:
            return []
        
        tables = self.db.query(DocumentTable).filter(
            DocumentTable.id.in_(table_ids)
        ).all()
        
        return [
            {
                "id": tbl.id,
                "image_path": tbl.image_path,
                "data": json.loads(tbl.data) if tbl.data else {},
                "caption": tbl.caption,
                "page_number": tbl.page_number,
                "rows": tbl.rows,
                "columns": tbl.columns
            }
            for tbl in tables
        ]
    
    async def get_related_content(
        self,
        chunk_ids: List[int]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get related images and tables for given chunks.
        
        TODO: Implement related content retrieval
        - Query DocumentImage and DocumentTable based on metadata
        - Return organized by type (images, tables)
        
        Args:
            chunk_ids: List of chunk IDs
            
        Returns:
            {
                "images": [...],
                "tables": [...]
            }
        """
        all_images = []
        all_tables = []
        seen_image_ids = set()
        seen_table_ids = set()
        
        # Get chunks and their metadata
        chunks = self.db.query(DocumentChunk).filter(
            DocumentChunk.id.in_(chunk_ids)
        ).all()
        
        for chunk in chunks:
            chunk_meta = chunk.chunk_metadata if chunk.chunk_metadata else {}
            
            # Collect related image IDs
            for img_id in chunk_meta.get("related_images", []):
                if img_id not in seen_image_ids:
                    seen_image_ids.add(img_id)
            
            # Collect related table IDs
            for tbl_id in chunk_meta.get("related_tables", []):
                if tbl_id not in seen_table_ids:
                    seen_table_ids.add(tbl_id)
        
        # Fetch images
        if seen_image_ids:
            all_images = await self._get_related_images(list(seen_image_ids))
        
        # Fetch tables
        if seen_table_ids:
            all_tables = await self._get_related_tables(list(seen_table_ids))
        
        return {
            "images": all_images,
            "tables": all_tables
        }
    
    async def store_text_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_id: int
    ) -> int:
        """
        Batch store multiple text chunks.
        
        Args:
            chunks: List of chunk dictionaries with content and metadata
            document_id: Document ID
            
        Returns:
            Number of successfully stored chunks
        """
        stored_count = 0
        
        for chunk in chunks:
            result = await self.store_chunk(
                content=chunk.get("content", ""),
                document_id=document_id,
                page_number=chunk.get("page_number", 1),
                chunk_index=chunk.get("chunk_index", stored_count),
                metadata=chunk.get("metadata", {})
            )
            if result:
                stored_count += 1
        
        return stored_count

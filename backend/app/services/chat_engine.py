"""
Chat engine service for multimodal RAG.

TODO: Implement this service to:
1. Process user messages
2. Search for relevant context using vector store
3. Find related images and tables
4. Generate responses using LLM
5. Support multi-turn conversations
"""
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from app.models.conversation import Conversation, Message
from app.services.vector_store import VectorStore
from app.core.config import settings
import time
import json


class ChatEngine:
    """
    Multimodal chat engine with RAG.
    
    This is a SKELETON implementation. You need to implement the core logic.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.vector_store = VectorStore(db)
        self.llm = None  # TODO: Initialize LLM (OpenAI, Ollama, etc.)
        self._initialize_llm()
        
        # Prompt templates
        self.system_prompt = """You are a helpful AI assistant that answers questions based on document content.

When answering questions:
1. Use ONLY the provided context to answer questions
2. If the context contains relevant images or tables, reference them in your answer
3. If you cannot find the answer in the context, say so clearly
4. Cite page numbers when referencing specific information
5. Be concise but thorough

Context will be provided in the following format:
- Text chunks with page numbers
- Related images with captions
- Related tables with data

Always ground your answers in the provided context."""

        self.rag_prompt_template = """Based on the following context, please answer the user's question.

CONTEXT:
{context}

RELATED IMAGES:
{images}

RELATED TABLES:
{tables}

CONVERSATION HISTORY:
{history}

USER QUESTION: {question}

Please provide a helpful, accurate answer based on the context above. If the context doesn't contain enough information to fully answer the question, acknowledge what you can answer and what requires additional information."""
    
    def _initialize_llm(self):
        """
        Initialize the LLM client.
        
        Supports:
        - OpenAI GPT models
        - Ollama (local, free)
        - Groq (free tier)
        - Google Gemini (free tier)
        """
        # Try OpenAI first
        try:
            import openai
            if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
                self.llm_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
                self.llm_provider = "openai"
                self.llm_model = getattr(settings, 'OPENAI_MODEL', 'gpt-4o-mini')
                print(f"Using OpenAI {self.llm_model} for chat")
                return
        except ImportError:
            pass
        
        # Try Groq (free tier available)
        try:
            from groq import Groq
            import os
            groq_key = os.getenv("GROQ_API_KEY") or getattr(settings, 'GROQ_API_KEY', None)
            if groq_key:
                self.llm_client = Groq(api_key=groq_key)
                self.llm_provider = "groq"
                self.llm_model = "llama-3.1-70b-versatile"
                print(f"Using Groq {self.llm_model} for chat")
                return
        except ImportError:
            pass
        
        # Try Google Gemini (free tier available)
        try:
            import google.generativeai as genai
            import os
            gemini_key = os.getenv("GOOGLE_API_KEY") or getattr(settings, 'GOOGLE_API_KEY', None)
            if gemini_key:
                genai.configure(api_key=gemini_key)
                self.llm_client = genai.GenerativeModel('gemini-1.5-flash')
                self.llm_provider = "gemini"
                self.llm_model = "gemini-1.5-flash"
                print(f"Using Google Gemini for chat")
                return
        except ImportError:
            pass
        
        # Try Ollama as fallback (local, free)
        try:
            import ollama
            # Test if Ollama is running
            ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': 'test'}])
            self.llm_client = ollama
            self.llm_provider = "ollama"
            self.llm_model = "llama3.2"
            print(f"Using Ollama {self.llm_model} for chat")
            return
        except:
            pass
        
        print("Warning: No LLM available. Install ollama or set API keys.")
        self.llm_client = None
        self.llm_provider = None
        self.llm_model = None
    
    async def process_message(
        self,
        conversation_id: int,
        message: str,
        document_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process a chat message and generate multimodal response.
        
        Implementation steps:
        1. Load conversation history (for multi-turn support)
        2. Search vector store for relevant context
        3. Find related images and tables
        4. Build prompt with context and history
        5. Generate response using LLM
        6. Format response with sources (text, images, tables)
        
        Args:
            conversation_id: Conversation ID
            message: User message
            document_id: Optional document ID to scope search
            
        Returns:
            {
                "answer": "...",
                "sources": [
                    {
                        "type": "text",
                        "content": "...",
                        "page": 3,
                        "score": 0.95
                    },
                    {
                        "type": "image",
                        "url": "/uploads/images/xxx.png",
                        "caption": "Figure 1: ...",
                        "page": 3
                    },
                    {
                        "type": "table",
                        "url": "/uploads/tables/yyy.png",
                        "caption": "Table 1: ...",
                        "page": 5,
                        "data": {...}  # structured table data
                    }
                ],
                "processing_time": 2.5
            }
        """
        start_time = time.time()
        
        try:
            # Step 1: Load conversation history
            history = await self._load_conversation_history(conversation_id)
            
            # Step 2: Search for relevant context
            context_chunks = await self._search_context(message, document_id)
            
            # Step 3: Find related images and tables
            related_media = await self._find_related_media(context_chunks)
            
            # Step 4: Build prompt
            prompt = self._build_prompt(
                question=message,
                context_chunks=context_chunks,
                related_media=related_media,
                history=history
            )
            
            # Step 5: Generate response using LLM
            answer = await self._generate_response(prompt)
            
            # Step 6: Save message to conversation
            await self._save_message(conversation_id, message, answer, context_chunks, related_media)
            
            # Step 7: Format response with sources
            processing_time = time.time() - start_time
            
            sources = self._format_sources(context_chunks, related_media)
            
            return {
                "answer": answer,
                "sources": sources,
                "processing_time": processing_time
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "answer": f"I apologize, but I encountered an error processing your question: {str(e)}",
                "sources": [],
                "processing_time": processing_time,
                "error": str(e)
            }
    
    async def _load_conversation_history(
        self,
        conversation_id: int,
        limit: int = 5
    ) -> List[Dict[str, str]]:
        """
        Load recent conversation history.
        
        TODO: Implement conversation history loading
        - Load last N messages from conversation
        - Format for LLM context
        - Include both user and assistant messages
        
        Returns:
            [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ]
        """
        messages = self.db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.created_at.desc()).limit(limit * 2).all()
        
        # Reverse to get chronological order
        messages = list(reversed(messages))
        
        history = []
        for msg in messages:
            history.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return history
    
    async def _search_context(
        self,
        query: str,
        document_id: Optional[int] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant context using vector store.
        
        TODO: Implement context search
        - Use vector store similarity search
        - Filter by document if specified
        - Return relevant chunks with metadata
        """
        return await self.vector_store.similarity_search(
            query=query,
            document_id=document_id,
            k=k
        )
    
    async def _find_related_media(
        self,
        context_chunks: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find related images and tables from context chunks.
        
        TODO: Implement related media finding
        - Extract image/table references from chunk metadata
        - Query database for actual image/table records
        - Return with URLs for frontend display
        """
        if not context_chunks:
            return {"images": [], "tables": []}
        
        # Collect all related images and tables from chunks
        all_images = []
        all_tables = []
        seen_image_ids = set()
        seen_table_ids = set()
        
        for chunk in context_chunks:
            # Get related images
            for img in chunk.get("related_images", []):
                img_id = img.get("id") if isinstance(img, dict) else img
                if img_id not in seen_image_ids:
                    seen_image_ids.add(img_id)
                    if isinstance(img, dict):
                        all_images.append(img)
            
            # Get related tables
            for tbl in chunk.get("related_tables", []):
                tbl_id = tbl.get("id") if isinstance(tbl, dict) else tbl
                if tbl_id not in seen_table_ids:
                    seen_table_ids.add(tbl_id)
                    if isinstance(tbl, dict):
                        all_tables.append(tbl)
        
        return {
            "images": all_images,
            "tables": all_tables
        }
    
    def _build_prompt(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        related_media: Dict[str, List[Dict[str, Any]]],
        history: List[Dict[str, str]]
    ) -> str:
        """
        Build the RAG prompt with context, media, and history.
        """
        # Format context
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(
                f"[Chunk {i}] (Page {chunk.get('page_number', 'N/A')}, Relevance: {chunk.get('score', 0):.2f})\n"
                f"{chunk.get('content', '')}"
            )
        context_text = "\n\n".join(context_parts) if context_parts else "No relevant context found."
        
        # Format images
        images_parts = []
        for img in related_media.get("images", []):
            images_parts.append(
                f"- [Image] Page {img.get('page_number', 'N/A')}: {img.get('caption', 'No caption')}"
            )
        images_text = "\n".join(images_parts) if images_parts else "No related images."
        
        # Format tables
        tables_parts = []
        for tbl in related_media.get("tables", []):
            table_info = f"- [Table] Page {tbl.get('page_number', 'N/A')}: {tbl.get('caption', 'No caption')}"
            if tbl.get("data"):
                data = tbl["data"]
                if isinstance(data, dict) and "headers" in data:
                    table_info += f"\n  Headers: {', '.join(str(h) for h in data.get('headers', []))}"
                    if data.get("rows"):
                        table_info += f"\n  Sample rows: {data['rows'][:3]}"
            tables_parts.append(table_info)
        tables_text = "\n".join(tables_parts) if tables_parts else "No related tables."
        
        # Format history
        history_parts = []
        for msg in history[-6:]:  # Last 3 exchanges
            role = "User" if msg["role"] == "user" else "Assistant"
            history_parts.append(f"{role}: {msg['content'][:200]}...")
        history_text = "\n".join(history_parts) if history_parts else "No previous conversation."
        
        # Build final prompt
        return self.rag_prompt_template.format(
            context=context_text,
            images=images_text,
            tables=tables_text,
            history=history_text,
            question=question
        )
    
    async def _generate_response(self, prompt: str) -> str:
        """
        Generate response using LLM.
        
        Supports multiple LLM backends.
        """
        if self.llm_client is None:
            return "I apologize, but no LLM is currently configured. Please set up OpenAI, Groq, Gemini, or Ollama."
        
        try:
            if self.llm_provider == "openai":
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                return response.choices[0].message.content
            
            elif self.llm_provider == "groq":
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                return response.choices[0].message.content
            
            elif self.llm_provider == "gemini":
                full_prompt = f"{self.system_prompt}\n\n{prompt}"
                response = self.llm_client.generate_content(full_prompt)
                return response.text
            
            elif self.llm_provider == "ollama":
                response = self.llm_client.chat(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response['message']['content']
            
            return "Unable to generate response."
            
        except Exception as e:
            print(f"LLM error: {e}")
            return f"I encountered an error generating a response: {str(e)}"
    
    async def _save_message(
        self,
        conversation_id: int,
        user_message: str,
        assistant_response: str,
        context_chunks: List[Dict[str, Any]],
        related_media: Dict[str, List[Dict[str, Any]]]
    ):
        """Save user message and assistant response to conversation."""
        try:
            # Save user message
            user_msg = Message(
                conversation_id=conversation_id,
                role="user",
                content=user_message,
                metadata=json.dumps({})
            )
            self.db.add(user_msg)
            
            # Save assistant message with sources
            assistant_msg = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=assistant_response,
                metadata=json.dumps({
                    "sources": [
                        {"chunk_id": c.get("id"), "score": c.get("score")}
                        for c in context_chunks
                    ],
                    "images": [img.get("id") for img in related_media.get("images", [])],
                    "tables": [tbl.get("id") for tbl in related_media.get("tables", [])]
                })
            )
            self.db.add(assistant_msg)
            
            self.db.commit()
            
        except Exception as e:
            print(f"Error saving messages: {e}")
            self.db.rollback()
    
    def _format_sources(
        self,
        context_chunks: List[Dict[str, Any]],
        related_media: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Format sources for API response."""
        sources = []
        
        # Add text chunks as sources
        for chunk in context_chunks:
            sources.append({
                "type": "text",
                "content": chunk.get("content", "")[:500] + "..." if len(chunk.get("content", "")) > 500 else chunk.get("content", ""),
                "page": chunk.get("page_number"),
                "score": chunk.get("score", 0)
            })
        
        # Add images as sources
        for img in related_media.get("images", []):
            sources.append({
                "type": "image",
                "url": img.get("file_path", ""),
                "caption": img.get("caption", ""),
                "page": img.get("page_number")
            })
        
        # Add tables as sources
        for tbl in related_media.get("tables", []):
            sources.append({
                "type": "table",
                "url": tbl.get("image_path", ""),
                "caption": tbl.get("caption", ""),
                "page": tbl.get("page_number"),
                "data": tbl.get("data", {})
            })
        
        return sources
    
    async def create_conversation(self, document_id: Optional[int] = None) -> Conversation:
        """Create a new conversation."""
        conversation = Conversation(
            document_id=document_id,
            metadata=json.dumps({})
        )
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        return conversation
    
    async def get_conversation_history(
        self,
        conversation_id: int
    ) -> List[Dict[str, Any]]:
        """Get full conversation history with sources."""
        messages = self.db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.created_at.asc()).all()
        
        return [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "metadata": json.loads(msg.metadata) if msg.metadata else {},
                "created_at": msg.created_at.isoformat() if msg.created_at else None
            }
            for msg in messages
        ]

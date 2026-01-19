"""
Multimodal Chat Engine with RAG Pipeline

Supports multiple LLM backends:
- Groq Llama (free tier available) - RECOMMENDED
- Google Gemini (free tier available)
- OpenAI GPT (requires API key)
- Ollama (local, completely free)
"""
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from app.models.conversation import Conversation, Message
from app.models.document import DocumentImage, DocumentTable
from app.services.vector_store import VectorStore
from app.core.config import settings
import time
import json
import os


class ChatEngine:
    """Multimodal chat engine with RAG (Retrieval-Augmented Generation)."""
    
    def __init__(self, db: Session):
        """Initialize chat engine with database session."""
        self.db = db
        self.vector_store = VectorStore(db)
        self.llm_client = None
        self.llm_provider = None
        self.llm_model = None
        
        self._initialize_llm()
        
        self.system_prompt = """You are a helpful AI assistant that answers questions based on document content.

When answering questions:
1. Use ONLY the provided context to answer questions
2. If the context contains relevant images or tables, reference them in your answer
3. If you cannot find the answer in the context, say so clearly
4. Cite page numbers when referencing specific information
5. Be concise but thorough

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

Please provide a helpful, accurate answer based on the context above."""
    
    def _initialize_llm(self):
        """Initialize the LLM client with fallback chain."""
        
        # Try Groq first (free tier, fast, recommended)
        try:
            from groq import Groq
            groq_key = os.getenv("GROQ_API_KEY") or getattr(settings, 'GROQ_API_KEY', None)
            if groq_key and groq_key.strip():
                self.llm_client = Groq(api_key=groq_key)
                self.llm_provider = "groq"
                self.llm_model = "llama-3.3-70b-versatile"  # Updated model name
                print(f"✓ Using Groq {self.llm_model} for chat")
                return
        except ImportError:
            print("groq not installed")
        except Exception as e:
            print(f"Groq init error: {e}")
        
        # Try Google Gemini
        try:
            import google.generativeai as genai
            gemini_key = os.getenv("GOOGLE_API_KEY") or getattr(settings, 'GOOGLE_API_KEY', None)
            if gemini_key and gemini_key.strip():
                genai.configure(api_key=gemini_key)
                model_name = 'gemini-2.0-flash'
                self.llm_client = genai.GenerativeModel(model_name)
                self.llm_provider = "gemini"
                self.llm_model = model_name
                print(f"✓ Using Google Gemini ({model_name}) for chat")
                return
        except ImportError:
            print("google-generativeai not installed")
        except Exception as e:
            print(f"Gemini init error: {e}")
        
        # Try OpenAI
        try:
            import openai
            api_key = os.getenv("OPENAI_API_KEY") or getattr(settings, 'OPENAI_API_KEY', None)
            if api_key and api_key.strip() and not api_key.startswith('sk-your'):
                self.llm_client = openai.OpenAI(api_key=api_key)
                self.llm_provider = "openai"
                self.llm_model = getattr(settings, 'OPENAI_MODEL', 'gpt-4o-mini')
                print(f"✓ Using OpenAI {self.llm_model} for chat")
                return
        except ImportError:
            print("openai not installed")
        except Exception as e:
            print(f"OpenAI init error: {e}")
        
        # Try Ollama as fallback (local, free)
        try:
            import ollama
            models = ollama.list()
            if models and models.get('models'):
                model_name = models['models'][0]['name'].split(':')[0]
                self.llm_client = ollama
                self.llm_provider = "ollama"
                self.llm_model = model_name
                print(f"✓ Using Ollama {self.llm_model} for chat")
                return
        except ImportError:
            print("ollama not installed")
        except Exception as e:
            print(f"Ollama init error: {e}")
        
        print("⚠ Warning: No LLM available. Set API keys or install Ollama.")
        self.llm_client = None
        self.llm_provider = None
        self.llm_model = None
    
    async def process_message(
        self,
        conversation_id: int,
        message: str,
        document_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Process a chat message and generate multimodal response."""
        start_time = time.time()
        
        try:
            history = await self._load_conversation_history(conversation_id)
            context_chunks = await self._search_context(message, document_id)
            related_media = await self._find_related_media(context_chunks, document_id)
            
            prompt = self._build_prompt(
                question=message,
                context_chunks=context_chunks,
                related_media=related_media,
                history=history
            )
            
            answer = await self._generate_response(prompt)
            
            processing_time = time.time() - start_time
            sources = self._format_sources(context_chunks, related_media)
            
            return {
                "answer": answer,
                "sources": sources,
                "processing_time": round(processing_time, 2)
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"Error processing message: {e}")
            import traceback
            traceback.print_exc()
            return {
                "answer": f"I encountered an error processing your question: {str(e)}",
                "sources": [],
                "processing_time": round(processing_time, 2)
            }
    
    async def _load_conversation_history(
        self,
        conversation_id: int,
        max_messages: int = 10
    ) -> List[Dict[str, str]]:
        """Load recent conversation history."""
        try:
            messages = self.db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).order_by(Message.created_at.desc()).limit(max_messages).all()
            
            messages = list(reversed(messages))
            return [{"role": msg.role, "content": msg.content} for msg in messages]
        except Exception as e:
            print(f"Error loading history: {e}")
            return []
    
    async def _search_context(
        self,
        query: str,
        document_id: Optional[int] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search vector store for relevant text chunks."""
        try:
            return await self.vector_store.search_similar(
                query=query,
                document_id=document_id,
                k=k
            )
        except Exception as e:
            print(f"Error searching context: {e}")
            return []
    
    async def _find_related_media(
        self,
        context_chunks: List[Dict[str, Any]],
        document_id: Optional[int] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Find images and tables related to the context chunks."""
        images = []
        tables = []
        
        try:
            page_numbers = set()
            for chunk in context_chunks:
                if chunk.get("page_number"):
                    page_numbers.add(chunk["page_number"])
            
            if not page_numbers and document_id:
                page_numbers = {1, 2, 3}
            
            if page_numbers:
                image_query = self.db.query(DocumentImage)
                if document_id:
                    image_query = image_query.filter(DocumentImage.document_id == document_id)
                image_query = image_query.filter(DocumentImage.page_number.in_(page_numbers))
                
                for img in image_query.limit(5).all():
                    images.append({
                        "id": img.id,
                        "file_path": img.file_path,
                        "page_number": img.page_number,
                        "caption": img.caption or "",
                        "width": img.width,
                        "height": img.height
                    })
            
            if page_numbers:
                table_query = self.db.query(DocumentTable)
                if document_id:
                    table_query = table_query.filter(DocumentTable.document_id == document_id)
                table_query = table_query.filter(DocumentTable.page_number.in_(page_numbers))
                
                for tbl in table_query.limit(5).all():
                    table_data = tbl.data
                    if isinstance(table_data, str):
                        try:
                            table_data = json.loads(table_data)
                        except:
                            table_data = {}
                    
                    tables.append({
                        "id": tbl.id,
                        "image_path": tbl.image_path,
                        "page_number": tbl.page_number,
                        "caption": tbl.caption or "",
                        "data": table_data,
                        "rows": tbl.rows,
                        "columns": tbl.columns
                    })
                    
        except Exception as e:
            print(f"Error finding related media: {e}")
        
        return {"images": images, "tables": tables}
    
    def _build_prompt(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        related_media: Dict[str, List[Dict[str, Any]]],
        history: List[Dict[str, str]]
    ) -> str:
        """Build the RAG prompt with context, media, and history."""
        # Format context
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            score = chunk.get('score', 0)
            page = chunk.get('page_number', 'N/A')
            content = chunk.get('content', '')
            context_parts.append(f"[Chunk {i}] (Page {page}, Score: {score:.2f})\n{content}")
        context_text = "\n\n".join(context_parts) if context_parts else "No relevant context found."
        
        # Format images
        images_parts = []
        for img in related_media.get("images", []):
            images_parts.append(f"- [Image on Page {img.get('page_number', 'N/A')}]: {img.get('caption', 'No caption')}")
        images_text = "\n".join(images_parts) if images_parts else "No related images."
        
        # Format tables
        tables_parts = []
        for tbl in related_media.get("tables", []):
            table_info = f"- [Table on Page {tbl.get('page_number', 'N/A')}]: {tbl.get('caption', 'No caption')}"
            data = tbl.get("data", {})
            if isinstance(data, dict) and "headers" in data:
                table_info += f"\n  Columns: {', '.join(str(h) for h in data.get('headers', []))}"
            tables_parts.append(table_info)
        tables_text = "\n".join(tables_parts) if tables_parts else "No related tables."
        
        # Format history
        history_parts = []
        for msg in history[-10:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
            history_parts.append(f"{role}: {content}")
        history_text = "\n".join(history_parts) if history_parts else "No previous conversation."
        
        return self.rag_prompt_template.format(
            context=context_text,
            images=images_text,
            tables=tables_text,
            history=history_text,
            question=question
        )
    
    async def _generate_response(self, prompt: str) -> str:
        """Generate response using the configured LLM."""
        if self.llm_client is None:
            return (
                "I apologize, but no LLM is currently configured. "
                "Please set up one of the following:\n"
                "- Groq: Set GROQ_API_KEY in .env (free at console.groq.com)\n"
                "- Gemini: Set GOOGLE_API_KEY in .env\n"
                "- OpenAI: Set OPENAI_API_KEY in .env\n"
                "- Ollama: Install and run locally"
            )
        
        try:
            if self.llm_provider == "groq":
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
            
            elif self.llm_provider == "openai":
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
            
            elif self.llm_provider == "ollama":
                response = self.llm_client.chat(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response['message']['content']
            
            return "Unable to generate response - unknown LLM provider."
            
        except Exception as e:
            print(f"LLM error: {e}")
            import traceback
            traceback.print_exc()
            return f"I encountered an error generating a response: {str(e)}"
    
    def _format_sources(
        self,
        context_chunks: List[Dict[str, Any]],
        related_media: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Format sources for the API response."""
        sources = []
        
        for chunk in context_chunks:
            content = chunk.get("content", "")
            if len(content) > 500:
                content = content[:500] + "..."
            sources.append({
                "type": "text",
                "content": content,
                "page": chunk.get("page_number"),
                "score": round(chunk.get("score", 0), 3)
            })
        
        for img in related_media.get("images", []):
            sources.append({
                "type": "image",
                "url": img.get("file_path", ""),
                "caption": img.get("caption", ""),
                "page": img.get("page_number")
            })
        
        for tbl in related_media.get("tables", []):
            sources.append({
                "type": "table",
                "url": tbl.get("image_path", ""),
                "caption": tbl.get("caption", ""),
                "page": tbl.get("page_number"),
                "data": tbl.get("data", {})
            })
        
        return sources

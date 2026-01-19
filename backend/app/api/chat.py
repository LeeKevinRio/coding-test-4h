"""
Chat API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from app.db.session import get_db
from app.models.conversation import Conversation, Message
from app.services.chat_engine import ChatEngine

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[int] = None
    document_id: Optional[int] = None


class ChatResponse(BaseModel):
    conversation_id: int
    message_id: int
    answer: str
    sources: List[dict]
    processing_time: float


@router.post("")
async def send_message(
    request: ChatRequest,
    db: Session = Depends(get_db)
) -> ChatResponse:
    """
    Send a chat message and get a response

    This endpoint:
    1. Creates or retrieves conversation
    2. Saves user message
    3. Processes message with ChatEngine (RAG + multimodal)
    4. Saves assistant response
    5. Returns answer with sources (text, images, tables)
    """
    # Create or get conversation
    if request.conversation_id:
        conversation = db.query(Conversation).filter(
            Conversation.id == request.conversation_id
        ).first()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        conversation = Conversation(
            title=request.message[:50],  # First 50 chars as title
            document_id=request.document_id
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    # Save user message
    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        content=request.message
    )
    db.add(user_message)
    db.commit()

    # Process message with ChatEngine
    chat_engine = ChatEngine(db)
    result = await chat_engine.process_message(
        conversation_id=conversation.id,
        message=request.message,
        document_id=request.document_id
    )

    # Save assistant message
    assistant_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=result["answer"],
        sources=result.get("sources", [])
    )
    db.add(assistant_message)
    db.commit()
    db.refresh(assistant_message)

    return ChatResponse(
        conversation_id=conversation.id,
        message_id=assistant_message.id,
        answer=result["answer"],
        sources=result.get("sources", []),
        processing_time=result.get("processing_time", 0.0)
    )


@router.get("/conversations")
async def list_conversations(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get list of all conversations
    """
    conversations = db.query(Conversation).order_by(
        Conversation.updated_at.desc()
    ).offset(skip).limit(limit).all()

    return {
        "conversations": [
            {
                "id": conv.id,
                "title": conv.title,
                "created_at": conv.created_at,
                "updated_at": conv.updated_at,
                "document_id": conv.document_id,
                "message_count": len(conv.messages)
            }
            for conv in conversations
        ],
        "total": db.query(Conversation).count()
    }


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: int,
    db: Session = Depends(get_db)
):
    """
    Get conversation details with all messages
    """
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {
        "id": conversation.id,
        "title": conversation.title,
        "created_at": conversation.created_at,
        "document_id": conversation.document_id,
        "messages": [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "sources": msg.sources,
                "created_at": msg.created_at
            }
            for msg in conversation.messages
        ]
    }


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a conversation and all its messages
    """
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    db.delete(conversation)
    db.commit()

    return {"message": "Conversation deleted successfully"}

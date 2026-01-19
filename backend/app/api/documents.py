"""
Document management API endpoints
"""
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
from app.db.session import get_db
from app.models.document import Document
from app.services.document_processor import DocumentProcessor
from app.core.config import settings
import os
import uuid
from datetime import datetime

router = APIRouter()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Upload a PDF document for processing
    
    This endpoint:
    1. Saves the uploaded file
    2. Creates a document record
    3. Triggers background processing (Docling extraction)
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Validate file size
    contents = await file.read()
    if len(contents) > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File size exceeds {settings.MAX_FILE_SIZE / 1024 / 1024}MB limit")
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{file_id}{file_extension}"
    file_path = os.path.join(settings.UPLOAD_DIR, "documents", unique_filename)
    
    # Save file
    with open(file_path, "wb") as f:
        f.write(contents)
    
    # Create document record
    document = Document(
        filename=file.filename,
        file_path=file_path,
        processing_status="pending"
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    
    # TODO: Trigger background processing
    # background_tasks.add_task(process_document_task, document.id, file_path, db)
    # For now, you can process synchronously or implement Celery
    
    return {
        "id": document.id,
        "filename": document.filename,
        "status": document.processing_status,
        "message": "Document uploaded successfully. Processing will begin shortly."
    }


@router.get("")
async def list_documents(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get list of all documents
    """
    documents = db.query(Document).offset(skip).limit(limit).all()
    
    return {
        "documents": [
            {
                "id": doc.id,
                "filename": doc.filename,
                "upload_date": doc.upload_date,
                "status": doc.processing_status,
                "total_pages": doc.total_pages,
                "text_chunks": doc.text_chunks_count,
                "images": doc.images_count,
                "tables": doc.tables_count
            }
            for doc in documents
        ],
        "total": db.query(Document).count()
    }


@router.get("/{document_id}")
async def get_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Get document details including extracted images and tables
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "id": document.id,
        "filename": document.filename,
        "upload_date": document.upload_date,
        "status": document.processing_status,
        "error_message": document.error_message,
        "total_pages": document.total_pages,
        "text_chunks": document.text_chunks_count,
        "images": [
            {
                "id": img.id,
                "url": f"/uploads/images/{os.path.basename(img.file_path)}",
                "page": img.page_number,
                "caption": img.caption,
                "width": img.width,
                "height": img.height
            }
            for img in document.images
        ],
        "tables": [
            {
                "id": tbl.id,
                "url": f"/uploads/tables/{os.path.basename(tbl.image_path)}",
                "page": tbl.page_number,
                "caption": tbl.caption,
                "rows": tbl.rows,
                "columns": tbl.columns,
                "data": tbl.data
            }
            for tbl in document.tables
        ]
    }


@router.post("/{document_id}/process")
async def process_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Manually trigger document processing
    """
    document = db.query(Document).filter(Document.id == document_id).first()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if document.processing_status == "processing":
        return {"message": "Document is already being processed", "status": document.processing_status}

    if document.processing_status == "completed":
        return {"message": "Document has already been processed", "status": document.processing_status}

    # Update status to processing
    document.processing_status = "processing"
    db.commit()

    try:
        processor = DocumentProcessor(db)
        result = await processor.process_document(document.file_path, document.id)

        return {
            "id": document.id,
            "status": "completed",
            "result": result
        }
    except Exception as e:
        document.processing_status = "error"
        document.error_message = str(e)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a document and all associated data
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete physical files
    if os.path.exists(document.file_path):
        os.remove(document.file_path)
    
    for img in document.images:
        if os.path.exists(img.file_path):
            os.remove(img.file_path)
    
    for tbl in document.tables:
        if os.path.exists(tbl.image_path):
            os.remove(tbl.image_path)
    
    # Delete database record (cascade will handle related records)
    db.delete(document)
    db.commit()
    
    return {"message": "Document deleted successfully"}

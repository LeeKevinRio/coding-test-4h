"""
Document-related database models
"""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.session import Base
from pgvector.sqlalchemy import Vector


class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)
    processing_status = Column(String, default="pending")  # pending, processing, completed, error
    error_message = Column(Text, nullable=True)
    total_pages = Column(Integer, default=0)
    text_chunks_count = Column(Integer, default=0)
    images_count = Column(Integer, default=0)
    tables_count = Column(Integer, default=0)
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    images = relationship("DocumentImage", back_populates="document", cascade="all, delete-orphan")
    tables = relationship("DocumentTable", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(384))  # OpenAI embedding dimension
    page_number = Column(Integer)
    chunk_index = Column(Integer)
    chunk_metadata = Column(JSON)  # {related_images: [...], related_tables: [...], ...}
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")


class DocumentImage(Base):
    __tablename__ = "document_images"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    file_path = Column(String, nullable=False)
    page_number = Column(Integer)
    caption = Column(Text, nullable=True)
    width = Column(Integer)
    height = Column(Integer)
    image_metadata = Column(JSON)  # Additional metadata from Docling
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="images")


class DocumentTable(Base):
    __tablename__ = "document_tables"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    image_path = Column(String, nullable=False)  # Rendered table as image
    data = Column(JSON, nullable=True)  # Structured table data
    page_number = Column(Integer)
    caption = Column(Text, nullable=True)
    rows = Column(Integer)
    columns = Column(Integer)
    table_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="tables")

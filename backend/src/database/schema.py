"""SQLAlchemy database models."""

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, Column, DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class MemoryFileModel(Base):
    """Memory file model with metadata."""

    __tablename__ = "memory_files"

    id = Column(Integer, primary_key=True, index=True)
    file_path = Column(String(512), unique=True, nullable=False, index=True)
    title = Column(String(256), nullable=False)
    category = Column(String(64), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    file_hash = Column(String(64), nullable=False)
    word_count = Column(Integer, default=0)
    tags = Column(ARRAY(String), default=list)
    file_metadata = Column(JSONB, default=dict)

    # Relationships
    chunks = relationship("MemoryChunkModel", back_populates="file", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_memory_files_category", "category"),
        Index("idx_memory_files_tags", "tags", postgresql_using="gin"),
        Index("idx_memory_files_updated", "updated_at"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "file_path": self.file_path,
            "title": self.title,
            "category": self.category,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "file_hash": self.file_hash,
            "word_count": self.word_count,
            "tags": self.tags or [],
            "metadata": self.file_metadata or {},
        }


class MemoryChunkModel(Base):
    """Memory chunk model with embedding vector."""

    __tablename__ = "memory_chunks"

    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(Integer, ForeignKey("memory_files.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False)
    embedding = Column(Vector(1536))  # Default dimension, configurable
    header_path = Column(ARRAY(Text), default=list)
    section_level = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    file = relationship("MemoryFileModel", back_populates="chunks")

    __table_args__ = (
        Index("idx_memory_chunks_file_id", "file_id"),
        Index("idx_memory_chunks_embedding", "embedding", postgresql_using="ivfflat"),
        # Fulltext search index created via raw SQL in migration
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "file_id": self.file_id,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "content_hash": self.content_hash,
            "header_path": self.header_path or [],
            "section_level": self.section_level,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ResearchSessionModel(Base):
    """Research session tracking."""

    __tablename__ = "research_sessions"

    id = Column(String(64), primary_key=True)
    mode = Column(String(16), nullable=False)  # speed, balanced, quality
    query = Column(Text, nullable=False)
    status = Column(String(32), nullable=False)  # running, completed, failed
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    completed_at = Column(DateTime(timezone=True))
    final_report = Column(Text)
    session_metadata = Column("metadata", JSONB, default=dict)

    __table_args__ = (
        Index("idx_research_sessions_status", "status"),
        Index("idx_research_sessions_created", "created_at"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "mode": self.mode,
            "query": self.query,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "final_report": self.final_report,
            "metadata": self.session_metadata or {},
        }


class ChatModel(Base):
    """Chat conversation model."""

    __tablename__ = "chats"

    id = Column(String(64), primary_key=True)
    title = Column(String(256), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    chat_metadata = Column("metadata", JSONB, default=dict)

    # Relationships
    messages = relationship("ChatMessageModel", back_populates="chat", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_chats_created", "created_at"),
        Index("idx_chats_updated", "updated_at"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "metadata": self.chat_metadata or {},
        }


class ChatMessageModel(Base):
    """Chat message model with embedding support."""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(String(64), ForeignKey("chats.id", ondelete="CASCADE"), nullable=False, index=True)
    message_id = Column(String(64), nullable=False, index=True)
    role = Column(String(16), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536))  # Default dimension, will be configurable
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    message_metadata = Column("metadata", JSONB, default=dict)

    # Relationships
    chat = relationship("ChatModel", back_populates="messages")

    __table_args__ = (
        Index("idx_chat_messages_chat_id", "chat_id"),
        Index("idx_chat_messages_created", "created_at"),
        Index("idx_chat_messages_embedding", "embedding", postgresql_using="ivfflat"),
        # Full-text search index will be created via migration SQL
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "chat_id": self.chat_id,
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": self.message_metadata or {},
        }

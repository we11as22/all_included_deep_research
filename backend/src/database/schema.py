"""SQLAlchemy database models."""

import os
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, Column, DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


def _get_embedding_dimension() -> int:
    """Get embedding dimension from environment or settings.
    
    This is used for database schema definition. The actual dimension
    should match the embedding provider's dimension from settings.
    """
    # Try environment variable first
    env_dim = os.getenv("EMBEDDING_DIMENSION")
    if env_dim:
        try:
            return int(env_dim)
        except ValueError:
            pass
    
    # Try to get from settings (lazy import to avoid circular dependencies)
    try:
        from src.config.settings import get_settings
        settings = get_settings()
        return settings.embedding_dimension
    except Exception:
        # Fallback to default
        return 1536


# Embedding dimension for database schema (must match embedding provider dimension)
EMBEDDING_DIMENSION = _get_embedding_dimension()


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
    embedding = Column(Vector(EMBEDDING_DIMENSION))  # Dimension from settings/environment
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
    """Deep research session tracking with multi-chat support.

    Key features:
    - Each session is tied to a specific chat_id
    - Only one active session per chat_id (enforced by DB constraint)
    - Session resume support through stable session_id
    - Stores all deep research artifacts (deep_search, draft_report, etc.)
    """

    __tablename__ = "research_sessions"

    id = Column(String(64), primary_key=True)
    chat_id = Column(String(64), ForeignKey("chats.id", ondelete="CASCADE"), nullable=False)
    original_query = Column(Text, nullable=False)  # Saved immediately on session creation
    mode = Column(String(16), nullable=False)  # quality, balanced, speed
    status = Column(String(32), nullable=False)  # active, waiting_clarification, researching, completed, superseded, cancelled, expired
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    completed_at = Column(DateTime(timezone=True))

    # Research artifacts
    deep_search_result = Column(Text)  # Initial deep search result
    clarification_answers = Column(Text)  # User answers to clarification questions
    draft_report = Column(Text)  # Working draft updated by agents/supervisor
    final_report = Column(Text)  # Final formatted report

    session_metadata = Column("metadata", JSONB, default=dict)

    # Relationships
    chat = relationship("ChatModel", back_populates="research_sessions")

    __table_args__ = (
        Index("idx_research_sessions_chat_id", "chat_id"),
        Index("idx_research_sessions_status", "status"),
        Index("idx_research_sessions_created", "created_at"),
        # UNIQUE partial index for one active session per chat (created in migration)
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "chat_id": self.chat_id,
            "original_query": self.original_query,
            "mode": self.mode,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "deep_search_result": self.deep_search_result,
            "clarification_answers": self.clarification_answers,
            "draft_report": self.draft_report,
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
    research_sessions = relationship("ResearchSessionModel", back_populates="chat", cascade="all, delete-orphan")

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
    """Chat message model with embedding support and research session tracking."""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(String(64), ForeignKey("chats.id", ondelete="CASCADE"), nullable=False, index=True)
    message_id = Column(String(64), nullable=False, index=True)
    role = Column(String(16), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIMENSION))  # Dimension from settings/environment
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    message_metadata = Column("metadata", JSONB, default=dict)

    # NEW: Research session tracking
    mode = Column(String(16))  # chat, search, deep_search, deep_research, quality, balanced, speed
    session_id = Column(String(64), ForeignKey("research_sessions.id", ondelete="SET NULL"))  # Links to deep research session
    original_query = Column(Text)  # Denormalized for quick access

    # Relationships
    chat = relationship("ChatModel", back_populates="messages")
    research_session = relationship("ResearchSessionModel", foreign_keys=[session_id])

    __table_args__ = (
        Index("idx_chat_messages_chat_id", "chat_id"),
        Index("idx_chat_messages_created", "created_at"),
        Index("idx_chat_messages_embedding", "embedding", postgresql_using="ivfflat"),
        Index("idx_chat_messages_mode", "mode"),
        Index("idx_chat_messages_session_id", "session_id"),
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
            "mode": self.mode,
            "session_id": self.session_id,
            "original_query": self.original_query,
        }

"""SQLite database schema for all_included_deep_research.

Migrated from PostgreSQL to SQLite for simpler deployment.
Vector search delegated to external store (Chroma/FAISS).
"""

import json
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


# Helper functions for JSON serialization in SQLite
def _serialize_json(value: Any) -> str:
    """Serialize value to JSON string."""
    if value is None:
        return "{}"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _deserialize_json(value: str | None) -> dict | list:
    """Deserialize JSON string to Python object."""
    if not value:
        return {}
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}


# ==================== Chat Tables ====================


class ChatModel(Base):
    """Chat conversation model."""

    __tablename__ = "chats"

    id = Column(String(64), primary_key=True)
    title = Column(String(256), nullable=False)
    created_at = Column(String, nullable=False)  # ISO timestamp
    updated_at = Column(String, nullable=False)  # ISO timestamp
    chat_metadata = Column(Text, default="{}")  # JSON string (renamed to avoid SQLAlchemy conflict)

    # Relationships
    messages = relationship(
        "ChatMessageModel", back_populates="chat", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_chats_created", "created_at"),
        Index("idx_chats_updated", "updated_at"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": _deserialize_json(self.metadata),
        }


class ChatMessageModel(Base):
    """Chat message model."""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chat_id = Column(
        String(64),
        ForeignKey("chats.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    message_id = Column(String(64), nullable=False, unique=True, index=True)
    role = Column(String(16), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    created_at = Column(String, nullable=False)  # ISO timestamp
    msg_metadata = Column(Text, default="{}")  # JSON string

    # Relationships
    chat = relationship("ChatModel", back_populates="messages")

    __table_args__ = (
        Index("idx_chat_messages_chat_id", "chat_id"),
        Index("idx_chat_messages_created", "created_at"),
        Index("idx_chat_messages_role", "role"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "chat_id": self.chat_id,
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
            "metadata": _deserialize_json(self.metadata),
        }


# ==================== Research Session Tables ====================


class ResearchSessionModel(Base):
    """Research session tracking for deep research mode."""

    __tablename__ = "research_sessions"

    id = Column(String(64), primary_key=True)
    chat_id = Column(
        String(64), ForeignKey("chats.id", ondelete="CASCADE"), nullable=True
    )
    mode = Column(String(16), nullable=False)  # speed, balanced, quality
    query = Column(Text, nullable=False)
    status = Column(String(32), default="running")  # running, completed, failed
    created_at = Column(String, nullable=False)  # ISO timestamp
    completed_at = Column(String, nullable=True)  # ISO timestamp
    final_report = Column(Text, nullable=True)
    msg_metadata = Column(Text, default="{}")  # JSON string

    # Relationships
    agent_memories = relationship(
        "AgentMemoryModel", back_populates="session", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_research_sessions_status", "status"),
        Index("idx_research_sessions_created", "created_at"),
        Index("idx_research_sessions_mode", "mode"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "chat_id": self.chat_id,
            "mode": self.mode,
            "query": self.query,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "final_report": self.final_report,
            "metadata": _deserialize_json(self.metadata),
        }


# ==================== Agent Memory Tables ====================


class AgentMemoryModel(Base):
    """Agent memory persistence for deep research.

    Stores todos, notes, and findings from agents across sessions.
    Enables persistent agent memory between research sessions.
    """

    __tablename__ = "agent_memory"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        String(64),
        ForeignKey("research_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    agent_id = Column(String(128), nullable=False)  # e.g., agent_r0_0
    memory_type = Column(String(32), nullable=False)  # todo, note, finding
    content = Column(Text, nullable=False)  # JSON or markdown content
    status = Column(String(32), nullable=True)  # For todos: pending/in_progress/done
    created_at = Column(String, nullable=False)  # ISO timestamp
    updated_at = Column(String, nullable=False)  # ISO timestamp
    msg_metadata = Column(Text, default="{}")  # JSON string (priority, urls, tags, etc.)

    # Relationships
    session = relationship("ResearchSessionModel", back_populates="agent_memories")

    __table_args__ = (
        Index("idx_agent_memory_session_agent", "session_id", "agent_id"),
        Index("idx_agent_memory_type", "memory_type"),
        Index("idx_agent_memory_status", "status"),
        Index("idx_agent_memory_created", "created_at"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "memory_type": self.memory_type,
            "content": self.content,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": _deserialize_json(self.metadata),
        }


# ==================== Memory Files Tables ====================


class MemoryFileModel(Base):
    """Memory file metadata (without vectors).

    Vector embeddings stored in external vector store (Chroma/FAISS).
    """

    __tablename__ = "memory_files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_path = Column(String(512), unique=True, nullable=False, index=True)
    title = Column(String(256), nullable=False)
    category = Column(String(64), nullable=False, index=True)
    created_at = Column(String, nullable=False)  # ISO timestamp
    updated_at = Column(String, nullable=False)  # ISO timestamp
    file_hash = Column(String(64), nullable=False)
    word_count = Column(Integer, default=0)
    tags = Column(Text, default="[]")  # JSON array as string
    msg_metadata = Column(Text, default="{}")  # JSON string

    # Relationships
    chunks = relationship(
        "MemoryChunkModel", back_populates="file", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_memory_files_category", "category"),
        Index("idx_memory_files_updated", "updated_at"),
        Index("idx_memory_files_hash", "file_hash"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "file_path": self.file_path,
            "title": self.title,
            "category": self.category,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "file_hash": self.file_hash,
            "word_count": self.word_count,
            "tags": _deserialize_json(self.tags) if self.tags else [],
            "metadata": _deserialize_json(self.metadata),
        }


class MemoryChunkModel(Base):
    """Memory chunk model (without embedding vector).

    Embeddings stored in external vector store with reference to chunk_id.
    """

    __tablename__ = "memory_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(
        Integer,
        ForeignKey("memory_files.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False)
    header_path = Column(Text, default="[]")  # JSON array as string
    section_level = Column(Integer, default=0)
    created_at = Column(String, nullable=False)  # ISO timestamp

    # Relationships
    file = relationship("MemoryFileModel", back_populates="chunks")

    __table_args__ = (
        Index("idx_memory_chunks_file_id", "file_id"),
        Index("idx_memory_chunks_hash", "content_hash"),
        Index("idx_memory_chunks_file_index", "file_id", "chunk_index"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "file_id": self.file_id,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "content_hash": self.content_hash,
            "header_path": _deserialize_json(self.header_path) if self.header_path else [],
            "section_level": self.section_level,
            "created_at": self.created_at,
        }


# ==================== Helper Functions ====================


def create_tables(engine):
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)


def drop_tables(engine):
    """Drop all tables from the database."""
    Base.metadata.drop_all(bind=engine)


def get_current_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()

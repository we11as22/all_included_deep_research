"""Pydantic models for memory system."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoryCategory(str, Enum):
    """Memory file categories."""

    MAIN = "main"
    PROJECT = "project"
    CONCEPT = "concept"
    CONVERSATION = "conversation"
    PREFERENCE = "preference"
    OTHER = "other"


class MemoryFileBase(BaseModel):
    """Base memory file model."""

    file_path: str
    title: str
    category: MemoryCategory
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryFileCreate(MemoryFileBase):
    """Create memory file model."""

    content: str
    file_hash: str
    word_count: int = 0


class MemoryFileUpdate(BaseModel):
    """Update memory file model."""

    title: str | None = None
    category: MemoryCategory | None = None
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None
    content: str | None = None
    file_hash: str | None = None
    word_count: int | None = None


class MemoryFile(MemoryFileBase):
    """Memory file model."""

    id: int
    created_at: datetime
    updated_at: datetime
    file_hash: str
    word_count: int

    class Config:
        from_attributes = True


class MemoryFileWithContent(MemoryFile):
    """Memory file with content."""

    content: str

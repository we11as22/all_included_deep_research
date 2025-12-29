"""Pydantic models for memory chunks."""

from datetime import datetime

from pydantic import BaseModel, Field


class ChunkBase(BaseModel):
    """Base chunk model."""

    file_id: int
    chunk_index: int
    content: str
    content_hash: str
    header_path: list[str] = Field(default_factory=list)
    section_level: int = 0


class ChunkCreate(ChunkBase):
    """Create chunk model."""

    embedding: list[float] | None = None


class Chunk(ChunkBase):
    """Chunk model."""

    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class ChunkWithEmbedding(Chunk):
    """Chunk with embedding vector."""

    embedding: list[float] | None = None

"""Pydantic models for search operations."""

from enum import Enum

from pydantic import BaseModel, Field


class SearchMode(str, Enum):
    """Search modes."""

    HYBRID = "hybrid"  # Vector + Fulltext with RRF
    VECTOR = "vector"  # Semantic search only
    FULLTEXT = "fulltext"  # Keyword search only


class SearchQuery(BaseModel):
    """Search query model."""

    query: str
    search_mode: SearchMode = SearchMode.HYBRID
    limit: int = Field(default=10, ge=1, le=100)
    category_filter: str | None = None
    tag_filter: list[str] = Field(default_factory=list)
    file_path: str | None = None  # Search within specific file


class SearchResult(BaseModel):
    """Search result model."""

    chunk_id: int
    file_id: int
    file_path: str
    file_title: str
    file_category: str
    content: str
    header_path: list[str]
    section_level: int
    score: float
    search_mode: SearchMode

    class Config:
        from_attributes = True


class SearchResponse(BaseModel):
    """Search response model."""

    query: str
    results: list[SearchResult]
    total: int
    search_mode: SearchMode

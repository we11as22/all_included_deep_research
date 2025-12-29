"""Memory system models."""

from pydantic import BaseModel, Field


class MemorySearchRequest(BaseModel):
    """Memory search request."""

    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results")
    min_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum relevance score")


class MemorySearchResult(BaseModel):
    """Single memory search result."""

    chunk_id: int
    file_path: str
    file_title: str
    content: str
    score: float
    header_path: list[str] = Field(default_factory=list)


class MemorySearchResponse(BaseModel):
    """Memory search response."""

    query: str
    results: list[MemorySearchResult]
    total: int


class MemoryCreateRequest(BaseModel):
    """Create memory file request."""

    file_path: str = Field(..., description="Relative path for memory file")
    title: str = Field(..., description="File title")
    content: str = Field(..., description="Markdown content")
    tags: list[str] = Field(default_factory=list, description="Optional tags")


class MemoryFileResponse(BaseModel):
    """Memory file information."""

    file_id: int
    file_path: str
    title: str
    chunks_count: int
    created_at: str
    updated_at: str


"""Search result models."""

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Single search result."""

    title: str = Field(..., description="Page title")
    url: str = Field(..., description="Page URL")
    snippet: str = Field(..., description="Page snippet/description")
    score: float = Field(default=0.0, description="Relevance score")
    published_date: str | None = Field(default=None, description="Publication date if available")


class ScrapedContent(BaseModel):
    """Scraped web page content."""

    url: str = Field(..., description="Page URL")
    title: str = Field(..., description="Page title")
    content: str = Field(..., description="Cleaned text content")
    markdown: str | None = Field(default=None, description="Content in markdown format")
    html: str | None = Field(default=None, description="Original HTML")
    images: list[str] = Field(default_factory=list, description="Image URLs")
    links: list[str] = Field(default_factory=list, description="Outbound links")


class SearchResponse(BaseModel):
    """Complete search response."""

    query: str = Field(..., description="Original search query")
    results: list[SearchResult] = Field(default_factory=list, description="Search results")
    total_results: int = Field(default=0, description="Total number of results found")

"""Configuration models."""

from pydantic import BaseModel, Field


class ConfigResponse(BaseModel):
    """Application configuration response."""

    search_provider: str = Field(..., description="Search provider (tavily, searxng)")
    embedding_provider: str = Field(..., description="Embedding provider (openai, ollama, etc)")
    speed_max_iterations: int = Field(..., description="Max iterations for speed mode")
    balanced_max_iterations: int = Field(..., description="Max iterations for balanced mode")
    quality_max_iterations: int = Field(..., description="Max iterations for quality mode")
    memory_enabled: bool = Field(..., description="Whether memory system is enabled")


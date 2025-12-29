"""Research-specific models."""

from enum import Enum

from pydantic import BaseModel, Field


class ResearchMode(str, Enum):
    """Research operation modes."""

    SPEED = "speed"
    BALANCED = "balanced"
    QUALITY = "quality"


class ResearchRequest(BaseModel):
    """Research request."""

    query: str = Field(..., description="Research question or topic")
    mode: ResearchMode = Field(default=ResearchMode.BALANCED, description="Research depth mode")
    save_to_memory: bool = Field(default=True, description="Save results to memory system")


class ResearchResponse(BaseModel):
    """Research response."""

    session_id: str = Field(..., description="Research session identifier")
    query: str = Field(..., description="Original query")
    mode: ResearchMode = Field(..., description="Research mode used")
    report: str | None = Field(None, description="Final research report")
    sources_count: int = Field(default=0, description="Number of sources consulted")
    findings_count: int = Field(default=0, description="Number of findings generated")


"""LangGraph state definitions for research workflows."""

from typing import Annotated, Any, Literal

from langgraph.graph import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


def override_reducer(existing: Any, new: Any) -> Any:
    """
    Override reducer for complete state replacement.

    This reducer allows nodes to completely replace a state field
    by providing a dict with 'type': 'override' and 'value': <new_value>.

    Args:
        existing: Current value
        new: New value (can be dict with override instruction)

    Returns:
        New value if override, otherwise appended value
    """
    if isinstance(new, dict) and new.get("type") == "override":
        return new.get("value")
    return new


# ===================================================================
# Main Research State
# ===================================================================


class MemoryContext(BaseModel):
    """Memory search context."""

    chunk_id: int
    file_path: str
    file_title: str
    content: str
    score: float
    header_path: list[str] = Field(default_factory=list)


class SourceReference(BaseModel):
    """Source reference with metadata."""

    url: str
    title: str
    snippet: str
    relevance_score: float = 0.0


class ResearchFinding(BaseModel):
    """Individual research finding."""

    researcher_id: str
    topic: str
    summary: str
    key_findings: list[str]
    sources: list[SourceReference]
    confidence: Literal["low", "medium", "high"] = "medium"


class ResearchState(TypedDict, total=False):
    """Main state for research workflow."""

    # Input
    query: str  # User's research question
    mode: Literal["speed", "balanced", "quality"]  # Research mode
    clarification_needed: bool  # Whether clarification is needed
    clarified_query: str | None  # Clarified version of query

    # Memory context
    memory_context: Annotated[list[MemoryContext], override_reducer]

    # Research planning
    research_plan: Annotated[str | None, override_reducer]  # High-level plan
    research_topics: Annotated[list[str], override_reducer]  # Specific topics to research

    # Messages (conversation history)
    messages: Annotated[list[dict], add_messages]

    # Research findings
    findings: Annotated[list[ResearchFinding], override_reducer]

    # Compressed research (intermediate)
    compressed_research: Annotated[str | None, override_reducer]

    # Final output
    final_report: Annotated[str | None, override_reducer]

    # Metadata
    iterations: int  # Current iteration count
    max_iterations: int  # Maximum iterations allowed
    max_concurrent_researchers: int  # Max parallel researchers

    # Streaming
    stream: Any
    suppress_final_report_stream: bool


# ===================================================================
# Supervisor State (Subgraph)
# ===================================================================


class SupervisorState(TypedDict, total=False):
    """State for supervisor subgraph."""

    # Input from parent
    query: str
    memory_context: list[MemoryContext]
    research_plan: str | None
    current_findings: list[ResearchFinding]

    # Messages for supervisor
    messages: Annotated[list[dict], add_messages]

    # Supervisor decisions
    research_topics: Annotated[list[str], override_reducer]  # Topics to assign to researchers
    supervisor_reasoning: Annotated[str | None, override_reducer]  # Supervisor's thinking

    # Tool calls (for delegation)
    pending_tool_calls: Annotated[list[dict], override_reducer]

    # Iteration control
    iteration: int
    max_iterations: int


# ===================================================================
# Researcher State (Subgraph)
# ===================================================================


class ResearcherState(TypedDict, total=False):
    """State for individual researcher subgraph."""

    # Input assignment
    researcher_id: str
    research_topic: str
    focus_areas: list[str]
    max_sources: int
    max_query_rounds: int
    queries_per_round: int

    # Context from parent
    memory_context: list[MemoryContext]
    existing_findings: list[ResearchFinding]  # Findings from other researchers

    # Messages for researcher
    messages: Annotated[list[dict], add_messages]

    # Research process
    search_queries: Annotated[list[str], override_reducer]  # Queries to execute
    sources_found: Annotated[list[SourceReference], override_reducer]  # Sources discovered
    analysis: Annotated[str | None, override_reducer]  # Researcher's analysis

    # Output
    summary: Annotated[str | None, override_reducer]  # Research summary
    key_findings: Annotated[list[str], override_reducer]  # Key insights
    confidence_level: Annotated[Literal["low", "medium", "high"], override_reducer]

    # Status
    completed: bool

    # Streaming
    stream: Any


# ===================================================================
# Output State Schemas (for subgraph returns)
# ===================================================================


class SupervisorOutput(BaseModel):
    """Output from supervisor subgraph."""

    research_topics: list[str] = Field(..., description="Topics assigned to researchers")
    supervisor_reasoning: str | None = Field(None, description="Supervisor's strategic thinking")
    findings: list[ResearchFinding] = Field(
        default_factory=list, description="Aggregated findings from all researchers"
    )


class ResearcherOutput(BaseModel):
    """Output from researcher subgraph."""

    researcher_id: str
    research_topic: str
    summary: str
    key_findings: list[str]
    sources: list[SourceReference]
    confidence_level: Literal["low", "medium", "high"] = "medium"

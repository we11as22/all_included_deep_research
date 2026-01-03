"""ConductResearch tool for supervisor to delegate research tasks."""

from typing import Annotated

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class ConductResearchInput(BaseModel):
    """Input for ConductResearch tool."""

    research_topic: str = Field(
        ...,
        description="Specific research topic or question for the researcher to investigate",
    )
    focus_areas: list[str] = Field(
        default_factory=list,
        description="Specific focus areas or subtopics to explore",
    )
    max_sources: int = Field(
        default=5,
        description="Maximum number of sources to search and analyze",
        ge=1,
        le=10,
    )


class ConductResearchOutput(BaseModel):
    """Output from ConductResearch tool."""

    researcher_id: str = Field(..., description="ID of the researcher assigned to this task")
    topic: str = Field(..., description="Research topic assigned")
    status: str = Field(default="assigned", description="Status of the research task")


@tool
def conduct_research_tool(
    research_topic: Annotated[str, "Specific research topic or question to investigate"],
    focus_areas: Annotated[
        list[str], "Specific focus areas or subtopics to explore"
    ] = [],
    max_sources: Annotated[int, "Maximum number of sources to search"] = 5,
) -> str:
    """
    Delegate a research task to a specialized researcher agent.

    This tool is used by the supervisor to assign research tasks to researcher agents.
    Each researcher will independently investigate the topic and return findings.

    Use this tool when you need to:
    - Gather information on a specific subtopic
    - Investigate different perspectives on an issue
    - Find evidence or examples for a claim
    - Explore related concepts in depth

    The researcher will:
    1. Search for relevant sources
    2. Extract and analyze key information
    3. Synthesize findings into a coherent summary
    4. Return structured research results

    Args:
        research_topic: Clear, specific question or topic for research
        focus_areas: List of specific aspects to focus on (optional)
        max_sources: Maximum number of sources to analyze (1-10)

    Returns:
        Confirmation message that research task has been delegated
    """
    # This is a placeholder - actual delegation happens in the workflow graph
    # The tool call itself signals to the graph to spawn a researcher subgraph
    return f"Research task assigned: {research_topic}"

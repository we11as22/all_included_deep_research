"""ResearchComplete tool for researcher to signal completion."""

from typing import Annotated

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class ResearchCompleteInput(BaseModel):
    """Input for ResearchComplete tool."""

    summary: str = Field(
        ...,
        description="Concise summary of research findings (2-3 paragraphs)",
    )
    key_findings: list[str] = Field(
        ...,
        description="List of key findings or insights discovered",
        min_length=1,
        max_length=10,
    )
    sources_used: list[str] = Field(
        default_factory=list,
        description="List of source URLs consulted",
    )
    confidence_level: str = Field(
        default="medium",
        description="Confidence level in findings: low, medium, high",
    )


@tool
def research_complete_tool(
    summary: Annotated[str, "Concise summary of research findings (2-3 paragraphs)"],
    key_findings: Annotated[list[str], "List of key findings or insights discovered"],
    sources_used: Annotated[list[str], "List of source URLs consulted"] = [],
    confidence_level: Annotated[
        str, "Confidence level in findings: low, medium, high"
    ] = "medium",
) -> str:
    """
    Signal completion of research task and return findings.

    Use this tool when you have finished investigating your assigned research topic
    and are ready to report your findings back to the supervisor.

    Before calling this tool, ensure you have:
    - Searched and analyzed relevant sources
    - Extracted key information and insights
    - Synthesized findings into a coherent summary
    - Identified the most important takeaways

    Args:
        summary: Clear, concise summary of what you learned (2-3 paragraphs)
        key_findings: Bullet points of the most important discoveries
        sources_used: URLs of sources you consulted (for citations)
        confidence_level: How confident you are in your findings (low/medium/high)

    Returns:
        Confirmation that research has been submitted
    """
    # This is a placeholder - actual handling happens in the workflow graph
    # The tool call signals to the graph that the researcher is done
    num_findings = len(key_findings)
    num_sources = len(sources_used)
    return f"Research complete: {num_findings} findings from {num_sources} sources (confidence: {confidence_level})"

"""Pydantic schemas for structured LLM outputs."""

from typing import Literal, Optional, Any
from pydantic import BaseModel, Field, ConfigDict


class AgentAction(BaseModel):
    """Structured output for agent actions."""
    
    model_config = ConfigDict(extra='forbid')
    
    action: Literal[
        "web_search",
        "scrape_urls",
        "scroll_page",
        "write_note",
        "update_note",
        "read_note",
        "add_todo",
        "update_todo",
        "complete_todo",
        "read_shared_notes",
        "read_agent_file",
        "read_main",
        "finish"
    ] = Field(..., description="Action to perform")
    
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="Action arguments as key-value pairs"
    )


class SupervisorAction(BaseModel):
    """Structured output for supervisor ReAct actions."""
    
    model_config = ConfigDict(extra='forbid')
    
    action: Literal[
        "plan_tasks",
        "create_agent",
        "write_to_main",
        "read_agent_file",
        "update_agent_todo",
        "read_note",
        "write_note",
        "read_main"
    ] = Field(..., description="Supervisor action to perform")
    
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="Action arguments"
    )


class SupervisorTasks(BaseModel):
    """Structured output for supervisor task planning."""
    
    tasks: list[str] = Field(
        ...,
        description="List of research tasks/topics",
        min_length=0,
        max_length=10
    )
    
    stop: bool = Field(
        default=False,
        description="Whether to stop generating more tasks"
    )
    
    reasoning: Optional[str] = Field(
        default=None,
        description="Reasoning behind the task selection"
    )


class SearchQueries(BaseModel):
    """Structured output for search query generation."""
    
    queries: list[str] = Field(
        ...,
        description="List of search queries",
        min_length=1,
        max_length=10
    )


class ResearchAnalysis(BaseModel):
    """Structured output for research analysis and synthesis."""
    
    summary: str = Field(
        ...,
        description="Comprehensive summary of research findings (2-3 paragraphs)",
        min_length=100
    )
    
    key_findings: list[str] = Field(
        ...,
        description="List of key findings (3-5 items)",
        min_length=1,
        max_length=10
    )
    
    confidence: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Confidence level based on source quality"
    )


class QueryRewrite(BaseModel):
    """Structured output for query rewriting."""
    
    rewritten_query: str = Field(
        ...,
        description="Rewritten search query",
        min_length=1
    )


class FollowupQueries(BaseModel):
    """Structured output for follow-up query generation."""
    
    queries: list[str] = Field(
        ...,
        description="List of follow-up search queries",
        min_length=0,
        max_length=5
    )


class SummarizedContent(BaseModel):
    """Structured output for content summarization."""
    
    summary: str = Field(
        ...,
        description="Summarized content",
        min_length=50
    )
    
    key_points: list[str] = Field(
        default_factory=list,
        description="Key points extracted from content"
    )


class SynthesizedAnswer(BaseModel):
    """Structured output for answer synthesis."""
    
    answer: str = Field(
        ...,
        description="Synthesized answer to the query",
        min_length=100
    )
    
    key_points: list[str] = Field(
        default_factory=list,
        description="Key points in the answer"
    )


class CompressedFindings(BaseModel):
    """Structured output for findings compression."""
    
    compressed_summary: str = Field(
        ...,
        description="Compressed summary of all findings",
        min_length=200
    )
    
    key_themes: list[str] = Field(
        ...,
        description="Key themes identified across findings",
        min_length=1
    )
    
    important_sources: list[str] = Field(
        default_factory=list,
        description="URLs of most important sources"
    )


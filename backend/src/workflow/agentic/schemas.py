"""Pydantic schemas for structured LLM outputs."""

from typing import Literal, Optional, Any
from pydantic import BaseModel, Field, ConfigDict


class AgentAction(BaseModel):
    """Structured output for agent actions."""
    
    model_config = ConfigDict(extra='forbid')
    
    reasoning: str = Field(..., description="Brief reasoning for the chosen action")

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
    
    reasoning: str = Field(..., description="Brief reasoning for the chosen action")

    action: Literal[
        "plan_tasks",
        "create_agent",
        "write_to_main",
        "read_agent_file",
        "update_agent_todo",
        "update_agent_todos",
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
    
    reasoning: str = Field(..., description="Reasoning behind the task selection")

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
    

class SearchQueries(BaseModel):
    """Structured output for search query generation."""
    
    reasoning: str = Field(..., description="Why these queries cover the topic")

    queries: list[str] = Field(
        ...,
        description="List of search queries",
        min_length=1,
        max_length=10
    )


class ResearchAnalysis(BaseModel):
    """Structured output for research analysis and synthesis."""
    
    reasoning: str = Field(..., description="Why the evidence supports this analysis")

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
    
    reasoning: str = Field(..., description="Why this rewrite best fits the intent")

    rewritten_query: str = Field(
        ...,
        description="Rewritten search query",
        min_length=1
    )


class FollowupQueries(BaseModel):
    """Structured output for follow-up query generation."""
    
    reasoning: str = Field(..., description="Why follow-up queries are or are not needed")

    should_continue: bool = Field(
        ...,
        description="Whether additional queries are needed to close gaps",
    )

    gap_summary: str = Field(
        ...,
        description="Brief summary of the remaining gaps or coverage sufficiency",
    )

    queries: list[str] = Field(
        ...,
        description="List of follow-up search queries",
        min_length=0,
        max_length=5
    )


class SummarizedContent(BaseModel):
    """Structured output for content summarization."""
    
    reasoning: str = Field(..., description="Why these points are most relevant")

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
    
    reasoning: str = Field(..., description="Why the answer follows from the evidence")

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
    
    reasoning: str = Field(..., description="Why these themes and sources were prioritized")

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


class GapTopics(BaseModel):
    """Structured output for gap topic generation."""

    reasoning: str = Field(..., description="Why additional topics are needed or not")

    topics: list[str] = Field(
        default_factory=list,
        description="Additional research topics to cover gaps",
        min_length=0,
        max_length=5,
    )


class FinalReport(BaseModel):
    """Structured output for final report generation."""

    reasoning: str = Field(..., description="Why the report structure and citations are sufficient")

    report: str = Field(
        ...,
        description="Final report text with sections and citations",
        min_length=200,
    )


class TodoItemSchema(BaseModel):
    """Strict todo item schema for agent plans."""

    reasoning: str = Field(..., description="Why this task is necessary")
    title: str = Field(..., description="Short task title")
    objective: str = Field(..., description="What the task should achieve")
    expected_output: str = Field(..., description="Concrete deliverable expected")
    sources_needed: list[str] = Field(default_factory=list, description="Types of sources to consult")
    priority: Literal["low", "medium", "high"] = Field(default="medium")
    status: Literal["pending", "in_progress", "done"] = Field(default="pending")
    note: str | None = Field(default=None, description="Optional task notes")
    url: str | None = Field(default=None, description="Optional related URL")


class TodoUpdateSchema(BaseModel):
    """Strict todo update schema."""

    reasoning: str = Field(..., description="Why this update is needed")
    title: str = Field(..., description="Task title to update")
    status: Literal["pending", "in_progress", "done"] | None = Field(default=None)
    note: str | None = Field(default=None)
    objective: str | None = Field(default=None)
    expected_output: str | None = Field(default=None)
    sources_needed: list[str] | None = Field(default=None)
    priority: Literal["low", "medium", "high"] | None = Field(default=None)
    url: str | None = Field(default=None)

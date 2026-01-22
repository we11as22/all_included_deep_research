"""Pydantic schemas for structured LLM outputs used across the application.

These schemas are used by ChatSearchService for web and deep search modes.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


class QueryRewrite(BaseModel):
    """Structured output for query rewriting."""

    reasoning: str = Field(..., description="Why this rewrite best fits the intent")

    rewritten_query: str = Field(
        ...,
        description="Rewritten search query",
        min_length=1
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
    
    model_config = ConfigDict(
        # Force key_points to be in required array for Azure/OpenRouter compatibility
        json_schema_extra={
            "required": ["summary", "key_points"]
        }
    )

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
    
    model_config = ConfigDict(
        # Force key_points to be in required array for Azure/OpenRouter compatibility
        # Even though it has a default, some providers require all properties in required
        json_schema_extra={
            "required": ["reasoning", "answer", "key_points"]
        }
    )

    reasoning: str = Field(..., description="Why the answer follows from the evidence")

    answer: str = Field(
        ...,
        description="Synthesized answer to the query in markdown format. MUST use proper markdown: ## for main sections (NOT #), ### for subsections, **bold**, *italic*, lists, links. Do NOT use plain text with large letters - use markdown headings! CRITICAL: Answer must be comprehensive (600-1500 words minimum) and fully formatted in markdown!",
        min_length=400  # Increased from 200 to ensure more comprehensive answers
    )

    # key_points has default but must be in required array for Azure/OpenRouter
    key_points: list[str] = Field(
        default_factory=list,
        description="Key points in the answer"
    )


class ChatTitle(BaseModel):
    """Structured output for chat title generation."""

    title: str = Field(
        ...,
        description="Concise, descriptive title for the conversation (max 60 characters)",
        min_length=1,
        max_length=60
    )


class ScrapedPageAnalysis(BaseModel):
    """Structured output for analyzing scraped page content."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "required": ["reasoning", "summary", "brief_info", "is_relevant"]
        }
    )
    
    reasoning: str = Field(
        ...,
        description="Your reasoning about the page content and its relevance to the research task. Explain what information you found, how it relates to the task, and why you made the decisions about summary and relevance.",
        min_length=50
    )
    
    summary: str = Field(
        ...,
        description="Comprehensive summary of the page content focused on the research task. This will be used for creating findings. Include all relevant facts, data, and insights. If content has markdown structure, preserve it. Target: 2000-4000 tokens.",
        min_length=200
    )
    
    brief_info: str = Field(
        ...,
        description="Brief description (2-3 sentences, max 200 chars) of what information is on this page. This will be shown in tool history to help agent understand what pages contain without storing full summary.",
        max_length=200
    )
    
    is_relevant: bool = Field(
        ...,
        description="Whether this page's content is relevant and needed for completing the research task. True if page contains information directly related to the task, False if it's not relevant or only tangentially related."
    )


class FindingContent(BaseModel):
    """Structured output for creating finding from scraped summaries and search snippets."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "required": ["summary", "key_findings"]
        }
    )
    
    summary: str = Field(
        ...,
        description="Comprehensive, detailed finding summary (2000-4000 words minimum, longer if many sources available) based on ALL scraped page summaries and search result snippets. MUST include ALL relevant facts, data, insights, comparisons, statistics, examples, and context from ALL available sources. Use proper markdown formatting with sections, subsections, lists, and emphasis. The more sources available, the longer and more detailed the summary should be.",
        min_length=1500
    )
    
    key_findings: list[str] = Field(
        ...,
        description="List of 8-15 key findings extracted from the content. Each finding should be a specific fact, insight, or data point.",
        min_items=5,
        max_items=20
    )


class NoteContent(BaseModel):
    """Structured output for creating note from scraped summaries and search snippets."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "required": ["title", "summary"]
        }
    )
    
    title: str = Field(
        ...,
        description="Concise, descriptive title for the note (max 100 chars)",
        max_length=100
    )
    
    summary: str = Field(
        ...,
        description="Comprehensive note content (500-2000 words) based on scraped page summaries and search result snippets. Include all relevant information, facts, and context.",
        min_length=500
    )

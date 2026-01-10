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
        description="Synthesized answer to the query",
        min_length=100
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

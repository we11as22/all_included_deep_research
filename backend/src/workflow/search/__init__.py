"""Search workflow module (Perplexica-style two-stage architecture).

Exports:
- SearchService: Main entry point for all search modes
- classify_query: Query classifier
- research_agent: Research agent with tool-calling
- writer_agent: Writer agent with citations
- ActionRegistry: Tool registry
"""

from src.workflow.search.classifier import classify_query, QueryClassification
from src.workflow.search.researcher import research_agent
from src.workflow.search.writer import writer_agent, CitedAnswer
from src.workflow.search.actions import ActionRegistry
from src.workflow.search.service import SearchService, create_search_service

__all__ = [
    "SearchService",
    "create_search_service",
    "classify_query",
    "QueryClassification",
    "research_agent",
    "writer_agent",
    "CitedAnswer",
    "ActionRegistry",
]

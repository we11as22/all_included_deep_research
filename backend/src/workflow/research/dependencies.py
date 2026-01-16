"""Dependency injection container for research workflows."""

from dataclasses import dataclass
from typing import Any

from src.workflow.research.session.manager import SessionManager


@dataclass
class ResearchDependencies:
    """Container for all research workflow dependencies.

    This replaces the context variables approach with explicit dependency injection.
    All nodes receive this container in their constructor.
    """

    # Core LLM and search
    llm: Any
    search_provider: Any
    scraper: Any

    # Streaming
    stream: Any

    # Agent memory (for agent_sessions)
    agent_memory_service: Any
    agent_file_service: Any

    # Database
    session_factory: Any
    session_manager: SessionManager

    # Configuration
    settings: Any

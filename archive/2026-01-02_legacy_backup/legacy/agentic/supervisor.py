"""Deprecated shim: supervisor logic now lives in AgenticResearchCoordinator."""

from __future__ import annotations

from typing import Any

from src.workflow.agentic.coordinator import AgenticResearchCoordinator, get_supervisor_system_prompt
from src.workflow.agentic.models import SharedResearchMemory
from src.memory.agent_memory_service import AgentMemoryService
from src.memory.agent_file_service import AgentFileService


class AgenticSupervisor(AgenticResearchCoordinator):
    """Backward-compatible wrapper for the unified lead agent."""

    def __init__(
        self,
        llm: Any,
        shared_memory: SharedResearchMemory,
        memory_context: list[Any],
        chat_history: list[dict[str, str]] | None = None,
        agent_memory_service: AgentMemoryService | None = None,
        agent_file_service: AgentFileService | None = None,
    ) -> None:
        super().__init__(
            llm=llm,
            search_provider=None,
            web_scraper=None,
            memory_context=memory_context,
            chat_history=chat_history,
            stream=None,
            max_rounds=1,
            max_concurrent=1,
            max_sources=1,
            agent_memory_service=agent_memory_service,
            agent_file_service=agent_file_service,
        )
        if shared_memory is not None:
            self.shared_memory = shared_memory


__all__ = ["AgenticSupervisor", "get_supervisor_system_prompt"]

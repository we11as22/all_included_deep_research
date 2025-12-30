"""Coordinator for agentic deep research."""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from src.search.base import SearchProvider
from src.search.scraper import WebScraper
from src.workflow.agentic.models import SharedResearchMemory
from src.workflow.agentic.researcher import AgenticResearcher
from src.workflow.agentic.supervisor import AgenticSupervisor
from src.workflow.state import ResearchFinding
from src.utils.chat_history import format_chat_history
from src.memory.agent_memory_service import AgentMemoryService
from src.memory.agent_file_service import AgentFileService

logger = structlog.get_logger(__name__)


class AgenticResearchCoordinator:
    """Run supervisor + agentic researchers with shared memory."""

    def __init__(
        self,
        llm: Any,
        search_provider: SearchProvider,
        web_scraper: WebScraper,
        memory_context: list[Any],
        chat_history: list[dict[str, str]] | None = None,
        stream: Any | None = None,
        max_rounds: int = 3,
        max_concurrent: int = 4,
        max_sources: int = 10,
        agent_memory_service: AgentMemoryService | None = None,
        agent_file_service: AgentFileService | None = None,
    ) -> None:
        self.llm = llm
        self.search_provider = search_provider
        self.web_scraper = web_scraper
        self.memory_context = memory_context
        self.chat_history = chat_history or []
        self.stream = stream
        self.max_rounds = max_rounds
        self.max_concurrent = max_concurrent
        self.max_sources = max_sources
        self.agent_memory_service = agent_memory_service
        self.agent_file_service = agent_file_service
        self.shared_memory = SharedResearchMemory()
        self.supervisor = AgenticSupervisor(
            llm=llm,
            shared_memory=self.shared_memory,
            memory_context=memory_context,
            chat_history=self.chat_history,
            agent_memory_service=agent_memory_service,
            agent_file_service=agent_file_service,
        )
        self.query = ""

    async def run(self, query: str, seed_tasks: list[str] | None = None) -> list[ResearchFinding]:
        self.query = query
        self._reset_session()
        findings: list[ResearchFinding] = []
        
        # Limit seed_tasks to max_concurrent if provided
        if seed_tasks:
            tasks = seed_tasks[:self.max_concurrent]
        else:
            tasks = await self.supervisor.initial_tasks(query, max_tasks=self.max_concurrent)

        try:
            round_id = 0
            while tasks and round_id < self.max_rounds:
                if self.stream:
                    self.stream.emit_status(f"Supervisor assigned {len(tasks)} tasks", step="supervisor")
                    self.stream.emit_search_queries(tasks, label="supervisor_tasks")

                round_findings, agent_statuses = await self._run_round(tasks, round_id, existing_findings=findings)
                findings.extend(round_findings)
                
                # Write project status to main.md after each round
                if self.supervisor.agent_memory_service:
                    await self.supervisor.write_project_status(
                        query=query,
                        findings=findings,
                        active_agents=agent_statuses,
                    )

                round_id += 1
                if round_id >= self.max_rounds:
                    break

                # Limit gap tasks to max_concurrent
                gap_tasks = await self.supervisor.gap_tasks(query, findings, max_tasks=self.max_concurrent)
                tasks = gap_tasks[:self.max_concurrent] if gap_tasks else []

            return findings
        finally:
            self.shared_memory.clear()

    async def _run_round(
        self,
        tasks: list[str],
        round_id: int,
        existing_findings: list[ResearchFinding],
    ) -> tuple[list[ResearchFinding], dict[str, dict[str, Any]]]:
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def run_task(idx: int, topic: str) -> tuple[ResearchFinding | None, str, dict[str, Any]]:
            agent_id = f"agent_r{round_id}_{idx}"
            async with semaphore:
                researcher = AgenticResearcher(
                    llm=self.llm,
                    search_provider=self.search_provider,
                    web_scraper=self.web_scraper,
                    shared_memory=self.shared_memory,
                    memory_context=self.memory_context,
                    chat_history=self.chat_history,
                    stream=self.stream,
                    max_steps=6,
                    max_sources=self.max_sources,
                    agent_memory_service=self.agent_memory_service,
                    agent_file_service=self.agent_file_service,
                )
                assignment = self._build_assignment(topic)
                
                # Get agent character and preferences from supervisor if available
                character, preferences = await self._get_agent_character_preferences(agent_id, topic)
                
                result = await researcher.run(
                    agent_id=agent_id,
                    topic=topic,
                    existing_findings=existing_findings,
                    assignment=assignment,
                    character=character,
                    preferences=preferences,
                )
                self._share_finding(result.finding)
                
                # Collect agent status
                agent_status = {
                    "todos": [
                        {
                            "title": todo.title,
                            "status": todo.status,
                            "note": todo.note,
                            "url": todo.url,
                        }
                        for todo in result.memory.todos
                    ],
                    "notes": [
                        {
                            "title": note.title,
                            "summary": note.summary[:100],
                            "urls": note.urls,
                        }
                        for note in result.memory.notes
                    ],
                }
                
                return result.finding, agent_id, agent_status

        # Limit tasks to max_concurrent to prevent too many agents
        limited_tasks = tasks[:self.max_concurrent]
        
        results = await asyncio.gather(
            *[run_task(idx, topic) for idx, topic in enumerate(limited_tasks)],
            return_exceptions=True,
        )

        findings: list[ResearchFinding] = []
        agent_statuses: dict[str, dict[str, Any]] = {}
        
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Agent task failed", task=tasks[idx] if idx < len(tasks) else "unknown", error=str(result))
                continue
            if result:
                finding, agent_id, agent_status = result
                if finding:
                    findings.append(finding)
                agent_statuses[agent_id] = agent_status

        return findings, agent_statuses

    def _share_finding(self, finding: ResearchFinding) -> None:
        if not finding.summary:
            return
        try:
            from src.workflow.agentic.models import AgentNote

            shared_note = AgentNote(
                title=f"Finding: {finding.topic}",
                summary=finding.summary[:600],
                urls=[source.url for source in finding.sources[:3]],
                tags=["finding"],
            )
            self.shared_memory.add_note(shared_note)
        except Exception as exc:
            logger.warning("Shared note creation failed", error=str(exc))

    def _reset_session(self) -> None:
        self.shared_memory = SharedResearchMemory()
        self.supervisor = AgenticSupervisor(
            llm=self.llm,
            shared_memory=self.shared_memory,
            memory_context=self.memory_context,
            chat_history=self.chat_history,
        )

    def _build_assignment(self, topic: str) -> str:
        query = self.query or "unknown"
        chat_block = format_chat_history(self.chat_history, limit=len(self.chat_history))
        return (
            f"Primary query: {query}\n"
            f"Your focus: {topic}\n"
            f"{chat_block}\n"
            "Capture high-signal evidence with citations. Share notes that help other agents."
        )

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

logger = structlog.get_logger(__name__)


class AgenticResearchCoordinator:
    """Run supervisor + agentic researchers with shared memory."""

    def __init__(
        self,
        llm: Any,
        search_provider: SearchProvider,
        web_scraper: WebScraper,
        memory_context: list[Any],
        stream: Any | None = None,
        max_rounds: int = 3,
        max_concurrent: int = 4,
        max_sources: int = 10,
    ) -> None:
        self.llm = llm
        self.search_provider = search_provider
        self.web_scraper = web_scraper
        self.memory_context = memory_context
        self.stream = stream
        self.max_rounds = max_rounds
        self.max_concurrent = max_concurrent
        self.max_sources = max_sources
        self.shared_memory = SharedResearchMemory()
        self.supervisor = AgenticSupervisor(
            llm=llm,
            shared_memory=self.shared_memory,
            memory_context=memory_context,
        )
        self.query = ""

    async def run(self, query: str, seed_tasks: list[str] | None = None) -> list[ResearchFinding]:
        self.query = query
        self._reset_session()
        findings: list[ResearchFinding] = []
        tasks = seed_tasks or await self.supervisor.initial_tasks(query, max_tasks=self.max_concurrent + 1)

        try:
            round_id = 0
            while tasks and round_id < self.max_rounds:
                if self.stream:
                    self.stream.emit_status(f"Supervisor assigned {len(tasks)} tasks", step="supervisor")
                    self.stream.emit_search_queries(tasks, label="supervisor_tasks")

                round_findings = await self._run_round(tasks, round_id, existing_findings=findings)
                findings.extend(round_findings)

                round_id += 1
                if round_id >= self.max_rounds:
                    break

                tasks = await self.supervisor.gap_tasks(query, findings, max_tasks=self.max_concurrent)

            return findings
        finally:
            self.shared_memory.clear()

    async def _run_round(
        self,
        tasks: list[str],
        round_id: int,
        existing_findings: list[ResearchFinding],
    ) -> list[ResearchFinding]:
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def run_task(idx: int, topic: str) -> ResearchFinding | None:
            agent_id = f"agent_r{round_id}_{idx}"
            async with semaphore:
                researcher = AgenticResearcher(
                    llm=self.llm,
                    search_provider=self.search_provider,
                    web_scraper=self.web_scraper,
                    shared_memory=self.shared_memory,
                    memory_context=self.memory_context,
                    stream=self.stream,
                    max_steps=6,
                    max_sources=self.max_sources,
                )
                assignment = self._build_assignment(topic)
                result = await researcher.run(
                    agent_id=agent_id,
                    topic=topic,
                    existing_findings=existing_findings,
                    assignment=assignment,
                )
                self._share_finding(result.finding)
                return result.finding

        results = await asyncio.gather(
            *[run_task(idx, topic) for idx, topic in enumerate(tasks)],
            return_exceptions=True,
        )

        findings: list[ResearchFinding] = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Agent task failed", task=tasks[idx], error=str(result))
                continue
            if result:
                findings.append(result)

        return findings

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
        )

    def _build_assignment(self, topic: str) -> str:
        query = self.query or "unknown"
        return (
            f"Primary query: {query}\n"
            f"Your focus: {topic}\n"
            "Capture high-signal evidence with citations. Share notes that help other agents."
        )

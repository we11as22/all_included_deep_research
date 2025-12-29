"""Agentic supervisor for delegating research tasks."""

from __future__ import annotations

import json
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.workflow.agentic.models import SharedResearchMemory
from src.workflow.nodes.memory_search import format_memory_context_for_prompt
from src.workflow.state import ResearchFinding
from src.utils.chat_history import format_chat_history

logger = structlog.get_logger(__name__)


SUPERVISOR_SYSTEM_PROMPT = """You are the lead research supervisor.

Return JSON only:
{"tasks": ["topic1", "topic2"], "stop": false, "reasoning": "..."}

Rules:
- Tasks must be short, specific research directions.
- Avoid duplicating existing topics or findings.
- If coverage is sufficient, set stop to true and tasks to [].
"""


class AgenticSupervisor:
    """Supervisor for generating and updating research tasks."""

    def __init__(
        self,
        llm: Any,
        shared_memory: SharedResearchMemory,
        memory_context: list[Any],
        chat_history: list[dict[str, str]] | None = None,
    ) -> None:
        self.llm = llm
        self.shared_memory = shared_memory
        self.memory_context = memory_context
        self.chat_history = chat_history or []

    async def initial_tasks(self, query: str, max_tasks: int = 5) -> list[str]:
        if getattr(self.llm, "_llm_type", "") == "mock-chat":
            return [query]

        prompt = self._build_prompt(
            title="Initial task planning",
            query=query,
            findings=[],
            max_tasks=max_tasks,
        )
        return await self._run_prompt(prompt, max_tasks)

    async def gap_tasks(
        self,
        query: str,
        findings: list[ResearchFinding],
        max_tasks: int = 4,
    ) -> list[str]:
        if getattr(self.llm, "_llm_type", "") == "mock-chat":
            return []

        prompt = self._build_prompt(
            title="Gap analysis",
            query=query,
            findings=findings,
            max_tasks=max_tasks,
        )
        return await self._run_prompt(prompt, max_tasks)

    def _build_prompt(
        self,
        title: str,
        query: str,
        findings: list[ResearchFinding],
        max_tasks: int,
    ) -> str:
        memory_block = format_memory_context_for_prompt(self.memory_context[:6])
        shared_notes = self.shared_memory.render_notes(limit=8)
        findings_block = _format_findings(findings)
        chat_block = format_chat_history(self.chat_history, limit=len(self.chat_history))

        return f"""Task: {title}
Research query: {query}

{chat_block}

Memory context:
{memory_block}

Shared notes from researchers:
{shared_notes}

Existing findings:
{findings_block}

Provide up to {max_tasks} tasks in JSON only."""

    async def _run_prompt(self, prompt: str, max_tasks: int) -> list[str]:
        try:
            response = await self.llm.ainvoke(
                [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT), HumanMessage(content=prompt)]
            )
            content = response.content if hasattr(response, "content") else str(response)
            payload = json.loads(_extract_json(content))
            if payload.get("stop", False):
                return []
            tasks = payload.get("tasks") or []
            return [task.strip() for task in tasks if task.strip()][:max_tasks]
        except Exception as exc:
            logger.warning("Supervisor task generation failed", error=str(exc))
            return []


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return "{}"
    return text[start : end + 1]


def _format_findings(findings: list[ResearchFinding]) -> str:
    if not findings:
        return "None."
    lines = []
    for finding in findings[-6:]:
        lines.append(f"- {finding.topic}: {finding.summary[:200]}")
    return "\n".join(lines)

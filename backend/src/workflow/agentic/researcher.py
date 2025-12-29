"""Agentic researcher with per-agent todo list and memory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.search.base import SearchProvider
from src.search.scraper import WebScraper
from src.workflow.agentic.models import AgentMemory, SharedResearchMemory
from src.workflow.nodes.memory_search import format_memory_context_for_prompt
from src.utils.chat_history import format_chat_history
from src.workflow.state import ResearchFinding, SourceReference

logger = structlog.get_logger(__name__)


AGENTIC_SYSTEM_PROMPT = """You are a research agent with tools and a personal todo list.

You must respond with a single JSON object:
{"action": "<tool_name or finish>", "args": {...}}

Allowed actions:
- web_search: {"queries": ["..."], "max_results": 5}
- scrape_urls: {"urls": ["..."]}
- write_note: {"title": "...", "summary": "...", "urls": ["..."], "tags": ["..."], "share": true}
- add_todo: {"items": [{"title": "...", "note": "...", "url": "..."}, "..."]}
- complete_todo: {"titles": ["..."]}
- read_shared_notes: {"keyword": "...", "limit": 5}
- finish: {}

Rules:
- Prefer web_search first if you need sources.
- Use scrape_urls on the most promising links.
- Write notes for anything worth reusing later, include URLs.
- Share only high-signal notes.
- When the todo list is mostly done and you have enough evidence, use finish.
"""


@dataclass
class AgenticResearchResult:
    """Result from agentic researcher."""

    finding: ResearchFinding
    memory: AgentMemory


class AgenticResearcher:
    """ReAct-style researcher with per-agent memory and todo list."""

    def __init__(
        self,
        llm: Any,
        search_provider: SearchProvider,
        web_scraper: WebScraper,
        shared_memory: SharedResearchMemory,
        memory_context: list[Any],
        chat_history: list[dict[str, str]] | None = None,
        stream: Any | None = None,
        max_steps: int = 6,
        max_sources: int = 8,
    ) -> None:
        self.llm = llm
        self.search_provider = search_provider
        self.web_scraper = web_scraper
        self.shared_memory = shared_memory
        self.memory_context = memory_context
        self.chat_history = chat_history or []
        self.stream = stream
        self.max_steps = max_steps
        self.max_sources = max_sources
        self.agent_memory = AgentMemory()
        self.sources: list[SourceReference] = []
        self.current_topic: str | None = None

    async def run(
        self,
        agent_id: str,
        topic: str,
        existing_findings: list[ResearchFinding],
        assignment: str | None = None,
    ) -> AgenticResearchResult:
        self.current_topic = topic
        if assignment:
            self._seed_assignment(agent_id, assignment)
        self._seed_todos(topic)
        last_tool_result = "None"

        if self.stream:
            self.stream.emit_research_start(agent_id, topic)
            self.stream.emit_agent_todo(agent_id, self._todo_payload())

        if getattr(self.llm, "_llm_type", "") == "mock-chat":
            note = self.agent_memory.add_note(
                title=f"Mock findings for {topic}",
                summary=f"Mock summary for {topic}.",
                urls=[],
                tags=["mock"],
            )
            self.shared_memory.add_note(note)
            if self.stream:
                self.stream.emit_agent_note(
                    agent_id,
                    {"title": note.title, "summary": note.summary, "urls": note.urls, "shared": True},
                )
                self.stream.emit_finding(agent_id, topic, note.summary, ["Mock key point."])
            return AgenticResearchResult(
                finding=ResearchFinding(
                    researcher_id=agent_id,
                    topic=topic,
                    summary=note.summary,
                    key_findings=["Mock key point."],
                    sources=[],
                    confidence="low",
                ),
                memory=self.agent_memory,
            )

        for step in range(self.max_steps):
            prompt = self._build_prompt(
                agent_id=agent_id,
                topic=topic,
                existing_findings=existing_findings,
                last_tool_result=last_tool_result,
            )
            response = await self.llm.ainvoke(
                [SystemMessage(content=AGENTIC_SYSTEM_PROMPT), HumanMessage(content=prompt)]
            )
            content = response.content if hasattr(response, "content") else str(response)
            action, args = self._parse_action(content)
            if action == "finish":
                break

            last_tool_result = await self._execute_action(action, args, agent_id)

        finding = self._synthesize_finding(agent_id, topic)
        if self.stream:
            self.stream.emit_finding(agent_id, topic, finding.summary, finding.key_findings)
        return AgenticResearchResult(finding=finding, memory=self.agent_memory)

    def _seed_assignment(self, agent_id: str, assignment: str) -> None:
        note = self.agent_memory.add_note(
            title="Supervisor brief",
            summary=assignment.strip(),
            urls=[],
            tags=["assignment"],
        )
        self.shared_memory.add_note(note)
        if self.stream:
            self.stream.emit_agent_note(
                agent_id,
                {"title": note.title, "summary": note.summary, "urls": note.urls, "shared": True},
            )

    def _seed_todos(self, topic: str) -> None:
        if self.agent_memory.todos:
            return
        self.agent_memory.add_todo(f"Run broad search on {topic}")
        self.agent_memory.add_todo("Identify authoritative sources", note="Prefer official or primary sources")
        self.agent_memory.add_todo("Extract key facts and evidence")
        self.agent_memory.add_todo("Check for gaps or conflicting claims")

    def _build_prompt(
        self,
        agent_id: str,
        topic: str,
        existing_findings: list[ResearchFinding],
        last_tool_result: str,
    ) -> str:
        memory_block = format_memory_context_for_prompt(self.memory_context[:6])
        shared_notes = self.shared_memory.render_notes(limit=6)
        chat_block = format_chat_history(self.chat_history, limit=len(self.chat_history))
        todo_block = self.agent_memory.render_todos(limit=8)
        notes_block = self.agent_memory.render_notes(limit=6)
        findings_block = _format_findings(existing_findings)

        return f"""Agent: {agent_id}
Research topic: {topic}

{chat_block}

Memory context:
{memory_block}

Shared notes:
{shared_notes}

Your todo list:
{todo_block}

Your notes:
{notes_block}

Existing findings from other agents:
{findings_block}

Last tool result:
{last_tool_result}

Choose the next action (JSON only)."""

    async def _execute_action(self, action: str, args: dict[str, Any], agent_id: str) -> str:
        if action == "web_search":
            return await self._tool_web_search(args, agent_id)
        if action == "scrape_urls":
            return await self._tool_scrape_urls(args)
        if action == "write_note":
            return self._tool_write_note(args, agent_id)
        if action == "add_todo":
            return self._tool_add_todo(args, agent_id)
        if action == "complete_todo":
            return self._tool_complete_todo(args, agent_id)
        if action == "read_shared_notes":
            return self._tool_read_shared_notes(args)

        return "Unknown action."

    async def _tool_web_search(self, args: dict[str, Any], agent_id: str) -> str:
        queries = args.get("queries") or []
        max_results = int(args.get("max_results") or 5)
        if isinstance(queries, str):
            queries = [queries]

        if not queries:
            return "No queries provided."

        summaries = []
        for query in queries[:3]:
            response = await self.search_provider.search(query, max_results=max_results)
            for result in response.results[:max_results]:
                self.sources.append(
                    SourceReference(
                        url=result.url,
                        title=result.title,
                        snippet=result.snippet,
                        relevance_score=result.score,
                    )
                )
                if self.stream:
                    self.stream.emit_source(agent_id, {"url": result.url, "title": result.title})
            top = response.results[:3]
            summary = "; ".join([f"{item.title} ({item.url})" for item in top])
            summaries.append(f"{query}: {summary}")

        if self.current_topic:
            self.agent_memory.complete_todo(f"Run broad search on {self.current_topic}")
        if self.stream:
            self.stream.emit_agent_todo(agent_id, self._todo_payload())
        return "Search results: " + " | ".join(summaries)

    async def _tool_scrape_urls(self, args: dict[str, Any]) -> str:
        urls = args.get("urls") or []
        if isinstance(urls, str):
            urls = [urls]
        if not urls:
            return "No urls to scrape."

        snippets = []
        for url in urls[:3]:
            try:
                content = await self.web_scraper.scrape(url)
                snippets.append(f"{content.title}: {content.content[:400]}")
            except Exception as exc:
                snippets.append(f"{url}: scrape failed ({exc})")

        return "Scraped content: " + " | ".join(snippets)

    def _tool_write_note(self, args: dict[str, Any], agent_id: str) -> str:
        title = str(args.get("title") or "Note")
        summary = str(args.get("summary") or "")
        urls = args.get("urls") or []
        tags = args.get("tags") or []
        share = bool(args.get("share", True))

        note = self.agent_memory.add_note(title=title, summary=summary, urls=urls, tags=tags)
        if share:
            self.shared_memory.add_note(note)

        if self.stream:
            self.stream.emit_agent_note(
                agent_id,
                {"title": note.title, "summary": note.summary, "urls": note.urls, "shared": share},
            )

        return f"Note saved: {title}"

    def _tool_add_todo(self, args: dict[str, Any], agent_id: str) -> str:
        items = args.get("items") or []
        added = 0
        if isinstance(items, str):
            items = [items]
        for item in items:
            if isinstance(item, str):
                self.agent_memory.add_todo(item)
                added += 1
            elif isinstance(item, dict):
                self.agent_memory.add_todo(
                    title=item.get("title", "Task"),
                    note=item.get("note"),
                    url=item.get("url"),
                )
                added += 1

        if self.stream:
            self.stream.emit_agent_todo(agent_id, self._todo_payload())
        return f"Added {added} todo items."

    def _tool_complete_todo(self, args: dict[str, Any], agent_id: str) -> str:
        titles = args.get("titles") or []
        if isinstance(titles, str):
            titles = [titles]
        completed = 0
        for title in titles:
            if self.agent_memory.complete_todo(title):
                completed += 1

        if self.stream:
            self.stream.emit_agent_todo(agent_id, self._todo_payload())
        return f"Completed {completed} todo items."

    def _tool_read_shared_notes(self, args: dict[str, Any]) -> str:
        keyword = str(args.get("keyword") or "").lower()
        limit = int(args.get("limit") or 5)
        if not keyword:
            return self.shared_memory.render_notes(limit=limit)

        matches = []
        for note in self.shared_memory.notes:
            if keyword in note.title.lower() or keyword in note.summary.lower():
                matches.append(note)
        if not matches:
            return "No shared notes matched."
        return "\n".join(
            [f"- {note.title}: {note.summary} (urls: {', '.join(note.urls[:2])})" for note in matches[:limit]]
        )

    def _synthesize_finding(self, agent_id: str, topic: str) -> ResearchFinding:
        notes = self.agent_memory.notes[-5:]
        if notes:
            summary = " ".join([note.summary for note in notes])[:1200]
            key_findings = [note.summary for note in notes[:3]]
            confidence = "medium" if self.sources else "low"
        else:
            summary = f"No detailed notes produced for {topic}."
            key_findings = ["No structured notes available."]
            confidence = "low"

        return ResearchFinding(
            researcher_id=agent_id,
            topic=topic,
            summary=summary,
            key_findings=key_findings,
            sources=self.sources[: self.max_sources],
            confidence=confidence,
        )

    def _todo_payload(self) -> list[dict[str, Any]]:
        return [
            {"title": item.title, "status": item.status, "note": item.note, "url": item.url}
            for item in self.agent_memory.todos
        ]

    def _parse_action(self, content: str) -> tuple[str, dict[str, Any]]:
        try:
            payload = json.loads(_extract_json(content))
        except json.JSONDecodeError:
            logger.warning("Agent JSON parse failed", response=content)
            return "finish", {}

        action = str(payload.get("action", "finish"))
        args = payload.get("args") or {}
        return action, args


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
    for finding in findings[-5:]:
        lines.append(f"- {finding.topic}: {finding.summary[:200]}")
    return "\n".join(lines)

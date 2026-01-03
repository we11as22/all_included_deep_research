"""Agent memory models for deep research mode."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from datetime import datetime, timezone
from typing import Iterable
from uuid import uuid4


@dataclass
class AgentTodoItem:
    """Per-agent todo item."""

    reasoning: str
    title: str
    objective: str
    expected_output: str
    sources_needed: list[str] = field(default_factory=list)
    priority: str = "medium"
    status: str = "pending"
    note: str | None = None
    url: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    todo_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class AgentNote:
    """Research note with optional links."""

    title: str
    summary: str
    urls: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    note_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class AgentMemory:
    """Per-agent memory with todos and notes."""

    todos: list[AgentTodoItem] = field(default_factory=list)
    notes: list[AgentNote] = field(default_factory=list)
    visited_urls: set[str] = field(default_factory=set)

    def add_todo(
        self,
        title: str,
        objective: str,
        expected_output: str,
        sources_needed: list[str] | None = None,
        priority: str = "medium",
        reasoning: str = "",
        status: str = "pending",
        note: str | None = None,
        url: str | None = None,
    ) -> AgentTodoItem:
        normalized = title.strip().lower()
        for existing in self.todos:
            if existing.title.strip().lower() == normalized:
                return existing
        item = AgentTodoItem(
            reasoning=reasoning,
            title=title,
            objective=objective,
            expected_output=expected_output,
            sources_needed=list(sources_needed or []),
            priority=priority,
            status=status,
            note=note,
            url=url,
        )
        self.todos.append(item)
        return item

    def complete_todo(self, title: str) -> bool:
        for item in self.todos:
            if item.title == title and item.status != "done":
                item.status = "done"
                return True
        return False

    def add_note(self, title: str, summary: str, urls: Iterable[str] | None = None, tags: Iterable[str] | None = None) -> AgentNote:
        note = AgentNote(
            title=title,
            summary=summary,
            urls=list(urls or []),
            tags=list(tags or []),
        )
        self.notes.append(note)
        for url in note.urls:
            self.visited_urls.add(url)
        return note

    def pending_todos(self) -> list[AgentTodoItem]:
        return [item for item in self.todos if item.status != "done"]

    def render_todos(self, limit: int = 8) -> str:
        if not self.todos:
            return "None."
        lines = []
        for item in self.todos[:limit]:
            lines.append(
                json.dumps(
                    {
                        "reasoning": item.reasoning,
                        "title": item.title,
                        "objective": item.objective,
                        "expected_output": item.expected_output,
                        "sources_needed": item.sources_needed,
                        "priority": item.priority,
                        "status": item.status,
                        "note": item.note,
                        "url": item.url,
                    },
                    ensure_ascii=False,
                )
            )
        return "\n".join(f"- {line}" for line in lines)

    def render_notes(self, limit: int = 6) -> str:
        if not self.notes:
            return "None."
        lines = []
        for note in self.notes[-limit:]:
            urls = ", ".join(note.urls[:3]) if note.urls else "none"
            lines.append(f"- {note.title}: {note.summary} (urls: {urls})")
        return "\n".join(lines)

    def clear(self) -> None:
        self.todos.clear()
        self.notes.clear()
        self.visited_urls.clear()


@dataclass
class SharedResearchMemory:
    """Shared memory across agents."""

    notes: list[AgentNote] = field(default_factory=list)
    todo_directives: dict[str, list[dict]] = field(default_factory=dict)

    def add_note(self, note: AgentNote) -> None:
        self.notes.append(note)

    def add_todo_directives(self, agent_id: str, updates: list[dict]) -> None:
        if not agent_id or not updates:
            return
        existing = self.todo_directives.get(agent_id, [])
        existing.extend(updates)
        self.todo_directives[agent_id] = existing

    def pop_todo_directives(self, agent_id: str) -> list[dict]:
        if not agent_id:
            return []
        return self.todo_directives.pop(agent_id, [])

    def recent_notes(self, limit: int = 8) -> list[AgentNote]:
        return self.notes[-limit:]

    def render_notes(self, limit: int = 8) -> str:
        if not self.notes:
            return "None."
        lines = []
        for note in self.recent_notes(limit):
            urls = ", ".join(note.urls[:3]) if note.urls else "none"
            lines.append(f"- {note.title}: {note.summary} (urls: {urls})")
        return "\n".join(lines)

    def clear(self) -> None:
        self.notes.clear()
        self.todo_directives.clear()

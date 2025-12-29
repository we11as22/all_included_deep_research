"""Agentic research models for shared and per-agent memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable
from uuid import uuid4


@dataclass
class AgentTodoItem:
    """Per-agent todo item."""

    title: str
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

    def add_todo(self, title: str, note: str | None = None, url: str | None = None) -> AgentTodoItem:
        item = AgentTodoItem(title=title, note=note, url=url)
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
            suffix = f" (url: {item.url})" if item.url else ""
            note = f" - {item.note}" if item.note else ""
            lines.append(f"- [{item.status}] {item.title}{suffix}{note}")
        return "\n".join(lines)

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

    def add_note(self, note: AgentNote) -> None:
        self.notes.append(note)

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

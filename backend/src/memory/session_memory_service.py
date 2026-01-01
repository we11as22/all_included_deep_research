"""Session-scoped memory service with persistence.

Manages agent memory (todos, notes) for research sessions.
Hybrid: SQLite DB + Markdown files.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import structlog

logger = structlog.get_logger(__name__)


class SessionMemoryService:
    """Manages memory for single research session."""

    def __init__(
        self,
        session_id: str,
        base_memory_dir: str = "./memory_files",
        db_session: Any = None,
    ):
        """Initialize session memory service."""
        self.session_id = session_id
        self.session_dir = Path(base_memory_dir) / "sessions" / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.agents_dir = self.session_dir / "agents"
        self.agents_dir.mkdir(exist_ok=True)

        self.items_dir = self.session_dir / "items"
        self.items_dir.mkdir(exist_ok=True)

        self.db = db_session

        # Initialize main.md
        self.main_file = self.session_dir / "main.md"
        if not self.main_file.exists():
            self._init_main_file()

    def _init_main_file(self):
        """Initialize main.md with template."""
        template = f"""# Research Session {self.session_id}

## Overview
Research session started at {datetime.now().isoformat()}

## Project Status
Status: In Progress

## Agents
(Agents will appear here)

## Notes
(Shared notes will appear here)
"""
        self.main_file.write_text(template, encoding="utf-8")

    async def read_main(self) -> str:
        """Read main.md content."""
        return self.main_file.read_text(encoding="utf-8")

    async def save_agent_file(
        self,
        agent_id: str,
        todos: List[Dict],
        notes: List[str] | None = None,
        character: str | None = None,
    ):
        """Save agent's personal file."""
        agent_file = self.agents_dir / f"{agent_id}.md"

        content = f"""# Agent {agent_id}

## Character
{character or "Standard research agent"}

## Todo List

"""
        for todo in todos:
            status = todo.get("status", "pending")
            icon = "✅" if status == "done" else "⏸️" if status == "in_progress" else "⬜"
            content += f"{icon} **{todo.get('title', 'Untitled')}**\n"
            content += f"  - Objective: {todo.get('objective', '')}\n"
            content += f"  - Priority: {todo.get('priority', 'medium')}\n\n"

        if notes:
            content += "\n## Notes\n\n"
            for note in notes:
                content += f"- {note}\n"

        agent_file.write_text(content, encoding="utf-8")

        # Persist todos to DB
        if self.db:
            await self._persist_todos_to_db(agent_id, todos)

    async def save_note(
        self,
        agent_id: str,
        title: str,
        summary: str,
        urls: List[str],
        tags: List[str],
        share: bool = True,
    ) -> str:
        """Save research note as markdown file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"note_{timestamp}_{agent_id}.md"
        note_file = self.items_dir / filename

        content = f"""# {title}

**Agent**: {agent_id}
**Created**: {datetime.now().isoformat()}
**Tags**: {", ".join(tags)}

## Summary

{summary}

## Sources

"""
        for url in urls:
            content += f"- {url}\n"

        note_file.write_text(content, encoding="utf-8")

        logger.debug(f"Note saved: {filename}", agent_id=agent_id)

        return str(note_file.relative_to(self.session_dir))

    async def _persist_todos_to_db(self, agent_id: str, todos: List[Dict]):
        """Persist todos to database."""
        if not self.db:
            return

        from src.database.schema_sqlite import AgentMemoryModel

        # Delete old todos
        # Insert new todos
        # (Simplified - actual implementation would use SQLAlchemy)
        pass

    async def cleanup_session(self):
        """Archive or delete session files after completion."""
        # Move to archive or delete
        pass


def create_session_memory_service(
    session_id: str,
    base_memory_dir: str = "./memory_files",
) -> SessionMemoryService:
    """Factory function for creating session memory service."""
    return SessionMemoryService(session_id, base_memory_dir)

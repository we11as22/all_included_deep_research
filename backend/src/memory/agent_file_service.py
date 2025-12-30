"""Service for managing per-agent personal files."""

import re
from datetime import datetime, timezone
from typing import Any

import structlog

from src.memory.file_manager import FileManager
from src.workflow.agentic.models import AgentMemory, AgentTodoItem

logger = structlog.get_logger(__name__)


class AgentFileService:
    """Service for managing per-agent personal files with todo and notes."""

    def __init__(self, file_manager: FileManager):
        """
        Initialize agent file service.

        Args:
            file_manager: File manager instance
        """
        self.file_manager = file_manager
        self.agents_dir = "agents"

    async def read_agent_file(self, agent_id: str) -> dict[str, Any]:
        """
        Read agent's personal file.

        Args:
            agent_id: Agent identifier

        Returns:
            Dict with todos and notes
        """
        file_path = f"{self.agents_dir}/{agent_id}.md"
        try:
            content = await self.file_manager.read_file(file_path)
            return self._parse_agent_file(content)
        except FileNotFoundError:
            # Return empty structure
            return {
                "todos": [],
                "notes": [],
                "character": "",
                "preferences": "",
            }

    async def write_agent_file(
        self,
        agent_id: str,
        todos: list[AgentTodoItem] | None = None,
        notes: list[str] | None = None,
        character: str | None = None,
        preferences: str | None = None,
    ) -> None:
        """
        Write agent's personal file.

        Args:
            agent_id: Agent identifier
            todos: List of todo items
            notes: List of note strings
            character: Agent character description
            preferences: Agent preferences
        """
        file_path = f"{self.agents_dir}/{agent_id}.md"
        
        # Read existing file to preserve character/preferences if not updating
        existing = await self.read_agent_file(agent_id)
        
        todos = todos or existing.get("todos", [])
        notes = notes if notes is not None else existing.get("notes", [])
        character = character if character is not None else existing.get("character", "")
        preferences = preferences if preferences is not None else existing.get("preferences", "")

        content = self._format_agent_file(agent_id, todos, notes, character, preferences)
        await self.file_manager.write_file(file_path, content)
        logger.info("Agent file written", agent_id=agent_id, file_path=file_path)

    async def update_agent_todo(
        self,
        agent_id: str,
        todo_title: str,
        status: str | None = None,
        note: str | None = None,
    ) -> bool:
        """
        Update agent's todo item.

        Args:
            agent_id: Agent identifier
            todo_title: Todo item title
            status: New status (pending, in_progress, done)
            note: Additional note

        Returns:
            True if todo was found and updated
        """
        file_data = await self.read_agent_file(agent_id)
        todos = file_data.get("todos", [])
        
        updated = False
        for todo in todos:
            if todo.title == todo_title:
                if status:
                    todo.status = status
                if note is not None:
                    todo.note = note
                updated = True
                break
        
        if updated:
            await self.write_agent_file(agent_id, todos=todos)
        
        return updated

    def _parse_agent_file(self, content: str) -> dict[str, Any]:
        """Parse agent file content."""
        todos = []
        notes = []
        character = ""
        preferences = ""
        
        lines = content.split("\n")
        current_section = None
        current_note = []
        
        for line in lines:
            if line.startswith("# Agent:"):
                continue
            elif line.startswith("## Character"):
                current_section = "character"
            elif line.startswith("## Preferences"):
                current_section = "preferences"
            elif line.startswith("## Todo List"):
                current_section = "todos"
            elif line.startswith("## Notes"):
                current_section = "notes"
            elif line.startswith("- [") and current_section == "todos":
                # Parse todo: - [status] title (url: ...) - note
                match = re.match(r'- \[(\w+)\]\s+(.+?)(?:\s+\(url:\s+([^)]+)\))?(?:\s+-\s+(.+))?$', line)
                if match:
                    status, title, url, note = match.groups()
                    todos.append(AgentTodoItem(
                        title=title.strip(),
                        status=status,
                        url=url.strip() if url else None,
                        note=note.strip() if note else None,
                    ))
            elif line.startswith("- ") and current_section == "notes":
                note_text = line[2:].strip()
                if note_text:
                    notes.append(note_text)
            elif current_section == "character" and line.strip() and not line.startswith("#"):
                character += line + "\n"
            elif current_section == "preferences" and line.strip() and not line.startswith("#"):
                preferences += line + "\n"
        
        return {
            "todos": todos,
            "notes": notes,
            "character": character.strip(),
            "preferences": preferences.strip(),
        }

    def _format_agent_file(
        self,
        agent_id: str,
        todos: list[AgentTodoItem],
        notes: list[str],
        character: str,
        preferences: str,
    ) -> str:
        """Format agent file content."""
        lines = [
            f"# Agent: {agent_id}",
            "",
            f"**Last Updated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "## Character",
            "",
            character if character else "<!-- Agent character and personality -->",
            "",
            "## Preferences",
            "",
            preferences if preferences else "<!-- Agent preferences and working style -->",
            "",
            "## Todo List",
            "",
        ]
        
        if todos:
            for todo in todos:
                url_part = f" (url: {todo.url})" if todo.url else ""
                note_part = f" - {todo.note}" if todo.note else ""
                lines.append(f"- [{todo.status}] {todo.title}{url_part}{note_part}")
        else:
            lines.append("<!-- No todos -->")
        
        lines.extend([
            "",
            "## Notes",
            "",
        ])
        
        if notes:
            for note in notes:
                lines.append(f"- {note}")
        else:
            lines.append("<!-- No notes -->")
        
        return "\n".join(lines)


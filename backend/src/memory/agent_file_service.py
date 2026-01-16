"""Service for managing per-agent personal files."""

import json
import re
from datetime import datetime, timezone
from typing import Any

import structlog

from src.memory.file_manager import FileManager
from src.models.agent_models import AgentMemory, AgentTodoItem

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
        
        # CRITICAL: Protect done tasks - never remove or modify them
        # When writing todos, preserve all done tasks from existing file
        existing_todos = existing.get("todos", [])
        existing_done_todos = {t.title: t for t in existing_todos if t.status == "done"}  # Use dict for deduplication
        
        # Merge: keep all done tasks + new/updated todos
        if todos:
            # Separate done and non-done tasks from new todos
            new_done_todos = {t.title: t for t in todos if t.status == "done"}
            new_pending_todos = [t for t in todos if t.status != "done"]
            
            # CRITICAL: Merge done tasks - prefer existing (they are the source of truth)
            # If a done task appears in both, keep the existing one (it's already persisted)
            merged_done_todos = {**new_done_todos, **existing_done_todos}  # existing_done_todos overwrite new
            
            # Combine: all done tasks (merged) + new pending/in_progress todos
            todos = list(merged_done_todos.values()) + new_pending_todos
        else:
            # If no new todos provided, keep existing (including done)
            todos = existing_todos
        
        notes = notes if notes is not None else existing.get("notes", [])
        character = character if character is not None else existing.get("character", "")
        preferences = preferences if preferences is not None else existing.get("preferences", "")

        content = self._format_agent_file(agent_id, todos, notes, character, preferences)
        await self.file_manager.write_file(file_path, content)
        logger.info("Agent file written", agent_id=agent_id, file_path=file_path, 
                   total_todos=len(todos), done_todos=len([t for t in todos if t.status == "done"]))

    async def delete_agent_file(self, agent_id: str) -> bool:
        """
        Delete agent's personal file.

        Returns:
            True if file was deleted, False if it didn't exist.
        """
        file_path = f"{self.agents_dir}/{agent_id}.md"
        try:
            await self.file_manager.delete_file(file_path)
            return True
        except FileNotFoundError:
            return False
        except Exception as exc:
            logger.warning("Failed to delete agent file", agent_id=agent_id, error=str(exc))
            return False

    async def update_agent_todo(
        self,
        agent_id: str,
        todo_title: str,
        status: str | None = None,
        note: str | None = None,
        reasoning: str | None = None,
        objective: str | None = None,
        expected_output: str | None = None,
        sources_needed: list[str] | None = None,
        priority: str | None = None,
        url: str | None = None,
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
                # CRITICAL: Protect done tasks - they are immutable once completed
                # No one can modify or delete done tasks - they are permanent record
                if todo.status == "done":
                    logger.warning(f"Attempted to modify done task '{todo_title}' for agent {agent_id}. Done tasks are immutable and cannot be changed.",
                                 agent_id=agent_id, todo_title=todo_title, note="Done tasks are permanent records")
                    return False  # Don't update done tasks at all
                
                # CRITICAL: Protect in_progress tasks from status changes
                # Only allow status changes to in_progress tasks if changing to done
                # This prevents race conditions where supervisor/other agents change status while agent is working
                if status:
                    if todo.status == "in_progress" and status != "done":
                        logger.warning(f"Attempted to change status of in_progress task '{todo_title}' for agent {agent_id} from in_progress to {status}. Ignoring status change to prevent race condition.",
                                     agent_id=agent_id, todo_title=todo_title, current_status=todo.status, attempted_status=status)
                        # Don't update status, but allow other fields to be updated
                    else:
                        todo.status = status
                
                # Allow updating other fields even for in_progress tasks
                # (agent uses cached current_task, so changes to objective/guidance won't affect current work)
                if note is not None:
                    todo.note = note
                if reasoning is not None:
                    todo.reasoning = reasoning
                if objective is not None:
                    todo.objective = objective
                if expected_output is not None:
                    todo.expected_output = expected_output
                if sources_needed is not None:
                    todo.sources_needed = sources_needed
                if priority is not None:
                    todo.priority = priority
                if url is not None:
                    todo.url = url
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
            elif line.startswith("- ") and current_section == "todos":
                entry = line[2:].strip()
                if entry.startswith("{"):
                    try:
                        payload = json.loads(entry)
                        todos.append(
                            AgentTodoItem(
                                reasoning=str(payload.get("reasoning") or ""),
                                title=str(payload.get("title") or "").strip() or "Task",
                                objective=str(payload.get("objective") or "").strip() or "Investigate the topic",
                                expected_output=str(payload.get("expected_output") or "").strip() or "Summary with sources",
                                sources_needed=list(payload.get("sources_needed") or []),
                                priority=str(payload.get("priority") or "medium"),
                                status=str(payload.get("status") or "pending"),
                                note=payload.get("note"),
                                url=payload.get("url"),
                            )
                        )
                        continue
                    except Exception:
                        pass
                # Legacy format: - [status] title (url: ...) - note
                match = re.match(r'- \[(\w+)\]\s+(.+?)(?:\s+\(url:\s+([^)]+)\))?(?:\s+-\s+(.+))?$', line)
                if match:
                    status, title, url, note = match.groups()
                    todos.append(
                        AgentTodoItem(
                            reasoning="Legacy todo loaded",
                            title=title.strip(),
                            objective="Investigate the topic",
                            expected_output="Summary with sources",
                            sources_needed=[],
                            priority="medium",
                            status=status,
                            url=url.strip() if url else None,
                            note=note.strip() if note else None,
                        )
                    )
            elif line.startswith("- ") and current_section == "notes":
                note_text = line[2:].strip()
                if note_text:
                    notes.append(note_text)
            elif current_section == "character" and line.strip() and not line.startswith("#"):
                character += line + "\n"
            elif current_section == "preferences" and line.strip() and not line.startswith("#"):
                preferences += line + "\n"
        
        # Limit notes to prevent context bloat - only return last 20 notes
        # Even if file contains more, we only use recent important ones
        limited_notes = notes[-20:] if len(notes) > 20 else notes
        
        return {
            "todos": todos,
            "notes": limited_notes,  # Limited to prevent context bloat
            "all_notes_count": len(notes),  # Total count for reference
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
                payload = {
                    "reasoning": todo.reasoning,
                    "title": todo.title,
                    "objective": todo.objective,
                    "expected_output": todo.expected_output,
                    "sources_needed": todo.sources_needed,
                    "priority": todo.priority,
                    "status": todo.status,
                    "note": todo.note,
                    "url": todo.url,
                }
                lines.append(f"- {json.dumps(payload, ensure_ascii=False)}")
        else:
            lines.append("<!-- No todos -->")
        
        lines.extend([
            "",
            "## Notes",
            "",
            "<!-- Only important notes are stored here to prevent context bloat -->",
            "",
        ])
        
        if notes:
            # Only show last 20 notes in file (even if more are stored)
            # This prevents file from becoming too large
            display_notes = notes[-20:] if len(notes) > 20 else notes
            for note in display_notes:
                lines.append(f"- {note}")
            if len(notes) > 20:
                lines.append(f"\n<!-- {len(notes) - 20} older notes not shown -->")
        else:
            lines.append("<!-- No notes -->")
        
        return "\n".join(lines)

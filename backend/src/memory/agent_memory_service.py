"""Service for agents to save and read notes from memory files."""

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog

from src.memory.file_manager import FileManager
from src.workflow.agentic.models import AgentNote

logger = structlog.get_logger(__name__)


class AgentMemoryService:
    """Service for agents to interact with persistent memory files."""

    def __init__(self, file_manager: FileManager):
        """
        Initialize agent memory service.

        Args:
            file_manager: File manager instance
        """
        self.file_manager = file_manager
        self.main_file = "main.md"
        self.items_dir = "items"

    async def save_agent_note(
        self,
        note: AgentNote,
        agent_id: str,
    ) -> str:
        """
        Save agent note to items/ directory and update main.md.

        Args:
            note: Agent note to save
            agent_id: Optional agent ID for filename

        Returns:
            File path of saved note
        """
        # Generate filename from title
        safe_title = self._sanitize_filename(note.title)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        agent_suffix = f"_{agent_id[:8]}" if agent_id else ""
        filename = f"{timestamp}_{safe_title}{agent_suffix}.md"
        file_path = f"{self.items_dir}/{filename}"

        # Format note content with agent metadata
        content = self._format_note_content(note, agent_id)

        # Save note file
        await self.file_manager.write_file(file_path, content)

        # Update main.md with reference
        await self._update_main_file(file_path, note.title, note.summary, note.tags)

        logger.info("Agent note saved", file_path=file_path, agent_id=agent_id)
        return file_path

    async def read_main_file(self) -> str:
        """Read main.md content."""
        try:
            return await self.file_manager.read_file(self.main_file)
        except FileNotFoundError:
            # Create initial main.md if it doesn't exist
            initial_content = self._get_initial_main_content()
            await self.file_manager.write_file(self.main_file, initial_content)
            return initial_content

    async def list_items(self) -> list[dict[str, Any]]:
        """
        List all items in items/ directory.

        Returns:
            List of item metadata
        """
        try:
            files = await self.file_manager.list_files(f"{self.items_dir}/*.md")
            items = []
            for file_path in files:
                try:
                    content = await self.file_manager.read_file(file_path)
                    # Extract metadata from file
                    title = self._extract_title_from_content(content)
                    summary = self._extract_summary_from_content(content)
                    items.append({
                        "file_path": file_path,
                        "title": title,
                        "summary": summary,
                    })
                except Exception as e:
                    logger.warning("Failed to read item", file_path=file_path, error=str(e))
            return items
        except Exception as e:
            logger.warning("Failed to list items", error=str(e))
            return []

    def _sanitize_filename(self, title: str) -> str:
        """Sanitize title for use in filename."""
        # Remove special characters, keep only alphanumeric, spaces, hyphens, underscores
        safe = re.sub(r'[^\w\s-]', '', title)
        # Replace spaces with underscores
        safe = re.sub(r'\s+', '_', safe)
        # Limit length
        safe = safe[:50]
        return safe

    def _format_note_content(self, note: AgentNote, agent_id: str) -> str:
        """Format note as markdown content with agent metadata."""
        lines = [
            f"# {note.title}",
            "",
            f"**Created:** {note.created_at}",
            f"**Created by:** {agent_id}",
        ]
        if note.tags:
            lines.append(f"**Tags:** {', '.join(note.tags)}")
        if note.urls:
            lines.append("")
            lines.append("## Sources")
            for url in note.urls:
                lines.append(f"- {url}")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append(note.summary)
        return "\n".join(lines)

    def _extract_title_from_content(self, content: str) -> str:
        """Extract title from markdown content."""
        lines = content.split("\n")
        for line in lines:
            if line.startswith("# "):
                return line[2:].strip()
        return "Untitled"

    def _extract_summary_from_content(self, content: str) -> str:
        """Extract summary from markdown content."""
        # Look for ## Summary section
        if "## Summary" in content:
            parts = content.split("## Summary")
            if len(parts) > 1:
                summary = parts[1].strip()
                # Take first paragraph
                summary = summary.split("\n\n")[0] if "\n\n" in summary else summary.split("\n")[0]
                return summary[:200]  # Limit length
        return ""

    async def _update_main_file(self, file_path: str, title: str, summary: str, tags: list[str]) -> None:
        """Update main.md with reference to new item."""
        try:
            content = await self.read_main_file()
        except Exception:
            content = self._get_initial_main_content()

        # Find or create "## Items" section
        items_section_start = content.find("## Items")
        
        if items_section_start == -1:
            # Items section doesn't exist, add it after Overview
            if "## Overview" in content:
                overview_end = content.find("\n## ", content.find("## Overview") + len("## Overview"))
                if overview_end != -1:
                    content = content[:overview_end] + "\n\n## Items\n\n" + content[overview_end:]
                else:
                    content += "\n\n## Items\n\n"
            else:
                content += "\n\n## Items\n\n"
            items_section_start = content.find("## Items")
        
        # Check if item already exists (by title)
        items_section = content[items_section_start:]
        if f"### {title}" in items_section:
            # Item already exists, skip or update
            logger.info("Item already exists in main.md", title=title)
            return
        
        # Find where to insert (after ## Items header, before next ## section)
        items_content_start = items_section.find("\n", items_section.find("## Items")) + 1
        next_section = items_section.find("\n## ", items_content_start)
        
        if next_section != -1:
            # Insert before next section
            insert_pos = items_section_start + next_section
            new_item = f"### {title}\n\n**File:** [{file_path}]({file_path})\n\n{summary[:150]}...\n\n"
            content = content[:insert_pos] + new_item + content[insert_pos:]
        else:
            # Append to end of Items section
            insert_pos = items_section_start + items_content_start
            new_item = f"### {title}\n\n**File:** [{file_path}]({file_path})\n\n{summary[:150]}...\n\n"
            content = content[:insert_pos] + new_item + content[insert_pos:]

        # Update last_updated
        content = re.sub(
            r"Last Updated: \d{4}-\d{2}-\d{2}",
            f"Last Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
            content,
        )

        await self.file_manager.write_file(self.main_file, content)

    def _get_initial_main_content(self) -> str:
        """Get initial main.md content."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return f"""# Agent Memory - Main Index

Last Updated: {today}

## Overview

This is the main index file for agent memory. All agent notes and findings are stored in the `items/` directory and referenced here.

## Items

<!-- Agent notes and findings will be listed here -->

---

## Quick Reference

<!-- Frequently needed information -->

---

## Notes

<!-- Additional notes and context -->
"""


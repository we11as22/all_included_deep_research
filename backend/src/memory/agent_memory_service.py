"""Service for agents to save and read notes from memory files."""

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog

from src.memory.file_manager import FileManager
from src.models.agent_models import AgentNote
from src.utils.text import summarize_text

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
        agent_file_service: Any = None,
    ) -> str:
        """
        Save agent note to items/ directory.
        
        Also adds note to agent's personal file (agents/{agent_id}.md) Notes section.

        Args:
            note: Agent note to save
            agent_id: Agent ID for filename and personal file update
            agent_file_service: Optional agent file service to update agent's personal file

        Returns:
            File path of saved note
        """
        # Generate filename from title
        safe_title = self._sanitize_filename(note.title)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        agent_suffix = f"_{agent_id[:8]}" if agent_id else ""
        filename = f"{timestamp}_{safe_title}{agent_suffix}.md"
        file_path = f"{self.items_dir}/{filename}"

        # Format note content with agent metadata and useful context
        content = self._format_note_content(note, agent_id)

        # Save note file to items/
        await self.file_manager.write_file(file_path, content)

        # Add note to agent's personal file Notes section
        if agent_file_service:
            try:
                agent_file = await agent_file_service.read_agent_file(agent_id)
                existing_notes = agent_file.get("notes", [])
                
                # Filter notes by importance - only save informative notes
                # Check if note contains actual information, not just metadata
                note_summary_lower = (note.summary or "").lower()
                note_title_lower = (note.title or "").lower()
                
                # Skip notes that are just metadata (not informative)
                is_metadata_only = any([
                    "found" in note_summary_lower and "sources" in note_summary_lower and "query" in note_summary_lower,
                    "search:" in note_title_lower and len(note.summary or "") < 100,
                    note_summary_lower.count("found") > 0 and "relevant sources" in note_summary_lower,
                    "key sources:" in note_summary_lower and len(note.summary or "") < 150,
                ])
                
                # Check if note has substantial content (not just links or titles)
                has_substantial_content = (
                    len(note.summary or "") > 200 or  # Has detailed summary
                    any(keyword in note_summary_lower for keyword in [
                        "discovery", "finding", "insight", "conclusion", "fact", "data",
                        "information", "evidence", "analysis", "pattern", "trend"
                    ])
                )
                
                # Format note for agent's personal file
                note_text = f"{note.title}: {note.summary}"
                if note.urls:
                    note_text += f" | Sources: {len(note.urls)}"
                
                # Only save if note is informative and not duplicate
                if note_text and not is_metadata_only and has_substantial_content and note_text not in existing_notes:
                    existing_notes.append(note_text)
                    # Keep only last 20 important notes to prevent context bloat
                    existing_notes = existing_notes[-20:]
                    await agent_file_service.write_agent_file(
                        agent_id=agent_id,
                        notes=existing_notes
                    )
                    logger.info(f"Added important note to agent {agent_id} personal file", note_title=note.title, total_notes=len(existing_notes))
                else:
                    if is_metadata_only:
                        logger.debug(f"Metadata-only note skipped for agent {agent_id}", note_title=note.title)
                    elif not has_substantial_content:
                        logger.debug(f"Low-content note skipped for agent {agent_id}", note_title=note.title, summary_length=len(note.summary or ""))
                    elif note_text in existing_notes:
                        logger.debug(f"Duplicate note skipped for agent {agent_id}", note_title=note.title)
                    else:
                        logger.debug(f"Empty note skipped for agent {agent_id}", note_title=note.title)
            except Exception as e:
                logger.warning(f"Failed to add note to agent {agent_id} personal file", error=str(e))

        # Don't update main.md - items stay in items/ directory
        # Main.md should only contain key insights from supervisor
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
        """Format note as markdown content with agent metadata and useful context."""
        lines = [
            f"# {note.title}",
            "",
            f"**Created:** {note.created_at}",
            f"**Created by:** {agent_id}",
        ]
        if note.tags:
            lines.append(f"**Tags:** {', '.join(note.tags)}")
        
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append(note.summary)
        
        # Add detailed findings if available in summary
        # The summary should contain key findings, not just "Found X sources"
        if "Found" in note.summary and "sources" in note.summary and len(note.summary) < 200:
            # This is a basic note, add context about what was found
            lines.append("")
            lines.append("## Key Findings")
            lines.append("")
            lines.append("<!-- Add specific findings, facts, and insights discovered from these sources -->")
            lines.append("<!-- Include: important facts, data points, expert opinions, comparisons, historical context -->")
        
        if note.urls:
            lines.append("")
            lines.append("## Sources")
            lines.append("")
            lines.append("Use these sources to gather detailed information:")
            for i, url in enumerate(note.urls, 1):
                lines.append(f"{i}. [{url}]({url})")
            lines.append("")
            lines.append("**Action Items:**")
            lines.append("- Review each source for relevant information")
            lines.append("- Extract key facts, statistics, and expert opinions")
            lines.append("- Note any contradictions or gaps in information")
            lines.append("- Identify follow-up research directions")
        
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
                return summarize_text(summary, 1200)
        return ""

    def _extract_agent_from_content(self, content: str) -> str:
        """Extract agent_id from markdown content."""
        for line in content.split("\n"):
            if line.startswith("**Created by:**"):
                return line.replace("**Created by:**", "").strip()
        return ""

    async def _update_main_file(self, file_path: str, title: str, summary: str, tags: list[str]) -> None:
        """
        Update main.md with reference to new item.
        
        CRITICAL: main.md should only contain KEY INSIGHTS, not all items.
        Items are stored in items/ directory and can be referenced when needed.
        Only add to main.md if it contains significant findings or key insights.
        """
        try:
            content = await self.read_main_file()
        except Exception:
            content = self._get_initial_main_content()

        # DON'T automatically add all items to main.md
        # Items are stored in items/ directory for reference
        # Main.md should only contain key insights and progress updates
        # This prevents main.md from becoming bloated with duplicate links
        
        logger.debug("Item saved to items/ directory", file_path=file_path, title=title)
        # Items are available in items/ directory, main.md stays focused on key insights

    def _get_initial_main_content(self) -> str:
        """Get initial main.md content for session."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return f"""# Research Session - Main Index

Last Updated: {today}

## Overview

This is the main index file for this research session.

## File Index

### Agents
<!-- Agent files will be listed here -->

### Items
<!-- Agent notes and findings will be listed here -->

### Reports
<!-- Research reports will be listed here -->

---

## Quick Reference

<!-- Frequently needed information -->

---

## Notes

<!-- Additional notes and context -->
"""

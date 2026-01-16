"""Draft report service - unified logic for managing draft_report.md.

This replaces the duplicated draft_report logic that was previously in 3 places:
- nodes.py:1181-1280 (99 lines of auto-update)
- nodes.py:1499-1615 (116 lines of fallback)
- supervisor_agent.py (write_draft_report_handler)
"""

import structlog
from datetime import datetime
from typing import Any, Dict, List

from sqlalchemy import select, update

from src.database.schema import ResearchSessionModel

logger = structlog.get_logger(__name__)


class DraftReportService:
    """Unified service for managing draft_report with DB persistence.

    Key features:
    - Saves to research_sessions.draft_report (in DB, not file)
    - Single source of truth for draft_report logic
    - Used by both execute_agents and supervisor
    - Automatic size limiting (trim old sections if too large)
    """

    def __init__(self, session_id: str, session_factory):
        """Initialize draft report service.

        Args:
            session_id: Research session ID
            session_factory: AsyncSession factory for DB access
        """
        self.session_id = session_id
        self.session_factory = session_factory
        self.max_size = 100000  # Max 100KB

    async def initialize(self, query: str) -> None:
        """Initialize empty draft report.

        Args:
            query: Original research query
        """
        content = f"""# Research Report Draft

**Query:** {query}
**Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This is the working draft of the research report. Findings are automatically updated as agents complete their tasks.

---
"""
        await self._save(content)
        logger.info("Draft report initialized", session_id=self.session_id)

    async def append_findings(self, findings: List[Dict[str, Any]]) -> None:
        """Append new findings to draft report.

        Args:
            findings: List of finding dictionaries from agents
        """
        if not findings:
            return

        current = await self.get_content()

        # Build findings section
        sections = []
        for finding in findings:
            section = self._format_finding(finding)
            sections.append(section)

        findings_text = "\n\n".join(sections)

        # Append with timestamp
        update = f"\n\n---\n\n## New Findings - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{findings_text}\n"
        updated = current + update

        # Trim if too large
        if len(updated) > self.max_size:
            updated = self._trim_old_content(updated)

        await self._save(updated)
        logger.info(
            "Appended findings to draft",
            session_id=self.session_id,
            count=len(findings),
        )

    async def add_section(self, title: str, content: str) -> None:
        """Add custom section to draft report.

        Used by supervisor to add synthesized sections.

        Args:
            title: Section title
            content: Section content
        """
        current = await self.get_content()

        section = f"\n\n---\n\n## {title} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{content}\n"
        updated = current + section

        # Trim if too large
        if len(updated) > self.max_size:
            updated = self._trim_old_content(updated)

        await self._save(updated)
        logger.info(
            "Added section to draft", session_id=self.session_id, title=title
        )

    async def get_content(self) -> str:
        """Get current draft content.

        Returns:
            Draft report content
        """
        async with self.session_factory() as session:
            result = await session.execute(
                select(ResearchSessionModel.draft_report).where(
                    ResearchSessionModel.id == self.session_id
                )
            )
            content = result.scalar_one_or_none()
            return content or ""

    def _format_finding(self, finding: Dict[str, Any]) -> str:
        """Format single finding as markdown.

        Args:
            finding: Finding dictionary

        Returns:
            Formatted markdown string
        """
        topic = finding.get("topic", "Unknown Topic")
        agent_id = finding.get("agent_id", "unknown")
        confidence = finding.get("confidence", "unknown")
        summary = finding.get("summary", "No summary")
        key_findings = finding.get("key_findings", [])
        sources = finding.get("sources", [])

        # Build key findings list
        if key_findings:
            findings_list = "\n".join([f"- {kf}" for kf in key_findings])
        else:
            findings_list = "- No key findings"

        return f"""### {topic}

**Agent:** {agent_id}
**Confidence:** {confidence}

{summary}

**Key Findings:**
{findings_list}

**Sources:** {len(sources)} sources cited
"""

    def _trim_old_content(self, content: str) -> str:
        """Trim old sections if content too large.

        Args:
            content: Full content

        Returns:
            Trimmed content
        """
        sections = content.split("\n\n---\n\n")

        # Keep header + last 5 sections
        if len(sections) > 6:
            header = sections[0]
            recent = sections[-5:]
            trimmed_count = len(sections) - 6
            summary = f"\n\n[... {trimmed_count} older sections trimmed for size ...]\n\n"
            return header + summary + "\n\n---\n\n".join(recent)

        return content

    async def _save(self, content: str) -> None:
        """Save draft_report to research_sessions table.

        Args:
            content: Draft content
        """
        async with self.session_factory() as session:
            await session.execute(
                update(ResearchSessionModel)
                .where(ResearchSessionModel.id == self.session_id)
                .values(draft_report=content, updated_at=datetime.now())
            )
            await session.commit()

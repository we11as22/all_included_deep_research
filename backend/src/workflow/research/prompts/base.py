"""Base class for prompt builders."""

from abc import ABC
from typing import List


class PromptBuilder(ABC):
    """Base class for all prompt builders.

    Provides common utilities for formatting prompts.
    """

    def _format_section(self, title: str, content: str) -> str:
        """Format a section with title.

        Args:
            title: Section title
            content: Section content

        Returns:
            Formatted section
        """
        return f"**{title}:**\n{content}\n\n"

    def _format_sections(self, sections: List[str]) -> str:
        """Join multiple sections.

        Args:
            sections: List of formatted sections

        Returns:
            Combined sections
        """
        return "\n".join(s for s in sections if s)

    def _truncate(self, text: str, max_length: int = 2000) -> str:
        """Truncate text to max length.

        Args:
            text: Text to truncate
            max_length: Maximum length

        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."

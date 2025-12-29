"""Mock chat model for offline testing."""

from __future__ import annotations

import re
from typing import Any, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class MockChatModel(BaseChatModel):
    """Minimal chat model that returns deterministic responses."""

    model_name: str = "mock"

    @property
    def _llm_type(self) -> str:
        return "mock-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        content = self._compose_response(messages)
        message = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _compose_response(self, messages: List[BaseMessage]) -> str:
        if not messages:
            return "Mock response."

        last = messages[-1].content if hasattr(messages[-1], "content") else ""
        last_text = str(last)
        lower = last_text.lower()

        if "rewrite the user query" in lower:
            return self._extract_last_query(last_text)

        if "generate" in lower and "search queries" in lower:
            topic = self._extract_topic(last_text)
            return "\n".join(
                [
                    f"{topic} overview",
                    f"{topic} latest updates",
                    f"{topic} key stakeholders",
                ]
            )

        if "research plan" in lower or "research topics" in lower:
            topic = self._extract_topic(last_text)
            return (
                "## Overview\n"
                f"Quick plan for {topic} using memory and web sources.\n\n"
                "## Research Topics\n"
                f"1. {topic} background\n"
                f"2. {topic} current landscape\n"
                f"3. {topic} risks and opportunities\n\n"
                "## Rationale\n"
                "Cover foundations, current state, and implications."
            )

        if "provide a comprehensive analysis" in lower or "key findings" in lower:
            topic = self._extract_topic(last_text)
            return (
                "## Summary\n"
                f"This is a mock summary about {topic}. It outlines the main points "
                "found in the collected sources and memory context.\n\n"
                "## Key Findings\n"
                "- The topic has recent activity.\n"
                "- Multiple sources agree on core facts.\n"
                "- There are open questions worth tracking.\n\n"
                "## Confidence\n"
                "medium - mock confidence."
            )

        if "final report" in lower or "executive summary" in lower:
            topic = self._extract_topic(last_text)
            return (
                f"# Research Report: {topic}\n\n"
                "## Executive Summary\n"
                "This mock report summarizes gathered sources and memory context.\n\n"
                "## Detailed Findings\n"
                "- Point one supported by sources.\n"
                "- Point two supported by sources.\n\n"
                "## Key Takeaways\n"
                "- Takeaway one.\n"
                "- Takeaway two.\n\n"
                "## Sources\n"
                "- Source list generated from search results."
            )

        if "summarize the following source" in lower:
            return self._summarize_source(last_text)

        return "Mock response based on provided context."

    def _extract_topic(self, text: str) -> str:
        match = re.search(r"research topic:\s*(.+)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.search(r"research query:\s*(.+)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "the topic"

    def _extract_last_query(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else "query"

    def _summarize_source(self, text: str) -> str:
        lines = [line for line in text.splitlines() if line.strip()]
        snippet = " ".join(lines[-6:])[:280]
        return f"Summary: {snippet}"

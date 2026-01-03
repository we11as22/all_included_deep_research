"""Utilities for summarizing and trimming text safely."""

from __future__ import annotations

import re
from typing import Any

import structlog
from langchain_core.messages import SystemMessage, HumanMessage

logger = structlog.get_logger(__name__)

_sentence_splitter = re.compile(r"(?<=[.!?])\s+")


def summarize_text(text: str, max_chars: int) -> str:
    """
    Summarize text without hard truncation.

    Uses sentence boundaries when possible and appends ellipsis if shortened.
    """
    if not text:
        return ""
    if len(text) <= max_chars:
        return text

    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned

    sentences = [s.strip() for s in _sentence_splitter.split(cleaned) if s.strip()]
    if not sentences:
        return cleaned[: max_chars - 3].rstrip() + "..."

    target = max_chars - 3
    head_budget = max(60, int(target * 0.65))
    summary_parts: list[str] = []
    total = 0

    for sentence in sentences:
        if total + len(sentence) + 1 <= head_budget:
            summary_parts.append(sentence)
            total += len(sentence) + 1
        else:
            break

    if summary_parts:
        last_sentence = sentences[-1]
        if last_sentence not in summary_parts and total + len(last_sentence) + 1 <= target:
            summary_parts.append(last_sentence)
    else:
        summary_parts.append(sentences[0][:target].rstrip())

    summary = " ".join(summary_parts).strip()
    if len(summary) > target:
        summary = summary[:target].rstrip()

    return summary + "..."


def ellipsize(text: str, max_chars: int) -> str:
    """Shorten text with ellipsis, preferring whitespace boundaries."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    trimmed = text[: max_chars - 3].rstrip()
    return trimmed + "..."


async def summarize_text_llm(text: str, max_tokens: int, llm: Any) -> str:
    """
    Summarize text using LLM with structured output.

    Args:
        text: Text to summarize
        max_tokens: Target summary length in tokens (approximate)
        llm: LLM instance

    Returns:
        Summarized text
    """
    if not text or not llm:
        return ""

    # Trim input if too long (to fit in context)
    trimmed = summarize_text(text, 8000)

    # If already short enough, return as-is
    if len(trimmed) <= max_tokens * 4:  # Rough chars-to-tokens ratio
        return trimmed

    try:
        # Import here to avoid circular dependency
        from src.models.schemas import SummarizedContent

        prompt = (
            f"Summarize the following content concisely (target ~{max_tokens} tokens). "
            "Focus on key facts, data, and insights. Preserve important details and context."
        )

        structured_llm = llm.with_structured_output(SummarizedContent, method="function_calling")

        response = await structured_llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=trimmed)
        ])

        if hasattr(response, 'summary'):
            logger.debug("LLM summarization successful", original_length=len(text), summary_length=len(response.summary))
            return response.summary

        # Fallback if response is not structured
        logger.warning("LLM summarization returned non-structured response")
        return summarize_text(trimmed, max_tokens * 4)

    except Exception as e:
        logger.warning("LLM summarization failed, using fallback", error=str(e))
        # Fallback to simple truncation
        return summarize_text(trimmed, max_tokens * 4)

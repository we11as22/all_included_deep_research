"""Utilities for summarizing and trimming text safely."""

from __future__ import annotations

import re


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

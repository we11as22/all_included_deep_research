"""Helpers for formatting chat history."""

from __future__ import annotations

from typing import Iterable


def format_chat_history(messages: Iterable[dict] | None, limit: int) -> str:
    """Render the last N chat messages for prompts."""
    if not messages or limit <= 0:
        return "Chat history: None."

    cleaned = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").strip()
        content = str(msg.get("content") or "").strip()
        if not content or role.lower() == "system":
            continue
        cleaned.append({"role": role, "content": content})

    if not cleaned:
        return "Chat history: None."

    trimmed = cleaned[-limit:]
    lines = ["Chat history:"]
    for item in trimmed:
        lines.append(f"- {item['role']}: {item['content']}")
    return "\n".join(lines)

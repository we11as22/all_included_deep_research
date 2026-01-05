"""Chat completion models (OpenAI-compatible)."""

from enum import Enum

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Message roles."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMode(str, Enum):
    """Chat modes for different research workflows."""

    CHAT = "chat"
    SEARCH = "search"
    SPEED = "speed"
    DEEP_SEARCH = "deep_search"
    BALANCED = "balanced"
    DEEP_RESEARCH = "deep_research"
    QUALITY = "quality"


class ChatMessage(BaseModel):
    """Chat message."""

    role: MessageRole
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field(
        default="search",
        description="Chat mode: search (web), deep_search, deep_research (legacy: speed, balanced, quality)",
    )
    messages: list[ChatMessage] = Field(..., description="Conversation messages")
    stream: bool = Field(default=True, description="Whether to stream the response")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    user_id: str | None = Field(default=None, description="Optional user identifier")
    chat_id: str | None = Field(default=None, description="Optional chat ID to load messages from database")

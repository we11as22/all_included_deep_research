"""LLM provider abstraction for multi-provider support.

Unified interface for OpenAI, Anthropic, Ollama, OpenRouter, etc.
"""

import structlog
from typing import Any, Type

from pydantic import BaseModel

logger = structlog.get_logger(__name__)


class UnifiedLLM:
    """Unified interface for multiple LLM providers."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        """Initialize unified LLM."""
        self.provider = provider
        self.model = model
        self._client = self._create_client(
            provider, model, api_key, base_url, temperature, max_tokens
        )

    def _create_client(
        self, provider: str, model: str, api_key: str | None, base_url: str | None,
        temperature: float, max_tokens: int
    ):
        """Create provider-specific client."""

        if provider in ["openai", "openrouter", "302ai"]:
            from langchain_openai import ChatOpenAI

            kwargs = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            if api_key:
                kwargs["api_key"] = api_key

            if base_url:
                kwargs["base_url"] = base_url

                # Provider-specific headers
                if "openrouter" in (base_url or ""):
                    kwargs["default_headers"] = {
                        "HTTP-Referer": "https://github.com/all-included-deep-research",
                        "X-Title": "All-Included Deep Research"
                    }

            return ChatOpenAI(**kwargs)

        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens
            )

        elif provider == "ollama":
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(
                model=model,
                base_url=base_url or "http://localhost:11434",
                temperature=temperature,
            )

        elif provider == "mock":
            from src.llm.mock import MockChatModel
            return MockChatModel()

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def ainvoke(self, messages: list) -> Any:
        """Invoke LLM asynchronously."""
        return await self._client.ainvoke(messages)

    def with_structured_output(
        self, schema: Type[BaseModel], method: str = "function_calling"
    ):
        """Configure for structured output."""
        return self._client.with_structured_output(schema, method=method)


def create_llm(
    model_string: str,
    settings: Any,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> UnifiedLLM:
    """
    Create LLM from config string.

    Args:
        model_string: Format "provider:model" or just "model"
        settings: Settings instance
        temperature: Temperature
        max_tokens: Max tokens

    Returns:
        UnifiedLLM instance
    """
    if ":" in model_string:
        provider, model = model_string.split(":", 1)
    else:
        provider = "openai"
        model = model_string

    # Get provider-specific settings
    api_key = None
    base_url = None

    if provider in ["openai", "openrouter", "302ai"]:
        api_key = settings.openai_api_key
        base_url = settings.openai_base_url
    elif provider == "anthropic":
        api_key = settings.anthropic_api_key
    elif provider == "ollama":
        base_url = settings.ollama_base_url

    logger.info(f"Creating LLM: {provider}:{model}")

    return UnifiedLLM(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )

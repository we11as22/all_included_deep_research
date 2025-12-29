"""LLM factory for chat and research models."""

from __future__ import annotations

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.config.settings import Settings
from src.llm.mock import MockChatModel

logger = structlog.get_logger(__name__)


def create_chat_model(
    model_str: str,
    settings: Settings,
    max_tokens: int,
    temperature: float = 0.7,
) -> BaseChatModel:
    """Create a chat model from provider:model string."""
    if settings.llm_mode == "mock" or model_str.startswith("mock"):
        logger.info("using_mock_llm")
        return MockChatModel()

    if ":" in model_str:
        provider, model_name = model_str.split(":", 1)
    else:
        provider = "openai"
        model_name = model_str

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not configured")

        llm_kwargs = {
            "model": model_name,
            "api_key": settings.openai_api_key,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if settings.openai_base_url:
            llm_kwargs["base_url"] = settings.openai_base_url
            if "openrouter.ai" in settings.openai_base_url:
                llm_kwargs["default_headers"] = {
                    "HTTP-Referer": "https://github.com/all-included-deep-research",
                    "X-Title": "All-Included Deep Research",
                }

        logger.debug("creating_openai_model", model=model_name)
        return ChatOpenAI(**llm_kwargs)

    if provider in {"anthropic", "claude"}:
        if not settings.anthropic_api_key:
            raise ValueError("Anthropic API key not configured")

        logger.debug("creating_anthropic_model", model=model_name)
        return ChatAnthropic(
            model=model_name,
            api_key=settings.anthropic_api_key,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")

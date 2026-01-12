"""LLM factory for chat and research models."""

from __future__ import annotations

from typing import Optional, Type

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.config.settings import Settings
from src.llm.mock import MockChatModel

logger = structlog.get_logger(__name__)


def create_chat_model(
    model_str: str,
    settings: Settings,
    max_tokens: int,
    temperature: float = 0.7,
    structured_output: Optional[Type[BaseModel]] = None,
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
        if not settings.openai_api_key or not settings.openai_api_key.strip():
            raise ValueError("OpenAI API key not configured. Please set OPENAI_API_KEY in backend/.env file")

        llm_kwargs = {
            "model": model_name,
            "api_key": settings.openai_api_key,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Support for any OpenAI-compatible API (OpenRouter, 302.AI, etc.)
        if settings.openai_base_url:
            llm_kwargs["base_url"] = settings.openai_base_url
            
            # Build headers for OpenAI-compatible APIs
            headers = {}
            
            # Use explicit header settings if provided
            if settings.openai_api_http_referer:
                headers["HTTP-Referer"] = settings.openai_api_http_referer
            elif "openrouter.ai" in settings.openai_base_url:
                # Default headers for OpenRouter if not explicitly set
                headers["HTTP-Referer"] = "https://github.com/all-included-deep-research"
            
            if settings.openai_api_x_title:
                headers["X-Title"] = settings.openai_api_x_title
            elif "openrouter.ai" in settings.openai_base_url:
                # Default headers for OpenRouter if not explicitly set
                headers["X-Title"] = "All-Included Deep Research"
            
            if headers:
                llm_kwargs["default_headers"] = headers
                logger.debug(
                    "using_openai_compatible_api",
                    base_url=settings.openai_base_url,
                    headers=list(headers.keys()),
                )

        logger.info(
            "creating_openai_model",
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            base_url=settings.openai_base_url or "default (api.openai.com)",
        )
        llm = ChatOpenAI(**llm_kwargs)
        
        # CRITICAL: Verify max_tokens was set correctly
        if hasattr(llm, "max_tokens"):
            actual_max_tokens = llm.max_tokens
            if actual_max_tokens != max_tokens:
                logger.warning(
                    "max_tokens mismatch",
                    expected=max_tokens,
                    actual=actual_max_tokens,
                    model=model_name,
                )
            else:
                logger.debug("max_tokens verified", max_tokens=actual_max_tokens, model=model_name)
        
        # Apply structured output if requested
        if structured_output:
            try:
                # Use function_calling method for better OpenAI compatibility
                llm = llm.with_structured_output(structured_output, method="function_calling")
                # CRITICAL: Verify max_tokens is preserved after with_structured_output
                if hasattr(llm, "max_tokens"):
                    actual_max_tokens = llm.max_tokens
                    if actual_max_tokens != max_tokens:
                        logger.warning(
                            "max_tokens changed after with_structured_output",
                            original=max_tokens,
                            after=actual_max_tokens,
                            schema=structured_output.__name__,
                        )
                    else:
                        logger.debug(
                            "max_tokens preserved after with_structured_output",
                            max_tokens=actual_max_tokens,
                            schema=structured_output.__name__,
                        )
            except TypeError:
                # Fallback if method parameter not supported
                llm = llm.with_structured_output(structured_output)
            logger.debug("applied_structured_output", schema=structured_output.__name__)
        
        return llm

    if provider in {"anthropic", "claude"}:
        if not settings.anthropic_api_key:
            raise ValueError("Anthropic API key not configured")

        logger.debug("creating_anthropic_model", model=model_name)
        llm = ChatAnthropic(
            model=model_name,
            api_key=settings.anthropic_api_key,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        # Apply structured output if requested
        if structured_output:
            try:
                # Use function_calling method for better OpenAI compatibility
                llm = llm.with_structured_output(structured_output, method="function_calling")
            except TypeError:
                # Fallback if method parameter not supported
                llm = llm.with_structured_output(structured_output)
            logger.debug("applied_structured_output", schema=structured_output.__name__)
        
        return llm

    raise ValueError(f"Unsupported LLM provider: {provider}")

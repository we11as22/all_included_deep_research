"""LLM factory for chat and research models."""

from __future__ import annotations

import asyncio
import inspect
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
    provider_order: Optional[str] = None,
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

        # CRITICAL: Get max_retries from settings (default 3 if not set)
        max_retries = getattr(settings, "max_retries", 3)
        
        llm_kwargs = {
            "model": model_name,
            "api_key": settings.openai_api_key,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "max_retries": max_retries,  # CRITICAL: Enable retry on client level
        }

        # Support for any OpenAI-compatible API (OpenRouter, 302.AI, etc.)
        is_openrouter = False
        if settings.openai_base_url:
            llm_kwargs["base_url"] = settings.openai_base_url
            is_openrouter = "openrouter.ai" in settings.openai_base_url
            
            # Build headers for OpenAI-compatible APIs
            headers = {}
            
            # Use explicit header settings if provided
            if settings.openai_api_http_referer:
                headers["HTTP-Referer"] = settings.openai_api_http_referer
            elif is_openrouter:
                # Default headers for OpenRouter if not explicitly set
                headers["HTTP-Referer"] = "https://github.com/all-included-deep-research"
            
            if settings.openai_api_x_title:
                headers["X-Title"] = settings.openai_api_x_title
            elif is_openrouter:
                # Default headers for OpenRouter if not explicitly set
                headers["X-Title"] = "All-Included Deep Research"
            
            if headers:
                llm_kwargs["default_headers"] = headers
                logger.debug(
                    "using_openai_compatible_api",
                    base_url=settings.openai_base_url,
                    headers=list(headers.keys()),
                )
            
        # CRITICAL: Add provider order for OpenRouter (only works with OpenRouter API)
        # Parse provider order before creating client
        providers = None
        provider_order_set_in_kwargs = False
        if is_openrouter and provider_order and provider_order.strip():
            providers = [p.strip() for p in provider_order.split(",") if p.strip()]
            if providers:
                # Try to pass extra_body directly in kwargs (if ChatOpenAI supports it)
                # Note: This may not work in all versions, so we'll also patch the client after creation
                try:
                    if "extra_body" not in llm_kwargs:
                        llm_kwargs["extra_body"] = {}
                    llm_kwargs["extra_body"]["provider"] = {"order": providers}
                    provider_order_set_in_kwargs = True
                except Exception:
                    # If extra_body is not supported in kwargs, we'll patch after creation
                    pass

        logger.info(
            "creating_openai_model",
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            base_url=settings.openai_base_url or "default (api.openai.com)",
            has_provider_order=providers is not None,
        )
        llm = ChatOpenAI(**llm_kwargs)
        
        # CRITICAL: If extra_body wasn't set in kwargs, patch the client after creation
        if is_openrouter and providers and not provider_order_set_in_kwargs:
            try:
                # Patch the underlying OpenAI client to include provider parameter
                if hasattr(llm, "client") and hasattr(llm.client, "chat"):
                    original_create = llm.client.chat.completions.create
                    
                    async def create_with_provider(*args, **kwargs):
                        """Wrapper to add provider parameter to requests (async)."""
                        if "extra_body" not in kwargs:
                            kwargs["extra_body"] = {}
                        kwargs["extra_body"]["provider"] = {"order": providers}
                        return await original_create(*args, **kwargs)
                    
                    def create_with_provider_sync(*args, **kwargs):
                        """Wrapper to add provider parameter to requests (sync)."""
                        if "extra_body" not in kwargs:
                            kwargs["extra_body"] = {}
                        kwargs["extra_body"]["provider"] = {"order": providers}
                        return original_create(*args, **kwargs)
                    
                    # Check if original method is async
                    if inspect.iscoroutinefunction(original_create):
                        llm.client.chat.completions.create = create_with_provider
                    else:
                        llm.client.chat.completions.create = create_with_provider_sync
                    
                    logger.info(
                        "added_provider_order_via_patch",
                        model=model_name,
                        providers=providers,
                        note="Provider order added via client patching"
                    )
                else:
                    logger.warning(
                        "cannot_patch_client",
                        model=model_name,
                        note="ChatOpenAI client structure not as expected"
                    )
            except Exception as e:
                logger.warning(
                    "failed_to_add_provider_order",
                    model=model_name,
                    error=str(e),
                    note="Failed to add provider order, continuing without it"
                )
        elif is_openrouter and providers and provider_order_set_in_kwargs:
            logger.debug(
                "provider_order_set_via_kwargs",
                model=model_name,
                providers=providers,
                note="Provider order set via kwargs (extra_body)"
            )
        
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

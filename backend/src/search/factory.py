"""Search provider factory."""

import structlog

from src.config.settings import Settings
from src.search.base import SearchProvider
from src.search.mock_provider import MockSearchProvider
from src.search.searxng_provider import SearXNGSearchProvider
from src.search.tavily_provider import TavilySearchProvider

logger = structlog.get_logger(__name__)


def create_search_provider(settings: Settings) -> SearchProvider:
    """
    Create search provider based on configuration.

    Args:
        settings: Application settings

    Returns:
        Configured SearchProvider instance

    Raises:
        ValueError: If search provider configuration is invalid
    """
    if settings.search_provider == "mock":
        logger.info("Creating MockSearchProvider")
        return MockSearchProvider()

    if settings.search_provider == "tavily":
        if not settings.tavily_api_key:
            if settings.llm_mode == "mock":
                logger.info("Tavily key missing; falling back to MockSearchProvider in mock mode")
                return MockSearchProvider()
            raise ValueError("Tavily API key is required when using Tavily search provider")

        logger.info("Creating TavilySearchProvider")
        return TavilySearchProvider(api_key=settings.tavily_api_key)

    elif settings.search_provider == "searxng":
        if not settings.searxng_instance_url:
            raise ValueError(
                "SearXNG instance URL is required when using SearXNG search provider"
            )

        logger.info("Creating SearXNGSearchProvider", instance_url=settings.searxng_instance_url)
        return SearXNGSearchProvider(
            instance_url=settings.searxng_instance_url,
            language=settings.searxng_language,
            categories=settings.searxng_categories,
            engines=settings.searxng_engines,
            safesearch=settings.searxng_safesearch,
        )

    else:
        raise ValueError(
            f"Unknown search provider: {settings.search_provider}. "
            f"Supported providers: tavily, searxng, mock"
        )


def get_search_provider() -> SearchProvider:
    """
    Get search provider using global settings.

    Returns:
        Configured SearchProvider instance
    """
    from src.config.settings import get_settings

    settings = get_settings()
    return create_search_provider(settings)

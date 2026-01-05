"""Live LLM + search service smoke tests (no mocks)."""

import pytest

from src.config.settings import get_settings
from src.llm.provider_abstraction import create_llm
from src.search.factory import create_search_provider
from src.search.scraper import WebScraper
from src.streaming.sse import ResearchStreamingGenerator
from src.workflow.search.service import SearchService


def _build_service(settings) -> SearchService:
    classifier_llm = create_llm(settings.chat_model, settings, temperature=0.3, max_tokens=512)
    research_llm = create_llm(settings.research_model, settings, temperature=0.7, max_tokens=2048)
    writer_llm = create_llm(settings.research_model, settings, temperature=0.7, max_tokens=2048)
    search_provider = create_search_provider(settings)
    scraper = WebScraper(
        timeout=settings.scraper_timeout,
        use_playwright=settings.scraper_use_playwright,
        scroll_enabled=settings.scraper_scroll_enabled,
        scroll_pause=settings.scraper_scroll_pause,
        max_scrolls=settings.scraper_max_scrolls,
    )
    return SearchService(
        classifier_llm=classifier_llm,
        research_llm=research_llm,
        writer_llm=writer_llm,
        search_provider=search_provider,
        scraper=scraper,
    )


@pytest.mark.asyncio
async def test_search_service_web_live():
    settings = get_settings()
    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY not set")

    service = _build_service(settings)
    stream = ResearchStreamingGenerator(session_id="test_web_live")

    answer = await service.answer(
        query="What is Python programming language?",
        chat_history=[],
        stream=stream,
        force_mode="web",
    )

    assert isinstance(answer, str)
    assert len(answer) > 0


@pytest.mark.asyncio
async def test_search_service_deep_live():
    settings = get_settings()
    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY not set")

    service = _build_service(settings)
    stream = ResearchStreamingGenerator(session_id="test_deep_live")

    answer = await service.answer(
        query="How does photosynthesis work?",
        chat_history=[],
        stream=stream,
        force_mode="deep",
    )

    assert isinstance(answer, str)
    assert len(answer) > 0

"""OpenRouter + HuggingFace integration tests."""

from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.config.settings import Settings
from src.embeddings.huggingface_provider import HuggingFaceEmbeddingProvider


def _load_settings() -> Settings:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        return Settings(_env_file=env_path)
    return Settings()


def _build_llm(settings: Settings) -> ChatOpenAI:
    llm_kwargs = {
        "model": "gpt-4o-mini",
        "api_key": settings.openai_api_key,
        "max_tokens": 100,
        "temperature": 0.7,
    }

    if settings.openai_base_url:
        llm_kwargs["base_url"] = settings.openai_base_url
        if "openrouter.ai" in settings.openai_base_url:
            headers = {}
            if settings.openai_api_http_referer:
                headers["HTTP-Referer"] = settings.openai_api_http_referer
            if settings.openai_api_x_title:
                headers["X-Title"] = settings.openai_api_x_title
            if headers:
                llm_kwargs["default_headers"] = headers

    return ChatOpenAI(**llm_kwargs)


@pytest.mark.asyncio
async def test_settings_load():
    settings = _load_settings()

    assert settings.embedding_provider
    assert settings.research_model
    assert settings.embedding_dimension > 0


@pytest.mark.asyncio
async def test_huggingface_embeddings():
    settings = _load_settings()
    if settings.embedding_provider != "huggingface":
        pytest.skip("Embedding provider is not HuggingFace")
    pytest.importorskip("sentence_transformers")

    provider = HuggingFaceEmbeddingProvider(
        model=settings.huggingface_model,
        api_key=settings.huggingface_api_key,
        use_local=settings.huggingface_use_local,
    )

    embedding = await provider.embed_text("This is a test sentence for embeddings.")
    assert isinstance(embedding, list)
    assert len(embedding) == provider.get_dimension()


@pytest.mark.asyncio
async def test_openrouter_llm():
    settings = _load_settings()
    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY not set")

    llm = _build_llm(settings)
    response = await llm.ainvoke([HumanMessage(content="Say 'Hello from OpenRouter!' in one sentence.")])

    assert response.content


def test_workflow_factory_imports():
    from src.workflow import create_search_service, create_research_graph

    assert callable(create_search_service)
    assert callable(create_research_graph)

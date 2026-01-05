"""Basic tests for core functionality."""

import pytest


def test_imports():
    """Test that all main modules can be imported."""
    # Config
    from src.config.settings import Settings, get_settings

    settings = get_settings()
    assert settings is not None

    # Database
    from src.database.schema import Base, MemoryChunkModel, MemoryFileModel

    assert Base is not None

    # Workflow
    from src.workflow import (
        SearchService,
        create_search_service,
        create_research_graph,
        ResearchState,
    )

    assert SearchService is not None
    assert create_search_service is not None
    assert create_research_graph is not None
    assert ResearchState is not None

    # API
    from src.api.app import create_app

    app = create_app()
    assert app is not None

    # Streaming
    from src.streaming import OpenAIStreamingGenerator, ResearchStreamingGenerator

    assert OpenAIStreamingGenerator is not None
    assert ResearchStreamingGenerator is not None


def test_settings_validation():
    """Test settings validation."""
    from src.config.settings import Settings

    # Should work with minimal settings
    settings = Settings(
        postgres_password="test",
        openai_api_key="sk-test",
        tavily_api_key="tvly-test",
    )

    assert settings.postgres_password == "test"
    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 8000


def test_streaming_generator():
    """Test streaming generator basic functionality."""
    from src.streaming import StreamingGenerator

    gen = StreamingGenerator()

    # Add data
    gen.add("test1")
    gen.add("test2")
    gen.finish()

    # Should not add after finish
    gen.add("test3")

    assert gen._finished is True


def test_research_state():
    """Test research state structure."""
    from src.workflow.research.state import ResearchPlan, ResearchTopic

    topic = ResearchTopic(
        reasoning="Test reasoning",
        topic="Example topic",
        description="Test description",
        priority="high",
    )
    plan = ResearchPlan(reasoning="Plan reasoning", topics=[topic])

    assert plan.topics[0].topic == "Example topic"


def test_mode_configurations():
    """Test research mode configurations."""
    from src.config.settings import get_settings

    settings = get_settings()

    # Iterations should be positive and monotonic by mode.
    assert settings.speed_max_iterations > 0
    assert settings.balanced_max_iterations >= settings.speed_max_iterations
    assert settings.quality_max_iterations >= settings.balanced_max_iterations

    # Concurrency should be positive and monotonic by mode.
    assert settings.speed_max_concurrent > 0
    assert settings.balanced_max_concurrent >= settings.speed_max_concurrent
    assert settings.quality_max_concurrent >= settings.balanced_max_concurrent


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

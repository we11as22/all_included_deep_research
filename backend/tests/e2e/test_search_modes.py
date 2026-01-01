"""End-to-end tests for all search modes."""

import asyncio
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

import pytest


@pytest.mark.asyncio
async def test_chat_mode():
    """Test chat mode (no sources)."""
    from src.workflow.search.service import SearchService
    from src.streaming.sse import ResearchStreamingGenerator
    from tests.mocks.mock_llm import MockChatModel

    # Create service with mock LLM
    service = SearchService(
        classifier_llm=MockChatModel(["Chat response about Python"]),
        research_llm=MockChatModel(),
        writer_llm=MockChatModel(),
        search_provider=None,  # Not used in chat mode
        scraper=None,
    )

    stream = ResearchStreamingGenerator(session_id="test_chat")

    # Test chat mode
    answer = await service.answer(
        query="What is Python?",
        chat_history=[],
        stream=stream,
        force_mode="chat"
    )

    assert isinstance(answer, str)
    assert len(answer) > 0
    print(f"\n✓ Chat mode answer: {answer[:100]}...")


@pytest.mark.asyncio
async def test_web_search_mode():
    """Test web search mode (speed: 2 iterations)."""
    from src.workflow.search.service import SearchService
    from src.streaming.sse import ResearchStreamingGenerator
    from tests.mocks.mock_llm import MockChatModel
    from tests.mocks.mock_search import MockSearchProvider
    from tests.mocks.mock_scraper import MockScraper

    # Create mocks
    classifier_llm = MockChatModel()
    research_llm = MockChatModel()
    writer_llm = MockChatModel()
    search_provider = MockSearchProvider()
    scraper = MockScraper()

    service = SearchService(
        classifier_llm=classifier_llm,
        research_llm=research_llm,
        writer_llm=writer_llm,
        search_provider=search_provider,
        scraper=scraper,
    )

    stream = ResearchStreamingGenerator(session_id="test_web")

    # Test web search mode
    answer = await service.answer(
        query="What is Python?",
        chat_history=[],
        stream=stream,
        force_mode="web"
    )

    assert isinstance(answer, str)
    assert len(answer) > 0
    assert "Sources" in answer or "[1]" in answer  # Should have citations
    print(f"\n✓ Web search answer: {answer[:200]}...")


@pytest.mark.asyncio
async def test_deep_search_mode():
    """Test deep search mode (balanced: 6 iterations)."""
    from src.workflow.search.service import SearchService
    from src.streaming.sse import ResearchStreamingGenerator
    from tests.mocks.mock_llm import MockChatModel
    from tests.mocks.mock_search import MockSearchProvider
    from tests.mocks.mock_scraper import MockScraper

    # Create mocks
    classifier_llm = MockChatModel()
    research_llm = MockChatModel()
    writer_llm = MockChatModel()
    search_provider = MockSearchProvider()
    scraper = MockScraper()

    service = SearchService(
        classifier_llm=classifier_llm,
        research_llm=research_llm,
        writer_llm=writer_llm,
        search_provider=search_provider,
        scraper=scraper,
    )

    stream = ResearchStreamingGenerator(session_id="test_deep")

    # Test deep search mode
    answer = await service.answer(
        query="How does quantum computing work?",
        chat_history=[],
        stream=stream,
        force_mode="deep"
    )

    assert isinstance(answer, str)
    assert len(answer) > 0
    assert "Sources" in answer or "[1]" in answer  # Should have citations
    print(f"\n✓ Deep search answer: {answer[:200]}...")


@pytest.mark.asyncio
async def test_classifier_routing():
    """Test that classifier routes queries correctly."""
    from src.workflow.search.classifier import classify_query
    from tests.mocks.mock_llm import MockChatModel

    mock_llm = MockChatModel()

    # Test classification
    classification = await classify_query(
        query="What is Python?",
        chat_history=[],
        llm=mock_llm
    )

    assert classification.query_type in ["simple", "research", "factual", "opinion", "comparison", "news"]
    assert classification.suggested_mode in ["chat", "web", "deep", "research_speed", "research_balanced", "research_quality"]
    assert isinstance(classification.standalone_query, str)
    assert isinstance(classification.requires_sources, bool)

    print(f"\n✓ Classification:")
    print(f"  - Type: {classification.query_type}")
    print(f"  - Mode: {classification.suggested_mode}")
    print(f"  - Standalone: {classification.standalone_query}")
    print(f"  - Requires sources: {classification.requires_sources}")


@pytest.mark.asyncio
async def test_research_agent_loop():
    """Test research agent ReAct loop."""
    from src.workflow.search.researcher import research_agent
    from src.workflow.search.classifier import QueryClassification
    from src.streaming.sse import ResearchStreamingGenerator
    from tests.mocks.mock_llm import MockChatModel
    from tests.mocks.mock_search import MockSearchProvider
    from tests.mocks.mock_scraper import MockScraper

    # Create classification
    classification = QueryClassification(
        reasoning="Test",
        query_type="factual",
        standalone_query="What is Python?",
        suggested_mode="web",
        requires_sources=True,
        time_sensitive=False
    )

    # Run research agent
    results = await research_agent(
        query="What is Python?",
        classification=classification,
        mode="speed",  # 2 iterations
        llm=MockChatModel(),
        search_provider=MockSearchProvider(),
        scraper=MockScraper(),
        stream=ResearchStreamingGenerator(session_id="test_research"),
        chat_history=[]
    )

    assert "sources" in results
    assert "scraped_content" in results
    assert "reasoning_history" in results
    assert isinstance(results["sources"], list)
    assert isinstance(results["scraped_content"], list)

    print(f"\n✓ Research agent results:")
    print(f"  - Sources: {len(results['sources'])}")
    print(f"  - Scraped: {len(results['scraped_content'])}")
    print(f"  - Reasoning steps: {len(results['reasoning_history'])}")


@pytest.mark.asyncio
async def test_writer_agent():
    """Test writer agent citation synthesis."""
    from src.workflow.search.writer import writer_agent
    from src.streaming.sse import ResearchStreamingGenerator
    from tests.mocks.mock_llm import MockChatModel

    # Mock research results
    research_results = {
        "sources": [
            {"title": "Python Docs", "url": "https://python.org", "snippet": "Python is..."},
            {"title": "Wikipedia", "url": "https://wikipedia.org", "snippet": "Python language..."},
        ],
        "scraped_content": [
            {"title": "Tutorial", "url": "https://tutorial.com", "content": "Learn Python..."},
        ],
        "reasoning_history": ["Step 1", "Step 2"]
    }

    # Run writer agent
    answer = await writer_agent(
        query="What is Python?",
        research_results=research_results,
        llm=MockChatModel(),
        stream=ResearchStreamingGenerator(session_id="test_writer"),
        mode="balanced",
        chat_history=[]
    )

    assert isinstance(answer, str)
    assert len(answer) > 0
    assert "Sources" in answer  # Should have sources section

    print(f"\n✓ Writer agent answer:")
    print(f"  - Length: {len(answer)} chars")
    print(f"  - Preview: {answer[:150]}...")


if __name__ == "__main__":
    print("Running search modes E2E tests...\n")
    print("=" * 70)

    # Run tests
    asyncio.run(test_chat_mode())
    print()
    asyncio.run(test_web_search_mode())
    print()
    asyncio.run(test_deep_search_mode())
    print()
    asyncio.run(test_classifier_routing())
    print()
    asyncio.run(test_research_agent_loop())
    print()
    asyncio.run(test_writer_agent())

    print("\n" + "=" * 70)
    print("✅ All search mode tests completed!")

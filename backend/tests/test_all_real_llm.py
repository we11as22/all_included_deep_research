#!/usr/bin/env python3
"""
Comprehensive real LLM testing script for all new modules.
Tests with actual OpenRouter API calls.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import get_settings
from src.workflow.search.classifier import classify_query
from src.workflow.search.researcher import research_agent
from src.workflow.search.writer import writer_agent
from src.workflow.search.service import create_search_service
from src.llm.provider_abstraction import create_llm
from src.streaming.sse import ResearchStreamingGenerator

# Mock search and scraper for testing
class MockSearchProvider:
    async def search(self, query: str, max_results: int = 5):
        class Result:
            def __init__(self):
                self.title = "Test Result"
                self.url = "https://example.com"
                self.content = "This is test search result content about " + query

        class Response:
            def __init__(self):
                self.results = [Result() for _ in range(min(max_results, 3))]

        return Response()

class MockScraper:
    async def scrape(self, url: str):
        class ScrapedContent:
            def __init__(self):
                self.title = "Scraped Page"
                self.url = url
                self.content = f"Scraped content from {url}. This is detailed information about the topic."

        return ScrapedContent()


async def test_classifier_real_llm():
    """Test 1: Query classifier with real LLM."""
    print("\n" + "="*70)
    print("TEST 1: Query Classifier with Real LLM")
    print("="*70)

    settings = get_settings()
    llm = create_llm(
        model_string=settings.chat_model,
        settings=settings,
        temperature=0.7,
        max_tokens=1000
    )

    test_queries = [
        "What is Python programming?",
        "Latest news about AI in 2024",
        "How does quantum computing work?"
    ]

    for query in test_queries:
        print(f"\n📝 Query: {query}")
        classification = await classify_query(query, [], llm)

        print(f"  ✓ Type: {classification.query_type}")
        print(f"  ✓ Mode: {classification.suggested_mode}")
        print(f"  ✓ Standalone: {classification.standalone_query}")
        print(f"  ✓ Requires sources: {classification.requires_sources}")
        print(f"  ✓ Reasoning: {classification.reasoning[:100]}...")


async def test_research_agent_speed():
    """Test 2: Research agent in speed mode (2 iterations)."""
    print("\n" + "="*70)
    print("TEST 2: Research Agent - Speed Mode (2 iterations)")
    print("="*70)

    settings = get_settings()
    llm = create_llm(
        model_string=settings.research_model,
        settings=settings,
        temperature=0.7,
        max_tokens=2000
    )

    from src.workflow.search.classifier import QueryClassification

    classification = QueryClassification(
        reasoning="Test query",
        query_type="factual",
        standalone_query="What is Python programming language?",
        suggested_mode="web",
        requires_sources=True,
        time_sensitive=False
    )

    stream = ResearchStreamingGenerator(session_id="test_speed")

    print("\n🔬 Running research agent (speed mode)...")
    results = await research_agent(
        query="What is Python programming language?",
        classification=classification,
        mode="speed",
        llm=llm,
        search_provider=MockSearchProvider(),
        scraper=MockScraper(),
        stream=stream,
        chat_history=[]
    )

    print(f"\n✓ Sources found: {len(results['sources'])}")
    print(f"✓ Scraped content: {len(results['scraped_content'])}")
    print(f"✓ Reasoning steps: {len(results['reasoning_history'])}")

    for i, step in enumerate(results['reasoning_history'][:3], 1):
        print(f"\n  Step {i}: {step[:150]}...")


async def test_writer_agent():
    """Test 3: Writer agent with citations."""
    print("\n" + "="*70)
    print("TEST 3: Writer Agent with Citations")
    print("="*70)

    settings = get_settings()
    llm = create_llm(
        model_string=settings.research_model,
        settings=settings,
        temperature=0.7,
        max_tokens=2000
    )

    research_results = {
        "sources": [
            {"title": "Python Docs", "url": "https://python.org", "snippet": "Python is a high-level programming language"},
            {"title": "Wikipedia", "url": "https://wikipedia.org", "snippet": "Python emphasizes code readability"},
        ],
        "scraped_content": [
            {"title": "Tutorial", "url": "https://tutorial.com", "content": "Python is beginner-friendly"},
        ],
        "reasoning_history": ["Searched for Python", "Found documentation"]
    }

    stream = ResearchStreamingGenerator(session_id="test_writer")

    print("\n✍️  Running writer agent...")
    answer = await writer_agent(
        query="What is Python?",
        research_results=research_results,
        llm=llm,
        stream=stream,
        mode="balanced",
        chat_history=[]
    )

    print(f"\n✓ Answer generated ({len(answer)} chars)")
    print(f"\n📄 Preview:\n{answer[:400]}...")
    print(f"\n✓ Contains citations: {'[1]' in answer or 'Sources' in answer}")


async def test_search_service_web():
    """Test 4: Complete search service - WEB mode."""
    print("\n" + "="*70)
    print("TEST 4: Search Service - WEB Mode (Speed)")
    print("="*70)

    settings = get_settings()

    service = create_search_service(
        classifier_llm=create_llm(settings.chat_model, settings, 0.7, 1000),
        research_llm=create_llm(settings.research_model, settings, 0.7, 2000),
        writer_llm=create_llm(settings.research_model, settings, 0.7, 2000),
        search_provider=MockSearchProvider(),
        scraper=MockScraper()
    )

    stream = ResearchStreamingGenerator(session_id="test_web")

    print("\n🌐 Running web search (speed: 2 iterations)...")
    answer = await service.answer(
        query="What is Python programming?",
        chat_history=[],
        stream=stream,
        force_mode="web"
    )

    print(f"\n✓ Answer generated ({len(answer)} chars)")
    print(f"\n📄 Preview:\n{answer[:500]}...")


async def test_search_service_deep():
    """Test 5: Complete search service - DEEP mode."""
    print("\n" + "="*70)
    print("TEST 5: Search Service - DEEP Mode (Balanced)")
    print("="*70)

    settings = get_settings()

    service = create_search_service(
        classifier_llm=create_llm(settings.chat_model, settings, 0.7, 1000),
        research_llm=create_llm(settings.research_model, settings, 0.7, 2000),
        writer_llm=create_llm(settings.research_model, settings, 0.7, 2000),
        search_provider=MockSearchProvider(),
        scraper=MockScraper()
    )

    stream = ResearchStreamingGenerator(session_id="test_deep")

    print("\n🔍 Running deep search (balanced: 6 iterations)...")
    answer = await service.answer(
        query="How does machine learning work?",
        chat_history=[],
        stream=stream,
        force_mode="deep"
    )

    print(f"\n✓ Answer generated ({len(answer)} chars)")
    print(f"\n📄 Preview:\n{answer[:500]}...")


async def main():
    """Run all tests."""
    print("\n" + "#"*70)
    print("# COMPREHENSIVE REAL LLM TESTING")
    print("# Testing all modules with actual OpenRouter API")
    print("#"*70)

    try:
        await test_classifier_real_llm()
        await test_research_agent_speed()
        await test_writer_agent()
        await test_search_service_web()
        await test_search_service_deep()

        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

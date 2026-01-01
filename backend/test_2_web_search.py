#!/usr/bin/env python3
"""Test 2: Web Search Mode (Speed: 2 iterations) with Real LLM"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import get_settings
from src.workflow.search.researcher import research_agent
from src.workflow.search.writer import writer_agent
from src.workflow.search.classifier import QueryClassification
from src.llm.provider_abstraction import create_llm
from src.streaming.sse import ResearchStreamingGenerator


# Mock search provider
class MockSearchProvider:
    async def search(self, query: str, max_results: int = 5):
        print(f"    🔍 Mock search: '{query}' (max {max_results} results)")

        class Result:
            def __init__(self, i):
                self.title = f"Result {i}: Python Programming Guide"
                self.url = f"https://python.org/guide/{i}"
                self.content = f"Python is a high-level programming language. Result {i} explains: Python was created by Guido van Rossum in 1991. It emphasizes code readability and simplicity."

        class Response:
            def __init__(self):
                self.results = [Result(i) for i in range(min(max_results, 3))]

        return Response()


class MockScraper:
    async def scrape(self, url: str):
        print(f"    📄 Mock scrape: {url}")

        class ScrapedContent:
            def __init__(self):
                self.title = "Python Official Documentation"
                self.url = url
                self.content = """Python is an interpreted, high-level, general-purpose programming language.
                Created by Guido van Rossum and first released in 1991.
                Python's design philosophy emphasizes code readability with its use of significant indentation.
                It provides constructs for clear programming on both small and large scales.
                Python features dynamic typing and automatic memory management."""

        return ScrapedContent()


async def main():
    print("\n" + "="*70)
    print("TEST 2: Web Search Mode (Speed: 2 iterations)")
    print("="*70)

    settings = get_settings()

    # Create LLMs
    research_llm = create_llm(settings.research_model, settings, 0.7, 2000)
    writer_llm = create_llm(settings.research_model, settings, 0.7, 2000)

    # Create classification
    classification = QueryClassification(
        reasoning="Factual query about Python programming",
        query_type="factual",
        standalone_query="What is Python programming language?",
        suggested_mode="web",
        requires_sources=True,
        time_sensitive=False
    )

    stream = ResearchStreamingGenerator(session_id="test_web")

    # Step 1: Research Agent
    print("\n🔬 STEP 1: Running Research Agent (speed mode: 2 iterations)...")
    results = await research_agent(
        query="What is Python programming language?",
        classification=classification,
        mode="speed",
        llm=research_llm,
        search_provider=MockSearchProvider(),
        scraper=MockScraper(),
        stream=stream,
        chat_history=[]
    )

    print(f"\n  ✓ Research complete:")
    print(f"    - Sources found: {len(results['sources'])}")
    print(f"    - Content scraped: {len(results['scraped_content'])}")
    print(f"    - Reasoning steps: {len(results['reasoning_history'])}")

    for i, step in enumerate(results['reasoning_history'], 1):
        print(f"\n    Step {i} reasoning: {step[:200]}...")

    # Step 2: Writer Agent
    print("\n✍️  STEP 2: Running Writer Agent (citations)...")
    answer = await writer_agent(
        query="What is Python programming language?",
        research_results=results,
        llm=writer_llm,
        stream=stream,
        mode="speed",
        chat_history=[]
    )

    print(f"\n  ✓ Answer generated ({len(answer)} chars)")
    print(f"\n📄 FINAL ANSWER:\n{answer}\n")

    has_citations = '[1]' in answer or 'Sources' in answer
    print(f"  ✓ Contains citations: {has_citations}")

    if not has_citations:
        print("  ⚠️  WARNING: No citations found in answer!")
        return 1

    print("\n✅ Web Search Test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

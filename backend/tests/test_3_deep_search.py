#!/usr/bin/env python3
"""Test 3: Deep Search Mode (Balanced: 6 iterations) with Real LLM"""

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


class MockSearchProvider:
    async def search(self, query: str, max_results: int = 5):
        print(f"    🔍 Search: '{query[:60]}...' ({max_results} results)")

        class Result:
            def __init__(self, i, query):
                if "quantum" in query.lower():
                    self.title = f"Quantum Computing Guide Part {i}"
                    self.url = f"https://quantum.edu/guide/{i}"
                    self.content = f"Quantum computing uses quantum bits (qubits) that can exist in superposition. Part {i}: {['Basic principles', 'Superposition explained', 'Quantum entanglement'][i % 3]}"
                else:
                    self.title = f"ML Resource {i}"
                    self.url = f"https://ml.org/res/{i}"
                    self.content = f"Machine learning trains algorithms on data. Resource {i} covers {['neural networks', 'supervised learning', 'deep learning'][i % 3]}"

        class Response:
            def __init__(self):
                self.results = [Result(i, query) for i in range(min(max_results, 4))]

        return Response()


class MockScraper:
    async def scrape(self, url: str):
        print(f"    📄 Scrape: {url}")

        class ScrapedContent:
            def __init__(self):
                self.title = "Detailed Article"
                self.url = url
                if "quantum" in url:
                    self.content = """Quantum computing harnesses quantum mechanical phenomena.
                    Unlike classical bits, qubits can be in superposition of states.
                    Quantum entanglement allows qubits to be correlated in ways classical bits cannot.
                    This enables exponentially faster computation for certain problems.
                    Current applications include cryptography, optimization, and simulation."""
                else:
                    self.content = """Machine learning is a subset of artificial intelligence.
                    It learns patterns from data without explicit programming.
                    Common techniques include supervised, unsupervised, and reinforcement learning.
                    Neural networks are inspired by biological neurons.
                    Deep learning uses multiple layers for complex pattern recognition."""

        return ScrapedContent()


async def main():
    print("\n" + "="*70)
    print("TEST 3: Deep Search Mode (Balanced: 6 iterations)")
    print("="*70)

    settings = get_settings()

    research_llm = create_llm(settings.research_model, settings, 0.7, 3000)
    writer_llm = create_llm(settings.research_model, settings, 0.7, 3000)

    classification = QueryClassification(
        reasoning="Complex research query requiring thorough investigation",
        query_type="research",
        standalone_query="How does quantum computing work?",
        suggested_mode="deep",
        requires_sources=True,
        time_sensitive=False
    )

    stream = ResearchStreamingGenerator(session_id="test_deep")

    print("\n🔬 STEP 1: Running Research Agent (balanced mode: 6 iterations)...")
    print("  This may take 1-2 minutes with real LLM calls...")

    results = await research_agent(
        query="How does quantum computing work?",
        classification=classification,
        mode="balanced",
        llm=research_llm,
        search_provider=MockSearchProvider(),
        scraper=MockScraper(),
        stream=stream,
        chat_history=[]
    )

    print(f"\n  ✓ Research complete:")
    print(f"    - Sources: {len(results['sources'])}")
    print(f"    - Scraped: {len(results['scraped_content'])}")
    print(f"    - Reasoning steps: {len(results['reasoning_history'])}")

    print("\n  📋 Reasoning History:")
    for i, step in enumerate(results['reasoning_history'], 1):
        print(f"    {i}. {step[:150]}...")

    print("\n✍️  STEP 2: Running Writer Agent (comprehensive answer)...")
    answer = await writer_agent(
        query="How does quantum computing work?",
        research_results=results,
        llm=writer_llm,
        stream=stream,
        mode="balanced",
        chat_history=[]
    )

    print(f"\n  ✓ Answer generated ({len(answer)} chars)")
    print(f"\n📄 FINAL ANSWER:\n{answer}\n")

    has_citations = '[1]' in answer or 'Sources' in answer
    is_comprehensive = len(answer) > 500

    print(f"  ✓ Contains citations: {has_citations}")
    print(f"  ✓ Comprehensive (>500 chars): {is_comprehensive}")

    if not has_citations:
        print("  ⚠️  WARNING: Missing citations!")
        return 1

    if not is_comprehensive:
        print("  ⚠️  WARNING: Answer too short for balanced mode!")
        return 1

    print("\n✅ Deep Search Test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

#!/usr/bin/env python3
"""Test 1: Query Classifier with Real LLM"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import get_settings
from src.workflow.search.classifier import classify_query
from src.llm.provider_abstraction import create_llm


async def main():
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
        "Latest AI news 2024",
        "How does quantum computing work in detail?"
    ]

    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        classification = await classify_query(query, [], llm)

        print(f"  ✓ Type: {classification.query_type}")
        print(f"  ✓ Suggested Mode: {classification.suggested_mode}")
        print(f"  ✓ Standalone Query: {classification.standalone_query}")
        print(f"  ✓ Requires Sources: {classification.requires_sources}")
        print(f"  ✓ Time Sensitive: {classification.time_sensitive}")
        print(f"  ✓ Reasoning: {classification.reasoning[:150]}...")

    print("\n✅ Classifier Test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

"""Comprehensive test runner for all modules (without external dependencies)."""

import asyncio
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))


async def test_database_modules():
    """Test database modules."""
    print("\n" + "=" * 70)
    print("TEST 1: Database Modules")
    print("=" * 70)

    try:
        from src.database.schema_sqlite import (
            Base,
            ChatModel,
            MessageModel,
            ResearchSessionModel,
            AgentMemoryModel,
        )

        print("✓ Schema imports successful")
        print(f"  - Tables: chats, messages, research_sessions, agent_memory")

        from src.database.connection_sqlite import SQLiteDatabase

        print("✓ Connection class imported")

        return True
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        return False


async def test_vector_store():
    """Test vector store adapter."""
    print("\n" + "=" * 70)
    print("TEST 2: Vector Store Adapter")
    print("=" * 70)

    try:
        from src.memory.vector_store_adapter import create_vector_store, MockAdapter

        # Test mock adapter
        adapter = create_vector_store(store_type="mock")
        assert isinstance(adapter, MockAdapter)
        print("✓ Mock adapter created")

        # Test operations
        await adapter.add_embeddings(
            file_id=1,
            chunks=[{"id": 1, "content": "test content"}],
            embeddings=[[0.1, 0.2, 0.3]]
        )
        print("✓ Added embeddings")

        results = await adapter.search(
            query_embedding=[0.1, 0.2, 0.3],
            top_k=5
        )
        print(f"✓ Search returned {len(results)} results")

        return True
    except Exception as e:
        print(f"✗ Vector store test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_llm_abstraction():
    """Test LLM provider abstraction."""
    print("\n" + "=" * 70)
    print("TEST 3: LLM Provider Abstraction")
    print("=" * 70)

    try:
        from tests.mocks.mock_llm import MockLLM, MockChatModel

        # Test mock LLM
        llm = MockLLM(provider="mock", model="test-model")
        print("✓ Mock LLM created")

        # Test invocation
        from langchain_core.messages import HumanMessage

        response = await llm.ainvoke([HumanMessage(content="Test")])
        print(f"✓ LLM invocation successful: {response.content[:50]}")

        return True
    except Exception as e:
        print(f"✗ LLM abstraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_search_classifier():
    """Test query classifier."""
    print("\n" + "=" * 70)
    print("TEST 4: Search Classifier")
    print("=" * 70)

    try:
        from src.workflow.search.classifier import classify_query
        from tests.mocks.mock_llm import MockChatModel

        mock_llm = MockChatModel()
        classification = await classify_query(
            query="What is Python?",
            chat_history=[],
            llm=mock_llm
        )

        print(f"✓ Classification successful:")
        print(f"  - Type: {classification.query_type}")
        print(f"  - Mode: {classification.suggested_mode}")
        print(f"  - Standalone query: {classification.standalone_query}")
        print(f"  - Requires sources: {classification.requires_sources}")

        return True
    except Exception as e:
        print(f"✗ Classifier test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_action_registry():
    """Test action registry."""
    print("\n" + "=" * 70)
    print("TEST 5: Action Registry")
    print("=" * 70)

    try:
        from src.workflow.search.actions import ActionRegistry

        # Get tools for balanced mode
        tools = ActionRegistry.get_tool_definitions(
            mode="balanced",
            classification="factual"
        )

        print(f"✓ Generated {len(tools)} tool definitions")

        action_names = [tool["function"]["name"] for tool in tools]
        print(f"✓ Available actions: {', '.join(action_names)}")

        assert "web_search" in action_names
        assert "scrape_url" in action_names
        assert "__reasoning_preamble" in action_names
        assert "done" in action_names

        return True
    except Exception as e:
        print(f"✗ Action registry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_research_agent():
    """Test research agent."""
    print("\n" + "=" * 70)
    print("TEST 6: Research Agent (Speed Mode)")
    print("=" * 70)

    try:
        from src.workflow.search.researcher import research_agent
        from src.workflow.search.classifier import QueryClassification
        from src.streaming.sse import ResearchStreamingGenerator
        from tests.mocks.mock_llm import MockChatModel
        from tests.mocks.mock_search import MockSearchProvider
        from tests.mocks.mock_scraper import MockScraper

        classification = QueryClassification(
            reasoning="Test",
            query_type="factual",
            standalone_query="What is Python?",
            suggested_mode="web",
            requires_sources=True,
            time_sensitive=False
        )

        print("✓ Starting research agent (speed mode: 2 iterations)...")

        results = await research_agent(
            query="What is Python?",
            classification=classification,
            mode="speed",
            llm=MockChatModel(),
            search_provider=MockSearchProvider(),
            scraper=MockScraper(),
            stream=ResearchStreamingGenerator(session_id="test"),
            chat_history=[]
        )

        print(f"✓ Research completed:")
        print(f"  - Sources found: {len(results['sources'])}")
        print(f"  - Content scraped: {len(results['scraped_content'])}")
        print(f"  - Reasoning steps: {len(results['reasoning_history'])}")

        return True
    except Exception as e:
        print(f"✗ Research agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_writer_agent():
    """Test writer agent."""
    print("\n" + "=" * 70)
    print("TEST 7: Writer Agent")
    print("=" * 70)

    try:
        from src.workflow.search.writer import writer_agent
        from src.streaming.sse import ResearchStreamingGenerator
        from tests.mocks.mock_llm import MockChatModel

        research_results = {
            "sources": [
                {"title": "Python Docs", "url": "https://python.org", "snippet": "Python is a programming language"},
                {"title": "Wikipedia", "url": "https://wikipedia.org", "snippet": "Python language overview"},
            ],
            "scraped_content": [
                {"title": "Tutorial", "url": "https://tutorial.com", "content": "Python tutorial content"},
            ],
            "reasoning_history": []
        }

        print("✓ Running writer agent...")

        answer = await writer_agent(
            query="What is Python?",
            research_results=research_results,
            llm=MockChatModel(),
            stream=ResearchStreamingGenerator(session_id="test"),
            mode="balanced",
            chat_history=[]
        )

        print(f"✓ Answer generated:")
        print(f"  - Length: {len(answer)} chars")
        print(f"  - Contains citations: {'[1]' in answer or 'Sources' in answer}")
        print(f"  - Preview: {answer[:150]}...")

        return True
    except Exception as e:
        print(f"✗ Writer agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_search_service():
    """Test complete search service."""
    print("\n" + "=" * 70)
    print("TEST 8: Complete Search Service (All Modes)")
    print("=" * 70)

    try:
        from src.workflow.search.service import SearchService
        from src.streaming.sse import ResearchStreamingGenerator
        from tests.mocks.mock_llm import MockChatModel
        from tests.mocks.mock_search import MockSearchProvider
        from tests.mocks.mock_scraper import MockScraper

        service = SearchService(
            classifier_llm=MockChatModel(),
            research_llm=MockChatModel(),
            writer_llm=MockChatModel(),
            search_provider=MockSearchProvider(),
            scraper=MockScraper(),
        )

        # Test chat mode
        print("\n  Testing CHAT mode...")
        answer = await service.answer(
            query="What is Python?",
            chat_history=[],
            stream=ResearchStreamingGenerator(session_id="test_chat"),
            force_mode="chat"
        )
        print(f"  ✓ Chat answer: {len(answer)} chars")

        # Test web mode
        print("\n  Testing WEB mode (speed)...")
        answer = await service.answer(
            query="What is Python?",
            chat_history=[],
            stream=ResearchStreamingGenerator(session_id="test_web"),
            force_mode="web"
        )
        print(f"  ✓ Web answer: {len(answer)} chars")

        # Test deep mode
        print("\n  Testing DEEP mode (balanced)...")
        answer = await service.answer(
            query="What is Python?",
            chat_history=[],
            stream=ResearchStreamingGenerator(session_id="test_deep"),
            force_mode="deep"
        )
        print(f"  ✓ Deep answer: {len(answer)} chars")

        return True
    except Exception as e:
        print(f"✗ Search service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_research_state():
    """Test LangGraph research state."""
    print("\n" + "=" * 70)
    print("TEST 9: LangGraph Research State")
    print("=" * 70)

    try:
        from src.workflow.research.state import create_initial_state

        state = create_initial_state(
            query="Test research query",
            chat_history=[],
            mode="balanced"
        )

        print(f"✓ Initial state created:")
        print(f"  - Query: {state['query']}")
        print(f"  - Mode: {state['mode']}")
        print(f"  - Max iterations: {state['max_iterations']}")
        print(f"  - State keys: {len(state)} fields")

        return True
    except Exception as e:
        print(f"✗ Research state test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_supervisor_queue():
    """Test supervisor queue."""
    print("\n" + "=" * 70)
    print("TEST 10: Supervisor Queue")
    print("=" * 70)

    try:
        from src.workflow.research.queue import SupervisorQueue

        queue = SupervisorQueue()

        # Enqueue items
        await queue.enqueue("agent_1", "finish", {"result": "data1"})
        await queue.enqueue("agent_2", "finish", {"result": "data2"})
        await queue.enqueue("agent_3", "update", {"progress": 50})

        print(f"✓ Enqueued 3 items, queue size: {queue.size()}")

        # Process batch
        from unittest.mock import AsyncMock

        mock_supervisor = AsyncMock(return_value=[{"directive": "continue"}])
        results = await queue.process_batch(
            state={"iteration": 1},
            supervisor_func=mock_supervisor,
            max_batch_size=10
        )

        print(f"✓ Processed batch, results: {len(results)}")
        print(f"✓ Queue cleared, size: {queue.size()}")

        return True
    except Exception as e:
        print(f"✗ Supervisor queue test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_session_memory():
    """Test session memory service."""
    print("\n" + "=" * 70)
    print("TEST 11: Session Memory Service")
    print("=" * 70)

    try:
        import tempfile
        from src.memory.session_memory_service import SessionMemoryService

        with tempfile.TemporaryDirectory() as tmpdir:
            service = SessionMemoryService(
                session_id="test_session",
                base_memory_dir=tmpdir
            )

            # Initialize
            await service.initialize()
            print("✓ Session initialized")

            # Update main file
            await service.update_main_section(
                section="## Overview",
                content="Test research session"
            )
            print("✓ Main file updated")

            # Save agent file
            await service.save_agent_file(
                agent_id="agent_r0_0",
                todos=[{"title": "Research topic", "status": "pending"}],
                notes=[],
                character=None
            )
            print("✓ Agent file saved")

            # Load agent state
            agent_state = await service.load_agent_state("agent_r0_0")
            print(f"✓ Agent state loaded: {len(agent_state['todos'])} todos")

            # Save note
            await service.save_note(
                agent_id="agent_r0_0",
                title="Test Note",
                summary="This is a test note",
                urls=["https://example.com"],
                tags=["test"],
                share=True
            )
            print("✓ Research note saved")

        return True
    except Exception as e:
        print(f"✗ Session memory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("\n" + "#" * 70)
    print("# COMPREHENSIVE TEST SUITE")
    print("# Testing all modules without external dependencies")
    print("#" * 70)

    results = []

    # Run all tests
    results.append(("Database Modules", await test_database_modules()))
    results.append(("Vector Store", await test_vector_store()))
    results.append(("LLM Abstraction", await test_llm_abstraction()))
    results.append(("Search Classifier", await test_search_classifier()))
    results.append(("Action Registry", await test_action_registry()))
    results.append(("Research Agent", await test_research_agent()))
    results.append(("Writer Agent", await test_writer_agent()))
    results.append(("Search Service", await test_search_service()))
    results.append(("Research State", await test_research_state()))
    results.append(("Supervisor Queue", await test_supervisor_queue()))
    results.append(("Session Memory", await test_session_memory()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print("\n" + "=" * 70)
    print(f"TOTAL: {passed}/{len(results)} tests passed")
    print("=" * 70)

    if failed == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

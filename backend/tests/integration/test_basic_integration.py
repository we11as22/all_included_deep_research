"""Basic integration tests for new modules."""

import asyncio
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_database_schema_imports():
    """Test that database schema can be imported."""
    from src.database.schema_sqlite import (
        Base,
        ChatModel,
        MessageModel,
        ResearchSessionModel,
        AgentMemoryModel
    )

    assert Base is not None
    assert ChatModel.__tablename__ == "chats"
    assert MessageModel.__tablename__ == "messages"
    assert ResearchSessionModel.__tablename__ == "research_sessions"
    assert AgentMemoryModel.__tablename__ == "agent_memory"


@pytest.mark.asyncio
async def test_vector_store_adapter():
    """Test vector store adapter factory."""
    from src.memory.vector_store_adapter import create_vector_store, MockAdapter

    # Test mock adapter creation
    adapter = create_vector_store(store_type="mock")
    assert isinstance(adapter, MockAdapter)

    # Test basic operations
    await adapter.add_embeddings(
        file_id=1,
        chunks=[{"id": 1, "content": "test"}],
        embeddings=[[0.1, 0.2, 0.3]]
    )

    results = await adapter.search(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=5
    )

    assert isinstance(results, list)
    assert len(results) > 0


@pytest.mark.asyncio
async def test_llm_provider_abstraction():
    """Test LLM provider abstraction."""
    from src.llm.provider_abstraction import create_llm
    from src.config.settings import Settings

    settings = Settings()

    # Test mock provider
    llm = create_llm(
        model_string="mock:test-model",
        settings=settings,
        temperature=0.7,
        max_tokens=1000
    )

    assert llm is not None
    assert llm.provider == "mock"


@pytest.mark.asyncio
async def test_search_classifier():
    """Test query classifier with mock LLM."""
    from src.workflow.search.classifier import classify_query, QueryClassification
    from pydantic import BaseModel, Field

    # Create mock LLM
    mock_llm = MagicMock()

    # Mock the structured output
    class MockStructuredLLM:
        async def ainvoke(self, messages):
            return QueryClassification(
                reasoning="Test reasoning",
                query_type="factual",
                standalone_query="What is Python?",
                suggested_mode="web",
                requires_sources=True,
                time_sensitive=False
            )

    mock_llm.with_structured_output = lambda schema, **kwargs: MockStructuredLLM()

    result = await classify_query(
        query="What is Python?",
        chat_history=[],
        llm=mock_llm
    )

    assert isinstance(result, QueryClassification)
    assert result.query_type == "factual"
    assert result.suggested_mode == "web"


@pytest.mark.asyncio
async def test_action_registry():
    """Test action registry."""
    from src.workflow.search.actions import ActionRegistry

    # Get tool definitions
    tools = ActionRegistry.get_tool_definitions(
        mode="balanced",
        classification="factual"
    )

    assert isinstance(tools, list)
    assert len(tools) > 0

    # Check for expected actions
    action_names = [tool["function"]["name"] for tool in tools]
    assert "web_search" in action_names
    assert "scrape_url" in action_names
    assert "done" in action_names


@pytest.mark.asyncio
async def test_research_state():
    """Test research state creation."""
    from src.workflow.research.state import create_initial_state, ResearchState

    state = create_initial_state(
        query="Test query",
        chat_history=[],
        mode="balanced"
    )

    assert isinstance(state, dict)
    assert state["query"] == "Test query"
    assert state["mode"] == "balanced"
    assert "research_plan" in state
    assert "active_agents" in state


@pytest.mark.asyncio
async def test_supervisor_queue():
    """Test supervisor queue."""
    from src.workflow.research.queue import SupervisorQueue

    queue = SupervisorQueue()

    # Enqueue items
    await queue.enqueue("agent_1", "finish", {"topic": "Test", "finding": "Result"})
    await queue.enqueue("agent_2", "update", {"note": "Progress"})

    # Check queue size
    assert queue.size() == 2

    # Process batch
    mock_supervisor = AsyncMock(return_value=[{"directive": "continue"}])
    results = await queue.process_batch(
        state={"iteration": 1},
        supervisor_func=mock_supervisor,
        max_batch_size=10
    )

    assert isinstance(results, list)
    assert queue.size() == 0  # Queue should be empty after processing


@pytest.mark.asyncio
async def test_session_memory_service():
    """Test session memory service."""
    import tempfile
    from src.memory.session_memory_service import SessionMemoryService

    with tempfile.TemporaryDirectory() as tmpdir:
        service = SessionMemoryService(
            session_id="test_session_123",
            base_memory_dir=tmpdir
        )

        # Test initialization
        await service.initialize()

        # Test main file update
        await service.update_main_section(
            section="## Overview",
            content="Test content"
        )

        # Read main file
        main_content = await service.read_main()
        assert "Test content" in main_content

        # Save agent file
        await service.save_agent_file(
            agent_id="agent_r0_0",
            todos=[{"title": "Research X", "status": "pending"}],
            notes=[],
            character=None
        )

        # Load agent state
        agent_state = await service.load_agent_state("agent_r0_0")
        assert "todos" in agent_state
        assert len(agent_state["todos"]) == 1


if __name__ == "__main__":
    print("Running integration tests...")
    pytest.main([__file__, "-v", "--tb=short"])

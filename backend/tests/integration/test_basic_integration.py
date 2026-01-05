"""Basic integration tests for new modules."""

import asyncio
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

import pytest


@pytest.mark.asyncio
async def test_database_schema_imports():
    """Test that database schema can be imported."""
    from src.database.schema_sqlite import (
        Base,
        ChatModel,
        ChatMessageModel,
        ResearchSessionModel,
        AgentMemoryModel
    )

    assert Base is not None
    assert ChatModel.__tablename__ == "chats"
    assert ChatMessageModel.__tablename__ == "chat_messages"
    assert ResearchSessionModel.__tablename__ == "research_sessions"
    assert AgentMemoryModel.__tablename__ == "agent_memory"


@pytest.mark.asyncio
async def test_vector_store_adapter():
    """Test vector store adapter factory."""
    pytest.importorskip("faiss")
    from src.memory.vector_store_adapter import create_vector_store

    adapter = create_vector_store(store_type="faiss", dimension=3)

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
    from src.config.settings import get_settings
    from langchain_core.messages import HumanMessage

    settings = get_settings()
    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY not set")

    llm = create_llm(
        model_string=settings.chat_model,
        settings=settings,
        temperature=0.7,
        max_tokens=256
    )

    response = await llm.ainvoke([HumanMessage(content="Reply with the word: ping")])
    assert response.content


@pytest.mark.asyncio
async def test_search_classifier():
    """Test query classifier with real LLM."""
    from src.config.settings import get_settings
    from src.llm.provider_abstraction import create_llm
    from src.workflow.search.classifier import classify_query

    settings = get_settings()
    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY not set")

    llm = create_llm(settings.chat_model, settings, temperature=0.3, max_tokens=512)

    result = await classify_query(
        query="What is Python?",
        chat_history=[],
        llm=llm,
    )

    assert result.standalone_query
    assert result.query_type
    assert result.suggested_mode


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
    from src.streaming.sse import ResearchStreamingGenerator
    from src.config.settings import get_settings

    settings = get_settings()
    mode_config = {
        "max_iterations": settings.balanced_max_iterations,
    }

    state = create_initial_state(
        query="Test query",
        chat_history=[],
        mode="balanced",
        stream=ResearchStreamingGenerator(session_id="test_state"),
        session_id="test_state",
        mode_config=mode_config,
        settings=settings,
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
    async def supervisor_func(state, batch):
        return {"directives": [{"directive": "continue"}], "should_continue": True}

    results = await queue.process_batch(
        state={"iteration": 1},
        supervisor_func=supervisor_func,
        max_batch_size=10
    )

    assert isinstance(results, dict)
    assert results["directives"]
    assert queue.size() == 0  # Queue should be empty after processing


@pytest.mark.asyncio
async def test_session_memory_service():
    """Test session memory service."""
    import tempfile
    from pathlib import Path
    from src.memory.session_memory_service import SessionMemoryService

    with tempfile.TemporaryDirectory() as tmpdir:
        service = SessionMemoryService(
            session_id="test_session_123",
            base_memory_dir=tmpdir
        )

        # Read main file
        main_content = await service.read_main()
        assert "Research Session" in main_content

        # Save agent file
        await service.save_agent_file(
            agent_id="agent_r0_0",
            todos=[{"title": "Research X", "status": "pending"}],
            notes=[],
            character=None
        )

        agent_file = Path(tmpdir) / "sessions" / "test_session_123" / "agents" / "agent_r0_0.md"
        assert agent_file.exists()

        note_path = await service.save_note(
            agent_id="agent_r0_0",
            title="Test Note",
            summary="Summary",
            urls=["https://example.com"],
            tags=["test"],
            share=True,
        )
        assert (Path(tmpdir) / "sessions" / "test_session_123" / note_path).exists()


if __name__ == "__main__":
    print("Running integration tests...")
    pytest.main([__file__, "-v", "--tb=short"])

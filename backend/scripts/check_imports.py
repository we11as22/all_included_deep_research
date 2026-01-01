"""Check imports for all new modules."""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

print("Checking imports...\n")

try:
    print("✓ Checking database modules...")
    from src.database.schema_sqlite import Base, ChatModel, MessageModel, ResearchSessionModel, AgentMemoryModel
    from src.database.connection_sqlite import SQLiteDatabase
    print("  ✓ Database modules OK")
except Exception as e:
    print(f"  ✗ Database modules FAILED: {e}")

try:
    print("\n✓ Checking memory modules...")
    from src.memory.vector_store_adapter import VectorStoreAdapter, FAISAdapter, ChromaAdapter, MockAdapter, create_vector_store
    from src.memory.session_memory_service import SessionMemoryService
    print("  ✓ Memory modules OK")
except Exception as e:
    print(f"  ✗ Memory modules FAILED: {e}")

try:
    print("\n✓ Checking LLM provider...")
    from src.llm.provider_abstraction import UnifiedLLM, create_llm
    print("  ✓ LLM provider OK")
except Exception as e:
    print(f"  ✗ LLM provider FAILED: {e}")

try:
    print("\n✓ Checking search workflow...")
    from src.workflow.search import (
        SearchService,
        create_search_service,
        classify_query,
        QueryClassification,
        research_agent,
        writer_agent,
        CitedAnswer,
        ActionRegistry
    )
    print("  ✓ Search workflow OK")
except Exception as e:
    print(f"  ✗ Search workflow FAILED: {e}")

try:
    print("\n✓ Checking research workflow...")
    from src.workflow.research import (
        create_research_graph,
        ResearchState,
        create_initial_state,
        SupervisorQueue,
        get_supervisor_queue,
        run_researcher_agent
    )
    print("  ✓ Research workflow OK")
except Exception as e:
    print(f"  ✗ Research workflow FAILED: {e}")

try:
    print("\n✓ Checking streaming...")
    from src.streaming.sse import StreamEventType, ResearchStreamingGenerator
    print("  ✓ Streaming OK")
except Exception as e:
    print(f"  ✗ Streaming FAILED: {e}")

try:
    print("\n✓ Checking settings...")
    from src.config.settings import Settings
    print("  ✓ Settings OK")
except Exception as e:
    print(f"  ✗ Settings FAILED: {e}")

print("\n" + "="*50)
print("Import check complete!")
print("="*50)

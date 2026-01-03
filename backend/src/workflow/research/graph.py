"""LangGraph workflow definition for deep research.

Defines the state machine for multi-agent research orchestration.
"""

from typing import Any

import structlog
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.workflow.research.state import ResearchState, create_initial_state
from src.workflow.research.nodes import (
    search_memory_node,
    run_deep_search_node,
    clarify_with_user_node,
    analyze_query_node,
    plan_research_enhanced_node,
    create_agent_characteristics_enhanced_node,
    execute_agents_enhanced_node,
    supervisor_review_enhanced_node,
    compress_findings_node,
    generate_final_report_enhanced_node,
)

logger = structlog.get_logger(__name__)


# Fields that should not be serialized (runtime dependencies)
NON_SERIALIZABLE_FIELDS = {
    "stream",
    "llm",
    "search_provider",
    "scraper",
    "supervisor_queue",
    "settings",
}


class FilteredMemorySaver(MemorySaver):
    """MemorySaver that excludes non-serializable fields from state before checkpointing.
    
    Note: LangGraph serializes state BEFORE calling put(), so we need to intercept
    at the serialization level. However, since we can't easily override the serializer,
    we'll use a different approach: filter state in the graph execution wrapper.
    """
    
    def put(self, config, checkpoint, metadata, new_versions):
        """Override put to filter out non-serializable fields."""
        # Filter checkpoint if it's a dict
        if isinstance(checkpoint, dict):
            filtered_checkpoint = {
                k: v for k, v in checkpoint.items() 
                if k not in NON_SERIALIZABLE_FIELDS
            }
        else:
            filtered_checkpoint = checkpoint
        
        return super().put(config, filtered_checkpoint, metadata, new_versions)
    
    def get_tuple(self, config):
        """Override get_tuple to restore filtered fields from config if needed."""
        result = super().get_tuple(config)
        if result is None:
            return None
        
        checkpoint, metadata, parent_config = result
        # Runtime deps are not in checkpoint, they're passed via config or restored elsewhere
        return checkpoint, metadata, parent_config


def should_continue_research(state: ResearchState) -> str:
    """Conditional routing from supervisor."""
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 25)
    should_continue = state.get("should_continue", True)
    replanning_needed = state.get("replanning_needed", False)

    # Stop if max iterations reached
    if iteration >= max_iterations:
        logger.info("Max iterations reached, compressing")
        return "compress"

    # Replan if gaps identified
    if replanning_needed:
        logger.info("Replanning needed")
        return "replan"

    # Continue if supervisor says so
    if should_continue:
        logger.info("Continuing research")
        return "continue"

    # Otherwise compress and finish
    logger.info("Research complete, compressing")
    return "compress"


def should_ask_clarification(state: ResearchState) -> str:
    """Conditional routing after clarification check."""
    clarification_needed = state.get("clarification_needed", False)
    
    if clarification_needed:
        logger.info("Clarification needed, pausing for user input")
        return "wait_for_user"
    else:
        logger.info("No clarification needed, proceeding with research")
        return "proceed"


def create_research_graph(checkpoint_path: str = "./research_checkpoints.db") -> StateGraph:
    """
    Create LangGraph state machine for deep research.

    Args:
        checkpoint_path: Path to SQLite checkpoint database (unused - using MemorySaver for now)

    Returns:
        Compiled StateGraph
    """
    logger.info("Creating research graph")

    # Initialize graph
    workflow = StateGraph(ResearchState)

    # Add nodes
    workflow.add_node("search_memory", search_memory_node)
    workflow.add_node("run_deep_search", run_deep_search_node)
    workflow.add_node("clarify", clarify_with_user_node)
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("plan_research", plan_research_enhanced_node)
    workflow.add_node("spawn_agents", create_agent_characteristics_enhanced_node)
    workflow.add_node("execute_agents", execute_agents_enhanced_node)
    workflow.add_node("supervisor_react", supervisor_review_enhanced_node)
    workflow.add_node("compress_findings", compress_findings_node)
    workflow.add_node("generate_report", generate_final_report_enhanced_node)

    # Define edges
    workflow.set_entry_point("search_memory")

    workflow.add_edge("search_memory", "run_deep_search")
    workflow.add_edge("run_deep_search", "clarify")
    
    # Conditional edge for clarification
    # For now, always proceed (interactive clarification would require different architecture)
    # To implement interactive: would need to pause graph, emit questions, wait for user response
    workflow.add_conditional_edges(
        "clarify",
        should_ask_clarification,
        {
            "wait_for_user": "analyze_query",  # Would pause here in interactive mode
            "proceed": "analyze_query"
        }
    )
    workflow.add_edge("analyze_query", "plan_research")
    workflow.add_edge("plan_research", "spawn_agents")
    workflow.add_edge("spawn_agents", "execute_agents")
    workflow.add_edge("execute_agents", "supervisor_react")

    # Conditional routing from supervisor
    workflow.add_conditional_edges(
        "supervisor_react",
        should_continue_research,
        {
            "continue": "execute_agents",  # More agent work
            "replan": "plan_research",  # New topics
            "compress": "compress_findings",  # Finish
        }
    )

    workflow.add_edge("compress_findings", "generate_report")
    workflow.add_edge("generate_report", END)

    # Compile with in-memory checkpointing (session data persists in SQLite via session_memory_service)
    # Note: stream, llm, search_provider, scraper, supervisor_queue, settings are runtime dependencies
    # and should not be serialized. They are excluded from state before checkpointing.
    checkpointer = FilteredMemorySaver()

    compiled_graph = workflow.compile(checkpointer=checkpointer)

    logger.info("Research graph created and compiled")

    return compiled_graph


# ==================== Graph Execution ==========


async def run_research_graph(
    query: str,
    chat_history: list,
    mode: str,
    llm: Any,
    search_provider: Any,
    scraper: Any,
    stream: Any,
    session_id: str,
    mode_config: dict,
    settings: Any = None,
) -> dict:
    """
    Execute research graph.

    Args:
        query: Research query
        chat_history: Chat history
        mode: Research mode (speed/balanced/quality)
        llm: LLM instance
        search_provider: Search provider
        scraper: Web scraper
        stream: Stream generator
        session_id: Session ID
        mode_config: Mode configuration
        settings: Application settings

    Returns:
        Final state dict
    """
    logger.info("Starting research graph execution", query=query[:100], mode=mode)

    # Create graph
    graph = create_research_graph()

    # Create initial state
    initial_state = create_initial_state(
        query=query,
        chat_history=chat_history,
        mode=mode,
        stream=stream,
        session_id=session_id,
        mode_config=mode_config,
        settings=settings,
    )

    # Store runtime dependencies separately (they can't be serialized)
    # Try to get memory services from stream.app_state if available
    stream_obj = initial_state.get("stream")
    agent_memory_service = None
    agent_file_service = None
    if stream_obj and hasattr(stream_obj, "app_state"):
        app_state = stream_obj.app_state
        if isinstance(app_state, dict):
            agent_memory_service = app_state.get("agent_memory_service") or app_state.get("_agent_memory_service")
            agent_file_service = app_state.get("agent_file_service") or app_state.get("_agent_file_service")
        else:
            agent_memory_service = getattr(app_state, "agent_memory_service", None) or getattr(app_state, "_agent_memory_service", None)
            agent_file_service = getattr(app_state, "agent_file_service", None) or getattr(app_state, "_agent_file_service", None)
    
    runtime_deps = {
        "stream": stream_obj,
        "llm": llm,
        "search_provider": search_provider,
        "scraper": scraper,
        "supervisor_queue": initial_state.get("supervisor_queue"),
        "settings": initial_state.get("settings"),
        "agent_memory_service": agent_memory_service,
        "agent_file_service": agent_file_service,
    }
    
    # Remove non-serializable fields from state before passing to graph
    # They will be restored in nodes via contextvars or passed through config
    filtered_state = {k: v for k, v in initial_state.items() if k not in NON_SERIALIZABLE_FIELDS}
    
    # Store runtime deps in a context variable or pass through config
    # Use the same contextvar from nodes.py to ensure consistency
    from src.workflow.research.nodes import runtime_deps_context
    runtime_deps_context.set(runtime_deps)

    try:
        # Run graph with filtered state (no non-serializable fields)
        final_state = await graph.ainvoke(
            filtered_state,
            config={"configurable": {"thread_id": session_id}}
        )
        
        # Restore runtime deps to final_state for return
        for key, value in runtime_deps.items():
            if value is not None:
                final_state[key] = value

        logger.info("Research graph completed successfully")

        return final_state

    except Exception as e:
        logger.error("Research graph execution failed", error=str(e), exc_info=True)
        raise

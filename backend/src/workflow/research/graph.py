"""LangGraph workflow definition for deep research.

Defines the state machine for multi-agent research orchestration.
"""

from typing import Any

import structlog
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.workflow.research.state import ResearchState, create_initial_state
from src.workflow.research.nodes import (
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
    chat_history = state.get("chat_history", [])
    
    if clarification_needed:
        # Check if user has answered the clarification questions
        # Look for assistant message with clarification, then check if there's a user response after it
        has_user_answer = False
        found_clarification = False
        
        for i, msg in enumerate(chat_history):
            if msg.get("role") == "assistant":
                content = msg.get("content", "").lower()
                if "clarification" in content or "🔍" in content or "clarify" in content:
                    found_clarification = True
                    # Check if next message is from user (user answered)
                    if i + 1 < len(chat_history) and chat_history[i + 1].get("role") == "user":
                        has_user_answer = True
                        logger.info("User has answered clarification questions", answer_preview=chat_history[i + 1].get("content", "")[:100])
                        break
        
        if found_clarification and not has_user_answer:
            # Questions were sent but user hasn't answered yet - STOP and wait
            logger.info("Clarification questions sent, waiting for user answer - INTERRUPTING GRAPH")
            # Return special value that will trigger interrupt
            # We'll use interrupt_before on analyze_query node
            return "wait_for_user"
        elif has_user_answer:
            # User answered, proceed with research
            logger.info("User answered clarification, proceeding with research")
            return "proceed"
        else:
            # Questions were just sent, wait for answer
            logger.info("Clarification needed, waiting for user input")
            return "wait_for_user"
    else:
        logger.info("No clarification needed, proceeding with research")
        return "proceed"


# Global checkpointer shared across all graph instances
# This ensures checkpoints persist between graph invocations
_global_checkpointer = None

def get_global_checkpointer():
    """Get or create global checkpointer for graph state persistence."""
    global _global_checkpointer
    if _global_checkpointer is None:
        _global_checkpointer = FilteredMemorySaver()
        logger.info("Created global checkpointer for graph state persistence")
    return _global_checkpointer

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
    # Note: search_memory_node removed - agent memory is created empty and populated during research
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
    workflow.set_entry_point("run_deep_search")
    workflow.add_edge("run_deep_search", "clarify")
    
    # Conditional edge for clarification
    # If user hasn't answered, stop graph (return END)
    # When user answers and graph is resumed, it will proceed to analyze_query
    workflow.add_conditional_edges(
        "clarify",
        should_ask_clarification,
        {
            "wait_for_user": END,  # Stop graph and wait for user input - graph will be resumed when user answers
            "proceed": "analyze_query"  # User answered, continue with research
        }
    )
    # Check if analyze_query stopped waiting for user
    def should_continue_after_analysis(state: ResearchState) -> str:
        """Check if we should continue after analysis or stop waiting for user."""
        if state.get("clarification_waiting", False) or state.get("should_stop", False):
            logger.info("Stopping graph - waiting for user clarification answer")
            return "stop"
        return "continue"
    
    workflow.add_conditional_edges(
        "analyze_query",
        should_continue_after_analysis,
        {
            "stop": END,  # Stop and wait for user
            "continue": "plan_research"  # Continue with research
        }
    )
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

    # Compile with global checkpointer to ensure state persists between invocations
    # Note: stream, llm, search_provider, scraper, supervisor_queue, settings are runtime dependencies
    # and should not be serialized. They are excluded from state before checkpointing.
    checkpointer = get_global_checkpointer()

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
    # Create graph
    graph = create_research_graph()

    # CRITICAL: Check if this is a continuation (user answered clarification)
    # If yes, check if checkpoint exists and use it instead of creating new state
    is_continuation = False
    for i in range(len(chat_history) - 1, -1, -1):
        msg = chat_history[i]
        if msg.get("role") == "assistant":
            content = msg.get("content", "").lower()
            if "clarification" in content or "🔍" in content or "clarify" in content:
                # Check if there's a user message after this
                if i + 1 < len(chat_history) and chat_history[i + 1].get("role") == "user":
                    is_continuation = True
                    logger.info("Detected continuation - user answered clarification, will resume from checkpoint")
                    break
    
    logger.info("Starting research graph execution", 
               query=query[:100] if query else None, 
               mode=mode,
               query_length=len(query) if query else 0,
               is_continuation=is_continuation)
    
    # Create initial state first
    initial_state = create_initial_state(
        query=query,
        chat_history=chat_history,
        mode=mode,
        stream=stream,
        session_id=session_id,
        mode_config=mode_config,
        settings=settings,
    )
    
    # CRITICAL: Set runtime dependencies in context variable so nodes can restore them
    from src.workflow.research.nodes import runtime_deps_context
    runtime_deps_context.set({
        "stream": stream,
        "llm": llm,
        "search_provider": search_provider,
        "scraper": scraper,
        "settings": settings,
    })
    logger.info("Runtime dependencies set in context", has_stream=stream is not None, has_llm=llm is not None)

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
    
    # CRITICAL: If continuation, try to get checkpoint state and merge it
    # LangGraph automatically resumes from checkpoint, but we need to ensure state is correct
    if is_continuation:
        try:
            # Get checkpoint state using graph's checkpointer
            checkpointer = graph.checkpointer if hasattr(graph, 'checkpointer') else None
            if checkpointer:
                config = {"configurable": {"thread_id": session_id}}
                checkpoint_tuple = checkpointer.get_tuple(config)
                if checkpoint_tuple:
                    checkpoint_state, metadata, parent_config = checkpoint_tuple
                    if checkpoint_state and isinstance(checkpoint_state, dict):
                        logger.info("Found checkpoint state for continuation", 
                                   state_keys=list(checkpoint_state.keys()),
                                   deep_search_result_exists="deep_search_result" in checkpoint_state)
                        # Merge checkpoint state into initial_state (checkpoint takes precedence)
                        # But update chat_history and query with latest
                        for key, value in checkpoint_state.items():
                            if key not in NON_SERIALIZABLE_FIELDS and key not in ["stream", "llm", "search_provider", "scraper", "settings"]:
                                initial_state[key] = value
                        # Always update chat_history and query with latest
                        # CRITICAL: Preserve original query from initial request, not from checkpoint!
                        initial_state["chat_history"] = chat_history
                        initial_state["query"] = query
                        logger.info("Checkpoint state merged", 
                                   deep_search_result_exists="deep_search_result" in initial_state,
                                   clarification_needed=initial_state.get("clarification_needed", False),
                                   query=query[:100] if query else None,
                                   checkpoint_query=checkpoint_state.get("query", "")[:100] if checkpoint_state.get("query") else None)
        except Exception as e:
            logger.warning("Failed to get checkpoint state, will use initial state", error=str(e), exc_info=True)
    
    # Remove non-serializable fields from state before passing to graph
    # They will be restored in nodes via contextvars or passed through config
    filtered_state = {k: v for k, v in initial_state.items() if k not in NON_SERIALIZABLE_FIELDS}
    
    # Store runtime deps in a context variable or pass through config
    # Use the same contextvar from nodes.py to ensure consistency
    from src.workflow.research.nodes import runtime_deps_context
    runtime_deps_context.set(runtime_deps)

    try:
        # Run graph with filtered state (no non-serializable fields)
        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": 100  # Increased from default 25 to handle complex workflows
        }
        
        # CRITICAL: For continuation, only pass updated fields (chat_history, query)
        # LangGraph will automatically load checkpoint and apply our updates
        # This ensures graph continues from where it stopped (after clarify node), not from entry point
        if is_continuation:
            logger.info("Continuation detected - passing only updated fields to resume from checkpoint",
                       has_deep_search_result="deep_search_result" in filtered_state,
                       deep_search_result_type=type(filtered_state.get("deep_search_result")).__name__ if "deep_search_result" in filtered_state else "none")
            # Only update chat_history and query - LangGraph will load the rest from checkpoint
            # BUT: CRITICAL - Preserve deep_search_result if it exists in filtered_state (from checkpoint merge)
            # CRITICAL: Always use the original query from the request, not from checkpoint!
            logger.info("Setting query for continuation", 
                       query=query[:100] if query else None,
                       query_source="original_request")
            update_state = {
                "chat_history": chat_history,
                "query": query,  # CRITICAL: This is the ORIGINAL query, not clarification answer!
            }
            # CRITICAL: If deep_search_result exists in filtered_state (from checkpoint), preserve it
            # This ensures deep search is not re-run when continuing after clarification
            if "deep_search_result" in filtered_state:
                update_state["deep_search_result"] = filtered_state["deep_search_result"]
                logger.info("CRITICAL: Preserving deep_search_result from checkpoint for continuation",
                           result_type=type(filtered_state["deep_search_result"]).__name__,
                           is_dict=isinstance(filtered_state["deep_search_result"], dict))
            # Remove non-serializable fields
            update_state = {k: v for k, v in update_state.items() if k not in NON_SERIALIZABLE_FIELDS}
            logger.info("Invoking graph with update state for continuation", 
                       update_keys=list(update_state.keys()),
                       has_checkpoint=True,
                       has_deep_search_result="deep_search_result" in update_state)
            final_state = await graph.ainvoke(update_state, config=config)
        else:
            # No checkpoint - start fresh with full state
            logger.info("No continuation - starting fresh with full state")
            final_state = await graph.ainvoke(filtered_state, config=config)
        
        # Restore runtime deps to final_state for return
        for key, value in runtime_deps.items():
            if value is not None:
                final_state[key] = value

        logger.info("Research graph completed successfully")

        return final_state

    except Exception as e:
        logger.error("Research graph execution failed", error=str(e), exc_info=True)
        raise

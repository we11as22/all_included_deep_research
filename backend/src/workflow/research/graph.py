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

        # LangGraph may return 3 or 4 values depending on version
        # Return result as-is since we don't modify it
        return result


def should_continue_research(state: ResearchState) -> str:
    """Conditional routing from supervisor.
    
    CRITICAL: This function MUST eventually return "compress" to ensure report generation.
    Multiple safety checks prevent infinite loops.
    """
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 25)
    should_continue = state.get("should_continue", True)
    replanning_needed = state.get("replanning_needed", False)
    supervisor_call_count = state.get("supervisor_call_count", 0)
    
    # Get max_supervisor_calls from settings
    try:
        from src.config.settings import get_settings
        settings = get_settings()
        max_supervisor_calls = settings.deep_research_max_supervisor_calls
    except:
        max_supervisor_calls = 12  # Default fallback

    # CRITICAL SAFETY CHECK 1: Stop if max iterations reached (hard limit)
    if iteration >= max_iterations:
        logger.warning(f"MANDATORY: Max iterations reached ({iteration}/{max_iterations}) - forcing compress to generate report")
        return "compress"

    # CRITICAL SAFETY CHECK 2: Stop if supervisor call limit reached (hard limit)
    if supervisor_call_count >= max_supervisor_calls:
        logger.warning(f"MANDATORY: Supervisor call limit reached ({supervisor_call_count}/{max_supervisor_calls}) - forcing compress to generate report")
        return "compress"

    # Replan if gaps identified (but only if limits not reached)
    if replanning_needed:
        logger.info("Replanning needed")
        return "replan"

    # Continue if supervisor says so (but only if limits not reached)
    if should_continue:
        logger.info("Continuing research")
        return "continue"

    # Otherwise compress and finish (supervisor said stop or all tasks done)
    logger.info("Research complete, compressing")
    return "compress"


def should_ask_clarification(state: ResearchState) -> str:
    """Conditional routing after clarification check based on session status."""
    clarification_needed = state.get("clarification_needed", False)
    session_status = state.get("session_status", "active")
    clarification_just_sent = state.get("clarification_just_sent", False)
    chat_history = state.get("chat_history", [])

    if clarification_needed:
        # CRITICAL: If clarification was just sent in THIS iteration, always wait
        # This prevents false positive when original user message is still last in chat_history
        if clarification_just_sent:
            logger.info("Clarification just sent, waiting for user answers",
                       session_status=session_status)
            return "wait_for_user"

        # Use session_status to determine if user has answered (not text markers!)
        if session_status == "waiting_clarification":
            # Check if user has provided new message (last message should be from user)
            if chat_history and chat_history[-1].get("role") == "user":
                # User has answered clarification
                logger.info("User answered clarification (based on session_status and chat_history)",
                           session_status=session_status,
                           last_message_role=chat_history[-1].get("role"))
                return "proceed"
            else:
                # Still waiting for user answer
                logger.info("Waiting for user clarification answer (based on session_status)",
                           session_status=session_status)
                return "wait_for_user"
        else:
            # Session is not in waiting_clarification state, proceed
            logger.info("Session not waiting for clarification (based on session_status)",
                       session_status=session_status)
            return "proceed"
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

    # CRITICAL: Conditional edge after clarify to handle waiting
    # If clarification_just_sent=True, we need to interrupt and wait for user
    def should_wait_for_clarification(state: ResearchState) -> str:
        """Check if we should wait for user clarification."""
        clarification_needed = state.get("clarification_needed", False)
        clarification_just_sent = state.get("clarification_just_sent", False)
        session_status = state.get("session_status", "")

        # CRITICAL: If user answered clarification (session_status changed to "researching"), proceed!
        if session_status == "researching" and not clarification_needed:
            logger.info("✅ User answered clarification - proceeding to analyze_query", session_status=session_status)
            return "continue"
        
        if clarification_needed and clarification_just_sent:
            logger.info("🛑 Waiting for user clarification - interrupting graph")
            return "wait"
        else:
            logger.info("✅ Clarification answered or not needed - proceeding to analyze_query")
            return "continue"

    workflow.add_conditional_edges(
        "clarify",
        should_wait_for_clarification,
        {
            "wait": END,  # Stop and wait for user
            "continue": "analyze_query",  # Proceed with research
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

    # Compile with global checkpointer to ensure state persists between invocations
    # Note: stream, llm, search_provider, scraper, supervisor_queue, settings are runtime dependencies
    # and should not be serialized. They are excluded from state before checkpointing.
    checkpointer = get_global_checkpointer()

    # CRITICAL: Clarify node uses conditional edge to decide if waiting is needed
    # If clarification_just_sent=True, conditional edge routes to END (graph stops)
    # On resume after user answers, clarify re-executes, returns clarification_needed=False,
    # and conditional edge routes to "analyze_query" (research continues)

    compiled_graph = workflow.compile(checkpointer=checkpointer)

    logger.info("Research graph created and compiled with interrupt support")

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
    session_manager: Any = None,
    session_factory: Any = None,
) -> dict:
    """
    Execute research graph.

    Args:
        query: Research query (original query for new sessions, or current message for continuations)
        chat_history: Chat history
        mode: Research mode (speed/balanced/quality)
        llm: LLM instance
        search_provider: Search provider
        scraper: Web scraper
        stream: Stream generator
        session_id: Session ID
        mode_config: Mode configuration
        settings: Application settings
        session_manager: SessionManager for loading session data
        session_factory: AsyncSession factory for database access

    Returns:
        Final state dict
    """
    # Create graph
    graph = create_research_graph()

    # Determine if continuation based on session status (not text markers!)
    is_continuation = False
    if session_manager:
        try:
            session = await session_manager.get_session(session_id)
            if session:
                # Check if session is in a state that indicates continuation
                is_continuation = session.status in {"waiting_clarification", "researching", "active"}
                logger.info("Session status check for continuation",
                           session_id=session_id,
                           status=session.status,
                           is_continuation=is_continuation)
        except Exception as e:
            logger.warning("Failed to check session status", error=str(e))

    logger.info("Starting research graph execution",
               query=query[:100] if query else None,
               mode=mode,
               query_length=len(query) if query else 0,
               is_continuation=is_continuation)

    # Create initial state (loads original_query from session if session_manager provided)
    initial_state = await create_initial_state(
        query=query,
        chat_history=chat_history,
        mode=mode,
        stream=stream,
        session_id=session_id,
        mode_config=mode_config,
        settings=settings,
        session_manager=session_manager,
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

    # CRITICAL: Set runtime dependencies in context variable so nodes can restore them
    # MUST include agent_memory_service and agent_file_service for agents to work!
    from src.workflow.research.nodes import runtime_deps_context
    runtime_deps_context.set({
        "stream": stream,
        "llm": llm,
        "search_provider": search_provider,
        "scraper": scraper,
        "settings": settings,
        "session_manager": session_manager,
        "session_factory": session_factory,
        "agent_memory_service": agent_memory_service,
        "agent_file_service": agent_file_service,
    })
    logger.info("Runtime dependencies set in context",
               has_stream=stream is not None,
               has_llm=llm is not None,
               has_agent_memory=agent_memory_service is not None,
               has_agent_file=agent_file_service is not None)

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
                    # LangGraph may return 3 or 4 values (checkpoint, metadata, parent_config, pending_writes)
                    checkpoint_state = checkpoint_tuple[0] if len(checkpoint_tuple) > 0 else None
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

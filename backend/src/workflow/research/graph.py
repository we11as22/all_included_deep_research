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
    CRITICAL: Before returning "compress", verify that NO agents are still working!
    This prevents frontend from hanging when deep research "completes" but agents are still working.
    """
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 25)
    should_continue = state.get("should_continue", True)
    replanning_needed = state.get("replanning_needed", False)

    # CRITICAL: Check if agents are still working BEFORE forcing compress
    # This flag is set in execute_agents.py when agents have pending/in_progress tasks
    # If agents are working, we MUST continue
    # This prevents frontend from hanging when deep research "completes" but agents are still working
    agents_still_working = state.get("_agents_still_working", False)
    
    # CRITICAL SAFETY CHECK: Stop if max iterations reached (hard limit)
    # BUT ONLY if agents are not still working
    if iteration >= max_iterations:
        if agents_still_working:
            logger.warning(f"Max iterations reached ({iteration}/{max_iterations}) but agents still working - forcing continue",
                         note="CRITICAL: Cannot finalize while agents are working - frontend would hang. Research will continue until all agents finish.")
            return "continue"
        logger.warning(f"MANDATORY: Max iterations reached ({iteration}/{max_iterations}) - forcing compress to generate report")
        return "compress"

    # Replan if gaps identified (but only if limits not reached)
    # CRITICAL: Prevent infinite replanning loops
    # Track replan count to prevent excessive replanning
    replan_count = state.get("replan_count", 0)
    max_replans = 3  # Maximum replans allowed
    
    if replanning_needed:
        if replan_count >= max_replans:
            logger.warning(f"Replan limit reached ({replan_count}/{max_replans}) - forcing continue instead of replan to prevent infinite loop",
                         iteration=iteration,
                         note="Too many replans - continuing with current plan instead")
            # Reset replanning_needed to prevent loop
            state["replanning_needed"] = False
            return "continue"  # Force continue instead of replan
        logger.info("Replanning needed", replan_count=replan_count, max_replans=max_replans)
        # Increment replan count in state (will be updated in next iteration)
        return "replan"

    # Continue if supervisor says so (but only if limits not reached)
    if should_continue:
        logger.info("Continuing research")
        return "continue"

    # Otherwise compress and finish (supervisor said stop or all tasks done)
    logger.info("Research complete, compressing")
    return "compress"


def should_ask_clarification(state: ResearchState) -> str:
    """Conditional routing after clarification check based on session status.
    
    CRITICAL: Uses session_status and clarification_answers from session state,
    NOT chat_history. Session state is the source of truth.
    """
    clarification_needed = state.get("clarification_needed", False)
    session_status = state.get("session_status", "active")
    clarification_just_sent = state.get("clarification_just_sent", False)
    clarification_answers = state.get("clarification_answers", "")  # Loaded from session in create_initial_state

    if clarification_needed:
        # CRITICAL: If clarification was just sent in THIS iteration, always wait
        # This prevents false positive when original user message is still last in chat_history
        if clarification_just_sent:
            logger.info("Clarification just sent, waiting for user answers",
                       session_status=session_status)
            return "wait_for_user"

        # CRITICAL: Use session_status and clarification_answers from session state
        # Session state is the source of truth, not chat_history
        if session_status == "waiting_clarification":
            # Check if clarification_answers exists in session state
            if clarification_answers and clarification_answers.strip():
                # User has answered clarification (answers are in session state)
                logger.info("User answered clarification (based on session_status and clarification_answers from session)",
                           session_status=session_status,
                           has_clarification_answers=bool(clarification_answers),
                           note="Using clarification_answers from session state, not chat_history")
                return "proceed"
            else:
                # Still waiting for user answer (no answers in session state)
                logger.info("Waiting for user clarification answer (no clarification_answers in session state)",
                           session_status=session_status,
                           note="Checking clarification_answers from session state, not chat_history")
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

    # CRITICAL: Entry point is always run_deep_search
    # The deep_search node itself checks DB FIRST and returns immediately if result exists
    # This is the MOST RELIABLE approach - DB check happens INSIDE the node, not in routing
    workflow.set_entry_point("run_deep_search")
    
    # CRITICAL: Conditional routing after deep_search
    # If continuation after clarification, skip clarify and go directly to analyze_query
    def should_skip_clarify_after_deep_search(state: ResearchState) -> str:
        """Check if we should skip clarify node after deep_search.
        
        CRITICAL: This ensures proper sequential workflow:
        - New session: deep_search → clarify → analyze_query
        - Continuation after clarification: deep_search (returns existing result) → analyze_query (skip clarify)
        
        Returns:
            "skip_clarify" - if user already answered clarification, go directly to analyze_query
            "clarify" - normal flow, go to clarify node
        """
        session_id = state.get("session_id")
        session_status = state.get("session_status", "active")
        clarification_answers = state.get("clarification_answers", "")
        deep_search_result = state.get("deep_search_result", "")
        
        # Handle dict format
        if isinstance(deep_search_result, dict):
            deep_search_result = deep_search_result.get("value", "")
        
        logger.info("🔀 ROUTING: should_skip_clarify_after_deep_search called",
                   session_id=session_id,
                   session_status=session_status,
                   has_clarification_answers=bool(clarification_answers),
                   clarification_answers_length=len(clarification_answers) if clarification_answers else 0,
                   has_deep_search_result=bool(deep_search_result),
                   deep_search_result_length=len(deep_search_result) if deep_search_result else 0,
                   note="Checking routing after deep_search node")
        
        # CRITICAL: If user already answered clarification, skip clarify and go to analyze_query
        # This ensures proper sequential workflow without double deep_search
        if session_status == "researching":
            logger.warning("⏭️ ROUTING: SKIP CLARIFY - session_status is 'researching' (user answered clarification)",
                      session_id=session_id,
                      session_status=session_status,
                      has_clarification_answers=bool(clarification_answers),
                      note="Proceeding directly to analyze_query after deep_search (continuation)")
            return "skip_clarify"
        
        if clarification_answers and clarification_answers.strip():
            logger.warning("⏭️ ROUTING: SKIP CLARIFY - clarification_answers exists (user answered)",
                      session_id=session_id,
                      answers_length=len(clarification_answers),
                      note="Proceeding directly to analyze_query after deep_search (continuation)")
            return "skip_clarify"
        
        # Normal flow: deep_search → clarify
        logger.info("✅ ROUTING: GO TO CLARIFY - normal flow",
                   session_id=session_id,
                   session_status=session_status,
                   note="Proceeding to clarify after deep_search (new session)")
        return "clarify"
    
    # CRITICAL: Conditional routing after deep_search
    # If continuation after clarification, skip clarify and go directly to analyze_query
    workflow.add_conditional_edges(
        "run_deep_search",
        should_skip_clarify_after_deep_search,
        {
            "skip_clarify": "analyze_query",  # Skip clarify, go directly to analyze
            "clarify": "clarify",  # Normal flow: go to clarify
        }
    )

    # CRITICAL: Conditional edge after clarify to handle waiting
    # If clarification_just_sent=True, we need to interrupt and wait for user
    def should_wait_for_clarification(state: ResearchState) -> str:
        """Check if we should wait for user clarification.
        
        CRITICAL: Uses session_status and clarification_answers from session state,
        NOT chat_history. Session state is the source of truth.
        """
        clarification_needed = state.get("clarification_needed", False)
        clarification_just_sent = state.get("clarification_just_sent", False)
        session_status = state.get("session_status", "")
        clarification_answers = state.get("clarification_answers", "")  # Loaded from session in create_initial_state

        # CRITICAL: If user answered clarification (session_status changed to "researching"), proceed!
        if session_status == "researching" and not clarification_needed:
            logger.info("✅ User answered clarification - proceeding to analyze_query", 
                       session_status=session_status,
                       has_clarification_answers=bool(clarification_answers),
                       note="Using session_status and clarification_answers from session state")
            return "continue"
        
        # CRITICAL: Also check clarification_answers from session state
        # If answers exist, user has answered, proceed
        if clarification_answers and clarification_answers.strip() and not clarification_needed:
            logger.info("✅ User answered clarification (clarification_answers in session state) - proceeding to analyze_query",
                       session_status=session_status,
                       answers_length=len(clarification_answers),
                       note="Using clarification_answers from session state, not chat_history")
            return "continue"
        
        # CRITICAL: If clarification needed but not answered, wait
        # This prevents multi-agent system from starting before user answers
        if clarification_needed and clarification_just_sent:
            logger.info("🛑 Waiting for user clarification - interrupting graph BEFORE multi-agent system",
                       clarification_needed=clarification_needed,
                       clarification_just_sent=clarification_just_sent,
                       session_status=session_status,
                       note="CRITICAL: Multi-agent system (execute_agents) will NOT start until user answers")
            return "wait"
        
        # CRITICAL: Also check if clarification_needed but answers are missing
        # This is a safety check in case clarification_just_sent was not set correctly
        # This prevents multi-agent system from starting if clarification is needed but not answered
        if clarification_needed and session_status == "waiting_clarification" and not clarification_answers:
            logger.warning("🛑 Clarification needed but answers missing - waiting BEFORE multi-agent system",
                         clarification_needed=clarification_needed,
                         session_status=session_status,
                         has_clarification_answers=bool(clarification_answers),
                         note="CRITICAL: Multi-agent system (execute_agents) will NOT start - waiting for user answers")
            return "wait"
        
        # CRITICAL: Additional safety check - if clarification_needed is True but we're here,
        # it means something went wrong. Better to wait than start multi-agent system prematurely
        if clarification_needed and not clarification_answers:
            logger.error("🛑 CRITICAL: clarification_needed=True but no answers - waiting to prevent premature multi-agent start",
                       clarification_needed=clarification_needed,
                       session_status=session_status,
                       has_clarification_answers=bool(clarification_answers),
                       note="CRITICAL ERROR: Multi-agent system should NOT start - waiting for clarification answers")
            return "wait"
        
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
    
    # CRITICAL: Conditional routing after plan_research
    # If clarification needed but not answered, stop and wait
    def should_stop_after_planning(state: ResearchState) -> str:
        """Check if we should stop after planning (waiting for clarification).
        
        CRITICAL: This prevents creating research plan before user answers clarification questions.
        """
        planning_waiting = state.get("planning_waiting", False)
        should_stop = state.get("should_stop", False)
        
        if planning_waiting or should_stop:
            logger.warning("⏸️ Stopping after planning - waiting for clarification answers",
                         planning_waiting=planning_waiting,
                         should_stop=should_stop,
                         note="Research plan should NOT be created before user answers clarification")
            return "wait"
        
        return "continue"
    
    workflow.add_conditional_edges(
        "plan_research",
        should_stop_after_planning,
        {
            "wait": END,  # Stop and wait for user
            "continue": "spawn_agents",  # Proceed with agent creation
        }
    )
    workflow.add_edge("spawn_agents", "execute_agents")
    workflow.add_edge("execute_agents", "supervisor_react")

    # Conditional routing from supervisor
    # CRITICAL: Track replan count to prevent infinite loops
    def should_continue_research_with_replan_tracking(state: ResearchState) -> str:
        result = should_continue_research(state)
        # Increment replan_count if replanning
        if result == "replan":
            current_count = state.get("replan_count", 0)
            state["replan_count"] = current_count + 1
            logger.info(f"Incremented replan_count to {current_count + 1}")
        return result
    
    workflow.add_conditional_edges(
        "supervisor_react",
        should_continue_research_with_replan_tracking,
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
    logger.info("🚀 RUN_RESEARCH_GRAPH: Starting",
               session_id=session_id,
               query_preview=query[:100] if query else None,
               mode=mode,
               has_session_manager=bool(session_manager),
               note="Creating research graph and initial state")
    
    # Create graph
    graph = create_research_graph()
    
    logger.info("✅ RUN_RESEARCH_GRAPH: Graph created",
               session_id=session_id,
               note="Research graph compiled successfully")

    # CRITICAL: Determine if continuation based on ACTUAL session state, not just status
    # Continuation means: session already has work done (deep_search_result, clarification_answers, etc.)
    # NOT just status "active" - new sessions also have status "active"!
    is_continuation = False
    current_session_status = "active"
    has_deep_search_result = False
    has_clarification_answers = False
    
    if session_manager:
        try:
            session = await session_manager.get_session(session_id)
            if session:
                current_session_status = session.status
                has_deep_search_result = bool(session.deep_search_result)
                has_clarification_answers = bool(session.clarification_answers)
                
                # CRITICAL: Continuation is determined by ACTUAL work done, not just status
                # Continuation if:
                # 1. deep_search_result exists (deep search was already done)
                # 2. clarification_answers exists (user already answered)
                # 3. status is "waiting_clarification" (waiting for user)
                # 4. status is "researching" (research in progress)
                # BUT NOT "active" alone - new sessions also have "active" status!
                
                if has_deep_search_result:
                    is_continuation = True
                    logger.warning("🔄 CONTINUATION DETECTED: deep_search_result exists in DB",
                                 session_id=session_id,
                                 status=session.status,
                                 result_length=len(session.deep_search_result) if session.deep_search_result else 0,
                                 note="Deep search was already done - this is continuation")
                elif has_clarification_answers:
                    is_continuation = True
                    logger.warning("🔄 CONTINUATION DETECTED: clarification_answers exists in DB",
                                 session_id=session_id,
                                 status=session.status,
                                 note="User already answered clarification - this is continuation")
                elif session.status == "waiting_clarification":
                    is_continuation = True
                    logger.warning("🔄 CONTINUATION DETECTED: session_status is 'waiting_clarification'",
                                 session_id=session_id,
                                 note="Waiting for user clarification - this is continuation")
                elif session.status == "researching":
                    is_continuation = True
                    logger.warning("🔄 CONTINUATION DETECTED: session_status is 'researching'",
                                 session_id=session_id,
                                 note="Research in progress - this is continuation")
                else:
                    # New session - status is "active" but no work done yet
                    is_continuation = False
                    logger.info("🆕 NEW SESSION: No work done yet",
                               session_id=session_id,
                               status=session.status,
                               note="This is a new session - deep search will execute")
                
                logger.info("Session continuation check",
                           session_id=session_id,
                           status=session.status,
                           has_deep_search_result=has_deep_search_result,
                           has_clarification_answers=has_clarification_answers,
                           is_continuation=is_continuation)
        except Exception as e:
            logger.error("Failed to check session status for continuation",
                        session_id=session_id,
                        error=str(e),
                        exc_info=True)

    logger.info("Starting research graph execution",
               query=query[:100] if query else None,
               mode=mode,
               query_length=len(query) if query else 0,
               is_continuation=is_continuation,
               session_id=session_id)

    # CRITICAL: Validate session_id - it should never be None or empty
    if not session_id:
        logger.warning("session_id is None or empty - generating fallback ID", 
                      session_id=session_id,
                      mode=mode)
        from uuid import uuid4
        session_id = str(uuid4())
        logger.info("Generated fallback session_id", session_id=session_id)
    
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
    
    # Get research_memory_service from stream.app_state if available
    research_memory_service = None
    if stream and hasattr(stream, "app_state"):
        app_state = stream.app_state
        if isinstance(app_state, dict):
            research_memory_service = app_state.get("research_memory_service") or app_state.get("_research_memory_service")
        else:
            research_memory_service = getattr(app_state, "research_memory_service", None) or getattr(app_state, "_research_memory_service", None)
    
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
        "research_memory_service": research_memory_service,
    })
    logger.warning("🔍 CRITICAL: Runtime dependencies set in context",
                 has_stream=stream is not None,
                 has_llm=llm is not None,
                 has_agent_memory=agent_memory_service is not None,
                 has_agent_file=agent_file_service is not None,
                 has_session_manager=bool(session_manager),
                 session_manager_type=type(session_manager).__name__ if session_manager else "None",
                 session_id=session_id,
                 note="CRITICAL: session_manager MUST be set for deep_search to work correctly!")

    # CRITICAL: Create runtime_deps dict with ALL dependencies including session_manager
    # This will be used to restore context variable later (line 657)
    # session_manager and session_factory MUST be included here!
    runtime_deps = {
        "stream": stream_obj,
        "llm": llm,
        "search_provider": search_provider,
        "scraper": scraper,
        "supervisor_queue": initial_state.get("supervisor_queue"),
        "settings": initial_state.get("settings"),
        "agent_memory_service": agent_memory_service,
        "agent_file_service": agent_file_service,
        "research_memory_service": research_memory_service,
        "session_manager": session_manager,  # CRITICAL: Must be included!
        "session_factory": session_factory,  # CRITICAL: Must be included!
    }
    
    logger.warning("🔍 CRITICAL: runtime_deps dict created",
                 has_session_manager=bool(session_manager),
                 has_session_factory=bool(session_factory),
                 runtime_deps_keys=list(runtime_deps.keys()),
                 session_id=session_id,
                 note="CRITICAL: session_manager and session_factory MUST be in runtime_deps dict!")
    
    # CRITICAL: Update session_status from DB in initial_state (before checkpoint merge)
    # This ensures we have the latest status even if checkpoint has old status
    if session_manager and session_id:
        try:
            session = await session_manager.get_session(session_id)
            if session:
                initial_state["session_status"] = session.status
                logger.info("Updated session_status in initial_state from DB",
                           session_id=session_id,
                           status=session.status)
        except Exception as e:
            logger.warning("Failed to update session_status in initial_state", error=str(e))
    
    # CRITICAL: If continuation, try to get checkpoint state and merge it
    # LangGraph automatically resumes from checkpoint, but we need to ensure state is correct
    # BUT: Only use checkpoint if this is REAL continuation (has work done), not new session
    if is_continuation:
        logger.warning("🔄 CONTINUATION: Loading checkpoint state",
                     session_id=session_id,
                     has_deep_search_result=has_deep_search_result,
                     has_clarification_answers=has_clarification_answers,
                     current_session_status=current_session_status,
                     note="This is continuation - will load checkpoint if available")
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
                        # But update chat_history, query, session_status, and session_id with latest
                        # CRITICAL: session_id MUST be preserved from current request, not from checkpoint!
                        # CRITICAL: session_status MUST be preserved from DB, not from checkpoint!
                        for key, value in checkpoint_state.items():
                            if key not in NON_SERIALIZABLE_FIELDS and key not in ["stream", "llm", "search_provider", "scraper", "settings"]:
                                # CRITICAL: Skip session_id and session_status - they will be set explicitly below
                                if key not in ["session_id", "session_status"]:
                                    # CRITICAL: Don't overwrite deep_search_result from DB if it exists in initial_state
                                    # Checkpoint might not have deep_search_result, but DB does (from previous execution)
                                    if key == "deep_search_result":
                                        # Only use checkpoint value if initial_state doesn't have it from DB
                                        if not initial_state.get("deep_search_result") or not initial_state.get("deep_search_result", "").strip():
                                            initial_state[key] = value
                                            logger.info("Using deep_search_result from checkpoint (not in DB initial_state)",
                                                       checkpoint_has_result=bool(value),
                                                       initial_state_has_result=bool(initial_state.get("deep_search_result")))
                                        else:
                                            logger.info("Preserving deep_search_result from DB initial_state (not overwriting with checkpoint)",
                                                       db_result_length=len(initial_state.get("deep_search_result", "")),
                                                       checkpoint_result_length=len(value) if value else 0)
                                    else:
                                        initial_state[key] = value
                        # Always update chat_history, query, session_status, and session_id with latest
                        # CRITICAL: Preserve original query from initial request, not from checkpoint!
                        # CRITICAL: Preserve session_status from DB, not from checkpoint!
                        # CRITICAL: Always preserve session_id from current request, not from checkpoint!
                        initial_state["chat_history"] = chat_history
                        initial_state["query"] = query
                        initial_state["session_id"] = session_id  # CRITICAL: Always use current session_id!
                        # session_status already updated from DB above, don't overwrite with checkpoint
                        logger.info("Session ID preserved from current request",
                                   session_id=session_id,
                                   session_id_in_checkpoint="session_id" in checkpoint_state,
                                   checkpoint_session_id=checkpoint_state.get("session_id") if "session_id" in checkpoint_state else None)
                        logger.info("Checkpoint state merged", 
                                   deep_search_result_exists="deep_search_result" in initial_state,
                                   clarification_needed=initial_state.get("clarification_needed", False),
                                   session_status=initial_state.get("session_status"),
                                   session_id=initial_state.get("session_id"),
                                   session_id_in_checkpoint="session_id" in checkpoint_state,
                                   query=query[:100] if query else None,
                                   checkpoint_query=checkpoint_state.get("query", "")[:100] if checkpoint_state.get("query") else None)
        except Exception as e:
            logger.warning("Failed to get checkpoint state, will use initial state", error=str(e), exc_info=True)
    
    # Remove non-serializable fields from state before passing to graph
    # They will be restored in nodes via contextvars or passed through config
    # CRITICAL: session_id MUST be included in filtered_state - it's serializable and needed by nodes!
    filtered_state = {k: v for k, v in initial_state.items() if k not in NON_SERIALIZABLE_FIELDS}
    
    # CRITICAL: Ensure session_id is always in filtered_state (it may have been lost during checkpoint merge)
    if "session_id" not in filtered_state and session_id:
        filtered_state["session_id"] = session_id
        logger.warning("session_id was missing from filtered_state - restored it",
                      session_id=session_id,
                      filtered_state_keys=list(filtered_state.keys())[:10])
    
    # CRITICAL: Store runtime deps in a context variable AND in config
    # LangGraph may lose contextvars in async execution, so we pass session_manager through config as backup
    from src.workflow.research.nodes import runtime_deps_context
    
    # CRITICAL: Verify session_manager is in runtime_deps before setting context
    if "session_manager" not in runtime_deps:
        logger.error("❌ CRITICAL: session_manager missing from runtime_deps! Adding it now.",
                    runtime_deps_keys=list(runtime_deps.keys()),
                    session_id=session_id,
                    note="CRITICAL ERROR: This should never happen - session_manager must be in runtime_deps!")
        runtime_deps["session_manager"] = session_manager
        runtime_deps["session_factory"] = session_factory
    
    logger.warning("🔍 CRITICAL: Setting runtime_deps_context (final check)",
                 has_session_manager="session_manager" in runtime_deps,
                 has_session_factory="session_factory" in runtime_deps,
                 runtime_deps_keys=list(runtime_deps.keys()),
                 session_id=session_id,
                 note="CRITICAL: Verifying session_manager is in runtime_deps before setting context!")
    
    runtime_deps_context.set(runtime_deps)

    try:
        # Run graph with filtered state (no non-serializable fields)
        # CRITICAL: Pass session_manager through config so nodes can access it even if contextvar is lost
        config = {
            "configurable": {
                "thread_id": session_id,
                "session_manager": session_manager,  # CRITICAL: Pass directly through config
                "session_factory": session_factory,  # Also pass session_factory for creating new SessionManager if needed
            },
            "recursion_limit": 100  # Increased from default 25 to handle complex workflows
        }
        
        # CRITICAL: For continuation, only pass updated fields (chat_history, query)
        # LangGraph will automatically load checkpoint and apply our updates
        # This ensures graph continues from where it stopped (after clarify node), not from entry point
        # BUT: Only if this is REAL continuation (has work done), not new session
        if is_continuation:
            logger.warning("🔄 CONTINUATION: Using update_state (not full state)",
                         session_id=session_id,
                         has_deep_search_result=has_deep_search_result,
                         has_clarification_answers=has_clarification_answers,
                         current_session_status=current_session_status,
                         note="This is continuation - will update checkpoint with new data")
            logger.info("Continuation detected - passing only updated fields to resume from checkpoint",
                       has_deep_search_result="deep_search_result" in filtered_state,
                       deep_search_result_type=type(filtered_state.get("deep_search_result")).__name__ if "deep_search_result" in filtered_state else "none",
                       current_session_status=current_session_status)
            # Only update chat_history and query - LangGraph will load the rest from checkpoint
            # BUT: CRITICAL - Preserve deep_search_result if it exists in filtered_state (from checkpoint merge)
            # CRITICAL: Always use the original query from the request, not from checkpoint!
            # CRITICAL: Update session_status from DB to reflect current state (user may have answered clarification)
            # CRITICAL: Get original_query from initial_state (loaded from DB session)
            # query parameter might be clarification answer, but original_query is the actual research topic
            original_query_from_state = initial_state.get("original_query", query)
            logger.info("Setting query for continuation", 
                       query=query[:100] if query else None,
                       original_query=original_query_from_state[:100] if original_query_from_state else None,
                       query_source="original_request",
                       note="Using original_query from initial_state (DB), not query parameter (might be clarification answer)")
            update_state = {
                "chat_history": chat_history,
                "query": query,  # Current query (might be clarification answer for continuation)
                "original_query": original_query_from_state,  # CRITICAL: Original research topic from DB session
                "session_status": current_session_status,  # CRITICAL: Update from DB, not checkpoint!
                "session_id": session_id,  # CRITICAL: Always include session_id so nodes can use it!
            }
            
            # CRITICAL: Check if user answered clarification
            # Priority: session_status from DB > clarification_answers in state
            # If clarification_answers exists but status is still "waiting_clarification", 
            # it means answers were just saved - update status and flags
            clarification_answers_in_state = initial_state.get("clarification_answers", "") or filtered_state.get("clarification_answers", "")
            has_clarification_answers = bool(clarification_answers_in_state and clarification_answers_in_state.strip())
            
            if current_session_status == "researching":
                # Status is already "researching" - user answered
                update_state["clarification_needed"] = False
                update_state["clarification_just_sent"] = False
                logger.info("✅ Session status is 'researching' - user answered clarification, updating flags",
                           session_id=session_id,
                           has_clarification_answers=has_clarification_answers)
            elif has_clarification_answers and current_session_status == "waiting_clarification":
                # CRITICAL: Answers exist but status is still "waiting_clarification"
                # This means answers were just saved - update status to "researching"
                # This handles race condition where answers were saved but status wasn't updated yet
                if session_manager:
                    try:
                        await session_manager.update_status(session_id, "researching")
                        current_session_status = "researching"
                        update_state["session_status"] = "researching"
                        logger.info("✅ Updated session status to 'researching' (answers exist but status was 'waiting_clarification')",
                                   session_id=session_id,
                                   answers_length=len(clarification_answers_in_state),
                                   note="Answers were saved but status wasn't updated - fixing now")
                    except Exception as e:
                        logger.warning("Failed to update session status", error=str(e), exc_info=True)
                update_state["clarification_needed"] = False
                update_state["clarification_just_sent"] = False
                logger.info("✅ User answered clarification (answers in state, status updated) - proceeding",
                           session_id=session_id,
                           answers_length=len(clarification_answers_in_state),
                           note="Answers exist in state - proceeding with research")
            elif current_session_status == "waiting_clarification":
                # Still waiting for user answer (no answers in state)
                update_state["clarification_needed"] = True
                update_state["clarification_just_sent"] = True
                logger.info("⏸️ Session status is 'waiting_clarification' - still waiting for user answer",
                           session_id=session_id,
                           has_clarification_answers=has_clarification_answers,
                           note="No answers in state - waiting for user response")
            
            # CRITICAL: Preserve deep_search_result from initial_state (loaded from DB) OR from filtered_state (checkpoint)
            # Priority: initial_state (from DB) > filtered_state (checkpoint) > empty
            # This ensures deep search is not re-run when continuing after clarification
            if "deep_search_result" in initial_state and initial_state.get("deep_search_result"):
                # Use deep_search_result from DB (initial_state) - highest priority
                update_state["deep_search_result"] = initial_state["deep_search_result"]
                logger.info("CRITICAL: Preserving deep_search_result from DB (initial_state) for continuation",
                           result_type=type(initial_state["deep_search_result"]).__name__,
                           is_dict=isinstance(initial_state["deep_search_result"], dict),
                           result_length=len(str(initial_state["deep_search_result"])) if initial_state.get("deep_search_result") else 0,
                           note="Using deep_search_result from DB, not checkpoint")
            elif "deep_search_result" in filtered_state:
                # Fallback to checkpoint if not in initial_state
                update_state["deep_search_result"] = filtered_state["deep_search_result"]
                logger.info("CRITICAL: Preserving deep_search_result from checkpoint for continuation",
                           result_type=type(filtered_state["deep_search_result"]).__name__,
                           is_dict=isinstance(filtered_state["deep_search_result"], dict),
                           note="Using deep_search_result from checkpoint (not in DB initial_state)")
            else:
                logger.warning("CRITICAL: deep_search_result not found in initial_state or filtered_state",
                             has_in_initial="deep_search_result" in initial_state,
                             has_in_filtered="deep_search_result" in filtered_state,
                             note="Deep search may execute again - this should not happen!")
            
            # CRITICAL: Preserve clarification_answers from initial_state (loaded from DB session)
            # This is the source of truth for clarification answers, not chat_history
            if "clarification_answers" in initial_state and initial_state.get("clarification_answers"):
                update_state["clarification_answers"] = initial_state["clarification_answers"]
                logger.info("CRITICAL: Preserving clarification_answers from DB (initial_state) for continuation",
                           answers_length=len(initial_state["clarification_answers"]) if initial_state.get("clarification_answers") else 0,
                           note="Using clarification_answers from DB session, not chat_history")
            elif "clarification_answers" in filtered_state:
                # Fallback to checkpoint if not in initial_state
                update_state["clarification_answers"] = filtered_state["clarification_answers"]
                logger.info("CRITICAL: Preserving clarification_answers from checkpoint for continuation",
                           note="Using clarification_answers from checkpoint (not in DB initial_state)")
            else:
                # No clarification_answers - this is normal for new sessions or before clarification
                logger.debug("No clarification_answers in initial_state or filtered_state",
                           has_in_initial="clarification_answers" in initial_state,
                           has_in_filtered="clarification_answers" in filtered_state,
                           note="This is normal for new sessions or before user answers clarification")
            # Remove non-serializable fields
            update_state = {k: v for k, v in update_state.items() if k not in NON_SERIALIZABLE_FIELDS}
            logger.info("Invoking graph with update state for continuation", 
                       update_keys=list(update_state.keys()),
                       has_checkpoint=True,
                       has_deep_search_result="deep_search_result" in update_state,
                       session_status=update_state.get("session_status"),
                       session_id=update_state.get("session_id"),
                       clarification_needed=update_state.get("clarification_needed"))
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

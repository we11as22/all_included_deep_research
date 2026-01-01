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
    plan_research_node,
    spawn_agents_node,
    execute_agents_node,
    supervisor_react_node,
    compress_findings_node,
    generate_report_node,
)

logger = structlog.get_logger(__name__)


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

    # Import run_deep_search_node
    from src.workflow.research.nodes import run_deep_search_node
    workflow.add_node("run_deep_search", run_deep_search_node)

    workflow.add_node("plan_research", plan_research_node)
    workflow.add_node("spawn_agents", spawn_agents_node)
    workflow.add_node("execute_agents", execute_agents_node)
    workflow.add_node("supervisor_react", supervisor_react_node)
    workflow.add_node("compress_findings", compress_findings_node)
    workflow.add_node("generate_report", generate_report_node)

    # Define edges
    workflow.set_entry_point("search_memory")

    workflow.add_edge("search_memory", "run_deep_search")
    workflow.add_edge("run_deep_search", "plan_research")
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
    checkpointer = MemorySaver()

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

    # Add LLM and providers to state
    initial_state["llm"] = llm
    initial_state["search_provider"] = search_provider
    initial_state["scraper"] = scraper

    try:
        # Run graph
        final_state = await graph.ainvoke(initial_state)

        logger.info("Research graph completed successfully")

        return final_state

    except Exception as e:
        logger.error("Research graph execution failed", error=str(e), exc_info=True)
        raise

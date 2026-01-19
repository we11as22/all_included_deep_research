"""LangGraph state schema for deep research workflow.

Defines the state structure for the multi-agent research system.
"""

import operator
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field, ConfigDict
import structlog

logger = structlog.get_logger(__name__)


# ==================== State Schema ====================


class ResearchState(TypedDict):
    """State schema for research graph.

    Uses TypedDict for LangGraph compatibility with reducers.
    """

    # ========== Input ==========
    query: str
    original_query: str  # Original query from session (for deep_research continuations)
    chat_history: list  # List of message dicts
    mode: str  # speed, balanced, quality
    user_language: str  # User's language (detected from query, e.g., "Russian", "English")

    # ========== Analysis ==========
    query_analysis: dict[str, Any]  # QueryAnalysis structured output

    # ========== Planning ==========
    research_plan: dict[str, Any]  # Research plan metadata (reasoning, depth, strategy)
    research_topics: list[dict]  # List of research topics to investigate
    completed_topics: Annotated[list[str], operator.add]  # Finished topics

    # ========== Agent Execution ==========
    active_agents: dict[str, dict]  # agent_id -> {topic, status, findings}
    agent_findings: Annotated[list[dict], operator.add]  # All agent findings
    agent_todos: dict[str, list[dict]]  # agent_id -> todo list
    agent_notes: dict[str, list[dict]]  # agent_id -> notes list

    # ========== Supervisor State ==========
    supervisor_directives: Annotated[list[dict], operator.add]  # Directive queue
    replanning_needed: bool
    gaps_identified: list[str]

    # ========== Deep Search ==========
    deep_search_result: str  # Initial deep search answer
    clarification_answers: str  # User answers to clarification questions (loaded from session, source of truth)

    # ========== Memory ==========
    memory_context: list[dict]  # Memory search results
    main_file_content: str  # Main research file content
    shared_notes: Annotated[list[dict], operator.add]  # Cross-agent shared notes

    # ========== Agent Characteristics ==========
    agent_characteristics: dict[str, dict]  # agent_id -> {role, expertise, personality}

    # ========== Output ==========
    final_report: str
    confidence: str  # low, medium, high

    # ========== Settings ==========
    settings: Any  # Settings object

    # ========== Dependencies (not persisted in state, added at runtime) ==========
    llm: Any  # LLM instance
    search_provider: Any  # Search provider
    scraper: Any  # Web scraper
    supervisor_queue: Any  # Supervisor queue for agent coordination

    # ========== Control Flow ==========
    iteration: int
    max_iterations: int
    should_continue: bool
    estimated_agent_count: int  # Estimated number of agents from analysis
    agent_count: int  # Actual number of agents created
    requires_deep_search: bool  # Whether deep search is needed
    clarification_needed: bool  # Whether user clarification is needed
    clarification_just_sent: bool  # Flag to prevent immediate continuation after sending clarification
    findings: list[dict]  # Findings from execute_agents (temporary, before adding to agent_findings)
    findings_count: int  # Count of findings
    compressed_research: str  # Compressed findings before final report
    coordination_notes: str  # Notes on how agents should coordinate

    # ========== Streaming ==========
    stream: Any  # Streaming generator

    # ========== Session Info ==========
    session_id: str
    session_status: str  # Session status from DB (active, waiting_clarification, researching, completed, etc.)

    # ========== Mode Config ==========
    mode_config: dict[str, Any]  # max_concurrent, max_sources, etc.


# ==================== Pydantic Models for Structured Outputs ==========


class ResearchTopic(BaseModel):
    """Single research topic."""
    
    model_config = ConfigDict(
        # Force all fields to be in required array for Azure/OpenRouter compatibility
        json_schema_extra={
            "required": ["reasoning", "topic", "description", "priority"]
        }
    )

    reasoning: str = Field(description="Why this topic is important")
    topic: str = Field(description="Research topic title")
    description: str = Field(description="Detailed description of what to research")
    priority: str = Field(default="medium", description="Priority: low/medium/high")


class ResearchPlan(BaseModel):
    """Initial research plan from supervisor."""
    
    model_config = ConfigDict(
        # Force all fields to be in required array for Azure/OpenRouter compatibility
        json_schema_extra={
            "required": ["reasoning", "topics", "stop"]
        }
    )

    reasoning: str = Field(description="Overall research strategy")
    topics: list[ResearchTopic] = Field(description="List of research topics")
    stop: bool = Field(default=False, description="Whether planning is complete")


class SupervisorReActOutput(BaseModel):
    """Supervisor's reaction after agent actions."""
    
    model_config = ConfigDict(
        # Force all fields to be in required array for Azure/OpenRouter compatibility
        json_schema_extra={
            "required": ["reasoning", "should_continue", "replanning_needed", "directives", "new_topics", "gaps_identified"]
        }
    )

    reasoning: str = Field(description="Analysis of current research state")
    should_continue: bool = Field(description="Whether research should continue")
    replanning_needed: bool = Field(description="Whether new topics are needed")
    directives: list[dict] = Field(
        default_factory=list,
        description="Todo updates for agents: [{agent_id, action, content}]"
    )
    new_topics: list[str] = Field(
        default_factory=list,
        description="New research topics to explore"
    )
    gaps_identified: list[str] = Field(
        default_factory=list,
        description="Identified research gaps"
    )


class AgentFinding(BaseModel):
    """Finding from a single agent."""

    agent_id: str
    topic: str
    summary: str
    key_findings: list[str]
    sources: list[dict[str, str]]  # [{title, url}]
    confidence: str = "medium"


class CompressedFindings(BaseModel):
    """Compressed research findings before final report."""

    reasoning: str = Field(description="Why these are the key findings")
    compressed_summary: str = Field(description="Synthesized summary (800-1200 words)")
    key_themes: list[str] = Field(description="Common themes across findings")
    important_sources: list[str] = Field(description="Most important source URLs")


# FinalReport is now defined in models.py - removed duplicate


# ==================== Helper Functions ==========


async def create_initial_state(
    query: str,
    chat_history: list,
    mode: str,
    stream: Any,
    session_id: str,
    mode_config: dict[str, Any],
    settings: Any = None,
    session_manager: Any = None,
) -> ResearchState:
    """Create initial state for research graph.

    Args:
        query: Current query (might be clarification answer)
        chat_history: Chat history
        mode: Research mode
        stream: Stream generator
        session_id: Session ID for checkpointing
        mode_config: Mode configuration
        settings: Settings object
        session_manager: SessionManager instance for loading session data

    Returns:
        Initial research state with original_query loaded from session
    """

    # Load original_query, session_status, deep_search_result, and clarification_answers from session if session_manager provided
    original_query = query  # Default to current query
    session_status = "active"  # Default status
    deep_search_result = ""  # Default to empty
    clarification_answers = ""  # Default to empty
    
    logger.info("📥 CREATE_INITIAL_STATE: Starting",
               session_id=session_id,
               query_preview=query[:100] if query else None,
               has_session_manager=bool(session_manager),
               note="Loading session data from DB")
    
    if session_manager:
        try:
            session = await session_manager.get_session(session_id)
            if session:
                original_query = session.original_query
                old_session_status = session_status
                session_status = session.status
                
                logger.info("📥 CREATE_INITIAL_STATE: Session loaded from DB",
                           session_id=session_id,
                           original_query_preview=original_query[:100] if original_query else None,
                           session_status=session_status,
                           old_session_status=old_session_status,
                           has_deep_search_result=bool(session.deep_search_result),
                           has_clarification_answers=bool(session.clarification_answers),
                           note="Session data loaded successfully")
                
                # CRITICAL: Load deep_search_result from DB session to prevent double execution
                # This ensures deep search runs only once per session
                if session.deep_search_result:
                    deep_search_result = session.deep_search_result
                    logger.warning("📥 CREATE_INITIAL_STATE: Loaded deep_search_result from DB",
                               session_id=session_id,
                               result_length=len(deep_search_result),
                               result_preview=deep_search_result[:200] if deep_search_result else None,
                               note="CRITICAL: This should prevent double deep search execution")
                else:
                    logger.info("📥 CREATE_INITIAL_STATE: No deep_search_result in DB",
                               session_id=session_id,
                               note="This is a new session - deep search will execute")
                
                # CRITICAL: Load clarification_answers from DB session
                # This is the source of truth for clarification answers, not chat_history
                if session.clarification_answers:
                    clarification_answers = session.clarification_answers
                    logger.warning("📥 CREATE_INITIAL_STATE: Loaded clarification_answers from DB",
                               session_id=session_id,
                               answers_length=len(clarification_answers),
                               answers_preview=clarification_answers[:200] if clarification_answers else None,
                               note="CRITICAL: User already answered clarification - deep search should be skipped")
                else:
                    logger.info("📥 CREATE_INITIAL_STATE: No clarification_answers in DB",
                               session_id=session_id,
                               note="User has not answered clarification yet")
            else:
                logger.warning("📥 CREATE_INITIAL_STATE: Session not found in DB",
                             session_id=session_id,
                             note="Using default values")
        except Exception as e:
            # Fallback to current query if session loading fails
            logger.error("📥 CREATE_INITIAL_STATE: Failed to load session data",
                        session_id=session_id,
                        error=str(e),
                        exc_info=True,
                        note="Using default values as fallback")
            pass
    
    logger.info("📥 CREATE_INITIAL_STATE: Final state",
               session_id=session_id,
               original_query_preview=original_query[:100] if original_query else None,
               session_status=session_status,
               has_deep_search_result=bool(deep_search_result),
               deep_search_result_length=len(deep_search_result) if deep_search_result else 0,
               has_clarification_answers=bool(clarification_answers),
               clarification_answers_length=len(clarification_answers) if clarification_answers else 0,
               note="Initial state created - ready for graph execution")

    # Detect user language from original query
    # CRITICAL: Use reliable detection method - check for Cyrillic characters first (most common case)
    # langdetect can be unreliable for short or mixed texts, so we use character-based detection as primary
    user_language = "English"  # Default
    text_to_check = original_query if original_query else query
    
    if text_to_check:
        # Method 1: Check for Cyrillic characters (Russian, Ukrainian, etc.) - most reliable
        if any('\u0400' <= char <= '\u04FF' for char in text_to_check):
            user_language = "Russian"
            logger.info("Detected Russian language from Cyrillic characters",
                       query_preview=text_to_check[:50])
        else:
            # Method 2: Try langdetect for other languages (less reliable, but useful for non-Cyrillic)
            try:
                from langdetect import detect, DetectorFactory
                # Set seed for reproducibility
                DetectorFactory.seed = 0
                detected = detect(text_to_check)
                if detected == "ru":
                    user_language = "Russian"
                elif detected == "en":
                    user_language = "English"
                elif detected == "es":
                    user_language = "Spanish"
                elif detected == "fr":
                    user_language = "French"
                elif detected == "de":
                    user_language = "German"
                elif detected == "zh-cn" or detected == "zh-tw":
                    user_language = "Chinese"
                elif detected == "uk":
                    user_language = "Russian"  # Ukrainian -> Russian for now
                logger.info("Detected language using langdetect",
                           detected=detected,
                           user_language=user_language,
                           query_preview=text_to_check[:50])
            except Exception as e:
                # Fallback to English if detection fails
                logger.warning("Language detection failed, using English default",
                             error=str(e),
                             query_preview=text_to_check[:50])
                user_language = "English"

    return {
        # Input
        "query": query,
        "original_query": original_query,
        "chat_history": chat_history,
        "mode": mode,
        "user_language": user_language,

        # Analysis
        "query_analysis": {},

        # Planning
        "research_plan": {},
        "research_topics": [],
        "completed_topics": [],

        # Deep Search
        "deep_search_result": deep_search_result,  # Loaded from DB session if exists
        "clarification_answers": clarification_answers,  # Loaded from DB session if exists

        # Agent execution
        "active_agents": {},
        "agent_findings": [],
        "agent_todos": {},
        "agent_notes": {},
        "agent_characteristics": {},

        # Supervisor
        "supervisor_directives": [],
        "replanning_needed": False,
        "gaps_identified": [],

        # Memory
        "memory_context": [],
        "main_file_content": "",
        "shared_notes": [],

        # Output
        "final_report": "",
        "confidence": "medium",

        # Control flow
        "iteration": 0,
        "max_iterations": mode_config.get("max_iterations", 25),
        "should_continue": True,

        # Streaming
        "stream": stream,

        # Session
        "session_id": session_id,
        "session_status": session_status,

        # Config
        "mode_config": mode_config,

        # Settings
        "settings": settings,

        # Additional fields
        "estimated_agent_count": 4,
        "agent_count": 0,
        "requires_deep_search": True,
        "clarification_needed": False,
        "findings": [],
        "findings_count": 0,
        "compressed_research": "",
        "coordination_notes": "",
    }

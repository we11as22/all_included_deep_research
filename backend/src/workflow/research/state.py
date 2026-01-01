"""LangGraph state schema for deep research workflow.

Defines the state structure for the multi-agent research system.
"""

import operator
from typing import Annotated, Any, Dict, List, TypedDict

from pydantic import BaseModel, Field


# ==================== State Schema ====================


class ResearchState(TypedDict):
    """State schema for research graph.

    Uses TypedDict for LangGraph compatibility with reducers.
    """

    # ========== Input ==========
    query: str
    chat_history: list  # List of message dicts
    mode: str  # speed, balanced, quality

    # ========== Planning ==========
    research_plan: Annotated[List[str], operator.add]  # Topics to research
    completed_topics: Annotated[List[str], operator.add]  # Finished topics

    # ========== Agent Execution ==========
    active_agents: Dict[str, Dict]  # agent_id -> {topic, status, findings}
    agent_findings: Annotated[List[Dict], operator.add]  # All agent findings
    agent_todos: Dict[str, List[Dict]]  # agent_id -> todo list
    agent_notes: Dict[str, List[Dict]]  # agent_id -> notes list

    # ========== Supervisor State ==========
    supervisor_directives: Annotated[List[Dict], operator.add]  # Directive queue
    replanning_needed: bool
    gaps_identified: List[str]

    # ========== Deep Search ==========
    deep_search_result: str  # Initial deep search answer

    # ========== Memory ==========
    memory_context: List[Dict]  # Memory search results
    main_file_content: str  # Main research file content
    shared_notes: Annotated[List[Dict], operator.add]  # Cross-agent shared notes

    # ========== Agent Characteristics ==========
    agent_characteristics: Dict[str, Dict]  # agent_id -> {role, expertise, personality}

    # ========== Output ==========
    final_report: str
    confidence: str  # low, medium, high

    # ========== Settings ==========
    settings: Any  # Settings object

    # ========== Control Flow ==========
    iteration: int
    max_iterations: int
    should_continue: bool

    # ========== Streaming ==========
    stream: Any  # Streaming generator

    # ========== Session Info ==========
    session_id: str

    # ========== Mode Config ==========
    mode_config: Dict[str, Any]  # max_concurrent, max_sources, etc.


# ==================== Pydantic Models for Structured Outputs ==========


class ResearchTopic(BaseModel):
    """Single research topic."""

    reasoning: str = Field(description="Why this topic is important")
    topic: str = Field(description="Research topic title")
    description: str = Field(description="Detailed description of what to research")
    priority: str = Field(default="medium", description="Priority: low/medium/high")


class ResearchPlan(BaseModel):
    """Initial research plan from supervisor."""

    reasoning: str = Field(description="Overall research strategy")
    topics: List[ResearchTopic] = Field(description="List of research topics")
    stop: bool = Field(default=False, description="Whether planning is complete")


class SupervisorReActOutput(BaseModel):
    """Supervisor's reaction after agent actions."""

    reasoning: str = Field(description="Analysis of current research state")
    should_continue: bool = Field(description="Whether research should continue")
    replanning_needed: bool = Field(description="Whether new topics are needed")
    directives: List[Dict] = Field(
        default_factory=list,
        description="Todo updates for agents: [{agent_id, action, content}]"
    )
    new_topics: List[str] = Field(
        default_factory=list,
        description="New research topics to explore"
    )
    gaps_identified: List[str] = Field(
        default_factory=list,
        description="Identified research gaps"
    )


class AgentFinding(BaseModel):
    """Finding from a single agent."""

    agent_id: str
    topic: str
    summary: str
    key_findings: List[str]
    sources: List[Dict[str, str]]  # [{title, url}]
    confidence: str = "medium"


class CompressedFindings(BaseModel):
    """Compressed research findings before final report."""

    reasoning: str = Field(description="Why these are the key findings")
    compressed_summary: str = Field(description="Synthesized summary (800-1200 words)")
    key_themes: List[str] = Field(description="Common themes across findings")
    important_sources: List[str] = Field(description="Most important source URLs")


class FinalReport(BaseModel):
    """Final research report."""

    reasoning: str = Field(description="Report structure rationale")
    report: str = Field(description="Complete markdown report with citations")
    key_findings: List[str] = Field(description="Summary of key findings")
    sources_count: int = Field(description="Number of unique sources")


# ==================== Helper Functions ==========


def create_initial_state(
    query: str,
    chat_history: list,
    mode: str,
    stream: Any,
    session_id: str,
    mode_config: Dict[str, Any],
    settings: Any = None,
) -> ResearchState:
    """Create initial state for research graph."""

    return {
        # Input
        "query": query,
        "chat_history": chat_history,
        "mode": mode,

        # Planning
        "research_plan": [],
        "completed_topics": [],

        # Deep Search
        "deep_search_result": "",

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

        # Config
        "mode_config": mode_config,

        # Settings
        "settings": settings,
    }

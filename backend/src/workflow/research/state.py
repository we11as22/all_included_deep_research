"""LangGraph state schema for deep research workflow.

Defines the state structure for the multi-agent research system.
"""

import operator
from typing import Annotated, Any, TypedDict

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
    findings: list[dict]  # Findings from execute_agents (temporary, before adding to agent_findings)
    findings_count: int  # Count of findings
    compressed_research: str  # Compressed findings before final report
    coordination_notes: str  # Notes on how agents should coordinate

    # ========== Streaming ==========
    stream: Any  # Streaming generator

    # ========== Session Info ==========
    session_id: str

    # ========== Mode Config ==========
    mode_config: dict[str, Any]  # max_concurrent, max_sources, etc.


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
    topics: list[ResearchTopic] = Field(description="List of research topics")
    stop: bool = Field(default=False, description="Whether planning is complete")


class SupervisorReActOutput(BaseModel):
    """Supervisor's reaction after agent actions."""

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


def create_initial_state(
    query: str,
    chat_history: list,
    mode: str,
    stream: Any,
    session_id: str,
    mode_config: dict[str, Any],
    settings: Any = None,
) -> ResearchState:
    """Create initial state for research graph."""

    return {
        # Input
        "query": query,
        "chat_history": chat_history,
        "mode": mode,

        # Analysis
        "query_analysis": {},

        # Planning
        "research_plan": {},
        "research_topics": [],
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

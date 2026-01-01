"""Research workflow orchestration with new architecture.

NEW:
- src.workflow.search - Perplexica-style two-stage search
- src.workflow.research - LangGraph multi-agent deep research
"""

# New Search Workflow (Perplexica pattern)
from src.workflow.search import (
    SearchService,
    create_search_service,
    classify_query,
    QueryClassification,
    research_agent,
    writer_agent,
    CitedAnswer,
    ActionRegistry,
)

# New Deep Research Workflow (LangGraph)
from src.workflow.research import (
    create_research_graph,
    ResearchState,
    create_initial_state,
    SupervisorQueue,
    get_supervisor_queue,
    cleanup_supervisor_queue,
    run_researcher_agent,
)

__all__ = [
    # New Search Workflow
    "SearchService",
    "create_search_service",
    "classify_query",
    "QueryClassification",
    "research_agent",
    "writer_agent",
    "CitedAnswer",
    "ActionRegistry",

    # New Deep Research Workflow
    "create_research_graph",
    "ResearchState",
    "create_initial_state",
    "SupervisorQueue",
    "get_supervisor_queue",
    "cleanup_supervisor_queue",
    "run_researcher_agent",
]

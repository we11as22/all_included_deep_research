"""Deep research workflow module (LangGraph-based multi-agent system).

Exports:
- create_research_graph: Main graph factory
- run_research_graph: Main graph executor
- ResearchState: LangGraph state schema
- SupervisorQueue: Concurrent agent completion queue
- run_researcher_agent: Individual researcher agent
"""

from src.workflow.research.graph import create_research_graph, run_research_graph
from src.workflow.research.state import ResearchState, create_initial_state
from src.workflow.research.queue import SupervisorQueue, get_supervisor_queue, cleanup_supervisor_queue
from src.workflow.research.researcher import run_researcher_agent
from src.workflow.research.nodes import (
    search_memory_node,
    run_deep_search_node,
    clarify_with_user_node,
    analyze_query_node,
    plan_research_enhanced_node as plan_research_node,
    create_agent_characteristics_enhanced_node as spawn_agents_node,
    execute_agents_enhanced_node as execute_agents_node,
    supervisor_review_enhanced_node as supervisor_react_node,
    compress_findings_node,
    generate_final_report_enhanced_node as generate_report_node,
)

__all__ = [
    "create_research_graph",
    "run_research_graph",
    "ResearchState",
    "create_initial_state",
    "SupervisorQueue",
    "get_supervisor_queue",
    "cleanup_supervisor_queue",
    "run_researcher_agent",
    "search_memory_node",
    "plan_research_node",
    "spawn_agents_node",
    "execute_agents_node",
    "supervisor_react_node",
    "compress_findings_node",
    "generate_report_node",
]

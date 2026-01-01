"""Deep research workflow module (LangGraph-based multi-agent system).

Exports:
- create_research_graph: Main graph factory
- ResearchState: LangGraph state schema
- SupervisorQueue: Concurrent agent completion queue
- run_researcher_agent: Individual researcher agent
"""

from src.workflow.research.graph import create_research_graph
from src.workflow.research.state import ResearchState, create_initial_state
from src.workflow.research.queue import SupervisorQueue, get_supervisor_queue, cleanup_supervisor_queue
from src.workflow.research.researcher import run_researcher_agent
from src.workflow.research.nodes import (
    search_memory_node,
    plan_research_node,
    spawn_agents_node,
    execute_agents_node,
    supervisor_react_node,
    compress_findings_node,
    generate_report_node,
)

__all__ = [
    "create_research_graph",
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

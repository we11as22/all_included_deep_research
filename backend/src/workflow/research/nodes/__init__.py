"""Research workflow nodes - modular architecture with DI pattern.

Each node is implemented as a class inheriting from ResearchNode base class.
Legacy wrappers are provided for backward compatibility during migration.
"""

import contextvars

# Context variable for runtime dependencies (legacy support during migration)
runtime_deps_context = contextvars.ContextVar('runtime_deps', default=None)

# Import legacy helper function from nodes_legacy.py
from src.workflow.research.nodes_legacy import _get_runtime_deps

# Import node classes
from src.workflow.research.nodes.base import ResearchNode
from src.workflow.research.nodes.deep_search import DeepSearchNode
from src.workflow.research.nodes.clarify import ClarifyNode
from src.workflow.research.nodes.analyze import AnalyzeQueryNode
from src.workflow.research.nodes.plan import PlanResearchNode
from src.workflow.research.nodes.spawn_agents import SpawnAgentsNode
from src.workflow.research.nodes.execute_agents import ExecuteAgentsNode
from src.workflow.research.nodes.supervisor_review import SupervisorReviewNode
from src.workflow.research.nodes.compress import CompressFindingsNode
from src.workflow.research.nodes.report import GenerateReportNode

# Import legacy wrapper functions for backward compatibility
from src.workflow.research.nodes.deep_search import run_deep_search_node
from src.workflow.research.nodes.clarify import clarify_with_user_node
from src.workflow.research.nodes.analyze import analyze_query_node
from src.workflow.research.nodes.plan import plan_research_enhanced_node
from src.workflow.research.nodes.spawn_agents import create_agent_characteristics_enhanced_node
from src.workflow.research.nodes.execute_agents import execute_agents_enhanced_node
from src.workflow.research.nodes.supervisor_review import supervisor_review_enhanced_node
from src.workflow.research.nodes.compress import compress_findings_node
from src.workflow.research.nodes.report import generate_final_report_enhanced_node

__all__ = [
    # Context variables (legacy)
    "runtime_deps_context",
    "_get_runtime_deps",

    # Node classes
    "ResearchNode",
    "DeepSearchNode",
    "ClarifyNode",
    "AnalyzeQueryNode",
    "PlanResearchNode",
    "SpawnAgentsNode",
    "ExecuteAgentsNode",
    "SupervisorReviewNode",
    "CompressFindingsNode",
    "GenerateReportNode",

    # Legacy wrapper functions
    "run_deep_search_node",
    "clarify_with_user_node",
    "analyze_query_node",
    "plan_research_enhanced_node",
    "create_agent_characteristics_enhanced_node",
    "execute_agents_enhanced_node",
    "supervisor_review_enhanced_node",
    "compress_findings_node",
    "generate_final_report_enhanced_node",
]

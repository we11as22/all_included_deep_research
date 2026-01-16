"""Execute agents node for running research agents."""

import structlog
from typing import Dict, Any

from src.workflow.research.state import ResearchState
from src.workflow.research.nodes.base import ResearchNode

logger = structlog.get_logger(__name__)


class ExecuteAgentsNode(ResearchNode):
    """Execute research agents in parallel.

    TODO: Full implementation with DraftReportService integration.
    Currently delegates to legacy implementation from nodes.py.
    """

    async def execute(self, state: ResearchState) -> Dict[str, Any]:
        """Execute agents node.

        Args:
            state: Current research state

        Returns:
            State updates with findings
        """
        logger.info("ExecuteAgentsNode: delegating to legacy implementation")

        # Delegate to legacy implementation from nodes_legacy.py
        from src.workflow.research.nodes_legacy import execute_agents_enhanced_node as legacy_execute_agents

        return await legacy_execute_agents(state)


# Legacy function wrapper for backward compatibility
async def execute_agents_enhanced_node(state: ResearchState) -> Dict:
    """Legacy wrapper for ExecuteAgentsNode.

    This function maintains backward compatibility with existing code
    that imports execute_agents_enhanced_node directly.

    TODO: Update imports to use ExecuteAgentsNode class directly,
    then remove this wrapper.
    """
    from src.workflow.research.nodes import runtime_deps_context

    runtime_deps = runtime_deps_context.get()
    if not runtime_deps:
        logger.warning("Runtime dependencies not found in context")
        return {"findings": [], "agent_findings": []}

    # Create dependencies container
    from src.workflow.research.dependencies import ResearchDependencies

    deps = ResearchDependencies(
        llm=runtime_deps.get("llm"),
        search_provider=runtime_deps.get("search_provider"),
        scraper=runtime_deps.get("scraper"),
        stream=runtime_deps.get("stream"),
        agent_memory_service=runtime_deps.get("agent_memory_service"),
        agent_file_service=runtime_deps.get("agent_file_service"),
        session_factory=runtime_deps.get("session_factory"),
        session_manager=runtime_deps.get("session_manager"),
        settings=runtime_deps.get("settings"),
    )

    # Execute node
    node = ExecuteAgentsNode(deps)
    return await node.execute(state)

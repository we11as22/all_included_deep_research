"""Spawn agents node for creating agent characteristics."""

import structlog
from typing import Dict, Any

from src.workflow.research.state import ResearchState
from src.workflow.research.nodes.base import ResearchNode

logger = structlog.get_logger(__name__)


class SpawnAgentsNode(ResearchNode):
    """Create agent characteristics for research agents.

    TODO: Full implementation with prompt builder.
    Currently delegates to legacy implementation from nodes.py.
    """

    async def execute(self, state: ResearchState) -> Dict[str, Any]:
        """Execute spawn agents node.

        Args:
            state: Current research state

        Returns:
            State updates with agent_characteristics
        """
        logger.info("SpawnAgentsNode: delegating to legacy implementation")

        # Delegate to legacy implementation from nodes_legacy.py
        from src.workflow.research.nodes_legacy import create_agent_characteristics_enhanced_node as legacy_create_agent_characteristics

        return await legacy_create_agent_characteristics(state)


# Legacy function wrapper for backward compatibility
async def create_agent_characteristics_enhanced_node(state: ResearchState) -> Dict:
    """Legacy wrapper for SpawnAgentsNode.

    This function maintains backward compatibility with existing code
    that imports create_agent_characteristics_enhanced_node directly.

    TODO: Update imports to use SpawnAgentsNode class directly,
    then remove this wrapper.
    """
    from src.workflow.research.nodes import runtime_deps_context

    runtime_deps = runtime_deps_context.get()
    if not runtime_deps:
        logger.warning("Runtime dependencies not found in context")
        return {"agent_characteristics": []}

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
    node = SpawnAgentsNode(deps)
    return await node.execute(state)

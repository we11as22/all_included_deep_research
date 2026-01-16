"""Supervisor review node for coordinating research."""

import structlog
from typing import Dict, Any

from src.workflow.research.state import ResearchState
from src.workflow.research.nodes.base import ResearchNode

logger = structlog.get_logger(__name__)


class SupervisorReviewNode(ResearchNode):
    """Supervisor reviews agent progress and coordinates research.

    TODO: Full implementation with SupervisorPromptBuilder and DraftReportService.
    Currently delegates to legacy implementation from nodes.py.
    """

    async def execute(self, state: ResearchState) -> Dict[str, Any]:
        """Execute supervisor review node.

        Args:
            state: Current research state

        Returns:
            State updates with supervisor decisions
        """
        logger.info("SupervisorReviewNode: delegating to legacy implementation")

        # Delegate to legacy implementation from nodes_legacy.py
        from src.workflow.research.nodes_legacy import supervisor_review_enhanced_node as legacy_supervisor_review

        return await legacy_supervisor_review(state)


# Legacy function wrapper for backward compatibility
async def supervisor_review_enhanced_node(state: ResearchState) -> Dict:
    """Legacy wrapper for SupervisorReviewNode.

    This function maintains backward compatibility with existing code
    that imports supervisor_review_enhanced_node directly.

    TODO: Update imports to use SupervisorReviewNode class directly,
    then remove this wrapper.
    """
    from src.workflow.research.nodes import runtime_deps_context

    runtime_deps = runtime_deps_context.get()
    if not runtime_deps:
        logger.warning("Runtime dependencies not found in context")
        return {"should_continue": False}

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
    node = SupervisorReviewNode(deps)
    return await node.execute(state)

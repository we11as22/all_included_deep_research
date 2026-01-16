"""Base class for all research workflow nodes."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from src.workflow.research.dependencies import ResearchDependencies
from src.workflow.research.state import ResearchState


class ResearchNode(ABC):
    """Base class for research workflow nodes with dependency injection.

    All nodes inherit from this class and receive dependencies through constructor.
    This replaces the context variables approach with explicit DI.

    Example:
        class DeepSearchNode(ResearchNode):
            async def execute(self, state: ResearchState) -> Dict[str, Any]:
                # Access dependencies
                llm = self.deps.llm
                search_provider = self.deps.search_provider
                stream = self.deps.stream

                # No need for _restore_runtime_deps()!

                # Perform work
                result = await search_provider.search(...)

                return {"deep_search_result": result}
    """

    def __init__(self, deps: ResearchDependencies):
        """Initialize node with dependencies.

        Args:
            deps: ResearchDependencies container with all runtime dependencies
        """
        self.deps = deps

    @abstractmethod
    async def execute(self, state: ResearchState) -> Dict[str, Any]:
        """Execute node logic.

        Args:
            state: Current research state

        Returns:
            Dictionary of state updates
        """
        pass

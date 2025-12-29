"""Workflow factory for creating research workflows."""

from typing import Literal, Union

import structlog
from src.config.settings import Settings
from src.llm.factory import create_chat_model
from src.memory.hybrid_search import HybridSearchEngine
from src.workflow.balanced_research import BalancedResearchWorkflow
from src.workflow.quality_research import QualityResearchWorkflow
from src.workflow.speed_research import SpeedResearchWorkflow

logger = structlog.get_logger(__name__)


class WorkflowFactory:
    """Factory for creating research workflows based on mode."""

    def __init__(self, settings: Settings, search_engine: HybridSearchEngine):
        """
        Initialize workflow factory.

        Args:
            settings: Application settings
            search_engine: Hybrid search engine for memory
        """
        self.settings = settings
        self.search_engine = search_engine

        # Initialize LLMs
        self.research_llm = create_chat_model(
            settings.research_model,
            settings,
            max_tokens=settings.research_model_max_tokens,
            temperature=0.7,
        )
        self.compression_llm = create_chat_model(
            settings.compression_model,
            settings,
            max_tokens=settings.compression_model_max_tokens,
            temperature=0.3,
        )
        self.final_report_llm = create_chat_model(
            settings.final_report_model,
            settings,
            max_tokens=settings.final_report_model_max_tokens,
            temperature=0.7,
        )

        logger.info(
            "WorkflowFactory initialized",
            research_model=settings.research_model,
            search_provider=settings.search_provider,
            embedding_provider=settings.embedding_provider,
        )

    def create_workflow(
        self,
        mode: Literal["speed", "balanced", "quality"],
    ) -> Union[SpeedResearchWorkflow, BalancedResearchWorkflow, QualityResearchWorkflow]:
        """
        Create workflow for specified research mode.

        Args:
            mode: Research mode (speed, balanced, or quality)

        Returns:
            Appropriate workflow instance

        Raises:
            ValueError: If mode is invalid
        """
        logger.info("Creating workflow", mode=mode)

        if mode == "speed":
            return SpeedResearchWorkflow(
                settings=self.settings,
                llm=self.research_llm,
                report_llm=self.final_report_llm,
                search_engine=self.search_engine,
            )

        elif mode == "balanced":
            return BalancedResearchWorkflow(
                settings=self.settings,
                llm=self.research_llm,
                report_llm=self.final_report_llm,
                search_engine=self.search_engine,
            )

        elif mode == "quality":
            return QualityResearchWorkflow(
                settings=self.settings,
                llm=self.research_llm,
                compression_llm=self.compression_llm,
                report_llm=self.final_report_llm,
                search_engine=self.search_engine,
            )

        else:
            raise ValueError(f"Invalid research mode: {mode}. Must be 'speed', 'balanced', or 'quality'")

    def get_available_modes(self) -> list[str]:
        """Get list of available research modes."""
        return ["speed", "balanced", "quality"]

    def get_mode_config(self, mode: str) -> dict:
        """
        Get configuration for a specific mode.

        Args:
            mode: Research mode

        Returns:
            Dict with mode configuration
        """
        if mode == "speed":
            return {
                "max_iterations": self.settings.speed_max_iterations,
                "max_concurrent": self.settings.speed_max_concurrent,
                "description": "Fast answers with web search (2 iterations, 1 researcher)",
            }

        elif mode == "balanced":
            return {
                "max_iterations": self.settings.balanced_max_iterations,
                "max_concurrent": self.settings.balanced_max_concurrent,
                "description": "Deep search with moderate quality (6 iterations, 3 researchers)",
            }

        elif mode == "quality":
            return {
                "max_iterations": self.settings.quality_max_iterations,
                "max_concurrent": self.settings.quality_max_concurrent,
                "description": "Comprehensive deep research (25 iterations, 5 researchers)",
            }

        else:
            raise ValueError(f"Invalid research mode: {mode}")

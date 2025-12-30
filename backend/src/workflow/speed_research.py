"""Speed mode research workflow - fast answers with web search."""

import asyncio
from typing import Any, Literal

import structlog
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph

from src.config.settings import Settings
from src.memory.hybrid_search import HybridSearchEngine
from src.search.factory import create_search_provider
from src.search.scraper import WebScraper
from src.workflow.nodes.memory_search import search_memory_node
from src.workflow.nodes.planner import plan_research_node
from src.workflow.nodes.reporter import generate_final_report_node
from src.workflow.nodes.researcher import researcher_node
from src.workflow.state import ResearchFinding, ResearchState

logger = structlog.get_logger(__name__)


class SpeedResearchWorkflow:
    """Speed mode research workflow (2 iterations, 1 researcher, quick answers)."""

    def __init__(
        self,
        settings: Settings,
        llm: BaseChatModel,
        search_engine: HybridSearchEngine,
        report_llm: BaseChatModel | None = None,
    ):
        """
        Initialize speed research workflow.

        Args:
            settings: Application settings
            llm: LangChain LLM instance
            search_engine: Hybrid search engine for memory
        """
        self.settings = settings
        self.llm = llm
        self.report_llm = report_llm or llm
        self.search_engine = search_engine

        # Initialize providers
        self.search_provider = create_search_provider(settings)
        self.web_scraper = WebScraper(
            timeout=settings.scraper_timeout,
            use_playwright=settings.scraper_use_playwright,
            scroll_enabled=settings.scraper_scroll_enabled,
            scroll_pause=settings.scraper_scroll_pause,
            max_scrolls=settings.scraper_max_scrolls,
        )

        # Build workflow graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""

        workflow = StateGraph(ResearchState)

        # Add nodes
        workflow.add_node("search_memory", self._search_memory)
        workflow.add_node("plan_research", self._plan_research)
        workflow.add_node("quick_research", self._quick_research)
        workflow.add_node("generate_report", self._generate_report)

        # Define edges
        workflow.set_entry_point("search_memory")
        workflow.add_edge("search_memory", "plan_research")
        workflow.add_edge("plan_research", "quick_research")
        workflow.add_edge("quick_research", "generate_report")
        workflow.add_edge("generate_report", END)

        return workflow.compile()

    async def _search_memory(self, state: ResearchState) -> dict:
        """Search memory node with fewer results for speed."""
        return await search_memory_node(
            state=state,
            search_engine=self.search_engine,
            max_results=5,  # Less memory context for speed
        )

    async def _plan_research(self, state: ResearchState) -> dict:
        """Planning node."""
        return await plan_research_node(
            state=state,
            llm=self.llm,
        )

    async def _quick_research(self, state: ResearchState) -> dict:
        """
        Quick research with single researcher on primary topic.

        Speed mode focuses on the most important topic only.
        """
        research_topics = state.get("research_topics", [])
        stream = state.get("stream")
        if not research_topics:
            logger.warning("No research topics found, using original query")
            research_topics = [state.get("query", "")]

        # Take only the first topic for speed
        primary_topic = research_topics[0]

        logger.info("Starting speed research", topic=primary_topic)

        try:
            researcher_state = {
                "researcher_id": "speed_researcher_0",
                "research_topic": primary_topic,
                "focus_areas": [],
                "max_sources": 3,  # Fewer sources for speed
                "max_query_rounds": 1,
                "queries_per_round": 2,
                "memory_context": state.get("memory_context", []),
                "existing_findings": [],
                "messages": state.get("messages", []),
                "completed": False,
                "stream": stream,
            }

            if stream:
                stream.emit_research_start(researcher_id="speed_researcher_0", topic=primary_topic)

            result = await researcher_node(
                state=researcher_state,
                llm=self.llm,
                search_provider=self.search_provider,
                web_scraper=self.web_scraper,
            )

            # Convert to ResearchFinding
            finding = ResearchFinding(
                researcher_id="speed_researcher_0",
                topic=primary_topic,
                summary=result.get("summary", {}).get("value", ""),
                key_findings=result.get("key_findings", {}).get("value", []),
                sources=result.get("sources_found", {}).get("value", []),
                confidence=result.get("confidence_level", {}).get("value", "medium"),
            )

            logger.info("Speed research completed")

            return {
                "findings": {"type": "override", "value": [finding]},
            }

        except Exception as e:
            logger.error("Speed research failed", error=str(e))
            return {
                "findings": {"type": "override", "value": []},
            }

    async def _generate_report(self, state: ResearchState) -> dict:
        """Generate concise report for speed mode."""
        return await generate_final_report_node(
            state=state,
            llm=self.report_llm,
        )

    async def run(
        self,
        query: str,
        stream: Any | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> dict:
        """
        Execute speed research workflow.

        Args:
            query: Research query

        Returns:
            Final state with research report
        """
        logger.info("Starting speed research workflow", query=query)

        # Initialize state
        initial_state = {
            "query": query,
            "mode": "speed",
            "clarification_needed": False,
            "clarified_query": None,
            "memory_context": [],
            "research_plan": None,
            "research_topics": [],
            "messages": messages or [],
            "findings": [],
            "compressed_research": None,
            "final_report": None,
            "iterations": 0,
            "max_iterations": self.settings.speed_max_iterations,
            "max_concurrent_researchers": self.settings.speed_max_concurrent,
            "stream": stream,
        }

        # Run workflow
        try:
            final_state = await self.graph.ainvoke(initial_state)

            logger.info("Speed research workflow completed")

            return final_state

        except Exception as e:
            logger.error("Workflow execution failed", error=str(e))
            raise

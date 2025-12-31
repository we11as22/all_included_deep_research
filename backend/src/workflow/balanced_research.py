"""Balanced mode research workflow."""

import asyncio
from typing import Any, Literal

import structlog
from langgraph.graph import END, StateGraph
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.config.settings import Settings
from src.memory.hybrid_search import HybridSearchEngine
from src.search.factory import create_search_provider
from src.search.scraper import WebScraper
from src.workflow.nodes.memory_search import search_memory_node
from src.workflow.nodes.planner import plan_research_node
from src.workflow.nodes.reporter import generate_final_report_node
from src.workflow.nodes.researcher import researcher_node
from src.workflow.state import ResearchFinding, ResearchState, SourceReference
from src.utils.text import summarize_text
from src.workflow.agentic.schemas import GapTopics

logger = structlog.get_logger(__name__)


class BalancedResearchWorkflow:
    """Balanced mode research workflow (6 iterations, 3 concurrent researchers)."""

    def __init__(
        self,
        settings: Settings,
        llm: BaseChatModel,
        search_engine: HybridSearchEngine,
        report_llm: BaseChatModel | None = None,
    ):
        """
        Initialize balanced research workflow.

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
        workflow.add_node("conduct_research", self._conduct_research)
        workflow.add_node("generate_report", self._generate_report)

        # Define edges
        workflow.set_entry_point("search_memory")
        workflow.add_edge("search_memory", "plan_research")
        workflow.add_edge("plan_research", "conduct_research")
        workflow.add_edge("conduct_research", "generate_report")
        workflow.add_edge("generate_report", END)

        return workflow.compile()

    async def _search_memory(self, state: ResearchState) -> dict:
        """Search memory node."""
        return await search_memory_node(
            state=state,
            search_engine=self.search_engine,
            max_results=10,
        )

    async def _plan_research(self, state: ResearchState) -> dict:
        """Planning node."""
        return await plan_research_node(
            state=state,
            llm=self.llm,
        )

    async def _conduct_research(self, state: ResearchState) -> dict:
        """
        Conduct parallel research on all topics.

        This node spawns multiple researchers in parallel to investigate
        different topics simultaneously.
        """
        research_topics = state.get("research_topics", [])
        if not research_topics:
            research_topics = [state.get("query", "")]
        max_concurrent = min(self.settings.balanced_max_concurrent, len(research_topics))
        stream = state.get("stream")

        logger.info(
            "Starting parallel research",
            topics_count=len(research_topics),
            max_concurrent=max_concurrent,
        )

        findings: list[ResearchFinding] = []
        completed_topics: set[str] = set()
        max_rounds = 2
        round_topics = list(research_topics)

        async def run_round(topics: list[str], round_id: int) -> list[ResearchFinding]:
            semaphore = asyncio.Semaphore(max_concurrent)

            async def run_topic(idx: int, topic: str):
                async with semaphore:
                    researcher_id = f"balanced_r{round_id}_{idx}"
                    researcher_state = {
                        "researcher_id": researcher_id,
                        "research_topic": topic,
                        "focus_areas": [],
                        "max_sources": 6,
                        "max_query_rounds": 2,
                        "queries_per_round": 3,
                        "memory_context": state.get("memory_context", []),
                        "existing_findings": findings,
                        "messages": state.get("messages", []),
                        "completed": False,
                        "stream": stream,
                    }

                    if stream:
                        stream.emit_research_start(researcher_id=researcher_id, topic=topic)

                    return await researcher_node(
                        state=researcher_state,
                        llm=self.llm,
                        search_provider=self.search_provider,
                        web_scraper=self.web_scraper,
                    )

            results = await asyncio.gather(
                *[run_topic(idx, topic) for idx, topic in enumerate(topics)],
                return_exceptions=True,
            )

            round_findings: list[ResearchFinding] = []
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        "Researcher failed",
                        researcher_id=f"balanced_r{round_id}_{idx}",
                        error=str(result),
                    )
                    continue

                if isinstance(result, dict):
                    round_findings.append(
                        ResearchFinding(
                            researcher_id=f"balanced_r{round_id}_{idx}",
                            topic=topics[idx],
                            summary=result.get("summary", {}).get("value", ""),
                            key_findings=result.get("key_findings", {}).get("value", []),
                            sources=result.get("sources_found", {}).get("value", []),
                            confidence=result.get("confidence_level", {}).get("value", "medium"),
                        )
                    )

            return round_findings

        round_id = 0
        while round_topics and round_id < max_rounds:
            if stream:
                stream.emit_status(f"Research round {round_id + 1} started", step="research_round")

            new_findings = await run_round(round_topics, round_id)
            findings.extend(new_findings)
            completed_topics.update(round_topics)

            round_id += 1
            if round_id >= max_rounds:
                break

            gap_topics = await self._identify_gap_topics(
                query=state.get("query", ""),
                findings=findings,
                completed_topics=completed_topics,
                max_topics=3,
            )
            if stream and gap_topics:
                stream.emit_search_queries(gap_topics, label="supervisor_gap")
            round_topics = [topic for topic in gap_topics if topic not in completed_topics]

        if stream and round_id > 1:
            stream.emit_status("Balanced research synthesis complete", step="research_done")

        logger.info("Parallel research completed", findings_count=len(findings))

        return {
            "findings": {"type": "override", "value": findings},
            "iterations": round_id,
        }

    async def _generate_report(self, state: ResearchState) -> dict:
        """Report generation node."""
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
        Execute balanced research workflow.

        Args:
            query: Research query

        Returns:
            Final state with research report
        """
        logger.info("Starting balanced research workflow", query=query)

        # Initialize state
        initial_state = {
            "query": query,
            "mode": "balanced",
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
            "max_iterations": self.settings.balanced_max_iterations,
            "max_concurrent_researchers": self.settings.balanced_max_concurrent,
            "stream": stream,
        }

        # Run workflow
        try:
            final_state = await self.graph.ainvoke(initial_state)

            logger.info("Balanced research workflow completed")

            return final_state

        except Exception as e:
            logger.error("Workflow execution failed", error=str(e))
            raise

    async def _identify_gap_topics(
        self,
        query: str,
        findings: list[ResearchFinding],
        completed_topics: set[str],
        max_topics: int = 3,
    ) -> list[str]:
        if not findings or self.settings.llm_mode == "mock":
            return []

        prompt = """You are a research supervisor. Identify missing angles or gaps.

Research Query: {query}

Current Findings:
{findings}

Return JSON with fields reasoning and topics. If coverage is sufficient, return an empty topics list."""

        findings_text = "\n".join(
            [f"- {finding.topic}: {summarize_text(finding.summary, 3000)}" for finding in findings[:6]]
        )

        structured_llm = self.llm.with_structured_output(GapTopics, method="function_calling")
        response = await structured_llm.ainvoke(
            [HumanMessage(content=prompt.format(query=query, findings=findings_text, max_topics=max_topics))]
        )
        if not isinstance(response, GapTopics):
            raise ValueError("GapTopics response was not structured")
        topics = response.topics

        deduped = []
        seen = {topic.lower() for topic in completed_topics}
        for topic in topics:
            normalized = topic.lower()
            if normalized in seen:
                continue
            deduped.append(topic)
            seen.add(normalized)
            if len(deduped) >= max_topics:
                break
        return deduped

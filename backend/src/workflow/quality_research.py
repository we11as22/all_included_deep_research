"""Quality mode research workflow - comprehensive deep research."""

from typing import Any

import structlog
from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph

from src.config.settings import Settings
from src.memory.hybrid_search import HybridSearchEngine
from src.search.factory import create_search_provider
from src.search.scraper import WebScraper
from src.workflow.nodes.memory_search import search_memory_node
from src.workflow.nodes.planner import plan_research_node
from src.workflow.nodes.reporter import generate_final_report_node
from src.workflow.agentic.coordinator import AgenticResearchCoordinator
from src.workflow.state import ResearchFinding, ResearchState

logger = structlog.get_logger(__name__)


class QualityResearchWorkflow:
    """Quality mode research workflow (25 iterations, 5 concurrent researchers, comprehensive)."""

    def __init__(
        self,
        settings: Settings,
        llm: BaseChatModel,
        search_engine: HybridSearchEngine,
        compression_llm: BaseChatModel | None = None,
        report_llm: BaseChatModel | None = None,
    ):
        """
        Initialize quality research workflow.

        Args:
            settings: Application settings
            llm: LangChain LLM instance
            search_engine: Hybrid search engine for memory
        """
        self.settings = settings
        self.llm = llm
        self.compression_llm = compression_llm or llm
        self.report_llm = report_llm or llm
        self.search_engine = search_engine

        # Initialize providers
        self.search_provider = create_search_provider(settings)
        self.web_scraper = WebScraper()

        # Build workflow graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""

        workflow = StateGraph(ResearchState)

        # Add nodes
        workflow.add_node("search_memory", self._search_memory)
        workflow.add_node("plan_research", self._plan_research)
        workflow.add_node("deep_research", self._deep_research)
        workflow.add_node("compress_findings", self._compress_findings)
        workflow.add_node("generate_report", self._generate_report)

        # Define edges
        workflow.set_entry_point("search_memory")
        workflow.add_edge("search_memory", "plan_research")
        workflow.add_edge("plan_research", "deep_research")
        workflow.add_edge("deep_research", "compress_findings")
        workflow.add_edge("compress_findings", "generate_report")
        workflow.add_edge("generate_report", END)

        return workflow.compile()

    async def _search_memory(self, state: ResearchState) -> dict:
        """Search memory node with more results for quality."""
        return await search_memory_node(
            state=state,
            search_engine=self.search_engine,
            max_results=20,  # More memory context for quality
        )

    async def _plan_research(self, state: ResearchState) -> dict:
        """Planning node."""
        return await plan_research_node(
            state=state,
            llm=self.llm,
        )

    async def _deep_research(self, state: ResearchState) -> dict:
        """
        Conduct comprehensive parallel research on all topics.

        Quality mode investigates all topics with more sources and depth.
        """
        research_topics = state.get("research_topics", [])
        if not research_topics:
            research_topics = [state.get("query", "")]
        stream = state.get("stream")
        max_concurrent = min(self.settings.quality_max_concurrent, len(research_topics))

        logger.info(
            "Starting agentic deep research",
            topics_count=len(research_topics),
            max_concurrent=max_concurrent,
        )

        coordinator = AgenticResearchCoordinator(
            llm=self.llm,
            search_provider=self.search_provider,
            web_scraper=self.web_scraper,
            memory_context=state.get("memory_context", []),
            chat_history=state.get("messages", []),
            stream=stream,
            max_rounds=3,
            max_concurrent=max_concurrent,
            max_sources=10,
        )

        findings = await coordinator.run(query=state.get("query", ""), seed_tasks=research_topics)

        logger.info("Agentic deep research completed", findings_count=len(findings))

        return {
            "findings": {"type": "override", "value": findings},
            "iterations": min(3, len(research_topics)),
        }

    async def _compress_findings(self, state: ResearchState) -> dict:
        """
        Compress and synthesize findings from all researchers.

        For quality mode, we create a comprehensive synthesis before final report.
        """
        findings = state.get("findings", [])
        stream = state.get("stream")

        if not findings:
            return {
                "compressed_research": {"type": "override", "value": "No findings to compress"},
            }

        logger.info("Compressing findings", findings_count=len(findings))

        try:
            # Create compression prompt
            findings_text = "\n\n".join(
                [
                    f"### Topic: {finding.topic}\n"
                    f"**Summary:** {finding.summary}\n"
                    f"**Key Findings:**\n" + "\n".join([f"- {kf}" for kf in finding.key_findings])
                    for finding in findings
                ]
            )

            sources_catalog = _build_sources_catalog(findings)

            prompt = f"""You are synthesizing research findings into a coherent narrative.

Research Findings from Multiple Researchers:

{findings_text}
{sources_catalog}

Create a comprehensive synthesis that:
1. Identifies common themes and patterns across findings
2. Highlights key insights and discoveries
3. Notes any contradictions or uncertainties
4. Organizes information logically
5. Preserves all sources and cite them inline where relevant

Keep the synthesis detailed but well-structured (aim for 800-1200 words)."""

            response = await self.compression_llm.ainvoke([HumanMessage(content=prompt)])
            compressed = response.content if hasattr(response, "content") else str(response)

            logger.info("Findings compressed successfully", compressed_length=len(compressed))

            if stream:
                stream.emit_compression(compressed)

            return {
                "compressed_research": {"type": "override", "value": compressed},
            }

        except Exception as e:
            logger.error("Compression failed", error=str(e))
            return {
                "compressed_research": {"type": "override", "value": findings_text},
            }

    async def _generate_report(self, state: ResearchState) -> dict:
        """Generate comprehensive report for quality mode."""
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
        Execute quality research workflow.

        Args:
            query: Research query

        Returns:
            Final state with research report
        """
        logger.info("Starting quality research workflow", query=query)

        # Initialize state
        initial_state = {
            "query": query,
            "mode": "quality",
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
            "max_iterations": self.settings.quality_max_iterations,
            "max_concurrent_researchers": self.settings.quality_max_concurrent,
            "stream": stream,
        }

        # Run workflow
        try:
            final_state = await self.graph.ainvoke(initial_state)

            logger.info("Quality research workflow completed")

            return final_state

        except Exception as e:
            logger.error("Workflow execution failed", error=str(e))
            raise

def _build_sources_catalog(findings: list[ResearchFinding]) -> str:
    sources = []
    for finding in findings:
        for source in finding.sources:
            sources.append((source.title, source.url))

    if not sources:
        return ""

    lines = ["\n## Source Catalog"]
    seen = set()
    idx = 1
    for title, url in sources:
        if url in seen:
            continue
        seen.add(url)
        lines.append(f"[{idx}] {title}: {url}")
        idx += 1

    return "\n".join(lines)

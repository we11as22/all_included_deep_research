"""Unified search service with Perplexica-style two-stage architecture.

Integrates: Classifier → Research Agent → Writer Agent
"""

import structlog
from typing import Any, Optional

from src.workflow.search.classifier import classify_query, QueryClassification
from src.workflow.search.researcher import research_agent
from src.workflow.search.writer import writer_agent

logger = structlog.get_logger(__name__)


class SearchService:
    """Main search service with intelligent routing."""

    def __init__(
        self,
        classifier_llm: Any,
        research_llm: Any,
        writer_llm: Any,
        search_provider: Any,
        scraper: Any,
    ):
        """
        Initialize search service.

        Args:
            classifier_llm: LLM for query classification
            research_llm: LLM for research agent
            writer_llm: LLM for writer agent
            search_provider: Search provider instance
            scraper: Web scraper instance
        """
        self.classifier_llm = classifier_llm
        self.research_llm = research_llm
        self.writer_llm = writer_llm
        self.search_provider = search_provider
        self.scraper = scraper

    async def answer(
        self,
        query: str,
        chat_history: list[dict],
        stream: Any,
        force_mode: Optional[str] = None,
    ) -> str:
        """
        Answer query with appropriate mode.

        Args:
            query: User query
            chat_history: Previous messages
            stream: Streaming generator for progress updates
            force_mode: Force specific mode (chat/web/deep)

        Returns:
            Final answer (markdown with citations)
        """
        logger.info("Search service processing query", query=query[:100], force_mode=force_mode)

        # Step 1: Classify query (unless mode is forced)
        if force_mode:
            # Create dummy classification for forced mode
            classification = QueryClassification(
                reasoning=f"Forced mode: {force_mode}",
                query_type="factual",
                standalone_query=query,
                suggested_mode=force_mode,
                requires_sources=force_mode != "chat",
                time_sensitive=False,
            )
            logger.info(f"Using forced mode: {force_mode}")
        else:
            if stream:
                stream.emit_status("Classifying query...", step="classification")

            classification = await classify_query(
                query, chat_history, self.classifier_llm
            )
            logger.info(
                "Query classified",
                type=classification.query_type,
                mode=classification.suggested_mode,
            )

        # Step 2: Route to appropriate handler
        mode = force_mode or classification.suggested_mode

        if mode == "chat":
            return await self._answer_chat(query, chat_history, stream)
        elif mode == "web":
            return await self._answer_web_search(query, classification, stream, chat_history)
        elif mode == "deep":
            return await self._answer_deep_search(query, classification, stream, chat_history)
        else:
            # For research modes, delegate to deep research (will be handled by LangGraph in Phase 3)
            logger.info(f"Research mode {mode} not yet implemented, using deep search")
            return await self._answer_deep_search(query, classification, stream, chat_history)

    async def _answer_chat(
        self, query: str, chat_history: list[dict], stream: Any
    ) -> str:
        """Simple chat without web search."""
        logger.info("Answering in chat mode (no sources)")

        if stream:
            stream.emit_status("Generating answer...", step="chat")

        from langchain_core.messages import SystemMessage, HumanMessage

        # Simple Q&A without sources
        system_prompt = """You are a helpful AI assistant. Answer questions concisely and accurately.

IMPORTANT: Only answer from your training data. Do NOT make up information.
If you don't know something or it requires recent information, say so clearly."""

        # Format chat history
        messages = [SystemMessage(content=system_prompt)]
        for msg in chat_history[-4:]:  # Last 4 messages for context
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                from langchain_core.messages import AIMessage
                messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=query))

        # Get response
        response = await self.classifier_llm.ainvoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)

        return answer

    async def _answer_web_search(
        self,
        query: str,
        classification: QueryClassification,
        stream: Any,
        chat_history: list[dict],
    ) -> str:
        """Quick web search (speed mode - 2 iterations)."""
        logger.info("Answering with web search (speed mode)")

        # Research stage
        research_results = await research_agent(
            query=query,
            classification=classification,
            mode="speed",
            llm=self.research_llm,
            search_provider=self.search_provider,
            scraper=self.scraper,
            stream=stream,
            chat_history=chat_history,
        )

        # Writer stage
        answer = await writer_agent(
            query=query,
            research_results=research_results,
            llm=self.writer_llm,
            stream=stream,
            mode="speed",
            chat_history=chat_history,
        )

        return answer

    async def _answer_deep_search(
        self,
        query: str,
        classification: QueryClassification,
        stream: Any,
        chat_history: list[dict],
    ) -> str:
        """Deep search with iterations (balanced mode - 6 iterations)."""
        logger.info("Answering with deep search (balanced mode)")

        # Research stage
        research_results = await research_agent(
            query=query,
            classification=classification,
            mode="balanced",
            llm=self.research_llm,
            search_provider=self.search_provider,
            scraper=self.scraper,
            stream=stream,
            chat_history=chat_history,
        )

        # Writer stage
        answer = await writer_agent(
            query=query,
            research_results=research_results,
            llm=self.writer_llm,
            stream=stream,
            mode="balanced",
            chat_history=chat_history,
        )

        return answer


# ==================== Factory ====================


def create_search_service(
    classifier_llm: Any,
    research_llm: Any,
    writer_llm: Any,
    search_provider: Any,
    scraper: Any,
) -> SearchService:
    """Create search service instance (factory function)."""
    return SearchService(
        classifier_llm=classifier_llm,
        research_llm=research_llm,
        writer_llm=writer_llm,
        search_provider=search_provider,
        scraper=scraper,
    )

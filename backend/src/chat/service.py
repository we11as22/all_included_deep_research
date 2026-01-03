"""Chat service for web search and deep search modes."""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.config.settings import Settings
from src.embeddings.base import EmbeddingProvider
from src.llm.factory import create_chat_model
from src.memory.hybrid_search import HybridSearchEngine
from src.search.factory import create_search_provider
from src.search.models import ScrapedContent, SearchResult
from src.search.reranker import SemanticReranker
from src.search.scraper import WebScraper
from src.utils.chat_history import format_chat_history
from src.utils.date import get_current_date
from src.utils.text import summarize_text
from src.models.schemas import QueryRewrite, SearchQueries, FollowupQueries, SummarizedContent, SynthesizedAnswer

logger = structlog.get_logger(__name__)


@dataclass
class ChatSearchResult:
    """Result from chat search pipeline."""

    answer: str
    sources: list[SearchResult]
    memory_context: list[dict[str, Any]]


@dataclass(frozen=True)
class SearchTuning:
    """Parameters for multi-query web search."""

    mode: str
    max_results: int
    queries: int
    iterations: int
    scrape_top_n: int
    rerank_top_k: int
    label: str


@dataclass
class SearchSessionMemory:
    """Ephemeral session memory for deep web search."""

    notes: list[str]

    def __init__(self) -> None:
        self.notes = []

    def add_observation(self, query: str, results: list[SearchResult]) -> None:
        if not results:
            return
        top = results[:3]
        formatted = []
        for item in top:
            if not item.snippet:
                continue
            snippet = item.snippet
            if len(snippet) > 2400:
                snippet = summarize_text(snippet, 2400)
            formatted.append(f"{item.title}: {snippet}")
        snippets = "; ".join(formatted)
        if snippets:
            self.notes.append(f"{query}: {snippets}")

    def render(self, limit: int = 6) -> str:
        if not self.notes:
            return "Session Memory: None."
        lines = ["Session Memory:"]
        for note in self.notes[-limit:]:
            lines.append(f"- {note}")
        return "\n".join(lines)

    def clear(self) -> None:
        self.notes.clear()


class ChatSearchService:
    """Provide web search and deep search answers with citations."""

    def __init__(
        self,
        settings: Settings,
        search_engine: HybridSearchEngine,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.settings = settings
        self.search_engine = search_engine
        self.embedding_provider = embedding_provider
        self.search_provider = create_search_provider(settings)
        self.scraper = WebScraper(
            timeout=settings.scraper_timeout,
            use_playwright=settings.scraper_use_playwright,
            scroll_enabled=settings.scraper_scroll_enabled,
            scroll_pause=settings.scraper_scroll_pause,
            max_scrolls=settings.scraper_max_scrolls,
        )
        self.reranker = SemanticReranker(embedding_provider)
        self.blocked_domains = _parse_blocklist(settings.search_blocked_domains)
        self.blocked_keywords = _parse_blocklist(settings.search_blocked_keywords)

        self.chat_llm = create_chat_model(
            settings.chat_model,
            settings,
            max_tokens=settings.chat_model_max_tokens,
            temperature=0.4,
        )
        self.summarizer_llm = create_chat_model(
            settings.search_summarization_model,
            settings,
            max_tokens=settings.search_summarization_model_max_tokens,
            temperature=0.2,
        )

    async def answer_simple(
        self,
        query: str,
        stream: Any | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> ChatSearchResult:
        """Answer with simple LLM conversation (no web search)."""
        self._emit_status(stream, "Generating response...", step="chat")
        
        chat_history = format_chat_history(messages, self.settings.chat_history_limit) if messages else None
        current_date = get_current_date()
        
        system_prompt = (
            f"You are a helpful AI assistant. Provide clear, accurate, and helpful responses. "
            f"Return JSON with fields reasoning, answer, key_points. "
            f"Current date: {current_date} - always consider this when providing information about dates, events, or current affairs."
        )
        
        history_block = chat_history or "Chat history: None."
        user_prompt = (
            f"User question: {query}\n"
            f"Current date: {current_date}\n\n"
            f"{history_block}\n\n"
            "Please provide a helpful and accurate response."
        )
        
        structured_llm = self.chat_llm.with_structured_output(SynthesizedAnswer, method="function_calling")
        try:
            response = await structured_llm.ainvoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            if not isinstance(response, SynthesizedAnswer):
                raise ValueError("SynthesizedAnswer response was not structured")
            answer = response.answer.strip()
        except Exception as exc:
            logger.error("Structured answer generation failed", error=str(exc))
            raise
        
        self._emit_status(stream, "Response generated", step="complete")
        
        return ChatSearchResult(
            answer=answer,
            sources=[],
            memory_context=[],
        )

    async def answer_web(
        self,
        query: str,
        stream: Any | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> ChatSearchResult:
        """Answer with base web search (rewrites + multiple queries)."""
        tuning = self._web_tuning()
        return await self._run_multi_query_search(
            query,
            stream=stream,
            tuning=tuning,
            messages=messages,
        )

    async def answer_deep(
        self,
        query: str,
        stream: Any | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> ChatSearchResult:
        """Answer with quality deep search (more iterations and sources)."""
        tuning = self._deep_tuning()
        return await self._run_multi_query_search(
            query,
            stream=stream,
            tuning=tuning,
            messages=messages,
        )

    async def _run_multi_query_search(
        self,
        query: str,
        stream: Any | None,
        tuning: SearchTuning,
        messages: list[dict[str, str]] | None = None,
    ) -> ChatSearchResult:
        """
        Run multi-query search using LLM-driven research agent (like Perplexica).
        
        LLM sees previous results and decides what to search next.
        """
        try:
            # Ensure query is a string
            if not isinstance(query, str):
                if isinstance(query, (list, tuple)):
                    logger.error("Query was passed as list/embedding to _run_multi_query_search, using empty string", query_type=type(query).__name__)
                    query = ""
                else:
                    query = str(query) if query else ""

            chat_history = format_chat_history(messages, self.settings.chat_history_limit)
            memory_context: list[dict[str, Any]] = []

            # Classify query
            from src.workflow.search.classifier import classify_query
            if stream:
                stream.emit_status("Classifying query...", step="classification")
            
            classification = await classify_query(
                query, 
                [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in (messages or [])],
                self.chat_llm
            )

            # Map tuning mode to research agent mode
            research_mode = "speed" if tuning.mode == "web" else "balanced" if tuning.mode == "deep" else "quality"
            
            # Use research agent (LLM-driven, like Perplexica)
            # LLM sees results and decides what to search next
            from src.workflow.search.researcher import research_agent
            
            if stream:
                stream.emit_status(f"Starting {research_mode} research...", step="research")
            
            research_results = await research_agent(
                query=query,
                classification=classification,
                mode=research_mode,
                llm=self.chat_llm,  # Use same LLM for research
                search_provider=self.search_provider,
                scraper=self.scraper,
                stream=stream,
                chat_history=[{"role": m.get("role", "user"), "content": m.get("content", "")} for m in (messages or [])],
            )

            # Extract sources from research results
            # research_agent returns sources as list of dicts from web_search action
            sources = research_results.get("sources", [])
            scraped_content = research_results.get("scraped_content", [])
            
            # Convert to SearchResult format
            results: list[SearchResult] = []
            for source in sources:
                if isinstance(source, dict):
                    # Format: {"title": "...", "url": "...", "snippet": "..."}
                    results.append(SearchResult(
                        title=source.get("title", ""),
                        url=source.get("url", ""),
                        snippet=source.get("snippet", source.get("content", "")),
                        score=source.get("score", 0.0)
                    ))
                elif hasattr(source, "title"):
                    # Already a SearchResult object
                    results.append(source)
            
            logger.info(
                "Research agent completed",
                sources_count=len(sources),
                results_count=len(results),
                scraped_count=len(scraped_content)
            )

            # Like Perplexica: simple deduplication only, no reranking, no filtering
            # Perplexica doesn't rerank or filter - it trusts SearXNG's ranking completely
            # Only dedupe by URL (merge content for duplicates)
            results = self._dedupe_results_simple(results)  # Simple dedupe like Perplexica
            # NO filtering - Perplexica doesn't filter results
            
            logger.info("Results after deduplication", 
                       query=query,
                       results_count=len(results),
                       top_titles=[r.title[:50] for r in results[:5]])
            
            # NO reranking - like Perplexica, trust SearXNG's ranking
            # Reranking can actually make results worse if embeddings are not good
            
            self._emit_sources(stream, results, label=tuning.label)

            # Use scraped content from research agent, or scrape top results
            if scraped_content:
                # Already scraped by research agent
                # Convert dicts to ScrapedContent objects
                summarized = []
                for item in scraped_content:
                    if isinstance(item, dict):
                        # Convert dict to ScrapedContent object
                        summarized.append(ScrapedContent(
                            url=item.get("url", ""),
                            title=item.get("title", ""),
                            content=item.get("content", item.get("summary", "")),
                            markdown=None,
                            html=None,
                            images=[],
                            links=[]
                        ))
                    elif isinstance(item, ScrapedContent):
                        # Already a ScrapedContent object
                        summarized.append(item)
                    else:
                        logger.warning(f"Unexpected scraped_content item type", item_type=type(item).__name__)
            else:
                # Fallback: scrape top results
                scraped = await self._scrape_results(results, tuning.scrape_top_n, stream=stream)
                logger.info("Scraping completed", scraped_count=len(scraped), total_results=len(results))
                summarized = await self._summarize_scraped(query, scraped, stream=stream)
                logger.info("Summarization completed", summarized_count=len(summarized))

            self._emit_status(stream, "Synthesizing answer from sources...", "synthesize")
            logger.info("Starting answer synthesis", sources_count=len(results), scraped_count=len(summarized))
            
            # Map tuning mode to writer mode
            writer_mode = "speed" if tuning.mode == "web" else "balanced" if tuning.mode == "deep" else "quality"
            answer = await self._synthesize_answer(
                query=query,
                search_query=query,  # Use original query
                sources=results,
                scraped=summarized,
                memory_context=memory_context,
                mode=writer_mode,
                chat_history=chat_history,
            )

            if not answer or not answer.strip():
                logger.error(
                    "Empty answer generated",
                    query=query,
                    sources_count=len(results),
                    scraped_count=len(summarized)
                )
                answer = "I apologize, but I was unable to generate a comprehensive answer from the available sources. Please try rephrasing your question or try again later."

            self._emit_finding(stream, f"{tuning.mode}_search", answer)
            self._emit_status(stream, "Answer ready", "complete")
            return ChatSearchResult(answer=answer, sources=results, memory_context=memory_context)
        except Exception as e:
            logger.error("Multi-query search failed", error=str(e), exc_info=True)
            raise

    def _web_tuning(self) -> SearchTuning:
        """Web Search (Speed mode) - fast search with 2 iterations."""
        return SearchTuning(
            mode="web",
            max_results=self.settings.deep_search_max_results,
            queries=self.settings.deep_search_queries,
            iterations=self.settings.speed_max_iterations,  # 2 iterations for speed
            scrape_top_n=self.settings.deep_search_scrape_top_n,
            rerank_top_k=self.settings.deep_search_rerank_top_k,
            label="web",
        )

    def _deep_tuning(self) -> SearchTuning:
        """Deep Search (Balanced mode) - quality search with 6 iterations."""
        return SearchTuning(
            mode="deep",
            max_results=self.settings.deep_search_quality_max_results,
            queries=self.settings.deep_search_quality_queries,
            iterations=self.settings.balanced_max_iterations,  # 6 iterations for balanced
            scrape_top_n=self.settings.deep_search_quality_scrape_top_n,
            rerank_top_k=self.settings.deep_search_quality_rerank_top_k,
            label="deep",
        )

    async def _rewrite_query(self, query: str, chat_history: str | None = None) -> str:
        # Ensure input is a string, not a list/embedding
        if isinstance(query, (list, tuple)):
            logger.warning("Query was passed as list/embedding to _rewrite_query, using original query string", query_type=type(query).__name__)
            return str(query) if query else ""
        if not isinstance(query, str):
            query = str(query) if query else ""
        
        if self.settings.llm_mode == "mock":
            return query

        current_date = get_current_date()
        history_block = chat_history or "Chat history: None."
        prompt = (
            f"Rewrite the user query into a precise, focused web search query that will find relevant results. "
            f"IMPORTANT: Preserve the core meaning and key terms from the original query. "
            f"Keep the query language the same as the original (do not translate unless asked). "
            f"Use the chat history below for context - if the user is continuing a conversation, "
            f"incorporate relevant context from previous messages to make the search query more specific. "
            f"Current date: {current_date} - consider this when rewriting queries about recent events. "
            f"Return JSON with fields reasoning, rewritten_query. "
            f"Example: 'расскажи про сорта пива' -> 'сорта пива типы классификация' (NOT 'какие-то' or unrelated terms)."
        )
        structured_llm = self.chat_llm.with_structured_output(QueryRewrite, method="function_calling")
        try:
            response = await structured_llm.ainvoke(
                [SystemMessage(content=prompt), HumanMessage(content=f"{history_block}\n\nUser query: {query}")]
            )
            if not isinstance(response, QueryRewrite):
                raise ValueError("QueryRewrite response was not structured")
            rewritten = response.rewritten_query or query
        except Exception as exc:
            logger.error("Structured query rewrite failed", error=str(exc))
            rewritten = query
        
        # Final validation - ensure rewritten is always a string
        if not isinstance(rewritten, str):
            logger.warning("Rewritten query is not a string, converting", rewritten_type=type(rewritten).__name__)
            rewritten = str(rewritten) if rewritten else query
        
        return rewritten

    async def _generate_search_queries(
        self,
        query: str,
        max_queries: int,
        memory_context: list[dict[str, Any]] | None = None,
        chat_history: str | None = None,
    ) -> list[str]:
        if self.settings.llm_mode == "mock":
            return [
                query,
                f"{query} latest updates",
                f"{query} key players",
            ][:max_queries]

        memory_block = self._format_memory(memory_context or [])
        history_block = chat_history or "Chat history: None."
        current_date = get_current_date()
        prompt = (
            f"Generate concise, targeted web search queries based on the user's original query. "
            f"CRITICAL: Preserve the core meaning and key terms from the original query. "
            f"Use the same language as the original query - DO NOT translate or change language. "
            f"Generate queries that are SEO-friendly (keywords, not full sentences). "
            f"Each query should focus on a specific aspect of the topic. "
            f"Use memory context and chat history if relevant to make queries more specific. "
            f"Current date: {current_date} - consider this when generating queries about recent events. "
            f"Example: 'расскажи про саблю' -> ['сабля', 'сабля история', 'сабля виды типы'] (NOT 'история сабли в России' or unrelated terms). "
            f"Return JSON with fields reasoning, queries."
        )
        structured_llm = self.chat_llm.with_structured_output(SearchQueries, method="function_calling")
        try:
            response = await structured_llm.ainvoke(
                [
                    SystemMessage(content=prompt),
                    HumanMessage(content=f"{history_block}\n\nOriginal query: {query}\n\n{memory_block}"),
                ]
            )
            if not isinstance(response, SearchQueries):
                raise ValueError("SearchQueries response was not structured")
            queries = response.queries[:max_queries]
            return queries if queries else [query]
        except Exception as exc:
            logger.error("Structured query generation failed", error=str(exc))
            return [query]

    async def _generate_followup_queries(
        self,
        query: str,
        existing_queries: list[str],
        results: list[SearchResult] | None = None,
        memory_context: list[dict[str, Any]] | None = None,
        session_memory: SearchSessionMemory | None = None,
        chat_history: str | None = None,
    ) -> list[str]:
        if self.settings.llm_mode == "mock":
            return []

        results = results or []
        memory_block = self._format_memory(memory_context or [])
        session_block = session_memory.render() if session_memory else "Session Memory: None."
        history_block = chat_history or "Chat history: None."
        results_block = "\n".join(
            [
                f"- {item.title}: {summarize_text(item.snippet, 2400)}"
                for item in results[:6]
            ]
        )
        current_date = get_current_date()
        prompt = (
            f"Given the original user query and existing search queries, propose follow-up queries "
            f"that would fill gaps or validate key claims. "
            f"CRITICAL: Preserve the core meaning and key terms from the original query. "
            f"Use the same language as the original query - DO NOT translate or change language. "
            f"Generate queries that are SEO-friendly (keywords, not full sentences). "
            f"Each query should focus on a specific aspect that hasn't been covered yet. "
            f"Current date: {current_date} - consider this when proposing queries about recent events. "
            f"Example: Original 'расскажи про саблю', already searched ['сабля', 'сабля история'] -> "
            f"['сабля виды', 'сабля характеристики', 'сабля применение'] (NOT 'история сабли в России' or unrelated). "
            f"Return JSON with fields reasoning, should_continue, gap_summary, queries."
        )
        structured_llm = self.chat_llm.with_structured_output(FollowupQueries, method="function_calling")
        try:
            response = await structured_llm.ainvoke(
                [
                    SystemMessage(content=prompt),
                    HumanMessage(
                        content=(
                            f"Query: {query}\n"
                            f"Current date: {current_date}\n"
                            f"Queries:\n"
                            + "\n".join(existing_queries)
                            + f"\n\nTop Results:\n{results_block}\n\n{memory_block}\n\n{session_block}\n\n{history_block}"
                        )
                    ),
                ]
            )
            if not isinstance(response, FollowupQueries):
                raise ValueError("FollowupQueries response was not structured")
            if not response.should_continue:
                return []
            queries = response.queries
        except Exception as exc:
            logger.error("Structured followup query generation failed", error=str(exc))
            return []
        
        deduped = []
        for item in queries:
            if item.lower() not in {q.lower() for q in existing_queries}:
                deduped.append(item)
        return deduped[:3]

    async def _search_memory(self, query: str, stream: Any | None = None) -> list[dict[str, Any]]:
        _ = (query, stream)
        return []

    async def _rerank_results(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        if not results:
            return []
        try:
            reranked = await self.reranker.rerank(query, results, top_k=top_k)
            if not reranked:
                logger.warning(
                    "Reranking returned no results",
                    query=query[:100],
                    original_count=len(results),
                )
                # Return original results if reranking filtered everything out
                return results[:top_k] if top_k else results
            return reranked
        except Exception as exc:
            logger.error("rerank_failed", error=str(exc), exc_info=True)
            return results[:top_k] if top_k else results

    def _dedupe_results_simple(
        self,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Simple deduplication like Perplexica: merge content for duplicate URLs.
        No domain limits - Perplexica doesn't limit results per domain.
        """
        seen_urls: dict[str, int] = {}  # url -> index in deduped list
        deduped: list[SearchResult] = []
        
        for result in results:
            if not result.url:
                # Keep results without URL
                deduped.append(result)
                continue
                
            normalized_url = result.url.lower().strip()
            
            if normalized_url in seen_urls:
                # Duplicate URL - merge content like Perplexica
                existing_index = seen_urls[normalized_url]
                existing_result = deduped[existing_index]
                # Merge snippets if different
                if result.snippet and result.snippet not in existing_result.snippet:
                    existing_result.snippet += f"\n\n{result.snippet}"
                # Keep the result with better title if available
                if result.title and len(result.title) > len(existing_result.title):
                    existing_result.title = result.title
            else:
                # New URL - add to results
                seen_urls[normalized_url] = len(deduped)
                deduped.append(result)
        
        return deduped
    
    def _dedupe_results(
        self,
        results: list[SearchResult],
        per_domain_limit: int | None = None,
    ) -> list[SearchResult]:
        """Legacy method - kept for compatibility."""
        if per_domain_limit:
            # Use old method with domain limit
            seen = set()
            domain_counts: dict[str, int] = defaultdict(int)
            deduped = []
            for result in results:
                if result.url in seen:
                    continue
                domain = _normalize_domain(result.url)
                if per_domain_limit and domain:
                    if domain_counts[domain] >= per_domain_limit:
                        continue
                    domain_counts[domain] += 1
                seen.add(result.url)
                deduped.append(result)
            return deduped
        else:
            # Use simple deduplication
            return self._dedupe_results_simple(results)

    def _filter_blocked_results(self, results: list[SearchResult]) -> list[SearchResult]:
        if not results:
            return results
        if not self.blocked_domains and not self.blocked_keywords:
            return results
        filtered: list[SearchResult] = []
        for result in results:
            domain = _normalize_domain(result.url)
            blocked = False
            for item in self.blocked_domains:
                if domain == item or domain.endswith(f".{item}"):
                    blocked = True
                    break
            if blocked:
                continue
            haystack = f"{result.title} {result.snippet}".lower()
            if any(keyword in haystack for keyword in self.blocked_keywords):
                continue
            filtered.append(result)
        return filtered

    async def _scrape_results(
        self,
        results: list[SearchResult],
        top_n: int,
        stream: Any | None = None,
    ) -> list[ScrapedContent]:
        async def scrape_one(result: SearchResult) -> ScrapedContent | None:
            try:
                if stream:
                    self._emit_status(stream, f"Scraping {result.title}", "scrape")
                return await self.scraper.scrape(result.url)
            except Exception as exc:
                error_msg = str(exc) if exc else "Unknown scraping error"
                error_type = type(exc).__name__ if exc else "UnknownError"
                logger.warning(
                    "scrape_failed",
                    url=result.url,
                    error=error_msg,
                    error_type=error_type
                )
                return None

        tasks = [scrape_one(result) for result in results[:top_n]]
        # Use return_exceptions=True to continue even if some scrapes fail
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        scraped = [item for item in results_list if item and not isinstance(item, Exception)]
        return scraped

    async def _summarize_scraped(
        self,
        query: str,
        scraped: list[ScrapedContent],
        stream: Any | None = None,
    ) -> list[ScrapedContent]:
        if not scraped:
            logger.debug("No scraped content to summarize")
            return []

        logger.info("Starting summarization", scraped_count=len(scraped))

        async def summarize(content: ScrapedContent) -> ScrapedContent:
            trimmed = summarize_text(content.content, self.settings.search_content_max_chars)
            if len(trimmed) <= 1800:
                if stream:
                    self._emit_status(stream, f"Using full text for {content.title}", "summarize")
                return ScrapedContent(
                    url=content.url,
                    title=content.title,
                    content=trimmed.strip(),
                    markdown=None,
                    html=None,
                    images=content.images,
                    links=content.links,
                )
            prompt = (
                "Summarize the following source for a research assistant. "
                "Focus on facts relevant to the query. Keep it under 200 words. "
                "Return JSON with fields summary, key_points."
            )
            structured_llm = self.summarizer_llm.with_structured_output(SummarizedContent, method="function_calling")
            try:
                response = await structured_llm.ainvoke(
                    [SystemMessage(content=prompt), HumanMessage(content=f"Query: {query}\n\n{trimmed}")]
                )
                if not isinstance(response, SummarizedContent):
                    raise ValueError("SummarizedContent response was not structured")
                summary = response.summary
            except Exception as exc:
                logger.error("Structured summarization failed", error=str(exc))
                summary = summarize_text(trimmed, 4000)
            if stream:
                self._emit_status(stream, f"Summarized {content.title}", "summarize")
            return ScrapedContent(
                url=content.url,
                title=content.title,
                content=summary.strip(),
                markdown=None,
                html=None,
                images=content.images,
                links=content.links,
            )

        summarized_results = await asyncio.gather(*[summarize(item) for item in scraped])
        logger.info("Summarization completed", summarized_count=len(summarized_results))
        return summarized_results

    async def _synthesize_answer(
        self,
        query: str,
        search_query: str,
        sources: list[SearchResult],
        scraped: list[ScrapedContent],
        memory_context: list[dict[str, Any]],
        mode: str,
        chat_history: str | None = None,
    ) -> str:
        sources_block = self._format_sources(sources, scraped)
        memory_block = self._format_memory(memory_context)
        history_block = chat_history or "Chat history: None."

        current_date = get_current_date()
        
        # Mode-specific length guidelines (matching writer prompts)
        length_guide = {
            "simple": "300-500 words",
            "web": "400-600 words",  # Web Search (speed)
            "speed": "400-600 words",
            "deep": "800-1200 words",  # Deep Search (balanced)
            "balanced": "800-1200 words",
            "quality": "1500-3000 words",
            "research": "1500-3000 words"
        }.get(mode, "500-800 words")
        
        system_prompt = (
            f"You are an expert research assistant synthesizing information from multiple sources. "
            f"Current date: {current_date}\n\n"
            f"CRITICAL INSTRUCTIONS:\n"
            f"1. Use ALL provided sources - each one has valuable information\n"
            f"2. Synthesize into a comprehensive, well-structured answer ({length_guide})\n"
            f"3. CITE EVERY FACT with inline references [1], [2], etc.\n"
            f"4. Include specific details, data, and examples from sources\n"
            f"5. If sources provide different perspectives, present them all\n"
            f"6. Structure with clear sections using markdown (##, ###)\n"
            f"7. Be thorough - don't leave out important information\n\n"
            f"Return JSON with: reasoning (why sources support answer), answer (full markdown with citations), key_points (list)"
        )
        user_prompt = (
            f"User question: {query}\n"
            f"Search query: {search_query}\n"
            f"Mode: {mode}\n"
            f"Current date: {current_date}\n\n"
            f"{history_block}\n\n"
            f"{memory_block}\n\n"
            f"Sources ({len(sources)} sources provided - USE THEM ALL):\n{sources_block}\n\n"
            f"Write a comprehensive answer using ALL sources above. Each source adds value - synthesize them together."
        )

        structured_llm = self.chat_llm.with_structured_output(SynthesizedAnswer, method="function_calling")
        try:
            logger.debug("Calling LLM for answer synthesis", prompt_length=len(user_prompt))
            response = await structured_llm.ainvoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            logger.debug("LLM response received", response_type=type(response).__name__)
            if not isinstance(response, SynthesizedAnswer):
                raise ValueError("SynthesizedAnswer response was not structured")
            answer = response.answer.strip()
            logger.info("Answer synthesis completed", answer_length=len(answer))
            return answer
        except Exception as exc:
            logger.error("Structured answer synthesis failed", error=str(exc), exc_info=True)
            raise

    def _format_sources(self, sources: list[SearchResult], scraped: list[ScrapedContent]) -> str:
        scraped_map = {item.url: item for item in scraped}
        lines = []
        for idx, source in enumerate(sources[: self.settings.sources_limit], 1):
            summary = source.snippet
            if source.url in scraped_map:
                summary = scraped_map[source.url].content
            lines.append(
                f"[{idx}] {source.title}\nURL: {source.url}\nSummary: {summary}"
            )
        return "\n\n".join(lines) if lines else "No sources available."

    def _format_memory(self, memory_context: list[dict[str, Any]]) -> str:
        if not memory_context:
            return "Memory Context: None."
        lines = ["Memory Context (from prior notes):"]
        for item in memory_context:
            lines.append(
                f"- {item['title']} ({item['file_path']}): {summarize_text(item['content'], 6000)}"
            )
        return "\n".join(lines)

    def _emit_status(self, stream: Any | None, message: str, step: str) -> None:
        if stream:
            stream.emit_status(message, step=step)

    def _emit_sources(self, stream: Any | None, sources: list[SearchResult], label: str) -> None:
        if not stream:
            return
        for source in sources[: self.settings.sources_limit]:
            stream.emit_source(
                researcher_id=label,
                source={"url": source.url, "title": source.title},
            )

    def _emit_search_queries(self, stream: Any | None, queries: list[str], label: str) -> None:
        if stream:
            stream.emit_search_queries(queries, label=label)

    def _emit_finding(self, stream: Any | None, topic: str, summary: str) -> None:
        if not stream:
            return
        findings = self._extract_key_findings(summary)
        stream.emit_finding({
            "researcher_id": "search",
            "topic": topic,
            "summary": summary,
            "key_findings": findings,
        })

    def _extract_key_findings(self, summary: str) -> list[str]:
        lines = [line.strip("- ").strip() for line in summary.splitlines() if line.strip()]
        findings = [line for line in lines if len(line.split()) > 3]
        return findings[:5] if findings else ["Summary available in final answer."]


def _normalize_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return ""
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def _parse_blocklist(raw: str | None) -> list[str]:
    if not raw:
        return []
    items = []
    for part in raw.split(","):
        item = part.strip().lower()
        if item:
            items.append(item)
    return items

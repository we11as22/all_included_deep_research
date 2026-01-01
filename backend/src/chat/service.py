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
from src.workflow.legacy.agentic.schemas import QueryRewrite, SearchQueries, FollowupQueries, SummarizedContent, SynthesizedAnswer

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
        session_memory = SearchSessionMemory()
        try:
            chat_history = format_chat_history(messages, self.settings.chat_history_limit)
            self._emit_status(stream, "Rewriting query...", "rewrite")
            # Ensure query is a string before rewriting
            if not isinstance(query, str):
                if isinstance(query, (list, tuple)):
                    logger.error("Query was passed as list/embedding to _run_multi_query_search, using empty string", query_type=type(query).__name__)
                    query = ""
                else:
                    query = str(query) if query else ""
            rewritten = await self._rewrite_query(query, chat_history=chat_history)
            self._emit_search_queries(stream, [rewritten], "rewrite")

            memory_context: list[dict[str, Any]] = []

            self._emit_status(stream, "Generating search queries...", "queries")
            queries = await self._generate_search_queries(
                rewritten,
                tuning.queries,
                memory_context=memory_context,
                chat_history=chat_history,
            )

            results: list[SearchResult] = []
            all_queries: list[str] = []

            for iteration in range(1, tuning.iterations + 1):
                if not queries:
                    break

                label = tuning.label if iteration == 1 else f"{tuning.label}_iter_{iteration}"
                self._emit_search_queries(stream, queries, label)
                all_queries.extend(queries)

                async def run_query(search_query: str) -> list[SearchResult]:
                    response = await self.search_provider.search(
                        search_query,
                        max_results=tuning.max_results,
                    )
                    # Rerank results for each query to improve relevance
                    reranked = await self._rerank_results(search_query, response.results, top_k=tuning.max_results)
                    return reranked

                search_batches = await asyncio.gather(*[run_query(q) for q in queries])
                for query_text, batch in zip(queries, search_batches):
                    results.extend(batch)
                    session_memory.add_observation(query_text, batch)

                if iteration < tuning.iterations:
                    followups = await self._generate_followup_queries(
                        rewritten,
                        all_queries,
                        results=results,
                        memory_context=memory_context,
                        session_memory=session_memory,
                        chat_history=chat_history,
                    )
                    queries = [item for item in followups if item.lower() not in {q.lower() for q in all_queries}]

            results = self._dedupe_results(results, per_domain_limit=2)
            results = self._filter_blocked_results(results)
            results = await self._rerank_results(rewritten, results, top_k=tuning.rerank_top_k)
            self._emit_sources(stream, results, label=tuning.label)

            scraped = await self._scrape_results(results, tuning.scrape_top_n, stream=stream)
            summarized = await self._summarize_scraped(query, scraped, stream=stream)

            self._emit_status(stream, "Synthesizing answer from sources...", "synthesize")
            answer = await self._synthesize_answer(
                query=query,
                search_query=rewritten,
                sources=results,
                scraped=summarized,
                memory_context=memory_context,
                mode=tuning.mode,
                chat_history=chat_history,
            )

            self._emit_finding(stream, f"{tuning.mode}_search", answer)
            self._emit_status(stream, "Answer ready", "complete")
            return ChatSearchResult(answer=answer, sources=results, memory_context=memory_context)
        finally:
            session_memory.clear()

    def _web_tuning(self) -> SearchTuning:
        return SearchTuning(
            mode="web",
            max_results=self.settings.deep_search_max_results,
            queries=self.settings.deep_search_queries,
            iterations=self.settings.deep_search_iterations,
            scrape_top_n=self.settings.deep_search_scrape_top_n,
            rerank_top_k=self.settings.deep_search_rerank_top_k,
            label="web",
        )

    def _deep_tuning(self) -> SearchTuning:
        return SearchTuning(
            mode="deep",
            max_results=self.settings.deep_search_quality_max_results,
            queries=self.settings.deep_search_quality_queries,
            iterations=self.settings.deep_search_quality_iterations,
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
        prompt = (
            f"Rewrite the user query into a precise, focused web search query that will find relevant results. "
            f"IMPORTANT: Preserve the core meaning and key terms from the original query. "
            f"Keep the query language the same as the original (do not translate unless asked). "
            f"Use the chat history for context if needed. "
            f"Current date: {current_date} - consider this when rewriting queries about recent events. "
            f"Return JSON with fields reasoning, rewritten_query. "
            f"Example: 'расскажи про сорта пива' -> 'сорта пива типы классификация' (NOT 'какие-то' or unrelated terms)."
        )
        history_block = chat_history or "Chat history: None."
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
            f"Generate concise web search queries for deeper research. "
            f"Use memory context if relevant. "
            f"Use the same language as the original query and avoid mixing languages. "
            f"Current date: {current_date} - consider this when generating queries about recent events. "
            f"Return JSON with fields reasoning, queries."
        )
        structured_llm = self.chat_llm.with_structured_output(SearchQueries, method="function_calling")
        try:
            response = await structured_llm.ainvoke(
                [
                    SystemMessage(content=prompt),
                    HumanMessage(content=f"{history_block}\n\n{query}\n\n{memory_block}"),
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
            f"Given the original query and existing search queries, propose follow-up queries "
            f"that would fill gaps or validate key claims. "
            f"Use the same language as the original query and avoid mixing languages. "
            f"Current date: {current_date} - consider this when proposing queries about recent events. "
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
            return await self.reranker.rerank(query, results, top_k=top_k)
        except Exception as exc:
            logger.warning("rerank_failed", error=str(exc))
            return results[:top_k] if top_k else results

    def _dedupe_results(
        self,
        results: list[SearchResult],
        per_domain_limit: int | None = None,
    ) -> list[SearchResult]:
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
                logger.warning("scrape_failed", url=result.url, error=str(exc))
                return None

        tasks = [scrape_one(result) for result in results[:top_n]]
        scraped = [item for item in await asyncio.gather(*tasks) if item]
        return scraped

    async def _summarize_scraped(
        self,
        query: str,
        scraped: list[ScrapedContent],
        stream: Any | None = None,
    ) -> list[ScrapedContent]:
        if not scraped:
            return []

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

        return await asyncio.gather(*[summarize(item) for item in scraped])

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
        system_prompt = (
            f"You are a research assistant. Answer with clear, concise reasoning. "
            f"Return JSON with fields reasoning, answer, key_points. "
            f"Use the provided sources and cite them with [number] references. "
            f"If information is missing, say so. "
            f"Current date: {current_date} - always consider this when evaluating information recency and relevance."
        )
        user_prompt = (
            f"User question: {query}\n"
            f"Search query: {search_query}\n"
            f"Mode: {mode}\n"
            f"Current date: {current_date}\n\n"
            f"{history_block}\n\n"
            f"{memory_block}\n\n"
            f"Sources:\n{sources_block}\n\n"
            "Answer with citations like [1], [2]."
        )

        structured_llm = self.chat_llm.with_structured_output(SynthesizedAnswer, method="function_calling")
        try:
            response = await structured_llm.ainvoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            if not isinstance(response, SynthesizedAnswer):
                raise ValueError("SynthesizedAnswer response was not structured")
            return response.answer.strip()
        except Exception as exc:
            logger.error("Structured answer synthesis failed", error=str(exc))
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
        stream.emit_finding(
            researcher_id="search",
            topic=topic,
            summary=summary,
            key_findings=findings,
        )

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

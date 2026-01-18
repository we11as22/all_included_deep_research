"""Chat service for web search and deep search modes."""

from __future__ import annotations

import asyncio
import inspect
import json
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
        
        # CRITICAL: Use simple LLM call WITHOUT structured output to preserve markdown formatting!
        # Structured output (JSON) loses \n characters during JSON parsing
        # Deep Research uses simple llm.ainvoke() and preserves formatting - we should do the same!
        system_prompt = (
            f"You are a helpful AI assistant. Provide comprehensive, detailed, and accurate responses in markdown format. "
            f"Current date: {current_date} - always consider this when providing information about dates, events, or current affairs.\n\n"
            f"CRITICAL MARKDOWN FORMATTING REQUIREMENT:\n"
            f"- Your answer MUST be valid markdown with proper formatting - NOT plain text!\n"
            f"- Use ## for main sections (NOT # - start with ##)\n"
            f"- Use ### for subsections\n"
            f"- Use **bold** for emphasis, *italic* for subtle emphasis\n"
            f"- Use proper markdown lists (- for unordered, 1. for ordered)\n"
            f"- Use proper markdown links: [text](url)\n"
            f"- Structure with clear markdown sections - do NOT use plain text!\n"
            f"- CRITICAL: Format your answer as markdown, not plain text with large letters!\n"
            f"- CRITICAL NEWLINE FORMATTING: Use TWO newlines (\\n\\n) between paragraphs and sections for proper markdown rendering!\n"
            f"- Each paragraph must be separated by a blank line (two newlines: \\n\\n)!\n"
            f"- Sections must be separated by blank lines!\n"
            f"- This ensures proper markdown rendering on the frontend - without blank lines, paragraphs will merge together!\n\n"
            f"IMPORTANT:\n"
            f"- Provide comprehensive, detailed answers (800-1500 words minimum - be thorough!)\n"
            f"- Be thorough and complete - don't write brief summaries!\n"
            f"- Use proper markdown formatting throughout your response\n"
            f"- CRITICAL: Your answer MUST be in markdown format with proper headings (##), lists, and formatting!\n"
            f"- Do NOT return plain text - always use markdown syntax!\n"
            f"- Return ONLY your answer in markdown format. Do NOT return JSON or any other format. Just the markdown answer."
        )
        
        # Format chat history to show actual messages from the chat
        if messages and len(messages) > 0:
            history_lines = []
            history_lines.append("**Previous messages in this chat:**")
            for msg in messages[-4:]:  # Last 4 messages
                role = msg.get("role", "user")
                content = msg.get("content", "").strip()
                if content:
                    role_label = "User" if role == "user" else "Assistant"
                    # Truncate long messages for context
                    if len(content) > 500:
                        content = content[:500] + "..."
                    history_lines.append(f"- {role_label}: {content}")
            history_block = "\n".join(history_lines)
        else:
            history_block = "**Previous messages in this chat:** None (this is the first message)."
        
        user_prompt = (
            f"User question: {query}\n"
            f"Current date: {current_date}\n\n"
            f"{history_block}\n\n"
            f"CRITICAL: Write your ENTIRE answer in markdown format with proper formatting:\n"
            f"- Use ## for main sections (NOT # - start with ##)\n"
            f"- Use ### for subsections\n"
            f"- Use **bold**, *italic*, lists, headings - proper markdown syntax!\n"
            f"- Do NOT use plain text with large letters - use markdown headings!\n"
            f"- CRITICAL NEWLINE FORMATTING: Use TWO newlines (\\n\\n) between paragraphs and sections!\n"
            f"- Each paragraph must be separated by a blank line (two newlines)!\n"
            f"- Be comprehensive and detailed (600-1200 words minimum)\n"
            f"- Provide a helpful, accurate, and complete response based on the conversation context above.\n"
            f"- Use proper markdown formatting throughout your response.\n\n"
            f"Return ONLY your answer in markdown format. Do NOT return JSON or any other format. Just the markdown answer."
        )
        
        # CRITICAL: Use simple LLM call (like DeepSearchNode) to preserve formatting!
        # This avoids structured output JSON parsing which loses \n characters
        logger.info(
            "Calling LLM for chat answer (simple call, no structured output)",
            prompt_length=len(user_prompt),
            system_prompt_length=len(system_prompt),
            note="Using simple llm.ainvoke() like DeepSearchNode to preserve markdown formatting"
        )
        
        from langchain_core.messages import SystemMessage, HumanMessage
        chat_result = await self.chat_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        # CRITICAL: Get answer directly from LLM response (like DeepSearchNode)
        answer = chat_result.content if hasattr(chat_result, 'content') else str(chat_result)
        
        # CRITICAL: Log EXACTLY what LLM returned - BEFORE any processing
        if answer:
            import re
            raw_newline_count = answer.count('\n')
            raw_double_newline_count = answer.count('\n\n')
            raw_triple_newline_count = answer.count('\n\n\n')
            has_markdown_headings = bool(re.search(r'^#{2,}\s+', answer, re.MULTILINE))
            
            logger.info(
                "RAW LLM CHAT RESPONSE (before any processing)",
                answer_length=len(answer),
                newline_count=raw_newline_count,
                double_newline_count=raw_double_newline_count,
                triple_newline_count=raw_triple_newline_count,
                has_markdown_headings=has_markdown_headings,
                first_200_chars=repr(answer[:200]),  # Use repr to see actual \n characters
                last_200_chars=repr(answer[-200:]) if len(answer) > 200 else repr(answer),
                note="This is EXACTLY what LLM returned - no structured output, no JSON parsing"
            )

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
                self._emit_status(stream, "Classifying query...", step="classification")
            
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
                self._emit_status(stream, f"Starting {research_mode} research...", step="research")
            
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

            # Deduplication only - we trust the search provider's ranking
            # Both Tavily and SearXNG provide their own ranking, so we don't rerank
            results = self._dedupe_results_simple(results)
            
            logger.info("Results after deduplication", 
                       query=query,
                       results_count=len(results),
                       top_titles=[r.title[:50] for r in results[:5]],
                       search_provider=self.settings.search_provider,
                       note="Using search provider's ranking, no reranking applied")
            
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
                # Fallback: scrape top results with prioritization of authoritative sources
                scraped = await self._scrape_results(results, tuning.scrape_top_n, stream=stream, query=query)
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
                    sources_count=len(sources) if tuning.mode != "deep" else 0,
                    scraped_count=len(scraped_content) if tuning.mode != "deep" else 0
                )
                answer = "I apologize, but I was unable to generate a comprehensive answer from the available sources. Please try rephrasing your question or try again later."

            # For deep search mode, sources are already in answer with citations
            # For web/search mode, convert sources to SearchResult format
            if tuning.mode == "deep":
                # Sources are already included in answer with citations
                results = []
            else:
                results = sources if isinstance(sources, list) else []

            # CRITICAL: Log formatting before returning
            newline_count = answer.count("\n") if answer else 0
            has_newlines = "\n" in (answer or "")
            logger.info("Returning ChatSearchResult", 
                       answer_length=len(answer),
                       newline_count=newline_count,
                       has_newlines=has_newlines,
                       has_markdown_headings=bool(re.search(r'^#{2,}\s+', answer, re.MULTILINE)) if answer else False,
                       mode=tuning.mode)
            
            self._emit_finding(stream, f"{tuning.mode}_search", answer)
            self._emit_status(stream, "Answer ready", "complete")
            # CRITICAL: Return answer as-is - preserve all formatting including \n
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
        query: str | None = None,
    ) -> list[ScrapedContent]:
        """
        Scrape top results.
        
        Uses search provider's ranking - we trust the search engine's relevance ordering.
        Simply scrape top N results as ranked by the search provider.
        """
        if not results:
            return []
        
        # Trust search provider's ranking - scrape top N results as they are ordered
        # Search providers (SearXNG, Tavily) already rank by relevance
        selected_results = results[:top_n]
        
        logger.info(
            "Selecting URLs for scraping",
            total_results=len(results),
            selected_total=len(selected_results),
            top_n=top_n,
            query=query[:100] if query else None,
            selected_urls=[r.url for r in selected_results[:5]]
        )
        
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

        tasks = [scrape_one(result) for result in selected_results]
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
        
        # CRITICAL: Detect user language from query
        def _detect_user_language(text: str) -> str:
            """Detect user language from query text."""
            if not text:
                return "English"
            try:
                from langdetect import detect
                detected = detect(text)
                if detected == "ru":
                    return "Russian"
                elif detected == "en":
                    return "English"
                elif detected == "es":
                    return "Spanish"
                elif detected == "fr":
                    return "French"
                elif detected == "de":
                    return "German"
                elif detected == "zh-cn" or detected == "zh-tw":
                    return "Chinese"
                # Check for Cyrillic (Russian, Ukrainian, etc.)
                if any('\u0400' <= char <= '\u04FF' for char in text):
                    return "Russian"
                return "English"
            except Exception:
                # Fallback: check for Cyrillic
                if any('\u0400' <= char <= '\u04FF' for char in text):
                    return "Russian"
                return "English"
        
        user_language = _detect_user_language(query)
        logger.info("Detected user language for synthesis", language=user_language, query_preview=query[:50])
        
        # Mode-specific length guidelines (matching writer prompts)
        # CRITICAL: Increased minimums to ensure comprehensive answers
        length_guide = {
            "simple": "500-800 words",
            "web": "600-1000 words",  # Web Search (speed) - increased from 400-600
            "speed": "600-1000 words",  # Increased from 400-600
            "deep": "1200-2000 words",  # Deep Search (balanced) - increased from 800-1200
            "balanced": "1200-2000 words",  # Increased from 800-1200
            "quality": "2000-4000 words",  # Increased from 1500-3000
            "research": "2000-4000 words"  # Increased from 1500-3000
        }.get(mode, "800-1200 words")
        
        system_prompt = (
            f"You are an expert research assistant synthesizing information from multiple sources. "
            f"Current date: {current_date}\n\n"
            f"CRITICAL LANGUAGE REQUIREMENT:\n"
            f"- Write your ENTIRE answer in {user_language} (the same language as the user's query)\n"
            f"- All text, headings, citations, and sources section must be in {user_language}\n"
            f"- Detect the language from the user's query and match it exactly\n"
            f"- Do NOT mix languages - use ONLY {user_language} throughout the entire answer\n\n"
            f"CRITICAL MARKDOWN FORMATTING REQUIREMENT (MANDATORY - NO EXCEPTIONS):\n"
            f"- Your answer field MUST be valid markdown with proper formatting - NOT plain text!\n"
            f"- You MUST use markdown syntax throughout the entire answer\n"
            f"- Use ## for main sections (NOT # - start with ##)\n"
            f"- Use ### for subsections\n"
            f"- Use **bold** for emphasis, *italic* for subtle emphasis\n"
            f"- Use proper markdown lists (- for unordered, 1. for ordered)\n"
            f"- Use proper markdown links: [text](url)\n"
            f"- Structure with clear markdown sections - do NOT use plain text!\n"
            f"- CRITICAL: Format your answer as markdown, not plain text with large letters!\n"
            f"- FORBIDDEN: Do NOT write plain text without markdown formatting!\n"
            f"- FORBIDDEN: Do NOT use large letters or bold text without markdown syntax!\n"
            f"- EXAMPLE CORRECT FORMAT:\n"
            f"  ## Main Section Title\n\n"
            f"  This is a paragraph with **bold text** and *italic text*.\n\n"
            f"  ### Subsection\n\n"
            f"  - List item 1\n"
            f"  - List item 2\n\n"
            f"CRITICAL INSTRUCTIONS:\n"
            f"1. Use ALL provided sources - each one has valuable information\n"
            f"2. Synthesize into a comprehensive, well-structured answer ({length_guide})\n"
            f"3. CITE EVERY FACT with inline references [1], [2], etc. in the text\n"
            f"4. Include specific details, data, and examples from sources\n"
            f"5. If sources provide different perspectives, present them all\n"
            f"6. Structure with clear sections using markdown (##, ###) - NOT plain text!\n"
            f"7. Be thorough and comprehensive - don't leave out important information\n"
            f"8. Use proper markdown formatting: **bold**, *italic*, lists, headings\n"
            f"9. Preserve all markdown formatting in your response\n"
            f"10. Be detailed and complete - don't write brief summaries!\n\n"
            f"CRITICAL SOURCES FORMATTING REQUIREMENT:\n"
            f"- You MUST include inline citations [1], [2], [3], etc. in your answer text for every fact or claim\n"
            f"- At the END of your answer, you MUST add a \"## Sources\" section with ALL sources used\n"
            f"- Sources section format: Each source on a new line as \"- [Title](URL)\" (NOT \"[1] [Title](URL)\")\n"
            f"- Example Sources section:\n"
            f"  ## Sources\n\n"
            f"  - [Source Title 1](https://example.com/1)\n"
            f"  - [Source Title 2](https://example.com/2)\n"
            f"- Include the Sources section directly in your answer field - it will be part of the markdown\n\n"
            f"Return JSON with: reasoning (why sources support answer), answer (MUST be valid markdown with proper formatting including Sources section!), key_points (list)"
        )
        user_prompt = (
            f"User question: {query}\n"
            f"Search query: {search_query}\n"
            f"Mode: {mode}\n"
            f"Current date: {current_date}\n\n"
            f"CRITICAL: Write your ENTIRE answer in {user_language} (the same language as the query above).\n"
            f"All text, headings, citations, and sources section must be in {user_language}.\n\n"
            f"CRITICAL MARKDOWN FORMATTING (MANDATORY - NO EXCEPTIONS):\n"
            f"- Your answer MUST be valid markdown with proper formatting - NOT plain text!\n"
            f"- You MUST use markdown syntax throughout the entire answer\n"
            f"- Use ## for main sections (NOT # - start with ##)\n"
            f"- Use ### for subsections\n"
            f"- Use **bold**, *italic*, lists, headings - proper markdown syntax!\n"
            f"- Do NOT use plain text with large letters - use markdown headings!\n"
            f"- FORBIDDEN: Do NOT write plain text without markdown formatting!\n"
            f"- FORBIDDEN: Do NOT use large letters or bold text without markdown syntax!\n\n"
            f"{history_block}\n\n"
            f"{memory_block}\n\n"
            f"Sources ({len(sources)} sources provided - USE THEM ALL):\n{sources_block}\n\n"
            f"Write a comprehensive, detailed answer using ALL sources above. Each source adds value - synthesize them together.\n"
            f"Be thorough and complete - don't write brief summaries! Use proper markdown formatting throughout.\n\n"
            f"CRITICAL: Include inline citations [1], [2], [3] in your answer text for every fact.\n"
            f"CRITICAL: At the END of your answer, you MUST add a \"## Sources\" section with ALL sources used.\n"
            f"Sources section format: Each source on a new line as \"- [Title](URL)\" (NOT \"[1] [Title](URL)\").\n"
            f"Example Sources section:\n"
            f"  ## Sources\n\n"
            f"  - [Source Title 1](https://example.com/1)\n"
            f"  - [Source Title 2](https://example.com/2)\n"
            f"The Sources section format matches deep_research draft_report format."
        )

        # CRITICAL: Use simple LLM call WITHOUT structured output to preserve markdown formatting!
        # Structured output (JSON) loses \n characters during JSON parsing
        # Deep Research uses simple llm.ainvoke() and preserves formatting - we should do the same!
        # 
        # Build prompt that asks LLM to return ONLY the answer (not JSON)
        # Remove JSON requirement from prompts
        simple_system_prompt = (
            f"You are an expert research assistant synthesizing information from multiple sources. "
            f"Current date: {current_date}\n\n"
            f"CRITICAL LANGUAGE REQUIREMENT:\n"
            f"- Write your ENTIRE answer in {user_language} (the same language as the user's query)\n"
            f"- All text, headings, citations, and sources section must be in {user_language}\n"
            f"- Detect the language from the user's query and match it exactly\n"
            f"- Do NOT mix languages - use ONLY {user_language} throughout the entire answer\n\n"
            f"CRITICAL MARKDOWN FORMATTING REQUIREMENT (MANDATORY - NO EXCEPTIONS):\n"
            f"- Your answer MUST be valid markdown with proper formatting - NOT plain text!\n"
            f"- You MUST use markdown syntax throughout the entire answer\n"
            f"- Use ## for main sections (NOT # - start with ##)\n"
            f"- Use ### for subsections\n"
            f"- Use **bold** for emphasis, *italic* for subtle emphasis\n"
            f"- Use proper markdown lists (- for unordered, 1. for ordered)\n"
            f"- Use proper markdown links: [text](url)\n"
            f"- Structure with clear markdown sections - do NOT use plain text!\n"
            f"- CRITICAL: Format your answer as markdown, not plain text with large letters!\n"
            f"- FORBIDDEN: Do NOT write plain text without markdown formatting!\n"
            f"- FORBIDDEN: Do NOT use large letters or bold text without markdown syntax!\n"
            f"- CRITICAL NEWLINE FORMATTING: Use TWO newlines (\\n\\n) between paragraphs and sections for proper markdown rendering!\n"
            f"- Each paragraph must be separated by a blank line (two newlines: \\n\\n)!\n"
            f"- Sections must be separated by blank lines!\n"
            f"- This ensures proper markdown rendering on the frontend - without blank lines, paragraphs will merge together!\n"
            f"- EXAMPLE CORRECT FORMAT:\n"
            f"  ## Main Section Title\n\n"
            f"  This is a paragraph with **bold text** and *italic text*.\n\n"
            f"  ### Subsection\n\n"
            f"  - List item 1\n"
            f"  - List item 2\n\n"
            f"CRITICAL INSTRUCTIONS:\n"
            f"1. Use ALL provided sources - each one has valuable information\n"
            f"2. Synthesize into a comprehensive, well-structured answer ({length_guide})\n"
            f"3. CITE EVERY FACT with inline references [1], [2], etc. in the text\n"
            f"4. Include specific details, data, and examples from sources\n"
            f"5. If sources provide different perspectives, present them all\n"
            f"6. Structure with clear sections using markdown (##, ###) - NOT plain text!\n"
            f"7. Be thorough and comprehensive - don't leave out important information\n"
            f"8. Use proper markdown formatting: **bold**, *italic*, lists, headings\n"
            f"9. Preserve all markdown formatting in your response\n"
            f"10. Be detailed and complete - don't write brief summaries!\n\n"
            f"CRITICAL SOURCES FORMATTING REQUIREMENT:\n"
            f"- You MUST include inline citations [1], [2], [3], etc. in your answer text for every fact or claim\n"
            f"- At the END of your answer, you MUST add a \"## Sources\" section with ALL sources used\n"
            f"- Sources section format: Each source on a new line as \"- [Title](URL)\" (NOT \"[1] [Title](URL)\")\n"
            f"- Example Sources section:\n"
            f"  ## Sources\n\n"
            f"  - [Source Title 1](https://example.com/1)\n"
            f"  - [Source Title 2](https://example.com/2)\n"
            f"- Include the Sources section directly in your answer - it will be part of the markdown\n\n"
            f"IMPORTANT: Return ONLY your answer in markdown format. Do NOT return JSON or any other format. Just the markdown answer."
        )
        
        simple_user_prompt = (
            f"User question: {query}\n"
            f"Search query: {search_query}\n"
            f"Mode: {mode}\n"
            f"Current date: {current_date}\n\n"
            f"CRITICAL: Write your ENTIRE answer in {user_language} (the same language as the query above).\n"
            f"All text, headings, citations, and sources section must be in {user_language}.\n\n"
            f"CRITICAL MARKDOWN FORMATTING (MANDATORY - NO EXCEPTIONS):\n"
            f"- Your answer MUST be valid markdown with proper formatting - NOT plain text!\n"
            f"- You MUST use markdown syntax throughout the entire answer\n"
            f"- Use ## for main sections (NOT # - start with ##)\n"
            f"- Use ### for subsections\n"
            f"- Use **bold**, *italic*, lists, headings - proper markdown syntax!\n"
            f"- Do NOT use plain text with large letters - use markdown headings!\n"
            f"- FORBIDDEN: Do NOT write plain text without markdown formatting!\n"
            f"- FORBIDDEN: Do NOT use large letters or bold text without markdown syntax!\n"
            f"- CRITICAL NEWLINE FORMATTING: Use TWO newlines (\\n\\n) between paragraphs and sections!\n"
            f"- Each paragraph must be separated by a blank line (two newlines)!\n\n"
            f"{history_block}\n\n"
            f"{memory_block}\n\n"
            f"Sources ({len(sources)} sources provided - USE THEM ALL):\n{sources_block}\n\n"
            f"Write a comprehensive, detailed answer using ALL sources above. Each source adds value - synthesize them together.\n"
            f"Be thorough and complete - don't write brief summaries! Use proper markdown formatting throughout.\n\n"
            f"CRITICAL: Include inline citations [1], [2], [3] in your answer text for every fact.\n"
            f"CRITICAL: At the END of your answer, you MUST add a \"## Sources\" section with ALL sources used.\n"
            f"Sources section format: Each source on a new line as \"- [Title](URL)\" (NOT \"[1] [Title](URL)\").\n"
            f"Example Sources section:\n"
            f"  ## Sources\n\n"
            f"  - [Source Title 1](https://example.com/1)\n"
            f"  - [Source Title 2](https://example.com/2)\n"
            f"The Sources section format matches deep_research draft_report format.\n\n"
            f"Return ONLY your answer in markdown format. Do NOT return JSON or any other format. Just the markdown answer."
        )
        
        # CRITICAL: Use simple LLM call (like DeepSearchNode) to preserve formatting!
        # This avoids structured output JSON parsing which loses \n characters
        logger.info(
            "Calling LLM for answer synthesis (simple call, no structured output)",
            prompt_length=len(simple_user_prompt),
            system_prompt_length=len(simple_system_prompt),
            sources_count=len(sources),
            scraped_count=len(scraped),
            mode=mode,
            note="Using simple llm.ainvoke() like DeepSearchNode to preserve markdown formatting"
        )
        
        from langchain_core.messages import SystemMessage, HumanMessage
        synthesis_result = await self.chat_llm.ainvoke([
            SystemMessage(content=simple_system_prompt),
            HumanMessage(content=simple_user_prompt)
        ])
        
        # CRITICAL: Get answer directly from LLM response (like DeepSearchNode)
        answer = synthesis_result.content if hasattr(synthesis_result, 'content') else str(synthesis_result)
        
        # CRITICAL: Log EXACTLY what LLM returned - BEFORE any processing
        if answer:
            import re
            raw_newline_count = answer.count('\n')
            raw_double_newline_count = answer.count('\n\n')
            raw_triple_newline_count = answer.count('\n\n\n')
            has_markdown_headings = bool(re.search(r'^#{2,}\s+', answer, re.MULTILINE))
            
            logger.info(
                "RAW LLM SYNTHESIS RESPONSE (before any processing)",
                answer_length=len(answer),
                newline_count=raw_newline_count,
                double_newline_count=raw_double_newline_count,
                triple_newline_count=raw_triple_newline_count,
                has_markdown_headings=has_markdown_headings,
                first_200_chars=repr(answer[:200]),  # Use repr to see actual \n characters
                last_200_chars=repr(answer[-200:]) if len(answer) > 200 else repr(answer),
                note="This is EXACTLY what LLM returned - no structured output, no JSON parsing"
            )
        
        # CRITICAL: Ensure Sources section is present and formatted correctly with numbered sources
        # Format: [N] [Title](URL) where N matches the citation number in text [N]
        import re
        has_sources_section = "## Sources" in answer
        sources_section_pattern = r'(##\s+Sources.*?)(?=\n\n##|\Z)'
        
        # Extract citation numbers from text (e.g., [1], [2], [3])
        # Match [N] but not markdown links [text](url) or LaTeX formulas
        citation_pattern = r'(?<!\[)\[(\d+)\](?!\()'
        citations_found = re.findall(citation_pattern, answer)
        citation_numbers = sorted(set(int(c) for c in citations_found))
        
        # Create mapping: citation number -> source (by order of appearance in text)
        source_map = {}
        source_index = 1
        seen_urls = set()
        
        for source in sources:
            url = source.url if hasattr(source, 'url') else source.get('url', '') if isinstance(source, dict) else ''
            title = source.title if hasattr(source, 'title') else source.get('title', 'Unknown') if isinstance(source, dict) else 'Unknown'
            # Deduplicate by URL
            url_normalized = url.lower().rstrip('/') if url else ""
            if url_normalized and url_normalized not in seen_urls:
                seen_urls.add(url_normalized)
                # Map to citation number if it exists in text, otherwise use sequential number
                if source_index in citation_numbers:
                    source_map[source_index] = {'title': title, 'url': url}
                elif not source_map:  # If no citations found, use sequential numbering
                    source_map[source_index] = {'title': title, 'url': url}
                source_index += 1
        
        # If citations were found, ensure all cited sources are in the map
        if citation_numbers:
            # Fill in missing sources for cited numbers
            for num in citation_numbers:
                if num not in source_map and num <= len(sources):
                    # Try to find source by index
                    if num - 1 < len(sources):
                        source = sources[num - 1]
                        url = source.url if hasattr(source, 'url') else source.get('url', '') if isinstance(source, dict) else ''
                        title = source.title if hasattr(source, 'title') else source.get('title', 'Unknown') if isinstance(source, dict) else 'Unknown'
                        source_map[num] = {'title': title, 'url': url}
        
        if not has_sources_section and sources:
            # Add Sources section if LLM didn't include it
            # CRITICAL: Always include ALL sources, not just cited ones!
            answer += "\n\n## Sources\n\n"
            
            # CRITICAL: Use ALL sources from source_map, not just citation_numbers!
            all_source_numbers = sorted(source_map.keys()) if source_map else list(range(1, len(sources) + 1))
            
            # If source_map is empty, create it from sources
            if not source_map and sources:
                logger.warning("source_map is empty when adding Sources section - creating from sources", sources_count=len(sources))
                for idx, source in enumerate(sources, 1):
                    url = source.url if hasattr(source, 'url') else source.get('url', '') if isinstance(source, dict) else ''
                    title = source.title if hasattr(source, 'title') else source.get('title', 'Unknown') if isinstance(source, dict) else 'Unknown'
                    source_map[idx] = {'title': title, 'url': url}
                all_source_numbers = sorted(source_map.keys())
            
            for num in all_source_numbers:
                if num in source_map:
                    source_info = source_map[num]
                    if source_info.get('url'):
                        answer += f"[{num}] [{source_info['title']}]({source_info['url']})\n"
                    else:
                        answer += f"[{num}] {source_info['title']}\n"
            logger.info("Added all sources to Sources section", sources_added=len(all_source_numbers), total_sources=len(sources))
        elif has_sources_section:
            # LLM included Sources section - check if it has content and fix format if needed
            match = re.search(sources_section_pattern, answer, re.DOTALL | re.IGNORECASE)
            if match:
                sources_section = match.group(1)
                # Check if Sources section is empty or has no actual sources (only header)
                sources_content = sources_section.replace("## Sources", "").strip()
                has_sources_content = bool(re.search(r'\[.*?\]\(.*?\)|-\s+\[.*?\]', sources_content, re.IGNORECASE))
                
                if not has_sources_content and sources:
                    # Sources section exists but is empty - replace it with proper numbered sources
                    # CRITICAL: Always include ALL sources, not just cited ones!
                    logger.info("Sources section is empty - adding ALL numbered sources", 
                               sources_count=len(sources), 
                               citation_numbers=citation_numbers,
                               source_map_size=len(source_map))
                    
                    # CRITICAL: Use ALL sources from source_map, not just citation_numbers!
                    all_source_numbers = sorted(source_map.keys()) if source_map else list(range(1, len(sources) + 1))
                    
                    # If source_map is empty, create it from sources
                    if not source_map and sources:
                        logger.warning("source_map is empty in Sources section replacement - creating from sources", sources_count=len(sources))
                        for idx, source in enumerate(sources, 1):
                            url = source.url if hasattr(source, 'url') else source.get('url', '') if isinstance(source, dict) else ''
                            title = source.title if hasattr(source, 'title') else source.get('title', 'Unknown') if isinstance(source, dict) else 'Unknown'
                            source_map[idx] = {'title': title, 'url': url}
                        all_source_numbers = sorted(source_map.keys())
                    
                    sources_list = []
                    for num in all_source_numbers:
                        if num in source_map:
                            source_info = source_map[num]
                            if source_info.get('url'):
                                sources_list.append(f"[{num}] [{source_info['title']}]({source_info['url']})")
                            else:
                                sources_list.append(f"[{num}] {source_info['title']}")
                    
                    # CRITICAL: Use \n\n between sources for proper markdown formatting (same as deep research)
                    new_sources_section = f"## Sources\n\n" + "\n".join(sources_list) + "\n"
                    answer = answer.replace(sources_section, new_sources_section)
                    logger.info("Replaced empty Sources section with all sources", 
                               sources_added=len(sources_list),
                               total_sources=len(sources))
                else:
                    # Sources section has content - verify format and fix if needed
                    # Check if it uses [1] format
                    if re.search(r'\[\d+\]\s*\[', sources_section):
                        # Replace [1] [Title](URL) with - [Title](URL)
                        fixed_section = re.sub(r'\[\d+\]\s*', '- ', sources_section)
                        answer = answer.replace(sources_section, fixed_section)
                        logger.info("Fixed Sources section format from [1] [Title](URL) to - [Title](URL)")
            elif sources:
                # Sources section header exists but regex didn't match - add numbered sources after it
                # CRITICAL: Always include ALL sources, not just cited ones!
                logger.info("Sources section header found but no content - adding ALL numbered sources", 
                           sources_count=len(sources), 
                           citation_numbers=citation_numbers,
                           source_map_size=len(source_map))
                
                # CRITICAL: Use ALL sources from source_map, not just citation_numbers!
                all_source_numbers = sorted(source_map.keys()) if source_map else list(range(1, len(sources) + 1))
                
                # If source_map is empty, create it from sources
                if not source_map and sources:
                    logger.warning("source_map is empty in Sources section addition - creating from sources", sources_count=len(sources))
                    for idx, source in enumerate(sources, 1):
                        url = source.url if hasattr(source, 'url') else source.get('url', '') if isinstance(source, dict) else ''
                        title = source.title if hasattr(source, 'title') else source.get('title', 'Unknown') if isinstance(source, dict) else 'Unknown'
                        source_map[idx] = {'title': title, 'url': url}
                    all_source_numbers = sorted(source_map.keys())
                
                sources_list = []
                for num in all_source_numbers:
                    if num in source_map:
                        source_info = source_map[num]
                        if source_info.get('url'):
                            sources_list.append(f"[{num}] [{source_info['title']}]({source_info['url']})")
                        else:
                            sources_list.append(f"[{num}] {source_info['title']}")
                
                # Find position of "## Sources" and add sources after it
                sources_header_pos = answer.rfind("## Sources")
                if sources_header_pos != -1:
                    # Find end of line after "## Sources"
                    end_of_line = answer.find("\n", sources_header_pos)
                    if end_of_line != -1:
                        # CRITICAL: Use \n\n between sources for proper markdown formatting
                        answer = answer[:end_of_line+1] + "\n" + "\n".join(sources_list) + "\n" + answer[end_of_line+1:]
                    else:
                        answer += "\n\n" + "\n".join(sources_list) + "\n"
                logger.info("Added all sources to Sources section", sources_added=len(sources_list), total_sources=len(sources))
        
        # CRITICAL: Log formatting preservation before returning
        newline_count = answer.count("\n") if answer else 0
        has_newlines = "\n" in (answer or "")
        logger.info("Answer synthesis completed", 
                   answer_length=len(answer), 
                   has_sources_section="## Sources" in answer,
                   newline_count=newline_count,
                   has_newlines=has_newlines,
                   has_markdown_headings=bool(re.search(r'^#{2,}\s+', answer, re.MULTILINE)) if answer else False,
                   answer_preview=answer[:200] if answer else "")
        # CRITICAL: Return answer as-is - preserve all formatting including \n
        return answer

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

    def _coerce_synthesized_answer(self, response: Any) -> SynthesizedAnswer:
        logger.debug(
            "Coercing synthesized answer",
            response_type=type(response).__name__,
            response_is_none=response is None,
            has_model_dump=hasattr(response, "model_dump") if response is not None else False,
            has_content=hasattr(response, "content") if response is not None else False,
        )
        
        if response is None:
            logger.error("Response is None - cannot coerce to SynthesizedAnswer")
            raise ValueError("SynthesizedAnswer response was None")
        
        if isinstance(response, SynthesizedAnswer):
            logger.debug("Response is already SynthesizedAnswer")
            return response
        
        if isinstance(response, list) and response:
            first = response[0]
            logger.debug("Response is list", list_length=len(response), first_type=type(first).__name__)
            if isinstance(first, SynthesizedAnswer):
                return first
            if isinstance(first, dict):
                try:
                    return SynthesizedAnswer.model_validate(first)
                except Exception as e:
                    logger.error("Failed to validate dict from list", error=str(e), dict_keys=list(first.keys()) if isinstance(first, dict) else None)
                    raise
        
        if isinstance(response, dict):
            logger.debug("Response is dict", dict_keys=list(response.keys()))
            try:
                return SynthesizedAnswer.model_validate(response)
            except Exception as e:
                logger.error("Failed to validate dict", error=str(e), dict_keys=list(response.keys()))
                raise
        
        if hasattr(response, "model_dump"):
            logger.debug("Response has model_dump method")
            try:
                dumped = response.model_dump()
                logger.debug("Model dumped", dumped_type=type(dumped).__name__, dumped_keys=list(dumped.keys()) if isinstance(dumped, dict) else None)
                return SynthesizedAnswer.model_validate(dumped)
            except Exception as e:
                logger.error("Failed to validate model_dump result", error=str(e))
                raise
        
        if hasattr(response, "content"):
            content = getattr(response, "content", "")
            logger.debug("Response has content", content_type=type(content).__name__, content_length=len(str(content)) if content else 0)
            if isinstance(content, str) and content.strip():
                try:
                    return SynthesizedAnswer.model_validate_json(content)
                except Exception as e:
                    logger.error("Failed to parse content as JSON", error=str(e), content_preview=content[:200] if content else None)
                    pass
        
        logger.error(
            "Cannot coerce response to SynthesizedAnswer",
            response_type=type(response).__name__,
            response_str=str(response)[:500] if response else None,
            response_repr=repr(response)[:500] if response else None,
        )
        raise ValueError("SynthesizedAnswer response was not structured")

    async def _invoke_structured_answer(
        self,
        structured_llm: Any,
        messages: list[Any],
        context: str,
    ) -> str:
        retries = max(1, self.settings.max_structured_output_retries)
        last_error: Exception | None = None

        logger.debug(
            "Invoking structured answer",
            context=context,
            retries=retries,
            messages_count=len(messages),
        )

        for attempt in range(1, retries + 1):
            try:
                # CRITICAL: Log max_tokens before calling
                max_tokens_before = None
                if hasattr(structured_llm, "max_tokens"):
                    max_tokens_before = structured_llm.max_tokens
                elif hasattr(structured_llm, "llm") and hasattr(structured_llm.llm, "max_tokens"):
                    max_tokens_before = structured_llm.llm.max_tokens
                
                logger.debug(
                    "Calling structured LLM",
                    context=context,
                    attempt=attempt,
                    retries=retries,
                    max_tokens=max_tokens_before,
                    messages_count=len(messages),
                )
                response = await structured_llm.ainvoke(messages)
                
                # CRITICAL: Log full response details before coercion
                logger.info(
                    "Structured answer response received",
                    context=context,
                    attempt=attempt,
                    response_type=type(response).__name__,
                    response_is_none=response is None,
                    response_str_preview=str(response)[:1000] if response else None,
                    response_repr_preview=repr(response)[:1000] if response else None,
                )
                
                # Additional logging for debugging
                if response is not None:
                    if hasattr(response, "__dict__"):
                        logger.debug("Response attributes", attrs=list(response.__dict__.keys()))
                    if hasattr(response, "content"):
                        content = getattr(response, "content", "")
                        logger.debug("Response content", content_type=type(content).__name__, content_length=len(str(content)) if content else 0, content_preview=str(content)[:500] if content else None)
                
                if response is None:
                    logger.error("LLM returned None response", context=context, attempt=attempt)
                    raise ValueError("LLM returned None response")
                
                synthesized = self._coerce_synthesized_answer(response)
                # CRITICAL: Don't use .strip() - it removes leading/trailing newlines which are important for markdown formatting!
                # Only strip if answer is truly empty (all whitespace)
                answer = synthesized.answer
                
                # CRITICAL: Log EXACTLY what structured output returned - BEFORE any processing
                if answer:
                    raw_newline_count = answer.count('\n')
                    raw_double_newline_count = answer.count('\n\n')
                    raw_triple_newline_count = answer.count('\n\n\n')
                    has_markdown_headings = bool(re.search(r'^#{2,}\s+', answer, re.MULTILINE))
                    
                    logger.info(
                        "RAW STRUCTURED OUTPUT RESPONSE (before any processing)",
                        context=context,
                        attempt=attempt,
                        answer_length=len(answer),
                        newline_count=raw_newline_count,
                        double_newline_count=raw_double_newline_count,
                        triple_newline_count=raw_triple_newline_count,
                        has_markdown_headings=has_markdown_headings,
                        first_200_chars=repr(answer[:200]),  # Use repr to see actual \n characters
                        last_200_chars=repr(answer[-200:]) if len(answer) > 200 else repr(answer),
                        note="This is EXACTLY what structured output returned - no modifications yet"
                    )
                    
                    # Show sample of actual content with newlines visible
                    sample_lines = answer.split('\n')[:10]
                    logger.debug(
                        "First 10 lines of raw structured output response (showing actual newlines)",
                        context=context,
                        lines=[repr(line) for line in sample_lines],
                        note="Use repr() to see actual \\n characters"
                    )
                
                if not answer or not answer.strip():
                    logger.warning("Synthesized answer is empty", context=context, attempt=attempt)
                    raise ValueError("Structured answer was empty")
                
                # CRITICAL: Check if answer contains markdown formatting
                has_markdown_headings = bool(re.search(r'^#{2,}\s+', answer, re.MULTILINE))
                has_markdown_bold = bool(re.search(r'\*\*.*?\*\*', answer))
                has_markdown_lists = bool(re.search(r'^[-*+]\s+', answer, re.MULTILINE))
                has_markdown = has_markdown_headings or has_markdown_bold or has_markdown_lists
                
                if not has_markdown:
                    logger.warning(
                        "Answer appears to lack markdown formatting",
                        context=context,
                        attempt=attempt,
                        answer_preview=answer[:500],
                        has_headings=has_markdown_headings,
                        has_bold=has_markdown_bold,
                        has_lists=has_markdown_lists,
                    )
                else:
                    logger.info(
                        "Answer contains markdown formatting",
                        context=context,
                        attempt=attempt,
                        has_headings=has_markdown_headings,
                        has_bold=has_markdown_bold,
                        has_lists=has_markdown_lists,
                    )
                
                logger.info(
                    "Structured answer generated successfully",
                    context=context,
                    attempt=attempt,
                    answer_length=len(answer),
                    has_markdown=has_markdown,
                )
                return answer
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Structured answer parsing failed",
                    context=context,
                    attempt=attempt,
                    retries=retries,
                    error=str(exc),
                    error_type=type(exc).__name__,
                    exc_info=True,
                )

        logger.error(
            "All attempts to generate structured answer failed",
            context=context,
            retries=retries,
            last_error=str(last_error) if last_error else None,
        )
        if last_error:
            raise last_error
        raise ValueError("Structured answer generation failed")

    def _emit_status(self, stream: Any | None, message: str, step: str) -> None:
        if stream:
            self._fire_and_forget(stream.emit_status(message, step=step))

    def _emit_sources(self, stream: Any | None, sources: list[SearchResult], label: str) -> None:
        if not stream:
            return
        for source in sources[: self.settings.sources_limit]:
            self._fire_and_forget(
                stream.emit_source(
                    researcher_id=label,
                    source={"url": source.url, "title": source.title},
                )
            )

    def _emit_search_queries(self, stream: Any | None, queries: list[str], label: str) -> None:
        if stream:
            self._fire_and_forget(stream.emit_search_queries(queries, label=label))

    def _emit_finding(self, stream: Any | None, topic: str, summary: str) -> None:
        if not stream:
            return
        findings = self._extract_key_findings(summary)
        self._fire_and_forget(stream.emit_finding({
            "researcher_id": "search",
            "topic": topic,
            "summary": summary,
            "key_findings": findings,
        }))

    def _fire_and_forget(self, result: Any) -> None:
        if inspect.isawaitable(result):
            if isinstance(result, asyncio.Task):
                return
            try:
                asyncio.create_task(result)
            except RuntimeError:
                return

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

"""Researcher node for conducting web research."""

from __future__ import annotations

import asyncio
import math
from collections import defaultdict
from typing import Any
from urllib.parse import urlparse

import structlog
from langchain_core.messages import HumanMessage

from src.search.base import SearchProvider
from src.search.scraper import WebScraper
from src.workflow.state import ResearchFinding, ResearcherState, SourceReference
from src.utils.chat_history import format_chat_history
from src.utils.date import get_current_date
from src.utils.text import summarize_text
from src.workflow.agentic.schemas import SearchQueries, ResearchAnalysis, FollowupQueries

logger = structlog.get_logger(__name__)


RESEARCHER_SYSTEM_PROMPT = """You are a specialized research agent conducting in-depth investigation.

Current date: {current_date}

Your assignment: {research_topic}

Relevant memory context:
{memory_context}

Existing findings from other researchers:
{existing_findings}

Your responsibilities:
1. **Search Strategy**: Plan effective search queries for your topic
2. **Source Analysis**: Critically evaluate sources for relevance and credibility
3. **Information Extraction**: Extract key insights from sources
4. **Synthesis**: Combine findings into a coherent summary

Available Tools:
- Reflect after each search round to identify gaps and refine queries

Guidelines:
- Focus exclusively on your assigned topic
- Use multiple search queries from different angles
- Prioritize authoritative and recent sources
- Be critical of source quality and bias
- Extract specific facts, examples, and insights
- Synthesize don't just summarize
- Always consider the current date ({current_date}) when evaluating information recency and relevance

Maximum sources to analyze: {max_sources}"""


async def researcher_node(
    state: ResearcherState,
    llm: any,  # LangChain LLM
    search_provider: SearchProvider,
    web_scraper: WebScraper,
) -> dict:
    """
    Researcher conducts web research on assigned topic.

    Args:
        state: Current researcher state
        llm: LangChain LLM for analysis
        search_provider: Search provider for web search
        web_scraper: Web scraper for content extraction

    Returns:
        State update with research findings
    """
    researcher_id = state.get("researcher_id", "unknown")
    topic = state.get("research_topic", "")
    max_sources = state.get("max_sources", 5)
    max_query_rounds = max(1, state.get("max_query_rounds", 2))
    queries_per_round = max(1, state.get("queries_per_round", 3))
    memory_context = state.get("memory_context", [])
    existing_findings = state.get("existing_findings", [])
    messages = state.get("messages", [])
    stream = state.get("stream")
    chat_history = format_chat_history(messages, limit=len(messages))

    logger.info(
        "Researcher starting investigation",
        researcher_id=researcher_id,
        topic=topic,
    )

    if stream:
        stream.emit_research_start(researcher_id=researcher_id, topic=topic)

    try:
        max_results_per_query = max(3, math.ceil(max_sources / max(1, queries_per_round)))

        # Step 1: Generate initial search queries
        search_queries = await _generate_search_queries(
            llm,
            topic,
            memory_context,
            existing_findings,
            max_queries=queries_per_round,
            chat_history=chat_history,
        )
        all_queries: list[str] = []
        all_sources: list[SourceReference] = []

        for round_idx in range(max_query_rounds):
            if round_idx > 0:
                followups = await _generate_followup_queries(
                    llm,
                    topic,
                    all_queries,
                    all_sources,
                    memory_context,
                    existing_findings,
                    max_queries=queries_per_round,
                    chat_history=chat_history,
                )
                search_queries = _dedupe_queries(followups, all_queries)

            if not search_queries:
                break

            if stream:
                stream.emit_status(
                    f"Searching round {round_idx + 1} for {topic}...",
                    step="search",
                )
                stream.emit_search_queries(search_queries, label=researcher_id)

            all_queries.extend(search_queries)

            async def run_query(query: str):
                return await search_provider.search(query, max_results=max_results_per_query)

            search_batches = await asyncio.gather(*[run_query(query) for query in search_queries])
            for batch in search_batches:
                for result in batch.results[:max_results_per_query]:
                    all_sources.append(
                        SourceReference(
                            url=result.url,
                            title=result.title,
                            snippet=result.snippet,
                            relevance_score=result.score,
                        )
                    )
                    if stream:
                        stream.emit_source(
                            researcher_id=researcher_id,
                            source={"url": result.url, "title": result.title},
                        )

            unique_sources = _dedupe_sources(all_sources, per_domain_limit=2, max_sources=max_sources)
            if len(unique_sources) >= max_sources:
                break

        if not all_sources:
            unique_sources = []
        else:
            unique_sources = _dedupe_sources(all_sources, per_domain_limit=2, max_sources=max_sources)

        logger.info(
            "Sources collected",
            researcher_id=researcher_id,
            sources_count=len(unique_sources),
        )

        # Step 2: Scrape and analyze top sources
        scraped_content = []
        for source in unique_sources[:3]:  # Scrape top 3
            try:
                content = await web_scraper.scrape(source.url)
                scraped_content.append({
                    "url": content.url,
                    "title": content.title,
                    "content": summarize_text(content.content, 12000),
                })
            except Exception as e:
                logger.warning("Scraping failed", url=source.url, error=str(e))
                continue

        # Step 3: Analyze and synthesize findings
        analysis_result = await _analyze_and_synthesize(
            llm=llm,
            topic=topic,
            sources=unique_sources,
            scraped_content=scraped_content,
            memory_context=memory_context,
            existing_findings=existing_findings,
            chat_history=chat_history,
        )

        logger.info(
            "Research completed",
            researcher_id=researcher_id,
            key_findings_count=len(analysis_result["key_findings"]),
        )

        if stream:
            stream.emit_finding(
                researcher_id=researcher_id,
                topic=topic,
                summary=analysis_result["summary"],
                key_findings=analysis_result["key_findings"],
            )

        return {
            "search_queries": {"type": "override", "value": all_queries},
            "sources_found": {"type": "override", "value": unique_sources},
            "summary": {"type": "override", "value": analysis_result["summary"]},
            "key_findings": {"type": "override", "value": analysis_result["key_findings"]},
            "confidence_level": {"type": "override", "value": analysis_result["confidence"]},
            "completed": True,
        }

    except Exception as e:
        logger.error("Researcher node failed", researcher_id=researcher_id, error=str(e))
        return {
            "summary": {"type": "override", "value": f"Research failed: {str(e)}"},
            "key_findings": {"type": "override", "value": []},
            "confidence_level": {"type": "override", "value": "low"},
            "completed": True,
        }


async def _generate_search_queries(
    llm: any,
    topic: str,
    memory_context: list[Any],
    existing_findings: list[ResearchFinding],
    max_queries: int = 3,
    chat_history: str | None = None,
) -> list[str]:
    """Generate effective search queries for the research topic."""
    memory_hint = ""
    if memory_context:
        memory_hint = "\nMemory Context:\n" + "\n".join(
            [f"- {ctx.file_title}: {summarize_text(ctx.content, 6000)}" for ctx in memory_context[:3]]
        )

    existing_hint = _format_existing_findings(existing_findings)

    history_block = chat_history or "Chat history: None."
    current_date = get_current_date()
    prompt = f"""Research Topic: {topic}
Current date: {current_date}
{history_block}
{memory_hint}
{existing_hint}

Generate {max_queries} effective web search queries to investigate this topic.

Requirements:
- Queries should approach the topic from different angles
- Use specific keywords, not full sentences
- Vary the query types (definition, examples, comparison, current state, etc.)
- Avoid repeating topics already covered in the existing findings

Return JSON with fields reasoning, queries."""

    try:
        structured_llm = llm.with_structured_output(SearchQueries, method="function_calling")
        response = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        if not isinstance(response, SearchQueries):
            raise ValueError("SearchQueries response was not structured")
        queries = response.queries[:max_queries]
        return queries if queries else [topic]
    except Exception as e:
        logger.warning("Query generation failed, using topic as query", error=str(e))
        return [topic]


async def _analyze_and_synthesize(
    llm: any,
    topic: str,
    sources: list[SourceReference],
    scraped_content: list[dict],
    memory_context: list[Any],
    existing_findings: list[ResearchFinding],
    chat_history: str | None = None,
) -> dict:
    """Analyze sources and synthesize findings."""

    # Format sources for analysis
    sources_text = "\n\n".join([
        f"**Source {idx + 1}:** {source.title}\n"
        f"URL: {source.url}\n"
        f"Summary: {summarize_text(source.snippet, 6000)}"
        for idx, source in enumerate(sources[:5])
    ])

    # Format scraped content
    content_text = "\n\n".join([
        f"**{sc['title']}**\n{summarize_text(sc['content'], 12000)}"
        for sc in scraped_content[:3]
    ])

    memory_text = ""
    if memory_context:
        memory_text = "\n\n## Memory Context\n" + "\n".join(
            [f"- {ctx.file_title}: {summarize_text(ctx.content, 6000)}" for ctx in memory_context[:3]]
        )

    existing_text = _format_existing_findings(existing_findings)

    history_block = chat_history or "Chat history: None."
    current_date = get_current_date()
    prompt = f"""Research Topic: {topic}
Current date: {current_date}
{history_block}

## Sources Found
{sources_text}

## Detailed Content (from top sources)
{content_text}
{memory_text}
{existing_text}

Based on these sources, provide a comprehensive analysis:

1. **Summary** (2-3 paragraphs): Synthesize the key information about this topic
2. **Key Findings** (3-5 bullet points): The most important insights or discoveries
3. **Confidence Level**: low/medium/high based on source quality and consensus

Return JSON with fields reasoning, summary, key_findings, confidence."""

    try:
        structured_llm = llm.with_structured_output(ResearchAnalysis, method="function_calling")
        response = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        if not isinstance(response, ResearchAnalysis):
            raise ValueError("ResearchAnalysis response was not structured")
        return {
            "summary": response.summary,
            "key_findings": response.key_findings,
            "confidence": response.confidence,
        }
    except Exception as e:
        logger.error("Analysis failed", error=str(e))
        return {
            "summary": f"Analysis failed for {topic}",
            "key_findings": ["Analysis error occurred"],
            "confidence": "low",
        }


def _format_existing_findings(existing_findings: list[ResearchFinding]) -> str:
    if not existing_findings:
        return ""
    lines = ["Existing Findings (avoid duplication):"]
    for finding in existing_findings[:4]:
        topic = getattr(finding, "topic", "") or ""
        summary = getattr(finding, "summary", "") or ""
        summary = summary.replace("\n", " ")
        if topic or summary:
            lines.append(f"- {topic}: {summarize_text(summary, 6000)}")
    return "\n" + "\n".join(lines)


async def _generate_followup_queries(
    llm: any,
    topic: str,
    existing_queries: list[str],
    sources: list[SourceReference],
    memory_context: list[Any],
    existing_findings: list[ResearchFinding],
    max_queries: int = 3,
    chat_history: str | None = None,
) -> list[str]:
    if getattr(llm, "_llm_type", "") == "mock-chat":
        return []
    if not sources:
        return []

    memory_hint = ""
    if memory_context:
        memory_hint = "\nMemory Context:\n" + "\n".join(
            [f"- {ctx.file_title}: {summarize_text(ctx.content, 6000)}" for ctx in memory_context[:2]]
        )

    sources_hint = "\n".join(
        [f"- {source.title}: {summarize_text(source.snippet, 6000)}" for source in sources[:5]]
    )

    history_block = chat_history or "Chat history: None."
    current_date = get_current_date()
    prompt = f"""Research Topic: {topic}
Current date: {current_date}
{history_block}
Existing Queries: {', '.join(existing_queries[-6:])}
Sources Summary:
{sources_hint}
{memory_hint}
{_format_existing_findings(existing_findings)}

Identify any gaps or missing angles. Return JSON with fields reasoning, should_continue, gap_summary, queries.

Rules:
- Max {max_queries} queries
- Queries should be keyword phrases, not sentences
- Avoid duplicates of existing queries
- If coverage is sufficient, set should_continue false and return empty queries.
"""

    try:
        structured_llm = llm.with_structured_output(FollowupQueries, method="function_calling")
        response = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        if not isinstance(response, FollowupQueries):
            raise ValueError("FollowupQueries response was not structured")
        if not response.should_continue:
            return []
        return [q.strip() for q in response.queries if q.strip()][:max_queries]
    except Exception as exc:
        logger.warning("Followup query generation failed", error=str(exc))
        return []


def _dedupe_queries(queries: list[str], existing: list[str]) -> list[str]:
    existing_set = {item.lower() for item in existing}
    output = []
    for query in queries:
        normalized = query.lower().strip()
        if normalized and normalized not in existing_set:
            output.append(query)
            existing_set.add(normalized)
    return output


def _dedupe_sources(
    sources: list[SourceReference],
    per_domain_limit: int = 2,
    max_sources: int | None = None,
) -> list[SourceReference]:
    seen_urls = set()
    domain_counts: dict[str, int] = defaultdict(int)
    deduped: list[SourceReference] = []

    for source in sources:
        if source.url in seen_urls:
            continue
        domain = _normalize_domain(source.url)
        if domain:
            if domain_counts[domain] >= per_domain_limit:
                continue
            domain_counts[domain] += 1
        seen_urls.add(source.url)
        deduped.append(source)

        if max_sources and len(deduped) >= max_sources:
            break

    return deduped


def _normalize_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return ""
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc

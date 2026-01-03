"""Writer agent for final answer synthesis with citations.

Second stage of the Perplexica two-stage architecture.
Synthesizes research results into cited answers.
"""

import structlog
from typing import Any, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from src.workflow.search.classifier import get_current_date, format_chat_history

logger = structlog.get_logger(__name__)


# ==================== Schemas ====================


class CitedAnswer(BaseModel):
    """Answer with inline citations."""

    reasoning: str = Field(description="Why the sources support this answer")
    answer: str = Field(
        description="Final answer with inline citations [1], [2], etc. in markdown format"
    )
    citations: list[str] = Field(
        description="List of source URLs as strings: ['https://example.com', 'https://example2.com', ...]"
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="Confidence in answer based on source quality and coverage"
    )


# ==================== Writer Prompts ====================


def get_writer_prompt(mode: str) -> str:
    """Get mode-specific writer prompt."""

    current_date = get_current_date()

    base_prompt = f"""You are an expert writer synthesizing research into comprehensive answers.

Current date: {current_date}

Your role:
1. Read the research sources provided
2. Synthesize information into a clear, well-structured answer
3. CITE EVERY FACT with inline citations [1], [2], etc.
4. Provide a list of all sources at the end
5. Be accurate and truthful

CRITICAL CITATION RULES:
- Every factual statement MUST have a citation
- Use inline citations: "According to [1], ..."
- Multiple sources for important claims: "Studies show [1][2] that..."
- Never make claims without sources
- If sources conflict, mention both: "While [1] suggests X, [2] indicates Y"

Formatting:
- Use markdown: **bold**, *italic*, lists, headings
- Structure with clear sections (##, ###)
- Write like a knowledgeable blog post
- Be engaging but factual
"""

    if mode == "speed":
        return base_prompt + """
MODE: SPEED

Provide a complete, informative answer:
- 400-600 words minimum
- Focus on key points but cover them fully
- Use ALL provided sources - each source has valuable information
- Clear structure with sections
- Still cite everything!

IMPORTANT: Don't just summarize snippets - synthesize information from ALL sources into a comprehensive answer.
"""

    elif mode == "balanced":
        return base_prompt + """
MODE: BALANCED

Provide thorough, comprehensive coverage:
- 800-1200 words minimum
- Well-organized sections with clear structure
- Use ALL provided sources - synthesize information from each
- Cover main aspects of the topic in depth
- Include specific details, data, and examples from sources
- Compare different perspectives if sources provide them

IMPORTANT: You have many sources available - use them all! Don't just pick a few.
Each source adds value - synthesize them into a complete picture.
"""

    else:  # quality
        return base_prompt + """
MODE: QUALITY

Create an in-depth, comprehensive response:
- Minimum 1500-3000 words
- Multiple sections with clear structure (Introduction, Main sections, Conclusion)
- Cover the topic from ALL angles
- Include background, detailed analysis, implications, examples
- Like a comprehensive research report or detailed article
- Deep analysis with specific data and evidence from sources

CRITICAL: Use EVERY source provided! You have extensive research - leverage all of it!
- Synthesize information from all sources into a coherent narrative
- Include specific quotes, data, and facts from sources
- Compare and contrast different perspectives
- Build a complete, authoritative answer
"""


# ==================== Writer Agent ====================


async def writer_agent(
    query: str,
    research_results: dict[str, Any],
    llm: Any,
    stream: Any,
    mode: str = "balanced",
    chat_history: list[dict] = None,
) -> str:
    """
    Synthesize final answer with citations from research.

    Args:
        query: Original user query
        research_results: Results from research_agent (sources, scraped_content, reasoning)
        llm: LLM instance
        stream: Streaming generator
        mode: Research mode (affects answer length/depth)
        chat_history: Chat history for context

    Returns:
        Formatted answer with inline citations and sources section
    """
    chat_history = chat_history or []

    sources = research_results.get("sources", [])
    scraped = research_results.get("scraped_content", [])

    if not sources and not scraped:
        logger.warning("No sources available for writer agent")
        return "I couldn't find enough information to answer this question reliably."

    # Prepare citation-ready sources
    all_sources = []

    # Add search results (use full snippet - it's already short from search engine)
    for source in sources:
        all_sources.append({
            "title": source.get("title", "Untitled"),
            "url": source.get("url", ""),
            "content": source.get("snippet", "")  # Full snippet, no truncation
        })

    # Add scraped content (richer information with summaries!)
    for scraped_item in scraped:
        if "error" not in scraped_item:
            # Prefer summary (comprehensive 800-token context) over content
            # Content field already contains summary from scrape_url_handler
            content = scraped_item.get("content", "")

            all_sources.append({
                "title": scraped_item.get("title", "Untitled"),
                "url": scraped_item.get("url", ""),
                "content": content  # Already summarized by LLM in scrape_url_handler
            })

    # Deduplicate by URL
    seen_urls = set()
    unique_sources = []
    for source in all_sources:
        url = source.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_sources.append(source)

    # Use all available sources - don't limit artificially
    # LLM can handle context and decide which to use
    # In speed mode: typically 3-5 sources
    # In balanced mode: typically 8-12 sources
    # In quality mode: typically 15-20 sources
    logger.info(f"Writer synthesizing from {len(unique_sources)} sources")

    # Build source context for LLM
    source_context = "\n\n".join([
        f"[{i+1}] **{src['title']}** ({src['url']})\n{src['content']}"
        for i, src in enumerate(unique_sources)
    ])

    # Get writer prompt
    system_prompt = get_writer_prompt(mode)

    # Build user prompt
    sources_count = len(unique_sources)
    user_prompt = f"""Query: {query}

Research sources ({sources_count} sources provided - USE ALL OF THEM):
{source_context}

Write a comprehensive answer with inline citations [1], [2], etc.

IMPORTANT INSTRUCTIONS:
- Use information from ALL {sources_count} sources - each one has valuable content
- Synthesize information from all sources into a coherent, complete answer
- Don't just use the first few sources - leverage ALL available research
- Include specific details, data, and examples from different sources
- If sources provide different perspectives, present them all with citations

Remember: CITE EVERY FACT. Return JSON with: reasoning, answer, citations (list of URL strings), confidence.
CRITICAL: citations must be a list of URL strings, e.g. ["https://example.com", "https://example2.com"]
"""

    try:
        if stream:
            stream.emit_status("Synthesizing answer...", step="synthesis")

        # Use structured output
        structured_llm = llm.with_structured_output(CitedAnswer, method="function_calling")

        result = await structured_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        # Format final answer
        final_answer = result.answer

        # Add sources section
        final_answer += "\n\n## Sources\n\n"
        for i, source in enumerate(unique_sources):
            final_answer += f"[{i+1}] [{source['title']}]({source['url']})\n"

        # Add confidence indicator (optional, for debugging)
        # final_answer += f"\n\n*Confidence: {result.confidence}*"

        logger.info(
            "Answer synthesized",
            length=len(final_answer),
            sources=len(unique_sources),
            confidence=result.confidence
        )

        return final_answer

    except Exception as e:
        logger.error("Writer agent failed", error=str(e), exc_info=True)

        # Fallback: Simple source listing
        fallback = f"Based on the research:\n\n{source_context}\n\n## Sources\n\n"
        for i, source in enumerate(unique_sources):
            fallback += f"[{i+1}] {source['title']} - {source['url']}\n"

        return fallback

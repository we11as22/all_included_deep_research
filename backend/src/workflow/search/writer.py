"""Writer agent for final answer synthesis with citations.

Second stage of the Perplexica two-stage architecture.
Synthesizes research results into cited answers.
"""

import structlog
from typing import Any, Dict, List, Literal
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
    citations: List[Dict[str, str]] = Field(
        description="List of sources: [{number: '1', url: '...', title: '...'}]"
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

Keep it concise but informative:
- 200-400 words
- Focus on key points
- Quick, clear answer
- Still cite everything!
"""

    elif mode == "balanced":
        return base_prompt + """
MODE: BALANCED

Provide thorough coverage:
- 500-800 words
- Well-organized sections
- Comprehensive but not exhaustive
- Cover main aspects of the topic
"""

    else:  # quality
        return base_prompt + """
MODE: QUALITY

Create an in-depth, comprehensive response:
- Minimum 1000-2000 words
- Multiple sections with clear structure
- Cover the topic from all angles
- Include background, details, implications
- Like a research report or detailed article
- Deep analysis, not just summary
"""


# ==================== Writer Agent ====================


async def writer_agent(
    query: str,
    research_results: Dict[str, Any],
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

    # Add search results
    for source in sources:
        all_sources.append({
            "title": source.get("title", "Untitled"),
            "url": source.get("url", ""),
            "content": source.get("snippet", "")[:500]
        })

    # Add scraped content (richer information)
    for scraped_item in scraped:
        if "error" not in scraped_item:
            all_sources.append({
                "title": scraped_item.get("title", "Untitled"),
                "url": scraped_item.get("url", ""),
                "content": scraped_item.get("content", "")[:1500]  # More context from scraped
            })

    # Deduplicate by URL
    seen_urls = set()
    unique_sources = []
    for source in all_sources:
        url = source.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_sources.append(source)

    # Limit sources (top 10 for writer context)
    unique_sources = unique_sources[:10]

    logger.info(f"Writer synthesizing from {len(unique_sources)} sources")

    # Build source context for LLM
    source_context = "\n\n".join([
        f"[{i+1}] **{src['title']}** ({src['url']})\n{src['content']}"
        for i, src in enumerate(unique_sources)
    ])

    # Get writer prompt
    system_prompt = get_writer_prompt(mode)

    # Build user prompt
    user_prompt = f"""Query: {query}

Research sources:
{source_context}

Write a comprehensive answer with inline citations [1], [2], etc.
Remember: CITE EVERY FACT. Return JSON with: reasoning, answer, citations (list), confidence.
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

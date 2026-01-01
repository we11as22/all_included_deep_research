"""Query classifier for intelligent routing to appropriate search mode.

Based on Perplexica's classification pattern.
Determines query type, reformulates for standalone context, and suggests mode.
"""

from datetime import datetime
from typing import Literal

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# ==================== Schema ====================


class QueryClassification(BaseModel):
    """Classification result with routing decision."""

    reasoning: str = Field(description="Why this classification was chosen")
    query_type: Literal["simple", "research", "factual", "opinion", "comparison", "news"] = Field(
        description="Type of query based on intent"
    )
    standalone_query: str = Field(
        description="Rewritten query that stands alone without chat context"
    )
    suggested_mode: Literal[
        "chat", "web", "deep", "research_speed", "research_balanced", "research_quality"
    ] = Field(description="Suggested mode for answering")
    requires_sources: bool = Field(description="Whether query requires web sources")
    time_sensitive: bool = Field(
        description="Whether query is time-sensitive (news, recent events)"
    )


# ==================== Classifier Implementation ====================


async def classify_query(
    query: str,
    chat_history: list[dict[str, str]],
    llm: any,
) -> QueryClassification:
    """
    Classify user query and determine routing.

    Args:
        query: User's current question
        chat_history: Previous messages [{"role": "user|assistant", "content": "..."}]
        llm: LLM instance with structured output support

    Returns:
        QueryClassification with routing decision
    """
    current_date = datetime.now().strftime("%Y-%m-%d")

    system_prompt = f"""You are a query classifier that routes questions to the appropriate search mode.

Current date: {current_date}

Your task:
1. Analyze the user's query in context of chat history
2. Classify the query type:
   - simple: Greetings, general knowledge, math, definitions (no sources needed)
   - research: Deep investigation, comprehensive analysis (needs extensive sources)
   - factual: Specific facts, statistics, current information (needs verification)
   - opinion: Subjective questions, recommendations, comparisons
   - comparison: Comparing options, alternatives, pros/cons
   - news: Recent events, current news, trending topics

3. Rewrite query as standalone:
   - Resolve pronouns and references from chat history
   - Add necessary context
   - Make it self-contained
   - Keep original intent

4. Suggest appropriate mode:
   - chat: No sources needed, simple Q&A
   - web: Quick web search (2 iterations)
   - deep: Iterative deep search (6 iterations)
   - research_speed: Fast multi-agent research
   - research_balanced: Balanced multi-agent research
   - research_quality: Comprehensive multi-agent research

5. Determine if sources are required
6. Check if query is time-sensitive

Return JSON with: reasoning, query_type, standalone_query, suggested_mode, requires_sources, time_sensitive

Rules:
- ALWAYS use 'web' or higher if user asks about current events, news, or anything after 2024
- Use 'research_*' only for complex, multi-faceted questions requiring deep analysis
- Use 'chat' only for greetings, simple math, or general knowledge from training data
- Standalone query MUST be grammatically correct and understandable without context
"""

    # Format chat history for context
    history_lines = []
    for msg in chat_history[-6:]:  # Last 6 messages for context
        role = msg.get("role", "user")
        content = msg.get("content", "")
        history_lines.append(f"{role.capitalize()}: {content}")

    history_block = "\n".join(history_lines) if history_lines else "(No previous context)"

    user_prompt = f"""Chat history:
{history_block}

Current query: {query}

Classify this query and provide routing decision."""

    try:
        # Use structured output
        structured_llm = llm.with_structured_output(
            QueryClassification, method="function_calling"
        )

        result = await structured_llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )

        logger.info(
            "Query classified",
            query=query[:100],
            type=result.query_type,
            mode=result.suggested_mode,
            requires_sources=result.requires_sources,
            standalone=result.standalone_query[:100],
        )

        return result

    except Exception as e:
        logger.error("Classification failed, using fallback", error=str(e))

        # Fallback to safe defaults
        return QueryClassification(
            reasoning="Classification failed, using safe defaults",
            query_type="factual",
            standalone_query=query,
            suggested_mode="web",
            requires_sources=True,
            time_sensitive=False,
        )


# ==================== Helper Functions ====================


def get_current_date() -> str:
    """Get current date in human-readable format."""
    return datetime.now().strftime("%B %d, %Y")


def format_chat_history(
    chat_history: list[dict[str, str]], limit: int = 6
) -> str:
    """Format chat history for prompts."""
    if not chat_history:
        return "(No previous conversation)"

    lines = []
    for msg in chat_history[-limit:]:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")

    return "\n".join(lines)

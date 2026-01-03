"""Memory search node for retrieving relevant context."""

import structlog

from src.memory.hybrid_search import HybridSearchEngine
from src.workflow.state import MemoryContext, ResearchState

logger = structlog.get_logger(__name__)


async def search_memory_node(
    state: ResearchState,
    search_engine: HybridSearchEngine,
    max_results: int = 10,
) -> dict:
    """
    Memory search is disabled. Return empty context.

    Args:
        state: Current research state
        search_engine: HybridSearchEngine instance
        max_results: Maximum memory results to retrieve

    Returns:
        State update with memory_context
    """
    _ = (state, search_engine, max_results)
    logger.info("Memory search disabled; returning empty context")
    return {
        "memory_context": {
            "type": "override",
            "value": [],
        }
    }


def format_memory_context_for_prompt(memory_context: list[MemoryContext]) -> str:
    """
    Format memory context for inclusion in LLM prompts.

    Args:
        memory_context: List of memory context items

    Returns:
        Formatted string for prompt
    """
    if not memory_context:
        return "No memory context."

    formatted = "## Relevant Memory Context\n\n"
    formatted += "The following information from past research may be relevant:\n\n"

    for idx, ctx in enumerate(memory_context, 1):
        formatted += f"### Source {idx}: {ctx.file_title}\n"
        formatted += f"**File:** {ctx.file_path}\n"

        if ctx.header_path:
            formatted += f"**Section:** {' > '.join(ctx.header_path)}\n"

        formatted += f"**Relevance Score:** {ctx.score:.3f}\n\n"
        formatted += f"{ctx.content}\n\n"
        formatted += "---\n\n"

    return formatted.strip()

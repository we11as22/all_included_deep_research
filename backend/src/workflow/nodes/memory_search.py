"""Memory search node for retrieving relevant context."""

import structlog

from src.memory.hybrid_search import HybridSearchEngine
from src.memory.models.search import SearchMode
from src.workflow.state import MemoryContext, ResearchState

logger = structlog.get_logger(__name__)


async def search_memory_node(
    state: ResearchState,
    search_engine: HybridSearchEngine,
    max_results: int = 10,
) -> dict:
    """
    Search memory for relevant context before research.

    This node runs early in the workflow to retrieve any existing
    knowledge that might be relevant to the research query.

    Args:
        state: Current research state
        search_engine: HybridSearchEngine instance
        max_results: Maximum memory results to retrieve

    Returns:
        State update with memory_context
    """
    query = state.get("clarified_query") or state.get("query", "")
    # Ensure query is a string, not a list/embedding
    # CRITICAL: SQL queries expect string, not vector/list - use empty string if query is not a string
    if not isinstance(query, str):
        if isinstance(query, (list, tuple)):
            logger.error("Query was passed as list/embedding in state! This is invalid. Using empty string.", query_type=type(query).__name__, query_length=len(query) if hasattr(query, '__len__') else 'unknown')
            query = ""  # Use empty string instead of str(query) to avoid passing vector as string
        else:
            logger.warning("Query was not a string, converting", query_type=type(query).__name__)
            query = str(query) if query else ""
    
    # Final validation - query must be a string
    if not isinstance(query, str):
        logger.error("Query is still not a string after conversion! Using empty string.", query_type=type(query).__name__)
        query = ""
    
    mode = state.get("mode", "balanced")
    stream = state.get("stream")

    logger.info("Searching memory", query=query[:100] if isinstance(query, str) else "non-string", mode=mode)

    try:
        # Adjust search depth based on mode
        if mode == "speed":
            search_limit = min(max_results, 3)  # Quick memory check
            search_mode = SearchMode.VECTOR  # Fast vector-only search
        elif mode == "balanced":
            search_limit = max_results
            search_mode = SearchMode.HYBRID  # Balanced hybrid search
        else:  # quality
            search_limit = max_results * 2  # Deep memory search
            search_mode = SearchMode.HYBRID

        if stream:
            stream.emit_status("Searching memory...", step="memory_search")

        # Final validation before search - query MUST be a string
        if not isinstance(query, str):
            logger.error("Query is not a string before search execution! Using empty string.", query_type=type(query).__name__)
            query = ""

        # Execute hybrid search
        search_results = await search_engine.search(
            query=query,
            search_mode=search_mode,
            limit=search_limit,
        )

        filtered_results = [result for result in search_results if result.file_category != "chat"]

        # Convert to MemoryContext objects
        memory_context = [
            MemoryContext(
                chunk_id=result.chunk_id,
                file_path=result.file_path,
                file_title=result.file_title,
                content=result.content,
                score=result.score,
                header_path=result.header_path,
            )
            for result in filtered_results
        ]

        logger.info(
            "Memory search completed",
            results_count=len(memory_context),
            mode=mode,
        )

        if stream:
            stream.emit_memory_context(
                [
                    {
                        "file_path": ctx.file_path,
                        "title": ctx.file_title,
                        "score": ctx.score,
                    }
                    for ctx in memory_context
                ]
            )

        # Return override update for memory_context
        return {
            "memory_context": {
                "type": "override",
                "value": memory_context,
            }
        }

    except Exception as e:
        logger.error("Memory search failed", error=str(e))
        # Return empty context on failure
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
        return "No relevant memory context found."

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

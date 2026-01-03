"""Action Registry for research agent actions.

Based on Perplexica's action registry pattern.
Provides tool definitions and execution handlers for research agents.
"""

import asyncio
import structlog
from typing import Any, Callable

from src.utils.text import summarize_text_llm

logger = structlog.get_logger(__name__)


# ==================== Action Registry ====================


class ActionRegistry:
    """Registry of available research actions (Perplexica pattern)."""

    _actions: dict[str, dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        description: str,
        args_schema: dict[str, Any],
        handler: Callable | None = None,
        enabled_condition: Callable | None = None,
    ):
        """
        Register an action.

        Args:
            name: Action name (e.g., "web_search")
            description: Human-readable description for LLM
            args_schema: JSON schema for arguments
            handler: Async function to execute action
            enabled_condition: Function to check if action is enabled for current context
        """
        cls._actions[name] = {
            "name": name,
            "description": description,
            "args_schema": args_schema,
            "handler": handler,
            "enabled_condition": enabled_condition or (lambda ctx: True),
        }
        logger.debug(f"Registered action: {name}")

    @classmethod
    def get_tool_definitions(
        cls, mode: str, classification: str | None = None, context: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Get tool definitions for LLM tool-calling.

        Args:
            mode: Research mode (speed, balanced, quality)
            classification: Query classification type
            context: Additional context for filtering

        Returns:
            List of tool definitions for LLM
        """
        context = context or {}
        tools = []

        for action_name, action_def in cls._actions.items():
            # Check if action is enabled for this context
            enabled = action_def["enabled_condition"](
                {"mode": mode, "classification": classification, **context}
            )

            if enabled:
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": action_name,
                            "description": action_def["description"],
                            "parameters": action_def["args_schema"],
                        },
                    }
                )

        logger.debug(f"Generated {len(tools)} tool definitions for mode={mode}")
        return tools

    @classmethod
    async def execute(
        cls, action: str, args: dict[str, Any], context: dict[str, Any]
    ) -> Any:
        """
        Execute a registered action.

        Args:
            action: Action name
            args: Action arguments
            context: Execution context (search_provider, scraper, stream, etc.)

        Returns:
            Action result
        """
        if action not in cls._actions:
            raise ValueError(f"Unknown action: {action}")

        handler = cls._actions[action]["handler"]
        if not handler:
            raise ValueError(f"No handler registered for action: {action}")

        logger.debug(f"Executing action: {action}", args=args)

        try:
            result = await handler(args, context)
            logger.debug(f"Action completed: {action}", result_type=type(result).__name__)
            return result
        except Exception as e:
            logger.error(f"Action failed: {action}", error=str(e), exc_info=True)
            return {"error": str(e), "action": action}


# ==================== Action Handlers ====================


async def web_search_handler(args: dict[str, Any], context: dict[str, Any]) -> dict:
    """Execute web search action with parallel queries."""
    queries = args.get("queries", [])
    max_results = args.get("max_results", 5)

    search_provider = context.get("search_provider")
    stream = context.get("stream")

    if not search_provider:
        return {"error": "Search provider not available"}

    # Emit all queries at once
    if stream:
        for query in queries[:3]:
            stream.emit_status(f"Searching: {query}", step="search")

    # Execute all searches in parallel
    async def search_single(query: str) -> list[dict]:
        try:
            # Validate query
            if not query or not isinstance(query, str):
                logger.warning(f"Invalid query: {query}", query_type=type(query).__name__)
                return []
            
            # Perform search
            response = await search_provider.search(query, max_results=max_results)
            results = response.results if hasattr(response, "results") else []

            # Format results (like Perplexica: title, url, content/snippet)
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.title if hasattr(result, "title") else "",
                    "url": result.url if hasattr(result, "url") else "",
                    "snippet": result.content if hasattr(result, "content") else (result.snippet if hasattr(result, "snippet") else ""),
                })

            logger.info(f"Search query completed: {query}", results=len(formatted_results))
            return formatted_results

        except Exception as e:
            logger.error(f"Search query failed: {query}", error=str(e), exc_info=True)
            return []

    # Validate queries
    if not queries:
        logger.warning("No queries provided to web_search")
        return {"results": [], "count": 0}
    
    # Limit to 3 queries (like Perplexica)
    queries_to_search = queries[:3]
    
    # Run all searches in parallel (like Perplexica) with error handling
    search_results = await asyncio.gather(*[search_single(q) for q in queries_to_search], return_exceptions=True)
    
    # Flatten results and filter out exceptions
    all_results = []
    for i, result in enumerate(search_results):
        if isinstance(result, Exception):
            logger.error(f"Search query {i} failed with exception", query=queries_to_search[i] if i < len(queries_to_search) else "unknown", error=str(result))
            continue
        if isinstance(result, list):
            all_results.extend(result)
        else:
            logger.warning(f"Unexpected result type from search_single", result_type=type(result).__name__)

    logger.info(f"Web search completed", queries_count=len(queries_to_search), results_count=len(all_results))
    return {"results": all_results, "count": len(all_results)}


async def scrape_url_handler(args: dict[str, Any], context: dict[str, Any]) -> dict:
    """Execute URL scraping action with parallel processing and summarization."""
    urls = args.get("urls", [])

    scraper = context.get("scraper")
    stream = context.get("stream")
    llm = context.get("llm")

    if not scraper:
        return {"error": "Scraper not available"}

    # Emit scraping status for all URLs
    if stream:
        for url in urls[:3]:
            stream.emit_status(f"Scraping: {url[:50]}...", step="scrape")

    # Scrape and summarize in parallel
    async def scrape_and_summarize(url: str) -> dict:
        try:
            # Step 1: Scrape URL
            content = await scraper.scrape(url)

            # Extract full content
            full_content = content.content if hasattr(content, "content") else ""
            title = content.title if hasattr(content, "title") else ""

            # Step 2: Summarize content in parallel with other URLs
            summary = ""
            if llm and full_content:
                try:
                    if stream:
                        stream.emit_status(f"Summarizing: {title[:40]}...", step="summarize")

                    summary = await summarize_text_llm(
                        full_content,
                        max_tokens=800,  # Comprehensive summary
                        llm=llm
                    )
                    logger.debug(f"Content summarized: {url}", summary_length=len(summary))
                except Exception as e:
                    logger.warning(f"Summarization failed: {url}", error=str(e))
                    # Fallback to smart truncation (not hard cut)
                    from src.utils.text import summarize_text
                    summary = summarize_text(full_content, 3200)  # ~800 tokens

            # If no summary and no LLM, use smart truncation
            if not summary and full_content:
                from src.utils.text import summarize_text
                summary = summarize_text(full_content, 3200)

            logger.debug(f"URL scraped and summarized: {url}")

            return {
                "url": url,
                "title": title,
                "content": summary,  # Always use summary (LLM or smart truncation)
                "summary": summary,  # Full context for writer!
            }

        except Exception as e:
            error_msg = str(e) if e else "Unknown scraping error"
            error_type = type(e).__name__ if e else "UnknownError"
            logger.warning(
                "URL scraping failed",
                url=url,
                error=error_msg,
                error_type=error_type
            )
            return {
                "url": url,
                "error": error_msg,
                "error_type": error_type
            }

    # Process all URLs in parallel (continue even if some fail)
    scraped_results = await asyncio.gather(
        *[scrape_and_summarize(url) for url in urls[:3]],
        return_exceptions=True
    )
    
    # Filter out exceptions and failed results, keep only successful ones
    successful_results = []
    for result in scraped_results:
        if isinstance(result, Exception):
            # Skip exceptions
            logger.debug("Filtered out exception from scrape results", error_type=type(result).__name__)
            continue
        if isinstance(result, dict) and result.get("error") is not None:
            # Skip results with error field
            logger.debug("Filtered out failed scrape result", url=result.get("url"), error=result.get("error"))
            continue
        # Keep successful results (dict without "error" field)
        successful_results.append(result)

    logger.info(
        "Scraping completed",
        total_urls=len(urls[:3]),
        successful=len(successful_results),
        failed=len(scraped_results) - len(successful_results)
    )

    return {"scraped": successful_results, "count": len(successful_results)}


async def done_handler(args: dict[str, Any], context: dict[str, Any]) -> dict:
    """Signal research completion."""
    summary = args.get("summary", "Research completed")

    stream = context.get("stream")
    if stream:
        stream.emit_status(summary, step="done")

    return {"done": True, "summary": summary}


async def reasoning_preamble_handler(args: dict[str, Any], context: dict[str, Any]) -> dict:
    """Handle reasoning preamble (chain-of-thought)."""
    reasoning = args.get("reasoning", "")

    stream = context.get("stream")
    # TODO: Add emit_agent_reasoning method to streaming
    # if stream:
    #     stream.emit_agent_reasoning(context.get("agent_id", "researcher"), reasoning)

    return {"reasoning": reasoning}


# ==================== Action Registration ====================


def register_actions():
    """Register all available actions."""

    # Web Search (always available)
    ActionRegistry.register(
        name="web_search",
        description="Search the web for information. Provide up to 3 targeted search queries. "
        "CRITICAL: When generating queries, ALWAYS preserve the core meaning and key terms from the original user query. "
        "Use the SAME LANGUAGE as the original query. Generate SEO-friendly keywords (2-5 words), NOT full sentences. "
        "**IMPORTANT**: If the user query contains a word that might be a typo (e.g., 'гинеса' -> 'Гиннесса'), intelligently correct it. "
        "Example: User asks 'расскажи про историю гинеса' -> queries should be ['Гиннесса история', 'Книга рекордов Гиннесса', 'Guinness World Records']. "
        "Example: User asks 'расскажи про саблю' -> queries should be ['сабля', 'сабля история', 'сабля виды']. "
        "DO NOT add unrelated terms. DO NOT drop key terms from the original query. Returns list of search results with title, URL, and snippet.",
        args_schema={
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of 1-3 search queries (SEO-friendly keywords, preserve original query's key terms and language)",
                    "minItems": 1,
                    "maxItems": 3,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results per query",
                    "default": 5,
                },
            },
            "required": ["queries"],
        },
        handler=web_search_handler,
    )

    # Scrape URLs
    ActionRegistry.register(
        name="scrape_url",
        description="Scrape full content from specific URLs. Use when user provides URLs or "
        "you need full article text. Returns scraped content.",
        args_schema={
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of 1-3 URLs to scrape",
                    "minItems": 1,
                    "maxItems": 3,
                },
            },
            "required": ["urls"],
        },
        handler=scrape_url_handler,
    )

    # Reasoning Preamble (for balanced/quality modes)
    ActionRegistry.register(
        name="__reasoning_preamble",
        description="MANDATORY: Start every response with your chain-of-thought reasoning. "
        "Explain what you've learned, what gaps remain, and what action you'll take next. "
        "This is NOT an action - it's your thinking process before calling tools.",
        args_schema={
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Your chain-of-thought reasoning in natural language. "
                    "Start with intent: 'Okay, the user wants to...'",
                },
            },
            "required": ["reasoning"],
        },
        handler=reasoning_preamble_handler,
        enabled_condition=lambda ctx: ctx.get("mode") in ["balanced", "quality"],
    )

    # Done (signal completion)
    ActionRegistry.register(
        name="done",
        description="Signal that research is complete and you have gathered sufficient information. "
        "Provide a brief summary of what was found.",
        args_schema={
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief summary of research findings",
                },
            },
            "required": ["summary"],
        },
        handler=done_handler,
    )

    logger.info(f"Registered {len(ActionRegistry._actions)} actions")


# Initialize actions on module import
register_actions()

"""Action Registry for research agent actions.

Based on Perplexica's action registry pattern.
Provides tool definitions and execution handlers for research agents.
"""

import asyncio
import structlog
from typing import Any, Callable, Dict, List

logger = structlog.get_logger(__name__)


# ==================== Action Registry ====================


class ActionRegistry:
    """Registry of available research actions (Perplexica pattern)."""

    _actions: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        description: str,
        args_schema: Dict[str, Any],
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
        cls, mode: str, classification: str | None = None, context: Dict | None = None
    ) -> List[Dict[str, Any]]:
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
        cls, action: str, args: Dict[str, Any], context: Dict[str, Any]
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


async def web_search_handler(args: Dict[str, Any], context: Dict[str, Any]) -> Dict:
    """Execute web search action."""
    queries = args.get("queries", [])
    max_results = args.get("max_results", 5)

    search_provider = context.get("search_provider")
    stream = context.get("stream")

    if not search_provider:
        return {"error": "Search provider not available"}

    all_results = []
    for query in queries[:3]:  # Limit to 3 queries
        try:
            if stream:
                stream.emit_status(f"Searching: {query}", step="search")

            response = await search_provider.search(query, max_results=max_results)
            results = response.results if hasattr(response, "results") else []

            for result in results:
                all_results.append(
                    {
                        "title": result.title if hasattr(result, "title") else "",
                        "url": result.url if hasattr(result, "url") else "",
                        "snippet": result.content if hasattr(result, "content") else "",
                    }
                )

            logger.debug(f"Search query completed: {query}", results=len(results))

        except Exception as e:
            logger.warning(f"Search query failed: {query}", error=str(e))
            continue

    return {"results": all_results, "count": len(all_results)}


async def scrape_url_handler(args: Dict[str, Any], context: Dict[str, Any]) -> Dict:
    """Execute URL scraping action."""
    urls = args.get("urls", [])

    scraper = context.get("scraper")
    stream = context.get("stream")

    if not scraper:
        return {"error": "Scraper not available"}

    scraped = []
    for url in urls[:3]:  # Limit to 3 URLs
        try:
            if stream:
                stream.emit_status(f"Scraping: {url[:50]}...", step="scrape")

            content = await scraper.scrape(url)

            scraped.append(
                {
                    "url": url,
                    "title": content.title if hasattr(content, "title") else "",
                    "content": content.content[:2000] if hasattr(content, "content") else "",
                }
            )

            logger.debug(f"URL scraped: {url}")

        except Exception as e:
            logger.warning(f"URL scraping failed: {url}", error=str(e))
            scraped.append({"url": url, "error": str(e)})

    return {"scraped": scraped, "count": len(scraped)}


async def done_handler(args: Dict[str, Any], context: Dict[str, Any]) -> Dict:
    """Signal research completion."""
    summary = args.get("summary", "Research completed")

    stream = context.get("stream")
    if stream:
        stream.emit_status(summary, step="done")

    return {"done": True, "summary": summary}


async def reasoning_preamble_handler(args: Dict[str, Any], context: Dict[str, Any]) -> Dict:
    """Handle reasoning preamble (chain-of-thought)."""
    reasoning = args.get("reasoning", "")

    stream = context.get("stream")
    if stream:
        stream.emit_agent_reasoning(context.get("agent_id", "researcher"), reasoning)

    return {"reasoning": reasoning}


# ==================== Action Registration ====================


def register_actions():
    """Register all available actions."""

    # Web Search (always available)
    ActionRegistry.register(
        name="web_search",
        description="Search the web for information. Provide up to 3 targeted search queries. "
        "Returns list of search results with title, URL, and snippet.",
        args_schema={
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of 1-3 search queries (SEO-friendly keywords)",
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

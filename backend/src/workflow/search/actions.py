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
    # CRITICAL: Increase default max_results for better coverage
    # 5 results per query is too few - increase to 10 for balanced/quality modes
    mode = context.get("mode", "speed")
    default_max_results = 10 if mode in ["balanced", "quality"] else 5
    max_results = args.get("max_results", default_max_results)

    search_provider = context.get("search_provider")
    stream = context.get("stream")
    original_query = context.get("original_query")  # Get original user query from context

    if not search_provider:
        return {"error": "Search provider not available"}

    # CRITICAL: Add original user query to search queries if not already present
    # This ensures we always search for the original query even if LLM generates different queries
    if original_query and original_query not in queries:
        # Add original query at the beginning to prioritize it
        queries = [original_query] + queries
        logger.info(f"Added original_query to search queries", original_query=original_query[:100], total_queries=len(queries))
    
    # Limit to 3 queries total (including original if added)
    queries = queries[:3]

    # Log queries for debugging relevance
    logger.info(f"web_search_handler received queries", queries=queries, queries_count=len(queries), original_query=original_query[:100] if original_query else None)

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

            # Extract content - prefer markdown if available, fallback to plain text
            # Step 1: Try markdown first (better structure for LLM)
            full_content = None
            if hasattr(content, "markdown") and content.markdown:
                full_content = content.markdown
                logger.debug(f"Using markdown content for summarization", url=url)
            
            # Step 2: Fallback to plain text content
            if not full_content:
                full_content = content.content if hasattr(content, "content") else ""
                logger.debug(f"Using plain text content for summarization", url=url)
            
            title = content.title if hasattr(content, "title") else ""

            # CRITICAL: Prefer markdown for summarization if available (better structure)
            content_to_summarize = None
            if hasattr(content, "markdown") and content.markdown:
                content_to_summarize = content.markdown
                logger.debug(f"Using markdown for summarization", url=url, markdown_length=len(content.markdown))
            elif full_content:
                content_to_summarize = full_content
                logger.debug(f"Using plain text content for summarization", url=url, content_length=len(full_content))

            # Step 2: Summarize content in parallel with other URLs
            summary = ""
            if llm and content_to_summarize:
                try:
                    if stream:
                        stream.emit_status(f"Summarizing: {title[:40]}...", step="summarize")

                    # CRITICAL: Log max_tokens to verify it's correct
                    max_tokens_value = None
                    if hasattr(llm, "max_tokens"):
                        max_tokens_value = llm.max_tokens
                    logger.debug(
                        "Scraping and summarizing URL",
                        url=url,
                        content_length=len(content_to_summarize),
                        llm_max_tokens=max_tokens_value,
                        summary_target_tokens=4096,
                        using_markdown=hasattr(content, "markdown") and content.markdown is not None,
                    )

                    summary = await summarize_text_llm(
                        content_to_summarize,
                        max_tokens=4096,  # Comprehensive summary (increased) - this is target summary length, not LLM max_tokens
                        llm=llm
                    )
                    logger.debug(f"Content summarized: {url}", summary_length=len(summary))
                except Exception as e:
                    logger.warning(f"Summarization failed: {url}", error=str(e))
                    # Fallback to smart truncation (not hard cut)
                    from src.utils.text import summarize_text
                    summary = summarize_text(content_to_summarize, 3200) if content_to_summarize else ""  # ~800 tokens

            # If no summary and no LLM, use smart truncation
            if not summary and content_to_summarize:
                from src.utils.text import summarize_text
                summary = summarize_text(content_to_summarize, 3200)

            logger.debug(f"URL scraped and summarized: {url}", summary_length=len(summary))

            # Return only url, title, summary (summary from markdown if available, otherwise from content)
            return {
                "url": url,
                "title": title,
                "summary": summary,  # Summary from markdown (if available) or content
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


async def save_note_handler(args: dict[str, Any], context: dict[str, Any]) -> dict:
    """Save a research note with title, summary, and optional URLs.
    
    Use this tool to save important findings, discoveries, insights, or information
    that you've gathered during research. Notes are stored with vector search
    and can be retrieved by you and other agents for future reference.
    
    CRITICAL: Only save notes when you have SUBSTANTIAL, ACTIONABLE INFORMATION:
    - Key discoveries, important facts, or significant insights
    - Critical information that directly relates to your current task
    - Important patterns, trends, or conclusions
    - Technical details, specifications, or data points
    - Expert opinions, analysis, or perspectives
    - Historical context, evolution, or development
    - Real-world examples, case studies, or applications
    
    DO NOT save routine notes like "Found X sources" or "Search: query".
    When you DO save a note, it MUST be LARGE and DETAILED (minimum 200-500 words).
    """
    from src.models.agent_models import AgentNote
    
    title = args.get("title", "")
    summary = args.get("summary", "")
    urls = args.get("urls", [])
    tags = args.get("tags", [])
    
    if not title or not summary:
        return {"error": "Title and summary are required"}
    
    agent_id = context.get("agent_id", "researcher")
    agent_memory_service = context.get("agent_memory_service")
    agent_file_service = context.get("agent_file_service")
    research_memory_service = context.get("research_memory_service")
    session_id = context.get("session_id")
    stream = context.get("stream")
    
    if not agent_memory_service:
        return {"error": "Agent memory service not available"}
    
    try:
        note = AgentNote(
            title=title,
            summary=summary,
            urls=urls if isinstance(urls, list) else [],
            tags=tags if isinstance(tags, list) else []
        )
        
        file_path = await agent_memory_service.save_agent_note(
            note,
            agent_id,
            agent_file_service=agent_file_service,
            research_memory_service=research_memory_service,
            session_id=session_id
        )
        
        if stream:
            stream.emit_agent_note(agent_id, {
                "title": note.title,
                "summary": note.summary,
                "urls": note.urls,
                "shared": True
            })
        
        logger.info(f"Agent {agent_id} saved note via save_note tool",
                   title=title[:100],
                   summary_length=len(summary),
                   urls_count=len(urls))
        
        return {
            "success": True,
            "file_path": file_path,
            "title": title,
            "note": "Note saved successfully. It will be available for vector search."
        }
    except Exception as e:
        logger.error(f"Agent {agent_id} failed to save note", error=str(e))
        return {"error": f"Failed to save note: {str(e)}"}


async def select_urls_to_scrape_handler(args: dict[str, Any], context: dict[str, Any]) -> dict:
    """Select URLs to scrape based on search results analysis.
    
    LLM analyzes search results (title + snippet) and selects most relevant URLs for scraping.
    This is similar to how deep research researchers evaluate results before scraping.
    """
    search_results = args.get("search_results", [])
    original_query = args.get("original_query", "")
    # CRITICAL: Fallback to context if original_query not provided in args
    if not original_query:
        original_query = context.get("original_query", "")
    
    max_urls = args.get("max_urls", 5)  # Default: select top 5 URLs
    
    llm = context.get("llm")
    stream = context.get("stream")
    
    logger.info(
        "select_urls_to_scrape called",
        search_results_count=len(search_results),
        original_query_provided=bool(args.get("original_query")),
        original_query_from_context=bool(context.get("original_query")),
        original_query_preview=original_query[:100] if original_query else "None"
    )
    
    if not llm:
        logger.warning("No LLM available for URL selection, using top results")
        # Fallback: return top results
        return {
            "selected_urls": [r.get("url", "") for r in search_results[:max_urls] if r.get("url")],
            "reasoning": "Fallback: selected top results (no LLM available)"
        }
    
    if not search_results:
        logger.warning("No search results provided for URL selection")
        return {"selected_urls": [], "reasoning": "No search results to analyze"}
    
    # Format search results for LLM analysis
    results_text = "\n\n".join([
        f"[{i+1}] Title: {r.get('title', 'No title')}\n"
        f"URL: {r.get('url', 'No URL')}\n"
        f"Snippet: {r.get('snippet', r.get('content', ''))[:300]}"
        for i, r in enumerate(search_results[:15])  # Analyze top 15 results
    ])
    
    # Create prompt for LLM to select URLs
    from langchain_core.messages import SystemMessage, HumanMessage
    from pydantic import BaseModel, Field
    
    class URLSelection(BaseModel):
        """Selected URLs for scraping."""
        selected_urls: list[str] = Field(description="List of URLs to scrape (max 5-10 URLs)")
        reasoning: str = Field(description="Why these URLs were selected")
    
    prompt = f"""Analyze the following search results and select the most relevant URLs to scrape for answering the user's query.

Original Query: {original_query}

Search Results:
{results_text}

Instructions:
1. Evaluate each result's RELEVANCE to the original query
2. Check the TITLE and SNIPPET - do they relate to the query?
3. Consider source CREDIBILITY (prefer authoritative sources)
4. Select 3-{max_urls} most relevant URLs that will provide comprehensive information
5. Prioritize sources that directly answer the query
6. Skip irrelevant or low-quality sources

Return the selected URLs and your reasoning."""
    
    try:
        if stream:
            stream.emit_status("Analyzing search results to select pages for scraping...", step="analyze")
        
        structured_llm = llm.with_structured_output(URLSelection, method="function_calling")
        result = await structured_llm.ainvoke([
            SystemMessage(content="You are an expert at evaluating search results and selecting the most relevant sources for information gathering."),
            HumanMessage(content=prompt)
        ])
        
        selected_urls = result.selected_urls[:max_urls]  # Limit to max_urls
        logger.info(f"URL selection completed", 
                   selected_count=len(selected_urls),
                   total_results=len(search_results),
                   reasoning=result.reasoning[:200])
        
        if stream:
            stream.emit_status(f"Selected {len(selected_urls)} URLs for scraping", step="analyze")
        
        return {
            "selected_urls": selected_urls,
            "reasoning": result.reasoning
        }
    except Exception as e:
        logger.error(f"URL selection failed", error=str(e), exc_info=True)
        # Fallback: return top results
        return {
            "selected_urls": [r.get("url", "") for r in search_results[:max_urls] if r.get("url")],
            "reasoning": f"Fallback selection due to error: {str(e)}"
        }


# ==================== Action Registration ====================


def register_actions():
    """Register all available actions."""

    # Web Search (always available)
    ActionRegistry.register(
        name="web_search",
        description="Search the web for information. Provide up to 3 search queries. "
        "**QUERY STRATEGY**: Write natural search queries as you would type in a browser. "
        "Keep queries targeted and specific to what you need. Use all 3 slots when possible to maximize information gathering. "
        "**REFORMULATION**: If search results are NOT relevant to your task, try DIFFERENT search queries with different keywords, "
        "synonyms, related terms, or more specific/general phrasing. Don't repeat the same query multiple times. "
        "**VERIFICATION STRATEGY**: When you find important information, search for it in different sources to verify accuracy. "
        "For critical claims, find the same information in 3-5 additional independent sources. "
        "**SOURCE QUALITY**: Prefer authoritative sources (academic publications, official sources, established news organizations, "
        "expert-authored content). Be critical of sources with bias, lack of citations, or questionable credibility. "
        "**RESULTS**: Returns list of search results with title, URL, and snippet. "
        "**COVERAGE**: For balanced/quality modes, use max_results=10 for better coverage. For speed mode, max_results=5 is sufficient. "
        "**NEXT STEP**: After web_search, you MUST call select_urls_to_scrape with ALL results to intelligently choose which pages to scrape.",
        args_schema={
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of 1-3 natural search queries (as you would type in a browser). Use all 3 slots when possible to maximize information gathering.",
                    "minItems": 1,
                    "maxItems": 3,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results per query. CRITICAL: For balanced/quality modes, use 10 for better coverage. For speed mode, 5 is sufficient.",
                    "default": 10,  # Increased from 5 to 10 for better coverage
                },
            },
            # Azure/OpenRouter require all properties to be in required array
            "required": ["queries", "max_results"],
        },
        handler=web_search_handler,
    )

    # Scrape URLs
    ActionRegistry.register(
        name="scrape_url",
        description="Scrape full content from specific URLs. Use when user provides URLs or "
        "you need full article text. Returns scraped content (url, title, summary). "
        "**VERIFICATION**: After scraping, verify important claims from the content by searching for them in other sources. "
        "**SOURCE QUALITY**: Evaluate the credibility of the source before trusting the information. "
        "Check if the source is authoritative, has proper citations, and is from a reputable publisher. "
        "**CROSS-REFERENCE**: For critical information found in scraped content, find the same information in 2-3 additional "
        "independent sources to verify accuracy. If sources contradict, investigate WHY and document both sides.",
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

    # Select URLs to scrape (analyze search results and choose best URLs)
    ActionRegistry.register(
        name="select_urls_to_scrape",
        description="Analyze search results (title + snippet) and select the most relevant URLs to scrape. "
        "**MANDATORY WORKFLOW**: Use this AFTER web_search to intelligently choose which pages to scrape. "
        "**CRITICAL**: When calling this tool, pass ALL results from web_search (all results_count), NOT just the first few! "
        "**EVALUATION CRITERIA**: Evaluate each result's RELEVANCE to your task AND CREDIBILITY of the source. "
        "Look at TITLE, SNIPPET, and SOURCE DOMAIN - do they relate to your task AND appear trustworthy? "
        "Only select URLs that are CLEARLY relevant to your task AND from authoritative, credible sources. "
        "Skip irrelevant results or those from questionable sources. "
        "This ensures you scrape only relevant, high-quality sources (like ai.meta.com, huggingface.co), not just top-N by order.",
        args_schema={
            "type": "object",
            "properties": {
                "search_results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "url": {"type": "string"},
                            "snippet": {"type": "string"}
                        }
                    },
                    "description": "Search results from web_search to analyze",
                },
                "original_query": {
                    "type": "string",
                    "description": "Original user query to evaluate relevance against",
                },
                "max_urls": {
                    "type": "integer",
                    "description": "Maximum number of URLs to select (default: 5)",
                    "default": 5,
                },
            },
            "required": ["search_results", "original_query"],
        },
        handler=select_urls_to_scrape_handler,
    )

    # Done (signal completion)
    ActionRegistry.register(
        name="done",
        description="Signal that research is complete and you have gathered sufficient information. "
        "**WHEN TO CALL**: Only call this when you have completed DEEP, COMPREHENSIVE research. "
        "**MANDATORY REQUIREMENTS BEFORE CALLING**: "
        "1. Verified important claims in 3-5 independent sources (NOT just 2!) "
        "2. Explored the topic from MULTIPLE angles (at least 4-5 different perspectives) "
        "3. Found specific examples, case studies, and detailed information (minimum 3-5 concrete examples) "
        "4. Cross-referenced key findings across different sources "
        "5. Investigated related aspects and follow-up questions "
        "6. Found expert opinions, critical analysis, and alternative viewpoints "
        "7. Documented limitations, challenges, and edge cases "
        "8. Used at least 80% of your available steps (if you have 8 steps, use at least 6-7 before done()) "
        "**FORBIDDEN**: Do NOT call done() if you only have surface-level information, basic definitions, or general overviews. "
        "**DEEP RESEARCH REQUIREMENTS**: You must have investigated: technical specifications, expert analysis, case studies, "
        "historical context, advanced features, industry trends, comparative analysis, critical perspectives, limitations, and challenges. "
        "**VERIFICATION**: All important claims, facts, and data MUST be verified in MULTIPLE independent sources (minimum 3-5 sources for critical claims).",
        args_schema={
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Comprehensive summary of research findings. Include: key discoveries, verified facts, "
                    "sources used (with verification status), expert opinions found, case studies, limitations identified, "
                    "and how the findings relate to the research objective.",
                },
            },
            "required": ["summary"],
        },
        handler=done_handler,
    )

    # Save Note (for deep research mode)
    ActionRegistry.register(
        name="save_note",
        description="Save an important research note with title, detailed summary, and optional URLs. "
        "**WHEN TO USE**: Use this when you discover SUBSTANTIAL, ACTIONABLE INFORMATION during research. "
        "**WHAT TO SAVE**: Key discoveries, important facts, significant insights, technical details, "
        "expert opinions, historical context, real-world examples, comparative analysis, or patterns/trends. "
        "**WHAT NOT TO SAVE**: Never save routine notes like 'Found X sources', 'Search: query', "
        "lists of URLs without context, or generic summaries without specific facts. "
        "**QUALITY REQUIREMENTS**: Notes MUST be LARGE and DETAILED (minimum 200-500 words) with full context, "
        "specific facts, data points, numbers, dates, analysis, explanations, and source URLs. "
        "**COORDINATION**: Consider other agents' active tasks (shown in your context) - if your finding relates "
        "to their research topics, make your note comprehensive so they can find it via vector search. "
        "**STORAGE**: Notes are stored with vector search and can be retrieved by you and other agents for future reference.",
        args_schema={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Clear, descriptive title for the note (e.g., 'Key Finding: X', 'Discovery: Y', 'Technical Analysis: Z'). "
                    "Make it searchable and informative.",
                },
                "summary": {
                    "type": "string",
                    "description": "Detailed summary of the finding (MINIMUM 200-500 words - this is CRITICAL). "
                    "Include: specific facts, data, numbers, dates, concrete information, full context (what, why, when, where, how), "
                    "detailed explanations (not just brief summaries), analysis/interpretation/synthesis, multiple related facts together, "
                    "relationships between different pieces of information, quotes/statistics/examples from sources, "
                    "clear explanation of WHY this information is important, and how it relates to the research objective. "
                    "Write it so other agents can find and understand it via vector search.",
                },
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of source URLs related to this note. Include ALL relevant sources that support your findings.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of tags for categorizing the note (e.g., 'technical', 'historical', 'expert_opinion', 'case_study').",
                },
            },
            "required": ["title", "summary"],
        },
        handler=save_note_handler,
        enabled_condition=lambda ctx: ctx.get("mode") in ["quality"],  # Only for deep research
    )

    logger.info(f"Registered {len(ActionRegistry._actions)} actions")


# Initialize actions on module import
register_actions()

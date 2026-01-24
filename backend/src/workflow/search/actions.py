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

    # CRITICAL: Get task context from context if available (for focused summarization)
    task_title = context.get("task_title", "")
    task_objective = context.get("task_objective", "")
    task_note = context.get("task_note", "")
    
    # Build task context for LLM
    task_context = ""
    if task_title or task_objective:
        task_context = f"Research task: {task_title}\n"
        if task_objective:
            task_context += f"Objective: {task_objective}\n"
        if task_note:
            task_context += f"Guidance: {task_note}\n"
    
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

            # Step 2: Use LLM to analyze content with focus on task
            summary = ""
            brief_info = ""
            reasoning = ""
            is_relevant = False
            
            if llm and content_to_summarize:
                try:
                    if stream:
                        stream.emit_status(f"Analyzing: {title[:40]}...", step="analyze")

                    # CRITICAL: Use structured output to get summary, brief_info, and is_relevant
                    from src.models.schemas import ScrapedPageAnalysis
                    
                    # Build prompt with task context - CRITICAL: Include full task description
                    analysis_prompt = f"""Analyze the following web page content and create a comprehensive summary focused on the research task.

**RESEARCH TASK CONTEXT:**
{task_context if task_context else "No specific task context provided - analyze the content generally."}

**Page Title:** {title}
**Page URL:** {url}

**Page Content:**
{content_to_summarize[:12000]}  # Limit input to avoid context overflow

**Instructions:**
1. **Reasoning**: First, explain your reasoning about the page content - what information you found, how it relates to the research task (shown above), and why you made decisions about summary and relevance.
2. **Summary**: Create a comprehensive summary (2000-4000 tokens) that focuses on information relevant to the research task. Include all facts, data, insights, and details that could be useful for findings.
3. **Brief Info**: Create a brief 2-3 sentence description (max 200 chars) of what information is on this page.
4. **Relevance**: Determine if this page is relevant to the research task (True) or not (False).

**CRITICAL:** 
- The summary will be used to create detailed findings, so make it comprehensive and focused on the task.
- Consider the research task context (title, objective, guidance) when analyzing the page.
- Your reasoning should explain how the page content relates to the task objectives."""
                    
                    analysis_result = await llm.with_structured_output(
                        ScrapedPageAnalysis,
                        method="json_schema"
                    ).ainvoke([
                        {"role": "system", "content": "You are an expert at analyzing web content and creating focused summaries for research tasks."},
                        {"role": "user", "content": analysis_prompt}
                    ])
                    
                    # CRITICAL: Check if analysis_result is None before accessing attributes
                    if analysis_result is None:
                        raise ValueError("LLM returned None for page analysis")
                    
                    # CRITICAL: Check if attributes exist and are not None
                    reasoning = getattr(analysis_result, 'reasoning', None) or ""
                    summary = getattr(analysis_result, 'summary', None) or ""
                    brief_info = getattr(analysis_result, 'brief_info', None) or ""
                    is_relevant = getattr(analysis_result, 'is_relevant', True)
                    
                    # Ensure all required fields have valid values
                    if not summary:
                        raise ValueError("LLM analysis returned empty summary")
                    
                    logger.debug(f"Page analyzed with reasoning: {url}", 
                               reasoning_length=len(reasoning) if reasoning else 0,
                               summary_length=len(summary) if summary else 0,
                               brief_info_length=len(brief_info) if brief_info else 0,
                               is_relevant=is_relevant)
                    
                    logger.debug(f"Page analyzed: {url}", 
                               summary_length=len(summary) if summary else 0,
                               brief_info_length=len(brief_info) if brief_info else 0,
                               is_relevant=is_relevant)
                               
                except Exception as e:
                    logger.warning(f"LLM analysis failed: {url}", error=str(e))
                    # Fallback to simple summarization
                    try:
                        summary = await summarize_text_llm(
                            content_to_summarize,
                            max_tokens=3000,
                            llm=llm
                        )
                        brief_info = f"Page about {title[:100]}" if title else "Web page content"
                        is_relevant = True  # Default to relevant if analysis fails
                    except:
                        from src.utils.text import summarize_text
                        summary = summarize_text(content_to_summarize, 3200) if content_to_summarize else ""
                        brief_info = f"Page about {title[:100]}" if title else "Web page content"
                        is_relevant = True

            # If no summary and no LLM, use smart truncation
            if not summary and content_to_summarize:
                from src.utils.text import summarize_text
                summary = summarize_text(content_to_summarize, 3200)
                brief_info = f"Page about {title[:100]}" if title else "Web page content"
                is_relevant = True

            logger.debug(f"URL scraped and analyzed: {url}", 
                        summary_length=len(summary),
                        brief_info=brief_info[:50],
                        is_relevant=is_relevant)

            # Return url, title, summary (for findings), brief_info (for history), is_relevant (for history)
            return {
                "url": url,
                "title": title,
                "summary": summary,  # Comprehensive summary for findings creation
                "brief_info": brief_info,  # Brief info for tool history (to save context)
                "is_relevant": is_relevant,  # Whether page is relevant to task (for tool history)
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


async def create_finding_handler(args: dict[str, Any], context: dict[str, Any]) -> dict:
    """Create a comprehensive finding using LLM from scraped page summaries and search snippets.
    
    This tool automatically generates a detailed finding from all collected information:
    - Summary from scraped pages (if is_relevant=True)
    - Snippets from web_search results (with substantial content >50 chars)
    
    The LLM synthesizes all this information into a comprehensive finding focused on the research task.
    This is the RESULTING tool - it should be called when research is complete to generate the final finding.
    """
    from src.models.schemas import FindingContent
    
    agent_id = context.get("agent_id", "researcher")
    llm = context.get("llm")
    stream = context.get("stream")
    
    # Get task context
    task_title = context.get("task_title", "")
    task_objective = context.get("task_objective", "")
    task_note = context.get("task_note", "")
    
    # Get scraped_pages and sources from context (passed by researcher)
    scraped_pages = context.get("scraped_pages", [])
    sources = context.get("sources", [])
    
    if not llm:
        return {"error": "LLM not available"}
    
    try:
        # Collect relevant data
        relevant_scraped_summaries = []
        for page in scraped_pages:
            if page.get("is_relevant", True) and page.get("summary"):
                relevant_scraped_summaries.append({
                    "url": page.get("url", ""),
                    "title": page.get("title", ""),
                    "summary": page.get("summary", "")
                })
        
        # Collect useful snippets (>50 chars, not metadata)
        useful_snippets = []
        for src in sources:
            snippet = src.get("snippet", "").strip()
            if snippet and len(snippet) > 50:
                snippet_lower = snippet.lower()
                is_metadata = any([
                    "found" in snippet_lower and "sources" in snippet_lower and "query" in snippet_lower,
                    snippet_lower.startswith("search:") or snippet_lower.startswith("query:"),
                    snippet_lower.count("http") > 2,
                ])
                if not is_metadata:
                    useful_snippets.append({
                        "title": src.get("title", ""),
                        "url": src.get("url", ""),
                        "snippet": snippet
                    })
        
        # Build prompt for LLM
        task_context = f"Research Task: {task_title}\n"
        if task_objective:
            task_context += f"Objective: {task_objective}\n"
        if task_note:
            task_context += f"Guidance: {task_note}\n"
        
        scraped_summaries_text = ""
        if relevant_scraped_summaries:
            scraped_summaries_text = "\n\n".join([
                f"## {page['title']} ({page['url']})\n\n{page['summary']}"
                for page in relevant_scraped_summaries
            ])
        
        snippets_text = ""
        if useful_snippets:
            # CRITICAL: Use more snippets if available - finding should be comprehensive
            # Increase limit from 20 to 40 to use more information
            snippets_text = "\n\n".join([
                f"**{s['title']}** ({s['url']}): {s['snippet']}"
                for s in useful_snippets[:40]
            ])
        
        # Count total information available
        total_scraped_pages = len(relevant_scraped_summaries)
        total_snippets = len(useful_snippets)
        total_info_sources = total_scraped_pages + total_snippets
        
        finding_prompt = f"""Create a comprehensive, detailed finding based on the research task and collected information.

{task_context}

**Scraped Page Summaries (comprehensive summaries of relevant pages):**
{scraped_summaries_text if scraped_summaries_text else "No scraped pages available."}

**Web Search Results (snippets from pages not scraped):**
{snippets_text if snippets_text else "No search snippets available."}

**CRITICAL INSTRUCTIONS - USE ALL AVAILABLE INFORMATION:**
You have access to {total_scraped_pages} scraped page summaries and {total_snippets} search result snippets (total: {total_info_sources} information sources).

1. **COMPREHENSIVE SUMMARY (2000-4000 words minimum):**
   - Synthesize ALL information from ALL {total_info_sources} sources - do NOT skip any relevant information
   - Include ALL relevant facts, data, insights, comparisons, statistics, examples, and context from scraped summaries AND search snippets
   - Use ALL available information - the user has collected extensive data, so your finding should reflect that depth
   - If you have many sources ({total_info_sources} sources), your summary MUST be proportionally longer and more detailed
   - Include specific details, numbers, dates, names, technical terms, and concrete examples from the sources

2. **STRUCTURE AND FORMATTING:**
   - Use proper markdown formatting with sections (##), subsections (###), lists, bold, and links
   - Organize information logically by themes or topics
   - Include subsections for different aspects covered in the sources
   - Use bullet points and numbered lists for clarity

3. **KEY FINDINGS (10-20 items):**
   - Extract 10-20 key findings - specific facts, insights, data points, or conclusions
   - Each finding should be substantial and informative
   - Include findings from BOTH scraped pages AND search snippets
   - Prioritize unique or important information from each source

4. **COMPLETENESS:**
   - The finding must be self-contained and comprehensive enough to stand alone
   - Include ALL relevant information from the sources - do not summarize too briefly
   - If sources contain detailed information, your finding should reflect that detail
   - The more sources you have, the more comprehensive your finding should be

**CRITICAL:** 
- You have {total_info_sources} information sources available - your finding MUST use information from ALL of them
- The finding will be used by supervisor to create a chapter in the draft report
- If you have extensive information ({total_info_sources} sources), create an extensive finding (2000-4000+ words)
- Do NOT create a brief summary when you have detailed information available - USE ALL THE INFORMATION!"""
        
        finding_result = await llm.with_structured_output(
            FindingContent,
            method="json_schema"
        ).ainvoke([
            {"role": "system", "content": "You are an expert at synthesizing research findings into comprehensive, detailed summaries."},
            {"role": "user", "content": finding_prompt}
        ])
        
        logger.info(f"Agent {agent_id} created finding via create_finding tool",
                   summary_length=len(finding_result.summary),
                   key_findings_count=len(finding_result.key_findings),
                   scraped_pages_used=len(relevant_scraped_summaries),
                   snippets_used=len(useful_snippets))
        
        return {
            "success": True,
            "summary": finding_result.summary,
            "key_findings": finding_result.key_findings,
            "scraped_pages_count": len(relevant_scraped_summaries),
            "snippets_count": len(useful_snippets),
            "note": "Finding created successfully. This will be used as the final result."
        }
    except Exception as e:
        logger.error(f"Agent {agent_id} failed to create finding", error=str(e))
        return {"error": f"Failed to create finding: {str(e)}"}


async def create_note_handler(args: dict[str, Any], context: dict[str, Any]) -> dict:
    """Create a comprehensive note using LLM from scraped page summaries and search snippets.
    
    This tool automatically generates a detailed note from all collected information:
    - Summary from scraped pages (if is_relevant=True)
    - Snippets from web_search results (with substantial content >50 chars)
    
    The LLM synthesizes all this information into a comprehensive note focused on the research task.
    """
    from src.models.agent_models import AgentNote
    from src.models.schemas import NoteContent
    
    agent_id = context.get("agent_id", "researcher")
    llm = context.get("llm")
    agent_memory_service = context.get("agent_memory_service")
    agent_file_service = context.get("agent_file_service")
    research_memory_service = context.get("research_memory_service")
    session_id = context.get("session_id")
    stream = context.get("stream")
    
    # Get task context
    task_title = context.get("task_title", "")
    task_objective = context.get("task_objective", "")
    task_note = context.get("task_note", "")
    
    # Get scraped_pages and sources from context (passed by researcher)
    scraped_pages = context.get("scraped_pages", [])
    sources = context.get("sources", [])
    
    if not agent_memory_service or not llm:
        return {"error": "Agent memory service or LLM not available"}
    
    try:
        # Collect relevant data
        relevant_scraped_summaries = []
        for page in scraped_pages:
            if page.get("is_relevant", True) and page.get("summary"):
                relevant_scraped_summaries.append({
                    "url": page.get("url", ""),
                    "title": page.get("title", ""),
                    "summary": page.get("summary", "")
                })
        
        # Collect useful snippets (>50 chars, not metadata)
        useful_snippets = []
        for src in sources:
            snippet = src.get("snippet", "").strip()
            if snippet and len(snippet) > 50:
                snippet_lower = snippet.lower()
                is_metadata = any([
                    "found" in snippet_lower and "sources" in snippet_lower and "query" in snippet_lower,
                    snippet_lower.startswith("search:") or snippet_lower.startswith("query:"),
                    snippet_lower.count("http") > 2,
                ])
                if not is_metadata:
                    useful_snippets.append({
                        "title": src.get("title", ""),
                        "url": src.get("url", ""),
                        "snippet": snippet
                    })
        
        # Build prompt for LLM
        task_context = f"Research Task: {task_title}\n"
        if task_objective:
            task_context += f"Objective: {task_objective}\n"
        if task_note:
            task_context += f"Guidance: {task_note}\n"
        
        scraped_summaries_text = ""
        if relevant_scraped_summaries:
            scraped_summaries_text = "\n\n".join([
                f"## {page['title']} ({page['url']})\n\n{page['summary']}"
                for page in relevant_scraped_summaries
            ])
        
        snippets_text = ""
        if useful_snippets:
            snippets_text = "\n\n".join([
                f"**{s['title']}** ({s['url']}): {s['snippet']}"
                for s in useful_snippets[:15]
            ])
        
        note_prompt = f"""Create a comprehensive research note based on the research task and collected information.

{task_context}

**Scraped Page Summaries:**
{scraped_summaries_text if scraped_summaries_text else "No scraped pages available."}

**Web Search Results (snippets):**
{snippets_text if snippets_text else "No search snippets available."}

**Instructions:**
1. Create a concise, descriptive title (max 100 chars) for the note.
2. Create comprehensive note content (500-2000 words) that synthesizes ALL information.
3. Focus on the research task and include all relevant facts, insights, and context.
4. Make the note detailed and informative.

**CRITICAL:** The note will be stored for vector search and used by other agents, so make it comprehensive!"""
        
        note_result = await llm.with_structured_output(
            NoteContent,
            method="json_schema"
        ).ainvoke([
            {"role": "system", "content": "You are an expert at creating comprehensive research notes from multiple sources."},
            {"role": "user", "content": note_prompt}
        ])
        
        # Save the note
        note = AgentNote(
            title=note_result.title,
            summary=note_result.summary,
            urls=[page.get("url") for page in relevant_scraped_summaries[:5] if page.get("url")],
            tags=["research_note"]
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
        
        logger.info(f"Agent {agent_id} created note via create_note tool",
                   title=note.title[:100],
                   summary_length=len(note.summary),
                   scraped_pages_used=len(relevant_scraped_summaries),
                   snippets_used=len(useful_snippets))
        
        return {
            "success": True,
            "file_path": file_path,
            "title": note.title,
            "summary_length": len(note.summary),
            "note": "Note created and saved successfully. It will be available for vector search."
        }
    except Exception as e:
        logger.error(f"Agent {agent_id} failed to create note", error=str(e))
        return {"error": f"Failed to create note: {str(e)}"}


async def save_note_handler(args: dict[str, Any], context: dict[str, Any]) -> dict:
    """Save a research note with title, summary, and optional URLs.
    
    **DEPRECATED**: Use create_note instead, which automatically generates comprehensive notes from collected data.
    This handler is kept for backward compatibility.
    
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
        
        # CRITICAL: Check if result is None before accessing attributes
        if result is None:
            raise ValueError("LLM returned None for URL selection")
        
        # CRITICAL: Check if selected_urls exists and is not None
        selected_urls_list = getattr(result, 'selected_urls', None)
        if selected_urls_list is None:
            selected_urls_list = []
        elif not isinstance(selected_urls_list, list):
            # If it's not a list, try to convert or use empty list
            try:
                selected_urls_list = list(selected_urls_list) if selected_urls_list else []
            except (TypeError, ValueError):
                selected_urls_list = []
        
        selected_urls = selected_urls_list[:max_urls] if selected_urls_list else []  # Limit to max_urls
        reasoning = getattr(result, 'reasoning', '') or 'URLs selected based on relevance to query'
        
        logger.info(f"URL selection completed", 
                   selected_count=len(selected_urls),
                   total_results=len(search_results),
                   reasoning=reasoning[:200] if reasoning else "No reasoning provided")
        
        if stream:
            stream.emit_status(f"Selected {len(selected_urls)} URLs for scraping", step="analyze")
        
        return {
            "selected_urls": selected_urls,
            "reasoning": reasoning
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

    # Create Finding (for deep research mode) - RESULTING tool for task completion
    ActionRegistry.register(
        name="create_finding",
        description="Create a comprehensive research finding automatically from all collected information. "
        "**MANDATORY RESULTING TOOL**: This is the FINAL tool you MUST call when research is complete. "
        "**FORBIDDEN**: Do NOT write text summaries - you MUST call this tool instead. Text responses will be IGNORED. "
        "**AUTOMATIC**: This tool uses LLM to synthesize information from scraped page summaries and search snippets into a detailed finding. "
        "**WHEN TO CALL**: Call this when you have completed your research and want to generate the final finding. "
        "**MANDATORY**: When you have gathered sufficient information (sources, scraped pages), you MUST call this tool - do NOT write text summaries. "
        "**WHAT IT DOES**: Automatically generates comprehensive summary (1500-3000 words) and key findings from all relevant scraped summaries and search snippets. "
        "**NO PARAMETERS NEEDED**: The tool automatically uses all collected data from your research session. "
        "**CRITICAL**: If you don't call this tool, it will be called automatically when you reach max_steps or call done(). "
        "**PREFERRED**: It's better to call this explicitly when you feel research is complete. "
        "**REMINDER**: You are in a tool-calling loop - you MUST call tools, not write text. When research is done, call create_finding(), not a text summary.",
        args_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=create_finding_handler,
        enabled_condition=lambda ctx: ctx.get("mode") in ["quality"],  # Only for deep research
    )
    
    # Create Note (for deep research mode) - automatically generates comprehensive note from collected data
    ActionRegistry.register(
        name="create_note",
        description="Create a comprehensive research note automatically from all collected information. "
        "**AUTOMATIC**: This tool uses LLM to synthesize information from scraped page summaries and search snippets into a detailed note. "
        "**WHEN TO USE**: Call this when you have collected enough information (scraped pages and search results) and want to save a comprehensive note. "
        "**WHAT IT DOES**: Automatically generates title and detailed content (500-2000 words) from all relevant scraped summaries and search snippets. "
        "**NO PARAMETERS NEEDED**: The tool automatically uses all collected data from your research session. "
        "**CRITICAL**: This is the preferred way to create notes - it ensures comprehensive, well-structured notes from all your research.",
        args_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=create_note_handler,
        enabled_condition=lambda ctx: ctx.get("mode") in ["quality"],  # Only for deep research
    )
    
    # Save Note (for deep research mode) - REMOVED, use create_note instead
    # ActionRegistry.register(
    #     name="save_note",
    #     description="Save an important research note with title, detailed summary, and optional URLs. "
    #     "**DEPRECATED**: Prefer create_note which automatically generates comprehensive notes from collected data. "
    #     "**WHEN TO USE**: Only use this if you need to manually specify note content. "
    #     "**WHAT TO SAVE**: Key discoveries, important facts, significant insights, technical details, "
    #     "expert opinions, historical context, real-world examples, comparative analysis, or patterns/trends. "
    #     "**WHAT NOT TO SAVE**: Never save routine notes like 'Found X sources', 'Search: query', "
    #     "lists of URLs without context, or generic summaries without specific facts. "
    #     "**QUALITY REQUIREMENTS**: Notes MUST be LARGE and DETAILED (minimum 200-500 words) with full context, "
    #     "specific facts, data points, numbers, dates, analysis, explanations, and source URLs.",
    #     args_schema={
    #         "type": "object",
    #         "properties": {
    #             "title": {
    #                 "type": "string",
    #                 "description": "Clear, descriptive title for the note (e.g., 'Key Finding: X', 'Discovery: Y', 'Technical Analysis: Z'). "
    #                 "Make it searchable and informative.",
    #             },
    #             "summary": {
    #                 "type": "string",
    #                 "description": "Detailed summary of the finding (MINIMUM 200-500 words - this is CRITICAL). "
    #                 "Include: specific facts, data, numbers, dates, concrete information, full context (what, why, when, where, how), "
    #                 "detailed explanations (not just brief summaries), analysis/interpretation/synthesis, multiple related facts together, "
    #                 "relationships between different pieces of information, quotes/statistics/examples from sources, "
    #                 "clear explanation of WHY this information is important, and how it relates to the research objective. "
    #                 "Write it so other agents can find and understand it via vector search.",
    #             },
    #             "urls": {
    #                 "type": "array",
    #                 "items": {"type": "string"},
    #                 "description": "List of source URLs related to this note. Include ALL relevant sources that support your findings.",
    #             },
    #             "tags": {
    #                 "type": "array",
    #                 "items": {"type": "string"},
    #                 "description": "Optional list of tags for categorizing the note (e.g., 'technical', 'historical', 'expert_opinion', 'case_study').",
    #             },
    #         },
    #         "required": ["title", "summary"],
    #     },
    #     handler=save_note_handler,
    #     enabled_condition=lambda ctx: ctx.get("mode") in ["quality"],  # Only for deep research
    # )

    logger.info(f"Registered {len(ActionRegistry._actions)} actions")


# Initialize actions on module import
register_actions()

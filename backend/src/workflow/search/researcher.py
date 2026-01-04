"""Research agent with tool-calling loop (Perplexica-style).

Two-stage architecture: Researcher (gather info) → Writer (synthesize answer).
"""

import json
import structlog
from typing import Any
from pydantic import BaseModel, Field

from src.workflow.search.actions import ActionRegistry
from src.workflow.search.classifier import QueryClassification, get_current_date, format_chat_history

logger = structlog.get_logger(__name__)


# ==================== Schemas ====================


class ResearchAction(BaseModel):
    """Single research action with reasoning."""

    reasoning: str = Field(description="Chain-of-thought for this action")
    action: str = Field(description="Action name from registry")
    args: dict[str, Any] = Field(default_factory=dict, description="Action arguments")


class ResearchPlan(BaseModel):
    """Research plan with multiple actions (for structured output)."""

    reasoning_preamble: str | None = Field(
        None,
        description="Chain-of-thought reasoning before actions (mandatory in balanced/quality)",
        alias="reasoning"
    )
    actions: list[ResearchAction] = Field(
        description="Sequence of research actions to execute"
    )
    done: bool = Field(
        default=False,
        description="Whether research is complete"
    )


# ==================== Mode-Specific Prompts ====================


def get_researcher_prompt(mode: str, iteration: int, max_iterations: int, original_query: str = "") -> str:
    """Get mode-specific researcher prompt."""

    current_date = get_current_date()

    base_prompt = f"""You are a research agent gathering information from the web.

Current date: {current_date}

Your knowledge cutoff is early 2024. For ANY information after that date, you MUST use web search.

When calling web_search tool:
- Write natural search queries as you would type in a browser
- Keep queries targeted and specific to what you need
- You can provide up to 3 queries at a time
- Use the same language as the user's query
"""

    if mode == "speed":
        return base_prompt + f"""
MODE: SPEED (iteration {iteration+1}/{max_iterations})

Your goal: Get information quickly and efficiently.

Strategy:
1. Make 1-2 targeted web searches with specific queries (use all 3 query slots if possible!)
2. Scrape 1-2 most promising URLs if needed
3. Call 'done' once you have enough information

IMPORTANT:
- You only have {max_iterations} iterations total
- Be concise and focused
- Don't over-analyze - gather key info and finish
- When calling web_search, provide up to 3 queries to maximize information gathering
- Your queries should be SEO-friendly keywords, NOT full sentences
- Example: ["сабля", "сабля история", "сабля виды"] ✅ NOT ["расскажи про саблю"] ❌

Available actions: {list(ActionRegistry._actions.keys())}
"""

    elif mode == "balanced":
        return base_prompt + f"""
MODE: BALANCED (iteration {iteration+1}/{max_iterations})

Your goal: Thorough research with good coverage. GO DEEP, NOT WIDE!

Strategy:
1. **MANDATORY**: Start with __reasoning_preamble explaining your thinking
2. Make 2-3 web searches from different angles (preserve key terms from user query!)
3. Scrape 2-3 most promising URLs
4. **CRITICAL**: After initial search, dig deeper into specific aspects:
   - Technical details and specifications
   - Historical context and evolution
   - Expert analysis and professional opinions
   - Real-world applications and case studies
   - Advanced features and capabilities
   - Industry trends and future developments
5. Iterate if gaps remain (you have {max_iterations} iterations)
6. Call 'done' when you have comprehensive information

IMPORTANT:
- EVERY response MUST start with __reasoning_preamble
- Explain: what you've learned, what gaps remain, next action
- **DO NOT just ask basic questions - dig into specifics!**
- After getting overview, search for: "advanced features", "technical details", "expert analysis", "case studies"
- Minimum 2 information-gathering calls (search/scrape)
- When generating search queries, preserve the original query's key terms and language
- Use all 3 query slots in web_search when possible to maximize information gathering
- Start with broader queries, then narrow down based on results
- Your queries should be SEO-friendly keywords, NOT full sentences
- **Iteration {iteration+1}**: If iteration > 1, you should be searching for SPECIFIC details, not general info

Available actions: {list(ActionRegistry._actions.keys())}

Example flow:
1. __reasoning_preamble: "Okay, the user wants to learn about X. I need to search for..."
2. web_search: ["X overview", "X basics"]
3. scrape_url: ["url1", "url2"]
4. __reasoning_preamble: "I've found basic info. Now I need SPECIFIC details: technical specs, expert opinions..."
5. web_search: ["X technical specifications", "X expert analysis", "X advanced features"]
6. scrape_url: ["url3", "url4"]
7. __reasoning_preamble: "I have comprehensive info. Let me check for case studies and real-world applications..."
8. web_search: ["X case studies", "X real-world applications"]
9. done: "Research completed with comprehensive coverage"
"""

    else:  # quality
        return base_prompt + f"""
MODE: QUALITY (iteration {iteration+1}/{max_iterations})

Your goal: Exhaustive, deep research leaving no stone unturned. GO DEEP INTO SPECIFICS!

Strategy:
1. **MANDATORY**: Start with __reasoning_preamble explaining your comprehensive plan
2. Explore multiple angles:
   - Definitions and core concepts
   - Features and capabilities
   - Comparisons with alternatives
   - Recent news and updates
   - Expert opinions and reviews
   - Use cases and applications
   - Limitations and criticisms
3. **CRITICAL**: After initial overview, dig into SPECIFIC details:
   - Technical specifications and implementation details
   - Historical evolution and context
   - Expert analysis and professional deep-dives
   - Real-world case studies and examples
   - Advanced features and edge cases
   - Industry trends and future developments
   - Performance metrics and benchmarks
   - Best practices and recommendations
4. Use 4-7 information-gathering calls (searches + scrapes)
5. Cross-reference from multiple authoritative sources
6. Call 'done' only when truly comprehensive (use most of your {max_iterations} iterations)

IMPORTANT:
- EVERY response MUST start with __reasoning_preamble
- Think deeply: what's missing? what angles are unexplored?
- **DO NOT just ask basic questions - dig into specifics!**
- **Iteration {iteration+1}**: 
  - If iteration <= 2: Get overview and basics
  - If iteration > 2: Search for SPECIFIC details, technical specs, expert analysis, case studies
- Minimum 5-6 iterations before considering 'done'
- Be meticulous - this is quality mode
- When generating search queries, preserve the original query's key terms and language
- Use all 3 query slots in web_search when possible to maximize information gathering
- Start with broader queries, then narrow down based on results
- Never stop before at least 5-6 iterations of searches unless the question is very simple
- Your queries should be SEO-friendly keywords, NOT full sentences
- **After iteration 2, your queries should target SPECIFIC aspects, not general info**

Available actions: {list(ActionRegistry._actions.keys())}

Research checklist:
- [ ] Core definition/overview
- [ ] Key features/capabilities
- [ ] Comparisons with alternatives
- [ ] Recent news/developments
- [ ] Expert opinions/reviews
- [ ] Practical use cases
- [ ] Limitations/criticisms
- [ ] Technical details
- [ ] Historical context and evolution
- [ ] Advanced features and edge cases
- [ ] Real-world case studies
- [ ] Industry trends and future
"""


# ==================== Research Agent ====================


async def research_agent(
    query: str,
    classification: QueryClassification,
    mode: str,  # speed, balanced, quality
    llm: Any,
    search_provider: Any,
    scraper: Any,
    stream: Any,
    chat_history: list[dict] = None,
) -> dict[str, Any]:
    """
    Run research agent with action loop.

    Args:
        query: User query
        classification: Query classification result
        mode: Research mode (speed/balanced/quality)
        llm: LLM instance
        search_provider: Search provider instance
        scraper: Web scraper instance
        stream: Streaming generator for progress updates
        chat_history: Chat history for context

    Returns:
        Dict with:
            - sources: List of search results
            - scraped_content: List of scraped content
            - reasoning_history: List of reasoning steps
    """
    chat_history = chat_history or []

    # Mode-specific iteration limits (Perplexica style)
    max_iterations = {
        "speed": 2,
        "balanced": 6,
        "quality": 25
    }[mode]

    # Get available tools based on mode
    # Handle case where classification might be None (shouldn't happen, but safety check)
    query_type = classification.query_type if classification else "factual"
    tools = ActionRegistry.get_tool_definitions(
        mode=mode,
        classification=query_type,
        context={}
    )

    # Track research state
    sources = []
    scraped_content = []
    reasoning_history = []
    agent_history = []

    # Add initial context
    history_context = format_chat_history(chat_history, limit=6)
    original_query = query  # Preserve original user query
    standalone_query = classification.standalone_query if classification else query
    
    agent_history.append({
        "role": "user",
        "content": f"Chat history:\n{history_context}\n\nUser query: {standalone_query}"
    })

    logger.info(
        "Starting research",
        mode=mode,
        max_iterations=max_iterations,
        query=query[:100]
    )

    # Research loop
    for iteration in range(max_iterations):
        try:
            # Get researcher prompt for current iteration - pass original query to ensure key terms are preserved
            system_prompt = get_researcher_prompt(mode, iteration, max_iterations, original_query=query)

            # Build messages for LLM
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

            messages = [SystemMessage(content=system_prompt)]

            # Add conversation history
            # Process history in pairs: AIMessage followed by its ToolMessages
            i = 0
            while i < len(agent_history):
                msg = agent_history[i]
                
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                    i += 1
                elif msg["role"] == "assistant":
                    # Restore AIMessage with tool_calls if they exist
                    tool_calls_from_history = msg.get("tool_calls", [])
                    if tool_calls_from_history:
                        # Convert tool_calls to LangChain format
                        from langchain_core.messages.tool import ToolCall
                        langchain_tool_calls = []
                        tool_call_ids = []
                        for tc in tool_calls_from_history:
                            # Support both formats
                            if isinstance(tc, dict):
                                tool_name = tc.get("name") or tc.get("function", {}).get("name")
                                tool_args = tc.get("args") or tc.get("function", {}).get("arguments", {})
                                # CRITICAL: tool_id must match the ID used in ToolMessage!
                                tool_id = tc.get("id") or tc.get("function", {}).get("id") or f"call_{iteration}_{len(langchain_tool_calls)}"
                            elif hasattr(tc, "name"):
                                tool_name = tc.name
                                tool_args = tc.args if hasattr(tc, "args") else {}
                                # CRITICAL: tool_id must match the ID used in ToolMessage!
                                tool_id = tc.id if hasattr(tc, "id") else f"call_{iteration}_{len(langchain_tool_calls)}"
                            else:
                                logger.warning(f"Skipping invalid tool_call format", tc=tc, tc_type=type(tc).__name__)
                                continue
                            
                            if tool_name and tool_id:
                                langchain_tool_calls.append(ToolCall(
                                    name=tool_name,
                                    args=tool_args,
                                    id=tool_id  # This ID must match tool_call_id in ToolMessage!
                                ))
                                tool_call_ids.append(tool_id)
                                logger.debug(f"Restored ToolCall from history", name=tool_name, id=tool_id, 
                                           tc_format=type(tc).__name__)
                            else:
                                logger.warning(f"Could not restore tool call - missing name or id", 
                                             tool_name=tool_name, tool_id=tool_id, tc=tc)
                        
                        aimessage = AIMessage(
                            content=msg.get("content", ""),
                            tool_calls=langchain_tool_calls
                        )
                        messages.append(aimessage)
                        
                        # Now look ahead for ToolMessages that correspond to this AIMessage
                        # They should come immediately after in agent_history
                        j = i + 1
                        while j < len(agent_history) and agent_history[j].get("role") == "tool":
                            tool_msg = agent_history[j]
                            tool_call_id = tool_msg.get("tool_call_id")
                            if tool_call_id in tool_call_ids:
                                logger.debug(f"Restoring ToolMessage immediately after AIMessage", 
                                           tool_call_id=tool_call_id,
                                           content_preview=tool_msg.get("content", "")[:100])
                                messages.append(ToolMessage(
                                    content=tool_msg["content"],
                                    tool_call_id=tool_call_id
                                ))
                                j += 1
                            else:
                                # This ToolMessage doesn't belong to this AIMessage, stop looking
                                break
                        i = j
                    else:
                        messages.append(AIMessage(content=msg.get("content", "")))
                        i += 1
                elif msg["role"] == "tool":
                    # This ToolMessage should have been processed with its AIMessage above
                    # If we reach here, it means it's orphaned - add it anyway
                    tool_call_id = msg.get("tool_call_id", f"call_{iteration}")
                    logger.debug(f"Adding orphaned ToolMessage", tool_call_id=tool_call_id,
                               content_preview=msg.get("content", "")[:100])
                    messages.append(ToolMessage(
                        content=msg["content"],
                        tool_call_id=tool_call_id
                    ))
                    i += 1
                else:
                    logger.warning(f"Unknown message role in history", role=msg.get("role"))
                    i += 1

            # Stream tool calls from LLM
            if stream:
                stream.emit_status(f"Research iteration {iteration+1}/{max_iterations}", step="research")

            # Bind tools to LLM for tool calling
            # Convert our tool definitions to format expected by bind_tools
            # Most LangChain LLMs accept tool definitions directly
            try:
                if hasattr(llm, "bind_tools"):
                    # Try binding tools directly with our definitions
                    llm_with_tools = llm.bind_tools(tools)
                else:
                    # Fallback: some LLMs need tools in different format
                    llm_with_tools = llm
            except Exception as e:
                logger.warning(f"Failed to bind tools, using LLM without tools", error=str(e))
                llm_with_tools = llm
            
            # Validate message order before sending to LLM
            # Check that every AIMessage with tool_calls has corresponding ToolMessages
            for i, msg in enumerate(messages):
                if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_call_ids = [tc.id if hasattr(tc, "id") else (tc.get("id") if isinstance(tc, dict) else None) for tc in msg.tool_calls]
                    # Check if following messages contain ToolMessages for these tool_calls
                    following_tool_messages = []
                    for j in range(i + 1, len(messages)):
                        if isinstance(messages[j], ToolMessage):
                            following_tool_messages.append(messages[j].tool_call_id)
                    
                    missing_tool_messages = [tid for tid in tool_call_ids if tid and tid not in following_tool_messages]
                    if missing_tool_messages:
                        logger.warning(f"AIMessage at index {i} has tool_calls without ToolMessages", 
                                     tool_call_ids=tool_call_ids,
                                     missing_ids=missing_tool_messages,
                                     following_tool_message_ids=following_tool_messages)
            
            # Get LLM response with tool calling
            response = await llm_with_tools.ainvoke(messages)

            # Extract tool calls from LLM response
            # LangChain returns tool_calls as a list of ToolCall objects
            tool_calls = []
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_calls = response.tool_calls
                logger.debug(f"Extracted {len(tool_calls)} tool calls from response", 
                           tool_call_ids=[tc.id if hasattr(tc, "id") else getattr(tc, "get", lambda k, d: d)("id", "unknown") for tc in tool_calls])

            # Check for done - support both formats:
            # 1. LangChain format: {"name": "...", "args": {...}}
            # 2. OpenAI format: {"function": {"name": "...", "arguments": {...}}}
            done = False
            for tc in tool_calls:
                tool_name = None
                if isinstance(tc, dict):
                    # Try LangChain format first
                    tool_name = tc.get("name")
                    if not tool_name and "function" in tc:
                        # Try OpenAI format
                        tool_name = tc.get("function", {}).get("name")
                elif hasattr(tc, "name"):
                    # ToolCall object
                    tool_name = tc.name
                
                if tool_name == "done":
                    done = True
                    break

            if not tool_calls or done:
                logger.info(f"Research completed at iteration {iteration+1}")
                break

            # Execute tools in parallel where possible
            import asyncio

            # Separate reasoning preamble from action tools
            reasoning_calls = []
            action_calls = []
            
            for tool_call in tool_calls:
                # Extract tool name - support both formats
                tool_name = None
                if isinstance(tool_call, dict):
                    # Try LangChain format first: {"name": "...", "args": {...}}
                    tool_name = tool_call.get("name")
                    if not tool_name and "function" in tool_call:
                        # Try OpenAI format: {"function": {"name": "...", "arguments": {...}}}
                        tool_name = tool_call.get("function", {}).get("name")
                elif hasattr(tool_call, "name"):
                    # ToolCall object from LangChain
                    tool_name = tool_call.name
                
                if not tool_name:
                    logger.warning(f"Could not extract tool name from tool_call", tool_call=tool_call)
                    continue
                
                if tool_name == "__reasoning_preamble":
                    reasoning_calls.append(tool_call)
                else:
                    action_calls.append(tool_call)
            
            action_results = []
            
            # Handle reasoning preamble first (synchronous, for logging)
            for tool_call in reasoning_calls:
                # Extract args and tool_call_id - support both formats
                tool_args = {}
                tool_call_id = f"call_{iteration}_{len(reasoning_calls)}"
                
                if isinstance(tool_call, dict):
                    # Try LangChain format first: {"name": "...", "args": {...}, "id": "..."}
                    tool_args = tool_call.get("args") or tool_call.get("function", {}).get("arguments", {})
                    tool_call_id = tool_call.get("id") or tool_call.get("function", {}).get("id") or tool_call_id
                elif hasattr(tool_call, "name"):
                    # ToolCall object from LangChain
                    tool_args = tool_call.args if hasattr(tool_call, "args") else {}
                    # CRITICAL: Use the exact ID from the ToolCall object
                    if hasattr(tool_call, "id") and tool_call.id:
                        tool_call_id = tool_call.id
                    else:
                        tool_call_id = f"call_{iteration}_{reasoning_calls.index(tool_call)}"
                
                reasoning = tool_args.get("reasoning", "")
                reasoning_history.append(reasoning)
                
                logger.debug(f"Processing reasoning preamble", 
                           tool_call_id=tool_call_id, 
                           tool_call_type=type(tool_call).__name__,
                           reasoning_preview=reasoning[:100])
                
                # CRITICAL: Ensure tool_call_id matches what will be saved in AIMessage tool_calls
                # Verify this ID will be found when restoring history
                action_results.append({
                    "tool_call_id": tool_call_id,
                    "output": json.dumps({"reasoning": reasoning})
                })
                logger.debug(f"Added reasoning result to action_results", tool_call_id=tool_call_id)
            
            # Execute action tools in parallel
            async def execute_tool(tool_call):
                # Extract tool name and args - support both formats
                tool_name = None
                tool_args = {}
                tool_call_id = f"call_{iteration}_{len(action_calls)}"
                
                if isinstance(tool_call, dict):
                    # Try LangChain format first: {"name": "...", "args": {...}, "id": "..."}
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args", {})
                    tool_call_id = tool_call.get("id", tool_call_id)
                    
                    if not tool_name and "function" in tool_call:
                        # Try OpenAI format: {"function": {"name": "...", "arguments": {...}}, "id": "..."}
                        tool_name = tool_call.get("function", {}).get("name")
                        tool_args = tool_call.get("function", {}).get("arguments", {})
                        tool_call_id = tool_call.get("id", tool_call_id)
                elif hasattr(tool_call, "name"):
                    # ToolCall object from LangChain
                    tool_name = tool_call.name
                    tool_args = tool_call.args if hasattr(tool_call, "args") else {}
                    # CRITICAL: Use the exact ID from the ToolCall object, or generate a unique one
                    if hasattr(tool_call, "id") and tool_call.id:
                        tool_call_id = tool_call.id
                    else:
                        # Generate unique ID based on iteration and position
                        tool_call_id = f"call_{iteration}_{action_calls.index(tool_call) if tool_call in action_calls else len(action_calls)}"
                
                logger.debug(f"Executing tool", tool_name=tool_name, tool_call_id=tool_call_id, 
                           tool_call_type=type(tool_call).__name__)
                
                if not tool_name:
                    logger.error(f"Could not extract tool name from tool_call", tool_call=tool_call)
                    return {
                        "tool_call_id": tool_call_id,
                        "output": json.dumps({"error": "Invalid tool call format"})
                    }
                
                try:
                    # Log search queries for debugging
                    search_queries = []
                    if tool_name == "web_search":
                        search_queries = tool_args.get("queries", [])
                        logger.info(f"Research agent generating search queries", 
                                   queries=search_queries, 
                                   original_query=query,
                                   iteration=iteration+1)
                    
                    result = await ActionRegistry.execute(
                        tool_name,
                        tool_args,
                        {
                            "search_provider": search_provider,
                            "scraper": scraper,
                            "stream": stream,
                            "llm": llm,  # Required for content summarization!
                            "mode": mode,
                            "agent_id": "researcher"
                        }
                    )

                    # Track sources and content
                    if tool_name == "web_search" and "results" in result:
                        search_results = result.get("results", [])
                        logger.info(f"Search results received", 
                                   queries=search_queries,
                                   results_count=len(search_results),
                                   top_titles=[r.get("title", "")[:50] for r in search_results[:3]])
                        sources.extend(search_results)
                    elif tool_name == "scrape_url" and "scraped" in result:
                        scraped_content.extend(result["scraped"])

                    # Format result for LLM to see - make it readable
                    # For web_search, show titles and snippets so LLM can see what was found
                    if tool_name == "web_search" and "results" in result:
                        formatted_output = {
                            "results_count": len(result.get("results", [])),
                            "results": [
                                {
                                    "title": r.get("title", ""),
                                    "url": r.get("url", ""),
                                    "snippet": r.get("snippet", "")[:200]  # Truncate for readability
                                }
                                for r in result.get("results", [])[:5]  # Show top 5 results
                            ]
                        }
                        output_str = json.dumps(formatted_output, ensure_ascii=False, indent=2)
                    else:
                        output_str = json.dumps(result, ensure_ascii=False, indent=2)
                    
                    return {
                        "tool_call_id": tool_call_id,
                        "output": output_str
                    }

                except Exception as e:
                    logger.warning(f"Action failed: {tool_name}", error=str(e))
                    return {
                        "tool_call_id": tool_call_id,
                        "output": json.dumps({"error": str(e)})
                    }
            
            # Execute all action tools in parallel
            if action_calls:
                parallel_results = await asyncio.gather(*[execute_tool(tc) for tc in action_calls])
                action_results.extend(parallel_results)

            # Add to history - preserve tool_calls in format that can be restored
            # Convert tool_calls to dict format for storage (preserve IDs!)
            tool_calls_for_history = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    # Already in dict format - preserve as is
                    tool_calls_for_history.append(tc)
                elif hasattr(tc, "name"):
                    # ToolCall object from LangChain - convert to dict preserving ID
                    tool_id = tc.id if hasattr(tc, "id") else f"call_{iteration}_{len(tool_calls_for_history)}"
                    tool_calls_for_history.append({
                        "name": tc.name,
                        "args": tc.args if hasattr(tc, "args") else {},
                        "id": tool_id,
                        "type": "tool_call"
                    })
                else:
                    logger.warning(f"Unexpected tool_call format", tool_call=tc, tool_call_type=type(tc).__name__)
            
            # Only save tool_calls if they exist (don't save empty list)
            assistant_msg = {
                "role": "assistant",
                "content": response.content if hasattr(response, "content") else "",
            }
            if tool_calls_for_history:
                assistant_msg["tool_calls"] = tool_calls_for_history
            
            agent_history.append(assistant_msg)
            
            # Add ToolMessages for all tool calls (reasoning + actions)
            # CRITICAL: These must come immediately after AIMessage in history
            for result in action_results:
                tool_call_id = result["tool_call_id"]
                logger.debug(f"Adding ToolMessage to history", 
                           tool_call_id=tool_call_id,
                           output_preview=result["output"][:100])
                agent_history.append({
                    "role": "tool",
                    "content": result["output"],
                    "tool_call_id": tool_call_id
                })

        except Exception as e:
            logger.error(f"Research iteration {iteration} failed", error=str(e), exc_info=True)
            # If it's a tool call error, try to continue with next iteration
            if "tool output found" in str(e) or "tool call" in str(e).lower():
                logger.warning(f"Skipping iteration {iteration} due to tool call error, continuing...")
                # Add error message to history so LLM knows what happened
                agent_history.append({
                    "role": "tool",
                    "content": json.dumps({"error": f"Tool call failed: {str(e)}. Please retry with a new approach."}),
                    "tool_call_id": f"call_error_{iteration}"
                })
                continue
            else:
                # For other errors, break the loop
                break

    logger.info(
        "Research completed",
        sources=len(sources),
        scraped=len(scraped_content),
        iterations=iteration+1
    )

    return {
        "sources": sources,
        "scraped_content": scraped_content,
        "reasoning_history": reasoning_history
    }

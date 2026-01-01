"""Research agent with tool-calling loop (Perplexica-style).

Two-stage architecture: Researcher (gather info) → Writer (synthesize answer).
"""

import json
import structlog
from typing import Any, Dict, List
from pydantic import BaseModel, Field

from src.workflow.search.actions import ActionRegistry
from src.workflow.search.classifier import QueryClassification, get_current_date, format_chat_history

logger = structlog.get_logger(__name__)


# ==================== Schemas ====================


class ResearchAction(BaseModel):
    """Single research action with reasoning."""

    reasoning: str = Field(description="Chain-of-thought for this action")
    action: str = Field(description="Action name from registry")
    args: Dict[str, Any] = Field(default_factory=dict, description="Action arguments")


class ResearchPlan(BaseModel):
    """Research plan with multiple actions (for structured output)."""

    reasoning_preamble: str | None = Field(
        None,
        description="Chain-of-thought reasoning before actions (mandatory in balanced/quality)",
        alias="reasoning"
    )
    actions: List[ResearchAction] = Field(
        description="Sequence of research actions to execute"
    )
    done: bool = Field(
        default=False,
        description="Whether research is complete"
    )


# ==================== Mode-Specific Prompts ====================


def get_researcher_prompt(mode: str, iteration: int, max_iterations: int) -> str:
    """Get mode-specific researcher prompt."""

    current_date = get_current_date()

    base_prompt = f"""You are a research agent gathering information from the web.

Current date: {current_date}

Your knowledge cutoff is early 2024. For ANY information after that date, you MUST use web search.
"""

    if mode == "speed":
        return base_prompt + f"""
MODE: SPEED (iteration {iteration+1}/{max_iterations})

Your goal: Get information quickly and efficiently.

Strategy:
1. Make 1-2 targeted web searches with specific queries
2. Scrape 1-2 most promising URLs if needed
3. Call 'done' once you have enough information

IMPORTANT:
- You only have {max_iterations} iterations total
- Be concise and focused
- Don't over-analyze - gather key info and finish

Available actions: {list(ActionRegistry._actions.keys())}
"""

    elif mode == "balanced":
        return base_prompt + f"""
MODE: BALANCED (iteration {iteration+1}/{max_iterations})

Your goal: Thorough research with good coverage.

Strategy:
1. **MANDATORY**: Start with __reasoning_preamble explaining your thinking
2. Make 2-3 web searches from different angles
3. Scrape 2-3 most promising URLs
4. Iterate if gaps remain (you have {max_iterations} iterations)
5. Call 'done' when you have comprehensive information

IMPORTANT:
- EVERY response MUST start with __reasoning_preamble
- Explain: what you've learned, what gaps remain, next action
- Be thorough but efficient
- Minimum 2 information-gathering calls (search/scrape)

Available actions: {list(ActionRegistry._actions.keys())}

Example flow:
1. __reasoning_preamble: "Okay, the user wants to learn about X. I need to search for..."
2. web_search: ["query 1", "query 2"]
3. scrape_url: ["url1", "url2"]
4. __reasoning_preamble: "I've found information about A and B, but need more on C..."
5. web_search: ["query 3"]
6. done: "Research completed with comprehensive coverage"
"""

    else:  # quality
        return base_prompt + f"""
MODE: QUALITY (iteration {iteration+1}/{max_iterations})

Your goal: Exhaustive, deep research leaving no stone unturned.

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
3. Use 4-7 information-gathering calls (searches + scrapes)
4. Cross-reference from multiple authoritative sources
5. Call 'done' only when truly comprehensive (use most of your {max_iterations} iterations)

IMPORTANT:
- EVERY response MUST start with __reasoning_preamble
- Think deeply: what's missing? what angles are unexplored?
- Minimum 5-6 iterations before considering 'done'
- Be meticulous - this is quality mode

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
) -> Dict[str, Any]:
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
    tools = ActionRegistry.get_tool_definitions(
        mode=mode,
        classification=classification.query_type,
        context={}
    )

    # Track research state
    sources = []
    scraped_content = []
    reasoning_history = []
    agent_history = []

    # Add initial context
    history_context = format_chat_history(chat_history, limit=6)
    agent_history.append({
        "role": "user",
        "content": f"Chat history:\n{history_context}\n\nResearch query: {classification.standalone_query}"
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
            # Get researcher prompt for current iteration
            system_prompt = get_researcher_prompt(mode, iteration, max_iterations)

            # Build messages for LLM
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

            messages = [SystemMessage(content=system_prompt)]

            # Add conversation history
            for msg in agent_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    # LangChain expects specific format for tool calls
                    messages.append(AIMessage(content=msg.get("content", "")))
                elif msg["role"] == "tool":
                    messages.append(ToolMessage(
                        content=msg["content"],
                        tool_call_id=msg.get("tool_call_id", f"call_{iteration}")
                    ))

            # Stream tool calls from LLM
            if stream:
                stream.emit_status(f"Research iteration {iteration+1}/{max_iterations}", step="research")

            # Get LLM response with tool calling
            response = await llm.ainvoke(messages)

            # Extract tool calls
            tool_calls = []
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_calls = response.tool_calls

            # Check for done
            done = any(tc.get("name") == "done" or tc["function"].get("name") == "done" for tc in tool_calls)

            if not tool_calls or done:
                logger.info(f"Research completed at iteration {iteration+1}")
                break

            # Execute tools in parallel
            import asyncio

            action_results = []
            for tool_call in tool_calls:
                tool_name = tool_call.get("name") or tool_call.get("function", {}).get("name")
                tool_args = tool_call.get("args") or tool_call.get("function", {}).get("arguments", {})

                # Handle reasoning preamble specially
                if tool_name == "__reasoning_preamble":
                    reasoning = tool_args.get("reasoning", "")
                    reasoning_history.append(reasoning)
                    if stream:
                        stream.emit_agent_reasoning("researcher", reasoning[:300])
                    action_results.append({
                        "tool_call_id": tool_call.get("id", f"call_{iteration}"),
                        "output": json.dumps({"reasoning": reasoning})
                    })
                    continue

                # Execute action via registry
                try:
                    result = await ActionRegistry.execute(
                        tool_name,
                        tool_args,
                        {
                            "search_provider": search_provider,
                            "scraper": scraper,
                            "stream": stream,
                            "mode": mode,
                            "agent_id": "researcher"
                        }
                    )

                    # Track sources and content
                    if tool_name == "web_search" and "results" in result:
                        sources.extend(result["results"])
                    elif tool_name == "scrape_url" and "scraped" in result:
                        scraped_content.extend(result["scraped"])

                    action_results.append({
                        "tool_call_id": tool_call.get("id", f"call_{iteration}"),
                        "output": json.dumps(result)
                    })

                except Exception as e:
                    logger.warning(f"Action failed: {tool_name}", error=str(e))
                    action_results.append({
                        "tool_call_id": tool_call.get("id", f"call_{iteration}"),
                        "output": json.dumps({"error": str(e)})
                    })

            # Add to history
            agent_history.append({
                "role": "assistant",
                "content": response.content if hasattr(response, "content") else "",
                "tool_calls": tool_calls
            })

            for result in action_results:
                agent_history.append({
                    "role": "tool",
                    "content": result["output"],
                    "tool_call_id": result["tool_call_id"]
                })

        except Exception as e:
            logger.error(f"Research iteration {iteration} failed", error=str(e), exc_info=True)
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

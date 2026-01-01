"""Individual researcher agent with ReAct loop for LangGraph.

Each agent researches a specific topic using web search and scraping.
"""

import structlog
from typing import Any, Dict

from src.workflow.search.actions import ActionRegistry

logger = structlog.get_logger(__name__)


async def run_researcher_agent(
    agent_id: str,
    topic: str,
    state: Dict[str, Any],
    llm: Any,
    search_provider: Any,
    scraper: Any,
    stream: Any,
    max_steps: int = 6,
) -> Dict:
    """
    Run single researcher agent with ReAct loop.

    Args:
        agent_id: Unique agent identifier
        topic: Research topic
        state: Current graph state
        llm: LLM instance
        search_provider: Search provider
        scraper: Web scraper
        stream: Stream generator
        max_steps: Maximum ReAct steps

    Returns:
        Finding dict with agent_id, topic, summary, key_findings, sources
    """
    logger.info(f"Agent {agent_id} starting research", topic=topic)

    if stream:
        stream.emit_research_start({"researcher_id": agent_id, "topic": topic})

    # Agent memory
    todos = state.get("agent_todos", {}).get(agent_id, [])
    notes = []
    sources = []

    # ReAct history
    agent_history = []
    agent_history.append({
        "role": "user",
        "content": f"Research topic: {topic}\n\nTodos: {todos}"
    })

    # Get agent characteristics if available
    agent_characteristics = state.get("agent_characteristics", {}).get(agent_id, {})
    role = agent_characteristics.get("role", f"Research Agent {agent_id}")
    expertise = agent_characteristics.get("expertise", topic)
    personality = agent_characteristics.get("personality", "thorough and analytical researcher")

    system_prompt = f"""You are {role}.

Your expertise: {expertise}
Your approach: {personality}

Research topic: {topic}

Your job:
1. Use web_search to find sources related to your topic
2. Use scrape_url to get full content from promising URLs
3. Gather comprehensive information leveraging your expertise
4. Call 'done' when you have thorough coverage

Available actions: web_search, scrape_url, done

As a specialist in {expertise}, be thorough and cite sources.
"""

    # ReAct loop
    for step in range(max_steps):
        try:
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

            messages = [SystemMessage(content=system_prompt)]
            for msg in agent_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg.get("content", "")))
                elif msg["role"] == "tool":
                    messages.append(ToolMessage(
                        content=msg["content"],
                        tool_call_id=msg.get("tool_call_id", f"call_{step}")
                    ))

            # Get LLM response
            response = await llm.ainvoke(messages)

            # Extract tool calls
            tool_calls = []
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_calls = response.tool_calls

            # Check for done
            done = any(
                tc.get("name") == "done" or tc.get("function", {}).get("name") == "done"
                for tc in tool_calls
            )

            if done or not tool_calls:
                break

            # Execute tools
            action_results = []
            for tool_call in tool_calls:
                tool_name = tool_call.get("name") or tool_call.get("function", {}).get("name")
                tool_args = tool_call.get("args") or tool_call.get("function", {}).get("arguments", {})

                result = await ActionRegistry.execute(
                    tool_name,
                    tool_args,
                    {
                        "search_provider": search_provider,
                        "scraper": scraper,
                        "stream": stream,
                        "agent_id": agent_id,
                    }
                )

                # Track sources
                if tool_name == "web_search" and "results" in result:
                    sources.extend(result["results"])
                    if stream:
                        for src in result["results"]:
                            stream.emit_source_found({
                                "researcher_id": agent_id,
                                "url": src.get("url"),
                                "title": src.get("title")
                            })

                import json
                action_results.append({
                    "tool_call_id": tool_call.get("id", f"call_{step}"),
                    "output": json.dumps(result)
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
            logger.error(f"Agent {agent_id} step {step} failed", error=str(e))
            break

    # Synthesize finding
    summary = f"Researched {topic} using {len(sources)} sources."
    key_findings = [f"Finding about {topic}"]  # Simplified

    finding = {
        "agent_id": agent_id,
        "topic": topic,
        "summary": summary,
        "key_findings": key_findings,
        "sources": sources[:10],  # Limit sources
        "confidence": "medium",
    }

    logger.info(f"Agent {agent_id} completed", sources=len(sources))

    if stream:
        stream.emit_finding({
            "researcher_id": agent_id,
            "topic": topic,
            "summary": summary[:200]
        })

    return finding

"""Enhanced researcher agent with full memory integration and structured outputs."""

import asyncio
import json
from typing import Any, Dict, Optional
import structlog

from src.workflow.search.actions import ActionRegistry
from src.workflow.research.models import AgentPlan, AgentReflection
from src.models.agent_models import AgentNote

logger = structlog.get_logger(__name__)


async def run_researcher_agent_enhanced(
    agent_id: str,
    state: Dict[str, Any],
    llm: Any,
    search_provider: Any,
    scraper: Any,
    stream: Any,
    supervisor_queue: Any,
    max_steps: int = 8,
) -> Dict:
    """Enhanced researcher agent - main implementation."""
    return await _run_researcher_agent_impl(
        agent_id, state, llm, search_provider, scraper, stream, supervisor_queue, max_steps
    )


async def run_researcher_agent(
    agent_id: str,
    topic: str,
    state: Dict[str, Any],
    llm: Any,
    search_provider: Any,
    scraper: Any,
    stream: Any,
    max_steps: int = 8,
) -> Dict:
    """Backward compatibility wrapper for run_researcher_agent."""
    return await _run_researcher_agent_impl(
        agent_id, state, llm, search_provider, scraper, stream, None, max_steps
    )


async def _run_researcher_agent_impl(
    agent_id: str,
    state: Dict[str, Any],
    llm: Any,
    search_provider: Any,
    scraper: Any,
    stream: Any,
    supervisor_queue: Any,
    max_steps: int = 8,
) -> Dict:
    """
    Enhanced researcher agent with full memory integration.

    Features:
    - Loads agent file (character, todos, notes) from memory
    - Works on ONE task at a time (enforced)
    - Uses structured outputs with reasoning
    - Writes notes to markdown files
    - Plans and replans based on reflection
    - Signals supervisor when task complete

    Args:
        agent_id: Unique agent identifier
        state: Current graph state
        llm: LLM instance
        search_provider: Search provider
        scraper: Web scraper
        stream: Stream generator
        supervisor_queue: Queue for supervisor coordination
        max_steps: Maximum ReAct steps

    Returns:
        Finding dict with results
    """
    # Get memory services from runtime dependencies (passed via contextvars)
    # First try to get from state (if passed directly)
    agent_memory_service = state.get("agent_memory_service")
    agent_file_service = state.get("agent_file_service")
    
    # If not in state, try to get from runtime deps via contextvars
    if not agent_memory_service or not agent_file_service:
        from src.workflow.research.nodes import _get_runtime_deps
        runtime_deps = _get_runtime_deps()
        agent_memory_service = runtime_deps.get("agent_memory_service")
        agent_file_service = runtime_deps.get("agent_file_service")
    
    # Last resort: try to get from stream.app_state
    if (not agent_memory_service or not agent_file_service) and stream:
        if hasattr(stream, "app_state"):
            app_state = stream.app_state
            if isinstance(app_state, dict):
                agent_memory_service = agent_memory_service or app_state.get("agent_memory_service") or app_state.get("_agent_memory_service")
                agent_file_service = agent_file_service or app_state.get("agent_file_service") or app_state.get("_agent_file_service")
            else:
                agent_memory_service = agent_memory_service or getattr(app_state, "agent_memory_service", None) or getattr(app_state, "_agent_memory_service", None)
                agent_file_service = agent_file_service or getattr(app_state, "agent_file_service", None) or getattr(app_state, "_agent_file_service", None)

    if not agent_memory_service or not agent_file_service:
        logger.error(
            f"Agent {agent_id}: Memory services not available",
            stream_has_app_state=hasattr(stream, "app_state") if stream else False,
            stream_type=type(stream).__name__ if stream else "None",
            state_has_services="agent_memory_service" in state or "agent_file_service" in state
        )
        raise RuntimeError(
            f"Memory services not available for agent {agent_id}. "
            f"This is required for deep research to work. "
            f"Please check that agent_memory_service and agent_file_service are properly initialized."
        )

    # Load agent file
    agent_file = await agent_file_service.read_agent_file(agent_id)
    character = agent_file.get("character", "")
    preferences = agent_file.get("preferences", "")
    todos = agent_file.get("todos", [])

    # Get agent characteristics from state
    agent_characteristics = state.get("agent_characteristics", {}).get(agent_id, {})
    role = agent_characteristics.get("role", f"Research Agent {agent_id}")
    expertise = agent_characteristics.get("expertise", "general research")
    personality = agent_characteristics.get("personality", "thorough and analytical")

    logger.info(f"Agent {agent_id} loaded", role=role, expertise=expertise, todos_count=len(todos))

    # ENFORCE: Only one task at a time
    in_progress_tasks = [t for t in todos if t.status == "in_progress"]
    if len(in_progress_tasks) > 1:
        error_msg = f"Agent {agent_id} has multiple in_progress tasks: {[t.title for t in in_progress_tasks]}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Get current task
    if in_progress_tasks:
        current_task = in_progress_tasks[0]
        logger.info(f"Agent {agent_id} resuming task", task=current_task.title)
    else:
        # Get next pending task
        pending_tasks = [t for t in todos if t.status == "pending"]
        if not pending_tasks:
            logger.info(f"Agent {agent_id} has no pending tasks")
            return {
                "agent_id": agent_id,
                "topic": "no_tasks",
                "summary": "No pending tasks",
                "key_findings": [],
                "sources": [],
                "confidence": "n/a"
            }

        current_task = pending_tasks[0]
        # Mark as in_progress
        await agent_file_service.update_agent_todo(
            agent_id,
            current_task.title,
            status="in_progress"
        )
        logger.info(f"Agent {agent_id} starting task", task=current_task.title)

        # Reload to get updated todos
        agent_file = await agent_file_service.read_agent_file(agent_id)
        todos = agent_file.get("todos", [])

    # Emit initial state
    if stream:
        stream.emit_research_start({"researcher_id": agent_id, "topic": current_task.title})
        # Convert todos to dict format for emission
        todos_dict = [
            {
                "title": t.title,
                "status": t.status,
                "note": t.note,
                "url": t.url
            }
            for t in todos
        ]
        stream.emit_agent_todo(agent_id, todos_dict)

    # Create research plan with structured output
    plan_prompt = f"""You are {role} with expertise in {expertise}.

Current task: {current_task.title}
Objective: {current_task.objective}
Expected output: {current_task.expected_output}
Sources needed: {', '.join(current_task.sources_needed)}

Create a detailed research plan for completing this task.
"""

    try:
        plan = await llm.with_structured_output(AgentPlan).ainvoke([
            {"role": "system", "content": f"You are a research planning expert."},
            {"role": "user", "content": plan_prompt}
        ])
        logger.info(f"Agent {agent_id} created plan", goal=plan.current_goal)
    except Exception as e:
        logger.error(f"Agent {agent_id} plan creation failed", error=str(e))
        plan = AgentPlan(
            reasoning="Fallback plan due to error",
            current_goal=current_task.objective,
            next_steps=["Search for sources", "Analyze findings"],
            expected_findings="Research results",
            search_strategy="Broad web search",
            fallback_if_stuck="Try alternative sources"
        )

    # Research execution (ReAct loop)
    sources = []
    notes = []
    agent_history = []

    system_prompt = f"""You are {role}.

Expertise: {expertise}
Personality: {personality}
Character: {character}

Current task: {current_task.title}
Objective: {current_task.objective}

Research plan:
Goal: {plan.current_goal}
Next steps: {', '.join(plan.next_steps)}
Strategy: {plan.search_strategy}

CRITICAL INSTRUCTIONS FOR NOTES:
- When you save notes, ALWAYS include clickable links (URLs) to all sources you found
- In your notes, include:
  1. Links to all relevant sources you discovered
  2. Possible research directions or questions that need further investigation
  3. Specific areas that might need deeper exploration
  4. Any gaps or limitations you noticed in the information found
- Your notes should help guide future research and provide clear paths for deeper investigation

Available actions:
- web_search(query: str): Search the web
- scrape_url(url: str): Get full content from URL
- done(): Signal completion

Be thorough, cite sources with links, and fulfill the objective. Go DEEP, not just surface-level!
"""

    agent_history.append({
        "role": "user",
        "content": f"Execute research plan. Current goal: {plan.current_goal}"
    })

    # Get tool definitions for LLM
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field
    
    # Create LangChain tools from ActionRegistry
    def create_tool_from_action(action_name: str, action_def: dict):
        """Create LangChain StructuredTool from ActionRegistry action."""
        schema = action_def["args_schema"]
        
        # Create Pydantic model dynamically from schema
        fields = {}
        for prop_name, prop_def in schema.get("properties", {}).items():
            field_type = str  # Default to str
            if prop_def.get("type") == "integer":
                field_type = int
            elif prop_def.get("type") == "array":
                field_type = list
            elif prop_def.get("type") == "boolean":
                field_type = bool
            
            fields[prop_name] = (
                field_type,
                Field(description=prop_def.get("description", ""))
            )
        
        # Create dynamic model class
        ArgsModel = type(f"{action_name}Args", (BaseModel,), fields)
        
        async def tool_handler(**kwargs):
            context = {
                "search_provider": search_provider,
                "scraper": scraper,
                "stream": stream,
                "llm": llm,
                "agent_id": agent_id,
            }
            result = await ActionRegistry.execute(action_name, kwargs, context)
            # Convert result to string for ToolMessage (LangChain expects string)
            if isinstance(result, dict):
                import json
                return json.dumps(result)
            return str(result)
        
        return StructuredTool(
            name=action_name,
            description=action_def["description"],
            args_schema=ArgsModel,
            func=tool_handler,
            coroutine=tool_handler,
        )
    
    # Get enabled actions for deep research mode
    tools = []
    for action_name, action_def in ActionRegistry._actions.items():
        # Check if action is enabled (skip reasoning_preamble for deep research)
        if action_name == "__reasoning_preamble":
            continue
        enabled = action_def["enabled_condition"]({
            "mode": "quality",  # Deep research uses quality mode
            "classification": None,
        })
        if enabled:
            tools.append(create_tool_from_action(action_name, action_def))
    
    logger.debug(f"Agent {agent_id} tools prepared", tool_count=len(tools), tool_names=[t.name for t in tools])

    # ReAct loop
    for step in range(max_steps):
        try:
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

            messages = [SystemMessage(content=system_prompt)]
            for msg in agent_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    # Reconstruct AIMessage with tool_calls if present
                    content = msg.get("content", "")
                    tool_calls_data = msg.get("tool_calls", [])
                    if tool_calls_data:
                        # Convert dict tool_calls to ToolCall objects
                        from langchain_core.messages.tool import ToolCall
                        tool_calls = []
                        for tc in tool_calls_data:
                            if isinstance(tc, dict):
                                tool_calls.append(ToolCall(
                                    name=tc.get("name", ""),
                                    args=tc.get("args", {}),
                                    id=tc.get("id", f"call_{step}_{len(tool_calls)}")
                                ))
                            else:
                                tool_calls.append(tc)
                        messages.append(AIMessage(content=content, tool_calls=tool_calls))
                    else:
                        messages.append(AIMessage(content=content))
                elif msg["role"] == "tool":
                    messages.append(ToolMessage(
                        content=msg["content"],
                        tool_call_id=msg.get("tool_call_id", f"call_{step}")
                    ))

            # Bind tools to LLM (tools created before loop)
            try:
                if hasattr(llm, "bind_tools") and tools:
                    llm_with_tools = llm.bind_tools(tools)
                    logger.debug(f"Agent {agent_id} step {step}: bound {len(tools)} tools to LLM", 
                               tool_names=[t.name for t in tools])
                else:
                    llm_with_tools = llm
                    if not hasattr(llm, "bind_tools"):
                        logger.warning(f"LLM does not support bind_tools")
                    if not tools:
                        logger.warning(f"No tools available for binding")
            except Exception as e:
                logger.error(f"Failed to bind tools", error=str(e), exc_info=True)
                llm_with_tools = llm

            # Get LLM response
            response = await llm_with_tools.ainvoke(messages)

            # Extract tool calls - handle both ToolCall objects and dicts
            tool_calls = []
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_calls = response.tool_calls
                logger.debug(f"Agent {agent_id} step {step}: extracted {len(tool_calls)} tool calls")
            else:
                logger.warning(f"Agent {agent_id} step {step}: no tool_calls in response", 
                             response_type=type(response).__name__,
                             has_tool_calls=hasattr(response, "tool_calls"))

            # Check for done - handle both ToolCall objects and dicts
            done = False
            for tc in tool_calls:
                tool_name = None
                if hasattr(tc, "name"):
                    tool_name = tc.name
                elif isinstance(tc, dict):
                    tool_name = tc.get("name") or tc.get("function", {}).get("name")
                
                if tool_name == "done":
                    done = True
                    break

            if done or not tool_calls:
                if done:
                    logger.info(f"Agent {agent_id} signaled done", step=step)
                else:
                    logger.warning(f"Agent {agent_id} step {step}: no tool calls, ending", 
                                 response_content_preview=str(response.content)[:200] if hasattr(response, "content") else "no content")
                break

            # Execute tools - can be parallel if multiple independent tools called
            action_results = []
            
            # Helper to extract tool name and args from ToolCall object or dict
            def extract_tool_info(tool_call):
                if hasattr(tool_call, "name"):
                    return tool_call.name, tool_call.args if hasattr(tool_call, "args") else {}
                elif isinstance(tool_call, dict):
                    return (
                        tool_call.get("name") or tool_call.get("function", {}).get("name"),
                        tool_call.get("args") or tool_call.get("function", {}).get("arguments", {})
                    )
                else:
                    logger.error(f"Unknown tool_call format: {type(tool_call)}")
                    return None, {}
            
            # Check if tools can be executed in parallel (multiple web_search or scrape_url)
            tool_names = [extract_tool_info(tc)[0] for tc in tool_calls]
            can_parallelize = len(tool_calls) > 1 and all(
                name in ["web_search", "scrape_url"] for name in tool_names if name
            )
            
            if can_parallelize:
                # Execute all tools in parallel
                async def execute_tool(tool_call):
                    tool_name, tool_args = extract_tool_info(tool_call)
                    if not tool_name:
                        return {"error": "Could not extract tool name"}
                    
                    return await ActionRegistry.execute(
                        tool_name,
                        tool_args,
                        {
                            "search_provider": search_provider,
                            "scraper": scraper,
                            "stream": stream,
                            "llm": llm,
                            "agent_id": agent_id,
                        }
                    )
                
                tool_results = await asyncio.gather(
                    *[execute_tool(tc) for tc in tool_calls],
                    return_exceptions=True
                )
                
                # Process results
                for tool_call, result in zip(tool_calls, tool_results):
                    if isinstance(result, Exception):
                        logger.error(f"Agent {agent_id} tool failed", error=str(result))
                        result = {"error": str(result)}
                    
                    tool_name, tool_args = extract_tool_info(tool_call)

                    # Track sources and create notes
                    # Handle both dict and string results
                    if isinstance(result, str):
                        try:
                            result = json.loads(result)
                        except:
                            result = {"error": "Could not parse result"}

                    if tool_name == "web_search" and isinstance(result, dict) and "results" in result:
                        new_sources = result["results"]
                        sources.extend(new_sources)

                        if stream:
                            for src in new_sources:
                                stream.emit_source_found({
                                    "researcher_id": agent_id,
                                    "url": src.get("url"),
                                    "title": src.get("title")
                                })

                        # Create note for this search
                        if new_sources:
                            # Extract query from tool_args
                            queries = tool_args.get("queries", []) if isinstance(tool_args, dict) else []
                            query_text = queries[0] if queries else "search"
                            
                            note = AgentNote(
                                title=f"Search results for {query_text}",
                                summary=f"Found {len(new_sources)} sources: {', '.join([s.get('title', 'Unknown')[:50] for s in new_sources[:3]])}",
                                urls=[s.get("url") for s in new_sources[:5] if s.get("url")],
                                tags=["web_search", current_task.title]
                            )
                            # Save note to file
                            try:
                                file_path = await agent_memory_service.save_agent_note(note, agent_id)
                                logger.info(f"Agent {agent_id} saved note", file_path=file_path)
                            except Exception as e:
                                logger.error(f"Agent {agent_id} failed to save note", error=str(e))

                            notes.append(note)

                            # Emit note to UI
                            if stream:
                                stream.emit_agent_note(agent_id, {
                                    "title": note.title,
                                    "summary": note.summary,
                                    "urls": note.urls,
                                    "shared": True
                                })

                    # Extract tool_call_id
                    tool_call_id = None
                    if hasattr(tool_call, "id"):
                        tool_call_id = tool_call.id
                    elif isinstance(tool_call, dict):
                        tool_call_id = tool_call.get("id")
                    
                    action_results.append({
                        "tool_call_id": tool_call_id or f"call_{step}_{len(action_results)}",
                        "output": json.dumps(result) if not isinstance(result, str) else result
                    })
            else:
                # Execute tools sequentially (default for mixed tool types)
                for tool_call in tool_calls:
                    tool_name, tool_args = extract_tool_info(tool_call)
                    if not tool_name:
                        logger.error(f"Agent {agent_id} step {step}: could not extract tool name", tool_call_type=type(tool_call).__name__)
                        continue

                    result = await ActionRegistry.execute(
                        tool_name,
                        tool_args,
                        {
                            "search_provider": search_provider,
                            "scraper": scraper,
                            "stream": stream,
                            "llm": llm,
                            "agent_id": agent_id,
                        }
                    )

                    # Track sources and create notes
                    # Handle both dict and string results
                    if isinstance(result, str):
                        try:
                            result = json.loads(result)
                        except:
                            result = {"error": "Could not parse result"}

                    if tool_name == "web_search" and isinstance(result, dict) and "results" in result:
                        new_sources = result["results"]
                        sources.extend(new_sources)

                        if stream:
                            for src in new_sources:
                                stream.emit_source_found({
                                    "researcher_id": agent_id,
                                    "url": src.get("url"),
                                    "title": src.get("title")
                                })

                        # Create note for this search
                        if new_sources:
                            # Extract query from tool_args
                            queries = tool_args.get("queries", []) if isinstance(tool_args, dict) else []
                            query_text = queries[0] if queries else "search"
                            
                            note = AgentNote(
                                title=f"Search results for {query_text}",
                                summary=f"Found {len(new_sources)} sources: {', '.join([s.get('title', 'Unknown')[:50] for s in new_sources[:3]])}",
                                urls=[s.get("url") for s in new_sources[:5] if s.get("url")],
                                tags=["web_search", current_task.title]
                            )
                            # Save note to file
                            try:
                                file_path = await agent_memory_service.save_agent_note(note, agent_id)
                                logger.info(f"Agent {agent_id} saved note", file_path=file_path)
                            except Exception as e:
                                logger.error(f"Agent {agent_id} failed to save note", error=str(e))

                            notes.append(note)

                            # Emit note to UI
                            if stream:
                                stream.emit_agent_note(agent_id, {
                                    "title": note.title,
                                    "summary": note.summary,
                                    "urls": note.urls,
                                    "shared": True
                                })

                    # Extract tool_call_id
                    tool_call_id = None
                    if hasattr(tool_call, "id"):
                        tool_call_id = tool_call.id
                    elif isinstance(tool_call, dict):
                        tool_call_id = tool_call.get("id")
                    
                    action_results.append({
                        "tool_call_id": tool_call_id or f"call_{step}_{len(action_results)}",
                        "output": json.dumps(result) if not isinstance(result, str) else result
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

            # Periodic reflection (every 3 steps)
            if step % 3 == 2 and step > 0:
                reflection_prompt = f"""Reflect on your research progress.

Current task: {current_task.title}
Objective: {current_task.objective}
Sources found so far: {len(sources)}
Notes created: {len(notes)}

Assess your progress and whether you need to adjust your approach.
"""
                try:
                    reflection = await llm.with_structured_output(AgentReflection).ainvoke([
                        {"role": "system", "content": "You are a reflective researcher."},
                        {"role": "user", "content": reflection_prompt}
                    ])

                    logger.info(
                        f"Agent {agent_id} reflection",
                        assessment=reflection.progress_assessment,
                        should_replan=reflection.should_replan
                    )

                    # Replan if needed
                    if reflection.should_replan and reflection.new_direction:
                        replan_prompt = f"""Your current approach needs adjustment.

Previous plan: {plan.current_goal}
New direction: {reflection.new_direction}

Create an updated research plan incorporating this new direction.
"""
                        plan = await llm.with_structured_output(AgentPlan).ainvoke([
                            {"role": "system", "content": "You are a research planning expert."},
                            {"role": "user", "content": replan_prompt}
                        ])
                        logger.info(f"Agent {agent_id} replanned", new_goal=plan.current_goal)

                        # Update agent history with new direction
                        agent_history.append({
                            "role": "user",
                            "content": f"Revised plan: {plan.current_goal}. Adjust your approach."
                        })

                except Exception as e:
                    logger.error(f"Agent {agent_id} reflection failed", error=str(e))

        except Exception as e:
            logger.error(f"Agent {agent_id} step {step} failed", error=str(e))
            break

    # Task completion
    summary = f"Completed '{current_task.title}' with {len(sources)} sources and {len(notes)} research notes."
    key_findings = [note.summary for note in notes[:5]]

    finding = {
        "agent_id": agent_id,
        "topic": current_task.title,
        "summary": summary,
        "key_findings": key_findings,
        "sources": sources[:20],
        "confidence": "high" if len(sources) >= 5 else "medium",
        "notes": notes
    }

    # Mark task as done
    await agent_file_service.update_agent_todo(
        agent_id,
        current_task.title,
        status="done",
        note=f"Completed with {len(sources)} sources"
    )

    # Reload todos
    agent_file = await agent_file_service.read_agent_file(agent_id)
    todos = agent_file.get("todos", [])

    # Emit completion
    if stream:
        todos_dict = [
            {
                "title": t.title,
                "status": t.status,
                "note": t.note,
                "url": t.url
            }
            for t in todos
        ]
        stream.emit_agent_todo(agent_id, todos_dict)

        # Final note
        final_note = AgentNote(
            title=f"Task complete: {current_task.title}",
            summary=summary,
            urls=[s.get("url") for s in sources[:5] if s.get("url")],
            tags=["task_complete"]
        )
        await agent_memory_service.save_agent_note(final_note, agent_id)

        stream.emit_agent_note(agent_id, {
            "title": final_note.title,
            "summary": final_note.summary,
            "urls": final_note.urls,
            "shared": True
        })

        stream.emit_finding({
            "researcher_id": agent_id,
            "topic": current_task.title,
            "summary": summary[:200]
        })

    logger.info(f"Agent {agent_id} completed task", task=current_task.title, sources=len(sources))

    # Signal supervisor queue
    if supervisor_queue:
        await supervisor_queue.agent_completed_task(
            agent_id=agent_id,
            task_title=current_task.title,
            result=finding
        )
        logger.info(f"Agent {agent_id} queued for supervisor review")

    return finding

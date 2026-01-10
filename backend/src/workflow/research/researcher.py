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
    max_steps: int = None,  # If None, will use settings.deep_research_agent_max_steps (old default: 8)
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
    max_steps: int = None,  # If None, will use settings.deep_research_agent_max_steps (old default: 8)
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
    
    # Get recent notes - LIMIT to prevent context bloat
    # Only include last 10 most recent important notes in context
    all_notes = agent_file.get("notes", [])
    # Notes are already limited to 20 in read_agent_file, but we limit further for context
    recent_notes = all_notes[-10:] if len(all_notes) > 10 else all_notes
    notes_context = "\n".join([f"- {note}" for note in recent_notes]) if recent_notes else "No previous notes."
    
    # Log note count for debugging
    if all_notes:
        logger.debug(f"Agent {agent_id} notes", total=agent_file.get("all_notes_count", len(all_notes)), in_context=len(recent_notes))

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

    # Get original user query and context from state
    original_query = state.get("query", "")
    user_language = state.get("user_language", "English")
    deep_search_result_raw = state.get("deep_search_result", "")
    if isinstance(deep_search_result_raw, dict):
        deep_search_result = deep_search_result_raw.get("value", "") if isinstance(deep_search_result_raw, dict) else ""
    else:
        deep_search_result = deep_search_result_raw or ""
    
    # Extract user clarification answers from chat history
    clarification_context = ""
    chat_history = state.get("chat_history", [])
    if chat_history:
        for i, msg in enumerate(chat_history):
            if msg.get("role") == "assistant" and ("clarification" in msg.get("content", "").lower() or "🔍" in msg.get("content", "")):
                if i + 1 < len(chat_history) and chat_history[i + 1].get("role") == "user":
                    user_answer = chat_history[i + 1].get("content", "")
                    clarification_context = f"\n\n**USER CLARIFICATION ANSWERS:**\n{user_answer}\n"
                    break

    system_prompt = f"""You are {role}.

Expertise: {expertise}
Personality: {personality}
Character: {character}

**IMPORTANT: Respond in {user_language} whenever generating text for the user.**

**CRITICAL: YOU DO NOT HAVE ACCESS TO THE ORIGINAL USER QUERY OR CHAT HISTORY**
- You ONLY see the task assigned to you below
- Use the task description to understand what you need to research
- The task objective and guidance contain all the context you need

Current task: {current_task.title}
Objective: {current_task.objective}
Guidance: {current_task.note if hasattr(current_task, 'note') else 'No specific guidance provided'}

Research plan:
Goal: {plan.current_goal}
Next steps: {', '.join(plan.next_steps)}
Strategy: {plan.search_strategy}

**NOTE: You do NOT have access to the original user query or chat history.**
**Your task description above contains all the context you need to complete this research.**

**CRITICAL: When you find information, provide DETAILED, COMPREHENSIVE summaries with full context, not just links.**
- Include specific facts, data, and insights in your summaries
- Explain what you found and why it's relevant
- Provide full context, not just "found X sources"
- Your findings should be self-contained and informative

Your previous notes (recent important findings):
{notes_context}

CRITICAL INSTRUCTIONS FOR NOTES AND MEMORY:
- **DO NOT save notes automatically** - you must THINK and decide what's truly important
- **ONLY save notes when you have SUBSTANTIAL, ACTIONABLE INFORMATION**:
  1. Key discoveries, important facts, or significant insights that answer the research question
  2. Critical information that directly relates to your current task objective
  3. Important patterns, trends, or conclusions you've identified
  4. Gaps or limitations that need further investigation
  5. Research directions or questions that are critical for completing the task
- **NEVER save routine notes** like:
  - "Found X sources" (this is not information, just metadata)
  - "Search: query" (this is not a finding)
  - Lists of URLs without context (this is not useful)
  - Generic summaries without specific facts
- **When you DO save a note, it must contain**:
  - Specific facts, data, or insights (not just "found sources")
  - Clear explanation of WHY this information is important
  - How it relates to the research objective
  - Clickable links (URLs) to all sources
- **Your notes should help guide future research** - they must be informative and actionable
- **Think before saving**: "Does this note contain valuable information that will help complete the research?" If not, don't save it.

Available actions:
- web_search(queries: list[str]): Search the web with natural queries (write as you would in a browser)
- scrape_url(urls: list[str]): Get full content from URLs
- done(): Signal completion

**CRITICAL: Evaluating search results before scraping:**
- When you receive search results, CAREFULLY evaluate each result's RELEVANCE to your current task
- Look at the TITLE and SNIPPET - do they relate to your task objective?
- Only scrape URLs that are CLEARLY relevant to your task
- If a result's title/snippet doesn't match your task, SKIP it - don't waste time scraping irrelevant content
- Example: If your task is "ВВС Германии техника", skip results about "ВВС США" or "техника вообще"
- Focus on scraping sources that directly help answer your specific task

**CRITICAL: Query reformulation strategy (to avoid getting stuck):**
- If your search results are NOT relevant to your task, you MUST try DIFFERENT search queries
- Don't repeat the same query multiple times - reformulate it with different keywords or phrasing
- Try different angles: synonyms, related terms, more specific or more general queries
- Example: If "ВВС Германии" gives irrelevant results, try "Luftwaffe техника", "немецкая военная авиация", "современные самолеты Германии"
- You have up to {max_steps} steps - use them ALL to thoroughly investigate the topic from multiple angles
- **MANDATORY**: Use your full step limit to dig DEEP - don't stop after finding basic information

**CRITICAL: VERIFICATION AND DEEP RESEARCH REQUIREMENTS:**
- **MANDATORY**: All important claims, facts, and data MUST be verified in MULTIPLE independent sources
- Never rely on a single source - always cross-reference important information
- If you find a key fact or claim, search for it in different sources to verify accuracy
- **DIG DEEP**: Don't stop at surface-level information - investigate:
  * Technical specifications and detailed parameters
  * Expert opinions and critical analysis from multiple perspectives
  * Historical context and evolution
  * Real-world case studies and practical applications
  * Comparative analysis with alternatives
  * Controversial aspects and different viewpoints
- **VERIFICATION STRATEGY**: For each important finding:
  1. Find the information in the first source
  2. Search for the same information in 2-3 additional independent sources
  3. Compare findings - note any discrepancies or different perspectives
  4. Document all sources that confirm or contradict the finding
- **DEEP DIVE**: When you find interesting information, don't just note it - investigate related aspects:
  * If you find technical specs, also find expert analysis of those specs
  * If you find historical info, also find current state and future trends
  * If you find one perspective, also find alternative or critical viewpoints
  * If you find general info, dig into specific examples and case studies
- **USE ALL YOUR STEPS**: You have {max_steps} steps - use them to thoroughly research, verify, and cross-reference information
- Only signal done() when you have:
  * Verified important claims in multiple sources
  * Explored the topic from multiple angles
  * Found specific examples, case studies, and detailed information
  * Cross-referenced key findings across different sources

Be thorough, cite sources with links, verify everything in multiple sources, and fulfill the objective. Go DEEP, not just surface-level!
"""

    agent_history.append({
        "role": "user",
        "content": f"Execute research plan. Current goal: {plan.current_goal}"
    })

    # Get tool definitions for LLM
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field, create_model
    from typing import Any
    
    # Create LangChain tools from ActionRegistry
    def create_tool_from_action(action_name: str, action_def: dict):
        """Create LangChain StructuredTool from ActionRegistry action."""
        schema = action_def["args_schema"]
        
        # Create Pydantic model dynamically from schema
        field_definitions = {}
        for prop_name, prop_def in schema.get("properties", {}).items():
            # Determine field type with proper annotations
            if prop_def.get("type") == "integer":
                field_type = int
            elif prop_def.get("type") == "array":
                # Check items type for arrays
                items_type = prop_def.get("items", {}).get("type", "string")
                if items_type == "string":
                    field_type = list[str]
                else:
                    field_type = list[Any]
            elif prop_def.get("type") == "boolean":
                field_type = bool
            else:
                field_type = str  # Default to str
            
            # Use tuple format for create_model: (type, Field(...))
            field_definitions[prop_name] = (
                field_type,
                Field(description=prop_def.get("description", ""))
            )
        
        # Create dynamic model class using create_model (Pydantic v2 way)
        ArgsModel = create_model(
            f"{action_name}Args",
            **field_definitions
        )
        
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

    # Get max_steps from settings if not provided
    if max_steps is None:
        from src.workflow.research.nodes import _get_runtime_deps
        runtime_deps = _get_runtime_deps()
        settings = runtime_deps.get("settings")
        if settings:
            max_steps = settings.deep_research_agent_max_steps
        else:
            max_steps = 5  # Default fallback
    
    logger.info(f"Agent {agent_id} starting ReAct loop", max_steps=max_steps)

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

                        # DO NOT automatically create notes for every search
                        # Notes should only be created when agent finds IMPORTANT information
                        # The agent will decide what to save based on actual findings
                        # We only track sources here for the agent's context
                    
                    # Handle scrape_url results - DO NOT automatically create notes
                    # The agent should analyze scraped content and decide what's important to save
                    # Only save notes when agent explicitly identifies valuable information

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

                        # DO NOT automatically create notes for search results
                        # Agent should analyze results and decide what's important to save
                        # Notes should only be created when agent finds IMPORTANT information
                    
                    # Handle scrape_url results - DO NOT automatically create notes
                    # Agent should analyze scraped content and decide what's important to save
                    # Only save notes when agent explicitly identifies valuable information

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

    # Task completion - create finding with ONLY useful information, NO metadata spam
    # Filter out metadata and garbage, keep only actual findings
    
    # Extract REAL findings from sources (snippets with actual information, not just titles)
    real_findings_from_sources = []
    for src in sources:
        snippet = src.get("snippet", "").strip()
        title = src.get("title", "").strip()
        
        # Skip if snippet is too short or just metadata
        if not snippet or len(snippet) < 30:
            continue
        
        # Skip if snippet is just metadata (contains "found", "sources", "query" etc.)
        snippet_lower = snippet.lower()
        is_metadata = any([
            "found" in snippet_lower and "sources" in snippet_lower,
            "search:" in snippet_lower,
            "query:" in snippet_lower,
            snippet_lower.count("http") > 1,  # Multiple URLs = likely metadata
        ])
        
        if not is_metadata and len(snippet) > 30:
            # Extract meaningful information
            finding_text = f"{title}: {snippet[:250]}" if title else snippet[:250]
            real_findings_from_sources.append(finding_text)
    
    # Extract REAL findings from notes (only informative ones, not metadata)
    important_notes = []
    for note in notes:
        if not note.summary or len(note.summary) < 100:
            continue
        
        # Skip metadata notes
        summary_lower = note.summary.lower()
        is_metadata = any([
            "found" in summary_lower and "sources" in summary_lower and "query" in summary_lower,
            "search:" in note.title.lower() and len(note.summary) < 150,
            "key sources:" in summary_lower and len(note.summary) < 200,
        ])
        
        if not is_metadata:
            important_notes.append(note)
    
    # Build summary with ONLY real findings, NO metadata
    # CRITICAL: Create comprehensive summary with detailed information, not just links
    summary_parts = []
    
    # Add comprehensive findings from sources with full context
    if real_findings_from_sources:
        # Include more detailed information from sources
        detailed_findings = []
        for finding in real_findings_from_sources[:15]:  # More findings for comprehensive summary
            # Extract more context from sources
            for src in sources:
                snippet = src.get("snippet", "").strip()
                title = src.get("title", "").strip()
                if finding in snippet or (title and finding.startswith(title)):
                    # Include full snippet if available
                    if len(snippet) > 100:
                        detailed_findings.append(f"{title}: {snippet[:400]}")
                    else:
                        detailed_findings.append(finding)
                    break
            else:
                detailed_findings.append(finding)
        
        findings_text = "\n\n".join([f"• {f}" for f in detailed_findings[:12]])
        summary_parts.append(f"**Detailed Research Findings:**\n\n{findings_text}")
    
    # Add comprehensive findings from notes with full context
    if important_notes:
        notes_text = "\n\n".join([
            f"**{note.title}:**\n{note.summary[:500]}{'...' if len(note.summary) > 500 else ''}" 
            for note in important_notes[:10]  # More notes for comprehensive summary
        ])
        summary_parts.append(f"\n**Important Discoveries:**\n\n{notes_text}")
    
    # If no real findings, indicate that research needs to go deeper
    if not summary_parts:
        summary_parts.append(f"Research completed on '{current_task.title}' but no substantial findings extracted. May need deeper investigation.")
    
    # Create comprehensive summary (NO metadata like "Found X sources")
    summary = "\n\n".join(summary_parts)
    
    # Ensure summary is substantial (at least 200 chars) - if too short, expand it
    if len(summary) < 200 and sources:
        # Add more context from sources
        additional_context = []
        for src in sources[:5]:
            snippet = src.get("snippet", "").strip()
            title = src.get("title", "").strip()
            if snippet and len(snippet) > 50:
                additional_context.append(f"{title}: {snippet[:300]}")
        if additional_context:
            summary += "\n\n**Additional Context:**\n\n" + "\n\n".join([f"• {ctx}" for ctx in additional_context])
    
    # Extract key findings - ONLY real information, NO metadata
    key_findings = []
    
    # From sources - only real findings
    for finding in real_findings_from_sources[:12]:
        key_findings.append(finding)
    
    # From notes - only informative ones
    for note in important_notes[:8]:
        if note.summary and len(note.summary) > 100:
            key_findings.append(f"{note.title}: {note.summary[:200]}")
    
    # Filter sources - keep only ones with actual content (NO metadata spam)
    filtered_sources = []
    for src in sources:
        snippet = src.get("snippet", "").strip()
        if snippet and len(snippet) > 30:
            snippet_lower = snippet.lower()
            # Skip metadata sources
            if not ("found" in snippet_lower and "sources" in snippet_lower):
                filtered_sources.append(src)
    
    # Use filtered sources, but keep reasonable limit
    useful_sources = filtered_sources[:30] if filtered_sources else sources[:10]
    
    # Create finding - ONLY real information, NO metadata spam
    finding = {
        "agent_id": agent_id,
        "topic": current_task.title,
        "summary": summary,  # Only real findings, no metadata
        "key_findings": key_findings,  # Only real findings, filtered
        "sources": useful_sources,  # Only sources with actual content
        "confidence": "high" if len(useful_sources) >= 5 else "medium",
        "notes": important_notes,  # Only informative notes, filtered
        "sources_count": len(useful_sources),
        "notes_count": len(important_notes),
        "key_findings_count": len(key_findings)
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
        # CRITICAL: Pass agent_file_service so note is added to agent's personal file
        await agent_memory_service.save_agent_note(final_note, agent_id, agent_file_service=agent_file_service)

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

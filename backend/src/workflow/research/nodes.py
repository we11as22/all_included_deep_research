"""LangGraph nodes for deep research workflow with structured outputs.

Each node is an async function that takes state and returns state updates.
All enhanced nodes use structured outputs with reasoning fields.
"""

import asyncio
import contextvars
import structlog
from typing import Any, Dict

from langchain_core.messages import SystemMessage, HumanMessage

# Context variable for runtime dependencies (not serialized in state)
runtime_deps_context = contextvars.ContextVar('runtime_deps', default=None)


def _get_runtime_deps() -> Dict[str, Any]:
    """Get runtime dependencies from context variable."""
    deps = runtime_deps_context.get()
    return deps or {}


def _restore_runtime_deps(state: Dict[str, Any]) -> Dict[str, Any]:
    """Restore runtime dependencies to state from context variable."""
    deps = _get_runtime_deps()
    for key, value in deps.items():
        if value is not None and key not in state:
            state[key] = value
    return state

from src.workflow.research.state import (
    ResearchState,
    CompressedFindings,
)
from src.workflow.research.models import (
    QueryAnalysis,
    ResearchPlan,
    ResearchTopic,
    AgentCharacteristics,
    AgentCharacteristic,
    AgentTodo,
    SupervisorAssessment,
    AgentDirective,
    FinalReport,
    ReportSection,
    ReportValidation,
    ClarificationNeeds,
)
from src.workflow.research.supervisor_queue import SupervisorQueue
from src.workflow.research.researcher import run_researcher_agent_enhanced
from src.workflow.research.supervisor_agent import run_supervisor_agent
from src.models.agent_models import AgentTodoItem

logger = structlog.get_logger(__name__)


# ==================== Memory Search Node ====================

async def search_memory_node(state: ResearchState) -> Dict:
    """Search vector memory for relevant context."""
    # Restore runtime dependencies if not in state
    state = _restore_runtime_deps(state)
    
    query = state["query"]
    stream = state.get("stream")

    if stream:
        stream.emit_status("Searching memory...", step="memory")

    # TODO: Implement actual memory search when vector store is integrated
    # For now, return empty context
    memory_results = []

    logger.info("Memory search completed", results=len(memory_results))

    return {
        "memory_context": {"type": "override", "value": memory_results},
    }


# ==================== Deep Search Node ====================

async def run_deep_search_node(state: ResearchState) -> Dict:
    """Run deep search to gather initial context before planning."""
    # Restore runtime dependencies if not in state
    state = _restore_runtime_deps(state)
    
    query = state["query"]
    chat_history = state.get("chat_history", [])
    stream = state.get("stream")

    if stream:
        stream.emit_status("Running deep search for initial context...", step="deep_search")

    logger.info("Running deep search before agent spawning")

    try:
        # Import search service
        from src.workflow.search.service import SearchService

        # Get dependencies from state
        llm = state.get("llm")
        search_provider = state.get("search_provider")
        scraper = state.get("scraper")

        if not all([llm, search_provider, scraper]):
            logger.warning("Missing dependencies for deep search, skipping")
            return {"deep_search_result": {"type": "override", "value": None}}

        # Create search service
        search_service = SearchService(
            classifier_llm=llm,
            research_llm=llm,
            writer_llm=llm,
            search_provider=search_provider,
            scraper=scraper,
        )

        # Run deep search
        result = await search_service._answer_deep_search(
            query=query,
            classification=None,  # Will be created internally
            stream=stream,
            chat_history=chat_history,
        )

        logger.info("Deep search completed", answer_length=len(result) if result else 0)

        # Emit deep search result to frontend via stream
        if stream and result:
            stream.emit_status(f"Deep search completed. Found {len(result)} characters of context.", step="deep_search")
            # Also emit as a report chunk so user can see it
            stream.emit_report_chunk(f"## Initial Deep Search Context\n\n{result}\n\n---\n\n*This context will be used to guide the research agents.*\n\n")

        return {
            "deep_search_result": {"type": "override", "value": result},
        }

    except Exception as e:
        logger.error("Deep search failed", error=str(e), exc_info=True)
        return {"deep_search_result": {"type": "override", "value": None}}


# ==================== Clarifying Questions Node ====================

async def clarify_with_user_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ask clarifying questions to user if needed before starting research.
    
    This helps narrow down research scope and ensure we understand the query correctly.
    """
    query = state.get("query", "")
    # Get deep_search_result - handle both dict and string formats
    deep_search_result_raw = state.get("deep_search_result", "")
    if isinstance(deep_search_result_raw, dict):
        # If it's a dict with "type": "override", extract the value
        deep_search_result = deep_search_result_raw.get("value", "") if isinstance(deep_search_result_raw, dict) else ""
    else:
        deep_search_result = deep_search_result_raw or ""
    chat_history = state.get("chat_history", [])
    # Restore runtime dependencies if not in state
    state = _restore_runtime_deps(state)
    
    llm = state.get("llm")
    stream = state.get("stream")
    settings = state.get("settings")
    
    # Check if clarifying questions are enabled
    if not settings or not settings.deep_research_enable_clarifying_questions:
        logger.info("Clarifying questions disabled, skipping")
        return {"clarification_needed": False}
    
    # Check if we already asked questions (look at chat history)
    # Only skip if there are multiple exchanges (user has already answered)
    if len(chat_history) > 4:  # More than 2 exchanges (user + assistant pairs)
        # User has already interacted extensively, skip clarification
        logger.info("Chat history is extensive, skipping clarification")
        return {"clarification_needed": False}
    
    if stream:
        stream.emit_status("Analyzing if clarification is needed...", step="clarification")
        logger.info("Clarification node: deep_search_result available", 
                   has_result=bool(deep_search_result), 
                   result_length=len(deep_search_result) if deep_search_result else 0,
                   result_preview=deep_search_result[:200] if deep_search_result else "")
    
    # Analyze if clarification is needed
    # Deep search result is already a synthesized answer, use it fully for context
    from src.utils.text import summarize_text
    deep_search_summary = summarize_text(deep_search_result, 1000) if deep_search_result and len(deep_search_result) > 1000 else deep_search_result
    deep_search_context = f"\n\nDeep search provided this context:\n{deep_search_summary}" if deep_search_summary else ""
    logger.info("Deep search context prepared", context_length=len(deep_search_context))
    
    prompt = f"""Analyze if this research query needs clarification from the user.

Query: {query}
{deep_search_context}

Assess whether:
1. The query is clear and specific enough to conduct research
2. There are ambiguous terms that need clarification
3. The scope is well-defined or too broad

If clarification is needed, ask 1-2 specific questions.
If you can proceed with reasonable assumptions, indicate so.
"""
    
    try:
        clarification = await llm.with_structured_output(ClarificationNeeds).ainvoke([
            {"role": "system", "content": "You are a research planning expert."},
            {"role": "user", "content": prompt}
        ])
        
        logger.info(
            "Clarification analysis",
            needs_clarification=clarification.needs_clarification,
            questions_count=len(clarification.questions),
            can_proceed=clarification.can_proceed_without
        )
        
        if clarification.needs_clarification and clarification.questions:
            # Send clarifying questions to user via stream
            if stream:
                questions_text = "\n\n".join([
                    f"**Q{i+1}:** {q.question}\n\n*Why needed:* {q.why_needed}\n\n*Default assumption if not answered:* {q.default_assumption}"
                    for i, q in enumerate(clarification.questions)
                ])
                
                clarification_message = f"""## 🔍 Clarification Needed

Before starting the research, I need to clarify a few points:

{questions_text}

---

*Note: Research will proceed with the default assumptions listed above. 
These questions help guide the research direction.*
"""
                
                stream.emit_status(clarification_message, step="clarification")
                
                # Emit as report chunk so it's visible to user in the main chat
                stream.emit_report_chunk(clarification_message)
                
                # Also emit as a separate message event to ensure it's visible
                logger.info("Clarifying questions sent to user", 
                           questions_count=len(clarification.questions),
                           message_length=len(clarification_message))
            
            # Check if we can proceed without answers
            if not clarification.can_proceed_without:
                logger.warning("Clarification needed but cannot proceed without - proceeding anyway with assumptions")
            
            # For now, always proceed with assumptions
            # To implement true interactive: would need to:
            # 1. Pause graph execution
            # 2. Store graph state in checkpoint
            # 3. Wait for user response via API
            # 4. Resume graph with user answers
            logger.info("Clarifying questions shown, proceeding with default assumptions")
            return {
                "clarification_needed": False,  # Always proceed for now
                "clarification_questions": [q.dict() for q in clarification.questions]  # Store for reference
            }
        
        return {"clarification_needed": False}
        
    except Exception as e:
        logger.error("Clarification analysis failed", error=str(e))
        return {"clarification_needed": False}


# ==================== Analysis Node (Enhanced) ====================

async def analyze_query_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze query to determine research approach.

    Uses structured output to assess query complexity and plan.
    """
    # Restore runtime dependencies if not in state
    state = _restore_runtime_deps(state)
    
    query = state.get("query", "")
    # Restore runtime dependencies if not in state
    state = _restore_runtime_deps(state)
    
    llm = state.get("llm")
    stream = state.get("stream")

    if stream:
        stream.emit_status("Analyzing query complexity...", step="analysis")

    prompt = f"""Analyze this research query to determine the best approach:

Query: {query}

Assess:
1. What are the main topics/subtopics?
2. How complex is this query?
3. Do we need deep search context first?
4. How many specialized research agents would be optimal?
"""

    try:
        analysis = await llm.with_structured_output(QueryAnalysis).ainvoke([
            {"role": "system", "content": "You are an expert research planner."},
            {"role": "user", "content": prompt}
        ])

        logger.info(
            "Query analyzed",
            complexity=analysis.complexity,
            agent_count=analysis.estimated_agent_count,
            topics=analysis.topics
        )

        return {
            "query_analysis": analysis.dict(),
            "requires_deep_search": analysis.requires_deep_search,
            "estimated_agent_count": analysis.estimated_agent_count
        }

    except Exception as e:
        logger.error("Query analysis failed", error=str(e))
        # Fallback
        return {
            "query_analysis": {
                "reasoning": "Fallback analysis due to error",
                "topics": [query],
                "complexity": "moderate",
                "requires_deep_search": True,
                "estimated_agent_count": 4
            },
            "requires_deep_search": True,
            "estimated_agent_count": 4
        }


# ==================== Planning Node (Enhanced) ====================

async def plan_research_enhanced_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create detailed research plan with structured output.

    Generates research topics, priorities, and coordination strategy.
    """
    # Restore runtime dependencies if not in state
    state = _restore_runtime_deps(state)
    
    query = state.get("query", "")
    query_analysis = state.get("query_analysis", {})
    # Get deep_search_result - handle both dict and string formats
    deep_search_result_raw = state.get("deep_search_result", "")
    if isinstance(deep_search_result_raw, dict):
        # If it's a dict with "type": "override", extract the value
        deep_search_result = deep_search_result_raw.get("value", "") if isinstance(deep_search_result_raw, dict) else ""
    else:
        deep_search_result = deep_search_result_raw or ""
    # Restore runtime dependencies if not in state
    state = _restore_runtime_deps(state)
    
    llm = state.get("llm")
    stream = state.get("stream")

    if stream:
        stream.emit_status("Creating research plan...", step="planning")

    context_info = f"\n\nDeep search context:\n{deep_search_result}" if deep_search_result else ""

    prompt = f"""Create a comprehensive research plan for this query:

Query: {query}

Query analysis: {query_analysis.get('reasoning', '')}
Identified topics: {', '.join(query_analysis.get('topics', []))}
Complexity: {query_analysis.get('complexity', 'moderate')}{context_info}

Create a detailed research plan with specific topics for different agents to investigate.
"""

    try:
        plan = await llm.with_structured_output(ResearchPlan).ainvoke([
            {"role": "system", "content": "You are a research strategy expert."},
            {"role": "user", "content": prompt}
        ])

        logger.info("Research plan created", topics=len(plan.topics))

        return {
            "research_plan": {
                "reasoning": plan.reasoning,
                "research_depth": plan.research_depth,
                "coordination_strategy": plan.coordination_strategy
            },
            "research_topics": [topic.dict() for topic in plan.topics]
        }

    except Exception as e:
        logger.error("Research planning failed", error=str(e))
        # Fallback with all required fields
        fallback_topic = ResearchTopic(
            topic=query,
            description=f"Research: {query}",
            priority="high",
            estimated_sources=5  # Default estimate for fallback
        )
        return {
            "research_plan": {
                "reasoning": "Fallback plan due to planning error",
                "research_depth": "standard",
                "coordination_strategy": "Parallel research"
            },
            "research_topics": [fallback_topic.dict()]
        }


# ==================== Agent Characteristics Creation (Enhanced) ====================

async def create_agent_characteristics_enhanced_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create specialized agent characteristics with initial todos.

    Each agent gets:
    - Unique role and expertise
    - Personality
    - Initial structured todos
    """
    # Restore runtime dependencies if not in state
    state = _restore_runtime_deps(state)
    
    query = state.get("query", "")
    research_plan = state.get("research_plan", {})
    research_topics = state.get("research_topics", [])
    # Restore runtime dependencies if not in state
    state = _restore_runtime_deps(state)
    
    llm = state.get("llm")
    stream = state.get("stream")
    settings = state.get("settings")

    # Get agent count from settings or state
    agent_count = state.get("estimated_agent_count", getattr(settings, "max_concurrent_agents", 4) if settings else 4)
    agent_count = min(agent_count, len(research_topics) + 1)  # At least one per topic

    if stream:
        stream.emit_status(f"Creating {agent_count} specialized research agents...", step="agent_characteristics")

    prompt = f"""Create a team of {agent_count} specialized research agents for this project:

Query: {query}

Research topics:
{chr(10).join([f"- {t.get('topic')}: {t.get('description')}" for t in research_topics])}

CRITICAL: Each agent must research DIFFERENT aspects to build a complete picture!

For each agent, create:
1. A unique role (e.g., "Senior Aviation Historian", "Economic Policy Analyst", "Technical Specifications Expert", "Case Study Researcher")
2. Specific expertise area - ensure DIFFERENT angles:
   - Historical development and evolution
   - Technical specifications and technical details
   - Expert opinions, analysis, and critical perspectives
   - Real-world applications, case studies, and practical examples
   - Industry trends, current state, and future prospects
   - Comparative analysis with alternatives/competitors
   - Economic, social, or cultural impact
   - Challenges, limitations, and controversies
3. Personality traits (thorough, analytical, critical, etc.)
4. Initial todo list with 2-3 specific research tasks - each agent should cover DIFFERENT aspects

DIVERSITY REQUIREMENT:
- Ensure agents cover DIFFERENT angles - avoid overlap!
- Each agent should contribute unique insights to build comprehensive understanding
- From diverse agent findings, the supervisor will assemble a COMPLETE picture
- Examples: one agent focuses on history, another on technical specs, another on expert views, another on applications, etc.
"""

    try:
        characteristics = await llm.with_structured_output(AgentCharacteristics).ainvoke([
            {"role": "system", "content": "You are an expert at designing research teams."},
            {"role": "user", "content": prompt}
        ])

        logger.info("Agent characteristics created", agent_count=len(characteristics.agents))

        # Get memory services from stream
        agent_memory_service = stream.app_state.get("agent_memory_service") if stream else None
        agent_file_service = stream.app_state.get("agent_file_service") if stream else None

        # Create agent files with initial todos
        agent_chars = {}
        for i, agent_char in enumerate(characteristics.agents):
            agent_id = f"agent_{i+1}"

            # Convert todos to AgentTodoItem
            agent_todos = [
                AgentTodoItem(
                    reasoning=todo.reasoning,
                    title=todo.title,
                    objective=todo.objective,
                    expected_output=todo.expected_output,
                    sources_needed=todo.sources_needed,
                    status="pending",
                    note=""
                )
                for todo in agent_char.initial_todos
            ]

            # Create agent file if services available
            if agent_file_service:
                try:
                    await agent_file_service.write_agent_file(
                        agent_id=agent_id,
                        todos=agent_todos,
                        character=f"""**Role**: {agent_char.role}
**Expertise**: {agent_char.expertise}
**Personality**: {agent_char.personality}
""",
                        preferences=f"Focus on: {agent_char.expertise}"
                    )
                    logger.info(f"Agent file created", agent_id=agent_id, todos=len(agent_todos))
                except Exception as e:
                    logger.error(f"Failed to create agent file", agent_id=agent_id, error=str(e))

            agent_chars[agent_id] = {
                "role": agent_char.role,
                "expertise": agent_char.expertise,
                "personality": agent_char.personality,
                "initial_todos": [todo.dict() for todo in agent_char.initial_todos]
            }

        return {
            "agent_characteristics": agent_chars,
            "agent_count": len(agent_chars),
            "coordination_notes": characteristics.coordination_notes
        }

    except Exception as e:
        logger.error("Agent characteristics creation failed", error=str(e))
        # Fallback: create simple agents
        fallback_chars = {}
        for i in range(agent_count):
            agent_id = f"agent_{i+1}"
            fallback_chars[agent_id] = {
                "role": f"Research Agent {i+1}",
                "expertise": "general research",
                "personality": "thorough and analytical",
                "initial_todos": []
            }
        return {
            "agent_characteristics": fallback_chars,
            "agent_count": agent_count,
            "coordination_notes": "Parallel research with supervisor coordination"
        }


# ==================== Execute Agents Node (Enhanced with Queue) ====================

async def execute_agents_enhanced_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute research agents in parallel with supervisor queue coordination.
    
    Agents work in parallel, each on ONE task at a time.
    When an agent completes a task, it queues for supervisor review.
    Agents continue working on next tasks until all todos are done.
    
    This implements the requirement: "МНОГО АГЕНТОВ ПОЧТИ ОДНОВРЕМЕННО 
    ВЫПОЛНЯЮТ ЗАДАЧУ И ВЫЗЫВАЮТ ГЛАВНОГО!"
    """
    # Restore runtime dependencies if not in state
    state = _restore_runtime_deps(state)
    
    agent_characteristics = state.get("agent_characteristics", {})
    agent_count = state.get("agent_count", 4)
    llm = state.get("llm")
    search_provider = state.get("search_provider")
    scraper = state.get("scraper")
    stream = state.get("stream")
    settings = state.get("settings")
    max_iterations = state.get("max_iterations", 25)
    current_iteration = state.get("iteration", 0)

    if stream:
        stream.emit_status(f"Executing {agent_count} research agents in parallel...", step="agents")

    # Create supervisor queue
    supervisor_queue = SupervisorQueue()

    # Store in state for agents to access
    state["supervisor_queue"] = supervisor_queue
    
    # All collected findings from all agent iterations
    all_findings = []
    
    # Run agents in continuous mode until all todos complete or max iterations
    agents_active = True
    iteration_count = 0
    
    while agents_active and iteration_count < max_iterations:
        iteration_count += 1
        logger.info(f"Agent execution cycle {iteration_count}")
        
        # Launch all agents in parallel for this iteration
        agent_tasks = []
        for agent_id in agent_characteristics.keys():
            task = asyncio.create_task(
                run_researcher_agent_enhanced(
                    agent_id=agent_id,
                    state=state,
                    llm=llm,
                    search_provider=search_provider,
                    scraper=scraper,
                    stream=stream,
                    supervisor_queue=supervisor_queue,
                    max_steps=8
                )
            )
            agent_tasks.append((agent_id, task))

        logger.info(f"Launched {len(agent_tasks)} agents for cycle {iteration_count}")

        # Collect results from this cycle - wait for all agents in parallel
        cycle_findings = []
        no_tasks_count = 0
        
        # Gather all agent tasks in parallel (not sequentially)
        agent_results = await asyncio.gather(
            *[task for _, task in agent_tasks],
            return_exceptions=True
        )
        
        # Process results
        for (agent_id, _), result in zip(agent_tasks, agent_results):
            if isinstance(result, Exception):
                logger.error(f"Agent {agent_id} failed", error=str(result), exc_info=result)
            elif result:
                if result.get("topic") == "no_tasks":
                    no_tasks_count += 1
                    logger.info(f"Agent {agent_id} has no tasks")
                else:
                    cycle_findings.append(result)
                    all_findings.append(result)
                    logger.info(f"Agent {agent_id} completed task", sources=len(result.get("sources", [])))

        # If all agents report no tasks, stop
        if no_tasks_count == len(agent_tasks):
            logger.info("All agents report no tasks, stopping agent execution")
            agents_active = False
            break
        
        # After each cycle, process supervisor queue if there are completions
        if supervisor_queue.size() > 0:
            logger.info(f"Processing {supervisor_queue.size()} agent completions in supervisor queue")
            
            # Call supervisor agent to review and assign new tasks
            from src.workflow.research.supervisor_agent import run_supervisor_agent
            
            try:
                decision = await run_supervisor_agent(
                    state=state,
                    llm=llm,
                    stream=stream,
                    max_iterations=10
                )
                
                # Update state with supervisor decision
                state["should_continue"] = decision.get("should_continue", False)
                state["replanning_needed"] = decision.get("replanning_needed", False)
                
                # Clear the queue after processing
                while not supervisor_queue.queue.empty():
                    try:
                        supervisor_queue.queue.get_nowait()
                        supervisor_queue.queue.task_done()
                    except:
                        break
                        
                logger.info("Supervisor queue processed", decision=decision.get("should_continue"))
                
            except Exception as e:
                logger.error("Supervisor processing failed", error=str(e))

    # Add findings to agent_findings (using reducer)
    return {
        "agent_findings": all_findings,
        "findings": all_findings,  # Keep for supervisor review
        "findings_count": len(all_findings),
        "iteration": current_iteration + iteration_count
    }


# ==================== Supervisor Review (Enhanced) ====================

async def supervisor_review_enhanced_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supervisor reviews all findings using LangGraph agent with ReAct format.
    
    The supervisor is now a full agent with tools:
    - read_main_document
    - write_main_document  
    - review_agent_progress
    - create_agent_todo
    - make_final_decision
    """
    # Restore runtime dependencies if not in state
    state = _restore_runtime_deps(state)
    
    llm = state.get("llm")
    stream = state.get("stream")

    # Use new supervisor agent with ReAct format
    try:
        decision = await run_supervisor_agent(
            state=state,
            llm=llm,
            stream=stream,
            max_iterations=10
        )
        
        logger.info("Supervisor agent completed", decision=decision)
        return decision
        
    except Exception as e:
        logger.error("Supervisor agent failed", error=str(e), exc_info=True)
        # Fallback: stop research
        return {
            "should_continue": False,
            "replanning_needed": False,
            "gaps_identified": [],
            "iteration": state.get("iteration", 0) + 1,
            "completion_criteria_met": True
        }


# ==================== Compress Findings Node ====================

async def compress_findings_node(state: ResearchState) -> Dict:
    # Restore runtime dependencies if not in state
    state = _restore_runtime_deps(state)
    """Compress all findings before final report."""
    findings = state.get("agent_findings", [])
    stream = state.get("stream")

    if not findings:
        return {"compressed_research": {"type": "override", "value": ""}}

    if stream:
        stream.emit_status("Compressing findings...", step="compression")

    findings_text = "\n\n".join([
        f"### {f.get('topic')}\n{f.get('summary', '')}"
        for f in findings if f
    ])

    system_prompt = """Compress research findings into a coherent synthesis.

Aim for 800-1200 words. Identify key themes and important sources.
Return JSON with: reasoning, compressed_summary, key_themes, important_sources
"""

    try:
        llm = state.get("llm")
        structured_llm = llm.with_structured_output(CompressedFindings, method="function_calling")

        result = await structured_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=findings_text)
        ])

        logger.info("Findings compressed", length=len(result.compressed_summary))

        if stream:
            stream.emit_compression({"message": "Findings compressed"})

        return {
            "compressed_research": {"type": "override", "value": result.compressed_summary},
        }

    except Exception as e:
        logger.error("Compression failed", error=str(e))
        return {"compressed_research": {"type": "override", "value": findings_text}}


# ==================== Final Report Generation (Enhanced) ====================

async def generate_final_report_enhanced_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate final research report with validation.

    Uses structured output for well-organized report.
    """
    # Restore runtime dependencies if not in state
    state = _restore_runtime_deps(state)
    
    query = state.get("query", "")
    # Use findings from execute_agents, fallback to agent_findings
    findings = state.get("findings", state.get("agent_findings", []))
    # Restore runtime dependencies if not in state
    state = _restore_runtime_deps(state)
    
    llm = state.get("llm")
    stream = state.get("stream")

    if stream:
        stream.emit_status("Generating final report...", step="report")

    # Get memory service to read main document
    agent_memory_service = stream.app_state.get("agent_memory_service") if stream else None
    main_document = ""
    if agent_memory_service:
        try:
            main_document = await agent_memory_service.read_main_file()
            logger.info("Read main document", length=len(main_document))
        except Exception as e:
            logger.warning("Could not read main document", error=str(e))

    # Compile all findings
    findings_text = "\n\n".join([
        f"### {f.get('topic')}\n{f.get('summary', '')}\n\nKey findings:\n" +
        "\n".join([f"- {kf}" for kf in f.get('key_findings', [])[:5]])
        for f in findings
    ])

    prompt = f"""Generate a comprehensive final research report.

Query: {query}

Main research document:
{main_document}

Additional findings:
{findings_text}

Create a well-structured report with:
1. Executive summary
2. Detailed sections with analysis
3. Evidence-based conclusions
4. Citations to sources
"""

    try:
        report = await llm.with_structured_output(FinalReport).ainvoke([
            {"role": "system", "content": "You are an expert research report writer."},
            {"role": "user", "content": prompt}
        ])

        # Validate report
        # Use summarize_text for intelligent truncation, not hard cut
        from src.utils.text import summarize_text
        exec_summary_preview = summarize_text(report.executive_summary, 500)
        conclusion_preview = summarize_text(report.conclusion, 300)
        
        validation_prompt = f"""Validate this research report for quality.

Query: {query}

Report title: {report.title}
Executive summary: {exec_summary_preview}
Sections: {len(report.sections)}
Conclusion: {conclusion_preview}

Check for completeness, accuracy, and quality.
"""

        validation = await llm.with_structured_output(ReportValidation).ainvoke([
            {"role": "system", "content": "You are a research quality auditor."},
            {"role": "user", "content": validation_prompt}
        ])

        logger.info(
            "Report generated and validated",
            is_complete=validation.is_complete,
            quality_score=validation.quality_score,
            sections=len(report.sections)
        )

        # Format as markdown
        report_markdown = f"""# {report.title}

## Executive Summary

{report.executive_summary}

"""
        for section in report.sections:
            report_markdown += f"""## {section.title}

{section.content}

"""

        report_markdown += f"""## Conclusion

{report.conclusion}

---

*Confidence Level: {report.confidence_level}*
*Research Quality Score: {validation.quality_score}/10*
"""

        # Stream report in chunks
        if stream:
            logger.info("Streaming final report", report_length=len(report_markdown))
            chunk_size = 500
            for i in range(0, len(report_markdown), chunk_size):
                chunk = report_markdown[i:i+chunk_size]
                stream.emit_report_chunk(chunk)
                await asyncio.sleep(0.02)  # Small delay for smooth streaming
            logger.info("Final report chunks streamed", total_chunks=(len(report_markdown) + chunk_size - 1) // chunk_size)

        logger.info("Returning final report from node", report_length=len(report_markdown))
        return {
            "final_report": report_markdown,
            "confidence": report.confidence_level
        }

    except Exception as e:
        logger.error("Report generation failed", error=str(e))
        # Fallback
        fallback_report = f"""# Research Report: {query}

{main_document if main_document else findings_text}
"""
        return {
            "final_report": fallback_report,
            "confidence": "low"
        }


# ==================== Aliases for Backward Compatibility ====================

# Old names point to enhanced versions
plan_research_node = plan_research_enhanced_node
spawn_agents_node = create_agent_characteristics_enhanced_node
execute_agents_node = execute_agents_enhanced_node
supervisor_react_node = supervisor_review_enhanced_node
generate_report_node = generate_final_report_enhanced_node

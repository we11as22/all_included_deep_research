"""LangGraph nodes for deep research workflow.

Each node is an async function that takes state and returns state updates.
"""

import asyncio
import structlog
from typing import Any, Dict

from langchain_core.messages import SystemMessage, HumanMessage

from src.workflow.research.state import (
    ResearchState,
    ResearchPlan,
    SupervisorReActOutput,
    AgentFinding,
    CompressedFindings,
    FinalReport,
)
from src.workflow.research.queue import get_supervisor_queue

logger = structlog.get_logger(__name__)


# ==================== Memory Search Node ==========


async def search_memory_node(state: ResearchState) -> Dict:
    """Search vector memory for relevant context."""
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


# ==================== Deep Search Node ==========


async def run_deep_search_node(state: ResearchState) -> Dict:
    """Run deep search to gather initial context before planning."""
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

        logger.info("Deep search completed", answer_length=len(result))

        return {
            "deep_search_result": {"type": "override", "value": result},
        }

    except Exception as e:
        logger.error("Deep search failed", error=str(e), exc_info=True)
        return {"deep_search_result": {"type": "override", "value": None}}


# ==================== Planning Node ==========


async def plan_research_node(state: ResearchState) -> Dict:
    """Initial research planning by supervisor."""
    query = state["query"]
    memory_context = state.get("memory_context", [])
    chat_history = state.get("chat_history", [])
    stream = state.get("stream")
    mode = state["mode"]
    deep_search_result = state.get("deep_search_result")

    if stream:
        stream.emit_status("Planning research...", step="planning")

    # Get settings
    settings = state.get("settings")
    num_agents = settings.deep_research_num_agents if settings else 4
    max_topics = num_agents  # Create exactly N agents

    # Include deep search context if available
    deep_search_context = ""
    if deep_search_result:
        deep_search_context = f"\n\nInitial Deep Search Result:\n{deep_search_result[:1000]}...\n\nUse this as context but identify gaps and areas that need deeper investigation."

    system_prompt = f"""You are a research planning supervisor.

Your task: Break down the research query into EXACTLY {max_topics} focused research topics.

Each topic should:
- Be specific and well-defined
- Cover a different aspect of the query
- Be suitable for parallel investigation by specialist agents
- Avoid overlap
- Address gaps in the initial research{deep_search_context}

Return JSON with: reasoning, topics (list), stop (bool)
"""

    user_prompt = f"""Query: {query}

Generate EXACTLY {max_topics} research topics to cover this comprehensively.
"""

    try:
        # Use structured output
        llm = state.get("llm")  # LLM should be in state
        if not llm:
            raise ValueError("LLM not in state")

        structured_llm = llm.with_structured_output(ResearchPlan, method="function_calling")

        result = await structured_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        topics = [topic.topic for topic in result.topics[:max_topics]]

        logger.info("Research plan generated", topics_count=len(topics))

        # Generate agent characteristics for each topic
        agent_characteristics = await _generate_agent_characteristics(
            topics=topics,
            query=query,
            llm=state.get("llm"),
            stream=stream
        )

        if stream:
            stream.emit_planning({"topics": topics, "reasoning": result.reasoning})

        return {
            "research_plan": topics,
            "agent_characteristics": {"type": "override", "value": agent_characteristics},
        }

    except Exception as e:
        logger.error("Planning failed, using fallback", error=str(e))
        # Fallback: Use query as single topic
        return {
            "research_plan": [query],
            "agent_characteristics": {"type": "override", "value": {}},
        }


async def _generate_agent_characteristics(topics: list, query: str, llm: Any, stream: Any) -> Dict[str, Dict]:
    """Generate specialist characteristics for each agent based on their topic."""
    from pydantic import BaseModel, Field

    class AgentCharacteristic(BaseModel):
        role: str = Field(description="Agent's role/title (e.g., 'Senior AI Policy Expert', 'Technical Standards Analyst')")
        expertise: str = Field(description="Specific domain expertise")
        personality: str = Field(description="Research approach and personality traits")

    class AgentCharacteristics(BaseModel):
        agents: list[AgentCharacteristic] = Field(description="List of agent characteristics for each topic")

    if stream:
        stream.emit_status("Creating specialized research agents...", step="agent_characteristics")

    try:
        system_prompt = """You are assigning expert researcher roles for parallel investigation.

For each research topic, create a unique specialist agent with:
- A specific professional role/title that matches the topic
- Relevant domain expertise
- A personality/approach that enhances research quality

Make agents diverse and complementary. Think like assembling a real research team."""

        user_prompt = f"""Research Query: {query}

Topics to assign specialists for:
{chr(10).join(f"{i+1}. {topic}" for i, topic in enumerate(topics))}

Create {len(topics)} specialist agent profiles."""

        structured_llm = llm.with_structured_output(AgentCharacteristics, method="function_calling")
        result = await structured_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        # Map to dict by index
        characteristics = {}
        for idx, agent_char in enumerate(result.agents[:len(topics)]):
            characteristics[f"agent_{idx}"] = {
                "role": agent_char.role,
                "expertise": agent_char.expertise,
                "personality": agent_char.personality,
            }

        logger.info(f"Generated characteristics for {len(characteristics)} agents")
        return characteristics

    except Exception as e:
        logger.warning(f"Failed to generate agent characteristics: {e}")
        # Fallback: Generic characteristics
        return {
            f"agent_{i}": {
                "role": f"Research Specialist {i+1}",
                "expertise": topic,
                "personality": "Thorough and analytical researcher",
            }
            for i, topic in enumerate(topics)
        }


# ==================== Spawn Agents Node ==========


async def spawn_agents_node(state: ResearchState) -> Dict:
    """Create agent instances for research topics."""
    topics = state.get("research_plan", [])
    iteration = state.get("iteration", 0)

    if not topics:
        return {"active_agents": {"type": "override", "value": {}}}

    active_agents = {}
    agent_todos = {}
    agent_notes = {}

    for idx, topic in enumerate(topics):
        agent_id = f"agent_r{iteration}_{idx}"
        active_agents[agent_id] = {
            "topic": topic,
            "status": "active",
            "findings": None,
        }

        # Initialize agent todos
        agent_todos[agent_id] = [
            {
                "title": f"Research {topic}",
                "objective": f"Gather comprehensive information about {topic}",
                "status": "pending",
                "priority": "high",
            }
        ]

        agent_notes[agent_id] = []

    logger.info(f"Spawned {len(active_agents)} agents", iteration=iteration)

    return {
        "active_agents": {"type": "override", "value": active_agents},
        "agent_todos": {"type": "override", "value": agent_todos},
        "agent_notes": {"type": "override", "value": agent_notes},
    }


# ==================== Execute Agents Node ==========


async def execute_agents_node(state: ResearchState) -> Dict:
    """Execute all active agents in parallel with semaphore."""
    active_agents = state.get("active_agents", {})
    max_concurrent = state["mode_config"].get("max_concurrent", 4)
    stream = state.get("stream")

    if not active_agents:
        return {}

    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_agent(agent_id: str, agent_data: Dict):
        async with semaphore:
            # Run researcher agent (import to avoid circular dependency)
            from src.workflow.research.researcher import run_researcher_agent

            try:
                result = await run_researcher_agent(
                    agent_id=agent_id,
                    topic=agent_data["topic"],
                    state=state,
                    llm=state.get("llm"),
                    search_provider=state.get("search_provider"),
                    scraper=state.get("scraper"),
                    stream=stream,
                    max_steps=6,
                )

                # Queue supervisor review
                queue = get_supervisor_queue(state["session_id"])
                await queue.enqueue(agent_id, "finish", result)

                return result

            except Exception as e:
                logger.error(f"Agent {agent_id} failed", error=str(e), exc_info=True)
                return None

    # Run all agents in parallel
    tasks = [run_agent(aid, adata) for aid, adata in active_agents.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect successful findings
    findings = []
    for result in results:
        if result and not isinstance(result, Exception):
            findings.append(result)

    logger.info(f"Agent execution completed", findings=len(findings))

    return {
        "agent_findings": findings,
        "iteration": state["iteration"] + 1,
    }


# ==================== Supervisor ReAct Node ==========


async def supervisor_react_node(state: ResearchState) -> Dict:
    """Supervisor analyzes progress and decides next action."""
    findings = state.get("agent_findings", [])
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 25)
    stream = state.get("stream")

    if stream:
        stream.emit_status("Supervisor reviewing progress...", step="supervisor")

    # Build summary for supervisor
    findings_summary = "\n\n".join([
        f"Agent {f.get('agent_id')}: {f.get('topic')}\n{f.get('summary', '')[:200]}"
        for f in findings if f
    ])

    system_prompt = f"""You are the research supervisor.

Iteration: {iteration}/{max_iterations}

Evaluate: Are the findings comprehensive? Should we continue? Are there gaps?

Return JSON with: reasoning, should_continue, replanning_needed, directives, new_topics, gaps_identified
"""

    user_prompt = f"""Current findings:
{findings_summary}

Should research continue?
"""

    try:
        llm = state.get("llm")
        structured_llm = llm.with_structured_output(SupervisorReActOutput, method="function_calling")

        result = await structured_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        logger.info(
            "Supervisor decision",
            should_continue=result.should_continue,
            replan=result.replanning_needed,
            gaps=len(result.gaps_identified)
        )

        if stream:
            stream.emit_supervisor_react({
                "reasoning": result.reasoning,
                "should_continue": result.should_continue,
                "gaps": result.gaps_identified,
            })

        return {
            "should_continue": {"type": "override", "value": result.should_continue},
            "replanning_needed": {"type": "override", "value": result.replanning_needed},
            "gaps_identified": {"type": "override", "value": result.gaps_identified},
            "supervisor_directives": result.directives,
        }

    except Exception as e:
        logger.error("Supervisor ReAct failed", error=str(e))
        # Fallback: Stop if near max iterations
        return {
            "should_continue": {"type": "override", "value": iteration < max_iterations - 1},
            "replanning_needed": {"type": "override", "value": False},
        }


# ==================== Compress Findings Node ==========


async def compress_findings_node(state: ResearchState) -> Dict:
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


# ==================== Generate Report Node ==========


async def generate_report_node(state: ResearchState) -> Dict:
    """Generate final markdown report."""
    compressed = state.get("compressed_research", "")
    query = state["query"]
    stream = state.get("stream")

    if stream:
        stream.emit_status("Generating final report...", step="report")

    system_prompt = """Generate a comprehensive research report in markdown.

Include:
- Clear structure with headings
- Inline citations [1], [2]
- Sources section at end

Return JSON with: reasoning, report (markdown), key_findings, sources_count
"""

    try:
        llm = state.get("llm")
        structured_llm = llm.with_structured_output(FinalReport, method="function_calling")

        result = await structured_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {query}\n\nResearch:\n{compressed}")
        ])

        logger.info("Final report generated", length=len(result.report))

        if stream:
            stream.emit_final_report(result.report)

        return {
            "final_report": {"type": "override", "value": result.report},
            "confidence": {"type": "override", "value": "high"},
        }

    except Exception as e:
        logger.error("Report generation failed", error=str(e))
        return {
            "final_report": {"type": "override", "value": compressed or "Research failed"},
            "confidence": {"type": "override", "value": "low"},
        }

"""Spawn agents node for creating agent characteristics."""

import structlog
from typing import Dict, Any
from collections import Counter

from src.workflow.research.state import ResearchState
from src.workflow.research.nodes.base import ResearchNode
from src.workflow.research.nodes.utils import _restore_runtime_deps
from src.workflow.research.models import (
    AgentCharacteristics,
    AgentCharacteristic,
    AgentTodo,
)
from src.models.agent_models import AgentTodoItem

logger = structlog.get_logger(__name__)


class SpawnAgentsNode(ResearchNode):
    """Create agent characteristics for research agents."""

    async def execute(self, state: ResearchState) -> Dict[str, Any]:
        """Execute spawn agents node.

        Args:
            state: Current research state

        Returns:
            State updates with agent_characteristics
        """
        # Convert ResearchState to dict for compatibility
        if isinstance(state, dict):
            state_dict = state
        else:
            state_dict = dict(state)
        
        # Restore runtime dependencies if not in state
        state_dict = _restore_runtime_deps(state_dict)
        
        query = state_dict.get("query", "")
        research_plan = state_dict.get("research_plan", {})
        research_topics = state_dict.get("research_topics", [])
        
        llm = state_dict.get("llm")
        stream = state_dict.get("stream")
        settings = state_dict.get("settings")

        # Get agent count from settings or state
        # CRITICAL: Use deep_research_num_agents from settings, not max_concurrent_agents
        # CRITICAL: LLM may suggest more agents, but we MUST respect settings limit
        if settings:
            max_agent_count = getattr(settings, "deep_research_num_agents", 3)
        else:
            from src.config.settings import get_settings
            settings_obj = get_settings()
            max_agent_count = getattr(settings_obj, "deep_research_num_agents", 3)
        
        # Get LLM's estimate, but cap it at settings limit
        estimated_from_llm = state_dict.get("estimated_agent_count", max_agent_count)
        agent_count = min(estimated_from_llm, max_agent_count)  # CRITICAL: Cap at settings limit
        
        session_id = state_dict.get("session_id")
        if not session_id:
            logger.warning("session_id not found in state - using 'unknown' for logging", state_keys=list(state_dict.keys())[:10])
            session_id = "unknown"
        logger.info("Creating agent characteristics", 
                   agent_count=agent_count, 
                   estimated_from_llm=estimated_from_llm,
                   max_from_settings=max_agent_count,
                   session_id=session_id,
                   note=f"Using {agent_count} agents (capped from LLM estimate {estimated_from_llm} by settings limit {max_agent_count})")

        # Don't emit status here - we'll emit after we know the actual agent count (after fallback)

        # CRITICAL: Log all context data before creating agent characteristics (лизонинг)
        original_query = state_dict.get("original_query", query)
        logger.info("🔍 SPAWN AGENTS: Context data check",
                   session_id=session_id,
                   original_query=original_query[:100] if original_query else None,
                   query=query[:100] if query else None,
                   has_deep_search_result="deep_search_result" in state_dict,
                   chat_history_length=len(state_dict.get("chat_history", [])),
                   note="Verifying all context data is available for agent creation")
        
        # Get deep search result and user clarification answers
        deep_search_result_raw = state_dict.get("deep_search_result", "")
        if isinstance(deep_search_result_raw, dict):
            deep_search_result = deep_search_result_raw.get("value", "") if isinstance(deep_search_result_raw, dict) else ""
        else:
            deep_search_result = deep_search_result_raw or ""
        
        logger.info("🔍 SPAWN AGENTS: Deep search result",
                   session_id=session_id,
                   result_length=len(deep_search_result) if deep_search_result else 0,
                   result_preview=deep_search_result[:200] if deep_search_result else None,
                   note="Deep search result loaded for agent creation")
        
        # CRITICAL: Use clarification_answers from session state (loaded from DB), not chat_history
        # clarification_answers is the source of truth, loaded from session in create_initial_state
        clarification_answers_from_state = state_dict.get("clarification_answers", "")
        clarification_context = ""
        
        if clarification_answers_from_state and clarification_answers_from_state.strip():
            clarification_context = f"\n\n**USER CLARIFICATION ANSWERS:**\n{clarification_answers_from_state}\n"
            logger.info("🔍 SPAWN AGENTS: Clarification answers",
                       session_id=session_id,
                       answer_preview=clarification_answers_from_state[:200] if clarification_answers_from_state else None,
                       source="session_state",
                       note="Using clarification_answers from session state (source of truth)")
        else:
            # Fallback: try to extract from chat_history (for backward compatibility)
            chat_history = state_dict.get("chat_history", [])
            if chat_history:
                for i, msg in enumerate(chat_history):
                    if msg.get("role") == "assistant" and ("clarification" in msg.get("content", "").lower() or "🔍" in msg.get("content", "")):
                        if i + 1 < len(chat_history) and chat_history[i + 1].get("role") == "user":
                            user_answer = chat_history[i + 1].get("content", "")
                            clarification_context = f"\n\n**USER CLARIFICATION ANSWERS:**\n{user_answer}\n"
                            logger.warning("🔍 SPAWN AGENTS: Clarification answers (fallback from chat_history)",
                                       session_id=session_id,
                                       answer_preview=user_answer[:200] if user_answer else None,
                                       source="chat_history_fallback",
                                       note="This should not happen - clarification_answers should be in session state")
                        break

        # Get user language from state (already detected in create_research_state)
        user_language = state_dict.get("user_language", "English")
        language_instruction = ""
        if user_language == "Russian":
            language_instruction = "\n\n**CRITICAL LANGUAGE REQUIREMENT:** The user's query is in RUSSIAN. All agent task descriptions, findings, and final reports MUST be in RUSSIAN language. Agents will conduct research using English sources but MUST write all outputs in RUSSIAN."
        else:
            language_instruction = f"\n\n**LANGUAGE REQUIREMENT:** All research outputs should be in {user_language}."

        prompt = f"""Create a team of {agent_count} specialized research agents for this project.

Query: {query}{language_instruction}

Initial Context:
{deep_search_result[:2000] if deep_search_result else "No initial context available."}
{clarification_context}

Research Topics:
{chr(10).join([f"- {t.get('topic')}: {t.get('description')}" for t in research_topics])}

**CRITICAL: REASONING REQUIREMENT - Before creating agents, document your thinking in the reasoning field:**
1. **Original Query Analysis**: What is the user asking for? What is the core topic? What aspects need to be covered?
2. **Deep Search Context**: What did the initial deep search reveal? What key aspects were found?
3. **Clarification Answers**: If clarification was provided, what did the user specify? How does it refine the original query?
4. **Research Topics Integration**: How do the research topics relate to the original query and deep search context?
5. **Coverage Verification**: Do the research topics together provide COMPLETE coverage of the query? What aspects are covered? Are there any gaps?
6. **Agent Team Strategy**: Why this specific team composition? How will each agent contribute unique insights?
7. **Task Distribution Strategy**: How should tasks be distributed among agents to ensure comprehensive coverage? Which topics should each agent handle? How will tasks ensure complete coverage without gaps?

**For each agent's reasoning field, document:**
1. Why this specific agent role is needed for this research
2. How this agent's expertise relates to the original query
3. How this agent will use deep search context and clarification answers
4. What unique angle this agent will cover

**For each task's reasoning field, document:**
1. How this task relates to the original query
2. What aspect of deep search context it addresses
3. How it incorporates clarification answers (if provided)
4. Why this specific task is important for comprehensive research

Requirements:
- Create exactly {agent_count} agents, each with 2-3 initial tasks
- Each agent must cover a different research angle to build a complete picture
- All tasks must be unique across agents - no duplicate or similar task titles
- Each task must be specific and include the query in the objective
- Tasks must be self-contained (agents only see their task description, not the full query)

Task Creation Guidelines - CRITICAL FOR COMPREHENSIVE COVERAGE:
- Every task objective MUST include the original user query
- Every task MUST be specific to the user's query - not generic
- Task format: Start each task objective with "The user asked: '[query]'. Research [specific aspect related to query]..."
- **MANDATORY**: Tasks must collectively ensure COMPLETE coverage of the query
- Each task should address a specific aspect that is ESSENTIAL for answering the query
- Think systematically: what questions need to be answered? What aspects must be covered?
- If clarification answers are provided, interpret them IN THE CONTEXT of the original query
  * Clarification specifies WHAT ASPECT of the original topic to focus on, NOT a new topic
  * Include clarification answers in task descriptions, but ALWAYS in context of original query
- Do NOT create generic tasks - be SPECIFIC and ensure each task contributes unique value
- Do NOT interpret clarification as a standalone query - it's ALWAYS about the original query topic
- Each task must be self-contained - the agent will NOT see the original query, only the task description
- **CRITICAL**: Before finalizing tasks, verify: "Do these tasks together fully answer the query?"

For each agent, create:
1. Unique role (e.g., "Aviation Historian", "Technical Analyst", "Case Study Researcher")
2. Specific expertise area - ensure different angles:
   - Historical development and evolution
   - Technical specifications and details
   - Expert analysis and critical perspectives
   - Real-world applications and case studies
   - Industry trends and current state
   - Comparative analysis
   - Impact and implications
   - Challenges and limitations
3. Personality traits (thorough, analytical, critical, etc.)
4. 2-3 research tasks with unique titles and objectives based on agent's expertise
   - Each agent's tasks MUST be UNIQUE and based on their specific expertise
   - Tasks must reflect the agent's unique expertise angle
   - Do NOT create identical or similar tasks for different agents
   - Each task title and objective must be DISTINCT and reflect the agent's unique expertise angle

Distribution Requirements:
- You MUST create EXACTLY {agent_count} agents
- EACH agent MUST have 2-3 initial tasks (NOT just 1 task per agent!)
- Each agent's tasks MUST be UNIQUE - NO duplicate or similar tasks across agents
- Ensure agents cover DIFFERENT angles - avoid overlap
- Each agent should contribute unique insights to build comprehensive understanding
- All angles must relate to the user's query - do NOT research unrelated topics
- If clarification was provided, interpret it IN CONTEXT of the original query

Verification: Before responding, check:
1. Do you have exactly {agent_count} agents?
2. Does each agent have 2-3 tasks?
3. Are ALL task titles UNIQUE across all agents? (NO duplicates!)
4. Do tasks reflect each agent's unique expertise angle?
If any answer is NO, adjust your response!
"""

        try:
            characteristics = await llm.with_structured_output(AgentCharacteristics).ainvoke([
                {"role": "system", "content": f"You are an expert at designing research teams. Create exactly {agent_count} agents, each with 2-3 unique tasks. Each agent's tasks must be unique and reflect their specific expertise angle."},
                {"role": "user", "content": prompt}
            ])

            # CRITICAL: Validate that each agent has 2-3 tasks
            agents_with_insufficient_tasks = []
            for i, agent_char in enumerate(characteristics.agents):
                todos_count = len(agent_char.initial_todos)
                if todos_count < 2:
                    agents_with_insufficient_tasks.append((i, agent_char.role, todos_count))
                    logger.warning(f"Agent {i+1} ({agent_char.role}) has only {todos_count} tasks, expected 2-3")
            
            logger.info("Agent characteristics created by LLM",
                       agent_count=len(characteristics.agents),
                       expected_count=agent_count,
                       shortfall=max(0, agent_count - len(characteristics.agents)),
                       agents_with_insufficient_tasks=len(agents_with_insufficient_tasks))

            # CRITICAL: Validate that each agent has 2-3 tasks AND that tasks are UNIQUE across agents
            # The structured output model now enforces min_length=2, so LLM should create 2-3 tasks per agent
            
            # Collect all task titles to check for duplicates
            all_task_titles = []
            all_task_objectives = []
            
            for agent_char in characteristics.agents:
                initial_count = len(agent_char.initial_todos)
                task_titles = [t.title for t in agent_char.initial_todos]
                task_objectives = [t.objective for t in agent_char.initial_todos]
                
                logger.info(f"Validating agent {agent_char.role} tasks",
                           initial_count=initial_count,
                           todos_titles=task_titles,
                           expertise=agent_char.expertise)
                
                all_task_titles.extend(task_titles)
                all_task_objectives.extend(task_objectives)
                
                # Model should enforce min_length=2, but double-check
                if initial_count < 2:
                    logger.error(f"CRITICAL: Agent {agent_char.role} has only {initial_count} tasks - structured output model should enforce min_length=2!",
                               initial_count=initial_count,
                               role=agent_char.role,
                               expertise=agent_char.expertise,
                               note="This should not happen - model has min_length=2 constraint")
                elif initial_count == 1:
                    # Fallback: if somehow only 1 task, add 1-2 more to reach 2-3
                    logger.warning(f"Agent {agent_char.role} has only 1 task (model should prevent this), adding tasks to reach 2-3")
                    target_tasks = 3  # Aim for 3 tasks
                    tasks_added = 0
                    while len(agent_char.initial_todos) < target_tasks:
                        task_num = len(agent_char.initial_todos) + 1
                        new_task = AgentTodo(
                            reasoning=f"Additional research task {task_num} for comprehensive coverage of {agent_char.expertise}",
                            title=f"{agent_char.role}: {agent_char.expertise} - Additional research task {task_num}",
                            objective=f"The user asked: '{query}'. Conduct additional research on {agent_char.expertise} to ensure comprehensive coverage. Focus on {agent_char.expertise} aspects not yet fully covered.",
                            expected_output=f"Additional findings about {agent_char.expertise}",
                            sources_needed=[],
                            guidance=f"Focus on {agent_char.expertise}. Use web search to find authoritative sources. Ensure comprehensive coverage."
                        )
                        agent_char.initial_todos.append(new_task)
                        tasks_added += 1
                    
                    final_count = len(agent_char.initial_todos)
                    logger.info(f"Added {tasks_added} tasks to agent {agent_char.role}, now has {final_count} tasks",
                               initial_count=initial_count,
                               final_count=final_count,
                               todos_titles=[t.title for t in agent_char.initial_todos])
                else:
                    logger.info(f"Agent {agent_char.role} has sufficient tasks", count=initial_count)
            
            # CRITICAL: Check for duplicate tasks across agents
            title_counts = Counter(all_task_titles)
            objective_counts = Counter([obj[:100] for obj in all_task_objectives])  # Check first 100 chars for similarity
            
            duplicates = {title: count for title, count in title_counts.items() if count > 1}
            similar_objectives = {obj: count for obj, count in objective_counts.items() if count > 1}
            
            if duplicates:
                logger.error(f"CRITICAL: Found duplicate task titles across agents!",
                           duplicates=duplicates,
                           note="Tasks must be unique per agent based on their expertise")
                
                # Fix duplicates by making tasks unique based on agent expertise and role
                for agent_char in characteristics.agents:
                    for todo in agent_char.initial_todos:
                        if todo.title in duplicates:
                            # Make title and objective unique by incorporating agent's specific expertise
                            original_title = todo.title
                            original_objective = todo.objective
                            
                            # Extract the core task from title (remove generic parts)
                            core_task = todo.title
                            if ":" in core_task:
                                core_task = core_task.split(":", 1)[1].strip()
                            
                            # Create unique title based on agent's expertise
                            todo.title = f"{agent_char.role}: {agent_char.expertise} - {core_task}"
                            
                            # Update objective to be specific to agent's expertise
                            if agent_char.expertise.lower() not in todo.objective.lower():
                                todo.objective = f"The user asked: '{query}'. Research {core_task} with focus on {agent_char.expertise}. {agent_char.expertise} aspects: {todo.objective}"
                            
                            logger.warning(f"Made task unique for agent {agent_char.role}",
                                         original_title=original_title,
                                         new_title=todo.title,
                                         expertise=agent_char.expertise)
            
            if similar_objectives:
                logger.warning(f"Found similar task objectives across agents",
                             similar_count=len(similar_objectives),
                             note="This may indicate agents have overlapping tasks - consider reviewing")
            
            # CRITICAL: Fallback if LLM returned fewer agents than requested
            # This happens with weaker models (GPT-4-mini, etc.) that struggle with large lists in structured output
            if len(characteristics.agents) < agent_count:
                logger.warning(f"LLM returned {len(characteristics.agents)} agents but {agent_count} were requested. Creating {agent_count - len(characteristics.agents)} additional agents with fallback logic.")

                # Get topics that weren't covered yet
                covered_topics = [agent.expertise.lower() for agent in characteristics.agents]
                remaining_topics = [
                    topic for topic in research_topics
                    if not any(covered in topic.get('topic', '').lower() or covered in topic.get('description', '').lower()
                              for covered in covered_topics)
                ]

                # Create fallback agents for remaining topics
                # Each fallback agent should have 2-3 tasks to match the expected distribution
                for i in range(len(characteristics.agents), agent_count):
                    agent_num = i + 1
                    if i - len(characteristics.agents) < len(remaining_topics):
                        topic = remaining_topics[i - len(characteristics.agents)]
                        topic_name = topic.get('topic', f'Research Area {agent_num}')
                        topic_desc = topic.get('description', '')

                        fallback_agent = AgentCharacteristic(
                            reasoning=f"Research Specialist {agent_num} needed to cover {topic_name} from research plan",
                            agent_id=f"agent_{agent_num}",
                            role=f"Research Specialist {agent_num}",
                            expertise=topic_name,
                            personality="Thorough, analytical, detail-oriented",
                            initial_todos=[
                                AgentTodo(
                                    reasoning=f"Research {topic_name} as specified in the research plan",
                                    title=f"Research: {topic_name}",
                                    objective=f"The user asked: '{query}'. Research {topic_name}: {topic_desc}",
                                    expected_output=f"Comprehensive findings about {topic_name} with verified sources",
                                    sources_needed=[],
                                    guidance="Use web search to find authoritative sources. Focus on accuracy and depth."
                                ),
                                AgentTodo(
                                    reasoning=f"Analyze findings and identify key insights about {topic_name}",
                                    title=f"Analyze {topic_name} findings",
                                    objective=f"Synthesize research on {topic_name} and extract key insights relevant to the user's query: '{query}'",
                                    expected_output=f"Key insights and analysis of {topic_name}",
                                    sources_needed=[],
                                    guidance="Focus on answering the user's original question with your findings."
                                ),
                                AgentTodo(
                                    reasoning=f"Verify and cross-reference key claims about {topic_name}",
                                    title=f"Verify {topic_name} findings",
                                    objective=f"The user asked: '{query}'. Verify important claims about {topic_name} by finding multiple independent sources and cross-referencing information.",
                                    expected_output=f"Verified and cross-referenced findings about {topic_name}",
                                    sources_needed=[],
                                    guidance="Find multiple independent sources to verify key claims. Cross-reference information for accuracy."
                                )
                            ]
                        )
                    else:
                        # If we ran out of topics, create a generic research agent with multiple tasks
                        fallback_agent = AgentCharacteristic(
                            reasoning=f"General Research Agent {agent_num} needed for additional research coverage",
                            agent_id=f"agent_{agent_num}",
                            role=f"General Research Agent {agent_num}",
                            expertise=f"General research and analysis",
                            personality="Thorough, analytical, detail-oriented",
                            initial_todos=[
                                AgentTodo(
                                    reasoning=f"Provide additional research coverage for the user's query",
                                    title=f"Additional research for: {query}",
                                    objective=f"The user asked: '{query}'. Conduct supplementary research to fill any gaps not covered by other agents.",
                                    expected_output="Additional relevant findings that complement other agents' work",
                                    sources_needed=[],
                                    guidance="Focus on aspects not fully covered by other agents. Use web search to find authoritative sources."
                                ),
                                AgentTodo(
                                    reasoning=f"Analyze and synthesize findings from multiple sources",
                                    title=f"Synthesize findings for: {query}",
                                    objective=f"The user asked: '{query}'. Analyze and synthesize findings from multiple sources to provide comprehensive coverage.",
                                    expected_output="Synthesized analysis combining multiple perspectives",
                                    sources_needed=[],
                                    guidance="Combine findings from different sources to provide a comprehensive view."
                                ),
                                AgentTodo(
                                    reasoning=f"Identify and investigate related aspects not yet covered",
                                    title=f"Explore related aspects for: {query}",
                                    objective=f"The user asked: '{query}'. Identify and investigate related aspects, connections, or implications that haven't been fully explored by other agents.",
                                    expected_output="Findings about related aspects and connections",
                                    sources_needed=[],
                                    guidance="Look for related topics, connections, or implications that add depth to the research."
                                )
                            ]
                        )

                    characteristics.agents.append(fallback_agent)
                    logger.info(f"Created fallback agent {i+1}", role=fallback_agent.role, expertise=fallback_agent.expertise)

            logger.info("Final agent team ready", agent_count=len(characteristics.agents))

            # Emit status with ACTUAL agent count (after fallback)
            if stream:
                stream.emit_status(f"Creating {len(characteristics.agents)} specialized research agents...", step="agent_characteristics")

            # Get memory services from stream
            agent_memory_service = stream.app_state.get("agent_memory_service") if stream else None
            agent_file_service = stream.app_state.get("agent_file_service") if stream else None

            # Create supervisor file if services available
            if agent_file_service:
                try:
                    await agent_file_service.write_agent_file(
                        agent_id="supervisor",
                        todos=[],
                        notes=[],
                        character="""**Role**: Research Supervisor
**Expertise**: Coordinating research teams, synthesizing findings, identifying gaps
**Personality**: Analytical, strategic, thorough
""",
                        preferences="Focus on comprehensive, diverse research coverage. Keep main.md minimal with only essential shared information."
                    )
                    logger.info("Supervisor file created")
                except Exception as e:
                    logger.error("Failed to create supervisor file", error=str(e))

            # Create agent files with initial todos
            agent_chars = {}
            for i, agent_char in enumerate(characteristics.agents):
                agent_id = f"agent_{i+1}"

                # CRITICAL: Log todos count BEFORE conversion to verify fallback worked
                initial_todos_count = len(agent_char.initial_todos)
                logger.info(f"Creating agent file for {agent_id}",
                           role=agent_char.role,
                           initial_todos_count=initial_todos_count,
                           todos_titles=[t.title for t in agent_char.initial_todos])

                # Convert todos to AgentTodoItem
                agent_todos = [
                    AgentTodoItem(
                        reasoning=todo.reasoning,
                        title=todo.title,
                        objective=todo.objective,
                        expected_output=todo.expected_output,
                        sources_needed=todo.sources_needed,
                        status="pending",
                        note=todo.guidance if hasattr(todo, 'guidance') and todo.guidance else ""
                    )
                    for todo in agent_char.initial_todos
                ]
                
                # CRITICAL: Verify todos count after conversion
                if len(agent_todos) != initial_todos_count:
                    logger.error(f"CRITICAL: Todos count mismatch for {agent_id}",
                               before_conversion=initial_todos_count,
                               after_conversion=len(agent_todos))
                if len(agent_todos) < 2:
                    logger.error(f"CRITICAL: Agent {agent_id} has only {len(agent_todos)} tasks after fallback - this should not happen!",
                               role=agent_char.role,
                               todos_titles=[t.title for t in agent_todos])

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

                        # Emit todos to frontend so user can see agent progress immediately
                        if stream and agent_todos:
                            todos_dict = [
                                {
                                    "title": t.title,
                                    "status": t.status,
                                    "objective": t.objective,
                                    "expected_output": t.expected_output,
                                }
                                for t in agent_todos
                            ]
                            stream.emit_agent_todo(agent_id, todos_dict)
                            logger.info(f"Agent todos emitted to frontend", agent_id=agent_id, todos_count=len(todos_dict))
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


# Legacy function wrapper for backward compatibility
async def create_agent_characteristics_enhanced_node(state: ResearchState) -> Dict:
    """Legacy wrapper for SpawnAgentsNode.

    This function maintains backward compatibility with existing code
    that imports create_agent_characteristics_enhanced_node directly.

    TODO: Update imports to use SpawnAgentsNode class directly,
    then remove this wrapper.
    """
    from src.workflow.research.nodes import runtime_deps_context

    runtime_deps = runtime_deps_context.get()
    if not runtime_deps:
        logger.warning("Runtime dependencies not found in context")
        return {"agent_characteristics": []}

    # Create dependencies container
    from src.workflow.research.dependencies import ResearchDependencies

    deps = ResearchDependencies(
        llm=runtime_deps.get("llm"),
        search_provider=runtime_deps.get("search_provider"),
        scraper=runtime_deps.get("scraper"),
        stream=runtime_deps.get("stream"),
        agent_memory_service=runtime_deps.get("agent_memory_service"),
        agent_file_service=runtime_deps.get("agent_file_service"),
        session_factory=runtime_deps.get("session_factory"),
        session_manager=runtime_deps.get("session_manager"),
        settings=runtime_deps.get("settings"),
    )

    # Execute node
    node = SpawnAgentsNode(deps)
    return await node.execute(state)

"""Execute agents node for running research agents."""

import asyncio
import structlog
import re
from typing import Dict, Any
from datetime import datetime

from src.workflow.research.state import ResearchState
from src.workflow.research.nodes.base import ResearchNode
from src.workflow.research.nodes.utils import _restore_runtime_deps
from src.workflow.research.queue import get_supervisor_queue
from src.workflow.research.researcher import run_researcher_agent_enhanced
from src.workflow.research.supervisor_chain import run_supervisor_chain

logger = structlog.get_logger(__name__)


class ExecuteAgentsNode(ResearchNode):
    """Execute research agents in parallel."""

    async def execute(self, state: ResearchState) -> Dict[str, Any]:
        """Execute agents node.

        Args:
            state: Current research state

        Returns:
            State updates with findings
        """
        # Convert ResearchState to dict for compatibility
        if isinstance(state, dict):
            state_dict = state
        else:
            state_dict = dict(state)
        
        # Restore runtime dependencies if not in state
        state_dict = _restore_runtime_deps(state_dict)
        
        agent_characteristics = state_dict.get("agent_characteristics", {})
        agent_count = state_dict.get("agent_count", 4)
        llm = state_dict.get("llm")
        search_provider = state_dict.get("search_provider")
        scraper = state_dict.get("scraper")
        stream = state_dict.get("stream")
        settings = state_dict.get("settings")
        # Get max_iterations from settings (centralized config)
        if settings:
            max_iterations = state_dict.get("max_iterations", settings.deep_research_default_max_iterations)
        else:
            from src.config.settings import get_settings
            settings_obj = get_settings()
            max_iterations = state_dict.get("max_iterations", settings_obj.deep_research_default_max_iterations)
        current_iteration = state_dict.get("iteration", 0)

        # Don't emit status here - we'll emit after discovering agents from files

        # CRITICAL: Get supervisor queue for this session (tied to session_id)
        # This ensures queue persists across graph invocations and is session-specific
        session_id = state_dict.get("session_id", "unknown")
        supervisor_queue = get_supervisor_queue(session_id)
        logger.info("Using supervisor queue for session",
                   session_id=session_id,
                   queue_size=supervisor_queue.size(),
                   note="Queue is tied to session - findings from all agents in this session go here")

        # Store in state for agents to access
        state_dict["supervisor_queue"] = supervisor_queue

        # All collected findings from all agent iterations
        all_findings = []
        
        # Лимиты на вызовы сняты - супервизор вызывается без ограничений
        # Трекинг task_addition_count для лимита на добавление задач (максимум 2 раза)
        task_addition_count = state_dict.get("task_addition_count", 0)
        
        # Run agents in continuous mode until all todos complete or max iterations
        agents_active = True
        iteration_count = 0
        
        # Агенты возвращаются к работе через agents_to_return в следующем цикле
        
        # CRITICAL: Hard limit to prevent infinite loops
        # If max_iterations reached, MUST stop and generate report
        while agents_active and iteration_count < max_iterations:
            iteration_count += 1
            logger.info(f"Agent execution cycle {iteration_count}")
            
            if stream:
                stream.emit_status(f"🔄 Agent execution cycle {iteration_count}/{max_iterations}", step="agents")
                logger.info(f"Emitting progress: cycle {iteration_count}/{max_iterations}")
            
            # Убрана логика немедленного перезапуска агентов
            # Агенты теперь возвращаются к работе через agents_to_return в следующем цикле
            
            # Launch all agents in parallel for this iteration
            # Get max_steps from settings (centralized config)
            if settings:
                agent_max_steps = settings.deep_research_agent_max_steps
            else:
                from src.config.settings import get_settings
                settings_obj = get_settings()
                agent_max_steps = settings_obj.deep_research_agent_max_steps
            
            # CRITICAL FIX (Bug #26): Load agents from files, not agent_characteristics
            # Supervisor can create new agents via create_agent_todo, but they won't be in agent_characteristics
            # So we need to load ALL agents from files to include agent_2, agent_3, etc.
            agent_memory_service = stream.app_state.get("agent_memory_service") if stream else None
            agent_file_service = stream.app_state.get("agent_file_service") if stream else None

            if agent_file_service:
                try:
                    # Get list of all agent files
                    file_manager = agent_file_service.file_manager
                    agent_files = await file_manager.list_files("agents/agent_*.md")

                    # Extract agent IDs from filenames (e.g., "agents/agent_1.md" -> "agent_1")
                    discovered_agents = []
                    for file_path in agent_files:
                        agent_id = file_path.replace("agents/", "").replace(".md", "")
                        if agent_id.startswith("agent_"):
                            discovered_agents.append(agent_id)

                    logger.info(f"Discovered {len(discovered_agents)} agents from files",
                               agents=discovered_agents,
                               note="Loading agents from files instead of agent_characteristics to include supervisor-created agents")

                    # Use discovered agents if found, otherwise fallback to agent_characteristics
                    # CRITICAL: Check if discovered_agents is not empty (not just truthy check)
                    if discovered_agents and len(discovered_agents) > 0:
                        discovered_agents_list = discovered_agents
                    else:
                        # Fallback to agent_characteristics
                        discovered_agents_list = list(agent_characteristics.keys()) if agent_characteristics else []
                        logger.warning(f"No agents discovered from files, using agent_characteristics fallback",
                                     agents_from_characteristics=discovered_agents_list,
                                     note="If this is first iteration, ensure spawn_agents created all agents")
                except Exception as e:
                    logger.warning(f"Failed to discover agents from files, falling back to agent_characteristics", error=str(e))
                    discovered_agents_list = list(agent_characteristics.keys())
            else:
                # No file service, use agent_characteristics
                discovered_agents_list = list(agent_characteristics.keys())
            
            # CRITICAL: Store original discovered agents list for first iteration
            # In first iteration, we launch ALL discovered agents, not just those with tasks
            original_discovered_agents = discovered_agents_list.copy() if discovered_agents_list else []
            
            # For filtering logic, use discovered_agents_list
            agents_to_run = discovered_agents_list
            
            # CRITICAL: In the FIRST iteration, launch ALL agents with tasks in parallel
            # Agents wait for THEIR OWN finding to be processed, but work independently otherwise
            # An agent should NOT run if it has a finding waiting in queue (its own finding)
            # But agents work independently - they don't wait for other agents' findings
            
            # Check which agents have pending/in_progress tasks AND don't have findings in queue
            agents_to_run_filtered = []
            agents_waiting_for_review = []
            agents_with_tasks = []
            
            if agent_file_service and supervisor_queue:
                try:
                    for agent_id in agents_to_run:
                        agent_file = await agent_file_service.read_agent_file(agent_id)
                        todos = agent_file.get("todos", [])
                        pending = [t for t in todos if t.status == "pending"]
                        in_progress = [t for t in todos if t.status == "in_progress"]
                        has_tasks = len(pending) > 0 or len(in_progress) > 0
                        
                        if has_tasks:
                            agents_with_tasks.append(agent_id)
                            
                            # CRITICAL: Check if agent has finding in queue (waiting for supervisor review)
                            # Agent sleeps ONLY if its finding is in queue OR no tasks
                            try:
                                has_finding_in_queue = supervisor_queue.has_finding_from_agent(agent_id)
                            except AttributeError as e:
                                # If method doesn't exist, log warning but continue (shouldn't happen after fix)
                                logger.warning(f"SUPERVISOR: has_finding_from_agent method not found, assuming no findings in queue",
                                             agent_id=agent_id,
                                             error=str(e),
                                             note="This should not happen - method should exist in SupervisorQueue")
                                has_finding_in_queue = False
                            
                            if has_finding_in_queue:
                                # Agent is waiting for supervisor to process its finding - don't run it
                                agents_waiting_for_review.append(agent_id)
                                logger.info(f"SUPERVISOR: Agent {agent_id} has finding in queue - waiting for supervisor review (agent sleeps)",
                                           agent_id=agent_id,
                                           pending_tasks=len(pending),
                                           in_progress_tasks=len(in_progress),
                                           note="Agent sleeps while its finding is in queue")
                            else:
                                # Agent has tasks and no finding in queue - can run
                                agents_to_run_filtered.append(agent_id)
                                logger.info(f"SUPERVISOR: Agent {agent_id} can run - has tasks and no finding in queue",
                                           agent_id=agent_id,
                                           pending_tasks=len(pending),
                                           in_progress_tasks=len(in_progress),
                                           note="Agent will run in parallel with other agents")
                except Exception as e:
                    logger.warning("SUPERVISOR: Failed to check agent tasks and queue status", error=str(e), exc_info=True)
                    # Fallback: in first iteration, run all discovered agents; in subsequent iterations, run all agents with tasks
                    if iteration_count == 1:
                        # First iteration: run all discovered agents (tasks may not be created yet)
                        agents_to_run_filtered = original_discovered_agents.copy() if original_discovered_agents else []
                        logger.warning("SUPERVISOR: Error checking tasks in first iteration - running all discovered agents as fallback",
                                     agents=agents_to_run_filtered,
                                     note="First iteration fallback - all discovered agents will run")
                    else:
                        # Subsequent iterations: run all agents with tasks
                        agents_to_run_filtered = agents_with_tasks
                        logger.warning("SUPERVISOR: Error checking tasks in subsequent iteration - running agents with tasks as fallback",
                                     agents=agents_to_run_filtered,
                                     note="Subsequent iteration fallback - only agents with tasks will run")
            else:
                # No file service or queue - run all discovered agents (fallback)
                agents_to_run_filtered = agents_to_run
                logger.warning("No agent_file_service or supervisor_queue - running all discovered agents as fallback",
                             agents_count=len(agents_to_run_filtered))
            
            # CRITICAL: In the FIRST iteration, launch ALL discovered agents in parallel
            # This ensures parallel execution from the start
            # In first iteration, queue is empty, and tasks may not be created yet
            # So we launch ALL discovered agents, not just those with tasks
            if iteration_count == 1:
                # In first iteration, launch ALL discovered agents in parallel
                # Tasks may be created during agent execution or by supervisor
                logger.info("FIRST ITERATION: Launching ALL discovered agents in parallel",
                           discovered_agents=original_discovered_agents,
                           agents_with_tasks=agents_with_tasks,
                           agents_to_run_filtered=agents_to_run_filtered,
                           agent_characteristics_keys=list(agent_characteristics.keys()) if agent_characteristics else [],
                           note="First iteration - all discovered agents launch in parallel, regardless of tasks")
                
                # CRITICAL: In first iteration, use ALL discovered agents OR agent_characteristics
                # This ensures all agents start working in parallel from the beginning
                if original_discovered_agents and len(original_discovered_agents) > 0:
                    agents_to_run = original_discovered_agents.copy()
                elif agent_characteristics and len(agent_characteristics) > 0:
                    # Fallback: use agent_characteristics if no agents discovered from files
                    agents_to_run = list(agent_characteristics.keys())
                    logger.warning("FIRST ITERATION: No agents discovered from files, using agent_characteristics",
                                 agents_from_characteristics=agents_to_run,
                                 note="This may indicate agents were not created in spawn_agents node")
                    
                    # CRITICAL: Create agent files with tasks from agent_characteristics if they don't exist
                    # This ensures agents have tasks even if spawn_agents didn't create files
                    if agent_file_service:
                        try:
                            from src.models.agent_models import AgentTodoItem
                            
                            for agent_id, agent_char in agent_characteristics.items():
                                # Check if agent file exists
                                agent_file = await agent_file_service.read_agent_file(agent_id)
                                existing_todos = agent_file.get("todos", [])
                                
                                # If agent file has no tasks, create them from agent_characteristics
                                if not existing_todos or len(existing_todos) == 0:
                                    # Get initial_todos from agent_characteristics
                                    initial_todos = agent_char.get("initial_todos", [])
                                    
                                    if initial_todos:
                                        # Convert to AgentTodoItem format
                                        agent_todos = [
                                            AgentTodoItem(
                                                reasoning=todo.get("reasoning", ""),
                                                title=todo.get("title", ""),
                                                objective=todo.get("objective", ""),
                                                expected_output=todo.get("expected_output", ""),
                                                sources_needed=todo.get("sources_needed", []),
                                                status="pending",
                                                note=todo.get("guidance", "") if isinstance(todo, dict) and "guidance" in todo else ""
                                            )
                                            for todo in initial_todos
                                        ]
                                        
                                        # Create agent file with tasks
                                        await agent_file_service.write_agent_file(
                                            agent_id=agent_id,
                                            todos=agent_todos,
                                            character=f"""**Role**: {agent_char.get("role", "")}
**Expertise**: {agent_char.get("expertise", "")}
**Personality**: {agent_char.get("personality", "")}
""",
                                            preferences=f"Focus on: {agent_char.get('expertise', '')}"
                                        )
                                        
                                        logger.info(f"Created agent file with tasks from agent_characteristics (fallback)",
                                                   agent_id=agent_id,
                                                   todos_count=len(agent_todos),
                                                   note="Agent file was missing - created from agent_characteristics fallback")
                                        
                                        # Emit todos to frontend
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
                        except Exception as e:
                            logger.error("Failed to create agent files from agent_characteristics (fallback)", error=str(e), exc_info=True)
                else:
                    agents_to_run = []
                    logger.error("FIRST ITERATION: No agents found! Check spawn_agents node.",
                               discovered_agents=original_discovered_agents,
                               agent_characteristics_keys=list(agent_characteristics.keys()) if agent_characteristics else [],
                               note="No agents discovered from files AND no agents in agent_characteristics")
                
                # But still filter out agents that have findings in queue (shouldn't happen in first iteration, but be safe)
                if supervisor_queue:
                    agents_before_filter = len(agents_to_run)
                    try:
                        agents_to_run = [
                            agent_id for agent_id in agents_to_run
                            if not supervisor_queue.has_finding_from_agent(agent_id)
                        ]
                        if agents_before_filter != len(agents_to_run):
                            logger.warning("FIRST ITERATION: Some agents filtered out due to findings in queue (unexpected!)",
                                         agents_before_filter=agents_before_filter,
                                         agents_after_filter=len(agents_to_run),
                                         note="This should not happen in first iteration - queue should be empty")
                    except AttributeError as e:
                        # If method doesn't exist, log warning but continue (shouldn't happen after fix)
                        logger.warning("FIRST ITERATION: has_finding_from_agent method not found, skipping filter",
                                     error=str(e),
                                     agents_count=len(agents_to_run),
                                     note="This should not happen - method should exist in SupervisorQueue. All agents will run.")
                
                logger.info("FIRST ITERATION: Final agents to launch",
                           agents_count=len(agents_to_run),
                           agents=agents_to_run,
                           note="First iteration - all discovered agents (without findings in queue) will launch in parallel")
            else:
                # In subsequent iterations, use filtered agents (only those with tasks and no findings in queue)
                agents_to_run = agents_to_run_filtered
                logger.info(f"ITERATION {iteration_count}: Using filtered agents",
                           agents_count=len(agents_to_run),
                           agents=agents_to_run,
                           agents_with_tasks=agents_with_tasks,
                           agents_waiting_for_review=agents_waiting_for_review,
                           note="Subsequent iteration - only agents with tasks and no findings in queue will run")
            
            # CRITICAL: agents_to_return is ONLY for logging - it doesn't affect which agents run
            # Agents are selected based on: has_tasks AND no_finding_in_queue
            # Processing other agents' findings doesn't affect running agents
            supervisor_decision = state_dict.get("supervisor_decision", {})
            agents_to_return = supervisor_decision.get("agents_to_return", [])
            
            logger.info(f"SUPERVISOR: Agent execution decision for cycle {iteration_count}",
                       agents_with_tasks=agents_with_tasks,
                       agents_waiting_for_review=agents_waiting_for_review,
                       agents_to_run=agents_to_run,
                       agents_to_return=agents_to_return,
                       note="Agents with tasks but no finding in queue will run. Agents waiting for review will NOT run. Processing other agents' findings doesn't affect running agents.")

            # Emit status with ACTUAL agent count
            if stream:
                stream.emit_status(f"Executing {len(agents_to_run)} research agents in parallel...", step="agents")

            # CRITICAL: Start continuous supervisor queue processing in background BEFORE launching agents
            # This ensures supervisor is ready to process findings as soon as agents add them to queue
            # Supervisor queue processing runs continuously in parallel with agents from the start
            supervisor_processing_task = None
            supervisor_processing_active = True
            
            # CRITICAL: Track dynamically added tasks (agents returned to work)
            dynamically_added_tasks = []
            
            # Create mapping for tracking agent tasks (will be populated when agents are launched)
            processed_agents = set()
            
            # CRITICAL: Launch continuous supervisor processing FIRST, before agents
            # CRITICAL: Lock for sequential processing of findings from start to finish
            # This ensures one finding is fully processed (including agent return) before starting the next
            supervisor_processing_lock = asyncio.Lock()
            
            # This ensures supervisor is ready when agents start adding findings to queue
            async def continuous_supervisor_processing():
                """Continuously process supervisor queue in background."""
                nonlocal supervisor_processing_active
                # Initialize task_addition_count from state_dict
                task_addition_count = state_dict.get("task_addition_count", 0)
                
                logger.info("SUPERVISOR: Starting continuous queue processing in background (BEFORE agents launch)",
                           agents_to_launch=len(agents_to_run),
                           initial_queue_size=supervisor_queue.size() if supervisor_queue else 0,
                           note="Supervisor queue processing starts FIRST, then agents launch in parallel. Processing is SEQUENTIAL - one finding at a time from start to finish. CRITICAL: Any findings already in queue will be processed immediately.")
                
                # CRITICAL: Check if queue already has findings (from previous iterations or agents that completed before processing started)
                if supervisor_queue and supervisor_queue.size() > 0:
                    logger.info("SUPERVISOR: Queue already has findings before processing starts",
                              queue_size=supervisor_queue.size(),
                              note="These findings will be processed immediately. This can happen if agents completed tasks before continuous processing started.")
                
                while supervisor_processing_active:
                    try:
                        # CRITICAL: Continuously process queue - researchers work in parallel and add findings
                        # Supervisor processes them sequentially from queue
                        if supervisor_queue:
                            queue_size = supervisor_queue.size()
                            
                            # CRITICAL: If queue has findings, process immediately
                            # Don't wait - researchers are adding findings in parallel
                            if queue_size > 0:
                                logger.info(
                                    "SUPERVISOR: Processing finding from queue (continuous processing)",
                                    queue_size=queue_size,
                                    note="Researchers work in parallel and add findings - supervisor processes sequentially. Using lock to ensure one finding at a time from start to finish."
                                )
                                
                                # CRITICAL: Use lock to ensure sequential processing of entire cycle
                                # This guarantees that one finding is fully processed (including agent return)
                                # before starting to process the next finding
                                # CRITICAL: Lock is acquired BEFORE processing finding and released AFTER agent return
                                # This ensures agents return to work sequentially, not simultaneously
                                logger.info(
                                    "SUPERVISOR: Attempting to acquire processing lock",
                                    queue_size=queue_size,
                                    note="Will wait for lock if another finding is being processed. This ensures sequential agent returns."
                                )
                                
                                async with supervisor_processing_lock:
                                    # CRITICAL: Log that we're starting processing with lock
                                    logger.info(
                                        "SUPERVISOR: Acquired processing lock - starting sequential processing",
                                        queue_size=queue_size,
                                        note="Lock acquired. One finding will be fully processed (including agent return) before next finding starts. Other findings in queue will wait."
                                    )
                                    
                                    try:
                                        if stream:
                                            stream.emit_status(f"👔 Supervisor processing finding from queue (queue size: {queue_size})", step="supervisor")
                                        
                                        # Process finding through supervisor chain
                                        # CRITICAL: run_supervisor_chain will get finding from queue internally
                                        # It uses queue.get() - if queue has items, it will get immediately
                                        decision = await run_supervisor_chain(
                                            state=state_dict,
                                            llm=llm,
                                            stream=stream,
                                            supervisor_queue=supervisor_queue
                                        )
                                        
                                        # CRITICAL: Check if we actually processed a finding
                                        processed_agent_id = decision.get("processed_agent_id")
                                        chapter_written_from_decision = decision.get("chapter_written", False)
                                        if not processed_agent_id:
                                            # No finding was processed (shouldn't happen if queue_size > 0, but handle gracefully)
                                            logger.warning(
                                                "SUPERVISOR: No finding processed despite queue_size > 0",
                                                queue_size=supervisor_queue.size(),
                                                decision_keys=list(decision.keys()) if isinstance(decision, dict) else None,
                                                note="This may indicate a race condition. Will retry."
                                            )
                                            await asyncio.sleep(0.1)  # Short wait before retry
                                            continue
                                        
                                        # CRITICAL: Log detailed finding processing result
                                        finding_topic = decision.get("finding_topic", "unknown")
                                        chapter_number = decision.get("chapter_number")
                                        chapter_title = decision.get("chapter_title")
                                        
                                        logger.info(
                                            "SUPERVISOR: Finding processed successfully",
                                            processed_agent_id=processed_agent_id,
                                            finding_topic=finding_topic,
                                            chapter_written=chapter_written_from_decision,
                                            chapter_number=chapter_number,
                                            chapter_title=chapter_title,
                                            action=decision.get("action", "unknown"),
                                            should_return_agent=decision.get("should_return_agent", False),
                                            note="CRITICAL: Finding was processed. If chapter_written=True, it's in draft_report. If False, check logs for reason (validation failed, rework needed, or write error)."
                                        )
                                        
                                        # CRITICAL: Log if chapter was NOT written
                                        if not chapter_written_from_decision:
                                            logger.error(
                                                "SUPERVISOR: Finding was processed but chapter was NOT written",
                                                processed_agent_id=processed_agent_id,
                                                finding_topic=finding_topic,
                                                action=decision.get("action", "unknown"),
                                                is_valid=decision.get("is_valid"),
                                                needs_rework=decision.get("needs_rework"),
                                                note="CRITICAL: This finding will NOT appear in draft_report. Check validation_result and chapter_result in supervisor_chain logs for details."
                                            )
                                        
                                        # Update task_addition_count
                                        if "task_management" in decision and decision["task_management"]:
                                            task_management = decision["task_management"]
                                            if isinstance(task_management, dict):
                                                task_addition_count = task_management.get("task_addition_count", task_addition_count)
                                            else:
                                                # If this is TaskManagementResult object
                                                task_addition_count = getattr(task_management, "task_addition_count", task_addition_count)
                                            state_dict["task_addition_count"] = task_addition_count
                                        
                                        state_dict["should_continue"] = decision.get("should_continue", False)
                                        state_dict["replanning_needed"] = decision.get("replanning_needed", False)
                                        state_dict["supervisor_decision"] = decision
                                        
                                        # CRITICAL: Return ONLY the processed agent to work immediately after processing its finding
                                        # Logic:
                                        # - If chapter NOT written (needs_rework=True) → agent returns to SAME task (in_progress)
                                        # - If chapter written (accepted) → agent returns to NEXT task (pending)
                                        # - Agent sleeps ONLY if its finding is in queue OR no tasks
                                        # - Agent launches IMMEDIATELY after its finding is processed
                                        chapter_written = decision.get("chapter_written", False)
                                        action = decision.get("action", "unknown")
                                        
                                        if agent_file_service and processed_agent_id:
                                            try:
                                                agent_file = await agent_file_service.read_agent_file(processed_agent_id)
                                                todos = agent_file.get("todos", [])
                                                pending = [t for t in todos if t.status == "pending"]
                                                in_progress = [t for t in todos if t.status == "in_progress"]
                                                has_tasks = len(pending) > 0 or len(in_progress) > 0
                                                
                                                # CRITICAL: Check if agent has finding in queue (waiting for supervisor)
                                                # Agent sleeps ONLY if its finding is in queue OR no tasks
                                                has_finding_in_queue = supervisor_queue.has_finding_from_agent(processed_agent_id) if supervisor_queue else False
                                                
                                                # Check if agent is already running (in any task list)
                                                agent_already_running = (
                                                    (processed_agent_id in tasks_by_agent and not tasks_by_agent[processed_agent_id].done()) or
                                                    any(aid == processed_agent_id for aid, _ in agent_tasks if aid in tasks_by_agent and not tasks_by_agent[aid].done())
                                                )
                                                
                                                # Check should_return_agent from decision
                                                should_return_agent = decision.get("should_return_agent", False)
                                                
                                                logger.info(
                                                    "SUPERVISOR: Checking if processed agent should return to work",
                                                    agent_id=processed_agent_id,
                                                    has_tasks=has_tasks,
                                                    pending_count=len(pending),
                                                    in_progress_count=len(in_progress),
                                                    has_finding_in_queue=has_finding_in_queue,
                                                    agent_already_running=agent_already_running,
                                                    should_return_agent=should_return_agent,
                                                    chapter_written=chapter_written,
                                                    action=action,
                                                    note="Agent will return to work if: has_tasks AND not has_finding_in_queue AND not agent_already_running"
                                                )
                                                
                                                # CRITICAL: Agent should return to work if:
                                                # 1. Has tasks (pending or in_progress)
                                                # 2. Its finding is NOT in queue (was just processed)
                                                # 3. Agent is not already running
                                                # 4. should_return_agent is True (from decision)
                                                if has_tasks and not has_finding_in_queue and not agent_already_running and should_return_agent:
                                                    # Determine which task agent will work on
                                                    task_to_work_on = None
                                                    if chapter_written:
                                                        # Chapter accepted → work on NEXT task (pending)
                                                        if pending:
                                                            task_to_work_on = pending[0]
                                                            task_status = "pending"
                                                        elif in_progress:
                                                            # No pending tasks, but has in_progress (shouldn't happen, but handle)
                                                            task_to_work_on = in_progress[0]
                                                            task_status = "in_progress"
                                                    else:
                                                        # Chapter NOT written (needs_rework) → work on SAME task (in_progress)
                                                        if in_progress:
                                                            task_to_work_on = in_progress[0]
                                                            task_status = "in_progress"
                                                        elif pending:
                                                            # No in_progress, but has pending (shouldn't happen, but handle)
                                                            task_to_work_on = pending[0]
                                                            task_status = "pending"
                                                    
                                                    if task_to_work_on:
                                                        logger.info(
                                                            "SUPERVISOR: Returning processed agent to work IMMEDIATELY after finding processing",
                                                            agent_id=processed_agent_id,
                                                            chapter_written=chapter_written,
                                                            action=action,
                                                            task_title=task_to_work_on.title,
                                                            task_status=task_status,
                                                            pending_tasks=len(pending),
                                                            in_progress_tasks=len(in_progress),
                                                            note=f"Agent's finding was processed. Chapter {'written' if chapter_written else 'NOT written (rework)'}. Returning to {'next task' if chapter_written else 'same task (rework)'} IMMEDIATELY. Other agents with findings in queue will wait for their turn."
                                                        )
                                                        
                                                        # CRITICAL: Launch agent immediately - it will pick up the appropriate task
                                                        # This happens INSIDE the lock, ensuring sequential agent returns
                                                        # CRITICAL: We need to ensure agent actually starts before releasing lock
                                                        # to prevent multiple agents from starting simultaneously
                                                        logger.info(
                                                            "SUPERVISOR: Creating agent task (inside lock)",
                                                            agent_id=processed_agent_id,
                                                            task_title=task_to_work_on.title,
                                                            note="Agent task will be created inside lock. Lock will be held until agent task is created and scheduled."
                                                        )
                                                        
                                                        # CRITICAL: Create task and ensure it's scheduled before releasing lock
                                                        # This prevents multiple agents from starting simultaneously
                                                        agent_task = asyncio.create_task(
                                                            run_researcher_agent_enhanced(
                                                                agent_id=processed_agent_id,
                                                                state=state_dict,
                                                                llm=llm,
                                                                search_provider=search_provider,
                                                                scraper=scraper,
                                                                stream=stream,
                                                                supervisor_queue=supervisor_queue,
                                                                max_steps=agent_max_steps
                                                            )
                                                        )
                                                        
                                                        # CRITICAL: Give event loop a chance to schedule the task
                                                        # This ensures agent task is actually scheduled before we release lock
                                                        # Without this, multiple agents might start simultaneously
                                                        await asyncio.sleep(0)  # Yield to event loop to schedule task
                                                        
                                                        # Add to tracking
                                                        agent_tasks.append((processed_agent_id, agent_task))
                                                        dynamically_added_tasks.append((processed_agent_id, agent_task))
                                                        tasks_by_agent[processed_agent_id] = agent_task
                                                        
                                                        logger.info(
                                                            "SUPERVISOR: Agent task created, scheduled, and added to tracking (still inside lock)",
                                                            agent_id=processed_agent_id,
                                                            total_agent_tasks=len(agent_tasks),
                                                            task_scheduled=not agent_task.done(),
                                                            note="Agent task is now scheduled and running. Lock will be released after this, ensuring next finding waits until this agent is started."
                                                        )
                                                        
                                                        if stream:
                                                            if chapter_written:
                                                                stream.emit_status(f"🚀 Agent {processed_agent_id} continuing work on next task", step="agents")
                                                            else:
                                                                stream.emit_status(f"🔄 Agent {processed_agent_id} reworking task", step="agents")
                                                    else:
                                                        logger.warning(
                                                            "SUPERVISOR: Processed agent has tasks but couldn't determine which task to work on",
                                                            agent_id=processed_agent_id,
                                                            pending_count=len(pending),
                                                            in_progress_count=len(in_progress),
                                                            chapter_written=chapter_written,
                                                            note="This shouldn't happen - agent has tasks but task_to_work_on is None"
                                                        )
                                                else:
                                                    logger.info(
                                                        "SUPERVISOR: Processed agent will NOT return to work",
                                                        agent_id=processed_agent_id,
                                                        has_tasks=has_tasks,
                                                        has_finding_in_queue=has_finding_in_queue,
                                                        agent_already_running=agent_already_running,
                                                        should_return_agent=should_return_agent,
                                                        note="Agent will wait: either no tasks, finding in queue, already running, or should_return_agent=False"
                                                    )
                                                
                                                # Additional logging for specific cases
                                                if has_finding_in_queue:
                                                    logger.info(
                                                        "SUPERVISOR: Processed agent still has finding in queue - agent sleeps",
                                                        agent_id=processed_agent_id,
                                                        note="Agent sleeps while its finding is in queue waiting for supervisor"
                                                    )
                                                elif agent_already_running:
                                                    logger.info(
                                                        "SUPERVISOR: Processed agent already running",
                                                        agent_id=processed_agent_id,
                                                        note="Agent is already working - no need to restart"
                                                    )
                                                else:
                                                    logger.info(
                                                        "SUPERVISOR: Processed agent has no more tasks - agent sleeps",
                                                        agent_id=processed_agent_id,
                                                        note="Agent sleeps - no tasks available"
                                                    )
                                            except Exception as e:
                                                logger.error(f"Failed to return processed agent {processed_agent_id} to work (continuous)", error=str(e), exc_info=True)
                                        
                                        # CRITICAL: Log completion of full processing cycle
                                        # NOTE: Lock is still held here - it will be released after this block
                                        # This ensures agent return happens before next finding is processed
                                        logger.info(
                                            "SUPERVISOR: Full processing cycle completed (finding processed + agent returned if needed)",
                                            agent_id=processed_agent_id if processed_agent_id else "none",
                                            queue_size_remaining=supervisor_queue.size() if supervisor_queue else 0,
                                            note="Entire cycle from finding retrieval to agent return completed. Lock will be released - next finding can be processed."
                                        )
                                        
                                        # CRITICAL: Give event loop a final chance to schedule any pending tasks
                                        # This ensures agent task is actually scheduled before lock is released
                                        # Without this, multiple agents might start simultaneously
                                        await asyncio.sleep(0)
                                        
                                        # CRITICAL: Log that we're about to release lock
                                        logger.info(
                                            "SUPERVISOR: About to release processing lock",
                                            agent_id=processed_agent_id if processed_agent_id else "none",
                                            queue_size_remaining=supervisor_queue.size() if supervisor_queue else 0,
                                            note="Lock will be released now. Next finding in queue can start processing. Agent task has been created and scheduled."
                                        )
                                        
                                        # Lock will be released when we exit this async with block
                                        # This ensures that:
                                        # 1. Finding is fully processed
                                        # 2. Agent is returned to work (task created and scheduled)
                                        # 3. Only then next finding can start processing
                                    
                                    except Exception as e:
                                        logger.error("SUPERVISOR: Error processing finding (continuous)", error=str(e), exc_info=True)
                                        # CRITICAL: Even if processing fails, continue processing queue
                                        # Don't break the loop - other findings need to be processed
                                        # The error is logged, but we continue to process next finding
                                        # Wait a bit before retrying to avoid tight error loop
                                        await asyncio.sleep(0.5)
                            else:
                                # Queue is empty - wait a bit before checking again
                                # But don't wait too long - researchers may add findings at any time
                                await asyncio.sleep(0.3)
                        else:
                            # No supervisor_queue available - wait a bit before checking again
                            await asyncio.sleep(0.5)
                    except Exception as e:
                        logger.error("SUPERVISOR: Error in continuous processing loop", error=str(e), exc_info=True)
                        await asyncio.sleep(1.0)  # Wait longer on error
                
                logger.info("SUPERVISOR: Continuous queue processing stopped")
            
            # CRITICAL: Initialize agent_tasks and tasks_by_agent BEFORE starting continuous supervisor processing
            # This ensures continuous supervisor processing can access them when needed
            agent_tasks = []
            tasks_by_agent = {}
            
            # Start continuous supervisor processing BEFORE launching agents
            # This ensures supervisor is ready when agents start adding findings to queue
            supervisor_processing_task = asyncio.create_task(continuous_supervisor_processing())
            logger.info("SUPERVISOR: Continuous queue processing started in background (BEFORE agents launch)",
                       agents_to_launch=len(agents_to_run),
                       agents_list=agents_to_run,
                       note="Supervisor queue processing starts FIRST, then all agents launch in parallel. Findings from agents go to queue, supervisor processes them immediately.")
            
            # CRITICAL: Launch ALL agents simultaneously, not sequentially
            # Create all tasks first, then they all run in parallel
            # All agents start working in parallel from the beginning
            for agent_id in agents_to_run:
                task = asyncio.create_task(
                    run_researcher_agent_enhanced(
                        agent_id=agent_id,
                        state=state_dict,
                        llm=llm,
                        search_provider=search_provider,
                        scraper=scraper,
                        stream=stream,
                        supervisor_queue=supervisor_queue,
                        max_steps=agent_max_steps
                    )
                )
                agent_tasks.append((agent_id, task))
                tasks_by_agent[agent_id] = task
                logger.info(f"Created task for agent {agent_id} in cycle {iteration_count}",
                           note="Task created, will run in parallel with other agents")

            # CRITICAL: All tasks are now created and running in parallel
            # asyncio.create_task immediately schedules them for execution
            logger.info(f"Launched {len(agent_tasks)} agents in parallel for cycle {iteration_count}",
                       agents=agents_to_run,
                       note="All agents run concurrently via asyncio.create_task - they execute simultaneously, not sequentially")
            
            if stream:
                stream.emit_status(f"🚀 Launched {len(agent_tasks)} agents in parallel for cycle {iteration_count}", step="agents")

            # Collect results from this cycle - process completions as they arrive
            # CRITICAL: Agents complete tasks independently and queue for supervisor immediately
            # We process supervisor queue as agents complete, not waiting for all
            cycle_findings = []
            no_tasks_count = 0
            
            # Use asyncio.as_completed to process agents as they finish
            # This allows supervisor to review immediately when agent completes
            logger.info(f"Waiting for {len(agent_tasks)} agents to complete (processing supervisor queue as agents finish)...")
            
            # Process agents as they complete using as_completed
            # This processes each agent immediately when it finishes
            completed_count = 0
            pending_tasks_list = [task for _, task in agent_tasks]
            
            # CRITICAL: Track all agent tasks to ensure we process all of them
            # Also track dynamically added tasks (agents returned to work after finding processing)
            logger.info(f"Processing {len(pending_tasks_list)} agent tasks as they complete",
                       agent_ids=agents_to_run)
            
            for completed_coro in asyncio.as_completed(pending_tasks_list):
                try:
                    result = await completed_coro
                    # Find which agent this result belongs to
                    agent_id = None
                    
                    # First try: get agent_id from result dict
                    if isinstance(result, dict):
                        agent_id = result.get("agent_id")
                    
                    # Second try: find by checking which task is done and not yet processed
                    if not agent_id:
                        for agent_id_check, task in tasks_by_agent.items():
                            if agent_id_check in processed_agents:
                                continue  # Skip already processed agents
                            if task.done():
                                try:
                                    task_result = task.result()
                                    # If task_result has agent_id matching agent_id_check, this is the one
                                    if isinstance(task_result, dict):
                                        if task_result.get("agent_id") == agent_id_check:
                                            agent_id = agent_id_check
                                            processed_agents.add(agent_id_check)
                                            break
                                except Exception as e:
                                    logger.debug(f"Error checking task result for {agent_id_check}", error=str(e))
                                    pass
                    
                    # Third try: if still not found, check all done tasks and pick first unprocessed
                    if not agent_id:
                        for agent_id_check, task in tasks_by_agent.items():
                            if agent_id_check in processed_agents:
                                continue
                            if task.done():
                                agent_id = agent_id_check
                                processed_agents.add(agent_id_check)
                                logger.info(f"Matched completed task to agent {agent_id} by process of elimination")
                                break
                    
                    if agent_id:
                        completed_count += 1
                        # Mark this agent as processed
                        processed_agents.add(agent_id)
                        if isinstance(result, Exception):
                            logger.error(f"Agent {agent_id} failed", error=str(result), exc_info=result)
                        elif result:
                            if result.get("topic") == "no_tasks":
                                no_tasks_count += 1
                                logger.info(f"Agent {agent_id} has no tasks",
                                          agent_id=agent_id,
                                          note="Agent has no pending or in_progress tasks - all tasks are done. This is normal if agent completed all tasks.")
                            else:
                                cycle_findings.append(result)
                                all_findings.append(result)
                                logger.info(f"Agent {agent_id} completed task", 
                                          task=result.get("topic", "unknown"),
                                          sources=len(result.get("sources", [])),
                                          completed_agents=f"{completed_count}/{len(agent_tasks)}",
                                          finding_agent_id=result.get("agent_id"),
                                          finding_topic=result.get("topic"),
                                          finding_summary_length=len(result.get("summary", "")),
                                          note="Result queued for supervisor. Continuous supervisor processing will handle it immediately. CRITICAL: Finding MUST be processed and written to draft_report as a chapter.")
                                
                                # CRITICAL: Finding is already in supervisor_queue (added by agent)
                                # Continuous supervisor processing will handle it immediately
                                # No need to call supervisor here - continuous processing does it
                                
                                # Add finding to state for reference
                                if result:
                                    existing_findings = state_dict.get("findings", [])
                                    finding_already_in_state = any(
                                        f.get("topic") == result.get("topic") and f.get("agent_id") == result.get("agent_id")
                                        for f in existing_findings
                                    )
                                    if not finding_already_in_state:
                                        existing_findings.append(result)
                                        state_dict["findings"] = existing_findings
                                        state_dict["agent_findings"] = existing_findings
                                        logger.info(f"Added finding to state",
                                                   finding_topic=result.get("topic", "unknown"),
                                                   finding_agent_id=result.get("agent_id"),
                                                   total_findings_in_state=len(existing_findings),
                                                   note="Finding added to state. Continuous supervisor processing will handle it and write chapter to draft_report.")
                    else:
                        logger.warning(f"Could not identify agent for completed task", 
                                     result_type=type(result).__name__,
                                     result_keys=list(result.keys()) if isinstance(result, dict) else None,
                                     processed_agents=list(processed_agents),
                                     total_agents=len(agent_tasks),
                                     note="This may indicate an issue with agent task completion tracking")
                except Exception as e:
                    logger.error(f"Error processing agent completion", error=str(e), exc_info=e)
                    # CRITICAL: Log which agents haven't completed yet
                    incomplete_agents = [aid for aid, task in tasks_by_agent.items() if aid not in processed_agents and not task.done()]
                    if incomplete_agents:
                        logger.warning(f"Agents not yet completed: {incomplete_agents}", 
                                     total_agents=len(agent_tasks),
                                     completed=len(processed_agents),
                                     incomplete=len(incomplete_agents))
            
            # CRITICAL: Verify all agents were processed
            unprocessed_agents = [aid for aid in agents_to_run if aid not in processed_agents]
            if unprocessed_agents:
                logger.warning(f"Cycle {iteration_count}: Some agents were not processed",
                             unprocessed_agents=unprocessed_agents,
                             processed_count=len(processed_agents),
                             total_agents=len(agent_tasks),
                             note="This may indicate agents are hanging or not completing")
                # Try to get results from unprocessed agents
                for agent_id in unprocessed_agents:
                    if agent_id in tasks_by_agent:
                        task = tasks_by_agent[agent_id]
                        if task.done():
                            try:
                                result = task.result()
                                if isinstance(result, dict):
                                    agent_id_from_result = result.get("agent_id")
                                    if agent_id_from_result == agent_id:
                                        logger.info(f"Found unprocessed result for agent {agent_id}, processing now")
                                        if result.get("topic") == "no_tasks":
                                            no_tasks_count += 1
                                        else:
                                            cycle_findings.append(result)
                                            all_findings.append(result)
                                        processed_agents.add(agent_id)
                                        completed_count += 1
                            except Exception as e:
                                logger.error(f"Error processing unprocessed agent {agent_id}", error=str(e))
            
            # CRITICAL: Process dynamically added tasks (agents returned to work immediately after finding processing)
            # These agents were launched during the main loop, but need to be processed separately
            if dynamically_added_tasks:
                logger.info(
                    "SUPERVISOR: Processing dynamically added agent tasks (agents returned to work immediately)",
                    dynamically_added_count=len(dynamically_added_tasks),
                    agent_ids=[aid for aid, _ in dynamically_added_tasks],
                    note="These agents were launched immediately after their finding was processed, not waiting for other agents"
                )
                
                # Process dynamically added tasks as they complete
                dynamic_tasks_list = [task for _, task in dynamically_added_tasks]
                for completed_coro in asyncio.as_completed(dynamic_tasks_list):
                    try:
                        result = await completed_coro
                        # Find which agent this result belongs to
                        agent_id = None
                        for aid, task in dynamically_added_tasks:
                            if task.done():
                                try:
                                    task_result = task.result()
                                    if isinstance(task_result, dict) and task_result.get("agent_id") == aid:
                                        agent_id = aid
                                        break
                                except:
                                    pass
                        
                        if agent_id:
                            if isinstance(result, Exception):
                                logger.error(f"Agent {agent_id} (dynamic) failed", error=str(result), exc_info=result)
                            elif result:
                                if result.get("topic") == "no_tasks":
                                    no_tasks_count += 1
                                    logger.info(f"Agent {agent_id} (dynamic) has no tasks")
                                else:
                                    cycle_findings.append(result)
                                    all_findings.append(result)
                                    logger.info(
                                        f"Agent {agent_id} (dynamic) completed task",
                                        task=result.get("topic", "unknown"),
                                        sources=len(result.get("sources", [])),
                                        note="Result queued for supervisor. Continuous supervisor processing will handle it immediately."
                                    )
                                    
                                    # CRITICAL: Finding is already in supervisor_queue (added by agent)
                                    # Continuous supervisor processing will handle it immediately
                                    # No need to call supervisor here - continuous processing does it
                                    
                                    # Add finding to state for reference
                                    if result:
                                        existing_findings = state_dict.get("findings", [])
                                        finding_already_in_state = any(
                                            f.get("topic") == result.get("topic") and f.get("agent_id") == result.get("agent_id")
                                            for f in existing_findings
                                        )
                                        if not finding_already_in_state:
                                            existing_findings.append(result)
                                            state_dict["findings"] = existing_findings
                                            state_dict["agent_findings"] = existing_findings
                                            logger.info(f"Added finding to state (dynamic)",
                                                       finding_topic=result.get("topic", "unknown"),
                                                       total_findings_in_state=len(existing_findings),
                                                       note="Continuous supervisor processing will handle it")
                    except Exception as e:
                        logger.error(f"Error processing dynamic agent completion", error=str(e), exc_info=True)
            
            # Stop continuous supervisor processing
            supervisor_processing_active = False
            if supervisor_processing_task:
                try:
                    await asyncio.wait_for(supervisor_processing_task, timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning("Supervisor processing task did not stop in time")
                except Exception as e:
                    logger.error("Error stopping supervisor processing task", error=str(e))
            
            logger.info(f"SUPERVISOR: Cycle {iteration_count} complete",
                       tasks_completed=len(cycle_findings),
                       agents_with_no_tasks=no_tasks_count,
                       agents_processed=f"{completed_count}/{len(agent_tasks)}",
                       dynamically_added_count=len(dynamically_added_tasks),
                       processed_agents=list(processed_agents),
                       unprocessed_agents=unprocessed_agents if unprocessed_agents else None,
                       supervisor_decision_agents_to_return=state_dict.get("supervisor_decision", {}).get("agents_to_return", []),
                       note="Agents returned to work immediately after finding processing, not waiting for other agents")
            
            # CRITICAL: Check if agents have pending tasks after cycle completes
            # This ensures agents continue working if supervisor assigned new tasks
            # IMPORTANT: Check even if agents_active is False - supervisor might have created new tasks before stopping
            if agent_file_service:
                logger.info(
                    "SUPERVISOR: Checking for pending tasks after cycle",
                    iteration_count=iteration_count
                )
                try:
                    # Reload agents list in case supervisor created new agents
                    agent_files = await agent_file_service.file_manager.list_files("agents/agent_*.md")
                    all_agent_ids = []
                    for file_path in agent_files:
                        agent_id = file_path.replace("agents/", "").replace(".md", "")
                        if agent_id.startswith("agent_") and agent_id != "supervisor":
                            all_agent_ids.append(agent_id)
                    
                    # Check if any agents have pending tasks
                    agents_with_pending_tasks = []
                    total_pending = 0
                    for agent_id in all_agent_ids:
                        agent_file = await agent_file_service.read_agent_file(agent_id)
                        todos = agent_file.get("todos", [])
                        pending_tasks = [t for t in todos if t.status == "pending"]
                        in_progress_tasks = [t for t in todos if t.status == "in_progress"]
                        if pending_tasks or in_progress_tasks:
                            agents_with_pending_tasks.append(agent_id)
                            total_pending += len(pending_tasks) + len(in_progress_tasks)
                    
                    if agents_with_pending_tasks:
                        logger.info(f"SUPERVISOR: After cycle {iteration_count}: {len(agents_with_pending_tasks)} agents have {total_pending} pending/in_progress tasks, continuing to next cycle",
                                   agents_with_tasks=agents_with_pending_tasks,
                                   total_pending_tasks=total_pending,
                                   note="Agents will continue working in next cycle - supervisor may have assigned new tasks")
                        # CRITICAL: Even if supervisor decided to stop, if there are pending tasks, we must continue
                        # Supervisor might have created new tasks before deciding to stop
                        agents_active = True
                        logger.info("SUPERVISOR: Reactivating agents because pending tasks found", 
                                   pending_tasks=total_pending,
                                   agents=agents_with_pending_tasks,
                                   note="Supervisor may have created new tasks before stopping - agents must complete them")
                        # Continue to next iteration - agents will pick up their pending tasks
                        # The while loop will continue because agents_active is now True
                    else:
                        logger.info(f"SUPERVISOR: After cycle {iteration_count}: no agents have pending tasks",
                                   iteration_count=iteration_count)
                        # Check if we should stop
                        if no_tasks_count == len(agent_tasks):
                            logger.info("All agents have no tasks, stopping agent execution")
                            if stream:
                                stream.emit_status("✅ All agents completed their tasks", step="agents")
                            
                            # CRITICAL: When all tasks are done, finalize report
                            logger.info("MANDATORY: All tasks completed - finalizing report")
                            if stream:
                                stream.emit_status("👔 Supervisor finalizing report...", step="supervisor")
                            
                            # Обработать оставшиеся файндинги из очереди перед финализацией
                            try:
                                # CRITICAL: Process all findings from queue using get_finding method
                                # This ensures all findings are processed and written to chapters
                                logger.info("MANDATORY: Processing all remaining findings from queue before finalization",
                                          queue_size=supervisor_queue.size() if supervisor_queue else 0,
                                          note="All findings MUST be processed and written to draft_report as chapters before finalization")
                                
                                # Обработать все файндинги через supervisor chain
                                # CRITICAL: Process ALL findings in queue - each one must become a chapter
                                processed_findings_count = 0
                                max_findings_to_process = 100  # Safety limit
                                while supervisor_queue and supervisor_queue.size() > 0 and processed_findings_count < max_findings_to_process:
                                    logger.info("Processing finding from queue during finalization",
                                              queue_size=supervisor_queue.size(),
                                              processed_count=processed_findings_count,
                                              note="Each finding will be validated and written to draft_report as a chapter")
                                    
                                    decision = await run_supervisor_chain(
                                        state=state_dict,
                                        llm=llm,
                                        stream=stream,
                                        supervisor_queue=supervisor_queue
                                    )
                                    
                                    processed_findings_count += 1
                                    
                                    # Check if finding was processed
                                    processed_agent_id = decision.get("processed_agent_id")
                                    chapter_written = decision.get("chapter_written", False)
                                    
                                    logger.info("Finding processed during finalization",
                                              processed_agent_id=processed_agent_id,
                                              chapter_written=chapter_written,
                                              action=decision.get("action", "unknown"),
                                              note="Finding was processed. If chapter_written=True, it was added to draft_report. If False, finding was rejected or needs rework.")
                                    
                                    # Обновить task_addition_count
                                    if "task_management" in decision and decision["task_management"]:
                                        task_management = decision["task_management"]
                                        if isinstance(task_management, dict):
                                            task_addition_count = task_management.get("task_addition_count", task_addition_count)
                                        else:
                                            task_addition_count = getattr(task_management, "task_addition_count", task_addition_count)
                                        state_dict["task_addition_count"] = task_addition_count
                                
                                if processed_findings_count > 0:
                                    logger.info("All findings from queue processed during finalization",
                                              processed_count=processed_findings_count,
                                              queue_size_remaining=supervisor_queue.size() if supervisor_queue else 0,
                                              note="All findings were processed. Each valid finding should now have a chapter in draft_report.")
                                elif supervisor_queue and supervisor_queue.size() == 0:
                                    logger.info("No findings in queue to process during finalization",
                                              note="Queue is empty - all findings were already processed during research")
                                
                                # Force should_continue to False to trigger report generation
                                state_dict["should_continue"] = False
                                state_dict["replanning_needed"] = False
                                
                                # CRITICAL: Update status after supervisor finalization
                                if stream:
                                    stream.emit_status("✅ Supervisor finalized report - generating final result...", step="supervisor")
                                
                                logger.info("Supervisor finalization completed", 
                                          note="Research will proceed to report generation")
                            except Exception as e:
                                logger.error("Failed to finalize supervisor", error=str(e), exc_info=True)
                                # Even if supervisor fails, set should_continue to False to proceed to report
                                state_dict["should_continue"] = False
                                state_dict["replanning_needed"] = False
                            
                            agents_active = False
                except Exception as e:
                    logger.error("Error checking agent tasks after cycle", error=str(e), exc_info=True)
                    # Continue anyway - don't stop on error
            
            # CRITICAL: Findings are NOT automatically added to draft_report
            # Findings are stored separately and supervisor receives them one-by-one via supervisor_queue
            # Supervisor adds findings to draft_report as structured chapters (one chapter = one finding)
            # This ensures draft_report is a clean research draft, not a dump of raw findings
            if cycle_findings and len(cycle_findings) > 0:
                logger.info("Cycle findings collected - supervisor will process them individually via queue",
                           cycle_findings_count=len(cycle_findings),
                           note="Findings stored separately, supervisor will add them to draft_report as chapters")

            # If all agents report no tasks, stop and force supervisor finalization
            if no_tasks_count == len(agent_tasks):
                logger.info("All agents report no tasks, stopping agent execution", 
                           no_tasks_count=no_tasks_count, total_agents=len(agent_tasks))
                if stream:
                    stream.emit_status("✅ All agents completed their tasks", step="agents")
                
                # CRITICAL: When all tasks are done, finalize report
                logger.info("MANDATORY: All tasks completed - finalizing report")
                if stream:
                    stream.emit_status("👔 Supervisor finalizing report...", step="supervisor")
                
                # Обработать оставшиеся файндинги из очереди перед финализацией
                try:
                    # CRITICAL: Process all findings from queue using get_finding method
                    # This ensures all findings are processed and written to chapters
                    logger.info("MANDATORY: Processing all remaining findings from queue before finalization",
                              queue_size=supervisor_queue.size() if supervisor_queue else 0,
                              note="All findings MUST be processed and written to draft_report as chapters before finalization")
                    
                    # Обработать все файндинги через supervisor chain
                    # CRITICAL: Process ALL findings in queue - each one must become a chapter
                    processed_findings_count = 0
                    max_findings_to_process = 100  # Safety limit
                    while supervisor_queue and supervisor_queue.size() > 0 and processed_findings_count < max_findings_to_process:
                        logger.info("Processing finding from queue during finalization",
                                  queue_size=supervisor_queue.size(),
                                  processed_count=processed_findings_count,
                                  note="Each finding will be validated and written to draft_report as a chapter")
                        
                        decision = await run_supervisor_chain(
                            state=state_dict,
                            llm=llm,
                            stream=stream,
                            supervisor_queue=supervisor_queue
                        )
                        
                        processed_findings_count += 1
                        
                        # Check if finding was processed
                        processed_agent_id = decision.get("processed_agent_id")
                        chapter_written = decision.get("chapter_written", False)
                        
                        logger.info("Finding processed during finalization",
                                  processed_agent_id=processed_agent_id,
                                  chapter_written=chapter_written,
                                  action=decision.get("action", "unknown"),
                                  note="Finding was processed. If chapter_written=True, it was added to draft_report. If False, finding was rejected or needs rework.")
                        
                        # Обновить task_addition_count
                        if "task_management" in decision and decision["task_management"]:
                            task_management = decision["task_management"]
                            if isinstance(task_management, dict):
                                task_addition_count = task_management.get("task_addition_count", task_addition_count)
                            else:
                                task_addition_count = getattr(task_management, "task_addition_count", task_addition_count)
                            state_dict["task_addition_count"] = task_addition_count
                    
                    if processed_findings_count > 0:
                        logger.info("All findings from queue processed during finalization",
                                  processed_count=processed_findings_count,
                                  queue_size_remaining=supervisor_queue.size() if supervisor_queue else 0,
                                  note="All findings were processed. Each valid finding should now have a chapter in draft_report.")
                    elif supervisor_queue and supervisor_queue.size() == 0:
                        logger.info("No findings in queue to process during finalization",
                                  note="Queue is empty - all findings were already processed during research")
                    
                    # Force should_continue to False to trigger report generation
                    state_dict["should_continue"] = False
                    state_dict["replanning_needed"] = False
                    
                    # CRITICAL: Update status after supervisor finalization
                    if stream:
                        stream.emit_status("✅ Supervisor finalized report - generating final result...", step="supervisor")
                    
                    logger.info("Supervisor finalization completed", 
                              total_findings=len(state_dict.get("findings", state_dict.get("agent_findings", []))),
                              note="Research will proceed to report generation")
                except Exception as e:
                    logger.error("Failed to finalize supervisor", error=str(e), exc_info=True)
                    # Even if supervisor fails, set should_continue to False to proceed to report
                    state_dict["should_continue"] = False
                    state_dict["replanning_needed"] = False
                
                agents_active = False
                # CRITICAL: After finalization, we should NOT process queue again - supervisor already finalized
                # Just break and return state with should_continue=False to trigger report generation
                logger.info("All tasks completed and supervisor finalized - breaking to trigger report generation",
                           note="should_continue=False will route to compress_findings -> generate_report")
                break
            
            # After cycle completes, check if there are any remaining items in supervisor queue
            # (Most items should have been processed during the cycle as agents completed)
            queue_size = supervisor_queue.size()
            logger.info(f"After cycle {iteration_count}: supervisor queue size = {queue_size}, cycle findings = {len(cycle_findings)}")
            
            # Process any remaining items in queue (should be rare, as we process during cycle)
            # CRITICAL: Only process queue if agents are still active (not finalized)
            if queue_size > 0 and agents_active:
                logger.info(f"Processing {supervisor_queue.size()} remaining findings in supervisor queue")
                
                if stream:
                    stream.emit_status(f"👔 Supervisor processing {supervisor_queue.size()} remaining findings", step="supervisor")
                
                # Обработать все оставшиеся файндинги из очереди
                try:
                    while supervisor_queue and supervisor_queue.size() > 0:
                        decision = await run_supervisor_chain(
                            state=state_dict,
                            llm=llm,
                            stream=stream,
                            supervisor_queue=supervisor_queue
                        )
                        
                        # Обновить task_addition_count
                        if "task_management" in decision:
                            task_addition_count = decision["task_management"].get("task_addition_count", task_addition_count)
                            state_dict["task_addition_count"] = task_addition_count
                        
                        # Update state with supervisor decision
                        state_dict["should_continue"] = decision.get("should_continue", False)
                        state_dict["replanning_needed"] = decision.get("replanning_needed", False)
                        
                        # Если supervisor решил остановиться, выйти из цикла
                        if not decision.get("should_continue", False):
                            logger.info("Supervisor decided to stop", 
                                       decision_reasoning=decision.get("reasoning", "")[:200])
                            agents_active = False
                            break
                    
                    # CRITICAL: Update status after supervisor review completes
                    if stream:
                        if not state_dict.get("should_continue", False):
                            stream.emit_status("✅ Supervisor finalized report - generating final result...", step="supervisor")
                        else:
                            # Check if there are pending tasks
                            if agent_file_service:
                                try:
                                    agent_files = await agent_file_service.file_manager.list_files("agents/agent_*.md")
                                    all_agent_ids = []
                                    for file_path in agent_files:
                                        agent_id = file_path.replace("agents/", "").replace(".md", "")
                                        if agent_id.startswith("agent_") and agent_id != "supervisor":
                                            all_agent_ids.append(agent_id)
                                    
                                    total_pending = 0
                                    for agent_id in all_agent_ids:
                                        agent_file = await agent_file_service.read_agent_file(agent_id)
                                        todos = agent_file.get("todos", [])
                                        pending_tasks = [t for t in todos if t.status == "pending"]
                                        in_progress_tasks = [t for t in todos if t.status == "in_progress"]
                                        total_pending += len(pending_tasks) + len(in_progress_tasks)
                                    
                                    if total_pending > 0:
                                        stream.emit_status(f"🚀 Agents continuing work ({total_pending} tasks remaining)", step="agents")
                                    else:
                                        stream.emit_status("🚀 Research continuing...", step="agents")
                                except Exception as e:
                                    logger.warning("Failed to check pending tasks for status update", error=str(e))
                                    stream.emit_status("🚀 Research continuing...", step="agents")
                    
                    logger.info("Supervisor queue processed", 
                               decision=state_dict.get("should_continue"),
                               note="Agents will continue working in next cycle if they have pending tasks")
                    
                except Exception as e:
                    logger.error("Supervisor processing failed", error=str(e), exc_info=True)
            
            # CRITICAL: DO NOT automatically add findings to draft_report - supervisor should add them as chapters
            # Automatic addition creates duplicate sections ("New Findings") and messes up the structure
            # Supervisor uses write_draft_report to add findings as proper chapters
            # Only finalize draft_report with ALL findings if supervisor limit reached (see below)
            # REMOVED: Automatic findings addition - it was creating duplicate sections and mess

        # CRITICAL: Check if max_iterations reached - if so, MUST stop and force finalization
        if iteration_count >= max_iterations:
            logger.warning(f"MANDATORY: Max iterations reached ({iteration_count}/{max_iterations}) - forcing stop and finalization",
                          note="Research will proceed to report generation even if tasks incomplete")
            agents_active = False
            # Force should_continue to False to trigger report generation
            state_dict["should_continue"] = False
            state_dict["replanning_needed"] = False
            if stream:
                stream.emit_status(f"⚠️ Max iterations reached ({iteration_count}/{max_iterations}) - finalizing report", step="agents")
        
        # Add findings to agent_findings (using reducer)
        # Update iteration in state
        new_iteration = current_iteration + iteration_count
        logger.info(f"Agent execution completed", cycles=iteration_count, total_iteration=new_iteration, max_iterations=max_iterations, task_addition_count=task_addition_count, max_reached=(iteration_count >= max_iterations))
        
        # CRITICAL: Supervisor continues to be called even after TODO limit is reached
        # Supervisor can still process findings and write chapters to draft_report
        # Finalization happens only when all tasks are completed, NOT when limit is reached
        agent_memory_service = stream.app_state.get("agent_memory_service") if stream else None
        agent_file_service = stream.app_state.get("agent_file_service") if stream else None
        
        # NOTE: We do NOT finalize draft_report here when limit is reached
        # Supervisor will continue to be called for findings processing and will write chapters
        # Finalization happens in generate_final_report_enhanced_node when all tasks are done

        # Проверка завершения: все задачи done + все главы записаны
        final_should_continue = state_dict.get("should_continue", True)
        
        # Проверить, есть ли еще задачи у агентов
        agents_still_working = False
        if agent_file_service:
            try:
                agent_files = await agent_file_service.file_manager.list_files("agents/agent_*.md")
                all_agent_ids = []
                for file_path in agent_files:
                    agent_id = file_path.replace("agents/", "").replace(".md", "")
                    if agent_id.startswith("agent_") and agent_id != "supervisor":
                        all_agent_ids.append(agent_id)
                
                for agent_id in all_agent_ids:
                    agent_file = await agent_file_service.read_agent_file(agent_id)
                    todos = agent_file.get("todos", [])
                    pending_tasks = [t for t in todos if t.status == "pending"]
                    in_progress_tasks = [t for t in todos if t.status == "in_progress"]
                    if pending_tasks or in_progress_tasks:
                        agents_still_working = True
                        break
            except Exception as e:
                logger.warning("Could not verify agent tasks status", error=str(e))
                agents_still_working = True  # Если не можем проверить, предполагаем что работают
        
        # Set flag in state so graph.py can check it
        state_dict["_agents_still_working"] = agents_still_working
        
        # Safety check: Max iterations reached
        if iteration_count >= max_iterations:
            if agents_still_working:
                logger.warning(f"Max iterations reached ({iteration_count}/{max_iterations}) but agents still working - NOT forcing completion",
                             note="Research will continue until all agents finish")
                final_should_continue = True
            else:
                logger.warning(f"MANDATORY: Max iterations reached ({iteration_count}/{max_iterations}) - forcing should_continue=False")
                final_should_continue = False
        
        # Проверка: все задачи завершены
        if not agents_still_working:
            logger.info("All agents have no tasks - research complete")
            final_should_continue = False
        
        logger.info("Final should_continue decision",
                  should_continue=final_should_continue,
                  iteration_count=iteration_count,
                  max_iterations=max_iterations,
                  findings_count=len(all_findings),
                  task_addition_count=task_addition_count,
                  note="If False, research will proceed to report generation")
        
        return {
            "agent_findings": all_findings,
            "findings": all_findings,
            "findings_count": len(all_findings),
            "iteration": new_iteration,
            "task_addition_count": task_addition_count,
            "should_continue": final_should_continue,
            "replanning_needed": False
        }


# Legacy function wrapper for backward compatibility
async def execute_agents_enhanced_node(state: ResearchState) -> Dict:
    """Legacy wrapper for ExecuteAgentsNode.

    This function maintains backward compatibility with existing code
    that imports execute_agents_enhanced_node directly.

    TODO: Update imports to use ExecuteAgentsNode class directly,
    then remove this wrapper.
    """
    from src.workflow.research.nodes import runtime_deps_context

    runtime_deps = runtime_deps_context.get()
    if not runtime_deps:
        logger.warning("Runtime dependencies not found in context")
        return {"findings": []}

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
    node = ExecuteAgentsNode(deps)
    return await node.execute(state)

"""Execute agents node for running research agents."""

import asyncio
import structlog
import re
from typing import Dict, Any
from datetime import datetime

from src.workflow.research.state import ResearchState
from src.workflow.research.nodes.base import ResearchNode
from src.workflow.research.nodes.utils import _restore_runtime_deps
from src.workflow.research.supervisor_queue import SupervisorQueue
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

        # Create supervisor queue
        supervisor_queue = SupervisorQueue()

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
                    agents_to_run = discovered_agents if discovered_agents else list(agent_characteristics.keys())
                except Exception as e:
                    logger.warning(f"Failed to discover agents from files, falling back to agent_characteristics", error=str(e))
                    agents_to_run = list(agent_characteristics.keys())
            else:
                # No file service, use agent_characteristics
                agents_to_run = list(agent_characteristics.keys())
            
            # CRITICAL: Agents wait for THEIR OWN finding to be processed, but work independently otherwise
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
                            has_finding_in_queue = supervisor_queue.has_finding_from_agent(agent_id)
                            
                            if has_finding_in_queue:
                                # Agent is waiting for supervisor to process its finding - don't run it
                                agents_waiting_for_review.append(agent_id)
                                logger.info(f"SUPERVISOR: Agent {agent_id} has finding in queue - waiting for supervisor review",
                                           agent_id=agent_id,
                                           pending_tasks=len(pending),
                                           in_progress_tasks=len(in_progress))
                            else:
                                # Agent has tasks and no finding in queue - can run
                                agents_to_run_filtered.append(agent_id)
                                logger.info(f"SUPERVISOR: Agent {agent_id} can run - has tasks and no finding in queue",
                                           agent_id=agent_id,
                                           pending_tasks=len(pending),
                                           in_progress_tasks=len(in_progress))
                except Exception as e:
                    logger.warning("SUPERVISOR: Failed to check agent tasks and queue status", error=str(e))
                    # Fallback: run all agents with tasks if check fails
                    agents_to_run_filtered = agents_with_tasks
            else:
                # No file service or queue - run all discovered agents (fallback)
                agents_to_run_filtered = agents_to_run
            
            agents_to_run = agents_to_run_filtered
            
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

            agent_tasks = []
            # CRITICAL: Launch ALL agents simultaneously, not sequentially
            # Create all tasks first, then they all run in parallel
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
            
            # Create a mapping of tasks to agent_ids for result processing
            # CRITICAL: Use a list to track which agents have been processed
            tasks_by_agent = {agent_id: task for agent_id, task in agent_tasks}
            processed_agents = set()  # Track which agents we've already processed
            
            # Process agents as they complete using as_completed
            # This processes each agent immediately when it finishes
            completed_count = 0
            pending_tasks_list = [task for _, task in agent_tasks]
            
            # CRITICAL: Track all agent tasks to ensure we process all of them
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
                                logger.info(f"Agent {agent_id} has no tasks")
                            else:
                                cycle_findings.append(result)
                                all_findings.append(result)
                                logger.info(f"Agent {agent_id} completed task", 
                                          task=result.get("topic", "unknown"),
                                          sources=len(result.get("sources", [])),
                                          completed_agents=f"{completed_count}/{len(agent_tasks)}",
                                          note="Result queued for supervisor, supervisor will review immediately")
                                
                                # CRITICAL: Process supervisor chain immediately when agent completes
                                # Don't wait for all agents - supervisor processes EACH agent's finding as soon as they complete
                                # This ensures supervisor updates draft_report and manages tasks in real-time
                                # CRITICAL: Other agents continue working in parallel - supervisor processing doesn't block them
                                
                                logger.info(
                                    "SUPERVISOR: Agent completed task, processing finding immediately",
                                    agent_id=agent_id,
                                    finding_topic=result.get("topic", "unknown"),
                                    queue_size_before=supervisor_queue.size() if supervisor_queue else 0
                                )
                                
                                try:
                                    if stream:
                                        stream.emit_status(f"👔 Supervisor processing finding from {agent_id}", step="supervisor")
                                    
                                    # CRITICAL: Ensure deep_search_result is in state before calling supervisor
                                    deep_search_result_in_state = state_dict.get("deep_search_result", "")
                                    if not deep_search_result_in_state:
                                        logger.warning("deep_search_result missing from state before supervisor call",
                                                     state_keys=list(state_dict.keys())[:20],
                                                     note="Supervisor may not have access to initial deep search context")
                                    else:
                                        logger.debug("deep_search_result available in state for supervisor",
                                                    has_deep_search=bool(deep_search_result_in_state),
                                                    deep_search_type=type(deep_search_result_in_state).__name__)
                                    
                                    # CRITICAL: Add current finding to state so supervisor can see it
                                    if result:
                                        # Use only "findings" field (not "agent_findings" - avoid duplication)
                                        existing_findings = state_dict.get("findings", [])
                                        # Check if this finding is already in state (avoid duplicates)
                                        finding_already_in_state = any(
                                            f.get("topic") == result.get("topic") and f.get("agent_id") == result.get("agent_id")
                                            for f in existing_findings
                                        )
                                        if not finding_already_in_state:
                                            existing_findings.append(result)
                                            state_dict["findings"] = existing_findings
                                            state_dict["agent_findings"] = existing_findings
                                            logger.info(f"Added finding to state for supervisor",
                                                       finding_topic=result.get("topic", "unknown"),
                                                       total_findings_in_state=len(existing_findings),
                                                       note="Supervisor will now see this finding in state")
                                    
                                    # Вызвать supervisor chain вместо ReAct агента
                                    decision = await run_supervisor_chain(
                                        state=state_dict,
                                        llm=llm,
                                        stream=stream,
                                        supervisor_queue=supervisor_queue
                                    )
                                    
                                    # Обновить task_addition_count из результата
                                    if "task_management" in decision:
                                        task_addition_count = decision["task_management"].get("task_addition_count", task_addition_count)
                                        state_dict["task_addition_count"] = task_addition_count
                                    
                                    state_dict["should_continue"] = decision.get("should_continue", False)
                                    state_dict["replanning_needed"] = decision.get("replanning_needed", False)
                                    
                                    # CRITICAL: Store supervisor decision with agents_to_return for logging only
                                    # This does NOT affect running agents - they continue working independently
                                    # agents_to_return is used only to track which agent's finding was processed
                                    # Agents are selected for next cycle based on: has_tasks AND no_finding_in_queue
                                    state_dict["supervisor_decision"] = decision
                                    
                                    agents_to_return = decision.get("agents_to_return", [])
                                    logger.info(
                                        "SUPERVISOR: Supervisor decision stored (for logging only)",
                                        finding_agent_id=agent_id,
                                        agents_to_return=agents_to_return,
                                        should_continue=decision.get("should_continue", False),
                                        queue_size_remaining=supervisor_queue.size() if supervisor_queue else 0,
                                        note="Processing this finding doesn't affect other running agents. agents_to_return is for logging only - agents are selected in next cycle based on has_tasks AND no_finding_in_queue"
                                    )
                                    
                                    # CRITICAL: After supervisor review, check if supervisor created new tasks
                                    # This is ONLY for setting should_continue flag - it does NOT restart agents
                                    # Running agents continue working independently - they are NOT affected by processing other agents' findings
                                    # Agents are restarted ONLY at the beginning of next cycle (in while loop)
                                    new_pending_tasks = 0  # Initialize before use
                                    if agent_file_service:
                                        try:
                                            # Check if any agents have new pending tasks after supervisor review
                                            # NOTE: This doesn't affect running agents - they continue working on their current tasks
                                            # Only pending tasks (not in_progress) can be changed, and this doesn't break current work
                                            agent_files = await agent_file_service.file_manager.list_files("agents/agent_*.md")
                                            all_agent_ids = []
                                            for file_path in agent_files:
                                                agent_id = file_path.replace("agents/", "").replace(".md", "")
                                                if agent_id.startswith("agent_") and agent_id != "supervisor":
                                                    all_agent_ids.append(agent_id)
                                            
                                            new_pending_tasks = 0
                                            agents_with_new_tasks = []
                                            for agent_id in all_agent_ids:
                                                agent_file = await agent_file_service.read_agent_file(agent_id)
                                                todos = agent_file.get("todos", [])
                                                pending_tasks = [t for t in todos if t.status == "pending"]
                                                in_progress_tasks = [t for t in todos if t.status == "in_progress"]
                                                if pending_tasks or in_progress_tasks:
                                                    agents_with_new_tasks.append(agent_id)
                                                    new_pending_tasks += len(pending_tasks) + len(in_progress_tasks)
                                            
                                            if new_pending_tasks > 0:
                                                logger.info(f"SUPERVISOR: After processing finding from {agent_id}: {len(agents_with_new_tasks)} agents have {new_pending_tasks} pending/in_progress tasks",
                                                           finding_agent_id=agent_id,
                                                           agents_with_tasks=agents_with_new_tasks,
                                                           total_pending=new_pending_tasks,
                                                           note="Processing this finding doesn't affect running agents. New/updated tasks will be picked up in next cycle. Running agents continue working on their current tasks.")
                                                # CRITICAL: If supervisor created new tasks, we MUST continue
                                                # Override supervisor's decision to stop if there are new tasks
                                                # This only sets should_continue flag - agents are NOT restarted here
                                                state_dict["should_continue"] = True
                                                logger.info("SUPERVISOR: Overriding supervisor's stop decision because new tasks were created",
                                                           note="This only sets should_continue flag. Agents are restarted only at beginning of next cycle.")
                                        except Exception as e:
                                            logger.error("SUPERVISOR: Error checking for new tasks after supervisor review", error=str(e), exc_info=True)
                                    
                                    # CRITICAL: Update status after supervisor review completes
                                    # If supervisor decided to continue or created new tasks, show that agents are working
                                    if stream:
                                        if state_dict.get("should_continue", False) or new_pending_tasks > 0:
                                            # Supervisor decided to continue or created new tasks - agents are working
                                            if new_pending_tasks > 0:
                                                stream.emit_status(f"🚀 Agents continuing work ({new_pending_tasks} tasks remaining)", step="agents")
                                            else:
                                                stream.emit_status("🚀 Agents continuing research...", step="agents")
                                        elif not decision.get("should_continue", False):
                                            # Supervisor decided to stop
                                            stream.emit_status("✅ Supervisor decided research is complete", step="supervisor")
                                    
                                    # CRITICAL: Even if supervisor says stop, check if there are pending tasks
                                    # Supervisor might have created new tasks before deciding to stop
                                    # We need to check pending tasks before actually stopping
                                    if not decision.get("should_continue", False) and not state_dict.get("should_continue", False):
                                        logger.info("Supervisor decided to stop", decision_reasoning=decision.get("reasoning", "")[:200])
                                        # Don't break immediately - check for pending tasks first
                                        # The check below will verify if there are pending tasks
                                        # If there are pending tasks, we'll continue despite supervisor's decision
                                except Exception as e:
                                    logger.error("Supervisor review failed", error=str(e), exc_info=True)
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
            
            logger.info(f"SUPERVISOR: Cycle {iteration_count} complete",
                       tasks_completed=len(cycle_findings),
                       agents_with_no_tasks=no_tasks_count,
                       agents_processed=f"{completed_count}/{len(agent_tasks)}",
                       processed_agents=list(processed_agents),
                       unprocessed_agents=unprocessed_agents if unprocessed_agents else None,
                       supervisor_decision_agents_to_return=state_dict.get("supervisor_decision", {}).get("agents_to_return", []),
                       note="Agents will be returned to work in next cycle via agents_to_return")
            
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
                                # Извлечь все файндинги из очереди
                                if supervisor_queue and supervisor_queue.size() > 0:
                                    findings_from_queue = []
                                    temp_events = []
                                    queue_size = supervisor_queue.size()
                                    for _ in range(queue_size):
                                        try:
                                            event = supervisor_queue.queue.get_nowait()
                                            temp_events.append(event)
                                            if event.result:
                                                findings_from_queue.append(event.result)
                                        except:
                                            break
                                    
                                    # Вернуть события обратно для обработки
                                    for event in temp_events:
                                        await supervisor_queue.queue.put(event)
                                    
                                    # Добавить файндинги в state
                                    if findings_from_queue:
                                        existing_findings = state_dict.get("findings", state_dict.get("agent_findings", []))
                                        for new_finding in findings_from_queue:
                                            finding_already_exists = any(
                                                f.get("topic") == new_finding.get("topic") and 
                                                f.get("agent_id") == new_finding.get("agent_id")
                                                for f in existing_findings
                                            )
                                            if not finding_already_exists:
                                                existing_findings.append(new_finding)
                                        
                                        state_dict["findings"] = existing_findings
                                        state_dict["agent_findings"] = existing_findings
                                        logger.info(f"Extracted {len(findings_from_queue)} findings from supervisor_queue before finalization",
                                                   total_findings=len(existing_findings))
                                
                                # Обработать оставшиеся файндинги через supervisor chain
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
                    # Извлечь все файндинги из очереди
                    if supervisor_queue and supervisor_queue.size() > 0:
                        findings_from_queue = []
                        temp_events = []
                        queue_size = supervisor_queue.size()
                        for _ in range(queue_size):
                            try:
                                event = supervisor_queue.queue.get_nowait()
                                temp_events.append(event)
                                if event.result:
                                    findings_from_queue.append(event.result)
                            except:
                                break
                        
                        # Вернуть события обратно для обработки
                        for event in temp_events:
                            await supervisor_queue.queue.put(event)
                        
                        # Добавить файндинги в state
                        if findings_from_queue:
                            existing_findings = state_dict.get("findings", state_dict.get("agent_findings", []))
                            for new_finding in findings_from_queue:
                                finding_already_exists = any(
                                    f.get("topic") == new_finding.get("topic") and 
                                    f.get("agent_id") == new_finding.get("agent_id")
                                    for f in existing_findings
                                )
                                if not finding_already_exists:
                                    existing_findings.append(new_finding)
                            
                            state_dict["findings"] = existing_findings
                            state_dict["agent_findings"] = existing_findings
                            logger.info(f"Extracted {len(findings_from_queue)} findings from supervisor_queue before finalization",
                                       total_findings=len(existing_findings))
                    
                    # Обработать оставшиеся файндинги через supervisor chain
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

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
from src.workflow.research.supervisor_agent import run_supervisor_agent

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
        
        # Track supervisor calls (not ReAct iterations, but actual supervisor invocations)
        supervisor_call_count = state_dict.get("supervisor_call_count", 0)
        # Get max_supervisor_calls from settings (centralized config)
        settings = state_dict.get("settings")
        if settings:
            max_supervisor_calls = settings.deep_research_max_supervisor_calls
        else:
            from src.config.settings import get_settings
            settings_obj = get_settings()
            max_supervisor_calls = settings_obj.deep_research_max_supervisor_calls
        
        # Run agents in continuous mode until all todos complete or max iterations
        agents_active = True
        iteration_count = 0
        
        # CRITICAL: Hard limit to prevent infinite loops
        # If max_iterations reached, MUST stop and generate report
        while agents_active and iteration_count < max_iterations:
            iteration_count += 1
            logger.info(f"Agent execution cycle {iteration_count}")
            
            if stream:
                stream.emit_status(f"🔄 Agent execution cycle {iteration_count}/{max_iterations} (Supervisor calls: {supervisor_call_count}/{max_supervisor_calls})", step="agents")
                logger.info(f"Emitting progress: cycle {iteration_count}/{max_iterations}, supervisor calls {supervisor_call_count}/{max_supervisor_calls}")
            
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
                                
                                # CRITICAL: Process supervisor review immediately when agent completes
                                # Don't wait for all agents - supervisor should review EACH agent's work as soon as they complete
                                # This ensures supervisor updates draft_report and manages tasks in real-time
                                # CRITICAL: Supervisor is ALWAYS called for findings processing and draft_report writing
                                # Limit applies ONLY to TODO operations (create_agent_todo, update_agent_todo), NOT to findings processing
                                # CRITICAL: Other agents continue working in parallel - supervisor review doesn't block them
                                
                                # Track call count for TODO operations limit, but ALWAYS call supervisor for findings
                                is_todo_operations_available = supervisor_call_count < max_supervisor_calls
                                
                                if is_todo_operations_available:
                                    # Increment counter only for TODO operations tracking
                                    supervisor_call_count += 1
                                    state_dict["supervisor_call_count"] = supervisor_call_count
                                    logger.info(f"Agent {agent_id} completed task - calling supervisor for review (call {supervisor_call_count}/{max_supervisor_calls}, TODO operations available)",
                                              note="Other agents continue working in parallel during supervisor review")
                                else:
                                    # Don't increment counter, but STILL call supervisor for findings processing
                                    logger.info(f"Agent {agent_id} completed task - calling supervisor for findings processing (TODO limit reached: {supervisor_call_count}/{max_supervisor_calls})",
                                              note="Supervisor will process findings and write to draft_report, but TODO operations are disabled")
                                
                                try:
                                    if settings:
                                        supervisor_max_iterations = settings.deep_research_supervisor_max_iterations
                                    else:
                                        from src.config.settings import get_settings
                                        settings_obj = get_settings()
                                        supervisor_max_iterations = settings_obj.deep_research_supervisor_max_iterations
                                    
                                    if stream:
                                        if is_todo_operations_available:
                                            stream.emit_status(f"👔 Supervisor reviewing findings from {agent_id} (call {supervisor_call_count}/{max_supervisor_calls})", step="supervisor")
                                        else:
                                            stream.emit_status(f"👔 Supervisor processing findings from {agent_id} (TODO limit reached, writing to draft_report)", step="supervisor")
                                    
                                    # CRITICAL: Supervisor review happens while other agents continue working
                                    # This is non-blocking - agents run in parallel via asyncio.create_task
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
                                    # Supervisor reads findings from state, not from cycle_findings
                                    if result:
                                        existing_findings = state_dict.get("findings", state_dict.get("agent_findings", []))
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
                                    
                                    decision = await run_supervisor_agent(
                                        state=state_dict,
                                        llm=llm,
                                        stream=stream,
                                        supervisor_queue=supervisor_queue,
                                        max_iterations=supervisor_max_iterations
                                    )
                                    
                                    state_dict["should_continue"] = decision.get("should_continue", False)
                                    state_dict["replanning_needed"] = decision.get("replanning_needed", False)
                                    
                                    # CRITICAL: After supervisor review, check if supervisor created new tasks
                                    # Even if supervisor says stop, we must check for new tasks before stopping
                                    # Supervisor might have created new tasks before deciding to stop
                                    new_pending_tasks = 0  # Initialize before use
                                    if agent_file_service:
                                        try:
                                            # Check if any agents have new pending tasks after supervisor review
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
                                                logger.info(f"After supervisor review: {len(agents_with_new_tasks)} agents have {new_pending_tasks} pending/in_progress tasks",
                                                           agents_with_tasks=agents_with_new_tasks,
                                                           total_pending=new_pending_tasks,
                                                           note="Supervisor created new tasks - agents will continue working in next cycle")
                                                # CRITICAL: If supervisor created new tasks, we MUST continue
                                                # Override supervisor's decision to stop if there are new tasks
                                                state_dict["should_continue"] = True
                                                logger.info("Overriding supervisor's stop decision because new tasks were created",
                                                           note="Agents must complete all tasks before finalization")
                                        except Exception as e:
                                            logger.error("Error checking for new tasks after supervisor review", error=str(e), exc_info=True)
                                    
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
            
            logger.info(f"Cycle {iteration_count} complete: {len(cycle_findings)} tasks completed, {no_tasks_count} agents with no tasks, {completed_count}/{len(agent_tasks)} agents processed",
                       processed_agents=list(processed_agents),
                       unprocessed_agents=unprocessed_agents if unprocessed_agents else None)
            
            # CRITICAL: Check if agents have pending tasks after cycle completes
            # This ensures agents continue working if supervisor assigned new tasks
            # IMPORTANT: Check even if agents_active is False - supervisor might have created new tasks before stopping
            if agent_file_service:
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
                        logger.info(f"After cycle {iteration_count}: {len(agents_with_pending_tasks)} agents have {total_pending} pending/in_progress tasks, continuing to next cycle",
                                   agents_with_tasks=agents_with_pending_tasks,
                                   total_pending_tasks=total_pending,
                                   note="Agents will continue working in next cycle - supervisor may have assigned new tasks")
                        # CRITICAL: Even if supervisor decided to stop, if there are pending tasks, we must continue
                        # Supervisor might have created new tasks before deciding to stop
                        agents_active = True
                        logger.info("Reactivating agents because pending tasks found", 
                                   pending_tasks=total_pending,
                                   agents=agents_with_pending_tasks,
                                   note="Supervisor may have created new tasks before stopping - agents must complete them")
                        # Continue to next iteration - agents will pick up their pending tasks
                        # The while loop will continue because agents_active is now True
                    else:
                        logger.info(f"After cycle {iteration_count}: no agents have pending tasks")
                        # Check if we should stop
                        if no_tasks_count == len(agent_tasks):
                            logger.info("All agents have no tasks, stopping agent execution")
                            if stream:
                                stream.emit_status("✅ All agents completed their tasks", step="agents")
                            
                            # CRITICAL: When all tasks are done, FORCE supervisor to finalize report
                            # This ensures final report is generated even if supervisor didn't call make_final_decision
                            # MANDATORY: Call supervisor EVEN IF limit reached - this is finalization call
                            logger.info("MANDATORY: All tasks completed - forcing supervisor to finalize report (bypassing call limit if needed)")
                            if stream:
                                stream.emit_status("👔 Supervisor finalizing report...", step="supervisor")
                            
                            # CRITICAL: Always call supervisor for finalization, even if limit reached
                            # This is a special finalization call that bypasses the normal limit
                            try:
                                # Increment counter but don't check limit - this is mandatory finalization
                                supervisor_call_count += 1
                                state_dict["supervisor_call_count"] = supervisor_call_count
                                
                                if settings:
                                    supervisor_max_iterations = settings.deep_research_supervisor_max_iterations
                                else:
                                    from src.config.settings import get_settings
                                    settings_obj = get_settings()
                                    supervisor_max_iterations = settings_obj.deep_research_supervisor_max_iterations
                                
                                logger.info("Calling supervisor for MANDATORY finalization (bypassing call limit)",
                                           call_count=supervisor_call_count,
                                           max_calls=max_supervisor_calls,
                                           note="This is a special finalization call when all tasks are done")
                                
                                decision = await run_supervisor_agent(
                                    state=state_dict,
                                    llm=llm,
                                    stream=stream,
                                    supervisor_queue=supervisor_queue,
                                    max_iterations=supervisor_max_iterations
                                )
                                
                                # Force should_continue to False to trigger report generation
                                state_dict["should_continue"] = False
                                state_dict["replanning_needed"] = False
                                
                                # CRITICAL: Update status after supervisor finalization
                                if stream:
                                    stream.emit_status("✅ Supervisor finalized report - generating final result...", step="supervisor")
                                
                                logger.info("Forced supervisor finalization completed", 
                                          decision=decision.get("should_continue"),
                                          note="Research will proceed to report generation")
                            except Exception as e:
                                logger.error("Failed to force supervisor finalization", error=str(e), exc_info=True)
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
                
                # CRITICAL: When all tasks are done, FORCE supervisor to finalize report
                # MANDATORY: Call supervisor EVEN IF limit reached - this is finalization call
                logger.info("MANDATORY: All tasks completed - forcing supervisor to finalize report (bypassing call limit if needed)")
                if stream:
                    stream.emit_status("👔 Supervisor finalizing report...", step="supervisor")
                
                # CRITICAL: Always call supervisor for finalization, even if limit reached
                # This is a special finalization call that bypasses the normal limit
                try:
                    # Increment counter but don't check limit - this is mandatory finalization
                    supervisor_call_count += 1
                    state_dict["supervisor_call_count"] = supervisor_call_count
                    
                    if settings:
                        supervisor_max_iterations = settings.deep_research_supervisor_max_iterations
                    else:
                        from src.config.settings import get_settings
                        settings_obj = get_settings()
                        supervisor_max_iterations = settings_obj.deep_research_supervisor_max_iterations
                    
                    logger.info("Calling supervisor for MANDATORY finalization (bypassing call limit)",
                               call_count=supervisor_call_count,
                               max_calls=max_supervisor_calls,
                               note="This is a special finalization call when all tasks are done")
                    
                    # CRITICAL: Extract ALL findings from supervisor_queue and add to state BEFORE finalization
                    # This ensures supervisor sees all findings when finalizing the report
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
                        
                        # Put events back in queue (they'll be processed properly by supervisor)
                        for event in temp_events:
                            await supervisor_queue.queue.put(event)
                        
                        # Add findings to state so supervisor can see them
                        if findings_from_queue:
                            existing_findings = state_dict.get("findings", state_dict.get("agent_findings", []))
                            # Combine existing findings with queue findings (avoid duplicates)
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
                                       total_findings=len(existing_findings),
                                       note="Supervisor will now see ALL findings when finalizing report")
                    
                    # Also extract findings from cycle_findings if available
                    if cycle_findings and len(cycle_findings) > 0:
                        existing_findings = state_dict.get("findings", state_dict.get("agent_findings", []))
                        for new_finding in cycle_findings:
                            finding_already_exists = any(
                                f.get("topic") == new_finding.get("topic") and 
                                f.get("agent_id") == new_finding.get("agent_id")
                                for f in existing_findings
                            )
                            if not finding_already_exists:
                                existing_findings.append(new_finding)
                        
                        state_dict["findings"] = existing_findings
                        state_dict["agent_findings"] = existing_findings
                        logger.info(f"Added {len(cycle_findings)} findings from cycle_findings to state before finalization",
                                   total_findings=len(existing_findings),
                                   note="Supervisor will now see ALL findings when finalizing report")
                    
                    decision = await run_supervisor_agent(
                        state=state_dict,
                        llm=llm,
                        stream=stream,
                        supervisor_queue=supervisor_queue,
                        max_iterations=supervisor_max_iterations
                    )
                    
                    # Force should_continue to False to trigger report generation
                    state_dict["should_continue"] = False
                    state_dict["replanning_needed"] = False
                    
                    # CRITICAL: Update status after supervisor finalization
                    if stream:
                        stream.emit_status("✅ Supervisor finalized report - generating final result...", step="supervisor")
                    
                    logger.info("Forced supervisor finalization completed", 
                              decision=decision.get("should_continue"),
                              total_findings=len(state_dict.get("findings", state_dict.get("agent_findings", []))),
                              note="Research will proceed to report generation")
                except Exception as e:
                    logger.error("Failed to force supervisor finalization", error=str(e), exc_info=True)
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
                # CRITICAL: Supervisor is ALWAYS called for findings processing and draft_report writing
                # Limit applies ONLY to TODO operations, NOT to findings processing
                is_todo_operations_available = supervisor_call_count < max_supervisor_calls
                
                if is_todo_operations_available:
                    # Increment counter only for TODO operations tracking
                    supervisor_call_count += 1
                    state_dict["supervisor_call_count"] = supervisor_call_count
                    logger.info(f"Processing {supervisor_queue.size()} agent completions in supervisor queue (call {supervisor_call_count}/{max_supervisor_calls}, TODO operations available)")
                else:
                    # Don't increment counter, but STILL call supervisor for findings processing
                    logger.info(f"Processing {supervisor_queue.size()} agent completions in supervisor queue (TODO limit reached: {supervisor_call_count}/{max_supervisor_calls})",
                              note="Supervisor will process findings and write to draft_report, but TODO operations are disabled")
                
                if stream:
                    if is_todo_operations_available:
                        stream.emit_status(f"👔 Supervisor reviewing findings (call {supervisor_call_count}/{max_supervisor_calls})", step="supervisor")
                    else:
                        stream.emit_status(f"👔 Supervisor processing findings (TODO limit reached, writing to draft_report)", step="supervisor")
                
                # CRITICAL: Call supervisor agent to review and process findings
                # Supervisor is ALWAYS called - limit only blocks TODO operations, NOT findings processing
                
                try:
                    
                    # Get max_iterations from settings (centralized config)
                    if settings:
                        supervisor_max_iterations = settings.deep_research_supervisor_max_iterations
                    else:
                        from src.config.settings import get_settings
                        settings_obj = get_settings()
                        supervisor_max_iterations = settings_obj.deep_research_supervisor_max_iterations
                    
                    # CRITICAL: Extract findings from supervisor_queue and add to state
                    # Supervisor needs findings in state to process them
                    if supervisor_queue and supervisor_queue.size() > 0:
                        findings_from_queue = []
                        # Get all pending findings from queue (peek without removing)
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
                        
                        # Put events back in queue (they'll be processed properly by supervisor)
                        for event in temp_events:
                            await supervisor_queue.queue.put(event)
                        
                        # Add findings to state so supervisor can see them
                        if findings_from_queue:
                            existing_findings = state_dict.get("findings", state_dict.get("agent_findings", []))
                            # Combine existing findings with queue findings (avoid duplicates)
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
                            logger.info(f"Extracted {len(findings_from_queue)} findings from supervisor_queue and added to state",
                                       total_findings=len(existing_findings),
                                       note="Supervisor will now see these findings in state")
                    
                    decision = await run_supervisor_agent(
                        state=state_dict,
                        llm=llm,
                        stream=stream,
                        supervisor_queue=supervisor_queue,  # Pass supervisor_queue
                        max_iterations=supervisor_max_iterations
                    )
                    
                    # Update state with supervisor decision
                    state_dict["should_continue"] = decision.get("should_continue", False)
                    state_dict["replanning_needed"] = decision.get("replanning_needed", False)
                    
                    # CRITICAL: Update status after supervisor review completes
                    if stream:
                        if not decision.get("should_continue", False):
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
                    
                    # If supervisor says stop, break the loop
                    if not decision.get("should_continue", False):
                        logger.info("Supervisor decided to stop, breaking agent execution loop", 
                                   decision_reasoning=decision.get("reasoning", "")[:200])
                        agents_active = False
                        break
                    
                    # Check if we've reached the limit after this call
                    if supervisor_call_count >= max_supervisor_calls:
                        logger.warning(f"Supervisor call limit reached after decision ({supervisor_call_count}/{max_supervisor_calls}), agents will complete tasks without supervisor")
                        # Don't break - let agents complete their tasks
                        # Clear queue and continue
                        while not supervisor_queue.queue.empty():
                            try:
                                supervisor_queue.queue.get_nowait()
                                supervisor_queue.queue.task_done()
                            except:
                                break
                        # Continue loop - agents will finish their tasks
                        continue  # Skip further supervisor calls but continue agent execution
                    
                    # Clear the queue after processing
                    while not supervisor_queue.queue.empty():
                        try:
                            supervisor_queue.queue.get_nowait()
                            supervisor_queue.queue.task_done()
                        except:
                            break
                            
                    logger.info("Supervisor queue processed", 
                               decision=decision.get("should_continue"),
                               note="Agents will continue working in next cycle if they have pending tasks")
                    
                    # CRITICAL: After supervisor review, agents should continue working if they have pending tasks
                    # The while loop will continue and agents will be launched again in next iteration
                    
                except Exception as e:
                    logger.error("Supervisor processing failed", error=str(e))
            
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
            # CRITICAL: Force supervisor finalization even if limit reached
            state_dict["_force_supervisor_finalization"] = True
            if stream:
                stream.emit_status(f"⚠️ Max iterations reached ({iteration_count}/{max_iterations}) - finalizing report", step="agents")
        
        # Add findings to agent_findings (using reducer)
        # Update iteration in state
        new_iteration = current_iteration + iteration_count
        logger.info(f"Agent execution completed", cycles=iteration_count, total_iteration=new_iteration, max_iterations=max_iterations, supervisor_calls=supervisor_call_count, max_reached=(iteration_count >= max_iterations))
        
        # CRITICAL: Supervisor continues to be called even after TODO limit is reached
        # Supervisor can still process findings and write chapters to draft_report
        # Finalization happens only when all tasks are completed, NOT when limit is reached
        agent_memory_service = stream.app_state.get("agent_memory_service") if stream else None
        agent_file_service = stream.app_state.get("agent_file_service") if stream else None
        
        # NOTE: We do NOT finalize draft_report here when limit is reached
        # Supervisor will continue to be called for findings processing and will write chapters
        # Finalization happens in generate_final_report_enhanced_node when all tasks are done

        # CRITICAL: Multiple safety checks to ensure research ALWAYS completes and generates result
        # 1. Check if max_iterations reached
        # 2. Check if supervisor call limit reached
        # 3. Check if all tasks done
        # ANY of these conditions MUST trigger report generation
        
        final_should_continue = state_dict.get("should_continue", True)
        
        # Safety check 1: Max iterations reached
        if iteration_count >= max_iterations:
            logger.warning(f"MANDATORY: Max iterations reached ({iteration_count}/{max_iterations}) - forcing should_continue=False")
            final_should_continue = False
        
        # Safety check 2: Supervisor call limit reached
        if supervisor_call_count >= max_supervisor_calls:
            logger.warning(f"MANDATORY: Supervisor call limit reached ({supervisor_call_count}/{max_supervisor_calls}) - forcing should_continue=False")
            final_should_continue = False
        
        # Safety check 3: All tasks done
        if not final_should_continue:
            logger.info("should_continue is False - research will proceed to report generation")
        else:
            # Check if all agents really have no tasks
            if agent_file_service:
                try:
                    agent_files = await agent_file_service.file_manager.list_files("agents/agent_*.md")
                    all_agent_ids = []
                    for file_path in agent_files:
                        agent_id = file_path.replace("agents/", "").replace(".md", "")
                        if agent_id.startswith("agent_") and agent_id != "supervisor":
                            all_agent_ids.append(agent_id)
                    
                    all_agents_have_no_tasks = True
                    for agent_id in all_agent_ids:
                        agent_file = await agent_file_service.read_agent_file(agent_id)
                        todos = agent_file.get("todos", [])
                        pending_tasks = [t for t in todos if t.status == "pending"]
                        in_progress_tasks = [t for t in todos if t.status == "in_progress"]
                        if pending_tasks or in_progress_tasks:
                            all_agents_have_no_tasks = False
                            break
                    
                    if all_agents_have_no_tasks:
                        logger.info("MANDATORY: All agents have no tasks - forcing should_continue=False to trigger report generation")
                        final_should_continue = False
                except Exception as e:
                    logger.warning("Could not verify agent tasks status", error=str(e))
        
        # CRITICAL: Final guarantee - if we have findings, we MUST generate report
        # Even if should_continue is True, if we have findings and limits reached, force completion
        if all_findings and len(all_findings) > 0:
            if iteration_count >= max_iterations or supervisor_call_count >= max_supervisor_calls:
                logger.warning(f"MANDATORY: Limits reached but findings exist - forcing completion to generate report",
                              findings_count=len(all_findings),
                              iteration_count=iteration_count,
                              max_iterations=max_iterations,
                              supervisor_calls=supervisor_call_count,
                              max_supervisor_calls=max_supervisor_calls)
                final_should_continue = False
        
        # CRITICAL: If no findings and limits reached, still generate report (even if empty)
        # This ensures user always gets a result, not infinite loop
        if not all_findings or len(all_findings) == 0:
            if iteration_count >= max_iterations or supervisor_call_count >= max_supervisor_calls:
                logger.warning(f"MANDATORY: Limits reached with no findings - forcing completion to generate report (may be empty)",
                              iteration_count=iteration_count,
                              max_iterations=max_iterations,
                              supervisor_calls=supervisor_call_count,
                              max_supervisor_calls=max_supervisor_calls)
                final_should_continue = False
        
        logger.info("Final should_continue decision",
                  should_continue=final_should_continue,
                  iteration_count=iteration_count,
                  max_iterations=max_iterations,
                  supervisor_calls=supervisor_call_count,
                  max_supervisor_calls=max_supervisor_calls,
                  findings_count=len(all_findings),
                  note="If False, research will proceed to report generation")
        
        return {
            "agent_findings": all_findings,
            "findings": all_findings,  # Keep for supervisor review
            "findings_count": len(all_findings),
            "iteration": new_iteration,
            "supervisor_call_count": supervisor_call_count,
            "should_continue": final_should_continue,  # CRITICAL: Ensure this is False when limits reached or tasks done
            "replanning_needed": False  # CRITICAL: Don't replan when limits reached or tasks done
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
        return {"findings": [], "agent_findings": []}

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

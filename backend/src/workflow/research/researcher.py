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
    research_memory_service = state.get("research_memory_service")
    
    # If not in state, try to get from runtime deps via contextvars
    if not agent_memory_service or not agent_file_service:
        from src.workflow.research.nodes import _get_runtime_deps
        runtime_deps = _get_runtime_deps()
        agent_memory_service = runtime_deps.get("agent_memory_service")
        agent_file_service = runtime_deps.get("agent_file_service")
        research_memory_service = runtime_deps.get("research_memory_service")
    
    # Last resort: try to get from stream.app_state
    if (not agent_memory_service or not agent_file_service) and stream:
        if hasattr(stream, "app_state"):
            app_state = stream.app_state
            if isinstance(app_state, dict):
                agent_memory_service = agent_memory_service or app_state.get("agent_memory_service") or app_state.get("_agent_memory_service")
                agent_file_service = agent_file_service or app_state.get("agent_file_service") or app_state.get("_agent_file_service")
                research_memory_service = research_memory_service or app_state.get("research_memory_service") or app_state.get("_research_memory_service")
            else:
                agent_memory_service = agent_memory_service or getattr(app_state, "agent_memory_service", None) or getattr(app_state, "_agent_memory_service", None)
                agent_file_service = agent_file_service or getattr(app_state, "agent_file_service", None) or getattr(app_state, "_agent_file_service", None)
                research_memory_service = research_memory_service or getattr(app_state, "research_memory_service", None) or getattr(app_state, "_research_memory_service", None)

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
    # CRITICAL: Do NOT truncate character - keep it full for proper agent behavior
    character = agent_file.get("character", "")
    preferences = agent_file.get("preferences", "")
    todos = agent_file.get("todos", [])
    
    # CRITICAL: Use vector search to find relevant notes and findings instead of just recent notes
    # Search for top 5 most relevant memories (notes + findings) based on task description
    notes_context = "No previous notes or findings."
    session_id = state.get("session_id")
    if research_memory_service and session_id:
        try:
            # Build search query from task description
            current_task = None
            in_progress_tasks = [t for t in todos if t.status == "in_progress"]
            if in_progress_tasks:
                current_task = in_progress_tasks[0]
            else:
                pending_tasks = [t for t in todos if t.status == "pending"]
                if pending_tasks:
                    current_task = pending_tasks[0]
            
            if current_task:
                # Use task title + objective + guidance as search query
                # CRITICAL: Limit query length for embedding generation performance (most embedding models have token limits)
                # Take first 2000 characters to ensure fast embedding generation while keeping key information
                task_title = current_task.title or ""
                task_objective = current_task.objective or ""
                task_note = current_task.note if hasattr(current_task, 'note') and current_task.note else ""
                
                # Build search query prioritizing title and objective (most important)
                search_query_parts = []
                if task_title:
                    search_query_parts.append(task_title)
                if task_objective:
                    search_query_parts.append(task_objective)
                if task_note:
                    search_query_parts.append(task_note)
                
                search_query = "\n".join(search_query_parts)
                
                # Limit to 2000 chars for fast embedding generation (most models handle this well)
                if len(search_query) > 2000:
                    # Prioritize title and objective, truncate note if needed
                    if len(task_title + "\n" + task_objective) <= 2000:
                        search_query = f"{task_title}\n{task_objective}\n{task_note[:2000 - len(task_title) - len(task_objective) - 2]}"
                    else:
                        # If even title+objective is too long, just use title (most important)
                        search_query = task_title[:2000]
                
                # Only search if we have a non-empty query
                if search_query and search_query.strip():
                    # Search for top 5 relevant memories (both notes and findings)
                    relevant_memories = await research_memory_service.search_memories(
                        session_id=session_id,
                        query=search_query,
                        memory_types=None,  # Search both notes and findings
                        limit=5
                    )
                else:
                    relevant_memories = []
                    logger.debug(f"Agent {agent_id} empty search query, skipping vector search", task=current_task.title if current_task else "no task")
                
                if relevant_memories:
                    # Format memories for context (full content, not truncated)
                    memory_parts = []
                    for mem in relevant_memories:
                        mem_type_label = "Note" if mem["memory_type"] == "note" else "Finding"
                        agent_label = f" (from {mem['agent_id']})" if mem.get("agent_id") else ""
                        memory_parts.append(
                            f"**{mem_type_label}{agent_label}**: {mem['title']}\n{mem['content']}"
                        )
                    notes_context = "\n\n".join(memory_parts)
                    # Log similarity for debugging/monitoring
                    avg_similarity = sum(m.get("similarity", 0.0) for m in relevant_memories) / len(relevant_memories) if relevant_memories else 0.0
                    logger.info(f"Agent {agent_id} found relevant memories via vector search",
                               memories_count=len(relevant_memories),
                               task=current_task.title,
                               avg_similarity=avg_similarity,
                               min_similarity=min((m.get("similarity", 0.0) for m in relevant_memories), default=0.0),
                               max_similarity=max((m.get("similarity", 0.0) for m in relevant_memories), default=0.0),
                               note="Using vector search instead of recent notes")
                else:
                    logger.debug(f"Agent {agent_id} no relevant memories found via vector search", task=current_task.title if current_task else "no task")
        except Exception as e:
            logger.warning(f"Agent {agent_id} vector search failed, using fallback", error=str(e))
            # Fallback to recent notes if vector search fails
            all_notes = agent_file.get("notes", [])
            recent_notes = all_notes[-5:] if len(all_notes) > 5 else all_notes
            notes_context = "\n".join([f"- {note}" for note in recent_notes]) if recent_notes else "No previous notes."
    else:
        # Fallback: use recent notes if research_memory_service not available
        all_notes = agent_file.get("notes", [])
        recent_notes = all_notes[-5:] if len(all_notes) > 5 else all_notes
        notes_context = "\n".join([f"- {note}" for note in recent_notes]) if recent_notes else "No previous notes."
        if not research_memory_service:
            logger.debug(f"Agent {agent_id} research_memory_service not available, using recent notes fallback")

    # Get agent characteristics from state
    agent_characteristics = state.get("agent_characteristics", {})
    role = agent_characteristics.get(agent_id, {}).get("role", f"Research Agent {agent_id}")
    expertise = agent_characteristics.get(agent_id, {}).get("expertise", "general research")
    personality = agent_characteristics.get(agent_id, {}).get("personality", "thorough and analytical")

    # Get tasks from other agents to help with note creation
    other_agents_tasks = []
    if agent_file_service:
        try:
            # Get all agent IDs from characteristics (typically agent_1, agent_2, agent_3)
            all_agent_ids = list(agent_characteristics.keys())
            for other_agent_id in all_agent_ids:
                if other_agent_id != agent_id:
                    try:
                        other_agent_file = await agent_file_service.read_agent_file(other_agent_id)
                        other_todos = other_agent_file.get("todos", [])
                        # Get pending and in_progress tasks
                        other_active_tasks = [
                            t for t in other_todos 
                            if t.status in ["pending", "in_progress"]
                        ]
                        if other_active_tasks:
                            other_agents_tasks.append({
                                "agent_id": other_agent_id,
                                "tasks": other_active_tasks
                            })
                    except Exception as e:
                        logger.debug(f"Could not load tasks from {other_agent_id}", error=str(e))
        except Exception as e:
            logger.warning(f"Failed to load other agents' tasks", error=str(e))

    logger.info(f"Agent {agent_id} loaded", role=role, expertise=expertise, todos_count=len(todos), other_agents_tasks_count=sum(len(ot["tasks"]) for ot in other_agents_tasks))

    # ENFORCE: Only one task at a time
    # CRITICAL: Check for duplicate tasks by title and remove duplicates before checking status
    # CRITICAL: Always keep only ONE task per title - prefer the most recent/active one
    seen_titles = {}
    unique_todos = []
    for todo in todos:
        if todo.title in seen_titles:
            # Duplicate found - decide which one to keep
            existing = seen_titles[todo.title]
            
            # Priority order for keeping a task:
            # 1. Status priority: in_progress > done > pending
            # 2. If same status: prefer one with more data (supervisor_message, return_count, additional_steps)
            # 3. If still same: prefer the one that appears later in the list (more recent)
            
            status_priority = {"in_progress": 3, "done": 2, "pending": 1}
            todo_priority = status_priority.get(todo.status, 0)
            existing_priority = status_priority.get(existing.status, 0)
            
            should_replace = False
            
            if todo_priority > existing_priority:
                # New task has higher status priority
                should_replace = True
            elif todo_priority == existing_priority:
                # Same status - check for data richness
                todo_data_score = 0
                existing_data_score = 0
                
                if hasattr(todo, "supervisor_message") and todo.supervisor_message:
                    todo_data_score += 3
                if hasattr(todo, "return_count") and getattr(todo, "return_count", 0) > 0:
                    todo_data_score += 2
                if hasattr(todo, "additional_steps") and getattr(todo, "additional_steps", 0) > 0:
                    todo_data_score += 1
                
                if hasattr(existing, "supervisor_message") and existing.supervisor_message:
                    existing_data_score += 3
                if hasattr(existing, "return_count") and getattr(existing, "return_count", 0) > 0:
                    existing_data_score += 2
                if hasattr(existing, "additional_steps") and getattr(existing, "additional_steps", 0) > 0:
                    existing_data_score += 1
                
                if todo_data_score > existing_data_score:
                    should_replace = True
                elif todo_data_score == existing_data_score:
                    # Same data score - prefer the one that appears later (more recent)
                    # Since we iterate in order, the current todo is more recent
                    should_replace = True
            
            if should_replace:
                unique_todos.remove(existing)
                unique_todos.append(todo)
                seen_titles[todo.title] = todo
                logger.debug(f"Agent {agent_id} replaced duplicate task",
                           title=todo.title,
                           old_status=existing.status,
                           new_status=todo.status,
                           note="Kept more recent/active task")
            # Otherwise keep existing (don't add current todo)
        else:
            unique_todos.append(todo)
            seen_titles[todo.title] = todo
    
    # Update todos list if duplicates were removed
    if len(unique_todos) < len(todos):
        logger.warning(f"Agent {agent_id} had duplicate tasks - removed {len(todos) - len(unique_todos)} duplicates",
                     original_count=len(todos),
                     unique_count=len(unique_todos),
                     duplicate_titles=[title for title, count in {t.title: sum(1 for t2 in todos if t2.title == t.title) for t in todos}.items() if count > 1])
        todos = unique_todos
        # Update agent file with deduplicated todos
        try:
            agent_file["todos"] = todos
            await agent_file_service.write_agent_file(
                agent_id=agent_id,
                role=role,
                expertise=expertise,
                todos=todos,
                notes=notes
            )
        except Exception as e:
            logger.warning(f"Failed to save deduplicated todos for agent {agent_id}", error=str(e))
    
    in_progress_tasks = [t for t in todos if t.status == "in_progress"]
    if len(in_progress_tasks) > 1:
        # CRITICAL BUG FIX: Multiple in_progress tasks detected - fix automatically
        # Keep the first one (oldest or most important), set others back to pending
        logger.error(f"CRITICAL BUG: Agent {agent_id} has {len(in_progress_tasks)} in_progress tasks: {[t.title for t in in_progress_tasks]}. Fixing automatically - keeping first, setting others to pending.",
                    agent_id=agent_id,
                    in_progress_tasks=[t.title for t in in_progress_tasks],
                    note="Agent can only work on ONE task at a time. Automatically fixing by keeping first in_progress task and setting others to pending.")
        
        # Keep the first in_progress task, set others to pending
        first_in_progress = in_progress_tasks[0]
        for task_to_fix in in_progress_tasks[1:]:
            try:
                await agent_file_service.update_agent_todo(
                    agent_id,
                    task_to_fix.title,
                    status="pending"
                )
                logger.warning(f"Fixed: Set task '{task_to_fix.title}' back to pending (agent {agent_id} can only work on one task)",
                             agent_id=agent_id,
                             task=task_to_fix.title,
                             kept_task=first_in_progress.title)
            except Exception as e:
                logger.error(f"Failed to fix task status", agent_id=agent_id, task=task_to_fix.title, error=str(e))
        
        # Reload todos after fix
        agent_file = await agent_file_service.read_agent_file(agent_id)
        todos = agent_file.get("todos", [])
        in_progress_tasks = [t for t in todos if t.status == "in_progress"]
        
        # Verify fix worked
        if len(in_progress_tasks) > 1:
            logger.error(f"CRITICAL: Fix failed - agent {agent_id} still has {len(in_progress_tasks)} in_progress tasks after fix attempt",
                        agent_id=agent_id,
                        in_progress_tasks=[t.title for t in in_progress_tasks])
            # Last resort: keep only first, manually set others
            for task_to_fix in in_progress_tasks[1:]:
                task_to_fix.status = "pending"
            # Save manually
            try:
                await agent_file_service.write_agent_file(
                    agent_id=agent_id,
                    role=role,
                    expertise=expertise,
                    todos=todos,
                    notes=notes
                )
                logger.info(f"Manually fixed multiple in_progress tasks for agent {agent_id}",
                           agent_id=agent_id,
                           kept_task=in_progress_tasks[0].title if in_progress_tasks else "none")
            except Exception as e:
                logger.error(f"Failed to manually fix tasks", agent_id=agent_id, error=str(e))
                raise ValueError(f"Agent {agent_id} has multiple in_progress tasks and automatic fix failed: {[t.title for t in in_progress_tasks]}")
        
        # Reload one more time to get fixed state
        agent_file = await agent_file_service.read_agent_file(agent_id)
        todos = agent_file.get("todos", [])
        in_progress_tasks = [t for t in todos if t.status == "in_progress"]

    # Get current task
    if in_progress_tasks:
        current_task = in_progress_tasks[0]
        has_supervisor_message = hasattr(current_task, "supervisor_message") and current_task.supervisor_message
        return_count = getattr(current_task, "return_count", 0)
        
        # Check if this task was returned by supervisor (has supervisor_message and return_count > 0)
        task_was_returned = has_supervisor_message and return_count > 0
        
        logger.info(f"Agent {agent_id} resuming task", 
                   task=current_task.title,
                   total_todos=len(todos),
                   pending_todos=len([t for t in todos if t.status == "pending"]),
                   done_todos=len([t for t in todos if t.status == "done"]),
                   in_progress_count=len(in_progress_tasks),
                   has_supervisor_message=has_supervisor_message,
                   return_count=return_count,
                   task_was_returned=task_was_returned,
                   note=f"✅ Agent continuing work on in_progress task. {'⚠️ This task was RETURNED by supervisor for rework - agent must address supervisor feedback!' if task_was_returned else 'Agent is continuing work on a task that was already in_progress.'} Agent will work on this task until it's completed and creates a new finding.")
    else:
        # Get next pending task
        pending_tasks = [t for t in todos if t.status == "pending"]
        if not pending_tasks:
            logger.info(f"Agent {agent_id} has no pending tasks",
                       total_todos=len(todos),
                       done_todos=len([t for t in todos if t.status == "done"]),
                       in_progress_todos=len([t for t in todos if t.status == "in_progress"]))
            return {
                "agent_id": agent_id,
                "topic": "no_tasks",
                "summary": "No pending tasks",
                "key_findings": [],
                "sources": [],
                "confidence": "n/a"
            }

        current_task = pending_tasks[0]
        
        # CRITICAL: Double-check that no other in_progress tasks exist before marking this as in_progress
        # This prevents race conditions where supervisor might have set another task to in_progress
        # Reload todos to get latest state (may have changed since we loaded them)
        agent_file = await agent_file_service.read_agent_file(agent_id)
        todos = agent_file.get("todos", [])
        existing_in_progress = [t for t in todos if t.status == "in_progress"]
        
        if existing_in_progress:
            # Another task is already in_progress - this should not happen after our fix above, but handle it
            logger.error(f"CRITICAL: Agent {agent_id} attempted to start new task '{current_task.title}', but found {len(existing_in_progress)} existing in_progress task(s): {[t.title for t in existing_in_progress]}. Will continue existing task instead.",
                        agent_id=agent_id,
                        new_task=current_task.title,
                        existing_in_progress_tasks=[t.title for t in existing_in_progress],
                        note="Agent can only work on ONE task at a time. Continuing existing in_progress task instead of starting new one.")
            # Use the existing in_progress task instead
            current_task = existing_in_progress[0]
        else:
            # Mark as in_progress - this should be safe now
            update_result = await agent_file_service.update_agent_todo(
                agent_id,
                current_task.title,
                status="in_progress"
            )
            
            # CRITICAL: Emit updated todos to frontend immediately after status change
            # This ensures frontend shows correct "in_progress" status
            if stream and update_result:
                # Reload todos to get updated status
                agent_file = await agent_file_service.read_agent_file(agent_id)
                todos = agent_file.get("todos", [])
                todos_dict = [
                    {
                        "title": t.title,
                        "status": t.status,
                        "objective": t.objective if hasattr(t, "objective") else "",
                        "expected_output": t.expected_output if hasattr(t, "expected_output") else "",
                        "note": t.note if hasattr(t, "note") else "",
                        "url": t.url if hasattr(t, "url") else None
                    }
                    for t in todos
                ]
                stream.emit_agent_todo(agent_id, todos_dict)
                logger.info(f"Agent {agent_id} todos emitted to frontend after marking task as in_progress",
                           task=current_task.title,
                           todos_count=len(todos_dict),
                           in_progress_count=sum(1 for t in todos if t.status == "in_progress"))
            
            if not update_result:
                # Update failed - reload and check what happened
                logger.warning(f"Agent {agent_id} failed to mark task '{current_task.title}' as in_progress - reloading to check state",
                             agent_id=agent_id,
                             task=current_task.title)
                agent_file = await agent_file_service.read_agent_file(agent_id)
                todos = agent_file.get("todos", [])
                existing_in_progress = [t for t in todos if t.status == "in_progress"]
                if existing_in_progress:
                    logger.info(f"After failed update, found {len(existing_in_progress)} in_progress task(s) for agent {agent_id}. Using first one.",
                               agent_id=agent_id,
                               in_progress_tasks=[t.title for t in existing_in_progress])
                    current_task = existing_in_progress[0]
                else:
                    # Check if our task is now in_progress (maybe update succeeded but returned False)
                    for t in todos:
                        if t.title == current_task.title and t.status == "in_progress":
                            current_task = t
                            logger.info(f"Task '{current_task.title}' is now in_progress (update may have succeeded)",
                                      agent_id=agent_id,
                                      task=current_task.title)
                            break
        
        # CRITICAL: Check if there are done tasks (completed and reviewed by supervisor)
        done_tasks = [t for t in todos if t.status == "done"]
        done_tasks_count = len(done_tasks)
        
        # Check if supervisor modified this task (has updated objective, guidance, etc.)
        task_was_updated = False
        if hasattr(current_task, "objective") and current_task.objective:
            # If task has detailed objective/guidance, it might have been updated by supervisor
            # We can't definitively know, but we log that agent sees the current state
            task_was_updated = True
        
        logger.info(f"Agent {agent_id} starting NEW task", 
                   task=current_task.title,
                   pending_tasks=len(pending_tasks),
                   total_todos=len(todos),
                   remaining_pending=len(pending_tasks) - 1,
                   done_tasks_count=done_tasks_count,
                   done_tasks_titles=[t.title[:50] for t in done_tasks[:3]],
                   task_may_be_updated=task_was_updated,
                   note=f"✅ Agent picked up next pending task. This means: 1) Previous task was completed and ACCEPTED by supervisor (chapter added to draft_report), OR 2) Agent has no in_progress tasks and is starting a new pending task. Agent sees ALL task changes made by supervisor (updates, new tasks). {done_tasks_count} tasks already completed and reviewed by supervisor (chapters added). {len(pending_tasks) - 1} pending tasks waiting after this one.")

        # Reload to get updated todos
        agent_file = await agent_file_service.read_agent_file(agent_id)
        todos = agent_file.get("todos", [])

    # Emit initial state
    if stream:
        stream.emit_research_start({"researcher_id": agent_id, "topic": current_task.title})
        # Convert todos to dict format for emission
        # CRITICAL: Include all fields to match supervisor's format for consistency
        todos_dict = [
            {
                "title": t.title,
                "status": t.status,
                "objective": t.objective if hasattr(t, "objective") else "",
                "expected_output": t.expected_output if hasattr(t, "expected_output") else "",
                "note": t.note if hasattr(t, "note") else "",
                "url": t.url if hasattr(t, "url") else None
            }
            for t in todos
        ]
        stream.emit_agent_todo(agent_id, todos_dict)

    # Create research plan with structured output
    # Extract user query from task if present (it should be there per supervisor instructions)
    task_guidance = current_task.note if hasattr(current_task, 'note') and current_task.note else ""
    task_objective = current_task.objective if hasattr(current_task, 'objective') else ""
    
    plan_prompt = f"""You are {role} with expertise in {expertise}.

Current task: {current_task.title}
Objective: {current_task.objective}
Guidance: {task_guidance}
Expected output: {current_task.expected_output}
Sources needed: {', '.join(current_task.sources_needed) if hasattr(current_task, 'sources_needed') and current_task.sources_needed else 'Various reliable sources'}

**CRITICAL**: The task objective and guidance above contain all the context you need. If the task mentions a user query (e.g., "The user asked: ..."), that is the specific topic you must research. Focus your research plan on exactly what is described in the task.

**MANDATORY: Keep your plan BRIEF and CONCISE.**
- reasoning: 1-2 sentences maximum
- current_goal: 1 sentence maximum (DO NOT repeat the full task objective, just state the goal)
- next_steps: 1-3 short action items (1-5 words each)
- expected_findings: 1 sentence maximum
- search_strategy: 1 sentence maximum
- fallback_if_stuck: 1 sentence maximum

Create a BRIEF, actionable research plan for completing this task.
"""

    try:
        # Use structured output - it handles JSON parsing automatically
        # CRITICAL: The issue is LLM generating 131k tokens in RESPONSE, not in input
        # We need to limit the RESPONSE tokens, not truncate input
        try:
            # Try to bind max_tokens if LLM supports it
            llm_for_plan = llm
            if hasattr(llm, "bind") or hasattr(llm, "with_config"):
                try:
                    if hasattr(llm, "with_config"):
                        llm_for_plan = llm.with_config({"max_tokens": 500})  # Limit response to 500 tokens
                    elif hasattr(llm, "bind"):
                        llm_for_plan = llm.bind(max_tokens=500)
                except:
                    pass  # If binding fails, use original LLM
            
            plan = await llm_for_plan.with_structured_output(
                AgentPlan,
                method="json_schema"  # Use JSON schema mode for better token control
            ).ainvoke([
                {"role": "system", "content": f"You are a research planning expert. Create VERY BRIEF, concise, actionable plans. Each field must be SHORT (1-2 sentences max). DO NOT write long explanations - keep it minimal and structured. CRITICAL: Your response must be under 500 tokens total."},
                {"role": "user", "content": plan_prompt}
            ])
        except Exception as e:
            logger.warning(f"Plan creation with max_tokens failed, using fallback", error=str(e))
            # Fallback without max_tokens
            plan = await llm.with_structured_output(
                AgentPlan,
                method="json_schema"
            ).ainvoke([
                {"role": "system", "content": f"You are a research planning expert. Create VERY BRIEF, concise, actionable plans. Each field must be SHORT (1-2 sentences max). DO NOT write long explanations - keep it minimal and structured."},
                {"role": "user", "content": plan_prompt}
            ])
        logger.info(f"Agent {agent_id} created plan", goal=plan.current_goal[:100] if plan.current_goal else "N/A")
    except Exception as e:
        error_msg = str(e)
        # Structured output should handle JSON parsing, but if it fails, log details
        # This usually happens if LLM response is malformed or too long
        if "Expecting value" in error_msg or "JSON" in error_msg or "parse" in error_msg.lower():
            logger.error(f"Agent {agent_id} plan creation failed - structured output parsing error", 
                        error=error_msg[:300],
                        error_type=type(e).__name__,
                        note="Structured output failed to parse LLM response (may be too long or malformed), using fallback plan")
        else:
            logger.error(f"Agent {agent_id} plan creation failed", error=error_msg[:300], error_type=type(e).__name__)
        # Fallback plan - structured output failed, use simple plan
        plan = AgentPlan(
            reasoning="Fallback plan due to structured output error",
            current_goal=current_task.objective,
            next_steps=["Search for sources", "Analyze findings"],
            expected_findings="Research results",
            search_strategy="Broad web search",
            fallback_if_stuck="Try alternative sources"
        )

    # Research execution (ReAct loop)
    sources = []  # Web search results (snippets)
    scraped_pages = []  # Scraped pages with summary, brief_info, is_relevant
    notes = []
    agent_history = []

    # Format other agents' tasks for context
    other_agents_tasks_context = "No other agents have active tasks."
    if other_agents_tasks:
        task_parts = []
        for agent_info in other_agents_tasks:
            agent_id_other = agent_info["agent_id"]
            tasks_list = agent_info["tasks"]
            task_lines = []
            for task in tasks_list:
                status_icon = "⏸️" if task.status == "in_progress" else "⬜"
                task_lines.append(f"  {status_icon} {task.title}")
                if hasattr(task, "objective") and task.objective:
                    task_lines.append(f"    Objective: {task.objective[:150]}")
            if task_lines:
                task_parts.append(f"**{agent_id_other}**:\n" + "\n".join(task_lines))
        if task_parts:
            other_agents_tasks_context = "\n\n".join(task_parts)
            other_agents_tasks_context = f"**Use this information to create notes that might be relevant to other agents' research:**\n\n{other_agents_tasks_context}"

    # Get user language from state (needed for response language)
    user_language = state.get("user_language", "English")
    # NOTE: We don't extract original_query, deep_search_result, or clarification_context here
    # because researchers should NOT see them - they only see the task description

    # CRITICAL: Build supervisor message section separately to avoid backslash in f-string
    supervisor_message_section = ""
    if current_task and hasattr(current_task, 'supervisor_message') and current_task.supervisor_message:
        supervisor_message_section = (
            f"**SUPERVISOR MESSAGE (REWORK REQUIRED):**\n"
            f"{current_task.supervisor_message}\n\n"
            f"**YOUR RESPONSE:** After completing the task, you must write a message to the supervisor "
            f"in the finding explaining how you addressed their concerns and what you found. "
            f"This message will be included in the finding as 'supervisor_message' field."
        )

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
Guidance: {current_task.note if hasattr(current_task, 'note') and current_task.note else 'No specific guidance provided'}

{supervisor_message_section}

**CRITICAL: YOU MUST WORK STRICTLY ON YOUR CURRENT TASK!**
- **MANDATORY**: Your research MUST be focused EXACTLY on the task objective and guidance above
- **MANDATORY**: The task description is your PRIMARY source of what to research - follow it strictly
- **FORBIDDEN**: Do NOT deviate from the task topic - research exactly what is described in the task!
- **FORBIDDEN**: Do NOT research topics from notes or other agents' tasks unless they DIRECTLY relate to YOUR current task
- If the task mentions "The user asked: ...", that is the SPECIFIC topic you must focus on - nothing else!
- The task description is self-contained and contains all context you need

Research plan:
Goal: {plan.current_goal}
Next steps: {', '.join(plan.next_steps)}
Strategy: {plan.search_strategy}

**CRITICAL: When you find information, provide DETAILED, COMPREHENSIVE summaries with full context, not just links.**
- Include specific facts, data, and insights in your summaries
- Explain what you found and why it's relevant
- Provide full context, not just "found X sources"
- Your findings should be self-contained and informative

**CRITICAL: UNDERSTANDING NOTES AND OTHER AGENTS' TASKS:**
- The notes below are from other agents or your previous work - they are for COORDINATION ONLY
- **MANDATORY**: You must work on YOUR task topic, NOT on topics from notes
- **MANDATORY**: Notes are shown so you can see what other agents need and write helpful notes for them
- **MANDATORY**: When you find information relevant to OTHER agents' tasks (shown below), write detailed notes about it
- **MANDATORY**: Your notes should help other agents find information they need for THEIR tasks
- **FORBIDDEN**: Do NOT switch your research focus to topics from notes - stay on YOUR task!

Your previous notes (for coordination - see what others might need):
{notes_context}

**OTHER AGENTS' ACTIVE TASKS (use this to write helpful notes for them):**
{other_agents_tasks_context}

**CRITICAL: NOTES FOR COORDINATION - READ CAREFULLY:**
- Use save_note to write detailed notes when you find information relevant to OTHER agents' tasks
- **MANDATORY**: When you find something that relates to other agents' research, write a comprehensive note about it
- **MANDATORY**: Your notes help other agents find information for THEIR tasks - write them clearly and in detail
- **MANDATORY**: Your notes are searchable by other agents - write them so they can find and understand the information
- **FORBIDDEN**: Do NOT use notes to change your research topic - work on YOUR task, write notes for OTHERS

**CRITICAL: READ TOOL DESCRIPTIONS CAREFULLY:**
- All detailed instructions for verification, source quality, deep research, and completion requirements are in the tool descriptions
- Read web_search, scrape_url, and done tool descriptions for complete guidance on verification and deep research
- You have up to {max_steps} steps - use EVERY SINGLE ONE to thoroughly research, verify, and cross-reference information
- Be thorough, cite sources with links, verify everything in multiple sources, and fulfill the objective. Go DEEP, not just surface-level!
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
            # CRITICAL: Pass task context to scrape_url for focused summarization
            # Also pass scraped_pages and sources to create_note handler
            context = {
                "search_provider": search_provider,
                "scraper": scraper,
                "stream": stream,
                "llm": llm,
                "agent_id": agent_id,
                "agent_memory_service": agent_memory_service,
                "agent_file_service": agent_file_service,
                "research_memory_service": research_memory_service,
                "session_id": session_id,
            }
            # Add task context for scrape_url handler
            if current_task:
                context["task_title"] = current_task.title
                context["task_objective"] = current_task.objective if hasattr(current_task, "objective") else ""
                context["task_note"] = current_task.note if hasattr(current_task, "note") and current_task.note else ""
            
            # CRITICAL: Pass scraped_pages and sources to create_note and create_finding handlers
            # Use closure to capture current values
            context["scraped_pages"] = scraped_pages
            context["sources"] = sources
            
            result = await ActionRegistry.execute(action_name, kwargs, context)
            # Convert result to string for ToolMessage (LangChain expects string)
            if isinstance(result, dict):
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
        # Check if action is enabled (reasoning_preamble is now enabled for deep research)
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
            base_max_steps = settings.deep_research_agent_max_steps
        else:
            base_max_steps = 5  # Default fallback
        
        # CRITICAL: Check if current task has additional_steps (task was returned for continuation)
        # If task was returned, increase step limit and preserve history
        if current_task and hasattr(current_task, "additional_steps") and current_task.additional_steps > 0:
            max_steps = base_max_steps + current_task.additional_steps
            logger.info(f"Agent {agent_id} task continuation detected",
                       task=current_task.title,
                       base_steps=base_max_steps,
                       additional_steps=current_task.additional_steps,
                       total_steps=max_steps,
                       return_count=getattr(current_task, "return_count", 0),
                       supervisor_message_present=bool(getattr(current_task, "supervisor_message", None)),
                       note="Task was returned for continuation - step limit increased, history will be preserved")
        else:
            max_steps = base_max_steps
    else:
        # If max_steps was provided, check if we need to add additional_steps
        if current_task and hasattr(current_task, "additional_steps") and current_task.additional_steps > 0:
            max_steps = max_steps + current_task.additional_steps
            logger.info(f"Agent {agent_id} adding additional steps for continuation",
                       task=current_task.title,
                       original_max_steps=max_steps - current_task.additional_steps,
                       additional_steps=current_task.additional_steps,
                       total_steps=max_steps)
    
    logger.info(f"Agent {agent_id} starting ReAct loop", max_steps=max_steps)
    
    # CRITICAL: If task was returned for continuation, preserve previous history
    # Check if there's supervisor_message (indicates task continuation)
    if current_task and hasattr(current_task, "supervisor_message") and current_task.supervisor_message:
        logger.info(f"Agent {agent_id} task continuation - supervisor message present",
                   task=current_task.title,
                   supervisor_message_preview=current_task.supervisor_message[:200],
                   previous_notes_count=len(notes),
                   previous_sources_count=len(sources) if 'sources' in locals() else 0,
                   previous_scraped_pages_count=len(scraped_pages) if 'scraped_pages' in locals() else 0,
                   note="Previous history preserved through notes_context - agent will see all previous notes and respond to supervisor's message")
        # CRITICAL: History is preserved through:
        # 1. notes_context - contains all previous notes from agent_file (loaded above via vector search)
        # 2. agent_file contains all todos with their history (including supervisor_message)
        # 3. When agent continues, it sees all previous notes in notes_context, so history is preserved
        # 4. ReAct history (agent_history) is local to this call, but important info is in notes

    # CRITICAL: Log task start with full context
    logger.info(f"Agent {agent_id} starting ReAct loop",
               task=current_task.title,
               max_steps=max_steps,
               pending_tasks_after_this=len([t for t in todos if t.status == "pending"]),
               total_todos=len(todos),
               note="Agent will work on this task until done() or max_steps reached. Other pending tasks will wait.")
    
    # ReAct loop
    for step in range(max_steps):
        try:
            # CRITICAL: Log each step to track agent progress
            logger.info(f"Agent {agent_id} ReAct step {step + 1}/{max_steps}",
                       agent_id=agent_id,
                       step=step + 1,
                       max_steps=max_steps,
                       task=current_task.title,
                       sources_count=len(sources),
                       notes_count=len(notes),
                       note="Starting ReAct step - will call LLM to get next action")
            
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

            # CRITICAL: No context truncation - agent_history keeps all messages for full context
            # This allows agents to have complete context of their research process

            # CRITICAL: system_prompt contains the task (current_task.title, objective, guidance)
            # This is NEVER truncated - agent always sees their task
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

            # CRITICAL: Get LLM response with timeout and logging
            logger.info(f"Agent {agent_id} step {step + 1}: calling LLM",
                       agent_id=agent_id,
                       step=step + 1,
                       messages_count=len(messages),
                       note="Calling LLM to get next action - this may take time")
            
            try:
                # Add timeout to prevent infinite hanging (default 120 seconds per LLM call)
                response = await asyncio.wait_for(
                    llm_with_tools.ainvoke(messages),
                    timeout=120.0
                )
                logger.info(f"Agent {agent_id} step {step + 1}: received LLM response",
                           agent_id=agent_id,
                           step=step + 1,
                           has_tool_calls=hasattr(response, "tool_calls") and bool(response.tool_calls),
                           response_content_length=len(str(response.content)) if hasattr(response, "content") else 0,
                           note="LLM responded successfully")
            except asyncio.TimeoutError:
                logger.error(f"Agent {agent_id} step {step + 1}: LLM call timed out after 120 seconds",
                           agent_id=agent_id,
                           step=step + 1,
                           task=current_task.title,
                           note="LLM call exceeded timeout - agent may be stuck. This is a critical error.")
                # Continue to next step or break
                raise
            except Exception as e:
                logger.error(f"Agent {agent_id} step {step + 1}: LLM call failed",
                           agent_id=agent_id,
                           step=step + 1,
                           error=str(e),
                           error_type=type(e).__name__,
                           task=current_task.title,
                           note="LLM call failed - this may cause agent to hang")
                raise

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
                    # CRITICAL: Require extensive work before allowing done()
                    # Prevent agents from completing tasks too quickly - force deep research
                    MIN_STEPS_FOR_DONE = max(5, int(max_steps * 0.6))  # Minimum 60% of max_steps (or 5, whichever is higher)
                    MIN_SOURCES_FOR_DONE = 5  # Minimum 5 sources before done() is allowed (increased for deep research)
                    MIN_VERIFIED_CLAIMS = 3  # Minimum 3 verified claims in multiple sources
                    
                    if step < MIN_STEPS_FOR_DONE:
                        logger.warning(f"Agent {agent_id} tried to call done() too early (step {step + 1}/{max_steps}, min {MIN_STEPS_FOR_DONE} required)",
                                     step=step + 1,
                                     max_steps=max_steps,
                                     min_steps=MIN_STEPS_FOR_DONE,
                                     min_percentage=int((MIN_STEPS_FOR_DONE/max_steps)*100),
                                     task=current_task.title,
                                     pending_tasks_waiting=len([t for t in todos if t.status == "pending"]),
                                     note=f"Agent must complete at least {MIN_STEPS_FOR_DONE} steps ({int((MIN_STEPS_FOR_DONE/max_steps)*100)}% of max) before calling done() - continue deep research! Other pending tasks are waiting.")
                        # Don't allow done() - continue research
                        done = False
                    elif len(sources) < MIN_SOURCES_FOR_DONE:
                        logger.warning(f"Agent {agent_id} tried to call done() with insufficient sources (found {len(sources)}, min {MIN_SOURCES_FOR_DONE} required)",
                                     sources_count=len(sources),
                                     min_sources=MIN_SOURCES_FOR_DONE,
                                     task=current_task.title,
                                     step=step + 1,
                                     max_steps=max_steps,
                                     pending_tasks_waiting=len([t for t in todos if t.status == "pending"]),
                                     note=f"Agent must find at least {MIN_SOURCES_FOR_DONE} sources before calling done() - continue searching! Other pending tasks are waiting.")
                        # Don't allow done() - continue research
                        done = False
                    else:
                        done = True
                        logger.info(f"Agent {agent_id} signaled done after extensive deep research",
                                   step=step + 1,
                                   max_steps=max_steps,
                                   steps_used_percentage=int(((step + 1)/max_steps)*100),
                                   sources_count=len(sources),
                                   notes_count=len(notes),
                                   note=f"Used {step + 1}/{max_steps} steps ({int(((step + 1)/max_steps)*100)}%) and found {len(sources)} sources")
                    break

            # CRITICAL: Check if agent called create_finding tool
            finding_created = False
            create_finding_tool_call = None
            for tc in tool_calls:
                tool_name = None
                if hasattr(tc, "name"):
                    tool_name = tc.name
                elif isinstance(tc, dict):
                    tool_name = tc.get("name") or tc.get("function", {}).get("name")
                
                if tool_name == "create_finding":
                    finding_created = True
                    create_finding_tool_call = tc
                    break
            
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
            
            # CRITICAL: If create_finding was called, execute it IMMEDIATELY before break
            # This ensures the result is added to agent_history and can be found later
            if finding_created and create_finding_tool_call:
                logger.info(f"Agent {agent_id} called create_finding - executing immediately", 
                           step=step + 1, 
                           sources_count=len(sources), 
                           notes_count=len(notes),
                           task=current_task.title,
                           pending_tasks_waiting=len([t for t in todos if t.status == "pending"]),
                           note="Finding tool called - executing it now to ensure result is in history")
                
                try:
                    tool_name, tool_args = extract_tool_info(create_finding_tool_call)
                    
                    # Execute create_finding with proper context
                    finding_result = await ActionRegistry.execute(
                        tool_name,
                        tool_args,
                        {
                            "search_provider": search_provider,
                            "scraper": scraper,
                            "stream": stream,
                            "llm": llm,
                            "agent_id": agent_id,
                            "agent_memory_service": agent_memory_service,
                            "agent_file_service": agent_file_service,
                            "research_memory_service": research_memory_service,
                            "session_id": session_id,
                            "task_title": current_task.title,
                            "task_objective": current_task.objective if hasattr(current_task, "objective") else "",
                            "task_note": current_task.note if hasattr(current_task, "note") and current_task.note else "",
                            "scraped_pages": scraped_pages,
                            "sources": sources,
                        }
                    )
                    
                    # Handle result format
                    if isinstance(finding_result, str):
                        try:
                            finding_result = json.loads(finding_result)
                        except:
                            finding_result = {"error": "Could not parse result"}
                    
                    # Extract tool_call_id
                    tool_call_id = None
                    if hasattr(create_finding_tool_call, "id"):
                        tool_call_id = create_finding_tool_call.id
                    elif isinstance(create_finding_tool_call, dict):
                        tool_call_id = create_finding_tool_call.get("id")
                    
                    # Format output for history
                    output_str = json.dumps(finding_result, ensure_ascii=False) if not isinstance(finding_result, str) else finding_result
                    
                    # Add assistant message with tool call to history
                    agent_history.append({
                        "role": "assistant",
                        "content": response.content if hasattr(response, "content") else "",
                        "tool_calls": [create_finding_tool_call]
                    })
                    
                    # Add tool result to history - CRITICAL: this allows finding to be found later
                    agent_history.append({
                        "role": "tool",
                        "content": output_str,
                        "tool_call_id": tool_call_id or f"call_{step}_create_finding"
                    })
                    
                    logger.info(f"Agent {agent_id} create_finding executed and added to history",
                               step=step + 1,
                               result_success=finding_result.get("success") if isinstance(finding_result, dict) else False,
                               summary_length=len(finding_result.get("summary", "")) if isinstance(finding_result, dict) else 0,
                               note="Finding result now in agent_history - will be found during task completion check")
                except Exception as e:
                    logger.error(f"Agent {agent_id} create_finding execution failed", error=str(e), exc_info=True)
                    # Continue - will create finding automatically later
                
                # Now break after executing create_finding
                break
            
            if done or not tool_calls:
                if done:
                    logger.info(f"Agent {agent_id} signaled done", 
                               step=step + 1, 
                               max_steps=max_steps,
                               sources_count=len(sources), 
                               notes_count=len(notes),
                               task=current_task.title,
                               pending_tasks_waiting=len([t for t in todos if t.status == "pending"]),
                               note="Task completed - will create finding and move to next pending task in next cycle")
                else:
                    logger.warning(f"Agent {agent_id} step {step}: no tool calls, ending", 
                                 step=step + 1,
                                 max_steps=max_steps,
                                 task=current_task.title,
                                 response_content_preview=str(response.content)[:200] if hasattr(response, "content") else "no content",
                                 pending_tasks_waiting=len([t for t in todos if t.status == "pending"]),
                                 note="No tool calls - will create finding automatically and move to next pending task in next cycle")
                break

            # Execute tools - can be parallel if multiple independent tools called
            action_results = []
            
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
                            "agent_memory_service": agent_memory_service,
                            "agent_file_service": agent_file_service,
                            "research_memory_service": research_memory_service,
                            "session_id": session_id,
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
                    # CRITICAL: json is imported at module level (line 4)
                    if isinstance(result, str):
                        try:
                            result = json.loads(result)
                        except Exception as e:
                            logger.warning(f"Agent {agent_id} step {step}: failed to parse result as JSON", error=str(e))
                            result = {"error": "Could not parse result"}

                    if tool_name == "web_search" and isinstance(result, dict) and "results" in result:
                        new_sources = result["results"]
                        # CRITICAL: Log web_search results to diagnose snippet issues
                        new_sources_with_snippet = sum(1 for s in new_sources if s.get("snippet", "").strip())
                        new_sources_snippet_lengths = [len(s.get("snippet", "").strip()) for s in new_sources if s.get("snippet")]
                        logger.debug(f"Agent {agent_id} web_search results (parallel)",
                                   results_count=len(new_sources),
                                   with_snippet=new_sources_with_snippet,
                                   without_snippet=len(new_sources) - new_sources_with_snippet,
                                   snippet_lengths=new_sources_snippet_lengths[:5] if new_sources_snippet_lengths else [],
                                   note="Check if web_search returns snippets - if not, that's the problem!")
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
                        
                        # CRITICAL: Format result for LLM to see ALL results (up to 15) so agent can see authoritative sources
                        # Not just top 5, because authoritative sources like ai.meta.com might be ranked lower
                        all_results = result.get("results", [])
                        formatted_result = {
                            "results_count": len(all_results),
                            "results": [
                                {
                                    "title": r.get("title", ""),
                                    "url": r.get("url", ""),
                                    "snippet": r.get("snippet", "")[:200]  # Truncate for readability
                                }
                                for r in all_results[:15]  # Show top 15 results so agent sees authoritative sources
                            ],
                            "note": f"Total {len(all_results)} results found. When calling select_urls_to_scrape, pass ALL {len(all_results)} results, not just the first few!"
                        }
                        result = formatted_result
                    
                    # CRITICAL: Handle scrape_url results - store separately with summary, brief_info, is_relevant
                    # scrape_url returns: {"scraped": [{"url": ..., "title": ..., "summary": ..., "brief_info": ..., "is_relevant": ...}, ...]}
                    if tool_name == "scrape_url" and isinstance(result, dict):
                        # Handle both single result and list of results
                        scraped_items = []
                        if "scraped" in result:
                            # List format: {"scraped": [...]}
                            scraped_items = result.get("scraped", [])
                        elif "url" in result:
                            # Single result format: {"url": ...}
                            scraped_items = [result]
                        
                        # Store scraped pages separately (not in sources)
                        for item in scraped_items:
                            if isinstance(item, dict) and "url" in item:
                                scraped_page = {
                                    "url": item.get("url", ""),
                                    "title": item.get("title", ""),
                                    "summary": item.get("summary", ""),  # Comprehensive summary for findings
                                    "brief_info": item.get("brief_info", ""),  # Brief info for tool history
                                    "is_relevant": item.get("is_relevant", True),  # Relevance flag
                                }
                                if scraped_page["summary"] or scraped_page["title"]:
                                    scraped_pages.append(scraped_page)
                        
                        if scraped_items:
                            logger.info(f"Agent {agent_id} scrape_url results stored (parallel)",
                                       scraped_count=len(scraped_items),
                                       stored_pages=len(scraped_pages),
                                       note="Scraped pages stored separately with summary, brief_info, is_relevant")
                    
                    # Handle scrape_url results - DO NOT automatically create notes
                    # The agent should analyze scraped content and decide what's important to save
                    # Only save notes when agent explicitly identifies valuable information

                    # Extract tool_call_id
                    tool_call_id = None
                    if hasattr(tool_call, "id"):
                        tool_call_id = tool_call.id
                    elif isinstance(tool_call, dict):
                        tool_call_id = tool_call.get("id")
                    
                    # CRITICAL: For scrape_url, create compact output for history (save context!)
                    # Store only brief_info in history, not full summary
                    if tool_name == "scrape_url" and isinstance(result, dict):
                        # Create compact version for history
                        compact_result = {}
                        if "scraped" in result:
                            compact_result["scraped"] = [
                                {
                                    "url": item.get("url", ""),
                                    "title": item.get("title", ""),
                                    "brief_info": item.get("brief_info", ""),
                                    "is_relevant": item.get("is_relevant", True),
                                    "summary_length": len(item.get("summary", "")),
                                    "note": "Full summary stored separately, not in history to save context"
                                }
                                for item in result.get("scraped", [])
                            ]
                            compact_result["count"] = result.get("count", 0)
                        else:
                            compact_result = {
                                "url": result.get("url", ""),
                                "title": result.get("title", ""),
                                "brief_info": result.get("brief_info", ""),
                                "is_relevant": result.get("is_relevant", True),
                                "summary_length": len(result.get("summary", "")),
                                "note": "Full summary stored separately, not in history to save context"
                            }
                        output_str = json.dumps(compact_result, ensure_ascii=False)
                    else:
                        # For other tools, keep full output but truncate if too large
                        if isinstance(result, dict):
                            # Check if result is too large (e.g., web_search with many results)
                            result_str = json.dumps(result, ensure_ascii=False)
                            if len(result_str) > 5000:  # If too large, truncate
                                # For web_search, keep structure but truncate snippets
                                if "results" in result:
                                    truncated_result = result.copy()
                                    truncated_result["results"] = [
                                        {
                                            "title": r.get("title", ""),
                                            "url": r.get("url", ""),
                                            "snippet": r.get("snippet", "")[:150] + "..." if len(r.get("snippet", "")) > 150 else r.get("snippet", "")
                                        }
                                        for r in result.get("results", [])[:10]  # Limit to 10 results in history
                                    ]
                                    output_str = json.dumps(truncated_result, ensure_ascii=False)
                                else:
                                    output_str = result_str[:5000] + "... [truncated]"
                            else:
                                output_str = result_str
                        else:
                            output_str = str(result) if not isinstance(result, str) else result
                            if len(output_str) > 5000:
                                output_str = output_str[:5000] + "... [truncated]"
                    
                    action_results.append({
                        "tool_call_id": tool_call_id or f"call_{step}_{len(action_results)}",
                        "output": output_str
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
                    # CRITICAL: json is imported at module level (line 4)
                    if isinstance(result, str):
                        try:
                            result = json.loads(result)
                        except Exception as e:
                            logger.warning(f"Agent {agent_id} step {step}: failed to parse result as JSON", error=str(e))
                            result = {"error": "Could not parse result"}

                    if tool_name == "web_search" and isinstance(result, dict) and "results" in result:
                        new_sources = result["results"]
                        # CRITICAL: Log web_search results to diagnose snippet issues
                        new_sources_with_snippet = sum(1 for s in new_sources if s.get("snippet", "").strip())
                        new_sources_snippet_lengths = [len(s.get("snippet", "").strip()) for s in new_sources if s.get("snippet")]
                        logger.debug(f"Agent {agent_id} web_search results (sequential)",
                                   results_count=len(new_sources),
                                   with_snippet=new_sources_with_snippet,
                                   without_snippet=len(new_sources) - new_sources_with_snippet,
                                   snippet_lengths=new_sources_snippet_lengths[:5] if new_sources_snippet_lengths else [],
                                   note="Check if web_search returns snippets - if not, that's the problem!")
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
                        
                        # CRITICAL: Format result for LLM to see ALL results (up to 15) so agent can see authoritative sources
                        # Not just top 5, because authoritative sources like ai.meta.com might be ranked lower
                        all_results = result.get("results", [])
                        formatted_result = {
                            "results_count": len(all_results),
                            "results": [
                                {
                                    "title": r.get("title", ""),
                                    "url": r.get("url", ""),
                                    "snippet": r.get("snippet", "")[:200]  # Truncate for readability
                                }
                                for r in all_results[:15]  # Show top 15 results so agent sees authoritative sources
                            ],
                            "note": f"Total {len(all_results)} results found. When calling select_urls_to_scrape, pass ALL {len(all_results)} results, not just the first few!"
                        }
                        result = formatted_result
                    
                    # CRITICAL: Handle scrape_url results - store separately with summary, brief_info, is_relevant
                    # scrape_url returns: {"scraped": [{"url": ..., "title": ..., "summary": ..., "brief_info": ..., "is_relevant": ...}, ...]}
                    if tool_name == "scrape_url" and isinstance(result, dict):
                        # Handle both single result and list of results
                        scraped_items = []
                        if "scraped" in result:
                            # List format: {"scraped": [...]}
                            scraped_items = result.get("scraped", [])
                        elif "url" in result:
                            # Single result format: {"url": ...}
                            scraped_items = [result]
                        
                        # Store scraped pages separately (not in sources)
                        for item in scraped_items:
                            if isinstance(item, dict) and "url" in item:
                                scraped_page = {
                                    "url": item.get("url", ""),
                                    "title": item.get("title", ""),
                                    "summary": item.get("summary", ""),  # Comprehensive summary for findings
                                    "brief_info": item.get("brief_info", ""),  # Brief info for tool history
                                    "is_relevant": item.get("is_relevant", True),  # Relevance flag
                                }
                                if scraped_page["summary"] or scraped_page["title"]:
                                    scraped_pages.append(scraped_page)
                        
                        if scraped_items:
                            logger.info(f"Agent {agent_id} scrape_url results stored (sequential)",
                                       scraped_count=len(scraped_items),
                                       stored_pages=len(scraped_pages),
                                       note="Scraped pages stored separately with summary, brief_info, is_relevant")
                    
                    # Handle scrape_url results - DO NOT automatically create notes
                    # Agent should analyze scraped content and decide what's important to save
                    # Only save notes when agent explicitly identifies valuable information

                    # Extract tool_call_id
                    tool_call_id = None
                    if hasattr(tool_call, "id"):
                        tool_call_id = tool_call.id
                    elif isinstance(tool_call, dict):
                        tool_call_id = tool_call.get("id")
                    
                    # No truncation - keep full tool output for complete context
                    output_str = json.dumps(result) if not isinstance(result, str) else result
                    
                    action_results.append({
                        "tool_call_id": tool_call_id or f"call_{step}_{len(action_results)}",
                        "output": output_str
                    })

            # Add to history
            agent_history.append({
                "role": "assistant",
                "content": response.content if hasattr(response, "content") else "",
                "tool_calls": tool_calls
            })

            for result in action_results:
                # No truncation - keep full tool output in history for complete context
                output_content = result["output"]
                
                agent_history.append({
                    "role": "tool",
                    "content": output_content,
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
                        # CRITICAL: Log actual lengths to understand what's being passed
                        goal_length = len(plan.current_goal) if plan.current_goal else 0
                        direction_length = len(reflection.new_direction) if reflection.new_direction else 0
                        logger.info(f"Agent {agent_id} starting replan", 
                                   previous_goal_length=goal_length,
                                   new_direction_length=direction_length,
                                   previous_goal_preview=plan.current_goal[:150] if plan.current_goal else "N/A",
                                   new_direction_preview=reflection.new_direction[:150] if reflection.new_direction else "N/A")
                        
                        # CRITICAL: Only truncate if REALLY long (over 500 chars) - don't break normal usage
                        # The real issue is LLM generating 131k tokens in RESPONSE, not in input
                        # So we need to limit the RESPONSE, not the input
                        max_goal_length = 500  # Only truncate if extremely long
                        max_direction_length = 500
                        truncated_goal = plan.current_goal[:max_goal_length] + "..." if plan.current_goal and len(plan.current_goal) > max_goal_length else (plan.current_goal or "")
                        truncated_direction = reflection.new_direction[:max_direction_length] + "..." if reflection.new_direction and len(reflection.new_direction) > max_direction_length else (reflection.new_direction or "")
                        
                        replan_prompt = f"""Your current approach needs adjustment.

Previous plan goal: {truncated_goal}
New direction: {truncated_direction}

Create an updated research plan incorporating this new direction. Keep it concise and actionable.

**MANDATORY: Keep your plan BRIEF and CONCISE.**
- reasoning: 1-2 sentences maximum
- current_goal: 1 sentence maximum (DO NOT repeat the full previous goal, just state the new goal)
- next_steps: 1-3 short action items (1-5 words each)
- expected_findings: 1 sentence maximum
- search_strategy: 1 sentence maximum
- fallback_if_stuck: 1 sentence maximum
"""
                        
                        # CRITICAL: The issue is LLM generating 131k tokens in RESPONSE
                        # We need to add max_tokens limit to the LLM call itself, not truncate input
                        # Check if LLM supports max_tokens parameter
                        try:
                            # Try to bind max_tokens if LLM supports it
                            llm_for_replan = llm
                            if hasattr(llm, "bind") or hasattr(llm, "with_config"):
                                # Some LLMs support max_tokens via bind/with_config
                                try:
                                    if hasattr(llm, "with_config"):
                                        llm_for_replan = llm.with_config({"max_tokens": 500})  # Limit response to 500 tokens
                                    elif hasattr(llm, "bind"):
                                        llm_for_replan = llm.bind(max_tokens=500)
                                except:
                                    pass  # If binding fails, use original LLM
                            
                            plan = await llm_for_replan.with_structured_output(
                                AgentPlan,
                                method="json_schema"
                            ).ainvoke([
                                {"role": "system", "content": "You are a research planning expert. Create VERY BRIEF, concise, actionable plans. Each field must be SHORT (1-2 sentences max). DO NOT write long explanations - keep it minimal and structured. DO NOT repeat previous plan details - just create a new brief plan. CRITICAL: Your response must be under 500 tokens total."},
                                {"role": "user", "content": replan_prompt}
                            ])
                        except Exception as e:
                            logger.warning(f"Replan with max_tokens failed, using fallback", error=str(e))
                            # Fallback without max_tokens
                            plan = await llm.with_structured_output(
                                AgentPlan,
                                method="json_schema"
                            ).ainvoke([
                                {"role": "system", "content": "You are a research planning expert. Create VERY BRIEF, concise, actionable plans. Each field must be SHORT (1-2 sentences max). DO NOT write long explanations - keep it minimal and structured. DO NOT repeat previous plan details - just create a new brief plan."},
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

        except asyncio.TimeoutError as e:
            logger.error(f"Agent {agent_id} step {step + 1} failed: LLM timeout",
                        agent_id=agent_id,
                        step=step + 1,
                        max_steps=max_steps,
                        task=current_task.title,
                        sources_count=len(sources),
                        notes_count=len(notes),
                        error=str(e),
                        note="LLM call timed out - breaking ReAct loop. Agent will create finding with collected data.")
            # Break loop - agent will create finding with collected data
            break
        except Exception as e:
            logger.error(f"Agent {agent_id} step {step + 1} failed",
                        agent_id=agent_id,
                        step=step + 1,
                        max_steps=max_steps,
                        task=current_task.title,
                        sources_count=len(sources),
                        notes_count=len(notes),
                        error=str(e),
                        error_type=type(e).__name__,
                        note="Step failed - breaking ReAct loop. Agent will create finding with collected data.")
            # Break loop - agent will create finding with collected data
            break

    # Task completion - create finding using LLM from scraped summaries + search snippets
    # CRITICAL: This is ALWAYS called when task completes (either agent called create_finding, done(), or max_steps reached)
    # If agent already called create_finding, we use that result. Otherwise, we call it automatically here.
    
    # CRITICAL: Log task completion status
    final_step = step if 'step' in locals() else max_steps
    
    # Reload todos to get current status (may have changed)
    try:
        agent_file = await agent_file_service.read_agent_file(agent_id)
        todos = agent_file.get("todos", [])
    except:
        pass  # Use existing todos if reload fails
    
    pending_tasks_waiting = len([t for t in todos if t.status == "pending"])
    done_tasks_count = len([t for t in todos if t.status == "done"])
    current_task_status = "done"  # Will be marked as done below
    
    logger.info(f"Agent {agent_id} task completion - creating finding",
               task=current_task.title,
               final_step=final_step,
               max_steps=max_steps,
               sources_found=len(sources),
               scraped_pages=len(scraped_pages),
               notes_created=len(notes),
               pending_tasks_waiting=pending_tasks_waiting,
               done_tasks_count=done_tasks_count,
               note=f"Task completed (done() called or max_steps reached) - creating finding. After supervisor review: if chapter added → task stays 'done' and agent will pick next pending task in next cycle. If task returned → task becomes 'in_progress' and agent will continue this task in next cycle. {pending_tasks_waiting} pending tasks waiting.")
    
    # Check if finding was already created via create_finding tool
    finding_already_created = False
    finding_summary = None
    finding_key_findings = None
    
    # Check agent_history for create_finding tool call result
    for msg in reversed(agent_history):
        if msg.get("role") == "tool":
            try:
                content = msg.get("content", "")
                if isinstance(content, str):
                    result = json.loads(content)
                else:
                    result = content
                
                if isinstance(result, dict) and result.get("success") and "summary" in result:
                    finding_already_created = True
                    finding_summary = result.get("summary")
                    finding_key_findings = result.get("key_findings", [])
                    logger.info(f"Agent {agent_id} finding already created via create_finding tool",
                               summary_length=len(finding_summary) if finding_summary else 0,
                               key_findings_count=len(finding_key_findings) if finding_key_findings else 0)
                    break
            except:
                pass
    
    # If finding not created yet, create it automatically
    if not finding_already_created:
        logger.info(f"Agent {agent_id} creating finding automatically (not called by agent)",
                   step=step if 'step' in locals() else max_steps,
                   max_steps=max_steps,
                   note="Agent did not call create_finding - calling automatically as resulting tool")
    
    # Step 1: Collect all relevant data
    # - Summary from scraped pages that are relevant (is_relevant=True)
    # - Snippets from web_search results (with stricter filtering - >50 chars instead of 30)
    
    relevant_scraped_summaries = []
    for page in scraped_pages:
        if page.get("is_relevant", True) and page.get("summary"):
            relevant_scraped_summaries.append({
                "url": page.get("url", ""),
                "title": page.get("title", ""),
                "summary": page.get("summary", "")
            })
    
    # Collect snippets from web_search (stricter filtering - >50 chars)
    useful_snippets = []
    for src in sources:
        snippet = src.get("snippet", "").strip()
        title = src.get("title", "").strip()
        url = src.get("url", "").strip()
        
        # Stricter filtering for snippets (only substantial content)
        if snippet and len(snippet) > 50:  # More strict than 30
            snippet_lower = snippet.lower()
            is_metadata = any([
                "found" in snippet_lower and "sources" in snippet_lower and "query" in snippet_lower,
                snippet_lower.startswith("search:") or snippet_lower.startswith("query:"),
                snippet_lower.count("http") > 2,
            ])
            if not is_metadata:
                useful_snippets.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet
                })
    
    logger.info(f"Agent {agent_id} task completion - data collection",
               relevant_scraped_pages=len(relevant_scraped_summaries),
               useful_snippets=len(useful_snippets),
               total_scraped_pages=len(scraped_pages),
               total_sources=len(sources),
               note="Collected relevant scraped summaries and useful snippets for LLM finding generation")
    
    # Step 2: Use LLM to generate comprehensive finding from collected data (if not already created)
    llm_finding_created = False  # Track if LLM successfully created finding
    if finding_already_created:
        summary = finding_summary
        key_findings = finding_key_findings
        llm_finding_created = True
        logger.info(f"Agent {agent_id} using finding from create_finding tool call",
                   summary_length=len(summary) if summary else 0,
                   key_findings_count=len(key_findings) if key_findings else 0)
    elif relevant_scraped_summaries or useful_snippets:
        try:
            # CRITICAL: Call create_finding handler automatically
            from src.workflow.search.actions import create_finding_handler
            
            # Prepare context for create_finding_handler
            finding_context = {
                "agent_id": agent_id,
                "llm": llm,
                "stream": stream,
                "task_title": current_task.title,
                "task_objective": current_task.objective if hasattr(current_task, "objective") else "",
                "task_note": current_task.note if hasattr(current_task, "note") and current_task.note else "",
                "scraped_pages": scraped_pages,
                "sources": sources,
            }
            
            finding_result = await create_finding_handler({}, finding_context)
            
            if finding_result.get("success"):
                summary = finding_result.get("summary", "")
                key_findings = finding_result.get("key_findings", [])
                llm_finding_created = True  # Mark that LLM successfully created finding
                logger.info(f"Agent {agent_id} automatically created finding via create_finding handler",
                           summary_length=len(summary),
                           key_findings_count=len(key_findings),
                           scraped_pages_used=len(relevant_scraped_summaries),
                           snippets_used=len(useful_snippets),
                           note="Finding created automatically as resulting tool - will skip old fallback logic")
            else:
                raise Exception(finding_result.get("error", "Unknown error"))
            
        except Exception as e:
            logger.error(f"Agent {agent_id} automatic finding creation failed", error=str(e))
            # Fallback - will use old logic below
            summary = None
            key_findings = []
            llm_finding_created = False
    else:
        # No data collected - use fallback
        logger.warning(f"Agent {agent_id} no data collected for finding generation",
                     scraped_pages=len(scraped_pages),
                     sources=len(sources),
                     note="No relevant scraped summaries or useful snippets - using fallback")
        summary = None
        key_findings = []
        llm_finding_created = False
    
    # CRITICAL: If LLM successfully created finding, skip old fallback logic and go directly to creating finding object
    if llm_finding_created:
        # LLM already generated comprehensive summary - skip old fallback logic
        # Just collect sources and create finding object
        logger.info(f"Agent {agent_id} skipping old fallback logic - using LLM-generated finding",
                   summary_length=len(summary) if summary else 0,
                   key_findings_count=len(key_findings) if key_findings else 0)
        
        # Extract notes for finding (filter metadata)
        important_notes = []
        for note in notes:
            if not note.summary or len(note.summary) < 100:
                continue
            summary_lower = note.summary.lower()
            is_metadata = any([
                "found" in summary_lower and "sources" in summary_lower and "query" in summary_lower,
                "search:" in note.title.lower() and len(note.summary) < 150,
                "key sources:" in summary_lower and len(note.summary) < 200,
            ])
            if not is_metadata:
                important_notes.append(note)
        
        # Collect all sources for finding (scraped pages + web search results)
        all_sources = []
        
        # Add scraped pages as sources
        for page in scraped_pages:
            if page.get("url"):
                all_sources.append({
                    "url": page.get("url", ""),
                    "title": page.get("title", ""),
                    "snippet": page.get("brief_info", "")  # Use brief_info for snippet
                })
        
        # Add web search results as sources (filtered)
        filtered_sources = []
        for src in sources:
            snippet = src.get("snippet", "").strip()
            if snippet and len(snippet) > 30:
                snippet_lower = snippet.lower()
                is_obvious_metadata = (
                    "found" in snippet_lower and "sources" in snippet_lower and "query" in snippet_lower
                )
                if not is_obvious_metadata:
                    filtered_sources.append(src)
        
        all_sources.extend(filtered_sources[:20])  # Limit to 20 web search sources
        
        # Skip to supervisor message generation and finding creation (continue from line 1868)
    else:
        # OLD FALLBACK LOGIC - Only use if LLM failed to create finding
        # Extract REAL findings from sources (snippets with actual information, not just titles)
        # CRITICAL: If snippet is missing but we have title and url, use title as finding basis
        real_findings_from_sources = []
        skipped_short = 0
        skipped_metadata = 0
        skipped_no_content = 0
        for src in sources:
            snippet = src.get("snippet", "").strip()
            title = src.get("title", "").strip()
            url = src.get("url", "").strip()
            
            # CRITICAL: If snippet is missing but we have title, use title as content
            # This handles cases where web_search didn't return snippets but we have scraped content
            if not snippet and title:
                # Use title as snippet if no snippet available (better than skipping)
                snippet = title
                logger.debug(f"Agent {agent_id} using title as snippet (no snippet available)",
                            title=title[:100],
                            url=url[:100] if url else "no url")
            
            # Skip if we have no content at all (no snippet and no title)
            if not snippet and not title:
                skipped_no_content += 1
                continue
            
            # Skip if snippet is too short (but allow if we have title)
            if snippet and len(snippet) < 30:
                # If we have a good title, use it even if snippet is short
                if title and len(title) > 20:
                    finding_text = f"{title}" + (f": {snippet}" if snippet else "")
                    if url:
                        finding_text += f" (Source: {url})"
                    real_findings_from_sources.append(finding_text)
                    continue
                skipped_short += 1
                continue
            
            # Skip if snippet is just metadata (contains "found", "sources", "query" etc.)
            # CRITICAL: Be less strict - only filter obvious metadata, not legitimate content
            snippet_lower = snippet.lower() if snippet else ""
            is_metadata = any([
                # Only filter if it's clearly metadata (all three words together)
                "found" in snippet_lower and "sources" in snippet_lower and "query" in snippet_lower,
                # Filter search: and query: only if they're at the start (likely metadata)
                snippet_lower.startswith("search:") or snippet_lower.startswith("query:"),
                # Multiple URLs (>2) might be metadata, but 1-2 URLs is normal content
                snippet_lower.count("http") > 2,  # More than 2 URLs = likely metadata
            ])
            
            if is_metadata:
                skipped_metadata += 1
                continue
            
            # Extract meaningful information
            if snippet and len(snippet) > 30:
                finding_text = f"{title}: {snippet[:250]}" if title else snippet[:250]
                if url and url not in finding_text:
                    finding_text += f" (Source: {url})"
                real_findings_from_sources.append(finding_text)
            elif title and len(title) > 20:
                # Use title if snippet is missing or too short
                finding_text = title
                if url:
                    finding_text += f" (Source: {url})"
                real_findings_from_sources.append(finding_text)
        
        # CRITICAL: Log filtering results to diagnose why findings might be empty
        logger.info(f"Agent {agent_id} findings extraction from sources",
                   total_sources=len(sources),
                   real_findings_count=len(real_findings_from_sources),
                   skipped_short=skipped_short,
                   skipped_metadata=skipped_metadata,
                   skipped_no_content=skipped_no_content,
                   note="If real_findings_count is 0, check why sources were filtered out")
        
        # Extract REAL findings from notes (only informative ones, not metadata)
        important_notes = []
        skipped_short_notes = 0
        skipped_metadata_notes = 0
        for note in notes:
            if not note.summary or len(note.summary) < 100:
                skipped_short_notes += 1
                continue
            
            # Skip metadata notes
            summary_lower = note.summary.lower()
            is_metadata = any([
                "found" in summary_lower and "sources" in summary_lower and "query" in summary_lower,
                "search:" in note.title.lower() and len(note.summary) < 150,
                "key sources:" in summary_lower and len(note.summary) < 200,
            ])
            
            if is_metadata:
                skipped_metadata_notes += 1
                continue
            
            important_notes.append(note)
        
        # CRITICAL: Log notes filtering results
        logger.info(f"Agent {agent_id} findings extraction from notes",
                   total_notes=len(notes),
                   important_notes_count=len(important_notes),
                   skipped_short=skipped_short_notes,
                   skipped_metadata=skipped_metadata_notes,
                   note="If important_notes_count is 0, check why notes were filtered out")
        
        # Build summary with ONLY real findings, NO metadata
        # CRITICAL: Create comprehensive summary with detailed information, not just links
        summary_parts = []
        
        # Add comprehensive findings from sources with full context
        if real_findings_from_sources:
            # CRITICAL: Use findings directly - they already contain title and snippet
            # The complex matching logic was causing issues - findings are already formatted correctly
            findings_text = "\n\n".join([f"• {f}" for f in real_findings_from_sources[:12]])
            
            # CRITICAL: Ensure findings_text is not empty before adding to summary
            if findings_text and findings_text.strip():
                findings_section = f"**Detailed Research Findings:**\n\n{findings_text}"
                summary_parts.append(findings_section)
                logger.info(f"Agent {agent_id} added findings to summary_parts",
                            findings_count=len(real_findings_from_sources[:12]),
                            findings_text_length=len(findings_text),
                            findings_text_preview=findings_text[:200] if findings_text else "EMPTY",
                            section_length=len(findings_section),
                            section_preview=findings_section[:300] if findings_section else "EMPTY",
                            note="Findings should now be in summary - VERIFY findings_text is not empty!")
            else:
                logger.error(f"Agent {agent_id} findings_text is EMPTY after join!",
                            real_findings_count=len(real_findings_from_sources),
                            sample_findings=real_findings_from_sources[:3] if real_findings_from_sources else [],
                            note="CRITICAL: findings_text is empty - this will cause empty summary!")
        else:
            logger.warning(f"Agent {agent_id} real_findings_from_sources is EMPTY - no findings to add to summary!",
                          total_sources=len(sources),
                          skipped_short=skipped_short if 'skipped_short' in locals() else 0,
                          skipped_metadata=skipped_metadata if 'skipped_metadata' in locals() else 0,
                          skipped_no_content=skipped_no_content if 'skipped_no_content' in locals() else 0,
                          note="CRITICAL: This will cause empty summary!")
        
        # Add comprehensive findings from notes with full context
        if important_notes:
            notes_text = "\n\n".join([
                f"**{note.title}:**\n{note.summary[:500]}{'...' if len(note.summary) > 500 else ''}" 
                for note in important_notes[:10]  # More notes for comprehensive summary
            ])
            summary_parts.append(f"\n**Important Discoveries:**\n\n{notes_text}")
        
        # CRITICAL: If no real findings from filtered sources/notes, but we have sources, use them directly
        # This prevents "no substantial findings" when filters are too strict
        direct_findings = []  # Initialize outside if block so it's available for key_findings
        if not summary_parts and sources:
            # Use sources directly if filters filtered everything out
            logger.warning(f"Agent {agent_id} no findings after filtering, using sources directly", 
                          sources_count=len(sources),
                          filtered_sources_count=len(real_findings_from_sources),
                          notes_count=len(notes),
                          filtered_notes_count=len(important_notes),
                          note="Filters may be too strict - using sources directly to ensure findings are extracted")
            
            # Extract findings directly from sources (less strict filtering)
            direct_skipped = 0
            for src in sources[:10]:  # Use first 10 sources
                snippet = src.get("snippet", "").strip()
                title = src.get("title", "").strip()
                url = src.get("url", "")
                
                if not snippet or len(snippet) <= 30:
                    direct_skipped += 1
                    continue
                
                # Only skip obvious metadata
                snippet_lower = snippet.lower()
                is_obvious_metadata = (
                    "found" in snippet_lower and "sources" in snippet_lower and "query" in snippet_lower
                )
                
                if is_obvious_metadata:
                    direct_skipped += 1
                    continue
                
                finding_text = f"{title}: {snippet[:400]}" if title else snippet[:400]
                if url:
                    finding_text += f" (Source: {url})"
                direct_findings.append(finding_text)
            
            logger.info(f"Agent {agent_id} direct findings extraction (fallback)",
                       sources_checked=min(10, len(sources)),
                       direct_findings_count=len(direct_findings),
                       direct_skipped=direct_skipped,
                       note="Fallback extraction with less strict filtering")
            
            if direct_findings:
                findings_text = "\n\n".join([f"• {f}" for f in direct_findings])
                summary_parts.append(f"**Research Findings:**\n\n{findings_text}")
            else:
                # CRITICAL: Even fallback failed - log sample snippets to diagnose
                sample_snippets = []
                for src in sources[:3]:
                    snippet = src.get("snippet", "").strip()
                    if snippet:
                        sample_snippets.append(f"Length: {len(snippet)}, Preview: {snippet[:100]}")
                
                logger.error(f"Agent {agent_id} FALLBACK EXTRACTION FAILED - NO FINDINGS",
                           task=current_task.title,
                           sources_checked=min(10, len(sources)),
                           sample_snippets=sample_snippets,
                           note="Even fallback extraction with less strict filtering failed - check source quality!")
        
        # If still no findings, indicate that research needs to go deeper
        if not summary_parts:
            # CRITICAL: Log detailed diagnostics when no findings extracted
            logger.error(f"Agent {agent_id} NO FINDINGS EXTRACTED - DIAGNOSTICS",
                       task=current_task.title,
                       total_sources=len(sources),
                       sources_with_snippet=sum(1 for s in sources if s.get("snippet", "").strip()),
                       sources_snippet_lengths=[len(s.get("snippet", "").strip()) for s in sources[:5] if s.get("snippet")],
                       real_findings_from_sources_count=len(real_findings_from_sources),
                       total_notes=len(notes),
                       important_notes_count=len(important_notes),
                       direct_findings_attempted=bool(sources),
                       note="CRITICAL: No findings extracted despite having sources/notes - check filtering logic!")
            
            summary_parts.append(f"Research completed on '{current_task.title}' but no substantial findings extracted. May need deeper investigation.")
        
        # Create comprehensive summary (NO metadata like "Found X sources")
        summary = "\n\n".join(summary_parts)
        
        # CRITICAL: Log summary creation to diagnose empty summaries
        logger.info(f"Agent {agent_id} summary creation",
                   summary_length=len(summary),
                   summary_parts_count=len(summary_parts),
                   summary_parts_preview=[part[:100] for part in summary_parts[:3]] if summary_parts else [],
                   summary_preview=summary[:500] if summary else "EMPTY",
                   real_findings_count=len(real_findings_from_sources),
                   direct_findings_count=len(direct_findings) if 'direct_findings' in locals() else 0,
                   important_notes_count=len(important_notes),
                   note="If summary is empty or only has header, check why summary_parts is empty")
        
        # CRITICAL: If summary is empty or only contains header, use fallback
        if not summary or len(summary.strip()) < 50 or (summary.count("**") > 0 and summary.count("\n") < 3):
            logger.error(f"Agent {agent_id} summary is EMPTY or only has header - using fallback!",
                        summary_length=len(summary),
                        summary_preview=summary[:200],
                        total_sources=len(sources),
                        note="CRITICAL: Summary is empty - will use sources directly as fallback")
            
            # Fallback: Use sources directly if summary is empty
            if sources:
                fallback_findings = []
                for src in sources[:10]:
                    snippet = src.get("snippet", "").strip()
                    title = src.get("title", "").strip()
                    url = src.get("url", "").strip()
                    
                    if snippet and len(snippet) > 30:
                        finding_text = f"{title}: {snippet[:300]}" if title else snippet[:300]
                        if url:
                            finding_text += f" (Source: {url})"
                        fallback_findings.append(finding_text)
                    elif title and len(title) > 20:
                        finding_text = title
                        if url:
                            finding_text += f" (Source: {url})"
                        fallback_findings.append(finding_text)
                
                if fallback_findings:
                    fallback_text = "\n\n".join([f"• {f}" for f in fallback_findings])
                    summary = f"**Detailed Research Findings:**\n\n{fallback_text}"
                    logger.info(f"Agent {agent_id} used fallback summary",
                               fallback_findings_count=len(fallback_findings),
                               fallback_summary_length=len(summary),
                               note="Fallback summary created from sources directly")
        
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
        
        # CRITICAL: If LLM didn't generate key_findings, extract them from collected data
        if not key_findings:
            # Extract from scraped summaries
            for page in relevant_scraped_summaries[:5]:
                if page.get("summary"):
                    # Extract first sentence or key point from summary
                    summary_text = page["summary"]
                    first_sentence = summary_text.split('.')[0] if '.' in summary_text else summary_text[:150]
                    key_findings.append(f"{page['title']}: {first_sentence}")
            
            # Extract from snippets
            for snippet_data in useful_snippets[:7]:
                if snippet_data.get("snippet"):
                    key_findings.append(f"{snippet_data['title']}: {snippet_data['snippet'][:150]}")
        
        # Collect all sources for finding (scraped pages + web search results)
        all_sources = []
        
        # Add scraped pages as sources
        for page in scraped_pages:
            if page.get("url"):
                all_sources.append({
                    "url": page.get("url", ""),
                    "title": page.get("title", ""),
                    "snippet": page.get("brief_info", "")  # Use brief_info for snippet
                })
        
        # Add web search results as sources (filtered)
        filtered_sources = []
        for src in sources:
            snippet = src.get("snippet", "").strip()
            if snippet and len(snippet) > 30:
                snippet_lower = snippet.lower()
                is_obvious_metadata = (
                    "found" in snippet_lower and "sources" in snippet_lower and "query" in snippet_lower
                )
                if not is_obvious_metadata:
                    filtered_sources.append(src)
        
        all_sources.extend(filtered_sources[:20])  # Limit to 20 web search sources
    
    # CRITICAL: Generate message to supervisor if task was continued (has supervisor_message)
    supervisor_message_response = ""
    if current_task and hasattr(current_task, "supervisor_message") and current_task.supervisor_message:
        # Agent should respond to supervisor's message explaining how they addressed concerns
        try:
            from src.models.schemas import FindingContent
            
            supervisor_response_prompt = f"""You completed a research task that was returned to you by the supervisor for improvement.

**SUPERVISOR'S ORIGINAL MESSAGE:**
{current_task.supervisor_message}

**YOUR RESEARCH RESULTS:**
Summary: {summary[:1000]}
Key Findings: {', '.join(key_findings[:5]) if key_findings else 'None'}

**INSTRUCTIONS:**
Write a message to the supervisor (2-4 sentences) explaining:
1. How you addressed their concerns and instructions
2. What additional information you found
3. How the finding now meets their requirements

This message will be included in the finding as a response to the supervisor's message."""
            
            # Use LLM to generate response (or create simple response if LLM fails)
            try:
                response_result = await llm.ainvoke([
                    {"role": "system", "content": "You are a researcher responding to supervisor feedback. Write a concise, professional message."},
                    {"role": "user", "content": supervisor_response_prompt}
                ])
                supervisor_message_response = response_result.content if hasattr(response_result, "content") else str(response_result)
            except:
                # Fallback: create simple response
                supervisor_message_response = f"I have addressed the supervisor's concerns and improved the research. The finding now includes the requested information and meets the requirements."
            
            logger.info(f"Agent {agent_id} generated supervisor message response",
                       response_length=len(supervisor_message_response),
                       task=current_task.title)
        except Exception as e:
            logger.warning(f"Agent {agent_id} failed to generate supervisor message response", error=str(e))
            supervisor_message_response = f"I have addressed the supervisor's concerns and improved the research based on their instructions."
    
    # Create finding - ONLY real information, NO metadata spam
    # CRITICAL: Include agent_id so supervisor queue can identify which agent completed
    finding = {
        "agent_id": agent_id,  # CRITICAL: Must be included for supervisor queue processing
        "topic": current_task.title,
        "summary": summary,  # LLM-generated comprehensive summary
        "key_findings": key_findings,  # LLM-generated or extracted key findings
        "sources": all_sources[:30],  # All sources (scraped + web search)
        "confidence": "high" if len(relevant_scraped_summaries) >= 3 else "medium",
        "notes": important_notes,  # Only informative notes, filtered
        "sources_count": len(all_sources),
        "notes_count": len(important_notes),
        "key_findings_count": len(key_findings),
        "scraped_pages_count": len(relevant_scraped_summaries),
        "supervisor_message": supervisor_message_response  # Response to supervisor's message (empty if first completion)
    }

    # Mark task as done FIRST, before emitting
    # CRITICAL: Agent can always mark its own task as done, even if supervisor tried to change status
    # The protection in update_agent_todo allows status change to "done" for in_progress tasks
    update_result = await agent_file_service.update_agent_todo(
        agent_id,
        current_task.title,
        status="done",
        note=f"Completed with {len(sources)} sources"
    )
    if not update_result:
        logger.warning(f"Agent {agent_id} failed to mark task as done - task may have been modified", task=current_task.title)
    else:
        # Reload todos to get updated status after marking as done
        try:
            agent_file = await agent_file_service.read_agent_file(agent_id)
            todos = agent_file.get("todos", [])
            pending_after = len([t for t in todos if t.status == "pending"])
            done_after = len([t for t in todos if t.status == "done"])
        except:
            pending_after = 0
            done_after = 0
        
        logger.info(f"Agent {agent_id} marked task as done", 
                   task=current_task.title,
                   pending_tasks_after=pending_after,
                   done_tasks_after=done_after,
                   note=f"Task marked as 'done'. Finding queued for supervisor review. GUARANTEE: After review: if chapter added (task stays 'done') → agent will pick next pending task ({pending_after} waiting). If task returned (task becomes 'in_progress') → agent will continue this task. Agent picks next task ONLY when chapter is added.")

    # Reload todos to get updated status
    agent_file = await agent_file_service.read_agent_file(agent_id)
    todos = agent_file.get("todos", [])

    # Emit updated todos with correct status
    if stream:
        # CRITICAL: Include all fields to match supervisor's format for consistency
        todos_dict = [
            {
                "title": t.title,
                "status": t.status,  # Should be "done" for completed task
                "objective": t.objective if hasattr(t, "objective") else "",
                "expected_output": t.expected_output if hasattr(t, "expected_output") else "",
                "note": t.note if hasattr(t, "note") else "",
                "url": t.url if hasattr(t, "url") else None
            }
            for t in todos
        ]
        stream.emit_agent_todo(agent_id, todos_dict)
        logger.info(f"Agent {agent_id} todos emitted to frontend", 
                   todos_count=len(todos_dict),
                   done_count=sum(1 for t in todos if t.status == "done"),
                   pending_count=sum(1 for t in todos if t.status == "pending"),
                   in_progress_count=sum(1 for t in todos if t.status == "in_progress"))

        # Final note
        # CRITICAL: Log summary before creating AgentNote to diagnose issues
        logger.info(f"Agent {agent_id} creating final note with summary",
                   summary_length=len(summary),
                   summary_preview=summary[:300] if summary else "EMPTY",
                   summary_ends_with=summary[-100:] if summary and len(summary) > 100 else summary,
                   note="Summary will be saved to AgentNote and written to file")
        
        final_note = AgentNote(
            title=f"Task complete: {current_task.title}",
            summary=summary,
            urls=[s.get("url") for s in sources[:5] if s.get("url")],
            tags=["task_complete"]
        )
        
        # CRITICAL: Verify summary was set correctly
        logger.info(f"Agent {agent_id} final_note created",
                   note_summary_length=len(final_note.summary) if final_note.summary else 0,
                   note_summary_preview=final_note.summary[:200] if final_note.summary else "EMPTY",
                   note="Verifying summary is in final_note before saving")
        # CRITICAL: Pass agent_file_service so note is added to agent's personal file
        # Also pass research_memory_service and session_id for vector search
        await agent_memory_service.save_agent_note(
            final_note, 
            agent_id, 
            agent_file_service=agent_file_service,
            research_memory_service=research_memory_service,
            session_id=session_id
        )
        
        # CRITICAL: Save finding to research_memory_service for vector search
        if research_memory_service and session_id:
            try:
                # Save finding summary with embedding
                finding_content = f"{summary}\n\nKey findings:\n" + "\n".join([f"- {kf}" for kf in key_findings[:10]])
                await research_memory_service.save_finding(
                    session_id=session_id,
                    agent_id=agent_id,
                    title=current_task.title,
                    content=finding_content,
                    metadata={
                        "sources_count": len(all_sources),
                        "key_findings_count": len(key_findings),
                        "confidence": finding.get("confidence", "medium"),
                        "sources": [s.get("url") for s in all_sources[:10] if s.get("url")],
                    }
                )
                logger.info(f"Agent {agent_id} saved finding to research_memory_service",
                           session_id=session_id,
                           title=current_task.title)
            except Exception as e:
                logger.warning(f"Agent {agent_id} failed to save finding to research_memory_service", error=str(e))

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

    # CRITICAL: Log finding details to verify all information is present
    logger.info(f"Agent {agent_id} completed task", 
               task=current_task.title, 
               sources=len(sources),
               finding_summary_length=len(summary),
               finding_key_findings_count=len(key_findings),
               finding_sources_count=len(all_sources),
               finding_supervisor_message=bool(supervisor_message_response),
               finding_keys=list(finding.keys()),
               note="Task completed, queuing for supervisor review - VERIFY all finding fields are present")

    # Signal supervisor queue - agents queue results asynchronously
    # CRITICAL: Pass complete finding object with ALL fields
    # CRITICAL: If task was returned for rework, this is a NEW finding (old one was deleted)
    task_was_returned = current_task and hasattr(current_task, "supervisor_message") and current_task.supervisor_message and getattr(current_task, "return_count", 0) > 0
    
    if supervisor_queue:
        await supervisor_queue.agent_completed_task(
            agent_id=agent_id,
            task_title=current_task.title,
            result=finding  # Complete finding object with all fields: summary, key_findings, sources, supervisor_message, etc.
        )
        queue_size = supervisor_queue.size()
        logger.info(f"Agent {agent_id} queued for supervisor review", 
                   task=current_task.title,
                   queue_size=queue_size,
                   finding_fields=list(finding.keys()),
                   finding_summary_preview=summary[:200] if summary else "EMPTY",
                   task_was_returned=task_was_returned,
                   return_count=getattr(current_task, "return_count", 0) if current_task else 0,
                   note=f"{'🔄 NEW finding created after rework - old finding was deleted. ' if task_was_returned else ''}Complete finding object added to supervisor queue - supervisor will receive all fields. After supervisor review: if chapter added → agent will pick next pending task. If task returned again → task becomes in_progress and agent will create another new finding.")
    else:
        logger.warning(f"Agent {agent_id} completed task but no supervisor_queue available", task=current_task.title)

    return finding

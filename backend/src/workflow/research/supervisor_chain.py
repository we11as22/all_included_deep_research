"""Supervisor chain - LLM chain with strict workflow instead of ReAct agent.

Workflow:
1. Finding validation from queue
2. Add chapter to draft_report (sources added automatically)
3. Agent progress review
4. Task management (limit: maximum 2 additions across all iterations)
5. Return agent to work
"""

import structlog
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.workflow.research.supervisor_agent import SupervisorToolsRegistry

logger = structlog.get_logger(__name__)


# ==================== Structured Output Models ====================

class FindingValidationResult(BaseModel):
    """Finding validation result."""
    reasoning: str = Field(
        description="Analysis of finding: does it match the task, quality, completeness of information"
    )
    is_valid: bool = Field(description="Is the finding valid for adding to draft_report")
    needs_rework: bool = Field(description="Does it need rework (maximum 1 time per task)")
    rework_instructions: Optional[str] = Field(
        default=None,
        description="Instructions for rework, if needs_rework=True"
    )


class ChapterContent(BaseModel):
    """Chapter content for adding to draft_report."""
    reasoning: str = Field(
        description="How the finding was reformulated considering summaries of existing chapters, what was added/changed"
    )
    chapter_title: str = Field(description="Chapter title (without '## Chapter N:')")
    content: str = Field(
        description="Chapter content (markdown, 1000-2500 words). Sources are added automatically - DO NOT include them!"
    )


class ProgressReviewResult(BaseModel):
    """Agent progress review result."""
    reasoning: str = Field(
        description="Progress analysis: statuses of all agent tasks, chapter summaries, identified gaps"
    )
    agents_status: List[Dict[str, Any]] = Field(
        description="Statuses of all agents: agent_id, pending_tasks, in_progress_tasks, done_tasks"
    )
    gaps_identified: List[str] = Field(description="Identified gaps in research")
    new_tasks_needed: List[Dict[str, Any]] = Field(
        description="New tasks to create (agent_id, title, objective, etc.)"
    )
    tasks_to_update: List[Dict[str, Any]] = Field(
        description="Tasks to update (only those that agents haven't started yet)"
    )
    can_add_tasks: bool = Field(
        description="Can new tasks be added (limit check: maximum 2 times across all iterations)"
    )


class TaskManagementResult(BaseModel):
    """Task management result."""
    reasoning: str = Field(description="Why these tasks were created/updated, how they cover gaps")
    tasks_created: int = Field(description="Number of tasks created")
    tasks_updated: int = Field(description="Number of tasks updated")
    task_addition_count: int = Field(description="Current task addition counter (for limit)")


# ==================== Supervisor Chain ====================

class SupervisorChain:
    """LLM цепочка для обработки файндингов с жестким workflow."""
    
    def __init__(
        self,
        llm: Any,
        context: Dict[str, Any],
    ):
        """Initialize chain.
        
        Args:
            llm: LLM instance for structured output
            context: Context with query, deep_search_result, clarification_context, chapter_summaries, etc.
        """
        self.llm = llm
        self.context = context
        self.agent_memory_service = context.get("agent_memory_service")
        self.agent_file_service = context.get("agent_file_service")
        self.session_id = context.get("session_id")
        self.session_factory = context.get("session_factory")
        
    async def validate_finding(
        self,
        finding: Dict[str, Any],
        agent_id: str,
    ) -> FindingValidationResult:
        """Step 1: Finding validation.
        
        Args:
            finding: Finding from agent
            agent_id: Agent ID
            
        Returns:
            Validation result
        """
        logger.info(
            "SUPERVISOR: Step 1 - Starting finding validation",
            agent_id=agent_id,
            finding_topic=finding.get("topic", "unknown"),
            finding_sources_count=finding.get("sources_count", 0),
            finding_confidence=finding.get("confidence", "unknown")
        )
        
        # Get agent task to check correspondence
        agent_file = await self.agent_file_service.read_agent_file(agent_id)
        todos = agent_file.get("todos", [])
        
        # Find task that corresponds to this finding
        finding_topic = finding.get("topic", "")
        task_for_finding = None
        for todo in todos:
            # Search task by title (finding.topic = task.title)
            if todo.title == finding_topic:
                task_for_finding = todo
                break
        
        if task_for_finding:
            return_count = getattr(task_for_finding, "return_count", 0)
            logger.info(
                "SUPERVISOR: Found matching task for finding",
                agent_id=agent_id,
                task_title=task_for_finding.title,
                task_status=task_for_finding.status,
                return_count=return_count
            )
        else:
            logger.warning(
                "SUPERVISOR: No matching task found for finding",
                agent_id=agent_id,
                finding_topic=finding_topic,
                available_tasks=[t.title for t in todos]
            )
        
        # Prompt for validation
        query = self.context.get("query", "")
        deep_search_result_raw = self.context.get("deep_search_result", "")
        # Обработать deep_search_result (может быть dict)
        if isinstance(deep_search_result_raw, dict):
            deep_search_result = deep_search_result_raw.get("value", "") if "value" in deep_search_result_raw else str(deep_search_result_raw)
        else:
            deep_search_result = deep_search_result_raw or ""
        clarification_context = self.context.get("clarification_context", "")
        
        task_description = ""
        if task_for_finding:
            return_count = getattr(task_for_finding, "return_count", 0)
            task_description = f"""
**Agent Task:**
- Title: {task_for_finding.title}
- Objective: {task_for_finding.objective}
- Expected Output: {task_for_finding.expected_output}
- Guidance: {task_for_finding.note if hasattr(task_for_finding, 'note') else ''}
- Status: {task_for_finding.status}
- Return count: {return_count}/1 (maximum 1 return per task)
"""
        
        # Get full finding summary and key findings without truncation
        finding_summary = finding.get('summary', '')
        key_findings = finding.get('key_findings', [])
        key_findings_text = ', '.join(key_findings) if key_findings else 'None'
        
        prompt = f"""Analyze the finding from the agent and determine if it matches the assigned task.

**Original User Query:** {query}

**Deep Search Context:**
{deep_search_result if deep_search_result else "No context available"}

**Clarification (if provided):**
{clarification_context if clarification_context else "No clarification"}

{task_description}

**Finding from Agent:**
- Topic: {finding.get('topic', 'Unknown')}
- Summary: {finding_summary}
- Key Findings: {key_findings_text}
- Sources: {finding.get('sources_count', 0)} sources
- Confidence: {finding.get('confidence', 'unknown')}

**Validation Criteria:**
1. Does the finding match the agent's assigned task?
2. Is there sufficient information and details?
3. Are there sources and facts?
4. Is the finding too superficial?

**Important:**
- If the finding doesn't match the task or is too superficial → needs_rework=True
- If the task was already returned for rework (return_count >= 1) → DO NOT return again, accept as-is
- If the finding is valid → is_valid=True, needs_rework=False

Return structured output with reasoning at the beginning."""
        
        logger.info(
            "SUPERVISOR: Calling LLM for finding validation",
            agent_id=agent_id,
            finding_topic=finding.get("topic", "unknown"),
            prompt_length=len(prompt)
        )
        
        result = await self.llm.with_structured_output(FindingValidationResult).ainvoke([
            {"role": "system", "content": "You are an expert at analyzing research findings. Analyze thoroughly and provide clear recommendations."},
            {"role": "user", "content": prompt}
        ])
        
        logger.info(
            "SUPERVISOR: Step 1 - Finding validation completed",
            agent_id=agent_id,
            is_valid=result.is_valid,
            needs_rework=result.needs_rework,
            has_rework_instructions=bool(result.rework_instructions),
            reasoning_preview=result.reasoning[:200] if result.reasoning else ""
        )
        
        return result
    
    async def write_chapter(
        self,
        finding: Dict[str, Any],
        validation_result: FindingValidationResult,
    ) -> Dict[str, Any]:
        """Step 2: Add chapter to draft_report.
        
        Args:
            finding: Finding from agent
            validation_result: Validation result
            
        Returns:
            Result of chapter addition
        """
        logger.info(
            "SUPERVISOR: Step 2 - Starting chapter writing",
            finding_topic=finding.get("topic", "unknown"),
            is_valid=validation_result.is_valid
        )
        
        if not validation_result.is_valid:
            logger.warning("SUPERVISOR: Skipping chapter writing - finding is not valid")
            return {"success": False, "reason": "finding_not_valid"}
        
        # Get summaries of existing chapters
        chapter_summaries = self.context.get("chapter_summaries", [])
        
        logger.info(
            "SUPERVISOR: Chapter writing context",
            existing_chapters_count=len(chapter_summaries),
            user_language=self.context.get("user_language", "English"),
            finding_sources_count=finding.get("sources_count", 0)
        )
        
        # Prompt for writing chapter
        query = self.context.get("query", "")
        deep_search_result_raw = self.context.get("deep_search_result", "")
        # Обработать deep_search_result (может быть dict)
        if isinstance(deep_search_result_raw, dict):
            deep_search_result = deep_search_result_raw.get("value", "") if "value" in deep_search_result_raw else str(deep_search_result_raw)
        else:
            deep_search_result = deep_search_result_raw or ""
        clarification_context = self.context.get("clarification_context", "")
        
        # Get user language for chapter writing
        user_language = self.context.get("user_language", "English")
        
        # Build chapter summaries text without truncation
        chapter_summaries_text = ""
        if chapter_summaries:
            summaries_parts = []
            for ch in chapter_summaries:
                if isinstance(ch, dict):
                    # Use full summary, not truncated
                    summary = ch.get('summary', '')
                    summaries_parts.append(
                        f"Chapter {ch.get('chapter_number', '?')}: {ch.get('chapter_title', 'Unknown')}\n"
                        f"Topic: {ch.get('topic', 'Unknown')}\n"
                        f"Summary: {summary}\n"
                    )
            chapter_summaries_text = "\n".join(summaries_parts)
        
        # Get full finding data without truncation
        finding_summary = finding.get('summary', '')
        finding_key_findings = finding.get('key_findings', [])
        key_findings_text = '\n'.join([f"- {kf}" for kf in finding_key_findings]) if finding_key_findings else "None"
        
        prompt = f"""Write a comprehensive, informative chapter for the draft_report based on the finding from the agent.

**CRITICAL: Write the chapter in {user_language} - the same language as the user's query.**

**Original User Query:** {query}

**Deep Search Context:**
{deep_search_result if deep_search_result else "No context available"}

**Clarification (if provided):**
{clarification_context if clarification_context else "No clarification"}

**Existing Chapters in draft_report:**
{chapter_summaries_text if chapter_summaries_text else "No existing chapters - this is the first chapter"}

**Finding from Agent:**
- Topic: {finding.get('topic', 'Unknown')}
- Summary: {finding_summary}
- Key Findings:
{key_findings_text}
- Sources: {finding.get('sources_count', 0)} sources

**Requirements:**
1. Reasoning at the beginning: explain how you reformulated the finding considering the chapter summaries, how it relates to other chapters, and how it contributes to answering the query

2. Chapter title: clear, descriptive chapter title (WITHOUT '## Chapter N:' - this is added automatically)

3. Content: full, comprehensive chapter content (1500-3000 words, markdown format)
   - Use ### for subsections (organize logically)
   - Use **bold** and *italic* for emphasis
   - Include ALL details, facts, data, examples, and insights from the finding
   - **CRITICAL: Be comprehensive and detailed** - don't summarize briefly, provide full information
   - Include context, background, explanations, implications
   - Use specific examples, data points, and concrete information from the finding
   - **MANDATORY: Reference other chapters when relevant** - use phrases like "As discussed in Chapter X..." or "Building on the analysis in Chapter Y..." to create connections
   - Adapt to existing chapters: avoid repetitions, build upon them, create logical flow
   - Ensure the chapter fully addresses its topic and contributes to answering the query
   - **CRITICAL: DO NOT include sources** - they are added automatically at the end of the chapter

**Chapter Quality Standards:**
- **Completeness**: The chapter must fully cover its topic - no superficial treatment
- **Depth**: Include detailed explanations, not just surface-level information
- **Integration**: Reference and connect to other chapters when relevant (use chapter summaries to identify connections)
- **Clarity**: Write clearly and structured, with logical flow
- **Information Density**: Include all relevant facts, data, and insights from the finding
- **Context**: Provide sufficient background and context for readers to understand

**Important:**
- Sources are added automatically from finding.sources - DO NOT write them in content!
- Use chapter summaries to identify which chapters to reference and how they relate
- Write comprehensively and in detail - this is a research report, not a summary
- **MANDATORY: Write in {user_language} - match the language of the user's query**
- When referencing other chapters, use their chapter numbers and titles from the summaries

Return structured output with reasoning at the beginning."""
        
        logger.info(
            "SUPERVISOR: Calling LLM for chapter writing",
            finding_topic=finding.get("topic", "unknown"),
            user_language=user_language,
            prompt_length=len(prompt),
            existing_chapters_count=len(chapter_summaries)
        )
        
        chapter_content = await self.llm.with_structured_output(ChapterContent).ainvoke([
            {"role": "system", "content": f"You are an expert at writing research chapters. Write comprehensively, structured, with context awareness. Always write in {user_language}."},
            {"role": "user", "content": prompt}
        ])
        
        logger.info(
            "SUPERVISOR: Chapter content generated",
            chapter_title=chapter_content.chapter_title,
            content_length=len(chapter_content.content),
            reasoning_length=len(chapter_content.reasoning) if chapter_content.reasoning else 0
        )
        
        # Call handler to add chapter (sources will be added automatically)
        from src.workflow.research.supervisor_agent import write_draft_report_handler
        
        # Ensure user_language is in context for write_draft_report_handler
        handler_context = self.context.copy()
        if "user_language" not in handler_context:
            handler_context["user_language"] = user_language
        
        result = await write_draft_report_handler(
            args={
                "content": chapter_content.content,
                "chapter_title": chapter_content.chapter_title,
                "finding": finding,  # Pass finding for automatic source addition
            },
            context=handler_context
        )
        
        # CRITICAL: Reload chapter_summaries from session_metadata after writing chapter
        # This ensures the next chapter will have access to the newly created chapter summary
        if result.get("success") and self.session_factory and self.session_id:
            try:
                from src.workflow.research.session.manager import SessionManager
                session_manager = SessionManager(self.session_factory)
                session_data = await session_manager.get_session(self.session_id)
                if session_data:
                    metadata = getattr(session_data, "session_metadata", None) or {}
                    if isinstance(metadata, dict):
                        updated_chapter_summaries = metadata.get("chapter_summaries", [])
                        self.context["chapter_summaries"] = updated_chapter_summaries
                        logger.info(
                            "Chapter summaries reloaded after writing chapter",
                            chapter_title=chapter_content.chapter_title,
                            total_summaries=len(updated_chapter_summaries)
                        )
            except Exception as e:
                logger.warning("Failed to reload chapter_summaries after writing chapter", error=str(e))
        
        logger.info(
            "SUPERVISOR: Step 2 - Chapter written successfully",
            chapter_title=chapter_content.chapter_title,
            chapter_number=result.get("chapter_number"),
            content_length=len(chapter_content.content),
            sources_added=result.get("sources_count", 0),
            chapter_summaries_count=len(self.context.get("chapter_summaries", [])),
            reasoning_preview=chapter_content.reasoning[:200] if chapter_content.reasoning else ""
        )
        
        return result
    
    async def review_progress(
        self,
        all_agents: List[str],
    ) -> ProgressReviewResult:
        """Step 3: Agent progress review.
        
        Args:
            all_agents: List of all agent_id
            
        Returns:
            Progress review result
        """
        logger.info(
            "SUPERVISOR: Step 3 - Starting progress review",
            agents_count=len(all_agents),
            agent_ids=all_agents
        )
        
        # Get statuses of all agents
        agents_status = []
        for agent_id in all_agents:
            try:
                agent_file = await self.agent_file_service.read_agent_file(agent_id)
                todos = agent_file.get("todos", [])
                pending = [t for t in todos if t.status == "pending"]
                in_progress = [t for t in todos if t.status == "in_progress"]
                done = [t for t in todos if t.status == "done"]
                
                agents_status.append({
                    "agent_id": agent_id,
                    "pending_tasks": len(pending),
                    "in_progress_tasks": len(in_progress),
                    "done_tasks": len(done),
                    "pending_titles": [t.title for t in pending[:3]],
                    "in_progress_titles": [t.title for t in in_progress],
                })
            except Exception as e:
                logger.warning(f"Failed to get status for agent {agent_id}", error=str(e))
        
        # Get chapter summaries
        chapter_summaries = self.context.get("chapter_summaries", [])
        
        # Check limit on task addition
        task_addition_count = self.context.get("task_addition_count", 0)
        can_add_tasks = task_addition_count < 2
        
        # Prompt for review
        query = self.context.get("query", "")
        deep_search_result_raw = self.context.get("deep_search_result", "")
        # Обработать deep_search_result (может быть dict)
        if isinstance(deep_search_result_raw, dict):
            deep_search_result = deep_search_result_raw.get("value", "") if "value" in deep_search_result_raw else str(deep_search_result_raw)
        else:
            deep_search_result = deep_search_result_raw or ""
        
        # Build chapter summaries text without truncation
        chapter_summaries_text = ""
        if chapter_summaries:
            summaries_parts = []
            for ch in chapter_summaries:
                if isinstance(ch, dict):
                    # Use full summary, not truncated
                    summary = ch.get('summary', '')
                    summaries_parts.append(
                        f"Chapter {ch.get('chapter_number', '?')}: {ch.get('chapter_title', 'Unknown')} "
                        f"(Topic: {ch.get('topic', 'Unknown')}, Summary: {summary})"
                    )
            chapter_summaries_text = "\n".join(summaries_parts)
        
        agents_status_text = "\n".join([
            f"- {s['agent_id']}: {s['pending_tasks']} pending, {s['in_progress_tasks']} in_progress, {s['done_tasks']} done"
            for s in agents_status
        ])
        
        prompt = f"""Review the progress of all agents and determine if new tasks are needed to ensure complete coverage of the research query.

**Original User Query:** {query}

**Deep Search Context:**
{deep_search_result if deep_search_result else "No context available"}

**Existing Chapters in draft_report:**
{chapter_summaries_text if chapter_summaries_text else "No chapters"}

**Agent Statuses:**
{agents_status_text}

**Task Addition Limit:** {task_addition_count}/2 (maximum 2 additions across all iterations)
**Can Add Tasks:** {can_add_tasks}

**Requirements:**
1. Reasoning at the beginning: 
   - Analyze current progress: what aspects of the query are covered? What's missing?
   - Compare existing chapters against the original query: are all aspects addressed?
   - Identify gaps: what critical aspects of the query are not yet covered or insufficiently covered?
   - Determine if new tasks are needed to ensure COMPLETE coverage of the query

2. Agents status: current statuses of all agents (pending/in_progress/done tasks)

3. Gaps identified: specific gaps in research coverage that prevent fully answering the query
   - What aspects of the query are missing or incomplete?
   - What topics need additional research?
   - What connections or perspectives are lacking?

4. New tasks needed: new tasks to create (if can_add_tasks=True)
   - **CRITICAL**: Only create tasks that fill identified gaps in coverage
   - Each task should address a specific missing aspect of the query
   - Tasks should be specific, actionable, and contribute unique value
   - Ensure tasks together with existing ones provide COMPLETE coverage of the query

5. Tasks to update: tasks to update (only those that agents haven't started yet - status 'pending')
   - Update if task description is unclear or needs refinement
   - Update if task doesn't align with identified gaps
   - DO NOT update tasks that are 'in_progress' or 'done'

6. Can add tasks: whether tasks can be added (limit check)

**Gap Analysis Guidelines:**
- Compare the original query against existing chapters: what's covered? What's missing?
- Consider all aspects: what, why, how, when, where, who, implications, challenges
- Identify both missing topics AND insufficiently covered topics
- Ensure the research will fully answer the query when all tasks are complete

**Important:**
- Can edit tasks with status 'pending' (agents haven't started yet)
- DO NOT edit tasks 'in_progress' or 'done'
- If limit reached (task_addition_count >= 2) → can_add_tasks=False
- **CRITICAL**: Only add tasks that fill genuine gaps - don't add redundant tasks
- Focus on ensuring COMPLETE coverage of the query

Return structured output with reasoning at the beginning."""
        
        logger.info(
            "SUPERVISOR: Calling LLM for progress review",
            agents_count=len(agents_status),
            chapter_summaries_count=len(chapter_summaries),
            task_addition_count=task_addition_count,
            can_add_tasks=can_add_tasks,
            prompt_length=len(prompt)
        )
        
        result = await self.llm.with_structured_output(ProgressReviewResult).ainvoke([
            {"role": "system", "content": "You are an expert at coordinating research teams. Analyze progress and identify gaps."},
            {"role": "user", "content": prompt}
        ])
        
        logger.info(
            "SUPERVISOR: Step 3 - Progress review completed",
            agents_count=len(agents_status),
            gaps_identified_count=len(result.gaps_identified),
            new_tasks_needed_count=len(result.new_tasks_needed),
            tasks_to_update_count=len(result.tasks_to_update),
            can_add_tasks=result.can_add_tasks,
            reasoning_preview=result.reasoning[:200] if result.reasoning else ""
        )
        
        return result
    
    async def manage_tasks(
        self,
        progress_review: ProgressReviewResult,
    ) -> TaskManagementResult:
        """Step 4: Task management (create/update).
        
        Args:
            progress_review: Progress review result
            
        Returns:
            Task management result
        """
        logger.info(
            "SUPERVISOR: Step 4 - Starting task management",
            can_add_tasks=progress_review.can_add_tasks,
            new_tasks_needed_count=len(progress_review.new_tasks_needed),
            tasks_to_update_count=len(progress_review.tasks_to_update),
            current_task_addition_count=self.context.get("task_addition_count", 0)
        )
        
        tasks_created = 0
        tasks_updated = 0
        task_addition_count = self.context.get("task_addition_count", 0)
        
        # Create new tasks (if possible)
        if progress_review.can_add_tasks and progress_review.new_tasks_needed:
            logger.info(
                "SUPERVISOR: Creating new tasks",
                tasks_to_create=len(progress_review.new_tasks_needed),
                task_addition_count_before=task_addition_count
            )
            from src.workflow.research.supervisor_agent import create_agent_todo_handler
            
            for task_data in progress_review.new_tasks_needed:
                try:
                    result = await create_agent_todo_handler(
                        args={
                            "agent_id": task_data.get("agent_id"),
                            "title": task_data.get("title"),
                            "objective": task_data.get("objective"),
                            "expected_output": task_data.get("expected_output", "Comprehensive findings"),
                            "priority": task_data.get("priority", "medium"),
                            "guidance": task_data.get("guidance", ""),
                            "reasoning": task_data.get("reasoning", ""),
                        },
                        context=self.context
                    )
                    if result.get("success"):
                        tasks_created += 1
                        task_addition_count += 1
                        logger.info(
                            "SUPERVISOR: Task created successfully",
                            agent_id=task_data.get("agent_id"),
                            title=task_data.get("title"),
                            tasks_created_so_far=tasks_created,
                            task_addition_count=task_addition_count
                        )
                    else:
                        logger.warning(
                            "SUPERVISOR: Failed to create task",
                            agent_id=task_data.get("agent_id"),
                            title=task_data.get("title"),
                            error=result.get("error")
                        )
                except Exception as e:
                    logger.error(
                        "SUPERVISOR: Exception while creating task",
                        error=str(e),
                        task_data=task_data
                    )
        
        # Update existing tasks (only pending)
        if progress_review.tasks_to_update:
            logger.info(
                "SUPERVISOR: Updating existing tasks",
                tasks_to_update_count=len(progress_review.tasks_to_update)
            )
            from src.workflow.research.supervisor_agent import update_agent_todo_handler
            
            for task_data in progress_review.tasks_to_update:
                try:
                    # Check that task is in pending status
                    agent_id = task_data.get("agent_id")
                    todo_title = task_data.get("todo_title")
                    
                    agent_file = await self.agent_file_service.read_agent_file(agent_id)
                    todos = agent_file.get("todos", [])
                    task = next((t for t in todos if t.title == todo_title), None)
                    
                    if task and task.status == "pending":
                        result = await update_agent_todo_handler(
                            args={
                                "agent_id": agent_id,
                                "todo_title": todo_title,
                                "status": task_data.get("status", ""),
                                "objective": task_data.get("objective", ""),
                                "expected_output": task_data.get("expected_output", ""),
                                "guidance": task_data.get("guidance", ""),
                                "priority": task_data.get("priority", ""),
                                "reasoning": task_data.get("reasoning", ""),
                            },
                            context=self.context
                        )
                        if result.get("success"):
                            tasks_updated += 1
                            logger.info(
                                "SUPERVISOR: Task updated successfully",
                                agent_id=agent_id,
                                title=todo_title,
                                tasks_updated_so_far=tasks_updated
                            )
                        else:
                            logger.warning(
                                "SUPERVISOR: Failed to update task",
                                agent_id=agent_id,
                                title=todo_title,
                                error=result.get("error")
                            )
                    else:
                        logger.warning(
                            "SUPERVISOR: Cannot update task - not in pending status",
                            agent_id=agent_id,
                            title=todo_title,
                            current_status=task.status if task else "not_found"
                        )
                except Exception as e:
                    logger.error(
                        "SUPERVISOR: Exception while updating task",
                        error=str(e),
                        task_data=task_data
                    )
        
        # Update counter in context
        self.context["task_addition_count"] = task_addition_count
        
        result = TaskManagementResult(
            reasoning=f"Created {tasks_created} tasks, updated {tasks_updated} tasks. Addition counter: {task_addition_count}/2",
            tasks_created=tasks_created,
            tasks_updated=tasks_updated,
            task_addition_count=task_addition_count
        )
        
        logger.info(
            "SUPERVISOR: Step 4 - Task management completed",
            tasks_created=tasks_created,
            tasks_updated=tasks_updated,
            task_addition_count=task_addition_count,
            reasoning_preview=result.reasoning[:200] if result.reasoning else ""
        )
        
        return result
    
    async def process_finding(
        self,
        finding: Dict[str, Any],
        agent_id: str,
        all_agents: List[str],
    ) -> Dict[str, Any]:
        """Process finding from queue (full workflow).
        
        Args:
            finding: Finding from agent
            agent_id: ID of agent that created the finding
            all_agents: List of all agent_id for progress review
            
        Returns:
            Processing result with agent return information
        """
        logger.info(
            "SUPERVISOR: Processing finding from queue - starting full workflow",
            agent_id=agent_id,
            finding_topic=finding.get("topic", "unknown"),
            finding_sources_count=finding.get("sources_count", 0),
            all_agents_count=len(all_agents)
        )
        
        # Step 1: Validation
        validation_result = await self.validate_finding(finding, agent_id)
        
        # If rework needed - return task
        if validation_result.needs_rework:
            logger.info(
                "SUPERVISOR: Finding needs rework - checking return count",
                agent_id=agent_id,
                finding_topic=finding.get("topic", "unknown")
            )
            from src.workflow.research.supervisor_agent import return_task_to_progress_handler
            
            # Проверить, не была ли уже возвращена
            agent_file = await self.agent_file_service.read_agent_file(agent_id)
            todos = agent_file.get("todos", [])
            task = next((t for t in todos if t.title == finding.get("topic", "")), None)
            
            if task:
                return_count = getattr(task, "return_count", 0)
                if return_count >= 1:
                    logger.warning(
                        "SUPERVISOR: Task already returned once, accepting finding as-is",
                        agent_id=agent_id,
                        task_title=task.title,
                        return_count=return_count,
                        note="Maximum 1 return per task - accepting finding"
                    )
                    # Accept as-is, continue to chapter writing
                    # (validation_result.is_valid will be True after this)
                else:
                    logger.info(
                        "SUPERVISOR: Returning task for rework",
                        agent_id=agent_id,
                        task_title=task.title,
                        return_count=return_count
                    )
                    # Вернуть на доработку
                    result = await return_task_to_progress_handler(
                        args={
                            "agent_id": agent_id,
                            "todo_title": task.title,
                            "motivation": validation_result.reasoning,
                            "instructions": validation_result.rework_instructions or "Improve the finding according to the feedback",
                        },
                        context=self.context
                    )
                    
                    if result.get("error"):
                        logger.error(
                            "Failed to return task to progress",
                            error=result.get("error"),
                            agent_id=agent_id,
                            task_title=task.title
                        )
                        # Continue to chapter writing even if return failed
                    else:
                        logger.info(
                            "SUPERVISOR: Task returned for rework successfully",
                            agent_id=agent_id,
                            task_title=task.title,
                            note="Agent will continue working on this task"
                        )
                        return {
                            "success": True,
                            "action": "rework",
                            "agent_id": agent_id,
                            "should_return_agent": True,  # Agent will continue working on this task
                        }
        
        # Step 2: Add chapter (only if valid)
        if validation_result.is_valid:
            chapter_result = await self.write_chapter(finding, validation_result)
            
            if not chapter_result.get("success"):
                logger.error(
                    "SUPERVISOR: Failed to write chapter",
                    result=chapter_result,
                    finding_topic=finding.get("topic", "unknown")
                )
                return {
                    "success": False,
                    "error": "chapter_write_failed",
                }
        else:
            logger.warning(
                "SUPERVISOR: Skipping chapter writing - finding is not valid",
                agent_id=agent_id,
                finding_topic=finding.get("topic", "unknown")
            )
        
        # Step 3: Progress review
        progress_review = await self.review_progress(all_agents)
        
        # Step 4: Task management
        task_management = await self.manage_tasks(progress_review)
        
        # Step 5: Return agent to work
        # Agent should continue working on next task
        agent_file = await self.agent_file_service.read_agent_file(agent_id)
        todos = agent_file.get("todos", [])
        pending_tasks = [t for t in todos if t.status == "pending"]
        in_progress_tasks = [t for t in todos if t.status == "in_progress"]
        
        should_return_agent = len(pending_tasks) > 0 or len(in_progress_tasks) > 0
        
        logger.info(
            "SUPERVISOR: Finding processing completed - full workflow finished",
            agent_id=agent_id,
            chapter_written=validation_result.is_valid,
            should_return_agent=should_return_agent,
            pending_tasks=len(pending_tasks),
            in_progress_tasks=len(in_progress_tasks),
            done_tasks=len([t for t in todos if t.status == "done"])
        )
        
        return {
            "success": True,
            "action": "chapter_added" if validation_result.is_valid else "skipped",
            "agent_id": agent_id,
            "should_return_agent": should_return_agent,
            "task_management": task_management.dict(),
        }


async def run_supervisor_chain(
    state: Dict[str, Any],
    llm: Any,
    stream: Any,
    supervisor_queue: Any = None,
) -> Dict[str, Any]:
    """Main function to run supervisor chain.
    
    Args:
        state: Research state
        llm: LLM instance
        stream: Stream for statuses
        supervisor_queue: Findings queue
        
    Returns:
        Processing result with should_continue and agents_to_return
    """
    # Get finding from queue (one at a time)
    current_finding = None
    current_agent_id = None
    
    logger.info(
        "SUPERVISOR: Starting supervisor chain",
        queue_size=supervisor_queue.size() if supervisor_queue else 0,
        session_id=state.get("session_id", "unknown")
    )
    
    if supervisor_queue and supervisor_queue.size() > 0:
        try:
            # Get first finding from queue (FIFO) - remove it from queue after processing
            event = await supervisor_queue.queue.get()
            current_finding = event.result
            current_agent_id = event.agent_id
            
            logger.info(
                "SUPERVISOR: Found finding in queue",
                agent_id=current_agent_id,
                finding_topic=current_finding.get("topic", "unknown") if current_finding else "unknown",
                queue_size_before=supervisor_queue.size() + 1,
                queue_size_after=supervisor_queue.size()
            )
        except Exception as e:
            logger.error("SUPERVISOR: Failed to get finding from queue", error=str(e))
    
    if not current_finding:
        logger.warning(
            "SUPERVISOR: No finding in queue to process",
            queue_size=supervisor_queue.size() if supervisor_queue else 0
        )
        return {
            "should_continue": True,  # Continue if there are tasks
            "agents_to_return": [],
        }
    
    # Prepare context
    query = state.get("original_query", state.get("query", ""))
    
    # Process deep_search_result (can be dict with "value" or string)
    deep_search_result_raw = state.get("deep_search_result", "")
    deep_search_result = ""
    if isinstance(deep_search_result_raw, dict):
        if "type" in deep_search_result_raw and deep_search_result_raw.get("type") == "override":
            deep_search_result = deep_search_result_raw.get("value", "")
        elif "value" in deep_search_result_raw:
            deep_search_result = deep_search_result_raw.get("value", "")
        else:
            deep_search_result = str(deep_search_result_raw)
    elif isinstance(deep_search_result_raw, str):
        deep_search_result = deep_search_result_raw
    else:
        deep_search_result = str(deep_search_result_raw) if deep_search_result_raw else ""
    
    clarification_answers = state.get("clarification_answers", "")
    
    # Get chapter_summaries from session metadata
    chapter_summaries = []
    if stream and hasattr(stream, "app_state"):
        session_factory = stream.app_state.get("session_factory")
        session_id = state.get("session_id")
        if session_factory and session_id:
            try:
                logger.info(
                    "SUPERVISOR: Loading chapter_summaries from session metadata",
                    session_id=session_id
                )
                from src.workflow.research.session.manager import SessionManager
                session_manager = SessionManager(session_factory)
                session_data = await session_manager.get_session(session_id)
                if session_data:
                    metadata = getattr(session_data, "session_metadata", None) or {}
                    if isinstance(metadata, dict):
                        chapter_summaries = metadata.get("chapter_summaries", [])
                        logger.info(
                            "SUPERVISOR: Chapter summaries loaded from session",
                            session_id=session_id,
                            chapter_summaries_count=len(chapter_summaries)
                        )
            except Exception as e:
                logger.warning("SUPERVISOR: Failed to get chapter_summaries", error=str(e))
    
    # Build clarification_context
    clarification_context = ""
    if clarification_answers and clarification_answers.strip():
        clarification_context = f"\n\n**USER CLARIFICATION ANSWERS:**\n{clarification_answers}\n"
    
    # Get all agents
    agent_file_service = stream.app_state.get("agent_file_service") if stream else None
    all_agents = []
    if agent_file_service:
        try:
            agent_files = await agent_file_service.file_manager.list_files("agents/agent_*.md")
            for file_path in agent_files:
                agent_id = file_path.replace("agents/", "").replace(".md", "")
                if agent_id.startswith("agent_") and agent_id != "supervisor":
                    all_agents.append(agent_id)
        except Exception as e:
            logger.warning("Failed to list agents", error=str(e))
    
    # Get user language from state
    user_language = state.get("user_language", "English")
    
    # Get settings from state or stream.app_state
    settings = state.get("settings")
    if not settings and stream and hasattr(stream, "app_state"):
        settings = stream.app_state.get("settings")
    
    # Create context for chain
    context = {
        "query": query,
        "deep_search_result": deep_search_result,
        "clarification_context": clarification_context,
        "chapter_summaries": chapter_summaries,
        "user_language": user_language,
        "agent_memory_service": stream.app_state.get("agent_memory_service") if stream else None,
        "agent_file_service": agent_file_service,
        "session_id": state.get("session_id"),
        "session_factory": stream.app_state.get("session_factory") if stream else None,
        "task_addition_count": state.get("task_addition_count", 0),
        "findings": state.get("findings", []),
        "current_finding": current_finding,
        "settings": settings,  # CRITICAL: Include settings for create_agent_todo_handler
    }
    
    logger.info(
        "SUPERVISOR: Context prepared, creating chain",
        context_keys=list(context.keys()),
        chapter_summaries_count=len(chapter_summaries),
        all_agents_count=len(all_agents),
        task_addition_count=context.get("task_addition_count", 0),
        has_deep_search_result=bool(deep_search_result),
        has_clarification_context=bool(clarification_context),
        user_language=user_language
    )
    
    # Create chain and process finding
    chain = SupervisorChain(llm, context)
    result = await chain.process_finding(current_finding, current_agent_id, all_agents)
    
    logger.info(
        "SUPERVISOR: Finding processing completed",
        agent_id=current_agent_id,
        result_success=result.get("success"),
        should_return_agent=result.get("should_return_agent", False),
        action=result.get("action", "unknown")
    )
    
    # Mark event as processed (remove from queue)
    if supervisor_queue:
        try:
            supervisor_queue.queue.task_done()
            logger.info(
                "SUPERVISOR: Finding processed and removed from queue",
                agent_id=current_agent_id,
                queue_size_remaining=supervisor_queue.size()
            )
        except Exception as e:
            logger.warning("SUPERVISOR: Failed to mark queue item as done", error=str(e))
    
    # Update task_addition_count in state
    if "task_management" in result and result["task_management"]:
        task_management = result["task_management"]
        if isinstance(task_management, dict):
            task_addition_count = task_management.get("task_addition_count", 0)
        else:
            # If this is TaskManagementResult object
            task_addition_count = getattr(task_management, "task_addition_count", 0)
        state["task_addition_count"] = task_addition_count
        logger.info(
            "SUPERVISOR: Updated task_addition_count in state",
            task_addition_count=task_addition_count,
            previous_count=state.get("task_addition_count", 0)
        )
    
    # Determine if should continue
    # Check if agents still have tasks
    should_continue = True
    agents_to_return = []
    
    if result.get("should_return_agent"):
        agents_to_return.append(current_agent_id)
        logger.info(
            "SUPERVISOR: Agent marked for return to work",
            agent_id=current_agent_id,
            reason=result.get("action", "unknown"),
            agents_to_return=agents_to_return
        )
    else:
        logger.info(
            "SUPERVISOR: Agent NOT marked for return",
            agent_id=current_agent_id,
            reason=result.get("action", "unknown"),
            should_return_agent=result.get("should_return_agent", False)
        )
    
    # CRITICAL: Check completion condition
    # Only check if the agent whose finding was processed has remaining tasks
    # Other agents work independently and don't need to be checked here
    if agent_file_service:
        logger.info(
            "SUPERVISOR: Checking completion condition",
            current_agent_id=current_agent_id,
            agents_to_return=agents_to_return
        )
        try:
            # Check if the current agent (whose finding was processed) has remaining tasks
            current_agent_has_tasks = False
            if current_agent_id:
                agent_file = await agent_file_service.read_agent_file(current_agent_id)
                todos = agent_file.get("todos", [])
                pending = [t for t in todos if t.status == "pending"]
                in_progress = [t for t in todos if t.status == "in_progress"]
                current_agent_has_tasks = len(pending) > 0 or len(in_progress) > 0
                logger.info(
                    "SUPERVISOR: Current agent task status",
                    agent_id=current_agent_id,
                    pending_count=len(pending),
                    in_progress_count=len(in_progress),
                    has_tasks=current_agent_has_tasks
                )
            
            # Check if ANY agents have remaining tasks (for completion condition)
            has_pending_tasks = False
            if current_agent_has_tasks:
                has_pending_tasks = True
            else:
                # Check all agents to see if research is complete
                for agent_id in all_agents:
                    agent_file = await agent_file_service.read_agent_file(agent_id)
                    todos = agent_file.get("todos", [])
                    pending = [t for t in todos if t.status == "pending"]
                    in_progress = [t for t in todos if t.status == "in_progress"]
                    if pending or in_progress:
                        has_pending_tasks = True
                        break
            
            if not has_pending_tasks:
                # Check that all findings have chapters
                # Get all findings from state
                findings = state.get("findings", [])
                chapter_summaries_count = len(chapter_summaries)
                findings_count = len(findings)
                
                logger.info(
                    "SUPERVISOR: Checking completion condition",
                    has_pending_tasks=has_pending_tasks,
                    findings_count=findings_count,
                    chapter_summaries_count=chapter_summaries_count
                )
                
                # If number of chapters is less than number of findings, not all chapters are written yet
                if chapter_summaries_count < findings_count:
                    logger.warning(
                        "SUPERVISOR: Not all findings have chapters yet - continuing",
                        findings_count=findings_count,
                        chapters_count=chapter_summaries_count,
                        note="Research will continue until all findings are added as chapters"
                    )
                    should_continue = True  # Continue to process remaining findings
                else:
                    should_continue = False
                    logger.info(
                        "SUPERVISOR: All tasks completed and all chapters written - research finished",
                        findings_count=findings_count,
                        chapters_count=chapter_summaries_count,
                        note="Research completion condition met"
                    )
            else:
                logger.info(
                    "SUPERVISOR: Agents still have tasks - continuing research",
                    has_pending_tasks=has_pending_tasks,
                    current_agent_id=current_agent_id,
                    current_agent_has_tasks=current_agent_has_tasks
                )
                should_continue = True  # Continue because agents have tasks
        except Exception as e:
            logger.warning("SUPERVISOR: Failed to check pending tasks", error=str(e))
    
    final_result = {
        "should_continue": should_continue,
        "agents_to_return": agents_to_return,
        "replanning_needed": False,
        "task_management": result.get("task_management"),
    }
    
    logger.info(
        "SUPERVISOR: Final result prepared",
        should_continue=should_continue,
        agents_to_return=agents_to_return,
        agents_to_return_count=len(agents_to_return),
        has_task_management=bool(result.get("task_management"))
    )
    
    return final_result

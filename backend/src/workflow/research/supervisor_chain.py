"""Supervisor chain - LLM chain with strict workflow instead of ReAct agent.

Workflow:
1. Finding validation from queue
2. Add chapter to draft_report (sources added automatically)
3. Agent progress review
4. Task management (limit: maximum 2 additions across all iterations)
5. Return agent to work
"""

import asyncio
import structlog
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict

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


class NewTaskRequired(BaseModel):
    """New task to create - strict model to ensure all fields are valid.
    
    CRITICAL: All fields are REQUIRED and validated. None, empty strings, or placeholders are FORBIDDEN.
    Structured output will enforce these constraints - LLM cannot return invalid values.
    """
    
    model_config = {
        "json_schema_extra": {
            "required": ["agent_id", "title", "objective", "expected_output"],
            "examples": [
                {
                    "agent_id": "agent_1",
                    "title": "Research historical development of quantum computing",
                    "objective": "Investigate the evolution of quantum computing from theoretical foundations to current implementations, including key milestones, breakthroughs, and technological transformations",
                    "expected_output": "Comprehensive timeline and analysis of quantum computing development with key dates, technologies, and impact",
                    "guidance": "Focus on both theoretical and practical aspects",
                    "priority": "high"
                }
            ]
        }
    }
    
    agent_id: str = Field(
        description="Agent ID to assign task to. MUST be exactly one of: 'agent_1', 'agent_2', 'agent_3'. NO OTHER VALUES ALLOWED. This field is REQUIRED and CANNOT be None, empty, or missing.",
        min_length=6,
        max_length=7,
        examples=["agent_1", "agent_2", "agent_3"]
    )
    title: str = Field(
        description="Task title. REQUIRED non-empty string, MUST be at least 10 characters. MUST be a meaningful, specific, descriptive task title describing what needs to be researched. CANNOT be None, empty string, placeholder, or generic text. Examples: 'Research historical development of X', 'Analyze current trends in Y', 'Investigate impact of Z'. Bad examples: '', 'Task', 'Research', None.",
        min_length=10,
        examples=["Research historical development of quantum computing", "Analyze current trends in AI regulation"]
    )
    objective: str = Field(
        description="Task objective. REQUIRED non-empty string, MUST be at least 20 characters. MUST be a clear, detailed, specific description of what the agent should accomplish. CANNOT be None, empty string, placeholder, or generic text. Must explain WHAT needs to be researched and WHY it's important. Examples: 'Investigate the evolution of X from origins to present, including milestones and transformations', 'Analyze current state of Y, focusing on key factors and implications'. Bad examples: '', 'Research X', 'Find information', None.",
        min_length=20,
        examples=["Investigate the evolution of quantum computing from theoretical foundations to current implementations, including key milestones and technological transformations"]
    )
    expected_output: str = Field(
        description="Expected output description. REQUIRED non-empty string, MUST be at least 10 characters. Description of what the agent should produce as a result. CANNOT be None, empty string, or placeholder. Examples: 'Comprehensive timeline and analysis of X development', 'Detailed report on Y with key findings and recommendations'. Bad examples: '', 'Report', None.",
        min_length=10,
        examples=["Comprehensive timeline and analysis of quantum computing development with key dates, technologies, and impact"]
    )
    guidance: str = Field(
        description="Specific guidance on how to approach this task. Optional string (can be empty but should be provided if helpful). If empty, use empty string '', not None.",
        default=""
    )
    priority: str = Field(
        description="Task priority. Optional string, default 'medium'. Must be one of: 'high', 'medium', 'low'. Default is 'medium'.",
        default="medium"
    )
    
    @field_validator('agent_id')
    @classmethod
    def validate_agent_id(cls, v: str) -> str:
        """Validate agent_id is one of the allowed values."""
        if not v or not isinstance(v, str):
            raise ValueError(f"agent_id must be a non-empty string, got {type(v).__name__}: {v}")
        v = v.strip()
        if v not in ["agent_1", "agent_2", "agent_3"]:
            raise ValueError(f"agent_id must be exactly 'agent_1', 'agent_2', or 'agent_3', got '{v}'")
        return v
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate title is non-empty and meaningful."""
        if not v or not isinstance(v, str):
            raise ValueError(f"title must be a non-empty string, got {type(v).__name__}: {v}")
        v = v.strip()
        if len(v) < 10:
            raise ValueError(f"title must be at least 10 characters, got {len(v)} characters: '{v[:50]}...'")
        # Check for single-word placeholders (too generic)
        if len(v.split()) == 1 and v.lower() in ["task", "todo", "research", "investigate", "analyze", "study"]:
            raise ValueError(f"title must be descriptive and specific, not a single generic word. Got: '{v}'")
        return v
    
    @field_validator('objective')
    @classmethod
    def validate_objective(cls, v: str) -> str:
        """Validate objective is non-empty and detailed."""
        if not v or not isinstance(v, str):
            raise ValueError(f"objective must be a non-empty string, got {type(v).__name__}: {v}")
        v = v.strip()
        if len(v) < 20:
            raise ValueError(f"objective must be at least 20 characters, got {len(v)} characters: '{v[:50]}...'")
        # Check for too-short objectives (likely placeholders)
        if len(v.split()) < 3:
            raise ValueError(f"objective must be detailed and specific, not too short. Got: '{v}' (only {len(v.split())} words)")
        return v
    
    @field_validator('expected_output')
    @classmethod
    def validate_expected_output(cls, v: str) -> str:
        """Validate expected_output is non-empty."""
        if not v or not isinstance(v, str):
            raise ValueError(f"expected_output must be a non-empty string, got {type(v).__name__}: {v}")
        v = v.strip()
        if len(v) < 10:
            raise ValueError(f"expected_output must be at least 10 characters, got {len(v)} characters: '{v[:50]}...'")
        return v


class AgentStatus(BaseModel):
    """Agent status information."""
    model_config = ConfigDict(extra='forbid')  # Azure requires additionalProperties: false
    
    agent_id: str = Field(description="Agent ID (e.g., 'agent_1', 'agent_2', 'agent_3')")
    pending_tasks: int = Field(description="Number of pending tasks")
    in_progress_tasks: int = Field(description="Number of in-progress tasks")
    done_tasks: int = Field(description="Number of completed tasks")


class TaskToUpdate(BaseModel):
    """Task to update - strict model for structured output compatibility with Azure."""
    model_config = ConfigDict(extra='forbid')  # Azure requires additionalProperties: false
    
    agent_id: str = Field(description="Agent ID (e.g., 'agent_1', 'agent_2', 'agent_3')")
    todo_title: str = Field(description="Title of the task to update")
    status: str = Field(default="", description="Updated status (if empty, keep current)")
    objective: str = Field(default="", description="Updated objective (if empty, keep current)")
    expected_output: str = Field(default="", description="Updated expected_output (if empty, keep current)")
    guidance: str = Field(default="", description="Updated guidance (if empty, keep current)")
    priority: str = Field(default="", description="Updated priority: high/medium/low (if empty, keep current)")
    reasoning: str = Field(default="", description="Updated reasoning for why this task is needed (if empty, keep current)")


class ProgressReviewResult(BaseModel):
    """Agent progress review result."""
    reasoning: str = Field(
        description="Progress analysis: statuses of all agent tasks, chapter summaries, identified gaps"
    )
    agents_status: List[AgentStatus] = Field(
        description="Statuses of all agents: agent_id, pending_tasks, in_progress_tasks, done_tasks"
    )
    gaps_identified: List[str] = Field(description="Identified gaps in research")
    new_tasks_needed: List[NewTaskRequired] = Field(
        description="New tasks to create. Each task MUST have: agent_id (REQUIRED, must be 'agent_1', 'agent_2', or 'agent_3'), title (REQUIRED, non-empty string, at least 10 characters, descriptive task title), objective (REQUIRED, non-empty string, at least 20 characters, clear description of what to achieve), expected_output (REQUIRED, non-empty string), guidance (optional string), priority (optional, default 'medium'). agent_id, title, and objective are MANDATORY and CANNOT be None, empty, or missing. title and objective must be meaningful strings, not placeholders."
    )
    tasks_to_update: List[TaskToUpdate] = Field(
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
        # CRITICAL: agent_memory_service and agent_file_service are NOT stored in context
        # They are accessed via stream.app_state when needed (to avoid storing in context)
        self.session_id = context.get("session_id")
        self.session_factory = context.get("session_factory")
        self.stream = context.get("stream")  # Store stream to access app_state
    
    def _get_agent_file_service(self):
        """Get agent_file_service from stream.app_state."""
        if self.stream and hasattr(self.stream, "app_state"):
            return self.stream.app_state.get("agent_file_service")
        return None
    
    def _get_agent_memory_service(self):
        """Get agent_memory_service from stream.app_state."""
        if self.stream and hasattr(self.stream, "app_state"):
            return self.stream.app_state.get("agent_memory_service")
        return None
        
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
        agent_file_service = self._get_agent_file_service()
        if not agent_file_service:
            logger.error("SUPERVISOR: agent_file_service not available", agent_id=agent_id)
            raise ValueError("agent_file_service not available in stream.app_state")
        agent_file = await agent_file_service.read_agent_file(agent_id)
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
        
        # Get current date and time for context
        from datetime import datetime
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        prompt = f"""Analyze the finding from the agent and determine if it matches the assigned task.

**CURRENT DATE AND TIME:**
- Date: {current_date}
- Full datetime: {current_datetime}

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
        validation_concerns: str = None,  # Optional: concerns about finding quality if accepted despite issues
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
        
        # Get current date and time for context
        from datetime import datetime
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Build validation concerns section separately to avoid backslash in f-string
        validation_concerns_section = ""
        if validation_concerns:
            validation_concerns_section = (
                f"**CRITICAL: VALIDATION CONCERNS**\n\n"
                f"This finding was accepted despite some quality concerns. The supervisor identified the following issues:\n"
                f"{validation_concerns}\n\n"
                f"**MANDATORY**: You MUST include a clear note in the chapter about these limitations or concerns. "
                f"Add a section (e.g., '### Limitations' or '### Note on Data Quality') that explicitly mentions these issues "
                f"so readers are aware of potential gaps or problems in the research.\n\n"
                f"Write the chapter normally, but be transparent about any limitations or concerns identified by the supervisor."
            )
        
        prompt = f"""Write a comprehensive, informative chapter for the draft_report based on the finding from the agent.

**CURRENT DATE AND TIME:**
- Date: {current_date}
- Full datetime: {current_datetime}

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
{validation_concerns_section}

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
        
        # CRITICAL: Retry logic with fallback for API errors (403, 429, etc.)
        chapter_content = None
        max_retries = 3
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                chapter_content = await self.llm.with_structured_output(ChapterContent).ainvoke([
                    {"role": "system", "content": f"You are an expert at writing research chapters. Write comprehensively, structured, with context awareness. Always write in {user_language}."},
                    {"role": "user", "content": prompt}
                ])
                break  # Success - exit retry loop
            except Exception as e:
                error_str = str(e)
                is_api_error = (
                    "403" in error_str or 
                    "429" in error_str or 
                    "Blocked by Google" in error_str or
                    "PermissionDeniedError" in str(type(e)) or
                    "RateLimitError" in str(type(e))
                )
                
                if is_api_error and attempt < max_retries - 1:
                    logger.warning(
                        "SUPERVISOR: API error during chapter writing, retrying",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        error=error_str,
                        finding_topic=finding.get("topic", "unknown"),
                        note="Will retry after delay. If all retries fail, will create fallback chapter."
                    )
                    await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    # Last attempt failed or non-retryable error - raise to trigger fallback
                    logger.error(
                        "SUPERVISOR: Failed to generate chapter content after retries",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        error=error_str,
                        finding_topic=finding.get("topic", "unknown"),
                        exc_info=True,
                        note="Will create fallback chapter from finding data directly"
                    )
                    raise  # Re-raise to trigger fallback logic below
        
        # CRITICAL: Fallback - if LLM call failed, create chapter directly from finding
        if chapter_content is None:
            logger.warning(
                "SUPERVISOR: Creating fallback chapter from finding data (LLM unavailable)",
                finding_topic=finding.get("topic", "unknown"),
                note="CRITICAL: Chapter will be created directly from finding to ensure it's written despite API errors"
            )
            
            # Create simple chapter from finding data
            # CRITICAL: ChapterContent is defined in this file, not in models.py - use it directly
            chapter_title = finding.get("topic", "Research Finding")
            finding_summary = finding.get("summary", "")
            key_findings = finding.get("key_findings", [])
            
            # Get user language for fallback chapter
            user_language = self.context.get("user_language", "English")
            
            # Build simple chapter content
            content_parts = []
            if finding_summary:
                content_parts.append(f"## {chapter_title}\n\n")
                content_parts.append(f"{finding_summary}\n\n")
            
            if key_findings:
                # Use appropriate language for section title
                section_title = "### Основные выводы\n\n" if user_language == "Russian" else "### Key Findings\n\n"
                content_parts.append(section_title)
                for kf in key_findings:
                    content_parts.append(f"- {kf}\n")
                content_parts.append("\n")
            
            if validation_concerns:
                note_title = "### Примечание\n\n" if user_language == "Russian" else "### Note\n\n"
                content_parts.append(note_title)
                content_parts.append(f"*{validation_concerns}*\n\n")
            
            # Add note about API unavailability
            api_note = (
                "\n\n*Примечание: Эта глава была создана автоматически из данных исследования из-за временной недоступности сервиса генерации текста.*\n"
                if user_language == "Russian"
                else "\n\n*Note: This chapter was automatically created from research data due to temporary unavailability of text generation service.*\n"
            )
            content_parts.append(api_note)
            
            fallback_content = "".join(content_parts) if content_parts else f"## {chapter_title}\n\n{finding_summary}\n\n"
            
            # Create ChapterContent object manually
            chapter_content = ChapterContent(
                reasoning=f"Chapter created directly from finding data due to LLM API unavailability. Finding topic: {chapter_title}",
                chapter_title=chapter_title,
                content=fallback_content
            )
            
            logger.info(
                "SUPERVISOR: Fallback chapter content created",
                chapter_title=chapter_title,
                content_length=len(fallback_content),
                note="Fallback chapter ensures finding is added to draft_report despite API errors"
            )
        
        logger.info(
            "SUPERVISOR: Chapter content generated",
            chapter_title=chapter_content.chapter_title,
            content_length=len(chapter_content.content),
            reasoning_length=len(chapter_content.reasoning) if chapter_content.reasoning else 0
        )
        
        # Call handler to add chapter (sources will be added automatically)
        from src.workflow.research.supervisor_agent import write_draft_report_handler
        
        # Ensure user_language and agent_memory_service are in context for write_draft_report_handler
        handler_context = self.context.copy()
        if "user_language" not in handler_context:
            handler_context["user_language"] = user_language
        # CRITICAL: Get agent_memory_service from stream.app_state and add to context
        agent_memory_service = self._get_agent_memory_service()
        if agent_memory_service:
            handler_context["agent_memory_service"] = agent_memory_service
        # Also add stream to context for fallback access
        if self.stream:
            handler_context["stream"] = self.stream
        
        # CRITICAL: Retry logic for write_draft_report_handler in case of errors
        result = None
        max_handler_retries = 2
        
        for handler_attempt in range(max_handler_retries):
            try:
                result = await write_draft_report_handler(
                    args={
                        "content": chapter_content.content,
                        "chapter_title": chapter_content.chapter_title,
                        "finding": finding,  # Pass finding for automatic source addition
                    },
                    context=handler_context
                )
                
                if result.get("success"):
                    break  # Success - exit retry loop
                elif handler_attempt < max_handler_retries - 1:
                    logger.warning(
                        "SUPERVISOR: Chapter write handler failed, retrying",
                        attempt=handler_attempt + 1,
                        max_retries=max_handler_retries,
                        result=result,
                        note="Will retry writing chapter"
                    )
                    await asyncio.sleep(1.0)
                    continue
                else:
                    logger.error(
                        "SUPERVISOR: Chapter write handler failed after retries",
                        attempt=handler_attempt + 1,
                        max_retries=max_handler_retries,
                        result=result,
                        note="CRITICAL: Chapter was NOT written to draft_report despite retries"
                    )
            except Exception as e:
                if handler_attempt < max_handler_retries - 1:
                    logger.warning(
                        "SUPERVISOR: Exception in chapter write handler, retrying",
                        attempt=handler_attempt + 1,
                        max_retries=max_handler_retries,
                        error=str(e),
                        note="Will retry writing chapter"
                    )
                    await asyncio.sleep(1.0)
                    continue
                else:
                    logger.error(
                        "SUPERVISOR: Exception in chapter write handler after retries",
                        attempt=handler_attempt + 1,
                        max_retries=max_handler_retries,
                        error=str(e),
                        exc_info=True,
                        note="CRITICAL: Chapter was NOT written to draft_report due to exception"
                    )
                    result = {"success": False, "error": str(e)}
        
        # CRITICAL: If result is None or failed, create error result
        if not result or not result.get("success"):
            logger.error(
                "SUPERVISOR: Chapter write failed - creating error result",
                result=result,
                chapter_title=chapter_content.chapter_title,
                note="CRITICAL: Chapter was NOT written to draft_report. This finding will not appear in the report."
            )
            result = result or {"success": False, "error": "Chapter write handler failed"}
        
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
        
        if result.get("success"):
            logger.info(
                "SUPERVISOR: Step 2 - Chapter written successfully",
                chapter_title=chapter_content.chapter_title,
                chapter_number=result.get("chapter_number"),
                content_length=len(chapter_content.content),
                sources_added=result.get("sources_count", 0),
                chapter_summaries_count=len(self.context.get("chapter_summaries", [])),
                reasoning_preview=chapter_content.reasoning[:200] if chapter_content.reasoning else ""
            )
        else:
            logger.error(
                "SUPERVISOR: Step 2 - Chapter write FAILED",
                chapter_title=chapter_content.chapter_title,
                result=result,
                note="CRITICAL: Chapter was NOT written to draft_report. Finding will not appear in report."
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
                agent_file_service = self._get_agent_file_service()
                if not agent_file_service:
                    logger.error("SUPERVISOR: agent_file_service not available", agent_id=agent_id)
                    continue
                agent_file = await agent_file_service.read_agent_file(agent_id)
                todos = agent_file.get("todos", [])
                pending = [t for t in todos if t.status == "pending"]
                in_progress = [t for t in todos if t.status == "in_progress"]
                done = [t for t in todos if t.status == "done"]
                
                agents_status.append(AgentStatus(
                    agent_id=agent_id,
                    pending_tasks=len(pending),
                    in_progress_tasks=len(in_progress),
                    done_tasks=len(done),
                ))
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
            f"- {s.agent_id}: {s.pending_tasks} pending, {s.in_progress_tasks} in_progress, {s.done_tasks} done"
            for s in agents_status
        ])
        
        # Get current date and time for context
        from datetime import datetime
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        prompt = f"""Review the progress of all agents and determine if new tasks are needed to ensure complete coverage of the research query.

**CURRENT DATE AND TIME:**
- Date: {current_date}
- Full datetime: {current_datetime}

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
   - **MANDATORY**: Each task MUST include "agent_id" field (MUST be exactly "agent_1", "agent_2", or "agent_3")
   - **MANDATORY**: Use existing agents (agent_1, agent_2, agent_3) - do NOT create new agent IDs
   - **MANDATORY**: Assign tasks to agents that have fewer tasks or are better suited for the topic
   - **CRITICAL - STRUCTURED OUTPUT VALIDATION**: Each task MUST have ALL of the following fields with valid values. These are ENFORCED by Pydantic validation - invalid values will cause errors:
     * "agent_id": REQUIRED string, MUST be exactly "agent_1", "agent_2", or "agent_3" (no other values, no None, no empty)
     * "title": REQUIRED non-empty string, MUST be at least 10 characters, MUST be a descriptive, meaningful task title (NOT a placeholder like "Research" or "Task", NOT generic, NOT empty, NOT None). Must be specific and descriptive.
     * "objective": REQUIRED non-empty string, MUST be at least 20 characters, MUST be a clear, detailed description of what to achieve (NOT a placeholder like "Find info" or "Research X", NOT generic, NOT empty, NOT None). Must explain WHAT and WHY.
     * "expected_output": REQUIRED non-empty string, MUST be at least 10 characters, description of expected result (NOT empty, NOT None, NOT placeholder)
     * "guidance": Optional string (can be empty string "" but should be provided if helpful)
     * "priority": Optional string (default "medium", can be "high", "medium", or "low")
   - **FORBIDDEN - WILL CAUSE VALIDATION ERRORS**: title, objective, expected_output, and agent_id CANNOT be None, empty strings, missing, or placeholders. These are REQUIRED fields with minimum length requirements enforced by structured output validation.
   - **VALIDATION**: Structured output uses Pydantic models that automatically validate all fields. If you provide None, empty strings, or placeholders, the structured output will FAIL.
   - **EXAMPLES OF VALID TASKS** (use these as templates):
     * ✅ VALID: agent_id="agent_1", title="Research historical development of quantum computing", objective="Investigate the evolution of quantum computing from theoretical foundations to current implementations, including key milestones, breakthroughs, and technological transformations", expected_output="Comprehensive timeline and analysis of quantum computing development with key dates, technologies, and impact"
     * ✅ VALID: agent_id="agent_2", title="Analyze current trends in AI regulation", objective="Examine recent developments in AI regulation across major jurisdictions, including policy changes, enforcement actions, and industry responses", expected_output="Detailed report on AI regulation trends with analysis of key policies and their implications"
     * ❌ INVALID: agent_id=None, title="", objective="", expected_output="" (ALL INVALID - structured output will reject)
     * ❌ INVALID: agent_id="agent_1", title="Research", objective="Find info", expected_output="Report" (TOO GENERIC - structured output will reject)

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

Return structured output with reasoning at the beginning.

**CRITICAL FOR NEW TASKS:**
- When creating new tasks in "new_tasks_needed", you MUST provide ALL required fields:
  * agent_id: MUST be exactly "agent_1", "agent_2", or "agent_3" (no other values)
  * title: MUST be a non-empty string with at least 10 characters, MUST be descriptive and meaningful (NOT a placeholder, NOT generic, NOT empty, NOT None)
  * objective: MUST be a non-empty string with at least 20 characters, MUST be detailed and specific (NOT a placeholder, NOT generic, NOT empty, NOT None)
  * expected_output: MUST be a non-empty string with at least 10 characters
  * guidance: Optional string (can be empty)
  * priority: Optional string (default "medium")
- If you cannot provide valid values for title and objective, DO NOT create the task
- Every field in new_tasks_needed is validated by Pydantic - invalid values will cause errors"""
        
        logger.info(
            "SUPERVISOR: Calling LLM for progress review",
            agents_count=len(agents_status),
            chapter_summaries_count=len(chapter_summaries),
            task_addition_count=task_addition_count,
            can_add_tasks=can_add_tasks,
            prompt_length=len(prompt)
        )
        
        # CRITICAL: Use structured output with NO token limits - LLM must generate complete, valid responses
        # Structured output enforces Pydantic validation - invalid values will be rejected
        system_message = """You are an expert at coordinating research teams. Analyze progress and identify gaps.

CRITICAL RULES FOR NEW TASKS (new_tasks_needed field):
1. agent_id: MUST be exactly "agent_1", "agent_2", or "agent_3" (no other values, no None, no empty)
2. title: MUST be a non-empty string with at least 10 characters, MUST be descriptive and meaningful (NOT placeholder, NOT generic, NOT empty, NOT None)
3. objective: MUST be a non-empty string with at least 20 characters, MUST be detailed and specific (NOT placeholder, NOT generic, NOT empty, NOT None)
4. expected_output: MUST be a non-empty string with at least 10 characters (NOT empty, NOT None)

VALIDATION: Structured output uses Pydantic models that ENFORCE these rules. If you provide invalid values (None, empty strings, placeholders), the structured output will FAIL and you will need to retry.

EXAMPLES:
✅ VALID: {"agent_id": "agent_1", "title": "Research historical development of quantum computing", "objective": "Investigate the evolution of quantum computing from theoretical foundations to current implementations, including key milestones and transformations", "expected_output": "Comprehensive timeline and analysis"}
❌ INVALID: {"agent_id": None, "title": "", "objective": "", "expected_output": ""}
❌ INVALID: {"agent_id": "agent_1", "title": "Research", "objective": "Find info", "expected_output": "Report"}

If you cannot provide valid values for ALL required fields, DO NOT create the task."""
        
        # CRITICAL: Retry logic with fallback for API errors (403, 429, etc.)
        result = None
        max_retries = 3
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                if attempt == 0:
                    # First attempt
                    result = await self.llm.with_structured_output(ProgressReviewResult).ainvoke([
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ])
                else:
                    # Retry with even more explicit instructions
                    retry_system = system_message + "\n\nRETRY: The previous attempt failed validation. You MUST provide valid, non-empty strings for ALL required fields. None, empty strings, or placeholders are FORBIDDEN and will cause validation errors."
                    retry_prompt = prompt + "\n\n**CRITICAL REMINDER - VALIDATION FAILED:** When creating new tasks, EVERY field (agent_id, title, objective, expected_output) MUST be a valid, non-empty string. None, empty strings, or placeholders are FORBIDDEN and will cause structured output validation to fail. Use the examples above as a guide."
                    result = await self.llm.with_structured_output(ProgressReviewResult).ainvoke([
                        {"role": "system", "content": retry_system},
                        {"role": "user", "content": retry_prompt}
                    ])
                break  # Success - exit retry loop
            except Exception as e:
                error_str = str(e)
                is_api_error = (
                    "403" in error_str or 
                    "429" in error_str or 
                    "Blocked by Google" in error_str or
                    "PermissionDeniedError" in str(type(e)) or
                    "RateLimitError" in str(type(e))
                )
                
                if is_api_error and attempt < max_retries - 1:
                    logger.warning(
                        "SUPERVISOR: API error during progress review, retrying",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        error=error_str,
                        note="Will retry after delay. If all retries fail, will use fallback (no new tasks)."
                    )
                    await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                elif attempt < max_retries - 1:
                    # Validation error - retry with more explicit instructions
                    logger.warning(
                        "SUPERVISOR: LLM call failed for progress review - structured output validation failed, retrying",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        error=error_str,
                        note="This indicates invalid structured output from LLM (likely None/empty values). Retrying with even more explicit instructions."
                    )
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    # Last attempt failed - use fallback
                    logger.error(
                        "SUPERVISOR: Failed to generate progress review after retries - using fallback",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        error=error_str,
                        exc_info=True,
                        note="Will use fallback ProgressReviewResult (no new tasks, continue research)"
                    )
                    # Create fallback result
                    # CRITICAL: ProgressReviewResult is defined in this file, not in models.py - use it directly
                    result = ProgressReviewResult(
                        reasoning="Progress review failed due to API error - using fallback. Research will continue with existing tasks.",
                        agents_status=[],
                        gaps_identified=[],
                        new_tasks_needed=[],
                        tasks_to_update=[],
                        can_add_tasks=False
                    )
                    break
        
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
                    # CRITICAL: task_data is now NewTaskRequired (Pydantic model), so all fields are validated
                    # Convert to dict for handler
                    if isinstance(task_data, NewTaskRequired):
                        task_dict = task_data.model_dump()
                    elif isinstance(task_data, dict):
                        task_dict = task_data
                    else:
                        logger.error(
                            "SUPERVISOR: Invalid task_data type",
                            task_data_type=type(task_data).__name__,
                            note="Task data must be NewTaskRequired or dict. Skipping this task."
                        )
                        continue
                    
                    # Extract fields (already validated by Pydantic if NewTaskRequired)
                    agent_id = task_dict.get("agent_id")
                    title = task_dict.get("title")
                    objective = task_dict.get("objective")
                    expected_output = task_dict.get("expected_output", "Comprehensive findings")
                    priority = task_dict.get("priority", "medium")
                    guidance = task_dict.get("guidance", "")
                    
                    # Final validation (should not be needed if Pydantic validation worked, but double-check)
                    if not agent_id or not isinstance(agent_id, str) or not agent_id.strip():
                        logger.error(
                            "SUPERVISOR: Cannot create task - agent_id is missing or invalid",
                            agent_id=agent_id,
                            note="Task data must include valid agent_id field. Skipping this task."
                        )
                        continue
                    
                    if not title or not isinstance(title, str) or not title.strip() or len(title.strip()) < 10:
                        logger.error(
                            "SUPERVISOR: Cannot create task - title is missing or invalid",
                            agent_id=agent_id,
                            title=title,
                            title_length=len(title) if title else 0,
                            note="Task data must include non-empty title field with at least 10 characters. Skipping this task."
                        )
                        continue
                    
                    if not objective or not isinstance(objective, str) or not objective.strip() or len(objective.strip()) < 20:
                        logger.error(
                            "SUPERVISOR: Cannot create task - objective is missing or invalid",
                            agent_id=agent_id,
                            title=title,
                            objective=objective,
                            objective_length=len(objective) if objective else 0,
                            note="Task data must include non-empty objective field with at least 20 characters. Skipping this task."
                        )
                        continue
                    
                    # Validate agent_id format
                    if not agent_id.startswith("agent_") or agent_id not in ["agent_1", "agent_2", "agent_3"]:
                        logger.error(
                            "SUPERVISOR: Invalid agent_id format",
                            agent_id=agent_id,
                            task_title=title,
                            note="agent_id must be 'agent_1', 'agent_2', or 'agent_3'. Skipping this task."
                        )
                        continue
                    
                    result = await create_agent_todo_handler(
                        args={
                            "agent_id": agent_id,
                            "title": title.strip(),
                            "objective": objective.strip(),
                            "expected_output": expected_output.strip() if expected_output else "Comprehensive findings",
                            "priority": priority if priority in ["high", "medium", "low"] else "medium",
                            "guidance": guidance.strip() if guidance else "",
                            "reasoning": "",
                        },
                        context=self.context
                    )
                    if result.get("success"):
                        tasks_created += 1
                        task_addition_count += 1
                        logger.info(
                            "SUPERVISOR: Task created successfully",
                            agent_id=agent_id,
                            title=title,
                            tasks_created_so_far=tasks_created,
                            task_addition_count=task_addition_count
                        )
                    else:
                        logger.warning(
                            "SUPERVISOR: Failed to create task",
                            agent_id=agent_id,
                            title=title,
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
                    # task_data is now TaskToUpdate model, not dict
                    agent_id = task_data.agent_id
                    todo_title = task_data.todo_title
                    
                    agent_file_service = self._get_agent_file_service()
                    if not agent_file_service:
                        logger.error("SUPERVISOR: agent_file_service not available", agent_id=agent_id)
                        continue
                    agent_file = await agent_file_service.read_agent_file(agent_id)
                    todos = agent_file.get("todos", [])
                    task = next((t for t in todos if t.title == todo_title), None)
                    
                    if task and task.status == "pending":
                        result = await update_agent_todo_handler(
                            args={
                                "agent_id": agent_id,
                                "todo_title": todo_title,
                                "status": task_data.status or "",
                                "objective": task_data.objective or "",
                                "expected_output": task_data.expected_output or "",
                                "guidance": task_data.guidance or "",
                                "priority": task_data.priority or "",
                                "reasoning": task_data.reasoning or "",
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
            finding_summary_length=len(finding.get("summary", "")),
            finding_sources_count=finding.get("sources_count", 0),
            finding_key_findings_count=finding.get("key_findings_count", 0),
            all_agents_count=len(all_agents),
            note="CRITICAL: This finding will be validated and written to draft_report as a chapter if valid. Topic MUST match completed task title."
        )
        
        # Step 1: Validation
        validation_result = await self.validate_finding(finding, agent_id)
        
        # CRITICAL: Handle all validation scenarios
        # 1. needs_rework=True → return task for rework (if return_count < 1)
        # 2. needs_rework=True AND return_count >= 1 → accept as-is (force is_valid=True)
        # 3. needs_rework=False AND is_valid=False → check return_count:
        #    - If return_count >= 1 → accept as-is (force is_valid=True) - already returned once
        #    - If return_count == 0 → this shouldn't happen, but accept as-is to avoid blocking
        # 4. needs_rework=False AND is_valid=True → accept and write chapter
        
        # First, check if task was already returned (to handle case 3)
        agent_file_service = self._get_agent_file_service()
        if not agent_file_service:
            logger.error("SUPERVISOR: agent_file_service not available", agent_id=agent_id)
            raise ValueError("agent_file_service not available in stream.app_state")
        agent_file = await agent_file_service.read_agent_file(agent_id)
        todos = agent_file.get("todos", [])
        task = next((t for t in todos if t.title == finding.get("topic", "")), None)
        return_count = getattr(task, "return_count", 0) if task else 0
        
        # Handle case: needs_rework=False AND is_valid=False
        # If task was already returned once, accept as-is (force is_valid=True)
        validation_concerns = None  # Store concerns to pass to write_chapter
        if not validation_result.needs_rework and not validation_result.is_valid:
            if return_count >= 1:
                logger.warning(
                    "SUPERVISOR: Finding is invalid but task was already returned once - accepting as-is",
                    agent_id=agent_id,
                    task_title=task.title if task else "unknown",
                    return_count=return_count,
                    note="Task was already returned for rework once. Accepting finding as-is and writing chapter with explicit note about issues."
                )
                # Accept as-is, force is_valid=True so chapter will be written
                # Store validation concerns to add explicit note in chapter
                validation_concerns = validation_result.reasoning or "The finding does not fully meet quality standards, but the task was already returned for rework once. This chapter includes the available information despite identified limitations."
                validation_result.is_valid = True
                validation_result.needs_rework = False
            else:
                # This shouldn't happen - LLM should set needs_rework=True if finding is invalid
                # But handle it gracefully: treat as needs_rework=True
                logger.warning(
                    "SUPERVISOR: Finding is invalid but needs_rework=False - treating as needs_rework=True",
                    agent_id=agent_id,
                    task_title=task.title if task else "unknown",
                    return_count=return_count,
                    note="LLM returned is_valid=False but needs_rework=False. This is unusual - treating as needs_rework=True."
                )
                validation_result.needs_rework = True
        
        # If rework needed - return task
        # CRITICAL: If needs_rework=True, we return the task and DO NOT write chapter
        # Chapter is written ONLY when finding is accepted (needs_rework=False AND is_valid=True)
        if validation_result.needs_rework:
            logger.info(
                "SUPERVISOR: Finding needs rework - checking return count",
                agent_id=agent_id,
                finding_topic=finding.get("topic", "unknown")
            )
            from src.workflow.research.supervisor_agent import return_task_to_progress_handler
            
            # Task already loaded above
            if task:
                return_count = getattr(task, "return_count", 0)
                if return_count >= 1:
                    logger.warning(
                        "SUPERVISOR: Task already returned once, accepting finding as-is",
                        agent_id=agent_id,
                        task_title=task.title,
                        return_count=return_count,
                        note="Maximum 1 return per task - accepting finding and writing chapter with explicit note about issues"
                    )
                    # Accept as-is, continue to chapter writing
                    # Store validation concerns to add explicit note in chapter
                    validation_concerns = validation_result.reasoning or "The finding still has some quality concerns, but the task was already returned for rework once. This chapter includes the available information despite identified limitations."
                    # Mark as valid so chapter will be written
                    validation_result.is_valid = True
                    validation_result.needs_rework = False
                else:
                    logger.info(
                        "SUPERVISOR: Returning task for rework - chapter will NOT be written",
                        agent_id=agent_id,
                        task_title=task.title,
                        return_count=return_count,
                        note="CRITICAL: Task returned for rework - chapter writing is SKIPPED. Agent will continue working on this task."
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
                            task_title=task.title,
                            note="Will continue to chapter writing even if return failed"
                        )
                        # Continue to chapter writing even if return failed
                        # Mark as valid so chapter will be written
                        validation_result.is_valid = True
                        validation_result.needs_rework = False
                    else:
                        logger.info(
                            "SUPERVISOR: Task returned for rework successfully - skipping chapter writing",
                            agent_id=agent_id,
                            task_title=task.title,
                            note="CRITICAL: Task returned for rework - chapter is NOT written. Agent will continue working on this task."
                        )
                        # CRITICAL: Return early - do NOT write chapter when task is returned for rework
                        # Chapter is written ONLY when finding is accepted (not returned for rework)
                        # Create empty task_management result for consistency
                        # CRITICAL: TaskManagementResult is defined in this file, not in models.py - use it directly
                        empty_task_management = TaskManagementResult(
                            reasoning="Task management skipped - task returned for rework",
                            tasks_created=0,
                            tasks_updated=0,
                            task_addition_count=self.context.get("task_addition_count", 0)
                        )
                        return {
                            "success": True,
                            "action": "rework",
                            "agent_id": agent_id,
                            "should_return_agent": True,  # Agent will continue working on this task
                            "chapter_written": False,  # CRITICAL: Chapter is NOT written when task is returned
                            "task_management": empty_task_management,  # Include for consistency
                        }
        
        # Step 2: Add chapter (only if valid AND not returned for rework)
        # CRITICAL: Chapter is written ONLY when finding is accepted (is_valid=True AND needs_rework=False)
        # CRITICAL: Chapter writing MUST happen before task management - even if task management fails, chapter should be written
        # CRITICAL: Even if API errors occur, fallback chapter MUST be written to ensure finding appears in report
        chapter_written = False
        chapter_result = None  # CRITICAL: Store chapter_result for later use
        if validation_result.is_valid and not validation_result.needs_rework:
            # Pass validation_concerns if finding was accepted despite issues
            logger.info(
                "SUPERVISOR: Finding is valid and accepted - will write chapter",
                agent_id=agent_id,
                finding_topic=finding.get("topic", "unknown"),
                is_valid=validation_result.is_valid,
                needs_rework=validation_result.needs_rework,
                note="Finding passed validation - proceeding to write chapter"
            )
            try:
                chapter_result = await self.write_chapter(finding, validation_result, validation_concerns=validation_concerns)
                # CRITICAL: Store chapter_result for later use in return value
                
                if not chapter_result.get("success"):
                    logger.error(
                        "SUPERVISOR: Failed to write chapter",
                        result=chapter_result,
                        finding_topic=finding.get("topic", "unknown"),
                        agent_id=agent_id,
                        chapter_result_keys=list(chapter_result.keys()) if isinstance(chapter_result, dict) else None,
                        note="CRITICAL: Chapter write failed - finding will NOT appear in draft_report. This is a critical error - finding is lost."
                    )
                    # CRITICAL: Don't return early - continue to task management even if chapter write failed
                    # This ensures other findings can still be processed
                    # But mark as not written
                    chapter_written = False
                else:
                    chapter_written = True
                    logger.info(
                        "SUPERVISOR: Chapter written successfully",
                        agent_id=agent_id,
                        finding_topic=finding.get("topic", "unknown"),
                        chapter_number=chapter_result.get("chapter_number"),
                        chapter_title=chapter_result.get("chapter_title", "unknown"),
                        note="Chapter successfully added to draft_report. Will continue to task management."
                    )
            except Exception as e:
                error_str = str(e)
                is_api_error = (
                    "403" in error_str or 
                    "429" in error_str or 
                    "Blocked by Google" in error_str or
                    "PermissionDeniedError" in str(type(e)) or
                    "RateLimitError" in str(type(e))
                )
                
                if is_api_error:
                    # API error - fallback chapter should have been created in write_chapter
                    # But if it still failed, log critical error
                    logger.error(
                        "SUPERVISOR: Exception while writing chapter (API error)",
                        error=error_str,
                        finding_topic=finding.get("topic", "unknown"),
                        agent_id=agent_id,
                        exc_info=True,
                        note="CRITICAL: API error during chapter write. Fallback chapter should have been created, but write_chapter still failed. Finding may be lost."
                    )
                else:
                    logger.error(
                        "SUPERVISOR: Exception while writing chapter (non-API error)",
                        error=error_str,
                        finding_topic=finding.get("topic", "unknown"),
                        agent_id=agent_id,
                        exc_info=True,
                        note="CRITICAL: Non-API exception during chapter write. Chapter was NOT written. Will continue to task management to avoid blocking other findings."
                    )
                chapter_written = False
        else:
            logger.warning(
                "SUPERVISOR: Skipping chapter writing - finding is not valid or was returned for rework",
                agent_id=agent_id,
                finding_topic=finding.get("topic", "unknown"),
                is_valid=validation_result.is_valid,
                needs_rework=validation_result.needs_rework,
                validation_reasoning=validation_result.reasoning[:200] if hasattr(validation_result, 'reasoning') and validation_result.reasoning else None,
                note="Chapter is written ONLY when finding is accepted (is_valid=True AND needs_rework=False). This finding will NOT appear in draft_report as a chapter."
            )
        
        # Step 3: Progress review
        # CRITICAL: Wrap in try-except to ensure chapter is written even if review fails
        progress_review = None
        try:
            progress_review = await self.review_progress(all_agents)
        except Exception as e:
            logger.error(
                "SUPERVISOR: Progress review failed - continuing without task management",
                error=str(e),
                agent_id=agent_id,
                finding_topic=finding.get("topic", "unknown"),
                exc_info=True,
                note="CRITICAL: Chapter was already written. Progress review failure will not prevent finding processing completion."
            )
            # Create empty progress review to continue
            # CRITICAL: ProgressReviewResult is defined in this file, not in models.py - use it directly
            progress_review = ProgressReviewResult(
                reasoning="Progress review failed - skipped due to error",
                agents_status=[],
                gaps_identified=[],
                new_tasks_needed=[],
                tasks_to_update=[],
                can_add_tasks=False
            )
        
        # Step 4: Task management
        # CRITICAL: Wrap in try-except to ensure chapter is written even if task management fails
        task_management = None
        try:
            if progress_review:
                task_management = await self.manage_tasks(progress_review)
            else:
                # Create empty task management result
                # CRITICAL: TaskManagementResult is defined in this file, not in models.py - use it directly
                task_management = TaskManagementResult(
                    reasoning="Task management skipped - progress review failed",
                    tasks_created=0,
                    tasks_updated=0,
                    task_addition_count=self.context.get("task_addition_count", 0)
                )
        except Exception as e:
            logger.error(
                "SUPERVISOR: Task management failed - continuing without task creation",
                error=str(e),
                agent_id=agent_id,
                finding_topic=finding.get("topic", "unknown"),
                exc_info=True,
                note="CRITICAL: Chapter was already written. Task management failure (e.g., None in title/objective) will not prevent finding processing completion."
            )
            # Create empty task management result to continue
            # CRITICAL: TaskManagementResult is defined in this file, not in models.py - use it directly
            task_management = TaskManagementResult(
                reasoning="Task management failed - skipped due to error",
                tasks_created=0,
                tasks_updated=0,
                task_addition_count=self.context.get("task_addition_count", 0)
            )
        
        # Step 5: Return agent to work
        # Agent should continue working on next task
        agent_file_service = self._get_agent_file_service()
        if not agent_file_service:
            logger.error("SUPERVISOR: agent_file_service not available", agent_id=agent_id)
            raise ValueError("agent_file_service not available in stream.app_state")
        agent_file = await agent_file_service.read_agent_file(agent_id)
        todos = agent_file.get("todos", [])
        pending_tasks = [t for t in todos if t.status == "pending"]
        in_progress_tasks = [t for t in todos if t.status == "in_progress"]
        
        should_return_agent = len(pending_tasks) > 0 or len(in_progress_tasks) > 0
        
        # CRITICAL: chapter_written is already set in Step 2 above
        
        # CRITICAL: Get chapter info from write_chapter result
        chapter_number = None
        chapter_title = None
        if chapter_written and chapter_result and isinstance(chapter_result, dict):
            chapter_number = chapter_result.get("chapter_number")
            chapter_title = chapter_result.get("chapter_title")
        
        logger.info(
            "SUPERVISOR: Finding processing completed - full workflow finished",
            agent_id=agent_id,
            finding_topic=finding.get("topic", "unknown"),
            chapter_written=chapter_written,
            chapter_number=chapter_number,
            chapter_title=chapter_title,
            is_valid=validation_result.is_valid,
            needs_rework=validation_result.needs_rework,
            should_return_agent=should_return_agent,
            pending_tasks=len(pending_tasks),
            in_progress_tasks=len(in_progress_tasks),
            done_tasks=len([t for t in todos if t.status == "done"]),
            note="CRITICAL: Chapter is written ONLY when finding is accepted (is_valid=True AND needs_rework=False). If chapter_written=False, this finding will NOT appear in draft_report."
        )
        
        # CRITICAL: Log detailed status for debugging missing chapters
        if not chapter_written:
            logger.error(
                "SUPERVISOR: Finding was NOT written to draft_report as chapter",
                agent_id=agent_id,
                finding_topic=finding.get("topic", "unknown"),
                is_valid=validation_result.is_valid,
                needs_rework=validation_result.needs_rework,
                validation_reasoning_preview=validation_result.reasoning[:200] if hasattr(validation_result, 'reasoning') and validation_result.reasoning else None,
                chapter_result_success=chapter_result.get("success") if chapter_result and isinstance(chapter_result, dict) else None,
                chapter_result_reason=chapter_result.get("reason") if chapter_result and isinstance(chapter_result, dict) else None,
                note="CRITICAL: This finding will NOT appear in draft_report. Possible reasons: finding not valid, needs rework, chapter write failed, or duplicate chapter."
            )
        else:
            logger.info(
                "SUPERVISOR: Finding successfully written to draft_report as chapter",
                agent_id=agent_id,
                finding_topic=finding.get("topic", "unknown"),
                chapter_number=chapter_number,
                chapter_title=chapter_title,
                note="SUCCESS: This finding is now a chapter in draft_report and will appear in final PDF."
            )
        
        # CRITICAL: Ensure task_management is always a dict, even if it's None or failed
        task_management_dict = {}
        if task_management:
            try:
                if hasattr(task_management, "dict"):
                    task_management_dict = task_management.dict()
                elif isinstance(task_management, dict):
                    task_management_dict = task_management
                else:
                    task_management_dict = {
                        "tasks_created": 0,
                        "tasks_updated": 0,
                        "task_addition_count": self.context.get("task_addition_count", 0),
                        "reasoning": "Task management result unavailable"
                    }
            except Exception as e:
                logger.warning("Failed to convert task_management to dict", error=str(e))
                task_management_dict = {
                    "tasks_created": 0,
                    "tasks_updated": 0,
                    "task_addition_count": self.context.get("task_addition_count", 0),
                    "reasoning": "Task management conversion failed"
                }
        
        # CRITICAL: Get chapter info from write_chapter result (chapter_result, not task_management result)
        chapter_number = None
        chapter_title = None
        if chapter_written and chapter_result and isinstance(chapter_result, dict):
            chapter_number = chapter_result.get("chapter_number")
            chapter_title = chapter_result.get("chapter_title")
        
        return {
            "success": True,
            "action": "chapter_added" if chapter_written else "skipped",
            "agent_id": agent_id,
            "finding_topic": finding.get("topic", "unknown"),  # CRITICAL: Include finding topic
            "chapter_written": chapter_written,  # CRITICAL: Include chapter_written flag
            "chapter_number": chapter_number,  # CRITICAL: Include chapter number if written
            "chapter_title": chapter_title,  # CRITICAL: Include chapter title if written
            "is_valid": validation_result.is_valid,  # CRITICAL: Include validation status
            "needs_rework": validation_result.needs_rework,  # CRITICAL: Include rework status
            "should_return_agent": should_return_agent,
            "task_management": task_management_dict,
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
    
    # CRITICAL: Get finding from queue but DON'T mark as done until processing completes successfully
    # This ensures finding is not lost if processing fails
    current_finding = None
    current_agent_id = None
    queue_event = None
    
    # CRITICAL: Get finding from queue with timeout to avoid blocking forever
    # If queue is empty, we'll return early and let continuous processing retry
    if supervisor_queue:
        try:
            # Get first finding from queue (FIFO) with timeout
            # This prevents blocking forever if queue is empty
            queue_size_before = supervisor_queue.size()
            logger.info(
                "SUPERVISOR: Attempting to get finding from queue",
                queue_size=queue_size_before,
                note="Will wait up to 2 seconds for finding to arrive"
            )
            
            # Use timeout to avoid blocking forever
            # CRITICAL: supervisor_queue uses deque, not asyncio.Queue
            # Use get_finding method which handles deque properly
            queue_event = await supervisor_queue.get_finding(timeout=2.0)
            
            if queue_event:
                # Extract finding and agent_id from queue event dict
                # queue_event structure from enqueue: {"agent_id": str, "action": str, "result": dict, "timestamp": float}
                # But agent_completed_task wraps result in {"task_title": str, "result": dict}
                # So queue_event["result"] = {"task_title": str, "result": finding_dict}
                # We need to extract the actual finding from queue_event["result"]["result"]
                result_wrapper = queue_event.get("result") if isinstance(queue_event, dict) else None
                if isinstance(result_wrapper, dict) and "result" in result_wrapper:
                    # Unwrap the finding from agent_completed_task structure
                    current_finding = result_wrapper.get("result")
                else:
                    # Fallback: use result_wrapper directly if structure is different
                    current_finding = result_wrapper
                current_agent_id = queue_event.get("agent_id") if isinstance(queue_event, dict) else None
                
                logger.info(
                    "SUPERVISOR: Found finding in queue",
                    agent_id=current_agent_id,
                    finding_topic=current_finding.get("topic", "unknown") if current_finding else "unknown",
                    finding_summary_length=len(current_finding.get("summary", "")) if current_finding else 0,
                    finding_sources_count=current_finding.get("sources_count", 0) if current_finding else 0,
                    queue_size_before=queue_size_before,
                    queue_size_after=supervisor_queue.size(),
                    note="CRITICAL: Finding retrieved from queue. Will be validated and written to draft_report as chapter if valid. Topic MUST match task title."
                )
            else:
                # Timeout - no finding available
                current_finding = None
                current_agent_id = None
                queue_event = None
                logger.debug(
                    "SUPERVISOR: No finding in queue within timeout",
                    queue_size=supervisor_queue.size() if supervisor_queue else 0,
                    note="This is normal - continuous processing will retry"
                )
        except Exception as timeout_error:
            # Queue is empty or no finding arrived within timeout, or other error
            # get_finding returns None on timeout, so this should only catch other errors
            if "timeout" in str(timeout_error).lower() or "TimeoutError" in str(type(timeout_error)):
                logger.debug(
                    "SUPERVISOR: No finding in queue within timeout",
                    queue_size=supervisor_queue.size() if supervisor_queue else 0,
                    note="This is normal - continuous processing will retry"
                )
            else:
                logger.error("SUPERVISOR: Error getting finding from queue", error=str(timeout_error), exc_info=True)
            current_finding = None
            current_agent_id = None
            queue_event = None
        except Exception as e:
            # This catch block should not be reached if get_finding works correctly
            # But keep it for safety
            logger.error("SUPERVISOR: Failed to get finding from queue", error=str(e), exc_info=True)
            current_finding = None
            current_agent_id = None
            queue_event = None
    
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
        # CRITICAL: agent_memory_service and agent_file_service are NOT stored in context
        # They are accessed via stream.app_state when needed (to avoid storing in context)
        "session_id": state.get("session_id"),
        "session_factory": stream.app_state.get("session_factory") if stream else None,
        "task_addition_count": state.get("task_addition_count", 0),
        "findings": state.get("findings", []),
        "current_finding": current_finding,
        "settings": settings,  # CRITICAL: Include settings for create_agent_todo_handler
        "stream": stream,  # CRITICAL: Include stream to access app_state for agent_file_service
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
    # CRITICAL: Wrap in try-except to ensure queue item is marked as done even if processing fails
    chain = SupervisorChain(llm, context)
    result = None
    processing_error = None
    
    try:
        result = await chain.process_finding(current_finding, current_agent_id, all_agents)
        
        # CRITICAL: Extract detailed info for logging
        finding_topic = result.get("finding_topic", "unknown") if result else "unknown"
        chapter_written = result.get("chapter_written", False) if result else False
        chapter_number = result.get("chapter_number") if result else None
        chapter_title = result.get("chapter_title") if result else None
        is_valid = result.get("is_valid") if result else None
        needs_rework = result.get("needs_rework") if result else None
        
        logger.info(
            "SUPERVISOR: Finding processing completed",
            agent_id=current_agent_id,
            finding_topic=finding_topic,
            result_success=result.get("success") if result else False,
            should_return_agent=result.get("should_return_agent", False) if result else False,
            action=result.get("action", "unknown") if result else "error",
            chapter_written=chapter_written,
            chapter_number=chapter_number,
            chapter_title=chapter_title,
            is_valid=is_valid,
            needs_rework=needs_rework,
            note="CRITICAL: If chapter_written=False, this finding will NOT appear in draft_report. Check validation and chapter write logs for details."
        )
        
        # CRITICAL: Log error if chapter was NOT written
        if not chapter_written:
            logger.error(
                "SUPERVISOR: Finding processing completed but chapter was NOT written",
                agent_id=current_agent_id,
                finding_topic=finding_topic,
                is_valid=is_valid,
                needs_rework=needs_rework,
                action=result.get("action", "unknown") if result else "error",
                note="CRITICAL: This finding will NOT appear in draft_report. Possible reasons: validation failed (is_valid=False), needs rework (needs_rework=True), or chapter write failed."
            )
    except Exception as e:
        processing_error = e
        logger.error(
            "SUPERVISOR: Exception during finding processing",
            error=str(e),
            agent_id=current_agent_id,
            finding_topic=current_finding.get("topic", "unknown") if current_finding else "unknown",
            exc_info=True,
            note="CRITICAL: Finding processing exception. Will mark queue item as done and continue processing other findings."
        )
        # Create error result to continue processing
        result = {
            "success": False,
            "error": str(e),
            "agent_id": current_agent_id,
            "should_return_agent": False,
            "chapter_written": False,
            "task_management": {}
        }
    
    # CRITICAL: Mark event as processed (remove from queue)
    # supervisor_queue uses deque, not asyncio.Queue, so there's no task_done() method
    # Finding was already removed from queue via get_finding() above (which uses popleft())
    # No need to call task_done() for deque
    if supervisor_queue and queue_event:
        logger.info(
            "SUPERVISOR: Finding processed and removed from queue",
            agent_id=current_agent_id,
            queue_size_remaining=supervisor_queue.size(),
            chapter_written=result.get("chapter_written", False) if (result and isinstance(result, dict)) else False,
            processing_error=bool(processing_error),
            note="Finding removed from queue via get_finding(). Processing continues even if errors occurred."
        )
    
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
    
    # CRITICAL: Always include processed_agent_id in return value
    # This allows continuous processing to know which agent's finding was processed
    
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
                    chapter_summaries_count=chapter_summaries_count,
                    note="All agent tasks are done - checking if all findings have chapters"
                )
                
                # CRITICAL: Check if there are findings in queue that haven't been processed yet
                # If queue has findings, continue processing
                queue_has_findings = supervisor_queue.size() > 0 if supervisor_queue else False
                
                if queue_has_findings:
                    logger.info(
                        "SUPERVISOR: Queue still has findings - continuing",
                        queue_size=supervisor_queue.size(),
                        findings_count=findings_count,
                        chapters_count=chapter_summaries_count,
                        note="Research will continue until all findings in queue are processed"
                    )
                    should_continue = True  # Continue to process remaining findings in queue
                # If number of chapters is less than number of findings, not all chapters are written yet
                elif chapter_summaries_count < findings_count:
                    logger.warning(
                        "SUPERVISOR: Not all findings have chapters yet - continuing",
                        findings_count=findings_count,
                        chapters_count=chapter_summaries_count,
                        missing_chapters=findings_count - chapter_summaries_count,
                        note="CRITICAL: Some findings were NOT written to draft_report as chapters! Research will continue until all findings are added as chapters. Some findings may have failed to write chapters due to API errors or validation failures."
                    )
                    should_continue = True  # Continue to process remaining findings
                    
                    # CRITICAL: Log which findings don't have chapters
                    if findings:
                        findings_with_chapters = {ch.get("topic", "").strip().lower() for ch in chapter_summaries if isinstance(ch, dict)}
                        findings_without_chapters = [
                            f for f in findings 
                            if isinstance(f, dict) and f.get("topic", "").strip().lower() not in findings_with_chapters
                        ]
                        if findings_without_chapters:
                            logger.error(
                                "SUPERVISOR: Findings without chapters detected",
                                findings_without_chapters_count=len(findings_without_chapters),
                                findings_without_chapters_topics=[f.get("topic", "unknown") for f in findings_without_chapters[:10]],
                                findings_without_chapters_agents=[f.get("agent_id", "unknown") for f in findings_without_chapters[:10]],
                                note="CRITICAL: These findings were NOT written to draft_report as chapters. They may have failed validation or chapter write failed. These findings will be LOST if not processed!"
                            )
                else:
                    should_continue = False
                    logger.info(
                        "SUPERVISOR: All tasks completed and all chapters written - research finished",
                        findings_count=findings_count,
                        chapters_count=chapter_summaries_count,
                        queue_size=supervisor_queue.size() if supervisor_queue else 0,
                        note="Research completion condition met: all tasks done, all findings have chapters, queue is empty"
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
    
    # CRITICAL: Extract finding topic and chapter info from result for detailed logging
    finding_topic = None
    chapter_number = None
    chapter_title = None
    if result and isinstance(result, dict):
        finding_topic = result.get("finding_topic")
        chapter_number = result.get("chapter_number")
        chapter_title = result.get("chapter_title")
    
    final_result = {
        "should_continue": should_continue,
        "agents_to_return": agents_to_return,
        "processed_agent_id": current_agent_id,  # CRITICAL: Include agent_id of processed finding
        "finding_topic": finding_topic,  # CRITICAL: Include finding topic for logging
        "chapter_number": chapter_number,  # CRITICAL: Include chapter number if written
        "chapter_title": chapter_title,  # CRITICAL: Include chapter title if written
        "replanning_needed": False,
        "task_management": result.get("task_management") if result else None,
    }
    
    logger.info(
        "SUPERVISOR: Final result prepared",
        should_continue=should_continue,
        agents_to_return=agents_to_return,
        agents_to_return_count=len(agents_to_return),
        has_task_management=bool(result.get("task_management"))
    )
    
    return final_result

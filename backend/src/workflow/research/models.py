"""Pydantic models for deep research workflow structured outputs."""

from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


# ==================== Query Analysis ====================

class QueryAnalysis(BaseModel):
    """Initial query analysis for research planning."""

    reasoning: str = Field(description="Detailed analysis of the query complexity and requirements")
    topics: list[str] = Field(description="Main topics to research", min_length=1, max_length=10)
    complexity: Literal["simple", "moderate", "complex"] = Field(description="Query complexity level")
    requires_deep_search: bool = Field(description="Whether deep search context is needed")
    estimated_agent_count: int = Field(description="Recommended number of research agents", ge=1, le=6)


# ==================== Research Planning ====================

class ResearchTopic(BaseModel):
    """Individual research topic."""

    topic: str = Field(description="Topic name")
    description: str = Field(description="Detailed description of what to research")
    priority: Literal["high", "medium", "low"] = Field(description="Topic priority")
    estimated_sources: int = Field(description="Estimated number of sources needed", ge=1)


class ResearchPlan(BaseModel):
    """Complete research plan with topics and approach."""

    reasoning: str = Field(description="Explanation of the research strategy and why these topics")
    topics: list[ResearchTopic] = Field(description="Research topics", min_length=1, max_length=10)
    research_depth: Literal["quick", "standard", "comprehensive"] = Field(description="Required research depth")
    coordination_strategy: str = Field(description="How agents should coordinate")


# ==================== Agent Characteristics ====================

class AgentTodo(BaseModel):
    """Individual todo item for agent."""
    
    model_config = ConfigDict(
        # Force all fields to be in required array for Azure/OpenRouter compatibility
        # NOTE: Fields with default values should NOT be in required array for proper structured output
        json_schema_extra={
            "required": ["reasoning", "title", "objective", "expected_output"]
        }
    )

    reasoning: str = Field(description="Why this task is important")
    title: str = Field(description="Task title", min_length=5)
    objective: str = Field(description="What should be achieved. MUST include the original user query if provided in context.")
    expected_output: str = Field(description="Expected result format")
    sources_needed: list[str] = Field(description="Types of sources to find", default_factory=list)
    priority: Literal["high", "medium", "low"] = Field(default="medium", description="Task priority")
    guidance: str = Field(description="Specific guidance on how to approach this task. MUST include the original user query and clarification answers if provided.", default="")


class AgentCharacteristic(BaseModel):
    """Characteristics for a specialized research agent."""

    reasoning: str = Field(description="Why this agent is needed for this role")
    agent_id: str = Field(description="Unique agent identifier")
    role: str = Field(description="Agent role (e.g., 'Senior Aviation Expert')")
    expertise: str = Field(description="Domain expertise")
    personality: str = Field(description="Research personality and approach")
    initial_todos: list[AgentTodo] = Field(description="Initial task list - MUST have 2-3 tasks per agent for comprehensive research", min_length=2, max_length=5)


class AgentCharacteristics(BaseModel):
    """Complete agent team characteristics."""

    reasoning: str = Field(description="Why this team composition is optimal")
    agents: list[AgentCharacteristic] = Field(description="Agent characteristics", min_length=1, max_length=6)
    coordination_notes: str = Field(description="How agents should work together")


# ==================== Agent Planning & Reflection ====================

class AgentPlan(BaseModel):
    """Agent's current research plan."""

    reasoning: str = Field(description="Analysis of current situation and plan rationale")
    current_goal: str = Field(description="Current main objective")
    next_steps: list[str] = Field(description="Next 1-3 steps to take", min_length=1, max_length=3)
    expected_findings: str = Field(description="What findings are expected")
    search_strategy: str = Field(description="How to search effectively")
    fallback_if_stuck: str = Field(description="Alternative approach if current doesn't work")


class AgentReflection(BaseModel):
    """Agent's reflection on research progress."""
    
    model_config = ConfigDict(
        # Force all fields to be in required array for Azure/OpenRouter compatibility
        json_schema_extra={
            "required": ["reasoning", "progress_assessment", "findings_quality", "should_replan", "new_direction", "ready_for_next_task"]
        }
    )

    reasoning: str = Field(description="Analysis of current progress and obstacles")
    progress_assessment: Literal["on_track", "needs_adjustment", "stuck", "complete"] = Field(
        description="Current progress status"
    )
    findings_quality: Literal["insufficient", "adequate", "good", "excellent"] = Field(
        description="Quality of findings so far"
    )
    should_replan: bool = Field(description="Whether replanning is needed")
    new_direction: Optional[str] = Field(default=None, description="New research direction if replanning")
    ready_for_next_task: bool = Field(description="Ready to move to next todo")


# ==================== Supervisor Assessment ====================

class ResearchGap(BaseModel):
    """Identified gap in research coverage."""
    
    model_config = ConfigDict(
        # Force all fields to be in required array for Azure/OpenRouter compatibility
        json_schema_extra={
            "required": ["gap_description", "severity", "suggested_action", "assign_to_agent"]
        }
    )

    gap_description: str = Field(description="What is missing")
    severity: Literal["critical", "important", "minor"] = Field(description="Gap severity")
    suggested_action: str = Field(description="How to address this gap")
    assign_to_agent: Optional[str] = Field(default=None, description="Which agent should handle this")


class AgentDirective(BaseModel):
    """New directive/todo for an agent from supervisor."""
    
    model_config = ConfigDict(
        # Force all fields to be in required array for Azure/OpenRouter compatibility
        json_schema_extra={
            "required": ["reasoning", "title", "objective", "expected_output", "priority", "guidance"]
        }
    )

    reasoning: str = Field(description="Why this task is needed")
    title: str = Field(description="Task title")
    objective: str = Field(description="What to achieve")
    expected_output: str = Field(description="Expected result")
    priority: Literal["high", "medium", "low"] = Field(default="high", description="Task priority")
    guidance: str = Field(description="Specific guidance on how to approach this")


class SupervisorAssessment(BaseModel):
    """Supervisor's assessment of research progress."""
    
    model_config = ConfigDict(
        # Force all fields to be in required array for Azure/OpenRouter compatibility
        json_schema_extra={
            "required": ["reasoning", "overall_progress", "gaps_found", "quality_assessment", "should_continue", "completion_criteria_met", "agent_directives", "main_document_update"]
        }
    )

    reasoning: str = Field(description="Detailed analysis of all agents' work and overall progress")
    overall_progress: int = Field(description="Overall progress percentage", ge=0, le=100)
    gaps_found: list[ResearchGap] = Field(description="Identified research gaps", default_factory=list)
    quality_assessment: Literal["poor", "fair", "good", "excellent"] = Field(
        description="Overall research quality"
    )
    should_continue: bool = Field(description="Whether research should continue")
    completion_criteria_met: bool = Field(description="Whether research meets completion criteria")
    agent_directives: dict[str, list[AgentDirective]] = Field(
        description="New tasks for each agent (agent_id -> directives)",
        default_factory=dict
    )
    main_document_update: str = Field(description="Update to add to main research document")


# ==================== Report Generation ====================

class ReportSection(BaseModel):
    """Section of the final report."""
    
    model_config = ConfigDict(
        # Force all fields to be in required array for Azure/OpenRouter compatibility
        json_schema_extra={
            "required": ["title", "content", "sources"]
        }
    )

    title: str = Field(description="Section title")
    content: str = Field(description="Detailed section content in markdown (300-800 words, comprehensive analysis with specific facts, data, and evidence)")
    sources: list[str] = Field(description="Source URLs cited in this section", default_factory=list)


class FinalReport(BaseModel):
    """Complete research report."""

    reasoning: str = Field(description="Explanation of report structure and content choices")
    title: str = Field(description="Report title")
    executive_summary: str = Field(description="Comprehensive executive summary (200-400 words, not brief)")
    sections: list[ReportSection] = Field(description="Report sections with detailed content (each section 300-800 words, multiple sections required)", min_length=3)  # Require at least 3 sections
    conclusion: str = Field(description="Comprehensive final conclusion (200-400 words, not brief)")
    total_sources: int = Field(description="Total number of sources used", ge=1)
    confidence_level: Literal["low", "medium", "high", "very high"] = Field(
        description="Confidence in report accuracy"
    )


class ReportValidation(BaseModel):
    """Validation of report before submission."""
    
    model_config = ConfigDict(
        # Force all fields to be in required array for Azure/OpenRouter compatibility
        json_schema_extra={
            "required": ["reasoning", "is_complete", "quality_score", "missing_aspects", "needs_revision", "revision_suggestions"]
        }
    )

    reasoning: str = Field(description="Analysis of report completeness and quality")
    is_complete: bool = Field(description="Whether report covers all required aspects")
    quality_score: int = Field(description="Quality score 1-10", ge=1, le=10)
    missing_aspects: list[str] = Field(description="Missing or weak aspects", default_factory=list)
    needs_revision: bool = Field(description="Whether revision is needed")
    revision_suggestions: list[str] = Field(description="Specific suggestions for improvement", default_factory=list)


# ==================== Clarifying Questions ====================

class ClarifyingQuestion(BaseModel):
    """Question to ask user for clarification."""

    question: str = Field(description="The question to ask")
    why_needed: str = Field(description="Why this clarification is needed")
    default_assumption: str = Field(description="What will be assumed if user doesn't answer")


class ClarificationNeeds(BaseModel):
    """Assessment of whether clarification is needed."""

    model_config = ConfigDict(
        # Force all fields to be in required array for Azure/OpenRouter compatibility
        json_schema_extra={
            "required": ["reasoning", "needs_clarification", "questions", "can_proceed_without"]
        }
    )

    reasoning: str = Field(description="Analysis of query clarity")
    needs_clarification: bool = Field(description="Whether user input is needed")
    questions: list[ClarifyingQuestion] = Field(
        description="Questions to ask user - MUST always return at least 2-3 questions, never empty list",
        default_factory=list,
        min_length=2,
        max_length=3
    )
    can_proceed_without: bool = Field(description="Whether research can proceed with assumptions")


# ==================== Findings Compression ====================

class CompressedFindings(BaseModel):
    """Compressed and structured research findings."""

    reasoning: str = Field(description="Analysis of findings and compression strategy")
    key_insights: list[str] = Field(
        description="Most important insights from all findings",
        min_length=3,
        max_length=15
    )
    themes: list[str] = Field(
        description="Main themes identified across findings",
        min_length=1,
        max_length=8
    )
    summary: str = Field(
        description="Comprehensive summary of all findings (500-1000 words)"
    )
    source_count: int = Field(
        description="Total number of unique sources",
        ge=0
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="Confidence in findings completeness"
    )

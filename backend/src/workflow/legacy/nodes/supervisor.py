"""Supervisor node for coordinating researchers."""

import structlog
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.workflow.state import ResearchFinding, SupervisorState

logger = structlog.get_logger(__name__)


SUPERVISOR_SYSTEM_PROMPT = """You are a research supervisor coordinating a team of specialized researchers.

Your responsibilities:
1. **Strategic Planning**: Analyze the research query and plan the investigation
2. **Task Delegation**: Assign specific research topics to individual researchers
3. **Quality Control**: Review findings and identify gaps or areas needing deeper investigation
4. **Decision Making**: Decide when enough research has been conducted

Guidelines:
- Always reflect on strategy before taking action
- Delegate research tasks that are specific and well-defined
- Each researcher should investigate a distinct aspect of the query
- Consider existing findings to avoid redundancy
- Know when to stop and move to synthesis

Return JSON with fields: reasoning, tool_calls, completed.
tool_calls is a list of objects with name (think|conduct_research) and args.

Current iteration: {iteration} / {max_iterations}"""


class SupervisorToolCall(BaseModel):
    """Structured tool call for supervisor decisions."""

    reasoning: str = Field(..., description="Why this tool call is needed")
    name: str = Field(..., description="Tool name: think or conduct_research")
    args: dict = Field(default_factory=dict, description="Arguments for the tool")


class SupervisorDecision(BaseModel):
    """Structured supervisor decision output."""

    reasoning: str = Field(..., description="Why these actions are chosen")
    tool_calls: list[SupervisorToolCall] = Field(default_factory=list)
    completed: bool = Field(default=False, description="Whether research should stop")


async def supervisor_node(
    state: SupervisorState,
    llm: any,  # LangChain LLM with tool binding
) -> dict:
    """
    Supervisor coordinates research by delegating to researchers.

    Args:
        state: Current supervisor state
        llm: LLM with tools bound

    Returns:
        State update with tool calls or completion signal
    """
    query = state.get("query", "")
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 6)
    current_findings = state.get("current_findings", [])

    logger.info(
        "Supervisor node executing",
        iteration=iteration,
        max_iterations=max_iterations,
        findings_count=len(current_findings),
    )

    # Check if we've reached max iterations
    if iteration >= max_iterations:
        logger.info("Max iterations reached, supervisor completing")
        return {
            "completed": True,
            "supervisor_reasoning": {"type": "override", "value": "Maximum iterations reached"},
        }

    # Prepare context for supervisor
    findings_summary = _summarize_findings(current_findings)

    # Create supervisor prompt
    user_message = f"""Research Query: {query}

{findings_summary}

Please:
1. Use the think tool to strategize your next steps
2. Delegate research tasks using conduct_research if more investigation is needed
3. If you have sufficient information, indicate completion

Remember: Be strategic and avoid redundant research."""

    messages = [
        SystemMessage(
            content=SUPERVISOR_SYSTEM_PROMPT.format(
                iteration=iteration,
                max_iterations=max_iterations,
            )
        ),
        HumanMessage(content=user_message),
    ]

    structured_llm = llm.with_structured_output(SupervisorDecision, method="function_calling")

    try:
        response = await structured_llm.ainvoke(messages)
        if not isinstance(response, SupervisorDecision):
            raise ValueError("SupervisorDecision response was not structured")

        tool_calls = [
            {"id": str(uuid4()), "name": call.name, "args": call.args}
            for call in response.tool_calls
            if call.name
        ]

        if response.completed or not tool_calls:
            logger.info("Supervisor completed without tool calls")
            return {
                "completed": True,
                "supervisor_reasoning": {
                    "type": "override",
                    "value": response.reasoning,
                },
            }

        logger.info("Supervisor issued tool calls", count=len(tool_calls))
        return {
            "pending_tool_calls": {"type": "override", "value": tool_calls},
            "supervisor_reasoning": {"type": "override", "value": response.reasoning},
        }

    except Exception as e:
        logger.error("Supervisor node failed", error=str(e))
        return {
            "completed": True,
            "supervisor_reasoning": {"type": "override", "value": f"Error: {str(e)}"},
        }


def _summarize_findings(findings: list[ResearchFinding]) -> str:
    """
    Summarize current research findings.

    Args:
        findings: List of research findings

    Returns:
        Formatted summary string
    """
    if not findings:
        return "**Current Findings:** None yet. This is the first iteration."

    summary = "**Current Findings:**\n\n"

    for idx, finding in enumerate(findings, 1):
        summary += f"{idx}. **{finding.topic}** (Confidence: {finding.confidence})\n"
        summary += f"   {finding.summary}\n"
        summary += f"   Key insights: {', '.join(finding.key_findings[:3])}\n\n"

    return summary.strip()

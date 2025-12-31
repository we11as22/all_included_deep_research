"""Research planning node."""

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.workflow.nodes.memory_search import format_memory_context_for_prompt
from src.workflow.state import ResearchState
from src.utils.chat_history import format_chat_history

logger = structlog.get_logger(__name__)


class ResearchPlan(BaseModel):
    """Structured research plan."""

    reasoning: str = Field(..., description="Why this plan best answers the query")

    overview: str = Field(..., description="High-level overview of the research approach")
    topics: list[str] = Field(
        ...,
        description="Specific topics or questions to investigate",
        min_length=1,
        max_length=10,
    )
    rationale: str = Field(..., description="Additional justification for the plan structure")


PLANNING_SYSTEM_PROMPT = """You are a research planning expert. Your role is to create a strategic research plan.

Given a research query and any relevant memory context, create a comprehensive plan that:
1. Breaks down the query into specific, researchable topics
2. Identifies key questions that need to be answered
3. Considers what we already know (from memory) and what we need to find out
4. Prioritizes topics by importance and relevance

Your plan should be:
- **Focused**: Each topic should be specific and well-defined
- **Comprehensive**: Cover all aspects of the query
- **Actionable**: Topics should be concrete enough for researchers to investigate
- **Non-redundant**: Avoid overlapping topics

Mode-specific guidance:
- **Speed mode**: 1-2 topics, very targeted
- **Balanced mode**: 3-5 topics, balanced coverage
- **Quality mode**: 5-8 topics, comprehensive exploration"""


PLANNING_USER_TEMPLATE = """Research Query: {query}

Research Mode: {mode}

{chat_history}

{memory_context}

Based on this information, create a strategic research plan.

Return JSON with fields: reasoning, overview, topics, rationale."""


async def plan_research_node(
    state: ResearchState,
    llm: any,  # LangChain LLM instance
) -> dict:
    """
    Create research plan based on query and memory context.

    Args:
        state: Current research state
        llm: LangChain LLM for planning

    Returns:
        State update with research_plan and research_topics
    """
    query = state.get("clarified_query") or state.get("query", "")
    mode = state.get("mode", "balanced")
    memory_context = state.get("memory_context", [])
    messages = state.get("messages", [])
    stream = state.get("stream")

    logger.info("Creating research plan", query=query, mode=mode)

    try:
        # Format memory context
        memory_str = format_memory_context_for_prompt(memory_context)

        chat_history = format_chat_history(messages, limit=len(messages))

        structured_llm = llm.with_structured_output(ResearchPlan, method="function_calling")
        try:
            plan_obj = await structured_llm.ainvoke(
                [
                    SystemMessage(content=PLANNING_SYSTEM_PROMPT),
                    HumanMessage(
                        content=PLANNING_USER_TEMPLATE.format(
                            query=query,
                            mode=mode,
                            chat_history=chat_history,
                            memory_context=memory_str,
                        )
                    ),
                ]
            )
            if not isinstance(plan_obj, ResearchPlan):
                raise ValueError("Planner returned non-structured output")
        except Exception as e:
            logger.error("Structured planning failed", error=str(e))
            plan_obj = ResearchPlan(
                reasoning="Fallback plan because structured planning failed.",
                overview=f"Directly investigate: {query}",
                topics=[query],
                rationale="Single-topic fallback to keep research moving.",
            )

        topics = plan_obj.topics
        plan_text = (
            f"## Reasoning\n\n{plan_obj.reasoning}\n\n"
            f"## Overview\n\n{plan_obj.overview}\n\n## Research Topics\n\n"
            + "\n".join(f"{i+1}. {t}" for i, t in enumerate(topics))
            + f"\n\n## Rationale\n\n{plan_obj.rationale}"
        )

        # Limit topics based on mode
        max_topics = _get_max_topics_for_mode(mode)
        topics = topics[:max_topics]

        logger.info(
            "Research plan created",
            topics_count=len(topics),
            mode=mode,
        )

        if stream:
            stream.emit_research_plan(plan_text, topics)

        return {
            "research_plan": {"type": "override", "value": plan_text},
            "research_topics": {"type": "override", "value": topics},
        }

    except Exception as e:
        logger.error("Research planning failed", error=str(e))

        # Fallback: use query as single topic
        return {
            "research_plan": {
                "type": "override",
                "value": f"Direct research on: {query}",
            },
            "research_topics": {"type": "override", "value": [query]},
        }


def _extract_topics_from_plan(plan_text: str) -> list[str]:
    """
    Extract numbered topics from plan text.

    Args:
        plan_text: Full plan text

    Returns:
        List of topic strings
    """
    topics = []
    in_topics_section = False

    for line in plan_text.split("\n"):
        line = line.strip()

        # Detect topics section
        if "## Research Topics" in line or "## Topics" in line:
            in_topics_section = True
            continue

        # End of topics section
        if in_topics_section and line.startswith("##"):
            break

        # Extract numbered items
        if in_topics_section and line:
            # Remove numbering (1., 2., -, *, etc.)
            cleaned = line.lstrip("0123456789.-*• \t")
            if cleaned:
                topics.append(cleaned)

    # If parsing failed, try simpler extraction
    if not topics:
        import re

        pattern = r"^\d+\.\s+(.+)$"
        for line in plan_text.split("\n"):
            match = re.match(pattern, line.strip())
            if match:
                topics.append(match.group(1))

    return topics


def _get_max_topics_for_mode(mode: str) -> int:
    """
    Get maximum topics for research mode.

    Args:
        mode: Research mode

    Returns:
        Maximum number of topics
    """
    if mode == "speed":
        return 2
    elif mode == "balanced":
        return 5
    else:  # quality
        return 8

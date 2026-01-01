"""ThinkTool for strategic reflection and reasoning."""

from typing import Annotated

from langchain_core.tools import tool


@tool
def think_tool(
    reflection: Annotated[
        str,
        "Your strategic thinking, analysis, or planning before taking action",
    ]
) -> str:
    """
    Strategic reflection and reasoning tool.

    Use this tool to think through your approach before taking action.
    This is a forced reflection step that helps ensure well-reasoned decisions.

    When to use this tool:
    - Before starting a complex research task
    - When planning your next steps
    - When synthesizing information from multiple sources
    - When deciding which research direction to pursue
    - Before concluding your research

    What to reflect on:
    - What do I already know about this topic?
    - What are the key questions I need to answer?
    - What's the best approach to find this information?
    - Are there any gaps or uncertainties in my current understanding?
    - What are the most important insights I've discovered?
    - How can I best organize and present my findings?

    Args:
        reflection: Your detailed thoughts, analysis, or strategic planning

    Returns:
        Confirmation that reflection has been recorded
    """
    # Reflection is recorded in the conversation history
    # This creates a clear separation between thinking and action
    word_count = len(reflection.split())
    return f"Strategic reflection recorded ({word_count} words). Ready to proceed with action."


@tool
def analyze_sources_tool(
    analysis: Annotated[
        str,
        "Your analysis of the sources you've found and how they relate to the research question",
    ]
) -> str:
    """
    Analyze and evaluate sources before proceeding.

    Use this tool to critically evaluate sources you've found:
    - Are the sources credible and authoritative?
    - Do they provide diverse perspectives?
    - Are there conflicting viewpoints that need reconciliation?
    - What are the gaps in the available information?
    - Which sources are most relevant to the research question?

    Args:
        analysis: Your detailed source analysis

    Returns:
        Confirmation that analysis has been recorded
    """
    return f"Source analysis recorded. Continuing with informed research strategy."

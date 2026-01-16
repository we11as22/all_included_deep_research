"""Planning prompt builder for research planning."""

from typing import Dict, Any
from src.workflow.research.prompts.base import PromptBuilder


class PlanningPromptBuilder(PromptBuilder):
    """Build prompts for research planning."""

    def build_planning_prompt(
        self,
        query: str,
        query_analysis: Dict[str, Any],
        deep_search_result: str,
        clarification_answers: str = "",
        mode: str = "quality",
    ) -> str:
        """Build prompt for creating research plan.

        Args:
            query: Original user query
            query_analysis: Query analysis results
            deep_search_result: Initial deep search results
            clarification_answers: User clarification answers
            mode: Research mode

        Returns:
            Complete planning prompt
        """
        deep_search_summary = self._truncate(deep_search_result, max_length=1500)
        key_aspects = query_analysis.get("key_aspects", [])
        complexity = query_analysis.get("complexity", {})

        sections = [
            self._format_role(),
            self._format_context(query, mode, deep_search_summary, clarification_answers),
            self._format_analysis_summary(key_aspects, complexity),
            self._format_planning_guidelines(query),
            self._format_output_structure(),
        ]

        return self._format_sections(sections)

    def _format_role(self) -> str:
        """Format planner role."""
        return """You are creating a research plan to guide comprehensive investigation."""

    def _format_context(
        self,
        query: str,
        mode: str,
        deep_search_summary: str,
        clarification_answers: str,
    ) -> str:
        """Format context section."""
        clarification_text = ""
        if clarification_answers:
            clarification_text = f"\n\n**User Clarification:**\n{clarification_answers}"

        context = f"""**Research Query:** {query}

**Research Mode:** {mode}

**Initial Context:**
{deep_search_summary}{clarification_text}"""

        return self._format_section("Context", context)

    def _format_analysis_summary(self, key_aspects: list, complexity: dict | str) -> str:
        """Format analysis summary.

        Args:
            key_aspects: List of key aspects identified
            complexity: Either a dict with 'level' and 'estimated_agents', or a string like 'simple'/'complex'
        """
        aspects_text = "\n".join([f"- {aspect}" for aspect in key_aspects]) if key_aspects else "To be determined"

        # Handle both dict and string complexity formats
        if isinstance(complexity, dict):
            complexity_level = complexity.get("level", "medium")
            estimated_agents = complexity.get("estimated_agents", 4)
        else:
            # String format like 'simple', 'medium', 'complex'
            complexity_level = str(complexity) if complexity else "medium"
            # Map complexity to estimated agents
            estimated_agents = {
                "simple": 2,
                "medium": 4,
                "complex": 5,
            }.get(complexity_level, 4)

        summary = f"""**Key Aspects Identified:**
{aspects_text}

**Complexity:** {complexity_level}
**Estimated Research Agents:** {estimated_agents}"""

        return self._format_section("Analysis Summary", summary)

    def _format_planning_guidelines(self, query: str) -> str:
        """Format planning guidelines."""
        guidelines = f"""**Planning Principles:**

1. **Comprehensive Coverage:**
   - Break down the topic into distinct research areas
   - Each topic should cover a different aspect/angle
   - Aim for 4-8 topics depending on complexity
   - Ensure topics collectively cover the full scope

2. **Topic Quality:**
   - Each topic should be specific and actionable
   - Provide clear description of what to research
   - Explain why this topic is important
   - Set appropriate priority (high/medium/low)

3. **Diversity:**
   - Cover multiple angles: historical, technical, practical, expert opinions, trends, comparisons, challenges
   - Avoid overlapping topics
   - Balance breadth and depth

4. **Context Retention:**
   - **CRITICAL**: All topics MUST relate to the original query: "{query}"
   - Include query context in topic descriptions
   - If clarification was provided, interpret it in context of original query

**FORBIDDEN:**
- Creating generic topics unrelated to the query
- Overlapping topics that duplicate research effort
- Topics too broad or too narrow for effective research"""

        return self._format_section("Planning Guidelines", guidelines)

    def _format_output_structure(self) -> str:
        """Format output structure."""
        output = """Generate research plan with this structure:

{
  "reasoning": "Overall research strategy explanation",
  "topics": [
    {
      "reasoning": "Why this topic is important",
      "topic": "Topic title",
      "description": "Detailed description of what to research",
      "priority": "high/medium/low"
    },
    ...
  ],
  "stop": false (always false - planning is one-shot)
}"""

        return self._format_section("Output Structure", output)

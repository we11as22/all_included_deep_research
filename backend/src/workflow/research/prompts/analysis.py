"""Analysis prompt builder for query analysis."""

from src.workflow.research.prompts.base import PromptBuilder


class AnalysisPromptBuilder(PromptBuilder):
    """Build prompts for query analysis."""

    def build_analysis_prompt(
        self,
        query: str,
        deep_search_result: str,
        chat_history: list,
        mode: str = "quality",
    ) -> str:
        """Build prompt for analyzing user query.

        Args:
            query: User query to analyze
            deep_search_result: Initial deep search results
            chat_history: Conversation history
            mode: Research mode (quality/balanced/speed)

        Returns:
            Complete analysis prompt
        """
        deep_search_summary = self._truncate(deep_search_result, max_length=1000)

        sections = [
            self._format_role(),
            self._format_context(query, deep_search_summary, mode),
            self._format_analysis_objectives(),
            self._format_output_structure(),
        ]

        return self._format_sections(sections)

    def _format_role(self) -> str:
        """Format analyzer role."""
        return """You are analyzing a research query to guide comprehensive research."""

    def _format_context(self, query: str, deep_search_summary: str, mode: str) -> str:
        """Format context section."""
        context = f"""**Query to Analyze:** {query}

**Research Mode:** {mode}

**Initial Search Context:**
{deep_search_summary}"""

        return self._format_section("Context", context)

    def _format_analysis_objectives(self) -> str:
        """Format analysis objectives."""
        objectives = """Analyze the query to determine:

1. **Complexity Assessment:**
   - How complex is this topic?
   - What depth of research is needed?
   - Estimated number of distinct aspects to research

2. **Ambiguity Detection:**
   - Are there multiple valid interpretations?
   - What clarifications would be most helpful?
   - What assumptions should be made if no clarification?

3. **Research Strategy:**
   - Key aspects/angles to investigate
   - Required expertise areas
   - Optimal research approach

4. **Scope Determination:**
   - Is the query too broad or too narrow?
   - What boundaries should be set?
   - What depth is appropriate for the mode?"""

        return self._format_section("Analysis Objectives", objectives)

    def _format_output_structure(self) -> str:
        """Format output structure."""
        output = """Provide analysis as structured output:

{
  "complexity": {
    "level": "low/medium/high",
    "reasoning": "Why this complexity level",
    "estimated_agents": 3-8
  },
  "ambiguities": ["ambiguity 1", "ambiguity 2", ...],
  "key_aspects": ["aspect 1", "aspect 2", ...],
  "research_strategy": "recommended approach",
  "requires_clarification": true/false
}"""

        return self._format_section("Output Structure", output)

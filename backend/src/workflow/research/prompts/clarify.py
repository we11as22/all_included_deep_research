"""Clarification prompt builder for user query clarification."""

from src.workflow.research.prompts.base import PromptBuilder


class ClarificationPromptBuilder(PromptBuilder):
    """Build prompts for user query clarification."""

    def build_clarification_prompt(
        self,
        query: str,
        deep_search_result: str,
        query_analysis: dict,
        user_language: str = "English",
    ) -> str:
        """Build prompt for generating clarification questions.

        Args:
            query: Original user query
            deep_search_result: Initial deep search results
            query_analysis: Query analysis with ambiguities and complexities
            user_language: Language for questions

        Returns:
            Prompt for clarification question generation
        """
        deep_search_summary = self._truncate(deep_search_result, max_length=1500)

        ambiguities = query_analysis.get("ambiguities", [])
        complexity = query_analysis.get("complexity", {})

        sections = [
            self._format_role(),
            self._format_context(query, deep_search_summary, user_language),
            self._format_analysis_findings(ambiguities, complexity),
            self._format_clarification_guidelines(),
            self._format_output_format(),
        ]

        return self._format_sections(sections)

    def _format_role(self) -> str:
        """Format clarification role."""
        return """You are helping to refine a research query by asking clarification questions."""

    def _format_context(self, query: str, deep_search_summary: str, user_language: str) -> str:
        """Format context section."""
        context = f"""**Original User Query:** {query}

**Initial Research Context:**
{deep_search_summary}

**Response Language:** {user_language} (all questions MUST be in this language)"""

        return self._format_section("Context", context)

    def _format_analysis_findings(self, ambiguities: list, complexity: dict) -> str:
        """Format analysis findings."""
        ambiguities_text = "\n".join([f"- {a}" for a in ambiguities]) if ambiguities else "None identified"

        complexity_text = ""
        if complexity:
            complexity_text = f"Estimated research depth: {complexity.get('depth', 'medium')}"

        findings = f"""**Potential Ambiguities:**
{ambiguities_text}

{complexity_text}"""

        return self._format_section("Analysis Findings", findings)

    def _format_clarification_guidelines(self) -> str:
        """Format guidelines for clarification questions."""
        guidelines = """**When to Ask Clarification:**
- Query has significant ambiguities (multiple valid interpretations)
- Unclear what depth/detail level the user wants
- Topic can be approached from multiple angles (historical, technical, practical, etc.)
- Scope is very broad or very narrow

**When NOT to Ask:**
- Query is clear and specific
- Only minor ambiguities that can be addressed in research
- Questions would be trivial or unnecessary

**Question Quality:**
- Maximum 3-5 focused questions
- Each question should meaningfully improve research direction
- Make questions specific and answerable
- Frame questions to understand user's goals and priorities"""

        return self._format_section("Clarification Guidelines", guidelines)

    def _format_output_format(self) -> str:
        """Format output format instructions."""
        output = """Generate clarification questions as a structured response:

{
  "needs_clarification": true/false,
  "reasoning": "Why clarification is/isn't needed",
  "questions": [
    "Question 1?",
    "Question 2?",
    ...
  ]
}

If needs_clarification is false, return empty questions array."""

        return self._format_section("Output Format", output)

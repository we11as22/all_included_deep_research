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

1. **Comprehensive Coverage - CRITICAL:**
   - Break down the topic into distinct research areas that TOGETHER provide complete coverage
   - Each topic should cover a different aspect/angle that is ESSENTIAL for answering the query
   - Aim for 4-8 topics depending on complexity, ensuring NO gaps in coverage
   - **MANDATORY**: Topics must collectively answer ALL aspects of the query: "{query}"
   - Think systematically: what questions need to be answered to fully address the query?
   - Consider: What? Why? How? When? Where? Who? What are the implications? What are the challenges?
   - Ensure topics complement each other and together form a complete picture

2. **Topic Quality:**
   - Each topic should be specific, actionable, and researchable
   - Provide clear, detailed description of what to research (not vague)
   - Explain why this topic is important for answering the query
   - Set appropriate priority (high/medium/low) based on importance to the query
   - Each topic should be substantial enough to generate a full, informative chapter

3. **Diversity and Completeness:**
   - Cover multiple angles: historical context, technical details, practical applications, expert opinions, current trends, comparisons, challenges, future implications
   - Avoid overlapping topics - each should be distinct
   - Balance breadth (covering all aspects) and depth (sufficient detail)
   - Consider both theoretical and practical perspectives
   - Include foundational topics (basics, definitions) AND advanced topics (implications, trends)

4. **Context Retention:**
   - **CRITICAL**: All topics MUST directly relate to and help answer the original query: "{query}"
   - Include query context in topic descriptions so agents understand the connection
   - If clarification was provided, interpret it in context of original query
   - Each topic should contribute unique value toward answering the query

5. **Gap Prevention:**
   - Before finalizing topics, verify: "Do these topics together fully answer the query?"
   - Identify potential gaps: what aspects of the query might not be covered?
   - If gaps exist, add topics to fill them
   - Ensure no critical aspect of the query is left unaddressed

**FORBIDDEN:**
- Creating generic topics unrelated to the query
- Overlapping topics that duplicate research effort
- Topics too broad (unfocused) or too narrow (insignificant) for effective research
- Leaving gaps in coverage - all aspects of the query must be addressed
- Creating topics that don't contribute to answering the query"""

        return self._format_section("Planning Guidelines", guidelines)

    def _format_output_structure(self) -> str:
        """Format output structure."""
        output = """Generate research plan with this structure:

{
  "reasoning": "**CRITICAL: Before creating the plan, think about and document in reasoning:**
  1. **Original Query Analysis**: What is the user asking for? What is the core topic?
  2. **Deep Search Context**: What did the initial deep search reveal? What key aspects were found?
  3. **Clarification Answers**: If clarification was provided, what did the user specify? How does it refine the original query?
  4. **Integration**: How do deep search results and clarification answers relate to the original query?
  5. **Research Strategy**: Based on all context, what research approach will be most effective?
  
  Document your thinking process in the reasoning field before listing topics.",
  "topics": [
    {
      "reasoning": "**Before defining this topic, think about:**
      1. How does this topic relate to the original query?
      2. What aspect of deep search context does it address?
      3. How does it incorporate clarification answers (if provided)?
      4. Why is this specific topic important for comprehensive research?
      
      Document your thinking in reasoning before defining the topic.",
      "topic": "Topic title",
      "description": "Detailed description of what to research",
      "priority": "high/medium/low"
    },
    ...
  ],
  "stop": false (always false - planning is one-shot)
}"""

        return self._format_section("Output Structure", output)

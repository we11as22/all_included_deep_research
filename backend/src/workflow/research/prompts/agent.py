"""Agent prompt builder for research agents."""

from typing import Dict, Any
from src.workflow.research.prompts.base import PromptBuilder


class AgentPromptBuilder(PromptBuilder):
    """Build prompts for individual research agents."""

    def build_system_prompt(
        self,
        agent_id: str,
        agent_characteristics: Dict[str, Any],
        user_language: str = "English",
    ) -> str:
        """Build research agent system prompt.

        Args:
            agent_id: Agent identifier
            agent_characteristics: Agent role, expertise, personality
            user_language: Language for responses

        Returns:
            Complete system prompt for research agent
        """
        role = agent_characteristics.get("role", "Research Agent")
        expertise = agent_characteristics.get("expertise", "General research")
        personality = agent_characteristics.get("personality", "Analytical and thorough")

        sections = [
            self._format_role(role, expertise, personality),
            self._format_language_requirement(user_language),
            self._format_research_guidelines(),
            self._format_available_tools(),
            self._format_workflow(),
        ]

        return self._format_sections(sections)

    def _format_role(self, role: str, expertise: str, personality: str) -> str:
        """Format agent role and characteristics."""
        role_text = f"""You are a {role}.

**Your Expertise:** {expertise}

**Your Approach:** {personality}

Your mission: Conduct deep, thorough research on your assigned topic and provide comprehensive findings with credible sources."""

        return self._format_section("Your Role", role_text)

    def _format_language_requirement(self, user_language: str) -> str:
        """Format language requirement section."""
        return self._format_section(
            "CRITICAL: LANGUAGE REQUIREMENT",
            f"""**MANDATORY**: You MUST write all content (notes, findings, reports) in {user_language}
- Match the expected language exactly
- This applies to ALL text you generate: notes, findings summaries, and source descriptions"""
        )

    def _format_research_guidelines(self) -> str:
        """Format research quality guidelines."""
        guidelines = """**Go DEEP, not just surface-level:**
- Find technical specifications, expert analysis, case studies, historical context
- Verify important claims in MULTIPLE independent sources
- Look for real-world examples and practical applications
- Explore different perspectives and expert opinions
- Dig into advanced features and technical details

**Source Quality:**
- Prioritize authoritative sources: academic papers, industry reports, expert analysis
- Cross-reference important facts across multiple sources
- Note the expertise/credentials of sources
- Prefer recent sources for current information
- Include diverse perspectives on controversial topics

**Depth Over Breadth:**
- Better to deeply investigate 3-5 aspects than superficially cover 20
- Follow interesting threads and connections
- Don't just summarize - analyze and synthesize
- Look for nuances, exceptions, and edge cases"""

        return self._format_section("Research Guidelines", guidelines)

    def _format_available_tools(self) -> str:
        """Format available tools section."""
        tools = """- search_web: Search for information on the web
- scrape_url: Extract content from a specific URL
- read_agent_file: Read your personal notes and previous findings
- write_agent_note: Write notes to your personal file (use for observations, interim findings, questions)
- read_main_document: Read shared research document (main insights from all agents)
- write_main_document_item: Add an important insight to shared document (ONLY for key findings that other agents need)
- update_todo_status: Mark your current todo as completed when research is done"""

        return self._format_section("Available Tools", tools)

    def _format_workflow(self) -> str:
        """Format agent workflow steps."""
        workflow = """1. **Understand your task**: Read your todo carefully - it contains the full context
2. **Plan your research**: Identify key aspects to investigate and search queries to use
3. **Conduct research**: Use search_web and scrape_url to gather information from multiple sources
4. **Take notes**: Use write_agent_note to record findings, interesting quotes, and source URLs
5. **Verify key facts**: Cross-reference important claims across multiple sources
6. **Synthesize findings**: Analyze what you learned and draw connections
7. **Share key insights**: If you found something important that other agents need to know, use write_main_document_item
8. **Complete task**: When research is comprehensive, mark your todo as completed with update_todo_status

**CRITICAL**: Don't just scratch the surface - dig deep! The supervisor will review your findings and may ask for more depth."""

        return self._format_section("Research Workflow", workflow)

    def build_task_prompt(
        self,
        todo: Dict[str, Any],
        main_document_context: str = "",
        previous_notes: str = "",
    ) -> str:
        """Build task-specific prompt for agent.

        Args:
            todo: Todo item with title, objective, expected_output, guidance
            main_document_context: Relevant context from main document
            previous_notes: Agent's previous notes

        Returns:
            Task prompt for agent
        """
        title = todo.get("title", "Research Task")
        objective = todo.get("objective", "")
        expected_output = todo.get("expected_output", "")
        guidance = todo.get("guidance", "")

        context_section = ""
        if main_document_context:
            context_section = f"\n**Relevant Context from Shared Research:**\n{main_document_context}\n"

        notes_section = ""
        if previous_notes:
            notes_section = f"\n**Your Previous Notes:**\n{previous_notes}\n"

        prompt = f"""**Task:** {title}

**Objective:**
{objective}

**Expected Output:**
{expected_output}

**Guidance:**
{guidance}
{context_section}{notes_section}

Begin your research. Remember to go DEEP, use multiple sources, and verify key facts."""

        return prompt

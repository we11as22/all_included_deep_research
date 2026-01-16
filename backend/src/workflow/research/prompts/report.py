"""Report prompt builder for final report generation."""

from src.workflow.research.prompts.base import PromptBuilder


class ReportPromptBuilder(PromptBuilder):
    """Build prompts for final report generation."""

    def build_report_prompt(
        self,
        query: str,
        compressed_findings: str,
        draft_report: str,
        user_language: str = "English",
        clarification_context: str = "",
    ) -> str:
        """Build prompt for generating final research report.

        Args:
            query: Original user query
            compressed_findings: Compressed/synthesized findings
            draft_report: Draft report from supervisor
            user_language: Language for report
            clarification_context: User clarification answers

        Returns:
            Complete report generation prompt
        """
        compressed_summary = self._truncate(compressed_findings, max_length=3000)
        draft_summary = self._truncate(draft_report, max_length=3000)

        sections = [
            self._format_role(),
            self._format_context(query, user_language, clarification_context),
            self._format_research_findings(compressed_summary, draft_summary),
            self._format_report_guidelines(),
            self._format_structure_guidelines(),
            self._format_quality_standards(),
        ]

        return self._format_sections(sections)

    def _format_role(self) -> str:
        """Format report writer role."""
        return """You are writing the final comprehensive research report."""

    def _format_context(self, query: str, user_language: str, clarification_context: str) -> str:
        """Format context section."""
        clarification_text = ""
        if clarification_context:
            clarification_text = f"\n\n**User Clarification:**\n{clarification_context}\n(This provides additional context about the user's priorities and focus areas)"

        context = f"""**Original Query:** {query}{clarification_text}

**Report Language:** {user_language}
**MANDATORY**: The entire report MUST be written in {user_language}"""

        return self._format_section("Context", context)

    def _format_research_findings(self, compressed_summary: str, draft_summary: str) -> str:
        """Format research findings section."""
        findings = f"""**Synthesized Research Findings:**
{compressed_summary}

**Draft Report from Supervisor:**
{draft_summary}

Use these as your primary source material for the final report."""

        return self._format_section("Research Findings", findings)

    def _format_report_guidelines(self) -> str:
        """Format report writing guidelines."""
        guidelines = """**Report Purpose:**
- Answer the user's query comprehensively
- Present findings in a clear, organized manner
- Provide actionable insights and conclusions
- Maintain academic/professional quality

**Content Principles:**
- Be thorough but concise - no unnecessary filler
- Lead with key insights and conclusions
- Support claims with evidence from research
- Present multiple perspectives when relevant
- Acknowledge limitations and uncertainties

**Tone:**
- Professional and authoritative
- Objective and balanced
- Accessible to educated readers
- Avoid overly technical jargon unless necessary"""

        return self._format_section("Report Guidelines", guidelines)

    def _format_structure_guidelines(self) -> str:
        """Format structure guidelines."""
        structure = """**Report Structure:**

1. **Executive Summary** (2-3 paragraphs)
   - High-level overview of findings
   - Key conclusions and insights
   - Direct answer to the user's query

2. **Main Body** (organized by themes/aspects)
   - Break down into logical sections
   - Each section covers a distinct aspect
   - Use clear headings and subheadings
   - Include relevant details, examples, and evidence
   - Connect findings to the original query

3. **Key Insights** (bullet points or numbered list)
   - Highlight the most important discoveries
   - Actionable conclusions
   - Novel or surprising findings

4. **Sources and References** (if applicable)
   - Cite key sources used in research
   - Provide URLs or references for credibility

**Formatting:**
- Use markdown for formatting
- Clear hierarchy with # ## ### headings
- Bullet points and numbered lists for clarity
- **Bold** for emphasis on key points
- Code blocks ``` for technical content if needed"""

        return self._format_section("Report Structure", structure)

    def _format_quality_standards(self) -> str:
        """Format quality standards."""
        standards = """**Quality Standards:**

✅ **Comprehensive**: Covers all major aspects of the query
✅ **Accurate**: Based on research findings, no speculation
✅ **Well-Organized**: Logical flow and clear structure
✅ **Insightful**: Goes beyond surface-level summaries
✅ **Actionable**: Provides useful conclusions and takeaways
✅ **Properly Cited**: References sources for key claims
✅ **Clear**: Accessible language, well-explained concepts
✅ **Objective**: Balanced perspective, acknowledges limitations

❌ **Avoid:**
- Generic statements without evidence
- Unnecessary repetition
- Disorganized stream-of-consciousness writing
- Missing key aspects from the query
- Overly promotional or biased language
- Unsupported speculation"""

        return self._format_section("Quality Standards", standards)

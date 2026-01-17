"""Report generation node for final research report."""

import structlog
from typing import Dict, Any
from datetime import datetime

from src.workflow.research.state import ResearchState
from src.workflow.research.nodes.base import ResearchNode
from src.workflow.research.models import FinalReport
from src.workflow.research.prompts.report import ReportPromptBuilder

logger = structlog.get_logger(__name__)


class GenerateReportNode(ResearchNode):
    """Generate final research report with validation.

    Uses draft_report.md from supervisor as primary source,
    falls back to main.md and findings if draft is not available.
    """

    async def execute(self, state: ResearchState) -> Dict[str, Any]:
        """Execute report generation node.

        Args:
            state: Current research state

        Returns:
            State updates with final_report
        """
        query = state.get("query", "")
        original_query = state.get("original_query", query)
        findings = state.get("findings", state.get("agent_findings", []))
        compressed_research = state.get("compressed_research", "")
        session_id = state.get("session_id", "unknown")
        chat_history = state.get("chat_history", [])

        # Access dependencies
        llm = self.deps.llm
        stream = self.deps.stream
        agent_memory_service = self.deps.agent_memory_service

        if stream:
            stream.emit_status("📄 Generating final report from draft...", step="report")

        logger.info("Starting final report generation",
                   findings_count=len(findings),
                   session_id=session_id)

        # Read draft_report.md and main.md
        draft_report = await self._read_draft_report(agent_memory_service, findings, query)
        main_document = await self._read_main_document(agent_memory_service)

        # CRITICAL: If draft_report is substantial (>= 1000 chars), return it directly WITHOUT generation
        # Supervisor wrote it throughout research, so it's the final report
        if draft_report and len(draft_report.strip()) >= 1000:
            logger.info("Draft report is substantial - returning it directly as final report (no LLM generation)",
                       draft_length=len(draft_report),
                       session_id=session_id,
                       note="Supervisor wrote this throughout research, using it as final report")
            
            # CRITICAL: Remove metadata from draft_report before returning
            # Remove "Research Report Draft", "Query:", "Started:", "Status:", "Overview" headers
            lines = draft_report.split('\n')
            cleaned_lines = []
            skip_metadata = False
            for i, line in enumerate(lines):
                # Skip "Research Report Draft" header and metadata
                if line.strip() == "# Research Report Draft" or line.strip().startswith("# Research Report Draft"):
                    skip_metadata = True
                    continue
                # Skip metadata lines
                if skip_metadata and (line.strip().startswith("**Query:**") or 
                                     line.strip().startswith("**Started:**") or 
                                     line.strip().startswith("**Status:**") or
                                     line.strip().startswith("**Generated:**") or
                                     line.strip() == "## Overview" or
                                     line.strip().startswith("This is the working draft")):
                    continue
                # Stop skipping after first chapter or section
                if skip_metadata and (line.startswith("## Chapter") or line.startswith("## ")):
                    skip_metadata = False
                if not skip_metadata:
                    cleaned_lines.append(line)
            
            cleaned_draft = '\n'.join(cleaned_lines).strip()
            
            # Format it as a proper report (add title if needed, but only if no chapters)
            if not cleaned_draft.strip().startswith("#") and not cleaned_draft.strip().startswith("## Chapter"):
                formatted_report = f"# Research Report: {original_query}\n\n{cleaned_draft}"
            else:
                formatted_report = cleaned_draft
            
            if stream:
                stream.emit_status("✅ Final report ready (from draft_report)", step="report")

            # Save report to DB if session_manager available
            if self.deps.session_manager:
                try:
                    await self.deps.session_manager.save_final_report(session_id, formatted_report)
                    await self.deps.session_manager.complete_session(session_id, formatted_report)
                    logger.info("Report saved to session in DB", session_id=session_id)
                except Exception as e:
                    logger.error("Failed to save report to session", error=str(e))

            return {
                "final_report": formatted_report,
                "report_generated": True,
                "should_continue": False
            }

        # Draft report is too short or missing - generate report using draft_report + findings summaries
        logger.info("Draft report too short or missing - generating report with draft_report + findings summaries",
                   draft_length=len(draft_report) if draft_report else 0,
                   findings_count=len(findings),
                   session_id=session_id)

        # Determine user language
        # Get user language from state (detected in create_initial_state)
        user_language = state.get("user_language", "English")

        # Extract clarification context
        clarification_context = self._extract_clarification_context(chat_history)

        # Combine draft_report (if exists) with findings summaries for generation
        if draft_report and len(draft_report.strip()) > 0:
            findings_summary = self._format_findings(findings)
            primary_source = f"{draft_report}\n\n## Additional Research Findings\n\n{findings_summary}"
            logger.info("Combining draft_report with findings summaries for generation",
                       draft_length=len(draft_report),
                       findings_count=len(findings),
                       session_id=session_id)
        elif main_document:
            primary_source = main_document
            logger.warning("No draft_report, using main.md as source",
                          session_id=session_id)
        elif compressed_research:
            primary_source = compressed_research
            logger.warning("No draft_report or main.md, using compressed_research as source",
                          session_id=session_id)
        else:
            primary_source = self._format_findings(findings)
            logger.warning("No draft report, main document, or compressed research - using raw findings",
                          session_id=session_id,
                          findings_count=len(findings))

        # Prepare content for prompt (handle very large content)
        draft_report_for_prompt = self._prepare_content_for_prompt(primary_source)

        # Build prompt using prompt builder
        prompt_builder = ReportPromptBuilder()
        prompt = prompt_builder.build_report_prompt(
            query=original_query,
            compressed_findings=draft_report_for_prompt,
            draft_report=draft_report_for_prompt,
            user_language=user_language,
            clarification_context=clarification_context
        )

        try:
            system_prompt = f"""You are an expert research report writer. Generate comprehensive, well-structured reports.

CRITICAL: Write the ENTIRE report in {user_language}.
Minimum report length: 1500 characters.
Include Executive Summary, Main Body (min 3 sections), and Conclusion."""

            report = await llm.with_structured_output(FinalReport).ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ])

            logger.info("Report generated successfully",
                       sections_count=len(report.sections) if hasattr(report, "sections") else 0,
                       session_id=session_id)

            # Format report as markdown
            formatted_report = self._format_report(report, original_query)

            # Validate report length
            if len(formatted_report) < 1500:
                logger.warning("Generated report too short, using combined source as fallback",
                             length=len(formatted_report),
                             session_id=session_id)
                formatted_report = self._create_fallback_report(original_query, draft_report_for_prompt, findings)

            if stream:
                stream.emit_status("✅ Final report generated", step="report")

            # Save report to DB if session_manager available
            if self.deps.session_manager:
                try:
                    await self.deps.session_manager.save_final_report(session_id, formatted_report)
                    await self.deps.session_manager.complete_session(session_id, formatted_report)
                    logger.info("Report saved to session in DB", session_id=session_id)
                except Exception as e:
                    logger.error("Failed to save report to session", error=str(e))

            return {
                "final_report": formatted_report,
                "report_generated": True,
                "should_continue": False
            }

        except Exception as e:
            logger.error("Report generation failed, using combined source as fallback", error=str(e), exc_info=True,
                        session_id=session_id,
                        source_available=bool(draft_report_for_prompt),
                        source_length=len(draft_report_for_prompt) if draft_report_for_prompt else 0)

            # Fallback: use combined source (draft_report + findings or just findings)
            fallback_report = self._create_fallback_report(original_query, draft_report_for_prompt, findings)
            logger.info("Using combined source as fallback report",
                       fallback_length=len(fallback_report),
                       session_id=session_id)

            return {
                "final_report": fallback_report,
                "report_generated": True,
                "should_continue": False
            }

    async def _read_draft_report(self, agent_memory_service: Any, findings: list, query: str) -> str:
        """Read draft_report.md or create from findings if not available.

        Args:
            agent_memory_service: Agent memory service
            findings: All agent findings
            query: Research query

        Returns:
            Draft report content
        """
        if not agent_memory_service:
            logger.warning("No agent_memory_service available")
            return ""

        try:
            draft_report = await agent_memory_service.file_manager.read_file("draft_report.md")
            logger.info("Read draft report", length=len(draft_report))

            # CRITICAL: If draft_report exists and is substantial (>= 1000 chars), use it as primary source
            # If it's too short (< 1000 chars), it means supervisor didn't write much, so we need fallback
            if draft_report and len(draft_report.strip()) >= 1000:
                logger.info("Using draft_report.md as primary source (written by supervisor, substantial length)",
                           draft_length=len(draft_report),
                           note="This is the main report source written by supervisor throughout research")
                return draft_report
            elif draft_report and len(draft_report.strip()) > 0:
                # Draft exists but is too short - supervisor didn't write much
                # Return it anyway, but it will be used WITH findings for generation (fallback)
                logger.warning("Draft report exists but is too short - will use with findings for generation",
                             draft_length=len(draft_report),
                             note="Draft will be combined with findings summaries for report generation")
                return draft_report
            else:
                logger.warning("Draft report is empty, creating from findings as fallback",
                             draft_length=len(draft_report) if draft_report else 0)
                return self._create_draft_from_findings(findings, query)

        except FileNotFoundError:
            logger.warning("Draft report not found, creating from findings")
            draft = self._create_draft_from_findings(findings, query)

            # Save created draft
            try:
                await agent_memory_service.file_manager.write_file("draft_report.md", draft)
                logger.info("Created and saved comprehensive draft report", length=len(draft))
            except Exception as e:
                logger.error("Failed to save created draft", error=str(e))

            return draft

        except Exception as e:
            logger.error("Error reading draft report", error=str(e))
            return ""

    async def _read_main_document(self, agent_memory_service: Any) -> str:
        """Read main.md for additional context.

        Args:
            agent_memory_service: Agent memory service

        Returns:
            Main document content
        """
        if not agent_memory_service:
            return ""

        try:
            main_doc = await agent_memory_service.read_main_file()
            logger.info("Read main document", length=len(main_doc))
            return main_doc
        except Exception as e:
            logger.warning("Could not read main document", error=str(e))
            return ""

    def _create_draft_from_findings(self, findings: list, query: str) -> str:
        """Create comprehensive draft report from all findings.

        Args:
            findings: All agent findings
            query: Research query

        Returns:
            Draft report content
        """
        if not findings:
            return f"# Research Report\n\n**Query:** {query}\n\nNo findings available."

        findings_sections = []
        for f in findings:
            full_summary = f.get('summary', 'No summary')
            all_key_findings = f.get('key_findings', [])
            sources = f.get('sources', [])

            findings_sections.append(f"""## {f.get('topic', 'Unknown Topic')}

**Agent:** {f.get('agent_id', 'unknown')}
**Confidence:** {f.get('confidence', 'unknown')}

### Summary

{full_summary}

### Key Findings

{chr(10).join([f"- {kf}" for kf in all_key_findings]) if all_key_findings else "No key findings"}

### Sources ({len(sources)})

{chr(10).join([f"- {s.get('title', 'Unknown')}: {s.get('url', 'N/A')}" for s in sources[:20]]) if sources else "No sources"}
""")

        findings_text = "\n\n".join(findings_sections)

        draft = f"""# Research Report Draft

**Query:** {query}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Findings:** {len(findings)}

## Executive Summary

This report synthesizes findings from research agents working on: {query}

## Detailed Findings

{findings_text}

## Conclusion

Research completed with {len(findings)} findings from multiple agents covering various aspects of the topic.
"""
        return draft

    def _prepare_content_for_prompt(self, content: str) -> str:
        """Prepare content for prompt (handle very large content).

        Args:
            content: Draft report or findings

        Returns:
            Content prepared for prompt
        """
        max_length = 50000  # characters

        if len(content) <= max_length:
            return content

        # For very large content, use intelligent chunking
        logger.info("Content very large, using first 40k chars + summary",
                   total_length=len(content))

        # Take first 40k chars + note about additional content
        return content[:40000] + f"\n\n[... additional {len(content) - 40000} characters of research content ...]\n\n"

    def _detect_language(self, text: str) -> str:
        """Detect language from text.

        Args:
            text: Text to analyze

        Returns:
            Language name (English, Russian, etc.)
        """
        try:
            from langdetect import detect
            detected = detect(text)
            if detected == "ru":
                return "Russian"
            elif detected == "en":
                return "English"
            else:
                return "English"
        except Exception:
            return "English"

    def _extract_clarification_context(self, chat_history: list) -> str:
        """Extract clarification context from chat history.

        Args:
            chat_history: Chat history

        Returns:
            Clarification context
        """
        if not chat_history:
            return ""

        for i, msg in enumerate(chat_history):
            if msg.get("role") == "assistant":
                content = msg.get("content", "").lower()
                if "clarification" in content or "🔍" in content:
                    if i + 1 < len(chat_history) and chat_history[i + 1].get("role") == "user":
                        return chat_history[i + 1].get("content", "")

        return ""

    def _format_findings(self, findings: list) -> str:
        """Format findings as text.

        Args:
            findings: Agent findings

        Returns:
            Formatted findings text
        """
        return "\n\n".join([
            f"### {f.get('topic')}\n{f.get('summary', '')}\n\nKey findings:\n" +
            "\n".join([f"- {kf}" for kf in f.get('key_findings', [])])
            for f in findings
        ])

    def _format_report(self, report: FinalReport, query: str) -> str:
        """Format report object as markdown.

        Args:
            report: FinalReport object
            query: Original query

        Returns:
            Formatted markdown report
        """
        sections_text = []

        # Executive Summary
        if hasattr(report, "executive_summary") and report.executive_summary:
            sections_text.append(f"## Executive Summary\n\n{report.executive_summary}")

        # Sections
        if hasattr(report, "sections") and report.sections:
            for section in report.sections:
                title = section.title if hasattr(section, "title") else "Section"
                content = section.content if hasattr(section, "content") else ""
                sections_text.append(f"## {title}\n\n{content}")

        # Conclusion
        if hasattr(report, "conclusion") and report.conclusion:
            sections_text.append(f"## Conclusion\n\n{report.conclusion}")

        # Sources
        if hasattr(report, "sources") and report.sources:
            sources_text = "\n".join([f"- {source}" for source in report.sources])
            sections_text.append(f"## Sources\n\n{sources_text}")

        final_report = f"# Research Report: {query}\n\n" + "\n\n".join(sections_text)

        return final_report

    def _create_fallback_report(self, query: str, draft: str, findings: list) -> str:
        """Create fallback report when generation fails.
        
        CRITICAL: This function should use draft_report (written by supervisor) as the main content.
        Findings are only used for metadata (count), not as content source.

        Args:
            query: Original query
            draft: Draft report content (should be draft_report.md written by supervisor)
            findings: Agent findings (used only for metadata, not content)

        Returns:
            Fallback report
        """
        # CRITICAL: If draft is the actual draft_report.md (has chapters), use it directly
        # Don't wrap it in extra structure - supervisor already structured it
        if "## Chapter" in draft or "# Chapter" in draft:
            # This is the structured draft_report from supervisor - use it as-is
            logger.info("Using structured draft_report as fallback (has chapters)",
                       draft_length=len(draft),
                       note="Supervisor wrote this, using it directly without modification")
            return f"# Research Report: {query}\n\n{draft}\n\n---\n\n*Note: This report was generated from the draft report written by the supervisor throughout the research process.*"
        
        # Otherwise, format it as a report
        return f"""# Research Report: {query}

## Executive Summary

This report presents the research findings for: {query}

{draft}

## Total Findings

{len(findings)} findings from research agents.

---

*Note: This is a comprehensive report generated from the draft report written by the supervisor throughout the research process.*
"""


# Legacy function wrapper for backward compatibility
async def generate_final_report_enhanced_node(state: ResearchState) -> Dict:
    """Legacy wrapper for GenerateReportNode.

    This function maintains backward compatibility with existing code
    that imports generate_final_report_enhanced_node directly.

    TODO: Update imports to use GenerateReportNode class directly,
    then remove this wrapper.
    """
    from src.workflow.research.nodes import runtime_deps_context

    runtime_deps = runtime_deps_context.get()
    if not runtime_deps:
        logger.warning("Runtime dependencies not found in context")
        return {
            "final_report": "Error: No runtime dependencies available",
            "report_generated": False,
            "should_continue": False
        }

    # Create dependencies container
    from src.workflow.research.dependencies import ResearchDependencies

    deps = ResearchDependencies(
        llm=runtime_deps.get("llm"),
        search_provider=runtime_deps.get("search_provider"),
        scraper=runtime_deps.get("scraper"),
        stream=runtime_deps.get("stream"),
        agent_memory_service=runtime_deps.get("agent_memory_service"),
        agent_file_service=runtime_deps.get("agent_file_service"),
        session_factory=runtime_deps.get("session_factory"),
        session_manager=runtime_deps.get("session_manager"),
        settings=runtime_deps.get("settings"),
    )

    # Execute node
    node = GenerateReportNode(deps)
    return await node.execute(state)

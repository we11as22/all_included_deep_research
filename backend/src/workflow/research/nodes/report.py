"""Report generation node for final research report."""

import re
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

        # Determine user language
        user_language = state.get("user_language", "English")
        
        # Get clarification answers for title generation
        clarification_answers_from_state = state.get("clarification_answers", "")
        clarification_context = clarification_answers_from_state if clarification_answers_from_state else ""
        
        # Fallback: if not in state, try to extract from chat_history (for backward compatibility)
        if not clarification_context:
            clarification_context = self._extract_clarification_context(chat_history)

        # CRITICAL: If draft_report is substantial (>= 1000 chars), return it directly WITHOUT generation
        # Supervisor wrote it throughout research, so it's the final report
        if draft_report and len(draft_report.strip()) >= 1000:
            logger.info("Draft report is substantial - returning it directly as final report (no LLM generation)",
                       draft_length=len(draft_report),
                       session_id=session_id,
                       note="Supervisor wrote this throughout research, using it as final report")
            
            # CRITICAL: Remove metadata from draft_report before returning
            # Remove "Research Report Draft", "Query:", "Started:", "Status:", "Overview" headers
            # CRITICAL: Also check for duplicate Sources sections at the end - sources are already in chapters
            lines = draft_report.split('\n')
            cleaned_lines = []
            skip_metadata = False
            in_sources_section_at_end = False
            sources_section_start = -1
            
            # First pass: identify if there's a Sources section at the end (after all chapters)
            # This would be a duplicate - sources are already in each chapter
            chapter_count = 0
            last_chapter_line = -1
            for i, line in enumerate(lines):
                if line.startswith("## Chapter"):
                    chapter_count += 1
                    last_chapter_line = i
            
            # Check if there's a "## Sources" section after the last chapter
            if last_chapter_line >= 0:
                for i in range(last_chapter_line + 1, len(lines)):
                    if re.match(r'^##\s+Sources', lines[i], re.IGNORECASE):
                        sources_section_start = i
                        logger.warning("Found duplicate Sources section at end of draft_report (after chapters)",
                                     line_number=i,
                                     note="Sources are already in each chapter - this section will be removed to prevent duplication")
                        break
            
            # Second pass: clean metadata and remove duplicate Sources section
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
                
                # CRITICAL: Skip duplicate Sources section at the end (after all chapters)
                if sources_section_start >= 0 and i >= sources_section_start:
                    # Check if this is the start of Sources section
                    if re.match(r'^##\s+Sources', line, re.IGNORECASE):
                        in_sources_section_at_end = True
                        logger.info("Skipping duplicate Sources section at end",
                                   line_number=i,
                                   note="Sources are already in each chapter - removing duplicate section")
                        continue
                    # Skip all lines in Sources section until next section or end
                    if in_sources_section_at_end:
                        # Check if this is a new section (starts with ##)
                        if re.match(r'^##\s+', line) and not re.match(r'^##\s+Sources', line, re.IGNORECASE):
                            # New section starts - stop skipping
                            in_sources_section_at_end = False
                        else:
                            # Still in Sources section - skip
                            continue
                
                if not skip_metadata:
                    cleaned_lines.append(line)
            
            cleaned_draft = '\n'.join(cleaned_lines).strip()
            
            # Generate report title using LLM based on original query, clarification questions, and answers
            report_title = await self._generate_report_title(
                original_query=original_query,
                clarification_answers=clarification_context,
                llm=llm,
                user_language=user_language
            )
            
            # Format it as a proper report (add title if needed, but only if no chapters)
            # CRITICAL: Check if draft already has a title to avoid duplication
            first_line = cleaned_draft.strip().split('\n')[0] if cleaned_draft.strip() else ""
            has_title = first_line.startswith("# ") and not first_line.startswith("## ")
            
            if not has_title and not cleaned_draft.strip().startswith("## Chapter"):
                formatted_report = f"# {report_title}\n\n{cleaned_draft}"
            else:
                # If draft has title, replace it with generated title to avoid duplication
                if has_title:
                    # Remove existing title and add new one
                    lines = cleaned_draft.split('\n')
                    # Skip first line if it's a title
                    if lines[0].startswith("# ") and not lines[0].startswith("## "):
                        content_lines = lines[1:]
                    else:
                        content_lines = lines
                    formatted_report = f"# {report_title}\n\n" + '\n'.join(content_lines)
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
            
            # CRITICAL: Clear research memories at the end of deep research
            if stream and hasattr(stream, "app_state"):
                app_state = stream.app_state
                if isinstance(app_state, dict):
                    research_memory_service = app_state.get("research_memory_service") or app_state.get("_research_memory_service")
                else:
                    research_memory_service = getattr(app_state, "research_memory_service", None) or getattr(app_state, "_research_memory_service", None)
                
                if research_memory_service:
                    try:
                        deleted_count = await research_memory_service.clear_session_memories(session_id)
                        logger.info("Cleared research memories at end of deep research",
                                   session_id=session_id,
                                   deleted_count=deleted_count)
                    except Exception as e:
                        logger.warning("Failed to clear research memories at end", error=str(e))

            # CRITICAL: Return format must match original - only final_report and confidence
            # Draft report is substantial (>= 1000 chars), so confidence is high
            return {
                "final_report": formatted_report,
                "confidence": "high"  # Draft report is substantial, written by supervisor throughout research
            }

        # Draft report is too short or missing - generate report using draft_report + findings summaries
        logger.info("Draft report too short or missing - generating report with draft_report + findings summaries",
                   draft_length=len(draft_report) if draft_report else 0,
                   findings_count=len(findings),
                   session_id=session_id)

        # Determine user language (already set above)
        # clarification_context already set above

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
            formatted_report = await self._format_report(report, query, original_query, clarification_context, llm, user_language)

            # Validate report length
            if len(formatted_report) < 1500:
                logger.warning("Generated report too short, using combined source as fallback",
                             length=len(formatted_report),
                             session_id=session_id)
                formatted_report = await self._create_fallback_report(original_query, draft_report_for_prompt, findings, state)

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
            
            # CRITICAL: Clear research memories at the end of deep research
            if stream and hasattr(stream, "app_state"):
                app_state = stream.app_state
                if isinstance(app_state, dict):
                    research_memory_service = app_state.get("research_memory_service") or app_state.get("_research_memory_service")
                else:
                    research_memory_service = getattr(app_state, "research_memory_service", None) or getattr(app_state, "_research_memory_service", None)
                
                if research_memory_service:
                    try:
                        deleted_count = await research_memory_service.clear_session_memories(session_id)
                        logger.info("Cleared research memories at end of deep research",
                                   session_id=session_id,
                                   deleted_count=deleted_count)
                    except Exception as e:
                        logger.warning("Failed to clear research memories at end", error=str(e))

            # CRITICAL: Return format must match original - only final_report and confidence
            # Original backup returns: {"final_report": ..., "confidence": report.confidence_level}
            return {
                "final_report": formatted_report,
                "confidence": report.confidence_level if hasattr(report, "confidence_level") else ("medium" if draft_report_for_prompt else "low")
            }

        except Exception as e:
            logger.error("Report generation failed, using combined source as fallback", error=str(e), exc_info=True,
                        session_id=session_id,
                        source_available=bool(draft_report_for_prompt),
                        source_length=len(draft_report_for_prompt) if draft_report_for_prompt else 0)

            # Fallback: use combined source (draft_report + findings or just findings)
            fallback_report = await self._create_fallback_report(original_query, draft_report_for_prompt, findings, state)
            logger.info("Using combined source as fallback report",
                       fallback_length=len(fallback_report),
                       session_id=session_id)

            # CRITICAL: Return format must match original - only final_report and confidence
            # Original backup returns: {"final_report": ..., "confidence": "medium" if draft_report else "low"}
            return {
                "final_report": fallback_report,
                "confidence": "medium" if draft_report_for_prompt else "low"
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
                return await self._create_draft_from_findings(findings, query, state)

        except FileNotFoundError:
            logger.warning("Draft report not found, creating from findings")
            draft = await self._create_draft_from_findings(findings, query, state)

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

    async def _create_draft_from_findings(self, findings: list, query: str, state: Dict[str, Any]) -> str:
        """Create comprehensive draft report from all findings.

        Args:
            findings: All agent findings
            query: Research query

        Returns:
            Draft report content
        """
        if not findings:
            # Generate report title
            user_language = state.get("user_language", "English")
            clarification_answers = state.get("clarification_answers", "")
            report_title = await self._generate_report_title(
                original_query=query,
                clarification_answers=clarification_answers,
                llm=self.deps.llm,
                user_language=user_language
            )
            return f"# {report_title}\n\n**Query:** {query}\n\nNo findings available."

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

    async def _generate_report_title(
        self,
        original_query: str,
        clarification_answers: str,
        llm: Any,
        user_language: str = "English"
    ) -> str:
        """Generate report title using LLM based on original query and clarification.
        
        Args:
            original_query: Original user query
            clarification_answers: User clarification answers (if provided)
            llm: LLM instance
            user_language: User's language
            
        Returns:
            Generated report title
        """
        from pydantic import BaseModel, Field
        
        class ReportTitle(BaseModel):
            reasoning: str = Field(description="Why this title was chosen")
            title: str = Field(description="Report title (without 'Research Report:' prefix, just the title itself)")
        
        prompt = f"""Generate a concise, descriptive title for a research report.

**Original User Query:** {original_query}

**Clarification Answers (if provided):**
{clarification_answers if clarification_answers else "No clarification provided"}

**Requirements:**
1. The title should be in {user_language} - the same language as the user's query
2. The title should be concise (5-15 words) but descriptive
3. The title should reflect the main topic of the research
4. If clarification was provided, incorporate it into the title
5. Do NOT include "Research Report:" prefix - just the title itself
6. The title should be professional and informative

**Examples:**
- Query: "расскажи про развитие видеокарт" → Title: "Развитие видеокарт: от появления до перспективных разработок"
- Query: "AI in healthcare" → Title: "Artificial Intelligence in Healthcare: Current Applications and Future Prospects"
- Query: "climate change effects" → Title: "Climate Change Effects: Environmental, Economic, and Social Impacts"

Return structured output with reasoning at the beginning."""
        
        try:
            result = await llm.with_structured_output(ReportTitle).ainvoke([
                {"role": "system", "content": f"You are an expert at creating concise, descriptive titles. Always create titles in {user_language}."},
                {"role": "user", "content": prompt}
            ])
            
            logger.info("Report title generated",
                       original_query=original_query[:100],
                       generated_title=result.title,
                       user_language=user_language)
            
            return result.title
        except Exception as e:
            logger.warning("Failed to generate report title with LLM, using fallback",
                          error=str(e),
                          original_query=original_query[:100])
            # Fallback: use original query as title
            return original_query[:100] if original_query else "Research Report"
    
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

    async def _format_report(self, report: FinalReport, query: str, original_query: str, clarification_context: str, llm: Any, user_language: str) -> str:
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

        # CRITICAL: Do NOT add Sources section if draft_report already has sources in chapters
        # Sources are already added to each chapter in draft_report by supervisor
        # Adding Sources section here would duplicate sources that are already in chapters
        # Only add Sources section if report was generated from scratch (not from draft_report)
        # Check if draft_report was used - if so, skip Sources section (sources are in chapters)
        # Note: This method is only called when draft_report is too short and LLM generates report
        # In that case, if LLM includes sources in FinalReport.sources, we should NOT add them
        # because they might duplicate sources from draft_report chapters
        # Skip Sources section to avoid duplication
        # if hasattr(report, "sources") and report.sources:
        #     sources_text = "\n".join([f"- {source}" for source in report.sources])
        #     sections_text.append(f"## Sources\n\n{sources_text}")

        # Generate report title
        report_title = await self._generate_report_title(
            original_query=original_query,
            clarification_answers=clarification_context,
            llm=llm,
            user_language=user_language
        )
        
        final_report = f"# {report_title}\n\n" + "\n\n".join(sections_text)

        return final_report

    async def _create_fallback_report(self, query: str, draft: str, findings: list, state: Dict[str, Any]) -> str:
        """Create fallback report when generation fails.
        
        CRITICAL: This function should use draft_report (written by supervisor) as the main content.
        Findings are only used for metadata (count), not as content source.

        Args:
            query: Original query
            draft: Draft report content (should be draft_report.md written by supervisor)
            findings: Agent findings (used only for metadata, not content)
            state: Research state (for user_language and clarification_answers)

        Returns:
            Fallback report
        """
        # Generate report title
        user_language = state.get("user_language", "English")
        clarification_answers = state.get("clarification_answers", "")
        report_title = await self._generate_report_title(
            original_query=query,
            clarification_answers=clarification_answers,
            llm=self.deps.llm,
            user_language=user_language
        )
        
        # CRITICAL: If draft is the actual draft_report.md (has chapters), use it directly
        # Don't wrap it in extra structure - supervisor already structured it
        if "## Chapter" in draft or "# Chapter" in draft:
            # This is the structured draft_report from supervisor - use it as-is
            logger.info("Using structured draft_report as fallback (has chapters)",
                       draft_length=len(draft),
                       note="Supervisor wrote this, using it directly without modification")
            
            # CRITICAL: Remove duplicate Sources section at the end if present
            # Sources are already in each chapter - any Sources section at the end is a duplicate
            lines = draft.split('\n')
            cleaned_lines = []
            in_sources_section_at_end = False
            sources_section_start = -1
            
            # Find last chapter line
            last_chapter_line = -1
            for i, line in enumerate(lines):
                if line.startswith("## Chapter") or line.startswith("# Chapter"):
                    last_chapter_line = i
            
            # Check if there's a "## Sources" section after the last chapter
            if last_chapter_line >= 0:
                for i in range(last_chapter_line + 1, len(lines)):
                    if re.match(r'^##\s+Sources', lines[i], re.IGNORECASE):
                        sources_section_start = i
                        logger.warning("Found duplicate Sources section at end of draft in fallback (after chapters)",
                                     line_number=i,
                                     note="Sources are already in each chapter - this section will be removed to prevent duplication")
                        break
            
            # Process lines: remove title and duplicate Sources section
            for i, line in enumerate(lines):
                # Skip first line if it's a title
                if i == 0 and line.startswith("# ") and not line.startswith("## "):
                    continue
                
                # Skip duplicate Sources section at the end
                if sources_section_start >= 0 and i >= sources_section_start:
                    if re.match(r'^##\s+Sources', line, re.IGNORECASE):
                        in_sources_section_at_end = True
                        logger.info("Skipping duplicate Sources section at end in fallback",
                                   line_number=i,
                                   note="Sources are already in each chapter - removing duplicate section")
                        continue
                    if in_sources_section_at_end:
                        # Check if this is a new section
                        if re.match(r'^##\s+', line) and not re.match(r'^##\s+Sources', line, re.IGNORECASE):
                            in_sources_section_at_end = False
                        else:
                            continue
                
                cleaned_lines.append(line)
            
            content = '\n'.join(cleaned_lines)
            return f"# {report_title}\n\n{content}\n\n---\n\n*Note: This report was generated from the draft report written by the supervisor throughout the research process.*"
        
        # Otherwise, format it as a report
        return f"""# {report_title}

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
        # CRITICAL: Return format must match original - only final_report and confidence
        return {
            "final_report": "Error: No runtime dependencies available",
            "confidence": "low"
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

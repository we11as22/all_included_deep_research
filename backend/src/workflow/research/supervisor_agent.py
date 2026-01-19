"""Supervisor agent as LangGraph agent with ReAct format and memory tools.

The supervisor is a full LangGraph agent that:
- Reviews agent findings and updates main research document
- Creates and edits agent todos
- Identifies research gaps
- Makes decisions about continuing/replanning/finishing
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List
import structlog

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.workflow.research.models import (
    SupervisorAssessment,
    AgentDirective,
    ResearchGap,
)
from src.models.agent_models import AgentTodoItem

logger = structlog.get_logger(__name__)


# ==================== Supervisor Tools Schema ====================


class ReadMainDocumentArgs(BaseModel):
    """Arguments for reading main research document."""
    max_length: int = Field(default=5000, description="Maximum characters to read")


class WriteMainDocumentArgs(BaseModel):
    """Arguments for writing/updating main research document."""
    content: str = Field(description="Content to append to main document")
    section_title: str = Field(description="Title for this section")


class CreateAgentTodoArgs(BaseModel):
    """Arguments for creating new todo for an agent."""
    agent_id: str = Field(description="Target agent ID")
    reasoning: str = Field(description="Why this task is needed")
    title: str = Field(description="Task title")
    objective: str = Field(description="What to achieve")
    expected_output: str = Field(description="Expected result")
    priority: str = Field(default="medium", description="Priority: high/medium/low")
    guidance: str = Field(description="Specific guidance for the agent")


class UpdateAgentTodoArgs(BaseModel):
    """Arguments for updating existing todo for an agent."""
    agent_id: str = Field(description="Target agent ID")
    todo_title: str = Field(description="Title of the existing todo to update")
    status: str = Field(default=None, description="New status (pending, in_progress, done)")
    objective: str = Field(default=None, description="Updated objective")
    expected_output: str = Field(default=None, description="Updated expected result")
    guidance: str = Field(default=None, description="Updated guidance")
    priority: str = Field(default=None, description="Updated priority: high/medium/low")
    reasoning: str = Field(default=None, description="Updated reasoning")


class ReviewAgentProgressArgs(BaseModel):
    """Arguments for reviewing specific agent's progress."""
    agent_id: str = Field(description="Agent ID to review")


class MakeFinalDecisionArgs(BaseModel):
    """Arguments for making final research decision."""
    reasoning: str = Field(description="Analysis of current research state")
    decision: str = Field(description="Decision: continue/replan/finish")


# ==================== Supervisor Tools Handlers ====================


async def read_main_document_handler(args: Dict[str, Any], context: Dict[str, Any]) -> Dict:
    """
    Read main research document - KEY INSIGHTS ONLY.
    
    CRITICAL: main.md should only contain essential shared information.
    This is a SHARED document - keep it minimal and focused on key insights only.
    """
    agent_memory_service = context.get("agent_memory_service")
    
    if not agent_memory_service:
        return {"error": "Memory service not available"}
    
    try:
        content = await agent_memory_service.read_main_file()
        max_length = args.get("max_length", 3000)  # Reduced default - main.md should be small
        
        # Extract only key sections - skip items section which can be huge
        # Focus on Overview and key insights
        lines = content.split("\n")
        key_sections = []
        current_section = []
        in_items_section = False
        
        for line in lines:
            if line.startswith("## Items"):
                in_items_section = True
                continue
            elif line.startswith("## ") and in_items_section:
                in_items_section = False
                current_section.append(line)
            elif not in_items_section:
                current_section.append(line)
                if line.startswith("## "):
                    if current_section:
                        key_sections.append("\n".join(current_section[:-1]))
                    current_section = [line]
        
        if current_section and not in_items_section:
            key_sections.append("\n".join(current_section))
        
        # Combine key sections (Overview, key insights, etc.)
        filtered_content = "\n\n".join(key_sections)
        
        # Limit size
        if len(filtered_content) > max_length:
            from src.utils.text import summarize_text
            preview = summarize_text(filtered_content, max_length)
        else:
            preview = filtered_content
            
        return {
            "content": preview,
            "full_length": len(content),
            "filtered_length": len(filtered_content),
            "truncated": len(filtered_content) > max_length,
            "note": "Main document filtered to show only key insights (items section excluded - items are in items/ directory)"
        }
    except Exception as e:
        logger.error("Failed to read main document", error=str(e))
        return {"error": str(e)}


async def write_main_document_handler(args: Dict[str, Any], context: Dict[str, Any]) -> Dict:
    """
    Write/update main research document with KEY INSIGHTS ONLY.
    
    CRITICAL: main.md should only contain key insights and progress updates, NOT all items.
    Items are stored in items/ directory. Main.md is for supervisor's key findings.
    """
    agent_memory_service = context.get("agent_memory_service")
    
    if not agent_memory_service:
        return {"error": "Memory service not available"}
    
    try:
        content = args.get("content", "")
        section_title = args.get("section_title", "Update")
        
        # Read current content
        current = await agent_memory_service.read_main_file()
        
        # CRITICAL: If updating Research Plan section, replace it instead of appending
        if section_title == "Research Plan" or section_title.lower() == "research plan":
            import re
            # Replace existing Research Plan section if it exists
            pattern = r"## Research Plan.*?(?=\n## |\Z)"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_plan_section = f"""## Research Plan

**Updated:** {timestamp}

{content}

---
**Note:** This research plan can be updated by the supervisor as research progresses.
"""
            if re.search(pattern, current, re.DOTALL):
                # Replace existing section
                updated = re.sub(pattern, new_plan_section.strip(), current, flags=re.DOTALL)
                logger.info("Research Plan section updated in main.md")
            else:
                # Append new section if not found
                updated = current + "\n\n" + new_plan_section
                logger.info("Research Plan section added to main.md")
        else:
            # Create structured update with key insights only
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            update = f"\n\n---\n\n## {section_title} - {timestamp}\n\n{content}\n"
            
            # Append to document
            updated = current + update
        
        # Limit main.md size - if too large, summarize older sections
        if len(updated) > 50000:  # ~50KB limit for main.md
            logger.warning("Main document too large, summarizing older content")
            from src.utils.text import summarize_text
            # Summarize everything except last 3 sections using simple truncation
            sections = updated.split("\n\n---\n\n")
            if len(sections) > 4:
                old_sections = "\n\n---\n\n".join(sections[:-3])
                summary = summarize_text(old_sections, 2000)
                updated = f"# Agent Memory - Main Index\n\n## Overview\n\n{summary}\n\n---\n\n" + "\n\n---\n\n".join(sections[-3:])
        
        await agent_memory_service.file_manager.write_file("main.md", updated)
        
        logger.info("Main document updated", section=section_title, content_length=len(content), total_length=len(updated))
        
        return {
            "success": True,
            "new_length": len(updated),
            "section": section_title
        }
    except Exception as e:
        logger.error("Failed to write main document", error=str(e))
        return {"error": str(e)}


async def write_draft_report_handler(args: Dict[str, Any], context: Dict[str, Any]) -> Dict:
    """
    Write/update draft research report (draft_report.md).
    
    CRITICAL: Draft report is structured by chapters (one chapter = one finding).
    Each finding from an agent becomes a new chapter in the draft report.
    Supervisor adds chapters iteratively as findings arrive.
    
    This is the supervisor's working document where the final report is assembled.
    
    Context available:
    - query: Original user query
    - deep_search_result: Initial deep search result
    - clarification_context: User clarification answers
    - chapter_summaries: Summaries of existing chapters (to avoid repetition)
    """
    agent_memory_service = context.get("agent_memory_service")
    session_id = context.get("session_id")
    session_factory = context.get("session_factory")
    query = context.get("query", "")
    deep_search_result = context.get("deep_search_result", "")
    clarification_context = context.get("clarification_context", "")
    chapter_summaries = context.get("chapter_summaries", [])
    
    if not agent_memory_service:
        return {"error": "Memory service not available"}
    
    try:
        content = args.get("content", "")
        # Support both chapter_title and section_title (for backward compatibility)
        chapter_title = args.get("chapter_title") or args.get("section_title", "Chapter")
        
        # CRITICAL: Clean chapter_title from any duplicate headers that LLM might have added
        # Remove patterns like "## Chapter 1: ## Chapter 1: Title" -> "Title"
        # LLM should NOT include headers in chapter_title, but sometimes does - remove them completely
        import re
        if chapter_title:
            # Remove ALL "## Chapter N:" patterns from title (can be multiple, anywhere in string)
            # First remove from start (most common case)
            while True:
                new_title = re.sub(r'^#+\s*Chapter\s+\d+:\s*', '', chapter_title, flags=re.IGNORECASE)
                if new_title == chapter_title:
                    break
                chapter_title = new_title
            # Also remove from anywhere in the string (in case LLM puts it in middle/end)
            chapter_title = re.sub(r'#+\s*Chapter\s+\d+:\s*', '', chapter_title, flags=re.IGNORECASE)
            # Remove standalone "## Chapter" or "# Chapter" (with or without number)
            chapter_title = re.sub(r'^#+\s*Chapter\s*:?\s*', '', chapter_title, flags=re.IGNORECASE)
            chapter_title = re.sub(r'#+\s*Chapter\s*:?\s*', '', chapter_title, flags=re.IGNORECASE)
            chapter_title = chapter_title.strip()
        
        # Validate chapter_title - must not be empty
        if not chapter_title or chapter_title == "Chapter":
            # Try to extract title from content if chapter_title is empty
            if content:
                # Try to find first meaningful line in content
                content_lines = content.split('\n')
                for line in content_lines[:5]:
                    line = line.strip()
                    if line and not line.startswith('#') and len(line) > 10:
                        chapter_title = line[:100]  # Use first meaningful line as title
                        break
            
            # If still empty, use fallback
            if not chapter_title or chapter_title == "Chapter":
                chapter_title = f"Research Finding {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                logger.warning("Chapter title was empty, using fallback", fallback_title=chapter_title)
        
        finding_data_raw = args.get("finding", None)  # Optional: full finding data for chapter summary
        
        # CRITICAL: Ensure finding_data is a dict, not a string
        # Findings should always be dict after structured output from researcher
        # If string is passed, try to find corresponding finding from state
        finding_data = None
        if finding_data_raw:
            if isinstance(finding_data_raw, dict):
                finding_data = finding_data_raw
            elif isinstance(finding_data_raw, str):
                # String passed - try to find corresponding finding from state by topic/chapter_title
                # This handles cases where LLM passes topic string instead of full finding dict
                findings_from_context = context.get("findings", [])
                if findings_from_context and chapter_title:
                    # Try to match by chapter_title (normalized)
                    chapter_title_normalized = chapter_title.strip().lower()
                    for f in findings_from_context:
                        if isinstance(f, dict):
                            # Match by topic (most common case)
                            topic = f.get("topic", "").strip().lower()
                            if topic == chapter_title_normalized:
                                finding_data = f
                                logger.info("Found finding from state by chapter_title",
                                           chapter_title=chapter_title,
                                           finding_topic=f.get("topic", "unknown"),
                                           note="String finding_data_raw was replaced with full finding dict from state")
                                break
                            # Also try matching finding_data_raw string with topic
                            if finding_data_raw.strip().lower() == topic:
                                finding_data = f
                                logger.info("Found finding from state by matching finding_data_raw string with topic",
                                           finding_data_raw_preview=finding_data_raw[:50],
                                           finding_topic=f.get("topic", "unknown"),
                                           note="String finding_data_raw was replaced with full finding dict from state")
                                break
                
                # If still not found, try JSON parse as fallback
                if not finding_data:
                    try:
                        import json
                        finding_data = json.loads(finding_data_raw)
                        logger.info("Parsed finding_data from JSON string",
                                   note="String was valid JSON, parsed to dict")
                    except (json.JSONDecodeError, TypeError):
                        logger.warning("finding_data is a string but not valid JSON and not found in state findings",
                                     finding_data_preview=finding_data_raw[:100] if finding_data_raw else None,
                                     findings_count=len(findings_from_context) if findings_from_context else 0,
                                     note="Will use fallback from state.findings if available")
                        finding_data = None
            else:
                logger.warning("finding_data has unexpected type, treating as None",
                             finding_data_type=type(finding_data_raw).__name__)
                finding_data = None
        
        # CRITICAL: If finding_data is still None, try to find it from context findings by chapter_title
        # This ensures we always have finding data if it exists in state
        if not finding_data and chapter_title:
            findings_from_context = context.get("findings", [])
            if findings_from_context:
                chapter_title_normalized = chapter_title.strip().lower()
                for f in findings_from_context:
                    if isinstance(f, dict):
                        topic = f.get("topic", "").strip().lower()
                        if topic == chapter_title_normalized:
                            finding_data = f
                            logger.info("Found finding from state by chapter_title (fallback)",
                                       chapter_title=chapter_title,
                                       finding_topic=f.get("topic", "unknown"),
                                       note="Finding was not passed in args, but found in context.findings")
                            break
        
        # CRITICAL: Log context for debugging
        logger.info("write_draft_report called - context available",
                   chapter_title=chapter_title,
                   query_preview=query[:100] if query else "None",
                   deep_search_length=len(deep_search_result) if deep_search_result else 0,
                   deep_search_preview=deep_search_result[:200] if deep_search_result else "None",
                   clarification_length=len(clarification_context) if clarification_context else 0,
                   clarification_preview=clarification_context[:200] if clarification_context else "None",
                   existing_chapters=len(chapter_summaries),
                   chapter_summaries_preview=[ch.get("chapter_title", "Unknown") if isinstance(ch, dict) else str(ch)[:50] for ch in chapter_summaries[-5:]],
                   finding_data_type=type(finding_data).__name__ if finding_data else "None",
                   note="Supervisor has access to: query, deep_search, clarification, existing chapters - use this to write adapted chapter")
        
        # CRITICAL: Build context summary for supervisor to see when writing chapter
        # This will be included in the tool response so supervisor knows what context is available
        context_summary = {
            "query": query,
            "deep_search_preview": deep_search_result if deep_search_result else "",
            "clarification_preview": clarification_context[:300] if clarification_context else "",
            "existing_chapters_count": len(chapter_summaries),
            "existing_chapters_summaries": [
                {
                    "chapter_number": ch.get("chapter_number") if isinstance(ch, dict) else None,
                    "chapter_title": ch.get("chapter_title", "Unknown") if isinstance(ch, dict) else str(ch)[:50],
                    "topic": ch.get("topic", "Unknown") if isinstance(ch, dict) else None,
                    "summary_preview": ch.get("summary", "")[:200] if isinstance(ch, dict) else ""
                }
                for ch in chapter_summaries[-5:]  # Last 5 chapters
            ]
        }
        
        # Read current draft report
        draft_file = "draft_report.md"
        try:
            current = await agent_memory_service.file_manager.read_file(draft_file)
        except FileNotFoundError:
            # Create initial draft report with structure for chapters
            # CRITICAL: Use datetime from module-level import, not local variable
            query = context.get('query', 'Unknown')
            from datetime import datetime as dt_module
            # CRITICAL: Create clean draft report WITHOUT metadata - just start with chapters
            # Metadata will be removed from final report anyway
            current = ""
        
        # Create new chapter (not just a section update)
        # CRITICAL: Simple structure - just chapter title and content, no metadata
        # CRITICAL: Parse existing chapters to find maximum chapter number (not just count)
        # This prevents duplicate chapter numbers when multiple agents write simultaneously
        # NOTE: Regex is used ONLY for numbering, NOT for duplicate detection
        # Duplicate detection is done via chapter_summaries (primary) and draft_report.md (fallback)
        import re
        # Match "## Chapter N:" pattern (must be at start of line)
        chapter_pattern = r'^##\s+Chapter\s+(\d+):'
        existing_chapter_numbers = []
        for line in current.split('\n'):
            match = re.match(chapter_pattern, line, re.IGNORECASE)
            if match:
                try:
                    chapter_num = int(match.group(1))
                    existing_chapter_numbers.append(chapter_num)
                except (ValueError, IndexError):
                    pass
        
        # Also check for "# Chapter" format (single #) for numbering
        single_hash_pattern = r'^#\s+Chapter\s+(\d+):'
        for line in current.split('\n'):
            match = re.match(single_hash_pattern, line, re.IGNORECASE)
            if match:
                try:
                    chapter_num = int(match.group(1))
                    existing_chapter_numbers.append(chapter_num)
                except (ValueError, IndexError):
                    pass
        
        # CRITICAL: Check for duplicate chapter titles
        # STRICT LOGIC: Only check chapter_summaries - this is the authoritative source
        # chapter_summaries is automatically updated when chapters are added, so it's always accurate
        # No need for fallback checks with regex - that's redundant and error-prone
        chapter_title_normalized = chapter_title.strip().lower()
        
        # Check chapter_summaries first
        if chapter_summaries:
            for ch in chapter_summaries:
                if isinstance(ch, dict):
                    existing_title = ch.get("chapter_title", "").strip().lower()
                    if existing_title and existing_title == chapter_title_normalized:
                        logger.warning("Chapter with this title already exists in chapter_summaries - skipping duplicate",
                                     chapter_title=chapter_title,
                                     existing_title=ch.get("chapter_title", "Unknown"),
                                     note="Supervisor has access to existing chapters and should not add duplicates")
                        return {
                            "success": False,
                            "message": f"Chapter '{chapter_title}' already exists in draft report. Check chapter_summaries before adding new chapters.",
                            "chapter_number": None,
                        }
        
        # Get next chapter number (max + 1, or 1 if no chapters exist)
        if existing_chapter_numbers:
            chapter_number = max(existing_chapter_numbers) + 1
        else:
            chapter_number = 1
        
        logger.info("Calculated chapter number",
                   existing_chapters=existing_chapter_numbers,
                   next_chapter=chapter_number,
                   chapter_summaries_count=len(chapter_summaries),
                   note="Parsed existing chapters to find max number, checked for duplicates via chapter_summaries")
        
        # CRITICAL: Extract sources from finding_data if available, add at end of chapter
        # If finding_data not provided or doesn't have sources, try to find finding in state's findings
        sources_section = ""
        sources = []
        
        # First, try to get sources from finding_data
        if finding_data and isinstance(finding_data, dict) and finding_data.get("sources"):
            sources = finding_data.get("sources", [])
            logger.info("Found sources in finding_data",
                       sources_count=len(sources),
                       chapter_title=chapter_title)
        else:
            # Fallback: try to find finding in state's findings by matching topic/chapter_title
            # This ensures sources are added even if supervisor didn't pass finding parameter
            findings_from_state = context.get("findings", [])
            if findings_from_state:
                # First, try to match by topic or chapter_title (exact or partial match)
                matched_finding = None
                for f in findings_from_state:
                    if isinstance(f, dict):
                        finding_topic = f.get("topic", "")
                        finding_title = f.get("title", "")
                        # Match if topic or title matches chapter_title (case-insensitive, partial match)
                        if (finding_topic and chapter_title and 
                            (finding_topic.lower() in chapter_title.lower() or 
                             chapter_title.lower() in finding_topic.lower())) or \
                           (finding_title and chapter_title and 
                            (finding_title.lower() in chapter_title.lower() or 
                             chapter_title.lower() in finding_title.lower())):
                            if f.get("sources"):
                                matched_finding = f
                                logger.info("Found sources in state findings by matching topic/title",
                                           sources_count=len(f.get("sources", [])),
                                           chapter_title=chapter_title,
                                           finding_topic=finding_topic,
                                           note="Sources extracted from state findings as fallback")
                                break
                
                # If no match found, try to find any finding with sources that hasn't been added yet
                # Check if this finding's topic is already in draft_report as a chapter
                if not matched_finding:
                    try:
                        current_draft = await agent_memory_service.file_manager.read_file("draft_report.md")
                        for f in findings_from_state:
                            if isinstance(f, dict) and f.get("sources"):
                                finding_topic = f.get("topic", "")
                                # Check if this finding's topic is already in draft_report
                                if finding_topic and finding_topic not in current_draft:
                                    matched_finding = f
                                    logger.info("Found sources in state findings (finding not yet added as chapter)",
                                               sources_count=len(f.get("sources", [])),
                                               chapter_title=chapter_title,
                                               finding_topic=finding_topic,
                                               note="Using first finding with sources that hasn't been added yet")
                                    break
                    except Exception as e:
                        logger.warning("Failed to check draft_report for finding matching", error=str(e))
                
                if matched_finding and matched_finding.get("sources"):
                    sources = matched_finding.get("sources", [])
        
        # Format sources section
        # CRITICAL: Deduplicate sources by URL to prevent duplicate sources in the same chapter
        if sources:
            seen_source_urls = set()
            sources_list = []
            for source in sources[:30]:  # Limit to 30 sources per chapter
                # Ensure source is a dict before calling .get()
                if isinstance(source, dict):
                    title = source.get("title", "Unknown")
                    url = source.get("url", "")
                    # Normalize URL for deduplication (lowercase, strip trailing slashes)
                    url_normalized = url.lower().rstrip('/') if url else ""
                    
                    # Skip if we've already seen this URL
                    if url_normalized and url_normalized in seen_source_urls:
                        continue
                    
                    if url_normalized:
                        seen_source_urls.add(url_normalized)
                    
                    if url:
                        sources_list.append(f"- [{title}]({url})")
                    else:
                        # Also check for duplicate titles if no URL
                        existing_titles = [s.split(']')[0].replace('- [', '').lower() for s in sources_list]
                        if title.lower() not in existing_titles:
                            sources_list.append(f"- {title}")
                elif isinstance(source, str):
                    # If source is a string (URL), check for duplicates
                    url_normalized = source.lower().rstrip('/')
                    if url_normalized not in seen_source_urls:
                        seen_source_urls.add(url_normalized)
                        sources_list.append(f"- {source}")
            
            if sources_list:
                sources_section = f"\n\n## Sources\n\n" + "\n".join(sources_list) + "\n"
                logger.info("Sources section created",
                           sources_count=len(sources_list),
                           original_sources_count=len(sources),
                           chapter_title=chapter_title,
                           note="Sources deduplicated and added at end of chapter")
            else:
                logger.warning("Sources list is empty after formatting and deduplication",
                             sources_count=len(sources),
                             chapter_title=chapter_title)
        else:
            logger.warning("No sources found for chapter",
                         chapter_title=chapter_title,
                         finding_data_provided=bool(finding_data),
                         finding_data_has_sources=bool(finding_data and isinstance(finding_data, dict) and finding_data.get("sources")),
                         note="Sources will NOT be added to this chapter")
        
        # CRITICAL: Clean content from any chapter headers that LLM might have added
        # LLM should NOT add headers, but sometimes does - remove them to prevent duplication
        # Remove any "## Chapter" or "# Chapter" lines from content
        content_lines = content.split('\n')
        cleaned_content_lines = []
        for line in content_lines:
            # Skip lines that look like chapter headers with number: "## Chapter 1:", "# Chapter 1:", etc.
            # Also match patterns like "## Chapter 1: ## Chapter 1:" (duplicate headers)
            if re.match(r'^#+\s*Chapter\s+\d+:', line, re.IGNORECASE):
                logger.warning("Removed duplicate chapter header from content",
                             line=line[:100],
                             chapter_title=chapter_title,
                             note="LLM added header but it's added automatically - removed to prevent duplication")
                continue
            # Skip lines that are just "## Chapter" or "# Chapter" without number
            if re.match(r'^#+\s*Chapter\s*:?\s*$', line, re.IGNORECASE):
                logger.warning("Removed duplicate chapter header from content",
                             line=line[:100],
                             chapter_title=chapter_title,
                             note="LLM added header but it's added automatically - removed to prevent duplication")
                continue
            # Skip lines that are "## Chapter: Title" or "# Chapter: Title" (without number but with title)
            if re.match(r'^#+\s*Chapter\s*:\s+', line, re.IGNORECASE):
                logger.warning("Removed duplicate chapter header from content (without number)",
                             line=line[:100],
                             chapter_title=chapter_title,
                             note="LLM added header but it's added automatically - removed to prevent duplication")
                continue
            # Also remove lines that are empty or just separators after chapter headers
            if line.strip() == "" and len(cleaned_content_lines) > 0 and cleaned_content_lines[-1].strip() == "":
                # Skip multiple empty lines
                continue
            cleaned_content_lines.append(line)
        cleaned_content = '\n'.join(cleaned_content_lines).strip()
        
        # Format chapter with clean structure - no metadata, just title, content, and sources
        # CRITICAL: Use ONLY "## Chapter N: Title" format (two #, not one #)
        # This is the ONLY allowed format - no variations!
        # Sources are added automatically at the end - LLM is instructed not to write them in content
        chapter = f"""

---

## Chapter {chapter_number}: {chapter_title}

{cleaned_content}{sources_section}

"""
        
        # Append chapter to draft
        updated = current + chapter
        await agent_memory_service.file_manager.write_file(draft_file, updated)
        
        # CRITICAL: Store chapter summary in session_metadata for fallback synthesis
        if finding_data and session_id and session_factory:
            try:
                from src.workflow.research.session.manager import SessionManager
                session_manager = SessionManager(session_factory)
                
                # Get current session metadata
                session_data = await session_manager.get_session(session_id)
                # CRITICAL: session_data is ResearchSessionModel object, not dict - use getattr
                current_metadata = getattr(session_data, "session_metadata", None) if session_data else {}
                if not isinstance(current_metadata, dict):
                    current_metadata = {}
                
                # Initialize chapter_summaries if not exists
                if "chapter_summaries" not in current_metadata:
                    current_metadata["chapter_summaries"] = []
                
                # Create chapter summary (full, not heavily truncated)
                # CRITICAL: datetime is already imported at module level, use it directly
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                chapter_summary = {
                    "chapter_number": chapter_number,
                    "chapter_title": chapter_title,
                    "timestamp": timestamp,
                    "agent_id": finding_data.get("agent_id", "unknown") if finding_data and isinstance(finding_data, dict) else "unknown",
                    "topic": finding_data.get("topic", chapter_title) if finding_data and isinstance(finding_data, dict) else chapter_title,
                    "summary": finding_data.get("summary", content[:500]) if finding_data and isinstance(finding_data, dict) else content[:500],
                    "key_findings": finding_data.get("key_findings", [])[:10] if finding_data and isinstance(finding_data, dict) else [],  # First 10 key findings
                    "sources_count": len(finding_data.get("sources", [])) if finding_data and isinstance(finding_data, dict) else 0,
                    "content_preview": content[:1000]  # First 1000 chars of content
                }
                
                current_metadata["chapter_summaries"].append(chapter_summary)
                
                # Update session metadata
                from sqlalchemy import update
                from src.database.schema import ResearchSessionModel
                async with session_factory() as session:
                    await session.execute(
                        update(ResearchSessionModel)
                        .where(ResearchSessionModel.id == session_id)
                        .values(session_metadata=current_metadata, updated_at=datetime.now())
                    )
                    await session.commit()
                
                logger.info("Chapter summary stored in session metadata",
                           chapter_number=chapter_number,
                           summaries_count=len(current_metadata["chapter_summaries"]))
            except Exception as e:
                logger.warning("Failed to store chapter summary in session metadata", error=str(e))
        
        logger.info("Draft report chapter added", 
                   chapter_number=chapter_number,
                   chapter_title=chapter_title,
                   content_length=len(content),
                   total_length=len(updated),
                   context_used={
                       "query": bool(query),
                       "deep_search": bool(deep_search_result),
                       "clarification": bool(clarification_context),
                       "existing_chapters": len(chapter_summaries)
                   })
        
        # Return success with context info for supervisor
        return {
            "success": True,
            "new_length": len(updated),
            "chapter_number": chapter_number,
            "chapter_title": chapter_title,
            "context_available": {
                "query": query[:100] if query else "",
                "has_deep_search": bool(deep_search_result),
                "has_clarification": bool(clarification_context),
                "existing_chapters": len(chapter_summaries)
            }
        }
    except Exception as e:
        logger.error("Failed to write draft report chapter", error=str(e))
        return {"error": str(e)}


async def update_synthesized_report_handler(args: Dict[str, Any], context: Dict[str, Any]) -> Dict:
    """
    Update the SYNTHESIZED REPORT section of draft_report.md.

    This is the main tool for supervisor to write structured report based on RAW findings.
    Replaces/updates the "SUPERVISOR SYNTHESIZED REPORT" section.
    """
    agent_memory_service = context.get("agent_memory_service")

    if not agent_memory_service:
        return {"error": "Memory service not available"}

    try:
        synthesized_content = args.get("content", "")
        mark_raw_as_processed = args.get("mark_raw_as_processed", False)
        processed_cycle = args.get("processed_cycle", None)  # Which RAW FINDINGS cycle was processed

        draft_file = "draft_report.md"
        try:
            current = await agent_memory_service.file_manager.read_file(draft_file)
        except FileNotFoundError:
            return {"error": "Draft report not found. Cannot update synthesized section."}

        # Find the SUPERVISOR SYNTHESIZED REPORT section
        synth_marker = "## 📝 SUPERVISOR SYNTHESIZED REPORT"
        raw_marker = "## 🔍 RAW FINDINGS"

        if synth_marker not in current:
            return {"error": "SUPERVISOR SYNTHESIZED REPORT section not found in draft"}

        # Split into parts
        parts = current.split(synth_marker)
        before_synth = parts[0] + synth_marker
        after_synth = parts[1]

        # Find where synthesized section ends (at first RAW FINDINGS marker)
        if raw_marker in after_synth:
            synth_section_end = after_synth.index(raw_marker)
            after_raw = after_synth[synth_section_end:]
        else:
            # No RAW FINDINGS yet
            after_raw = ""

        # Build new synthesized section
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_synth_section = f"""

**Last Updated:** {timestamp}

{synthesized_content}

---
"""

        # Handle marking RAW findings as processed
        if mark_raw_as_processed and processed_cycle is not None:
            # Mark specific cycle as processed
            cycle_marker = f"## 🔍 RAW FINDINGS - Cycle {processed_cycle}"
            if cycle_marker in after_raw:
                # Replace status line
                after_raw = after_raw.replace(
                    "**Status:** Awaiting supervisor synthesis",
                    f"**Status:** ✅ Processed by supervisor at {timestamp}"
                )

        # Reconstruct draft
        updated = before_synth + new_synth_section + after_raw
        await agent_memory_service.file_manager.write_file(draft_file, updated)

        logger.info("Synthesized report section updated",
                   content_length=len(synthesized_content),
                   marked_processed=mark_raw_as_processed,
                   cycle=processed_cycle)

        return {
            "success": True,
            "content_length": len(synthesized_content),
            "marked_cycle_processed": processed_cycle if mark_raw_as_processed else None
        }
    except Exception as e:
        logger.error("Failed to update synthesized report", error=str(e), exc_info=True)
        return {"error": str(e)}


async def read_draft_report_handler(args: Dict[str, Any], context: Dict[str, Any]) -> Dict:
    """Read draft research report."""
    agent_memory_service = context.get("agent_memory_service")

    if not agent_memory_service:
        return {"error": "Memory service not available"}
    
    try:
        draft_file = "draft_report.md"
        max_length = args.get("max_length", 10000)
        
        try:
            content = await agent_memory_service.file_manager.read_file(draft_file)
        except FileNotFoundError:
            return {
                "content": "Draft report not yet created.",
                "full_length": 0,
                "truncated": False
            }
        
        if len(content) > max_length:
            preview = content[:max_length] + f"\n\n[... truncated {len(content) - max_length} characters]"
        else:
            preview = content
            
        return {
            "content": preview,
            "full_length": len(content),
            "truncated": len(content) > max_length
        }
    except Exception as e:
        logger.error("Failed to read draft report", error=str(e))
        return {"error": str(e)}


async def read_supervisor_file_handler(args: Dict[str, Any], context: Dict[str, Any]) -> Dict:
    """Read supervisor's personal file (agents/supervisor.md) with notes and observations."""
    agent_file_service = context.get("agent_file_service")
    
    if not agent_file_service:
        return {"error": "File service not available"}
    
    try:
        supervisor_file = await agent_file_service.read_agent_file("supervisor")
        max_length = args.get("max_length", 5000)
        
        # Format supervisor file content
        notes = supervisor_file.get("notes", [])
        notes_text = "\n".join([f"- {note}" for note in notes[-20:]]) if notes else "No notes yet."
        
        content = f"""# Supervisor Personal File

## Notes
{notes_text}

## Character
{supervisor_file.get("character", "Research supervisor coordinating team of agents")}

## Preferences
{supervisor_file.get("preferences", "Focus on comprehensive, diverse research coverage")}
"""
        
        if len(content) > max_length:
            preview = content[:max_length] + f"\n\n[... truncated {len(content) - max_length} characters]"
        else:
            preview = content
            
        return {
            "content": preview,
            "full_length": len(content),
            "truncated": len(content) > max_length,
            "notes_count": len(notes)
        }
    except Exception as e:
        logger.error("Failed to read supervisor file", error=str(e))
        return {"error": str(e)}


async def write_supervisor_note_handler(args: Dict[str, Any], context: Dict[str, Any]) -> Dict:
    """
    Write note to supervisor's personal file (agents/supervisor.md).
    
    Use this for your personal observations, thoughts, and notes about the research process.
    This is YOUR file - use it to track your thinking, not to store everything in main.md.
    """
    agent_file_service = context.get("agent_file_service")
    
    if not agent_file_service:
        return {"error": "File service not available"}
    
    try:
        note_text = args.get("note", "")
        if not note_text:
            return {"error": "Note text is required"}
        
        # Read current supervisor file
        supervisor_file = await agent_file_service.read_agent_file("supervisor")
        existing_notes = supervisor_file.get("notes", [])
        
        # Add timestamp to note
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_note = f"[{timestamp}] {note_text}"
        
        # Add to notes (keep last 100 notes)
        existing_notes.append(formatted_note)
        existing_notes = existing_notes[-100:]
        
        # Update supervisor file
        await agent_file_service.write_agent_file(
            agent_id="supervisor",
            notes=existing_notes,
            character=supervisor_file.get("character", "Research supervisor coordinating team of agents"),
            preferences=supervisor_file.get("preferences", "Focus on comprehensive, diverse research coverage")
        )
        
        logger.info("Supervisor note written", note_length=len(note_text), total_notes=len(existing_notes))
        
        return {
            "success": True,
            "notes_count": len(existing_notes)
        }
    except Exception as e:
        logger.error("Failed to write supervisor note", error=str(e))
        return {"error": str(e)}


async def create_agent_todo_handler(args: Dict[str, Any], context: Dict[str, Any]) -> Dict:
    """Create new todo for an agent. If agent doesn't exist, create it with basic characteristics."""
    agent_file_service = context.get("agent_file_service")
    
    if not agent_file_service:
        return {"error": "File service not available"}
    
    # CRITICAL: Check supervisor call limit - TODO operations are limited
    # Findings processing (write_draft_report, update_synthesized_report) is NOT limited
    state = context.get("state", {})
    supervisor_call_count = state.get("supervisor_call_count", 0)
    settings = context.get("settings")
    if not settings:
        from src.config.settings import get_settings
        settings = get_settings()
    max_supervisor_calls = settings.deep_research_max_supervisor_calls
    
    if supervisor_call_count >= max_supervisor_calls:
        logger.warning(f"Supervisor TODO limit reached ({supervisor_call_count}/{max_supervisor_calls}) - cannot create new todos, but findings processing continues",
                      supervisor_calls=supervisor_call_count, max_calls=max_supervisor_calls)
        return {
            "error": f"Supervisor call limit reached ({supervisor_call_count}/{max_supervisor_calls}). Cannot create new todos, but findings processing (write_draft_report, update_synthesized_report) continues.",
            "supervisor_calls": supervisor_call_count,
            "max_calls": max_supervisor_calls,
            "note": "Findings processing is not limited - supervisor can still process findings and write to draft_report"
        }
    
    try:
        agent_id = args.get("agent_id")
        
        # CRITICAL: Check maximum agent count from settings
        max_agents = getattr(settings, "deep_research_num_agents", 3)
        
        # Extract agent number from agent_id (e.g., "agent_4" -> 4)
        try:
            agent_num = int(agent_id.replace("agent_", "")) if "agent_" in agent_id else None
        except:
            agent_num = None
        
        # CRITICAL: Prevent creating agents beyond the limit
        if agent_num and agent_num > max_agents:
            logger.warning(f"Attempted to create agent {agent_id} but limit is {max_agents} agents. Use existing agents (agent_1 to agent_{max_agents}) instead.",
                          agent_id=agent_id, max_agents=max_agents)
            return {
                "error": f"Cannot create agent {agent_id}. Maximum {max_agents} agents allowed. Use existing agents (agent_1 to agent_{max_agents}) or update_agent_todo to modify existing tasks.",
                "max_agents": max_agents,
                "suggestion": f"Use update_agent_todo to modify tasks for existing agents (agent_1 to agent_{max_agents})"
            }
        
        # Read current agent file (returns empty structure if agent doesn't exist)
        agent_file = await agent_file_service.read_agent_file(agent_id)
        current_todos = agent_file.get("todos", [])
        character = agent_file.get("character", "")
        preferences = agent_file.get("preferences", "")
        
        # If agent doesn't exist (no character), create basic characteristics
        is_new_agent = not character
        if is_new_agent:
            # Extract agent number from agent_id (e.g., "agent_2" -> "2")
            agent_num_str = agent_id.replace("agent_", "") if "agent_" in agent_id else "?"
            character = f"""**Role**: Research Agent {agent_num_str}
**Expertise**: General research and analysis
**Personality**: Thorough, analytical, detail-oriented
"""
            preferences = "Focus on comprehensive research coverage and accuracy."
            logger.info(f"Creating new agent {agent_id} with basic characteristics")
        
        # Create new todo
        new_todo = AgentTodoItem(
            reasoning=args.get("reasoning", ""),
            title=args.get("title", ""),
            objective=args.get("objective", ""),
            expected_output=args.get("expected_output", ""),
            sources_needed=[],
            priority=args.get("priority", "medium"),
            status="pending",
            note=args.get("guidance", "")
        )
        
        current_todos.append(new_todo)
        
        # Write updated todos
        await agent_file_service.write_agent_file(
            agent_id=agent_id,
            todos=current_todos,
            character=character,
            preferences=preferences
        )
        
        logger.info("Created agent todo", agent_id=agent_id, title=new_todo.title, is_new_agent=is_new_agent, total_todos=len(current_todos))
        
        # CRITICAL: Emit updated todos to frontend so user sees new tasks immediately
        stream = context.get("stream")
        if stream and current_todos:
            todos_dict = [
                {
                    "title": t.title,
                    "status": t.status,
                    "objective": t.objective,
                    "expected_output": t.expected_output,
                    "note": t.note,
                    "url": t.url if hasattr(t, "url") else None
                }
                for t in current_todos
            ]
            stream.emit_agent_todo(agent_id, todos_dict)
            logger.info(f"Agent {agent_id} todos emitted to frontend after supervisor created new task", 
                       todos_count=len(todos_dict),
                       new_task=new_todo.title)
        
        return {
            "success": True,
            "agent_id": agent_id,
            "todo_title": new_todo.title,
            "total_todos": len(current_todos),
            "is_new_agent": is_new_agent
        }
    except Exception as e:
        logger.error("Failed to create agent todo", error=str(e))
        return {"error": str(e)}


async def update_agent_todo_handler(args: Dict[str, Any], context: Dict[str, Any]) -> Dict:
    """Update existing todo for an agent."""
    agent_file_service = context.get("agent_file_service")
    
    if not agent_file_service:
        return {"error": "File service not available"}
    
    # CRITICAL: Check supervisor call limit - TODO operations are limited
    # Findings processing (write_draft_report, update_synthesized_report) is NOT limited
    state = context.get("state", {})
    supervisor_call_count = state.get("supervisor_call_count", 0)
    settings = context.get("settings")
    if not settings:
        from src.config.settings import get_settings
        settings = get_settings()
    max_supervisor_calls = settings.deep_research_max_supervisor_calls
    
    if supervisor_call_count >= max_supervisor_calls:
        logger.warning(f"Supervisor TODO limit reached ({supervisor_call_count}/{max_supervisor_calls}) - cannot update todos, but findings processing continues",
                      supervisor_calls=supervisor_call_count, max_calls=max_supervisor_calls)
        return {
            "error": f"Supervisor call limit reached ({supervisor_call_count}/{max_supervisor_calls}). Cannot update todos, but findings processing (write_draft_report, update_synthesized_report) continues.",
            "supervisor_calls": supervisor_call_count,
            "max_calls": max_supervisor_calls,
            "note": "Findings processing is not limited - supervisor can still process findings and write to draft_report"
        }
    
    try:
        agent_id = args.get("agent_id")
        todo_title = args.get("todo_title")
        
        # Read current agent file
        agent_file = await agent_file_service.read_agent_file(agent_id)
        current_todos = agent_file.get("todos", [])
        
        # Find and update todo
        updated = False
        for todo in current_todos:
            if todo.title == todo_title:
                # CRITICAL: Protect done tasks - they are immutable once completed
                # No one can modify or delete done tasks - they are permanent record
                if todo.status == "done":
                    logger.warning(f"Supervisor attempted to modify done task '{todo_title}' for agent {agent_id}. Done tasks are immutable and cannot be changed.",
                                 agent_id=agent_id, todo_title=todo_title, note="Done tasks are permanent records")
                    return {
                        "error": f"Cannot modify done task '{todo_title}' for agent {agent_id}. Done tasks are immutable and cannot be changed.",
                        "note": "Done tasks are permanent records of completed work"
                    }
                
                # CRITICAL: Protect in_progress tasks from status changes by supervisor
                # Supervisor should not change status of tasks that agents are currently working on
                # This prevents race conditions where supervisor changes status while agent is working
                if "status" in args and args.get("status"):
                    new_status = args["status"]
                    # CRITICAL: Do not allow supervisor to change status of in_progress tasks
                    # (except to done, which agent will do itself, or if explicitly needed)
                    if todo.status == "in_progress" and new_status != "done":
                        logger.warning(f"Supervisor attempted to change status of in_progress task '{todo_title}' for agent {agent_id} from in_progress to {new_status}. Ignoring status change to prevent race condition.",
                                     agent_id=agent_id, todo_title=todo_title, current_status=todo.status, attempted_status=new_status)
                        # Don't update status, but allow other fields to be updated
                    else:
                        todo.status = new_status
                
                # Allow updating other fields even for in_progress tasks
                # (objective, guidance can be refined while agent works, but agent uses cached current_task)
                if "objective" in args and args.get("objective"):
                    todo.objective = args["objective"]
                if "expected_output" in args and args.get("expected_output"):
                    todo.expected_output = args["expected_output"]
                if "guidance" in args and args.get("guidance"):
                    todo.note = args["guidance"]
                if "priority" in args and args.get("priority"):
                    todo.priority = args["priority"]
                if "reasoning" in args and args["reasoning"]:
                    todo.reasoning = args["reasoning"]
                updated = True
                break
        
        if not updated:
            return {"error": f"Todo '{todo_title}' not found for agent {agent_id}"}
        
        # Write updated todos
        await agent_file_service.write_agent_file(
            agent_id=agent_id,
            todos=current_todos,
            character=agent_file.get("character", ""),
            preferences=agent_file.get("preferences", "")
        )
        
        logger.info("Updated agent todo", agent_id=agent_id, todo_title=todo_title)
        
        # CRITICAL: Emit updated todos to frontend so user sees task updates immediately
        stream = context.get("stream")
        if stream and current_todos:
            todos_dict = [
                {
                    "title": t.title,
                    "status": t.status,
                    "objective": t.objective,
                    "expected_output": t.expected_output,
                    "note": t.note,
                    "url": t.url if hasattr(t, "url") else None
                }
                for t in current_todos
            ]
            stream.emit_agent_todo(agent_id, todos_dict)
            logger.info(f"Agent {agent_id} todos emitted to frontend after supervisor updated task", 
                       todos_count=len(todos_dict),
                       updated_task=todo_title)
        
        return {
            "success": True,
            "agent_id": agent_id,
            "todo_title": todo_title,
            "total_todos": len(current_todos)
        }
    except Exception as e:
        logger.error("Failed to update agent todo", error=str(e))
        return {"error": str(e)}


async def review_agent_progress_handler(args: Dict[str, Any], context: Dict[str, Any]) -> Dict:
    """Review specific agent's progress."""
    agent_file_service = context.get("agent_file_service")
    agent_memory_service = context.get("agent_memory_service")
    
    if not agent_file_service:
        return {"error": "File service not available"}
    
    try:
        agent_id = args.get("agent_id")
        
        # Read agent file
        agent_file = await agent_file_service.read_agent_file(agent_id)
        todos = agent_file.get("todos", [])
        
        # Get agent's notes count (but don't include content to prevent context bloat)
        all_notes = agent_file.get("notes", [])
        
        # Get items count (but don't include full content)
        items = await agent_memory_service.list_items() if agent_memory_service else []
        agent_items_count = len([item for item in items if agent_id in item.get("file_path", "")])
        
        # Calculate progress
        total_todos = len(todos)
        completed_todos = sum(1 for t in todos if t.status == "done")
        pending_todos = sum(1 for t in todos if t.status == "pending")
        in_progress_todos = sum(1 for t in todos if t.status == "in_progress")
        
        summary = {
            "agent_id": agent_id,
            "role": agent_file.get("character", ""),
            "total_todos": total_todos,
            "completed": completed_todos,
            "pending": pending_todos,
            "in_progress": in_progress_todos,
            "progress_percent": (completed_todos / total_todos * 100) if total_todos > 0 else 0,
            "notes_count": len(all_notes),
            "items_count": agent_items_count,  # Total items in items/ directory
            "current_todos": [
                {
                    "title": t.title,
                    "status": t.status,
                    "objective": t.objective,
                    "note": t.note
                }
                for t in todos
            ]
        }
        
        logger.info("Reviewed agent progress", agent_id=agent_id, progress=summary["progress_percent"])
        
        return summary
    except Exception as e:
        logger.error("Failed to review agent progress", error=str(e))
        return {"error": str(e)}


async def make_final_decision_handler(args: Dict[str, Any], context: Dict[str, Any]) -> Dict:
    """Make final decision about research continuation."""
    reasoning = args.get("reasoning", "")
    decision = args.get("decision", "continue")
    
    # Get current iteration from state
    state = context.get("state", {})
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 25)
    
    # Force finish if max iterations reached
    if iteration >= max_iterations:
        logger.warning(f"Max iterations reached ({iteration}/{max_iterations}), forcing finish")
        decision = "finish"
        reasoning = f"{reasoning}\n\n[FORCED] Max iterations reached ({iteration}/{max_iterations}), research must finish."
    
    should_continue = decision == "continue"
    replanning_needed = decision == "replan"
    
    logger.info("Supervisor decision", decision=decision, should_continue=should_continue, iteration=iteration, max_iterations=max_iterations, reasoning=reasoning[:200])
    
    return {
        "should_continue": should_continue,
        "replanning_needed": replanning_needed,
        "reasoning": reasoning,
        "decision": decision
    }


# ==================== Supervisor Tools Registry ====================


class SupervisorToolsRegistry:
    """Registry of supervisor tools."""
    
    _tools = {
        "read_main_document": {
            "name": "read_main_document",
            "description": "Read the main research document to see current progress and findings. "
                          "Returns the document content (may be truncated).",
            "args_schema": {
                "type": "object",
                "properties": {
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum characters to read (default 5000)",
                        "default": 5000
                    }
                },
                # Azure/OpenRouter require all properties to be in required array
                "required": ["max_length"]
            },
            "handler": read_main_document_handler
        },
        "write_main_document": {
            "name": "write_main_document",
            "description": "Write KEY INSIGHTS ONLY to the main research document. "
                          "CRITICAL: Only add key findings and progress updates here, NOT all items. "
                          "Items are stored in items/ directory. Main.md is for supervisor's key insights only. "
                          "Content will be added as a new section with timestamp. "
                          "You can also update the Research Plan section by using section_title='Research Plan'.",
            "args_schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Key insights to add (markdown format) - only important findings, not all details. If section_title is 'Research Plan', this will replace the Research Plan section."
                    },
                    "section_title": {
                        "type": "string",
                        "description": "Title for this section (e.g., 'Key Findings', 'Progress Update', 'Research Plan'). Use 'Research Plan' to update the research plan."
                    }
                },
                "required": ["content", "section_title"]
            },
            "handler": write_main_document_handler
        },
        "write_draft_report": {
            "name": "write_draft_report",
            "description": "Add a new CHAPTER to the draft research report (draft_report.md) based on a finding from an agent. "
                          "CRITICAL: Draft report is structured by chapters - each chapter = one finding from one agent task. "
                          "When an agent completes a task and you receive their finding, you MUST add it as a new chapter. "
                          "Write COMPREHENSIVE content based on the finding - include ALL details, facts, data, and evidence from the finding. "
                          "This file will be used to generate the final report for the user. "
                          "**CRITICAL**: Write DETAILED content based on the finding - include specific facts, dates, numbers, technical details from the finding. "
                          "DO NOT write brief summaries - write FULL, DETAILED chapter with extensive information from the finding. "
                          "Each chapter should be substantial (500-1500 words) and cover the finding comprehensively. "
                          "**CRITICAL FORMAT REQUIREMENT**: Chapter format is STRICTLY '## Chapter N: Title' (two #, space, Chapter, space, number, colon, space, title). "
                          "**FORBIDDEN**: Do NOT use '# Chapter' (single #) or any other format - ONLY '## Chapter N: Title'. "
                          "**FORBIDDEN**: Do NOT add multiple titles for the same chapter - use ONLY '## Chapter N: Title' format, no additional '# Chapter' or '## Title' lines. "
                          "**CRITICAL - DO NOT ADD CHAPTER HEADERS IN chapter_title PARAMETER**: The chapter header '## Chapter N: Title' is added AUTOMATICALLY by the tool. Your 'chapter_title' parameter should contain ONLY the title text (e.g., 'Historical Analysis of Topic X'), NOT the header format (NOT '## Chapter 2: Historical Analysis of Topic X'). The tool will automatically format it as '## Chapter N: [your_title]'. "
                          "**CRITICAL - DO NOT ADD CHAPTER HEADERS IN CONTENT**: The chapter header '## Chapter N: Title' is added AUTOMATICALLY by the tool. Your 'content' parameter should contain ONLY the chapter body text, NOT the chapter header. If you include '## Chapter' or '# Chapter' in your content, it will be removed automatically, but this wastes tokens. Start your content directly with the chapter body text. "
                          "**MANDATORY**: You have access to chapter_summaries in your context which automatically show all existing chapters. Check chapter_summaries BEFORE calling this tool to ensure the chapter title doesn't already exist. If it exists, do NOT add it again - the tool will return an error if you try to add a duplicate. "
                          "**MANDATORY WORKFLOW**: After EACH agent task completion, you MUST call this tool to add the finding as a chapter. "
                          "Draft report is structured by chapters - each chapter = one finding from one agent task. "
                          "When you receive a finding from an agent (in the findings list), you MUST add it as a new chapter - NO EXCEPTIONS. "
                          "**CONTENT REQUIREMENTS**: Write comprehensive, detailed content (1000-2500 words) based on the finding. "
                          "Chapters must be FULL and DETAILED. Include ALL details, facts, data, and evidence from the finding. "
                          "**MARKDOWN FORMAT**: Use proper markdown formatting: '##' for chapter titles (already added automatically - DO NOT include in content), "
                          "'###' for subsections, '**bold**' for emphasis, '*italic*' for emphasis, '-' for lists, '[text](url)' for links. "
                          "**CRITICAL SOURCES RULE**: Sources are added AUTOMATICALLY at the end of each chapter with clickable links in format '- [Title](URL)'. "
                          "**FORBIDDEN**: Do NOT write sources, references, links, or '## Sources' sections in your content - they are automatically added from finding data. "
                          "**FORBIDDEN**: Do NOT include any source lists, reference sections, citation lists, or links in the chapter content. "
                          "**FORBIDDEN**: Do NOT add duplicate sources with the same URL - sources are automatically deduplicated. "
                          "Focus ONLY on writing the chapter content itself - sources will be added automatically. "
                          "**CONTEXT FOR WRITING**: When you call this tool, you have access to: "
                          "1) Original user query (to understand the research goal), "
                          "2) Deep search result (initial context from deep search), "
                          "3) Clarification answers (user's additional requirements), "
                          "4) Summaries of existing chapters (to avoid repetition). "
                          "Use this context to adapt the finding content, avoiding repetition while ensuring NO information is lost. "
                          "Write the chapter as an adapted version of the finding that fits the overall research context. "
                          "**CHAPTER TITLE**: Use chapter_title based on the finding topic. Pass the full finding data in the 'finding' parameter for chapter summary storage.",
            "args_schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Chapter content based on the finding (markdown format, comprehensive and detailed, 1000-2500 words). Use proper markdown: ### for subsections, **bold** for emphasis, *italic* for emphasis, - for lists. **CRITICAL**: Sources are added AUTOMATICALLY at the end of the chapter with clickable links in format '- [Title](URL)'. **FORBIDDEN**: Do NOT write sources, references, or links in the content - they will be automatically added from the finding data. Do NOT include '## Sources', '## References', or any source lists in your content."
                    },
                    "chapter_title": {
                        "type": "string",
                        "description": "Title for this chapter based on the finding topic (e.g., 'Historical Analysis of Topic X', 'Technical Specifications of Y'). REQUIRED. **CRITICAL - DO NOT INCLUDE CHAPTER HEADER IN TITLE**: The chapter header '## Chapter N: Title' is added AUTOMATICALLY by the tool. Your 'chapter_title' parameter should contain ONLY the title text, NOT the header. Do NOT include '## Chapter', '# Chapter', or '## Chapter N:' in the chapter_title - just the title text itself (e.g., 'Historical Analysis of Topic X', NOT '## Chapter 2: Historical Analysis of Topic X'). **CRITICAL**: You have access to chapter_summaries in your context which automatically show all existing chapters. Check chapter_summaries BEFORE calling this tool to ensure this chapter title doesn't already exist. If it exists, do NOT add it again - the tool will return an error if you try to add a duplicate."
                    },
                    "section_title": {
                        "type": "string",
                        "description": "Alias for chapter_title (for backward compatibility). Use chapter_title instead."
                    },
                    "finding": {
                        "type": "object",
                        "description": "Full finding data from the agent (optional, but recommended for chapter summary storage)"
                    }
                },
                "required": ["content"]
            },
            "handler": write_draft_report_handler
        },
        "read_draft_report": {
            "name": "read_draft_report",
            "description": "Read the draft research report (draft_report.md) to see current progress. "
                          "Returns the draft report content (may be truncated).",
            "args_schema": {
                "type": "object",
                "properties": {
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum characters to read (default 10000)",
                        "default": 10000
                    }
                },
                # Azure/OpenRouter require all properties to be in required array
                "required": ["max_length"]
            },
            "handler": read_draft_report_handler
        },
        "update_synthesized_report": {
            "name": "update_synthesized_report",
            "description": "**PRIMARY TOOL** for writing the structured research report. "
                          "Updates the 'SUPERVISOR SYNTHESIZED REPORT' section with your analysis and synthesis of RAW findings. "
                          "This REPLACES the synthesized section (not append). "
                          "Use this after reading RAW FINDINGS to write structured report sections. "
                          "Can mark RAW findings as processed after synthesis.",
            "args_schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Your synthesized report content (markdown format). Should be COMPREHENSIVE and DETAILED with sections like Introduction, Key Findings, Analysis, etc. "
                                      "**CRITICAL**: Include ALL details, facts, dates, numbers, technical specifications, and comprehensive analysis. "
                                      "DO NOT write brief summaries - write FULL, DETAILED sections with extensive information (aim for 1000-3000+ words total). "
                                      "Include ALL information from RAW FINDINGS - don't skip or summarize too much."
                    },
                    "mark_raw_as_processed": {
                        "type": "boolean",
                        "description": "Whether to mark RAW findings as processed (default: false)",
                        "default": False
                    },
                    "processed_cycle": {
                        "type": "integer",
                        "description": "Which RAW FINDINGS cycle was synthesized (e.g., 1, 2, 3). Required if mark_raw_as_processed is true."
                    }
                },
                "required": ["content"]
            },
            "handler": update_synthesized_report_handler
        },
        "read_supervisor_file": {
            "name": "read_supervisor_file",
            "description": "Read YOUR personal file (agents/supervisor.md) with your notes and observations. "
                          "Use this to review your previous thoughts and notes.",
            "args_schema": {
                "type": "object",
                "properties": {
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum characters to read (default 5000)",
                        "default": 5000
                    }
                },
                # Azure/OpenRouter require all properties to be in required array
                "required": ["max_length"]
            },
            "handler": read_supervisor_file_handler
        },
        "write_supervisor_note": {
            "name": "write_supervisor_note",
            "description": "Write note to YOUR personal file (agents/supervisor.md). "
                          "Use this for your personal observations, thoughts, and notes about the research process. "
                          "This is YOUR file - use it to track your thinking, not to store everything in main.md. "
                          "When an agent completes a task, you can write notes about your review, observations, and next steps.",
            "args_schema": {
                "type": "object",
                "properties": {
                    "note": {
                        "type": "string",
                        "description": "Your note or observation (markdown format)"
                    }
                },
                "required": ["note"]
            },
            "handler": write_supervisor_note_handler
        },
        "create_agent_todo": {
            "name": "create_agent_todo",
            "description": "Create a new todo task for a specific research agent. "
                          "**CRITICAL CONTEXT RULE**: Researcher agents DO NOT have access to the original user query or chat history - "
                          "they ONLY see the task you assign. You MUST provide COMPREHENSIVE, EXHAUSTIVE task descriptions. "
                          "**MANDATORY**: Every task MUST include the original user query in the objective or guidance so the agent understands what they're researching. "
                          "**DIVERSIFICATION STRATEGY**: Ensure each agent gets DIFFERENT tasks covering different aspects "
                          "(history, technical, expert views, applications, trends, comparisons, impact, challenges) to build a complete picture. "
                          "Avoid duplicate/overlapping tasks between agents. "
                          "**TASK DISTRIBUTION**: Check each agent's current workload BEFORE creating new todos. "
                          "Prioritize assigning to agents with FEWER tasks. Aim for balanced distribution: each agent should have 2-4 active tasks maximum. "
                          "**AGENT LIMIT**: You have exactly {max_agents} agents (agent_1, agent_2, agent_3). DO NOT create tasks for agent_4, agent_5, etc.! "
                          "**DEEP RESEARCH**: ACTIVELY create multiple follow-up tasks to promote deep research and verification. "
                          "If an agent only provides basic/general information, create MULTIPLE todos forcing them to dig into SPECIFIC details from different angles. "
                          "**CLARIFICATION CONTEXT**: If clarification was provided, interpret it IN THE CONTEXT of the original query - it does NOT replace the original query! "
                          "Clarification specifies WHAT ASPECT of the original topic to focus on, not a new topic. "
                          "**EXAMPLES**: "
                          "1) User query: 'расскажи про историю советской палубной авиации' → Good task: 'Research the history of Soviet carrier aviation. The user asked about [query]. Investigate development, aircraft, key milestones.' "
                          "2) User query: 'обучение моделей qwen', Clarification: 'технические тонкости' → Good task: 'Research Qwen training. The user asked about [query] and wants technical details. Focus on Qwen-specific training: algorithms, hyperparameters, infrastructure.' "
                          "3) User query: 'оформление работников в РФ', Clarification: 'все режимы' → Good task: 'Research employee registration types in Russia. The user asked about [query] and wants all regimes/types. Investigate contracts, part-time, remote work, etc.' "
                          "**BAD EXAMPLES** (avoid these): "
                          "- 'Research history of technology' (ignores original query) "
                          "- 'Research technical details for all models' (ignores Qwen-specific focus) "
                          "- 'Research types of political regimes' (misinterprets 'режимы' out of context)",
            "args_schema": {
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "Agent identifier (e.g., 'agent_1', 'agent_2')"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": """**CRITICAL: Before creating this task, document your thinking in reasoning:**
1. **Original Query**: What is the user asking for? How does this task relate to the original query?
2. **Deep Search Context**: What did the initial deep search reveal? How does this task address those findings?
3. **Clarification Questions & Answers**: What clarification was asked? What did the user answer? How does this task incorporate those answers IN THE CONTEXT of the original query?
4. **Task Necessity**: Why is this specific task needed? What gap does it fill?
5. **Integration**: How do original query, deep search context, and clarification answers come together for this task?
6. **Agent Context**: Remember - the agent will NOT see the original query or chat history. This task description must be self-contained and include all necessary context.

Document your complete thinking process in the reasoning field before defining the task."""
                    },
                    "title": {
                        "type": "string",
                        "description": "Task title"
                    },
                    "objective": {
                        "type": "string",
                        "description": "What the agent should achieve. MUST be COMPREHENSIVE and include: THE ORIGINAL USER QUERY (quote it exactly), specific aspect to research related to that query, why this is important for answering the user's query, and any background information needed. The agent has NO access to dialogue context! Example: 'The user asked: [original query]. Research [specific aspect] because [why it matters for answering the query].'"
                    },
                    "expected_output": {
                        "type": "string",
                        "description": "Expected result format. Be specific about what kind of information is needed (technical specs, expert opinions, case studies, etc.)"
                    },
                    "priority": {
                        "type": "string",
                        "description": "Priority: high/medium/low",
                        "enum": ["high", "medium", "low"],
                        "default": "medium"
                    },
                    "guidance": {
                        "type": "string",
                        "description": "Specific guidance on how to approach this task. MUST include: THE ORIGINAL USER QUERY (quote it exactly: 'The user asked: [query]'), relevant context from deep search result (if available), clarification answers interpreted IN CONTEXT of original query (if provided), what specific information to find related to that query, how to verify findings in multiple sources, and what aspects to investigate deeply. Make it clear how this task helps answer the user's specific question. The agent has NO access to dialogue context - this guidance must be COMPREHENSIVE and self-contained!"
                    }
                },
                # Azure/OpenRouter require all properties to be in required array
                "required": ["agent_id", "title", "objective", "expected_output", "priority", "guidance", "reasoning"]
            },
            "handler": create_agent_todo_handler
        },
        "update_agent_todo": {
            "name": "update_agent_todo",
            "description": "Update an existing todo task for a specific research agent. "
                          "Use this to modify task details, change priority, update guidance, or change status. "
                          "This is OPTIMAL for refining tasks when agents need more specific instructions or when research direction changes.",
            "args_schema": {
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "Agent identifier (e.g., 'agent_1', 'agent_2')"
                    },
                    "todo_title": {
                        "type": "string",
                        "description": "Title of the existing todo to update"
                    },
                    "status": {
                        "type": "string",
                        "description": "New status (pending, in_progress, done)",
                        "enum": ["pending", "in_progress", "done"],
                        "default": ""
                    },
                    "objective": {
                        "type": "string",
                        "description": "Updated objective",
                        "default": ""
                    },
                    "expected_output": {
                        "type": "string",
                        "description": "Updated expected result format",
                        "default": ""
                    },
                    "guidance": {
                        "type": "string",
                        "description": "Updated guidance on how to approach this task",
                        "default": ""
                    },
                    "priority": {
                        "type": "string",
                        "description": "Updated priority: high/medium/low",
                        "enum": ["high", "medium", "low"],
                        "default": ""
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Updated reasoning for why this task is needed",
                        "default": ""
                    }
                },
                # Azure/OpenRouter require all properties to be in required array
                "required": ["agent_id", "todo_title", "status", "objective", "expected_output", "guidance", "priority", "reasoning"]
            },
            "handler": update_agent_todo_handler
        },
        "review_agent_progress": {
            "name": "review_agent_progress",
            "description": "Review specific agent's current progress, todos, and notes. "
                          "Returns detailed status including completed/pending tasks.",
            "args_schema": {
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "Agent identifier to review"
                    }
                },
                "required": ["agent_id"]
            },
            "handler": review_agent_progress_handler
        },
        "make_final_decision": {
            "name": "make_final_decision",
            "description": "Make final decision about whether research should continue, replan, or finish. "
                          "Call this after reviewing agent progress and main document.",
            "args_schema": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Analysis of current research state"
                    },
                    "decision": {
                        "type": "string",
                        "description": "Decision to make",
                        "enum": ["continue", "replan", "finish"]
                    }
                },
                "required": ["reasoning", "decision"]
            },
            "handler": make_final_decision_handler
        }
    }
    
    @classmethod
    def get_tool_definitions(cls) -> List[Dict[str, Any]]:
        """Get tool definitions for LLM (OpenAI format)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_def["description"],
                    "parameters": tool_def["args_schema"]
                }
            }
            for tool_name, tool_def in cls._tools.items()
        ]
    
    @classmethod
    def get_structured_tools(cls, context: Dict[str, Any]) -> List[StructuredTool]:
        """Get StructuredTool objects for LangChain bind_tools."""
        tools = []
        for tool_name, tool_def in cls._tools.items():
            # Create Pydantic model for args
            args_schema = tool_def["args_schema"]
            properties = args_schema.get("properties", {})
            required = args_schema.get("required", [])
            
            # Build field definitions for Pydantic model
            field_definitions = {}
            for prop_name, prop_schema in properties.items():
                field_type = str  # Default to str
                if prop_schema.get("type") == "integer":
                    field_type = int
                elif prop_schema.get("type") == "boolean":
                    field_type = bool
                
                field_info = Field(
                    description=prop_schema.get("description", ""),
                    default=prop_schema.get("default") if prop_name not in required else ...
                )
                field_definitions[prop_name] = (field_type, field_info)
            
            # Create dynamic Pydantic model
            from pydantic import create_model
            ToolArgsModel = create_model(f"{tool_name}_Args", **field_definitions)
            
            # Create async wrapper for handler - use closure to capture handler_func
            handler_func = tool_def["handler"]  # Capture in closure
            async def tool_wrapper(args: ToolArgsModel) -> str:
                args_dict = args.dict() if hasattr(args, "dict") else dict(args)
                result = await handler_func(args_dict, context)
                return json.dumps(result, ensure_ascii=False) if not isinstance(result, str) else result
            
            # Create StructuredTool
            tool = StructuredTool(
                name=tool_name,
                description=tool_def["description"],
                args_schema=ToolArgsModel,
                func=tool_wrapper,
                coroutine=tool_wrapper
            )
            tools.append(tool)
        
        return tools
    
    @classmethod
    async def execute(cls, tool_name: str, args: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute a supervisor tool."""
        if tool_name not in cls._tools:
            raise ValueError(f"Unknown supervisor tool: {tool_name}")
        
        handler = cls._tools[tool_name]["handler"]
        return await handler(args, context)


# ==================== Supervisor Agent Implementation ====================


async def run_supervisor_agent(
    state: Dict[str, Any],
    llm: Any,
    stream: Any,
    supervisor_queue: Any = None,  # Add supervisor_queue parameter
    max_iterations: int = None  # If None, will use settings.deep_research_supervisor_max_iterations (old default: 10)
) -> Dict[str, Any]:
    """
    Run supervisor agent with ReAct format.
    
    Supervisor is a LangGraph agent that:
    - Reviews agent findings
    - Updates main research document
    - Creates new todos for agents
    - Decides whether to continue/replan/finish
    
    Args:
        state: Current research state
        llm: LLM instance
        stream: Stream generator
        max_iterations: Max ReAct iterations
        
    Returns:
        Decision dict with should_continue, replanning_needed, etc.
    """
    query = state.get("query", "")
    # CRITICAL: Log query to ensure it's the original query, not clarification answer
    logger.info("Supervisor agent starting", 
               query=query[:100] if query else None,
               query_length=len(query) if query else 0,
               has_clarification_context=bool(state.get("clarification_context", "")))
    
    # CRITICAL: Extract findings from supervisor_queue if provided
    # This ensures supervisor sees findings even if they're only in queue
    findings = state.get("findings", state.get("agent_findings", []))
    if supervisor_queue and supervisor_queue.size() > 0:
        # Extract findings from queue events
        queue_findings = []
        temp_events = []
        queue_size = supervisor_queue.size()
        for _ in range(queue_size):
            try:
                event = supervisor_queue.queue.get_nowait()
                temp_events.append(event)
                if event.result:
                    queue_findings.append(event.result)
            except:
                break
        
        # Put events back (they'll be processed properly)
        for event in temp_events:
            await supervisor_queue.queue.put(event)
        
        # Combine with existing findings (avoid duplicates)
        if queue_findings:
            for new_finding in queue_findings:
                finding_already_exists = any(
                    f.get("topic") == new_finding.get("topic") and 
                    f.get("agent_id") == new_finding.get("agent_id")
                    for f in findings
                )
                if not finding_already_exists:
                    findings.append(new_finding)
            
            logger.info(f"Extracted {len(queue_findings)} findings from supervisor_queue",
                       total_findings=len(findings),
                       note="Supervisor will process these findings")
    agent_characteristics = state.get("agent_characteristics", {})
    research_plan = state.get("research_plan", {})
    iteration = state.get("iteration", 0)

    # CRITICAL: Check supervisor call count from state
    # IMPORTANT: Supervisor call limit applies ONLY to TODO operations (create_agent_todo, update_agent_todo)
    # Findings processing (write_draft_report, update_synthesized_report) is NOT limited and ALWAYS available
    supervisor_call_count = state.get("supervisor_call_count", 0)
    from src.config.settings import get_settings
    settings = get_settings()
    
    # Get max agents for supervisor prompt
    max_agents = getattr(settings, "deep_research_num_agents", 3)
    max_supervisor_calls = settings.deep_research_max_supervisor_calls
    
    # Check if TODO operations are limited
    todo_operations_limited = supervisor_call_count >= max_supervisor_calls
    
    # CRITICAL: Check if this is a forced finalization call (bypasses limit)
    force_finalization = state.get("_force_supervisor_finalization", False)
    if force_finalization:
        logger.info(f"MANDATORY finalization call - bypassing limit (call {supervisor_call_count + 1}, limit {max_supervisor_calls})",
                   note="This is a special call when all tasks are completed")
    else:
        logger.info(f"Supervisor call {supervisor_call_count + 1}/{max_supervisor_calls}")
    
    # Detect user language from query
    def _detect_user_language(text: str) -> str:
        """Detect user language from query text."""
        if not text:
            return "English"
        # Check for Cyrillic (Russian, Ukrainian, etc.)
        if any('\u0400' <= char <= '\u04FF' for char in text):
            return "Russian"
        # Check for common non-English patterns
        # For now, default to English if not clearly Russian
        return "English"
    
    user_language = _detect_user_language(query)
    
    # Get deep_search_result for context (from initial deep search before multi-agent system)
    # CRITICAL: This contains important background information that must be used to guide research
    deep_search_result_raw = state.get("deep_search_result", "")
    deep_search_result = ""
    
    # Handle multiple formats: dict with "type": "override" and "value", plain dict, or string
    if isinstance(deep_search_result_raw, dict):
        # Check for LangGraph override format: {"type": "override", "value": "..."}
        if "type" in deep_search_result_raw and deep_search_result_raw.get("type") == "override":
            deep_search_result = deep_search_result_raw.get("value", "")
        # Check for plain dict with "value" key
        elif "value" in deep_search_result_raw:
            deep_search_result = deep_search_result_raw.get("value", "")
        else:
            # Try to convert dict to string
            deep_search_result = str(deep_search_result_raw)
    elif isinstance(deep_search_result_raw, str):
        deep_search_result = deep_search_result_raw
    else:
        deep_search_result = str(deep_search_result_raw) if deep_search_result_raw else ""
    
    # Log if deep_search_result is missing (this should not happen in normal flow)
    # CRITICAL: deep_search_result can be empty string if deep search was skipped (continuation after clarification)
    # This is normal for continuations - don't warn if it's a continuation
    if not deep_search_result:
        # Check if this is a continuation (clarification was already sent)
        is_continuation = state.get("clarification_needed", False) or state.get("session_status") == "waiting_clarification"
        if not is_continuation:
            logger.warning("Supervisor: deep_search_result is empty or missing from state", 
                          state_keys=list(state.keys()) if isinstance(state, dict) else "not a dict",
                          has_deep_search_result="deep_search_result" in state,
                          deep_search_result_type=type(deep_search_result_raw).__name__ if deep_search_result_raw else "none",
                          deep_search_value_preview=str(deep_search_result_raw)[:200] if deep_search_result_raw else "none")
        else:
            logger.info("Supervisor: deep_search_result is empty (continuation after clarification - this is normal)",
                       session_status=state.get("session_status"))
    else:
        logger.info("Supervisor: deep_search_result available", 
                   length=len(deep_search_result),
                   preview=deep_search_result[:200] if deep_search_result else None)
    
    # CRITICAL: Log all context data to verify correctness (clarification_context will be set below)
    # This log will be updated after clarification_context is extracted
    
    # Get memory services
    agent_memory_service = stream.app_state.get("agent_memory_service") if stream else None
    agent_file_service = stream.app_state.get("agent_file_service") if stream else None
    
    if not agent_memory_service or not agent_file_service:
        logger.warning("Supervisor: Memory services not available, using fallback")
        return {
            "should_continue": False,
            "replanning_needed": False,
            "gaps_identified": [],
            "iteration": iteration + 1
        }
    
    if stream:
        findings_count = len(findings) if findings else 0
        stream.emit_status(f"Supervisor reviewing iteration #{iteration + 1}... ({findings_count} findings to process)", step="supervisor")
    
    # Build context for supervisor - LIMIT SIZE to prevent context bloat
    # Only include recent findings and summaries, not full details
    from src.utils.text import summarize_text
    
    findings_summary_parts = []
    # Filter findings - only include ones with REAL information, not metadata spam
    useful_findings = []
    for f in findings:
        summary = f.get('summary', '').strip()
        key_findings = f.get('key_findings', [])
        
        # Skip findings that are just metadata
        if not summary or len(summary) < 50:
            continue
        
        summary_lower = summary.lower()
        is_metadata_only = any([
            "found" in summary_lower and "sources" in summary_lower and "query" in summary_lower,
            "completed research" in summary_lower and len(summary) < 100,
            "no substantial findings" in summary_lower,
        ])
        
        # Skip if it's just metadata or has no real findings
        if not is_metadata_only and (len(key_findings) > 0 or len(summary) > 100):
            useful_findings.append(f)
    
    # Include last 15 useful findings (filtered, not all)
    for f in useful_findings[-15:]:
        summary = f.get('summary', '')
        key_findings = f.get('key_findings', [])
        
        # Filter key findings - remove metadata
        filtered_key_findings = []
        for kf in key_findings:
            if isinstance(kf, str):
                kf_lower = kf.lower()
                # Skip metadata findings
                if not ("found" in kf_lower and "sources" in kf_lower):
                    if len(kf) > 30:  # Only meaningful findings
                        filtered_key_findings.append(kf)
        
        # Build findings summary - only real information
        key_findings_str = '\n  - '.join(filtered_key_findings[:10]) if filtered_key_findings else 'No specific findings extracted'
        
        sources_count = f.get('sources_count', 0)
        confidence = f.get('confidence', 'unknown')
        
        # Only include if there's real information
        if summary and len(summary) > 50 and (filtered_key_findings or len(summary) > 150):
            findings_summary_parts.append(
                f"**{f.get('agent_id')}** - {f.get('topic')}:\n"
                f"{summary}\n"
                f"Key findings:\n  - {key_findings_str}\n"
                f"Sources: {sources_count}, Confidence: {confidence}"
            )
    
    findings_summary = "\n\n".join(findings_summary_parts)
    
    # Use smart summarization if too long
    if len(findings_summary) > 3000:
        findings_summary = summarize_text(findings_summary, 3000)  # Smart truncation preserves important info
    
    # Log findings summary for debugging
    logger.info("Supervisor received findings", 
               total_findings=len(findings),
               summary_length=len(findings_summary),
               findings_preview=findings_summary[:500] if findings_summary else "No findings",
               note="Supervisor should add each finding as a new chapter in draft_report using write_draft_report")
    
    # CRITICAL: Add findings to context for supervisor - supervisor needs to see ALL findings to add them as chapters
    # Format findings for supervisor prompt
    findings_for_supervisor = []
    for f in findings:
        findings_for_supervisor.append({
            "agent_id": f.get("agent_id", "unknown"),
            "topic": f.get("topic", "Unknown"),
            "summary": f.get("summary", ""),
            "key_findings": f.get("key_findings", [])[:10],  # First 10 for prompt
            "sources_count": len(f.get("sources", [])),
            "confidence": f.get("confidence", "unknown"),
            "full_finding": f  # Full finding for chapter creation
        })
    
    # CRITICAL: Get clarification context from session state, not chat_history
    # clarification_answers is loaded from DB session in create_initial_state - this is the source of truth
    clarification_context = state.get("clarification_context", "")
    clarification_questions_text = ""
    clarification_answers_from_state = state.get("clarification_answers", "")
    
    if not clarification_context and clarification_answers_from_state and clarification_answers_from_state.strip():
        # Use clarification_answers from session state (source of truth)
        # Note: We don't have clarification questions text in session, but answers are what matter
        clarification_context = f"\n\n**USER CLARIFICATION ANSWERS:**\n{clarification_answers_from_state}\n\n**CRITICAL INTERPRETATION RULES**:\n- The ORIGINAL USER QUERY above is still the PRIMARY topic to research: \"{query}\"\n- This clarification provides additional context about what aspects/depth the user wants to focus on WITHIN the original query topic\n- **MANDATORY**: Clarification MUST be interpreted IN THE CONTEXT of the original query\n- **CRITICAL**: If clarification mentions words that could have multiple meanings, they ALWAYS refer to those words IN THE CONTEXT of the original query\n- **FORBIDDEN**: Do NOT interpret clarification as a standalone query - it's ALWAYS about the original query topic\n- Example: If original query is \"оформление работников\" and clarification says \"все режимы\", this means \"режимы оформления работников\", NOT \"режимы\" in general\n- Use this clarification to understand what depth/angle to focus on, but ALWAYS within the context of the original query topic"
        logger.info("Using clarification_answers from session state for supervisor", 
                   answer_preview=clarification_answers_from_state[:200],
                   original_query=query[:100] if query else None,
                   source="session_state",
                   note="Clarification answers from session state (source of truth), not chat_history")
    
    # Fallback: if not in state, try to extract from chat_history (for backward compatibility)
    if not clarification_context:
        chat_history = state.get("chat_history", [])
        if chat_history:
            for i, msg in enumerate(chat_history):
                if msg.get("role") == "assistant" and ("clarification" in msg.get("content", "").lower() or "🔍" in msg.get("content", "")):
                    # Extract the questions themselves from assistant message
                    assistant_content = msg.get("content", "")
                    if "Clarification Needed" in assistant_content or "🔍" in assistant_content:
                        # Extract questions section (everything from "## 🔍 Clarification Needed" to "---" or end)
                        questions_start = assistant_content.find("## 🔍 Clarification Needed")
                        if questions_start != -1:
                            # Find the end of questions section (before user answers or at "---")
                            questions_end = assistant_content.find("\n---\n", questions_start)
                            if questions_end == -1:
                                questions_end = assistant_content.find("\n\n*Note:", questions_start)
                            if questions_end == -1:
                                questions_end = len(assistant_content)
                            
                            clarification_questions_text = assistant_content[questions_start:questions_end].strip()
                            logger.info("Extracted clarification questions for supervisor (fallback)", 
                                      questions_preview=clarification_questions_text[:300],
                                      clarification_index=i)
                    
                    # Extract user answers
                    if i + 1 < len(chat_history) and chat_history[i + 1].get("role") == "user":
                        user_answer = chat_history[i + 1].get("content", "")
                        if user_answer and user_answer.strip():
                            # Build comprehensive clarification context with both questions and answers
                            questions_section = f"\n\n**CLARIFICATION QUESTIONS ASKED:**\n{clarification_questions_text}\n" if clarification_questions_text else ""
                            clarification_context = f"{questions_section}\n\n**USER CLARIFICATION ANSWERS:**\n{user_answer}\n\n**CRITICAL INTERPRETATION RULES**:\n- The ORIGINAL USER QUERY above is still the PRIMARY topic to research: \"{query}\"\n- This clarification provides additional context about what aspects/depth the user wants to focus on WITHIN the original query topic\n- **MANDATORY**: Clarification MUST be interpreted IN THE CONTEXT of the original query\n- **CRITICAL**: If clarification mentions words that could have multiple meanings, they ALWAYS refer to those words IN THE CONTEXT of the original query\n- **FORBIDDEN**: Do NOT interpret clarification as a standalone query - it's ALWAYS about the original query topic\n- Example: If original query is \"оформление работников\" and clarification says \"все режимы\", this means \"режимы оформления работников\", NOT \"режимы\" in general\n- Use this clarification to understand what depth/angle to focus on, but ALWAYS within the context of the original query topic"
                            logger.warning("Extracted user clarification questions and answers from chat_history (fallback)", 
                                      questions_preview=clarification_questions_text[:200] if clarification_questions_text else "None",
                                      answer_preview=user_answer[:200],
                                      clarification_index=i,
                                      answer_index=i+1,
                                      original_query=query[:100] if query else None,
                                      source="chat_history_fallback",
                                      note="This should not happen - clarification_answers should be in session state")
                        break
        if not clarification_context:
            clarification_context = ""
            logger.info("No clarification context found in session state or chat_history", 
                       has_clarification_answers_in_state=bool(clarification_answers_from_state),
                       chat_history_length=len(chat_history) if chat_history else 0)
    
    # Format chat history to show actual messages from chat
    # For deep_research, use only 2 messages as they can be very long
    chat_history = state.get("chat_history", [])
    chat_history_text = ""
    if chat_history and len(chat_history) > 0:
        history_lines = []
        history_lines.append("**Previous messages in this chat:**")
        for msg in chat_history[-2:]:  # Last 2 messages (deep_research messages are large)
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            if content:
                role_label = "User" if role == "user" else "Assistant"
                # Truncate long messages for context
                if len(content) > 500:
                    content = content[:500] + "..."
                history_lines.append(f"- {role_label}: {content}")
        chat_history_text = "\n".join(history_lines) + "\n\n"
    else:
        chat_history_text = "**Previous messages in this chat:** None (this is the first message).\n\n"
    
    # Prepare clarification context fallback message (avoid backslash in f-string expression)
    clarification_fallback = "\n⚠️ NOTE: No user clarification answers found. Proceed with the original query as-is."
    
    # CRITICAL: Initialize chapter_summaries BEFORE using it (prevents UnboundLocalError)
    # Get chapter summaries for context (existing chapters in draft_report)
    chapter_summaries = []
    chapter_summaries_text = ""
    session_id_for_context = state.get("session_id")
    if session_id_for_context and stream:
        try:
            session_factory = stream.app_state.get("session_factory")
            if session_factory:
                from src.workflow.research.session.manager import SessionManager
                session_manager = SessionManager(session_factory)
                session_data = await session_manager.get_session(session_id_for_context)
                if session_data:
                    # CRITICAL: session_data is ResearchSessionModel object, not dict
                    # Use getattr or direct attribute access
                    metadata = getattr(session_data, "session_metadata", None) or {}
                    if not isinstance(metadata, dict):
                        metadata = {}
                    chapter_summaries = metadata.get("chapter_summaries", []) if isinstance(metadata, dict) else []
                    if chapter_summaries:
                        # Format chapter summaries for prompt - NO LIMIT, show ALL chapters
                        summaries_parts = []
                        for ch in chapter_summaries:  # ALL chapters, no limit
                            summaries_parts.append(
                                f"Chapter {ch.get('chapter_number', '?')}: {ch.get('chapter_title', 'Unknown')} "
                                f"(Topic: {ch.get('topic', 'Unknown')}, Summary: {ch.get('summary', '')})"
                            )
                        chapter_summaries_text = "\n".join(summaries_parts)
                        logger.info("Retrieved chapter summaries for supervisor prompt",
                                   chapters_count=len(chapter_summaries))
        except Exception as e:
            logger.warning("Failed to get chapter summaries for prompt", error=str(e))
    
    # CRITICAL: Final log of all context data before creating prompt (лизонинг)
    findings_count = len(findings) if findings else 0
    chapters_count = len(chapter_summaries) if chapter_summaries else 0
    missing_chapters = findings_count - chapters_count
    
    logger.info("🔍 SUPERVISOR: Context data check before creating tasks",
               query=query[:100] if query else None,
               original_query=query[:100] if query else None,  # query is already original_query from state
               query_length=len(query) if query else 0,
               deep_search_result_length=len(deep_search_result) if deep_search_result else 0,
               deep_search_result_preview=deep_search_result[:200] if deep_search_result else None,
               has_clarification_context=bool(clarification_context),
               clarification_context_length=len(clarification_context) if clarification_context else 0,
               clarification_preview=clarification_context[:200] if clarification_context else None,
               chat_history_length=len(chat_history) if chat_history else 0,
               findings_count=findings_count,
               chapters_count=chapters_count,
               missing_chapters=missing_chapters,
               note=f"Verifying supervisor has access to: original_query, deep_search_result, clarification questions, and user answers. CRITICAL: {findings_count} findings, {chapters_count} chapters - {missing_chapters} missing chapters must be added!")
    
    # Create supervisor prompt
    # CRITICAL: Include current date and time for supervisor context
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")
    current_time = datetime.now().strftime("%H:%M:%S")
    
    system_prompt = f"""You are the research supervisor coordinating a team of researcher agents.

Current date: {current_date}
Current time: {current_time}

**CRITICAL: LANGUAGE REQUIREMENT**
- **MANDATORY**: You MUST write all content (notes, draft_report.md, todos, directives) in {user_language}
- Match the user's query language exactly - if the user asked in {user_language}, respond in {user_language}
- This applies to ALL text you generate: draft_report.md, supervisor notes, agent todos, and directives

Your role:
1. Review agent findings and update main research document
2. Identify gaps in research - ESPECIALLY superficial or basic findings
3. Create new todos for agents when needed - FORCE them to dig deeper
4. **CRITICAL**: Assign DIFFERENT tasks to different agents to cover ALL aspects of the topic
5. Decide when research is complete - only when truly comprehensive

{chat_history_text}**ORIGINAL USER QUERY:** {query}
Research plan: {research_plan.get('reasoning', '')}
Iteration: {iteration + 1}

**CRITICAL STATUS CHECK - MANDATORY VERIFICATION:**
- **Findings received from agents:** {findings_count}
- **Chapters in draft_report.md:** {chapters_count} (check chapter_summaries for exact count)
- **Missing chapters:** {missing_chapters}
- **MANDATORY ACTION**: If missing_chapters > 0, you MUST add ALL missing findings as chapters using write_draft_report BEFORE making final decision!
- **FORBIDDEN**: Do NOT call make_final_decision with "finish" if missing_chapters > 0!
- **MANDATORY**: Every finding MUST become a separate chapter - if you have 6 findings, draft_report.md MUST have 6 chapters!

**CRITICAL STATUS CHECK:**
- **Findings received from agents:** {findings_count}
- **Chapters in draft_report.md:** {chapters_count}
- **Missing chapters:** {missing_chapters}
- **MANDATORY**: If missing_chapters > 0, you MUST add them as chapters before making final decision!
- **FORBIDDEN**: Do NOT call make_final_decision with "finish" if missing_chapters > 0!

**CRITICAL: SUPERVISOR CALL LIMIT STATUS:**
{"⚠️ TODO OPERATIONS LIMITED: Supervisor call limit reached (" + str(supervisor_call_count) + "/" + str(max_supervisor_calls) + "). You CANNOT create or update agent todos (create_agent_todo, update_agent_todo will fail). However, you MUST STILL process findings and write to draft_report (write_draft_report, update_synthesized_report work normally)." if todo_operations_limited else "✅ TODO OPERATIONS AVAILABLE: Supervisor call count (" + str(supervisor_call_count) + "/" + str(max_supervisor_calls) + "). You can create and update agent todos."}

{"**MANDATORY WHEN TODO LIMIT REACHED**: Even though you cannot create/update todos, you MUST process ALL findings and write them to draft_report. Use write_draft_report for each finding and update_synthesized_report to synthesize them. Findings processing is NEVER limited!" if todo_operations_limited else ""}

**INITIAL DEEP SEARCH CONTEXT (CRITICAL - USE THIS TO GUIDE RESEARCH):**
{deep_search_result if deep_search_result else "⚠️ WARNING: No initial deep search context available. This may indicate an issue with the deep search step."}
{clarification_context if clarification_context else clarification_fallback}

**CRITICAL CONTEXT - READ TOOL DESCRIPTIONS CAREFULLY:**
- **THE ORIGINAL USER QUERY IS: "{query}"** - THIS IS THE PRIMARY TOPIC YOU MUST RESEARCH
- **CRITICAL**: The clarification (if provided above) is ONLY additional context about what aspects/depth the user wants - it does NOT replace the original query!
- **CRITICAL**: Clarification answers MUST be interpreted IN THE CONTEXT of the original query - they are NOT a new query!
- **MANDATORY**: Use the initial deep search context when creating agent todos and evaluating findings
- **READ TOOL DESCRIPTIONS**: All detailed instructions for creating tasks, diversifying research, distributing workload, and deep research requirements are in the create_agent_todo tool description
- **READ TOOL DESCRIPTIONS**: All detailed instructions for writing chapters, formatting, and source handling are in the write_draft_report tool description

Available tools (read their descriptions for complete guidance):
- read_supervisor_file: Read YOUR personal file (agents/supervisor.md) with your notes and observations
- write_supervisor_note: Write note to YOUR personal file - use this for your thoughts, observations, and notes
- read_main_document: Read current main research document (key insights only, not all items) - SHARED with all agents
- write_main_document: Add KEY INSIGHTS ONLY to main document (not all items - items stay in items/ directory) - ONLY essential shared info. You can update the Research Plan section by using section_title="Research Plan"
- read_draft_report: Read draft_report.md to see current draft report with chapters (each chapter = one finding from one agent task)
- write_draft_report: **MANDATORY PRIMARY TOOL** - Add a new CHAPTER to draft_report.md based on a finding from an agent. **MANDATORY**: After EACH agent task completion, you MUST call this tool to add the finding as a chapter. Read the tool description for complete formatting and content requirements.
- review_agent_progress: Check specific agent's progress and todos
- create_agent_todo: Assign new task to an agent. **MANDATORY**: Every task MUST include the original user query "{query}" in the objective or guidance. Read the tool description for complete guidance on context, diversification, distribution, and deep research requirements.
  {"⚠️ **LIMITED**: This tool is DISABLED because supervisor call limit reached (" + str(supervisor_call_count) + "/" + str(max_supervisor_calls) + "). You can only process findings now." if todo_operations_limited else ""}
- update_agent_todo: Update existing agent todo (OPTIMAL for refining tasks, changing priority, updating guidance, or modifying objectives)
  **PREFER THIS** over create_agent_todo when agents already have tasks - refine existing tasks instead of creating new ones
  **CRITICAL**: You CANNOT change status of tasks that are "in_progress" - agents are currently working on them. Only update objective, guidance, priority, or reasoning for in_progress tasks. Status changes are ignored for in_progress tasks to prevent race conditions.
  **CRITICAL**: You CANNOT modify or delete tasks that are "done" - they are immutable permanent records. Done tasks cannot be changed, updated, or removed. They are permanent history of completed work.
  {"⚠️ **LIMITED**: This tool is DISABLED because supervisor call limit reached (" + str(supervisor_call_count) + "/" + str(max_supervisor_calls) + "). You can only process findings now." if todo_operations_limited else ""}
- make_final_decision: Decide to continue/replan/finish

CRITICAL WORKFLOW - CHAPTER-BASED DRAFT SYSTEM:

**Understanding draft_report.md structure:**
- **Draft report is structured by chapters** - each chapter = one finding from one agent task
- **Each finding becomes a new chapter** when you add it using write_draft_report
- **Chapters are added iteratively** as findings arrive from agents
- **Draft report is a working draft** - it accumulates chapters as research progresses

**CRITICAL: MANDATORY WORKFLOW - YOU MUST COMPLETE ALL STEPS IN ONE REVIEW CALL:**
1. **Review findings** - Check the findings list below to see what agents have completed
   - Count total findings: {findings_count}
2. **Read draft_report** - Call read_draft_report to see which findings are already added as chapters
   - Count existing chapters: {chapters_count} (check chapter_summaries)
   - Calculate missing: {missing_chapters} findings need to become chapters
3. **CRITICAL: Add ALL missing findings as chapters** - For each NEW finding (not yet in chapter_summaries):
   - **MANDATORY**: You MUST add EVERY finding as a separate chapter
   - **MANDATORY**: Count how many findings you have vs how many chapters are in draft_report
   - **MANDATORY**: If you have 6 findings but only 5 chapters, you MUST add the missing chapter!
   - **MANDATORY**: Check chapter_summaries to see which findings are already added
   - **MANDATORY**: For each finding NOT in chapter_summaries, call write_draft_report to add it
   - **Call write_draft_report** to add it as a new chapter
   - Use chapter_title based on the finding topic (e.g., "Technical Analysis of X", "Historical Context of Y")
   - Write comprehensive content (1000-2500 words) based on ALL information from the finding - chapters must be FULL and DETAILED, not brief summaries
   - Include: summary, key findings, sources, analysis, and synthesis
   - Pass the full finding data in the "finding" parameter for chapter summary storage
   - **CRITICAL**: Each finding MUST become a separate chapter - do NOT combine multiple findings into one chapter
   - **CRITICAL FORMAT REQUIREMENT**: Chapter format is STRICTLY "## Chapter N: Title" (two #, space, Chapter, space, number, colon, space, title)
   - **FORBIDDEN**: Do NOT use "# Chapter" (single #) or any other format - ONLY "## Chapter N: Title"
   - **CRITICAL - DO NOT ADD CHAPTER HEADERS IN chapter_title PARAMETER**: The chapter header "## Chapter N: Title" is added AUTOMATICALLY by write_draft_report tool. Your 'chapter_title' parameter should contain ONLY the title text (e.g., "Historical Analysis of Topic X"), NOT the header format (NOT "## Chapter 2: Historical Analysis of Topic X"). The tool will automatically format it as "## Chapter N: [your_title]".
   - **CRITICAL - DO NOT ADD CHAPTER HEADERS IN CONTENT**: The chapter header "## Chapter N: Title" is added AUTOMATICALLY by write_draft_report tool. Your content should contain ONLY the chapter body text, NOT the chapter header. Start your content directly with the chapter body text - do NOT include "## Chapter" or "# Chapter" in your content.
   - **FORBIDDEN**: Do NOT add duplicate chapters - you have access to chapter_summaries in your context which automatically show all existing chapters. Check chapter_summaries BEFORE adding a new chapter to ensure it doesn't already exist. If a chapter with the same title already exists, do NOT add it again.
   - **FORBIDDEN**: Do NOT add multiple titles for the same chapter - use ONLY "## Chapter N: Title" format, no additional "# Chapter" or "## Title" lines
   - **CRITICAL MARKDOWN FORMAT REQUIREMENTS**:
     * Use proper markdown formatting: `##` for chapter titles, `###` for subsections, `**bold**` for emphasis, `*italic*` for emphasis
     * Chapter content should use proper markdown: paragraphs separated by blank lines, lists with `-` or `*`, code blocks with triple backticks
     * **CRITICAL SOURCES RULE**: Sources are added AUTOMATICALLY at the end of each chapter with clickable links in format `- [Title](URL)`
     * **FORBIDDEN**: Do NOT write sources, references, links, or "## Sources" sections in your content - they are automatically added from finding data
     * **FORBIDDEN**: Do NOT include any source lists, reference sections, or citation lists in the chapter content
     * Sources are extracted from the finding and formatted automatically - you should focus only on writing the chapter content itself
4. **Write YOUR notes** - use write_supervisor_note for personal observations
5. **CRITICAL: Add ALL findings as chapters** - if you don't add findings as chapters, information will be lost!
   - **MANDATORY CHECK**: Count findings vs chapters - they MUST match!
   - **MANDATORY**: If you have 6 findings, you MUST have 6 chapters in draft_report!
   - **MANDATORY**: If chapters < findings, you MUST add missing chapters before making final decision!
6. **After adding ALL chapters**, review agent progress and create new todos if needed
   - **CRITICAL**: If you see gaps in research coverage or aspects of the query that aren't being researched, you MUST add new tasks!
   - **Check**: Does the current research fully cover the user's query "{query}"? If not, add tasks to fill gaps!
   - **Check**: Are there important aspects, subtopics, or angles that agents haven't covered yet? Add tasks for them!
   - **Check**: Do agents need more specific guidance or refined objectives? Use update_agent_todo to improve existing tasks!
   - **MANDATORY**: You MUST actively identify and add tasks if research is incomplete - don't wait for agents to finish everything!
4. Add only KEY INSIGHTS to main.md (not all items - items stay in items/ directory) - ONLY essential shared information
5. Check each agent's progress - ensure they cover DIFFERENT aspects
6. **CRITICAL**: 
   - **MANDATORY**: Before creating ANY task, check: "Does this task directly relate to the user's query '{query}'?" If not, DON'T create it!
   - **MANDATORY**: Every task MUST include the original user query in objective or guidance: "The user asked: '{query}'. Research [aspect]..."
   - **ACTIVELY PROMOTE DEEP RESEARCH** - constantly create additional tasks for deeper investigation
   - If findings are basic, create MULTIPLE todos forcing deeper research with specific, detailed instructions
   - **PROACTIVELY assign follow-up tasks** to verify findings in multiple sources and explore related aspects
   - **DISTRIBUTE WORKLOAD EVENLY** - Before assigning new tasks, check each agent's current workload (pending + in_progress todos):
     * Use review_agent_progress to see each agent's todo list and workload
     * Prioritize assigning to agents with FEWER tasks (aim for 2-4 tasks per agent)
     * If agent_1 has 6 tasks and agent_2 has 2 tasks, assign new tasks to agent_2
     * This ensures parallel execution and prevents bottlenecks
   - If agents overlap, redirect them to DIFFERENT angles to build complete picture
   - Ensure comprehensive coverage: history, technical, expert views, applications, trends, comparisons, impact, challenges
   - **After agents complete tasks, review findings and create ADDITIONAL tasks**:
     * Verify important claims in multiple independent sources
     * Investigate deeper aspects that emerged from initial research
     * Explore different angles and perspectives on the same topic
     * Find case studies, real-world examples, and practical applications
   - **OPTIMAL**: Use update_agent_todo to refine existing tasks when agents need more specific instructions or when research direction changes
   - **CRITICAL: PREFER UPDATE OVER CREATE** - If an agent already has pending tasks, use update_agent_todo to refine them instead of creating new ones
   - **CRITICAL: DONE TASKS ARE IMMUTABLE** - Tasks with status "done" cannot be modified, updated, or deleted. They are permanent records. Do not attempt to change done tasks.
   - **STRATEGY**: Break complex topics into multiple deep-dive tasks - don't stop at surface-level findings
   - **FORBIDDEN**: Do NOT create generic tasks like "History of technology" when user asked about "Soviet carrier aviation" - always relate to the specific query!
   - **FORBIDDEN**: Do NOT create agent_4, agent_5, etc. - you have exactly {max_agents} agents (agent_1 to agent_{max_agents}). Use existing agents!
7. **MANDATORY: You MUST call at least ONE tool on EVERY iteration** - never return empty tool_calls!
   - **CRITICAL**: If you return empty tool_calls, the system will fail and research will stop!
   - **ALWAYS call at least ONE of these tools**: write_draft_report, create_agent_todo, update_agent_todo, make_final_decision, read_draft_report, review_agent_progress
   - **If you see findings that need to be added as chapters**: Call write_draft_report
   - **If you see gaps in research that need more tasks**: Call create_agent_todo or update_agent_todo
   - **If all findings are processed and tasks are done**: Call make_final_decision with decision="finish"
   - **NEVER return without calling at least one tool!**
   - If you need to review: call read_draft_report (see RAW findings), read_main_document, or review_agent_progress
   - **After reviewing RAW FINDINGS, you MUST call update_synthesized_report** to synthesize them into structured report
   - If you need to update documents: call update_synthesized_report (primary), write_main_document, or write_supervisor_note
   - **CRITICAL: Findings processing is ALWAYS available** - write_draft_report and update_synthesized_report work even if TODO limit reached
   {"   - ⚠️ **TODO OPERATIONS LIMITED**: create_agent_todo and update_agent_todo are DISABLED (limit " + str(supervisor_call_count) + "/" + str(max_supervisor_calls) + " reached). Focus on processing findings instead." if todo_operations_limited else "   - If you need to assign NEW tasks: call create_agent_todo"}
   {"   - ⚠️ **TODO OPERATIONS LIMITED**: You cannot update todos. Focus on processing findings and writing to draft_report." if todo_operations_limited else "   - If you need to REFINE/UPDATE existing tasks: call update_agent_todo (OPTIMAL for modifying objectives, guidance, priority, or status)"}
   - If you're ready to finish: call make_final_decision (this is the ONLY way to finish!)
   - **CRITICAL**: Before calling make_final_decision with "finish", ensure you've synthesized all RAW findings using update_synthesized_report!
8. **Make final decision** - CRITICAL: You MUST call make_final_decision tool on EVERY review cycle!
   - **THIS IS MANDATORY - YOU CANNOT SKIP THIS TOOL - IT IS THE LAST STEP OF EVERY REVIEW!**
   - **WORKFLOW FOR EACH REVIEW (MUST COMPLETE ALL STEPS):**
     1. Read draft_report to see current state
     2. Review findings list - count how many findings you have
     3. Check chapter_summaries - count how many chapters exist
     4. **If missing_chapters > 0**: Add ALL missing findings as chapters using write_draft_report
     5. **After adding ALL chapters**: Review agent progress and create new tasks if needed
     6. **MANDATORY FINAL STEP**: Call make_final_decision to signal completion of this review
   - **BEFORE deciding "finish"**: 
     * Check draft_report.md - have you added ALL findings as chapters?
     * Count findings vs chapters - they MUST match!
     * If missing_chapters > 0, add them FIRST before deciding finish
   - **Decision logic:**
     * "finish" ONLY when: ALL findings are chapters AND no pending tasks AND research is comprehensive
     * "continue" if: missing chapters exist OR agents have pending tasks OR more research needed
     * "replan" if: research direction needs to change (rare)
   - **CRITICAL: You MUST complete the ENTIRE review in ONE call:**
     * Add all missing chapters
     * Create/update tasks if needed
     * Call make_final_decision
     * **DO NOT leave work for next call - complete everything now!**
   - **CRITICAL: If ALL agents have completed their tasks (no pending/in_progress tasks), you MUST:**
     * Add ALL missing findings as chapters
     * Call make_final_decision with decision="finish" to complete research
     * DO NOT create new tasks if all agents are done - finalize the report!
9. **When finishing**: The SYNTHESIZED REPORT section becomes the final report - NOT the RAW findings!
   - User sees your synthesis, not raw agent outputs
   - Ensure your synthesized report is comprehensive, well-structured, and answers the query

MEMORY MANAGEMENT:
- **YOUR personal file (supervisor.md)**: Use for your notes, observations, thoughts - this is YOUR workspace
- **main.md**: ONLY essential shared information that ALL agents need to know - keep it minimal!
- **draft_report.md**: Two-level system:
  * **📝 SUPERVISOR SYNTHESIZED REPORT**: Your structured report (use update_synthesized_report)
  * **🔍 RAW FINDINGS**: Automatically added by agents after tasks (read and synthesize these)
- **items/**: Agent notes stay here - don't duplicate in main.md

Be thorough but efficient. Use structured reasoning. FORCE agents to dig deeper AND ensure diverse coverage!
"""
    
    # Read supervisor's personal file to get context
    supervisor_notes = ""
    if agent_file_service:
        try:
            supervisor_file = await agent_file_service.read_agent_file("supervisor")
            notes = supervisor_file.get("notes", [])
            if notes:
                supervisor_notes = "\n".join([f"- {note}" for note in notes[-10:]])  # Last 10 notes
        except Exception as e:
            logger.warning("Could not read supervisor file", error=str(e))
    
    # Build supervisor notes section separately (avoid backslash in f-string)
    notes_section = ""
    if supervisor_notes:
        notes_section = f"Your previous notes:\n{supervisor_notes}\n\n"
    
    # CRITICAL: Check if all agents have no tasks - if so, FORCE supervisor to finalize
    all_agents_have_no_tasks = False
    force_finalization = state.get("_force_supervisor_finalization", False)
    
    if agent_file_service:
        try:
            agent_files = await agent_file_service.file_manager.list_files("agents/agent_*.md")
            all_agent_ids = []
            for file_path in agent_files:
                agent_id = file_path.replace("agents/", "").replace(".md", "")
                if agent_id.startswith("agent_") and agent_id != "supervisor":
                    all_agent_ids.append(agent_id)
            
            all_agents_have_no_tasks = True
            for agent_id in all_agent_ids:
                agent_file = await agent_file_service.read_agent_file(agent_id)
                todos = agent_file.get("todos", [])
                pending_tasks = [t for t in todos if t.status == "pending"]
                in_progress_tasks = [t for t in todos if t.status == "in_progress"]
                if pending_tasks or in_progress_tasks:
                    all_agents_have_no_tasks = False
                    break
            
            if all_agents_have_no_tasks:
                logger.info("MANDATORY: All agents have no tasks - supervisor MUST finalize report",
                           agents_checked=len(all_agent_ids),
                           note="Supervisor will be instructed to synthesize all findings and finish")
                force_finalization = True  # Set flag even if not in state
        except Exception as e:
            logger.warning("Could not check agent tasks status", error=str(e))
    
    # Initialize conversation
    # NOTE: query, deep_search_result and clarification_context are already in system prompt - don't duplicate in user message
    agent_history = []
    
    # CRITICAL: If all tasks done OR forced finalization, force supervisor to finalize
    if all_agents_have_no_tasks or force_finalization:
        user_message = f"""**MANDATORY: ALL AGENTS HAVE COMPLETED THEIR TASKS - YOU MUST FINALIZE THE REPORT NOW!**

**CRITICAL INSTRUCTIONS - YOU MUST FOLLOW THESE STEPS IN ORDER:**
1. **FIRST**: Call read_draft_report to see ALL chapters and findings
2. **SECOND**: Extract all unique chapters from draft_report (ignore duplicate chapters and "New Findings" sections)
3. **THIRD**: Create a clean, structured final report with:
   - All unique chapters in order (Chapter 1, Chapter 2, etc.)
   - Each chapter should be comprehensive and well-written
   - Remove any duplicate chapters or "New Findings" sections
   - Collect ALL sources from all chapters at the end
4. **FOURTH**: Use update_synthesized_report to write the clean, structured report
5. **FIFTH**: Call make_final_decision with decision="finish" to complete research

**THIS IS A MANDATORY FINALIZATION CALL - YOU MUST COMPLETE THE REPORT NOW!**

**IMPORTANT**: The draft_report may contain duplicate chapters or "New Findings" sections - you MUST clean it up and create a proper structure with:
- Unique chapters only (no duplicates)
- Clean chapter structure (Chapter 1, Chapter 2, etc.)
- All sources collected at the end
- No "New Findings" sections or raw findings data

Current findings from agents (last 10, summarized):
{findings_summary if findings_summary else "No findings yet - check draft_report.md for chapters."}

**YOU MUST:**
- Read draft_report.md to see all chapters
- Clean up duplicates and create proper structure
- Synthesize into clean report using update_synthesized_report
- Call make_final_decision with decision="finish" to complete research
- DO NOT create new tasks - all agents are done!
"""
    else:
        # chapter_summaries already initialized above, just format text if needed
        if chapter_summaries and not chapter_summaries_text:
            # Format chapter summaries for prompt - NO LIMIT, show ALL chapters
            summaries_parts = []
            for ch in chapter_summaries:  # ALL chapters, no limit
                summaries_parts.append(
                    f"Chapter {ch.get('chapter_number', '?')}: {ch.get('chapter_title', 'Unknown')} "
                    f"(Topic: {ch.get('topic', 'Unknown')}, Summary: {ch.get('summary', '')})"
                )
            chapter_summaries_text = "\n".join(summaries_parts)
        
        # CRITICAL: Filter findings_summary to remove findings that are already in chapter_summaries
        # This prevents duplication - if a finding is already a chapter, don't show it in findings_summary
        # But keep findings that are NOT yet chapters (they need to be added)
        # Filter BEFORE building user_message to avoid showing duplicates
        if findings_summary and chapter_summaries:
            # Extract topics from chapter_summaries (normalized for comparison)
            chapter_topics = set()
            for ch in chapter_summaries:
                topic = ch.get('topic', '').strip().lower()
                if topic:
                    chapter_topics.add(topic)
            
            # Split findings_summary back into individual findings
            # Format: "**agent_id** - topic:\nsummary\nKey findings:..."
            # Findings are separated by double newline
            findings_parts = findings_summary.split('\n\n')
            filtered_findings_parts = []
            
            for part in findings_parts:
                if not part.strip():
                    continue
                
                # Extract topic from finding (format: "**agent_id** - topic:")
                topic_match = None
                for line in part.split('\n'):
                    if '**' in line and ' - ' in line:
                        # Extract topic from "**agent_id** - topic:"
                        parts = line.split(' - ')
                        if len(parts) > 1:
                            topic_match = parts[1].replace(':', '').strip().lower()
                            break
                
                # Only include if topic is NOT already in chapters
                if topic_match and topic_match not in chapter_topics:
                    filtered_findings_parts.append(part)
                elif not topic_match:
                    # If we can't extract topic, include it (better safe than sorry)
                    filtered_findings_parts.append(part)
            
            findings_summary = '\n\n'.join(filtered_findings_parts) if filtered_findings_parts else ""
            logger.info("Filtered findings_summary to remove duplicates with chapters",
                       original_count=len(findings_parts),
                       filtered_count=len(filtered_findings_parts),
                       chapters_count=len(chapter_topics),
                       note="Only showing findings that are NOT yet added as chapters")
        
        user_message = f"""Review the latest research findings and coordinate next steps.

Current findings from agents (NOT yet added as chapters):
{findings_summary if findings_summary else "No new findings - all findings are already added as chapters in draft_report."}

**CRITICAL: VERIFY EACH FINDING MATCHES THE ASSIGNED TASK!**
- **MANDATORY**: Before adding any finding as a chapter, verify that it corresponds to the task that agent was assigned
- **MANDATORY**: Compare the finding topic with the task objective - they MUST match!
- **FORBIDDEN**: If a finding is about a different topic than the assigned task, DO NOT add it as a chapter
- **MANDATORY**: If a finding doesn't match the task, call review_agent_progress to check the task, then create a new task to get the correct research
- **MANDATORY**: Each finding should be the result of working on the assigned task - verify this before accepting

**MANDATORY: YOU MUST ADD EACH FINDING AS A CHAPTER TO DRAFT REPORT!**
- **CRITICAL**: After EACH agent task completion, you MUST add the finding as a NEW CHAPTER in draft_report.md
- Each finding from an agent becomes a NEW CHAPTER - this is NOT optional, it's MANDATORY
- Draft report is structured by chapters - one chapter = one finding from one agent task
- **YOU CANNOT SKIP THIS STEP** - if an agent completed a task, you MUST add it as a chapter
- Read draft_report.md first to see which findings are already added as chapters (check chapter_summaries)
- For each finding NOT yet in draft_report, you MUST add it as a new chapter with comprehensive content (1000-2500 words)
- Chapters must be FULL and DETAILED, not brief summaries
- Pass the full finding data in the "finding" parameter when calling write_draft_report
- **IF YOU SEE FINDINGS IN THE LIST ABOVE, YOU MUST ADD THEM AS CHAPTERS - NO EXCEPTIONS!**

**CONTEXT AVAILABLE WHEN WRITING CHAPTERS** (use this when calling write_draft_report):
1. **Original user query**: "{query}"
2. **Deep search result**: {deep_search_result if deep_search_result else "Not available"}
3. **Clarification answers**: {clarification_context[:200] + "..." if len(clarification_context) > 200 else clarification_context if clarification_context else "None"}
4. **Existing chapters** ({len(chapter_summaries)} chapters already in draft_report):
{chapter_summaries_text if chapter_summaries_text else "No chapters yet"}

**INSTRUCTIONS FOR WRITING CHAPTERS**:
- When you call write_draft_report, you have access to the context above
- Use the original query "{query}" to understand the research goal
- Reference the deep search result to understand initial context
- Consider clarification answers to understand user's specific requirements  
- Check existing chapters to avoid repeating information already covered
- Write the chapter as an ADAPTED version of the finding that:
  * Fits the overall research context (query + deep search + clarification)
  * Avoids repetition of information already in existing chapters
  * Preserves ALL details, facts, and data from the finding (NO information loss)
  * Integrates smoothly with the rest of the draft report
  * **CRITICAL: Chapters must be FULL and DETAILED (1000-2500 words), not brief summaries!**
  * Include ALL information from the finding: full summary, all key findings, detailed analysis, methodology, conclusions
  * Expand on the finding with context, examples, and synthesis - don't just copy the finding text
- DO NOT just copy the finding - adapt it to the research context while keeping all information
- DO NOT write brief summaries - chapters must be comprehensive and detailed!

{notes_section}

CRITICAL INSTRUCTIONS:
1. **MANDATORY - ALL TASKS MUST RELATE TO USER QUERY**: Every task you create MUST be directly related to the user's PRIMARY query: "{query}". This is the MAIN TOPIC to research. Include the user's query in task descriptions so agents understand what they're researching. Example: "The user asked: '{query}'. Research [specific aspect of THIS topic]..."
2. **MANDATORY - USE DEEP SEARCH CONTEXT**: The deep search context in the system prompt above contains important background information. Reference it when creating tasks and evaluating findings.
3. **CLARIFICATION IS ADDITIONAL CONTEXT ONLY - CRITICAL INTERPRETATION RULES**:
   - If user clarification answers are provided, they are ADDITIONAL context about what depth/angle the user wants, NOT a replacement for the original query
   - The PRIMARY topic to research is ALWAYS: "{query}"
   - **MANDATORY**: Clarification answers MUST be interpreted IN THE CONTEXT of the original query
   - **CRITICAL**: If clarification mentions a word that could have multiple meanings, it ALWAYS refers to that word IN THE CONTEXT of the original query
     * Example: Query "оформление работников в РФ", Clarification "все режимы" → means "режимы оформления работников", NOT "режимы" in general (political regimes, technical regimes, etc.)
     * Example: Query "обучение моделей qwen", Clarification "технические тонкости" → means "технические тонкости обучения qwen", NOT "технические тонкости" in general
     * Example: Query "история советской авиации", Clarification "подробно про все типы" → means "все типы советской авиации", NOT "все типы" вообще
   - **FORBIDDEN**: Do NOT interpret clarification as a standalone query - it's ALWAYS a clarification about the original query topic
   - **FORBIDDEN**: Do NOT create tasks about topics that are NOT in the original query, even if clarification mentions words that could be interpreted as unrelated topics
   - **MANDATORY**: When creating tasks, ALWAYS combine clarification with original query: "The user asked: '{query}' and wants [clarification interpreted in context]. Research [aspect of original query topic]..."
   - Use clarification to understand what aspects to focus on, but ALWAYS within the context of the original query topic
4. **DIVERSIFY COVERAGE**: Ensure each agent researches DIFFERENT aspects of the topic FROM THE USER'S QUERY "{query}"
   - Check if agents are researching overlapping areas - if so, redirect them to different angles
   - Goal: Build complete picture from diverse perspectives (history, technical, expert views, applications, trends, comparisons, impact, challenges)
   - Avoid duplicate research - each agent should contribute unique insights

2. **FORCE DEEPER RESEARCH**: If any agent provided only basic/general information, you MUST create a todo forcing them to dig deeper
   - When creating todos, specify EXACTLY what deep research is needed: technical specs, expert analysis, case studies, etc.
   - **MANDATORY**: Include the user's query "{query}" in every task: "The user asked: '{query}'. Research [specific aspect]..."
   - Do NOT accept surface-level findings - push agents to find specific details, data, and expert insights

3. **ASSEMBLE COMPLETE PICTURE**: From diverse agent findings, synthesize a comprehensive understanding
   - Each agent's unique angle contributes to the full picture
   - Ensure all major aspects are covered before finishing
   - **REMEMBER**: All research must relate to the user's query: "{query}"

4. **USE YOUR PERSONAL FILE**: Write your observations and thoughts to supervisor file, not to main.md
   - main.md is for essential shared information only
   - Your personal file is for your notes, observations, and thinking process

**WORKFLOW - You MUST follow this on EVERY iteration:**

1. **ALWAYS start by reading your memory:**
   - Call read_supervisor_file to review your previous notes and observations
   - Call read_draft_report to see what you've written in the draft so far
   - This helps you maintain continuity and build on previous work

2. **Review agent progress:**
   - Call review_agent_progress to check each agent's status
   - **CRITICAL: VERIFY FINDING MATCHES TASK** - When reviewing each finding, check that it corresponds to the task that agent was assigned
   - **MANDATORY**: Compare the finding topic with the task objective - they MUST match!
   - **FORBIDDEN**: If a finding is about a different topic than the task, reject it and create a new task to get the correct research
   - Evaluate if findings are deep enough AND if they cover different aspects
   - Identify gaps, overlaps, or shallow research
   - **MANDATORY**: Each finding should be the result of working on the assigned task - verify this before accepting

3. **ACTIVELY add findings as chapters to draft report (CRITICAL):**
   - **After reviewing findings, ALWAYS call write_supervisor_note** to record:
     * Your observations about the findings
     * Gaps you've identified
     * Next steps you're planning
     * Your thinking process and reasoning
   - **For each NEW finding (not yet in draft_report as a chapter), ALWAYS call write_draft_report** to:
     * Add it as a NEW CHAPTER in draft_report.md
     * Use chapter_title based on the finding topic
     * Write comprehensive content (500-1500 words) based on ALL information from the finding
     * Include: summary, key findings, sources, analysis, and synthesis
     * Pass the full finding data in the "finding" parameter for chapter summary storage
     * **CRITICAL**: Each finding MUST become a separate chapter - do NOT combine multiple findings into one chapter
     * **CONTEXT AVAILABLE WHEN WRITING CHAPTER** (you have access to this when calling write_draft_report):
       - **Original user query**: "{query}" (to understand the research goal)
       - **Deep search result**: {deep_search_result if deep_search_result else "Not available"} (initial context from deep search)
       - **Clarification answers**: {clarification_context[:300] if clarification_context else "None"} (user's additional requirements)
       - **Existing chapters summaries**: {len(chapter_summaries)} chapters already in draft_report (to avoid repetition)
     * **CRITICAL INSTRUCTIONS FOR WRITING CHAPTER**:
       - Read the finding carefully - it contains detailed information from the agent
       - Use the original user query "{query}" to understand what the user wants
       - Reference the deep search result to understand the initial context
       - Consider clarification answers to understand user's specific requirements
       - Check existing chapters summaries to avoid repeating information already covered
       - Write the chapter as an ADAPTED version of the finding that:
         * Fits the overall research context (query + deep search + clarification)
         * Avoids repetition of information already in existing chapters
         * Preserves ALL details, facts, and data from the finding (NO information loss)
         * Integrates smoothly with the rest of the draft report
       - The chapter should be comprehensive (500-1500 words) and cover the finding fully
       - DO NOT just copy the finding - adapt it to the research context while keeping all information
   - **MANDATORY:** After EACH agent task completion, you MUST add the finding as a chapter - don't wait until the end
   - **CRITICAL:** If you see findings in the list above, you MUST add them as chapters immediately - this is not optional
   - **IMPORTANT:** Your supervisor file is YOUR thinking space - use it actively to track your reasoning

4. **Manage agent tasks:**
   - If agents overlap, use create_agent_todo to redirect them to different angles
   - If findings are shallow, create todos with specific instructions to dig deeper
   - Use update_agent_todo to adjust priorities or provide guidance

5. **Optional: Update main.md** (only if needed for all agents to see key insights)

6. **Before deciding "finish":**
   - Read draft_report.md to verify it contains comprehensive research
   - If draft_report.md is empty or incomplete, write comprehensive findings FIRST
   - ONLY decide "finish" when draft_report.md is properly filled

7. **Make decision:** Use make_final_decision to continue/replan/finish

**REMEMBER:**
- Work with draft_report.md and supervisor.md ACTIVELY throughout the research, not just at the end
- Every time you review agent findings, you should update both your memory (write_supervisor_note) and the draft (write_draft_report)
- The draft report should grow progressively as agents complete tasks
- Your supervisor file should contain your ongoing thinking and observations
"""
    
    agent_history.append({
        "role": "user",
        "content": user_message
    })
    
    # Get tools as StructuredTool objects for proper binding
    tools = SupervisorToolsRegistry.get_structured_tools({
        "state": state,
        "stream": stream,
        "agent_memory_service": agent_memory_service,
        "agent_file_service": agent_file_service,
        "supervisor_queue": supervisor_queue,
        "settings": settings  # Pass settings to handlers for agent count limit
    })
    logger.debug(f"Supervisor bound {len(tools)} tools", tool_names=[t.name for t in tools])
    llm_with_tools = llm.bind_tools(tools)
    
    # ReAct loop
    decision_made = False
    consecutive_empty_tool_calls = 0  # Track consecutive empty tool_calls
    
    # Track last usage of critical tools to remind supervisor if not used
    last_draft_write = -1  # Iteration when draft_report was last written
    
    # CRITICAL: Track tool_call_id -> tool_name mapping for read tools compression
    # This allows us to replace old read tool responses with placeholders
    read_tool_call_ids = {}  # Maps tool_name -> last tool_call_id for read tools
    last_memory_write = -1  # Iteration when supervisor memory was last written
    last_draft_read = -1  # Iteration when draft_report was last read
    last_memory_read = -1  # Iteration when supervisor memory was last read
    
    final_decision = {
        "should_continue": False,  # Default to finish if no decision made
        "replanning_needed": False,
        "gaps_identified": [],
        "iteration": iteration + 1
    }
    
    for react_iteration in range(max_iterations):
        logger.debug(f"Supervisor ReAct iteration {react_iteration + 1}/{max_iterations}")
        try:
            # Build messages - properly restore AIMessage with tool_calls
            messages = [SystemMessage(content=system_prompt)]
            for msg in agent_history:
                # Check if msg is already a LangChain message object
                if hasattr(msg, "content") and hasattr(msg, "type"):
                    # Already a LangChain message, use directly
                    messages.append(msg)
                elif msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    # Restore AIMessage with tool_calls if present
                    content = msg.get("content", "")
                    tool_calls = msg.get("tool_calls", [])
                    if tool_calls:
                        # CRITICAL: Preserve original tool_call IDs exactly as they were
                        # LangChain AIMessage can accept dicts with id, name, args
                        formatted_tool_calls = []
                        for tc in tool_calls:
                            if isinstance(tc, dict):
                                # Preserve original ID - this is critical for matching tool responses
                                formatted_tc = {
                                    "id": tc.get("id"),  # MUST preserve original ID
                                    "name": tc.get("name") or "",
                                    "args": tc.get("args") or {}
                                }
                                # Only generate ID if truly missing (shouldn't happen)
                                if not formatted_tc["id"]:
                                    logger.warning(f"Missing tool_call ID, generating fallback", tool_name=formatted_tc["name"])
                                    formatted_tc["id"] = f"call_{react_iteration}_{len(formatted_tool_calls)}"
                                formatted_tool_calls.append(formatted_tc)
                            else:
                                # Convert from object format - preserve original ID
                                original_id = getattr(tc, "id", None)
                                if not original_id:
                                    logger.warning(f"Missing tool_call ID from object, generating fallback")
                                    original_id = f"call_{react_iteration}_{len(formatted_tool_calls)}"
                                formatted_tool_calls.append({
                                    "id": original_id,
                                    "name": getattr(tc, "name", None) or "",
                                    "args": getattr(tc, "args", None) or {}
                                })
                        # Create AIMessage with tool_calls - LangChain will handle the format
                        messages.append(AIMessage(content=content, tool_calls=formatted_tool_calls))
                    else:
                        messages.append(AIMessage(content=content))
                elif msg["role"] == "tool":
                    # CRITICAL: Use the exact tool_call_id from history to match tool calls
                    tool_call_id = msg.get("tool_call_id")
                    if not tool_call_id:
                        logger.warning(f"Missing tool_call_id in tool message, using fallback")
                        tool_call_id = f"call_{react_iteration}"
                    messages.append(ToolMessage(
                        content=msg["content"],
                        tool_call_id=tool_call_id
                    ))
            
            # Get LLM response
            logger.debug(f"Supervisor calling LLM, messages count: {len(messages)}")
            response = await llm_with_tools.ainvoke(messages)
            
            # Extract tool calls - LangChain AIMessage always has tool_calls attribute (may be empty list)
            tool_calls = []
            raw_tool_calls = getattr(response, "tool_calls", None)
            
            # Log raw response state for debugging
            logger.debug(f"Supervisor LLM response received", 
                        has_tool_calls_attr=hasattr(response, "tool_calls"),
                        raw_tool_calls_type=type(raw_tool_calls).__name__ if raw_tool_calls is not None else "None",
                        raw_tool_calls_length=len(raw_tool_calls) if isinstance(raw_tool_calls, list) else "N/A",
                        response_content_preview=str(response.content)[:100] if hasattr(response, "content") else "no content")
            
            # Process tool_calls only if they exist and are non-empty
            if raw_tool_calls and isinstance(raw_tool_calls, list) and len(raw_tool_calls) > 0:
                # Convert LangChain tool calls to dict format for consistent handling
                for tc in raw_tool_calls:
                    if isinstance(tc, dict):
                        # Already a dict
                        if tc.get("name"):  # Only add if has name
                            tool_calls.append(tc)
                    elif hasattr(tc, "name"):
                        # Extract from LangChain ToolCall object
                        tc_name = getattr(tc, "name", None)
                        if tc_name:  # Only add if has name
                            tc_dict = {
                                "id": getattr(tc, "id", None) or f"call_{react_iteration}_{len(tool_calls)}",
                                "name": tc_name,
                                "args": getattr(tc, "args", {})
                            }
                            logger.debug(f"Converted ToolCall to dict", tool_name=tc_dict["name"], tool_id=tc_dict["id"])
                            tool_calls.append(tc_dict)
                    else:
                        logger.warning(f"Unknown tool_call format", tc_type=type(tc).__name__)
            elif raw_tool_calls is not None:
                # tool_calls exists but is empty list - LLM chose not to call tools
                logger.debug(f"LLM returned empty tool_calls list - no tools called this iteration")
            
            # Log tool calls
            tool_names = [tc.get("name") for tc in tool_calls if tc.get("name")]
            logger.info(f"Supervisor iteration {react_iteration + 1}: {len(tool_calls)} tool calls", tools=tool_names)
            
            # Check for decision - handle both dict and object formats
            decision_call = None
            for tc in tool_calls:
                tool_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if tool_name == "make_final_decision":
                    decision_call = tc
                    break
            
            # Execute tools first to get results
            action_results = []
            decision_result = None
            
            for tool_call in tool_calls:
                # Extract tool name and args - handle both dict and object formats
                tool_name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)
                tool_args = tool_call.get("args", {}) if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
                tool_call_id = tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", None)
                
                if not tool_name:
                    logger.warning(f"Tool call missing name, skipping", tool_call=tool_call)
                    continue
                
                # Ensure we have a valid tool_call_id
                if not tool_call_id:
                    tool_call_id = f"call_{react_iteration}_{len(action_results)}"
                
                try:
                    # CRITICAL: Build comprehensive context for tool handlers
                    # Include all necessary data for write_draft_report to see:
                    # 1. Original user query
                    # 2. Deep search result
                    # 3. Clarification answers
                    # 4. Chapter summaries (existing chapters)
                    # Get settings for tool context (for TODO limit checking)
                    tool_settings = settings
                    if not tool_settings:
                        from src.config.settings import get_settings
                        tool_settings = get_settings()
                    
                    tool_context = {
                        "agent_memory_service": agent_memory_service,
                        "agent_file_service": agent_file_service,
                        "state": state,
                        "stream": stream,
                        "query": query,  # Original user query
                        "session_id": state.get("session_id"),
                        "session_factory": stream.app_state.get("session_factory") if stream else None,
                        "settings": tool_settings  # CRITICAL: Pass settings for TODO limit checking
                    }
                    
                    # Add deep_search_result to context
                    deep_search_result_raw = state.get("deep_search_result", "")
                    if isinstance(deep_search_result_raw, dict):
                        if "type" in deep_search_result_raw and deep_search_result_raw.get("type") == "override":
                            tool_context["deep_search_result"] = deep_search_result_raw.get("value", "")
                        elif "value" in deep_search_result_raw:
                            tool_context["deep_search_result"] = deep_search_result_raw.get("value", "")
                        else:
                            tool_context["deep_search_result"] = str(deep_search_result_raw)
                    else:
                        tool_context["deep_search_result"] = deep_search_result_raw or ""
                    
                    # Add clarification_context to context
                    tool_context["clarification_context"] = state.get("clarification_context", "")
                    
                    # CRITICAL: Add findings to context for write_draft_report to extract sources
                    # This allows write_draft_report to find sources even if finding parameter is not passed
                    findings_from_state = state.get("findings", state.get("agent_findings", []))
                    tool_context["findings"] = findings_from_state
                    logger.debug("Added findings to tool_context",
                               findings_count=len(findings_from_state),
                               note="write_draft_report can use this to extract sources if finding parameter not provided")
                    
                    # Add chapter summaries to context (for write_draft_report to see existing chapters)
                    # CRITICAL: Always initialize chapter_summaries, even if we can't load it from session
                    # This prevents "cannot access local variable" errors
                    if tool_name == "write_draft_report":
                        if tool_context.get("session_id") and tool_context.get("session_factory"):
                            try:
                                from src.workflow.research.session.manager import SessionManager
                                session_manager = SessionManager(tool_context["session_factory"])
                                session_data = await session_manager.get_session(tool_context["session_id"])
                                if session_data:
                                    # CRITICAL: session_data is ResearchSessionModel object, not dict
                                    # Use getattr or direct attribute access
                                    metadata = getattr(session_data, "session_metadata", None) or {}
                                    if not isinstance(metadata, dict):
                                        metadata = {}
                                    chapter_summaries = metadata.get("chapter_summaries", []) if isinstance(metadata, dict) else []
                                    tool_context["chapter_summaries"] = chapter_summaries
                                    logger.info("Added chapter summaries to context for write_draft_report",
                                               chapters_count=len(chapter_summaries))
                            except Exception as e:
                                logger.warning("Failed to get chapter summaries for context", error=str(e))
                                tool_context["chapter_summaries"] = []
                        else:
                            # No session_id or session_factory - initialize empty list
                            tool_context["chapter_summaries"] = []
                            logger.debug("No session_id or session_factory for chapter_summaries, using empty list")
                    
                    result = await SupervisorToolsRegistry.execute(
                        tool_name,
                        tool_args,
                        tool_context
                    )
                    
                    action_results.append({
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,  # CRITICAL: Store tool_name to identify read tools for history compression
                        "output": json.dumps(result, ensure_ascii=False)
                    })
                    
                    logger.info(f"Supervisor tool executed: {tool_name}", tool_call_id=tool_call_id)
                    
                    # Track usage of critical tools
                    if tool_name == "write_draft_report" or tool_name == "update_synthesized_report":
                        last_draft_write = react_iteration
                        logger.debug("Draft report updated", tool=tool_name, iteration=react_iteration)
                    elif tool_name == "write_supervisor_note":
                        last_memory_write = react_iteration
                        logger.debug("Supervisor memory written", iteration=react_iteration)
                    elif tool_name == "read_draft_report":
                        last_draft_read = react_iteration
                        logger.debug("Draft report read", iteration=react_iteration)
                    elif tool_name == "read_supervisor_file":
                        last_memory_read = react_iteration
                        logger.debug("Supervisor memory read", iteration=react_iteration)
                    
                    # Store decision result if this is make_final_decision
                    if tool_name == "make_final_decision":
                        try:
                            decision_result = json.loads(result) if isinstance(result, str) else result
                        except:
                            decision_result = result
                    
                except Exception as e:
                    logger.error(f"Supervisor tool failed: {tool_name}", error=str(e), exc_info=True)
                    action_results.append({
                        "tool_call_id": tool_call_id,
                        "output": json.dumps({"error": str(e)})
                    })
            
            # Process decision if make_final_decision was called
            if decision_result:
                final_decision["should_continue"] = decision_result.get("should_continue", False)
                final_decision["replanning_needed"] = decision_result.get("replanning_needed", False)
                final_decision["reasoning"] = decision_result.get("reasoning", "")
                decision_type = decision_result.get("decision", "finish" if not final_decision["should_continue"] else "continue")
                logger.info("Supervisor made decision", decision=decision_type, should_continue=final_decision["should_continue"], replanning_needed=final_decision["replanning_needed"], iteration=react_iteration)
                decision_made = True
                break  # Exit ReAct loop after decision
            
            # Track consecutive empty tool_calls FIRST (before any other checks)
            if not tool_calls:
                consecutive_empty_tool_calls += 1
                logger.debug(f"Empty tool_calls count: {consecutive_empty_tool_calls}")
            else:
                consecutive_empty_tool_calls = 0  # Reset counter if tools were called
                logger.debug("Tool calls present, resetting empty counter")
            
            # Force finish if supervisor repeatedly fails to call tools (2+ consecutive times)
            # This indicates the LLM is confused and not following instructions
            # Check this BEFORE adding reminder to history
            if consecutive_empty_tool_calls >= 2:
                logger.warning(f"Supervisor failed to call tools {consecutive_empty_tool_calls} times consecutively - forcing finish")
                # CRITICAL: If forced finalization, ensure we finish even if supervisor fails
                if force_finalization:
                    final_decision["reasoning"] = f"MANDATORY finalization: All tasks completed. Supervisor failed to call tools ({consecutive_empty_tool_calls} times) - proceeding to report generation."
                else:
                    final_decision["reasoning"] = f"Supervisor repeatedly failed to call tools ({consecutive_empty_tool_calls} times). Research completed after {react_iteration + 1} iterations."
                final_decision["should_continue"] = False
                final_decision["replanning_needed"] = False
                decision_made = True
                break
            
            # CRITICAL: Force decision if we're near max iterations and no decision made
            # Supervisor MUST call make_final_decision - if it doesn't, force it
            iterations_without_decision = react_iteration + 1
            # Force decision earlier (at 80% of max_iterations) to ensure supervisor completes review
            force_decision_threshold = int(max_iterations * 0.8)  # Force at 80% of max iterations
            if (iterations_without_decision >= force_decision_threshold) and not decision_made:
                # CRITICAL: If this is forced finalization, always finish regardless of limit
                if force_finalization:
                    logger.warning(f"MANDATORY finalization: forcing finish (call {supervisor_call_count + 1}, iterations {iterations_without_decision}/{max_iterations}) - supervisor did not call make_final_decision")
                    final_decision["should_continue"] = False  # Finish research - proceed to report generation
                    final_decision["replanning_needed"] = False
                    final_decision["reasoning"] = f"MANDATORY finalization: All tasks completed. Supervisor reached max iterations without calling make_final_decision - proceeding to report generation."
                    decision_made = True
                    break
                # Check supervisor call count - if at limit, finish; otherwise continue
                elif supervisor_call_count >= max_supervisor_calls:
                    logger.info(f"Supervisor call limit reached ({supervisor_call_count}/{max_supervisor_calls}) - finishing research")
                    final_decision["should_continue"] = False  # Finish research - proceed to report generation
                    final_decision["replanning_needed"] = False
                    final_decision["reasoning"] = f"Supervisor completed {supervisor_call_count} calls (max: {max_supervisor_calls}) - research complete, proceeding to report generation."
                else:
                    logger.info(f"Supervisor near max iterations ({iterations_without_decision}/{max_iterations}) - returning control to agents (call {supervisor_call_count}/{max_supervisor_calls})")
                    final_decision["should_continue"] = True  # Continue research - agents will work more and supervisor will be called again
                    final_decision["replanning_needed"] = False
                    final_decision["reasoning"] = f"Supervisor completed {iterations_without_decision} iterations - returning control to agents for continued research (call {supervisor_call_count}/{max_supervisor_calls})."
                decision_made = True
                break
            
            # Check if supervisor is not actively working with draft/memory and add reminders
            iterations_since_draft_write = react_iteration - last_draft_write if last_draft_write >= 0 else react_iteration + 1
            iterations_since_memory_write = react_iteration - last_memory_write if last_memory_write >= 0 else react_iteration + 1
            iterations_since_draft_read = react_iteration - last_draft_read if last_draft_read >= 0 else react_iteration + 1
            iterations_since_memory_read = react_iteration - last_memory_read if last_memory_read >= 0 else react_iteration + 1
            
            # Add reminders if supervisor hasn't worked with draft/memory recently
            reminders = []
            if iterations_since_draft_read >= 2:
                reminders.append("You haven't read draft_report.md recently - call read_draft_report to see RAW FINDINGS and current synthesis")
            if iterations_since_memory_read >= 2:
                reminders.append("You haven't read your supervisor file recently - call read_supervisor_file to review your notes")
            if iterations_since_draft_write >= 3:
                reminders.append("You haven't synthesized RAW findings in 3+ iterations - call update_synthesized_report to write structured report")
            if iterations_since_memory_write >= 3:
                reminders.append("You haven't written to your supervisor file in 3+ iterations - call write_supervisor_note to record observations")
            
            # Only add reminder if we haven't forced finish
            if not tool_calls:
                # No tool calls - LLM violated the instruction to always call tools
                # This is a problem - LLM should always call at least one tool
                response_content = str(response.content) if hasattr(response, "content") else ""
                logger.warning("Supervisor returned no tool calls - LLM violated instruction to always call tools", 
                             react_iteration=react_iteration + 1,
                             max_iterations=max_iterations,
                             response_content_preview=response_content[:300])
                
                # Add a reminder message to the conversation to enforce tool calling
                reminder_content = "ERROR: You must call at least one tool on every iteration. You returned no tool calls. Please call a tool now - either review findings, update documents, assign tasks, or make a final decision using make_final_decision."
                if reminders:
                    reminder_content += "\n\nALSO: " + " | ".join(reminders)
                reminder_message = {
                    "role": "user",
                    "content": reminder_content
                }
                agent_history.append(reminder_message)
                
                # Continue loop - LLM should call tools after reminder
                logger.debug("Added reminder to call tools, continuing ReAct loop")
            elif reminders:
                # Supervisor called tools but hasn't worked with draft/memory - add gentle reminder
                reminder_content = "REMINDER: " + " | ".join(reminders) + " - Work actively with draft_report.md and supervisor.md throughout the research."
                reminder_message = {
                    "role": "user",
                    "content": reminder_content
                }
                agent_history.append(reminder_message)
                logger.debug("Added reminder to work with draft/memory", reminders=reminders)
            
            # Add to history - store the actual AIMessage object to preserve tool_call IDs
            # This ensures call_ids match between tool calls and tool responses
            # Convert tool_calls to dict format for storage (they're already in dict format from extraction above)
            stored_tool_calls = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    # Ensure all required fields are present - preserve original id!
                    stored_tc = {
                        "id": tc.get("id"),  # Preserve original ID - don't generate new one!
                        "name": tc.get("name") or "",
                        "args": tc.get("args") or {}
                    }
                    # Only generate ID if it's truly missing
                    if not stored_tc["id"]:
                        stored_tc["id"] = f"call_{react_iteration}_{len(stored_tool_calls)}"
                    stored_tool_calls.append(stored_tc)
                else:
                    # Fallback: convert object to dict - preserve original id
                    stored_id = getattr(tc, "id", None)
                    if not stored_id:
                        stored_id = f"call_{react_iteration}_{len(stored_tool_calls)}"
                    stored_tool_calls.append({
                        "id": stored_id,
                        "name": getattr(tc, "name", None) or "",
                        "args": getattr(tc, "args", None) or {}
                    })
            
            # Store in history - preserve tool_call IDs exactly as they were
            # Store as dict with exact IDs preserved (don't store object, it may not serialize properly)
            agent_history.append({
                "role": "assistant",
                "content": response.content if hasattr(response, "content") else "",
                "tool_calls": stored_tool_calls  # Store with exact IDs preserved
            })
            
            # CRITICAL: For read tools (read_draft_report, read_main_document, read_supervisor_file),
            # only keep the LATEST response in full, replace older ones with placeholders
            # This prevents context bloat from accumulating large file contents
            read_tools_to_compress = {"read_draft_report", "read_main_document", "read_supervisor_file"}
            
            for result in action_results:
                tool_name = result.get("tool_name", "")
                tool_call_id = result["tool_call_id"]
                output = result["output"]
                
                # If this is a read tool, replace old responses from the same tool with placeholders
                if tool_name in read_tools_to_compress:
                    # Find previous tool_call_id for this read tool (if exists)
                    previous_tool_call_id = read_tool_call_ids.get(tool_name)
                    
                    if previous_tool_call_id:
                        # Replace old tool response with placeholder
                        for i, msg in enumerate(agent_history):
                            if msg.get("role") == "tool" and msg.get("tool_call_id") == previous_tool_call_id:
                                placeholder = f"[Previous {tool_name} output - see latest call for current content]"
                                agent_history[i] = {
                                    "role": "tool",
                                    "content": json.dumps({"placeholder": placeholder, "note": "Previous output replaced to save context"}),
                                    "tool_call_id": previous_tool_call_id
                                }
                                logger.debug(f"Replaced old {tool_name} response with placeholder",
                                           old_tool_call_id=previous_tool_call_id,
                                           new_tool_call_id=tool_call_id)
                                break
                    
                    # Update mapping: this is now the latest tool_call_id for this read tool
                    read_tool_call_ids[tool_name] = tool_call_id
                
                # Add current tool response (always add latest, even if it's a read tool)
                agent_history.append({
                    "role": "tool",
                    "content": output,
                    "tool_call_id": tool_call_id
                })
        
        except Exception as e:
            logger.error(f"Supervisor iteration {react_iteration} failed", error=str(e))
            break
    
    if not decision_made:
        # CRITICAL: Supervisor did NOT call make_final_decision - this is a problem!
        # Force a decision to prevent hanging
        logger.warning(f"Supervisor did NOT call make_final_decision after {max_iterations} iterations - forcing decision",
                      max_iterations=max_iterations,
                      agent_history_length=len(agent_history),
                      note="Supervisor should ALWAYS call make_final_decision - this is a bug!")
        
        # CRITICAL: If this is forced finalization, always finish regardless of limit
        if force_finalization:
            logger.info(f"MANDATORY finalization: forcing finish (no decision made, call {supervisor_call_count + 1})",
                       max_iterations=max_iterations,
                       agent_history_length=len(agent_history))
            final_decision["should_continue"] = False
            final_decision["reasoning"] = f"MANDATORY finalization: All tasks completed. Synthesizing findings and finishing research."
        # Check supervisor call count - if at limit, finish; otherwise continue
        elif supervisor_call_count >= max_supervisor_calls:
            logger.info(f"Supervisor call limit reached ({supervisor_call_count}/{max_supervisor_calls}) - finishing research",
                         max_iterations=max_iterations,
                         agent_history_length=len(agent_history))
            final_decision["should_continue"] = False
            final_decision["reasoning"] = f"Supervisor reached call limit ({supervisor_call_count}/{max_supervisor_calls}) - research complete."
        else:
            logger.info(f"Supervisor reached max iterations, continuing research (call {supervisor_call_count}/{max_supervisor_calls})",
                         max_iterations=max_iterations,
                         agent_history_length=len(agent_history))
            final_decision["should_continue"] = True
            final_decision["reasoning"] = f"Supervisor reached max iterations ({max_iterations}) for this call - returning control to agents."
    
    # CRITICAL: Before finishing, ensure draft_report.md is filled with findings
    # If supervisor didn't write draft_report, we must create it from findings
    if not final_decision["should_continue"]:
        # Write final note to supervisor file about completion
        if agent_file_service:
            try:
                supervisor_file = await agent_file_service.read_agent_file("supervisor")
                existing_notes = supervisor_file.get("notes", [])
                final_note = f"Research completed. Decision: finish. Total findings: {len(findings)}. Draft report finalized."
                # CRITICAL: Use datetime from import at top of file
                from datetime import datetime as dt
                existing_notes.append(f"[{dt.now().strftime('%Y-%m-%d %H:%M:%S')}] {final_note}")
                existing_notes = existing_notes[-100:]
                await agent_file_service.write_agent_file(
                    agent_id="supervisor",
                    notes=existing_notes,
                    character=supervisor_file.get("character", "Research supervisor coordinating team of agents"),
                    preferences=supervisor_file.get("preferences", "Focus on comprehensive, diverse research coverage")
                )
                logger.info("Supervisor final note written")
            except Exception as e:
                logger.warning("Failed to write supervisor final note", error=str(e))
        
        agent_memory_service = stream.app_state.get("agent_memory_service") if stream else None
        if agent_memory_service:
            try:
                # Check if draft_report exists and has content
                draft_content = await agent_memory_service.file_manager.read_file("draft_report.md")
                if len(draft_content) < 500:  # Too short or empty
                    logger.warning("Draft report is too short - creating comprehensive draft from findings", 
                                 draft_length=len(draft_content), findings_count=len(findings))
                    
                    # Create comprehensive draft report from all findings (NO TRUNCATION)
                    from datetime import datetime
                    findings_text = "\n\n".join([
                        f"## {f.get('topic', 'Unknown Topic')}\n\n"
                        f"**Agent:** {f.get('agent_id', 'unknown')}\n"
                        f"**Summary:** {f.get('summary', 'No summary')}\n\n"  # FULL summary, no truncation
                        f"**Key Findings:**\n" + "\n".join([f"- {kf}" for kf in f.get('key_findings', [])]) + "\n\n"  # ALL key findings, no truncation
                        f"**Sources:** {len(f.get('sources', []))}\n"
                        f"**Confidence:** {f.get('confidence', 'unknown')}\n"
                        for f in findings  # ALL findings, no truncation
                    ])
                    
                    comprehensive_draft = f"""# Research Report Draft

**Query:** {query}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Findings:** {len(findings)}

## Executive Summary

This report synthesizes findings from {len(agent_characteristics)} research agents working on: {query}

## Detailed Findings

{findings_text}

## Conclusion

Research completed with {len(findings)} findings from multiple agents covering various aspects of the topic.
"""
                    
                    await agent_memory_service.file_manager.write_file("draft_report.md", comprehensive_draft)
                    logger.info("Created comprehensive draft report from findings", 
                              draft_length=len(comprehensive_draft), findings_count=len(findings))
            except FileNotFoundError:
                # Draft report doesn't exist - create it (NO TRUNCATION)
                logger.warning("Draft report not found - creating from findings", findings_count=len(findings))
                from datetime import datetime
                findings_text = "\n\n".join([
                    f"## {f.get('topic', 'Unknown Topic')}\n\n"
                    f"**Agent:** {f.get('agent_id', 'unknown')}\n"
                    f"**Summary:** {f.get('summary', 'No summary')}\n\n"  # FULL summary, no truncation
                    f"**Key Findings:**\n" + "\n".join([f"- {kf}" for kf in f.get('key_findings', [])]) + "\n\n"  # ALL key findings, no truncation
                    for f in findings  # ALL findings, no truncation
                ])
                
                comprehensive_draft = f"""# Research Report Draft

**Query:** {query}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Findings:** {len(findings)}

## Executive Summary

This report synthesizes findings from {len(agent_characteristics)} research agents working on: {query}

## Detailed Findings

{findings_text}

## Conclusion

Research completed with {len(findings)} findings from multiple agents covering various aspects of the topic.
"""
                await agent_memory_service.file_manager.write_file("draft_report.md", comprehensive_draft)
                logger.info("Created draft report from findings", draft_length=len(comprehensive_draft))
            except Exception as e:
                logger.error("Failed to ensure draft_report exists", error=str(e))
    
    logger.info("Supervisor agent returning decision", 
               should_continue=final_decision["should_continue"],
               decision_made=decision_made,
               reasoning_preview=final_decision.get("reasoning", "")[:200])
    
    # CRITICAL: Update status after supervisor review completes
    # This ensures frontend doesn't show stale "Supervisor reviewing iteration #1..." status
    if stream:
        findings_count = len(findings) if findings else 0
        if not final_decision.get("should_continue", False):
            # Supervisor decided to finish
            stream.emit_status(f"✅ Supervisor finalized report ({findings_count} findings processed) - generating final result...", step="supervisor")
        else:
            # Supervisor decided to continue - check if there are pending tasks
            if agent_file_service:
                try:
                    agent_files = await agent_file_service.file_manager.list_files("agents/agent_*.md")
                    all_agent_ids = []
                    for file_path in agent_files:
                        agent_id = file_path.replace("agents/", "").replace(".md", "")
                        if agent_id.startswith("agent_") and agent_id != "supervisor":
                            all_agent_ids.append(agent_id)
                    
                    total_pending = 0
                    for agent_id in all_agent_ids:
                        agent_file = await agent_file_service.read_agent_file(agent_id)
                        todos = agent_file.get("todos", [])
                        pending_tasks = [t for t in todos if t.status == "pending"]
                        in_progress_tasks = [t for t in todos if t.status == "in_progress"]
                        total_pending += len(pending_tasks) + len(in_progress_tasks)
                    
                    if total_pending > 0:
                        stream.emit_status(f"🚀 Agents continuing work ({total_pending} tasks remaining)", step="agents")
                    else:
                        stream.emit_status("🚀 Research continuing...", step="agents")
                except Exception as e:
                    logger.warning("Failed to check pending tasks for status update", error=str(e))
                    stream.emit_status("🚀 Research continuing...", step="agents")
            else:
                stream.emit_status("🚀 Research continuing...", step="agents")
    
    return final_decision


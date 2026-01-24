"""Supervisor agent as LangGraph agent with ReAct format and memory tools.

The supervisor is a full LangGraph agent that:
- Reviews agent findings and updates main research document
- Creates and edits agent todos
- Identifies research gaps
- Makes decisions about continuing/replanning/finishing
"""

import asyncio
import json
import re
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

# CRITICAL: Lock for sequential chapter writing to prevent concurrent writes
# This ensures chapters are written one at a time, even if multiple findings are processed
_draft_report_write_lock = asyncio.Lock()


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


class ReturnTaskToProgressArgs(BaseModel):
    """Arguments for returning a task to progress (rejecting finding and asking agent to rework)."""
    agent_id: str = Field(description="Agent ID whose task should be returned to progress")
    todo_title: str = Field(description="Title of the task to return to progress")
    motivation: str = Field(description="Your motivation for asking the agent to continue work - explain why the finding needs improvement and what you want to achieve")
    instructions: str = Field(description="Specific instructions for the agent on what to do - what information to find, what aspects to investigate, what to improve")


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
    # Fallback: try to get from stream.app_state if not in context
    if not agent_memory_service:
        stream = context.get("stream")
        if stream and hasattr(stream, "app_state"):
            agent_memory_service = stream.app_state.get("agent_memory_service")
    
    session_id = context.get("session_id")
    session_factory = context.get("session_factory")
    query = context.get("query", "")
    deep_search_result = context.get("deep_search_result", "")
    clarification_context = context.get("clarification_context", "")
    chapter_summaries = context.get("chapter_summaries", [])
    user_language = context.get("user_language", "English")
    
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
                # CRITICAL BUG FIX: Check if sources field is a number instead of list
                # This happens when LLM passes sources_count instead of sources, or when finding structure is corrupted
                if "sources" in finding_data and not isinstance(finding_data["sources"], list):
                    sources_value = finding_data.get("sources")
                    logger.error("CRITICAL BUG: finding_data.sources is not a list!",
                               sources_type=type(sources_value).__name__,
                               sources_value=sources_value,
                               finding_keys=list(finding_data.keys()),
                               has_sources_count="sources_count" in finding_data,
                               sources_count_value=finding_data.get("sources_count") if "sources_count" in finding_data else None,
                               chapter_title=chapter_title,
                               note="This is a BUG - sources should always be a list. Finding may be corrupted or LLM passed wrong structure.")
                    # Try to fix: if sources_count exists and sources is a number, remove wrong sources field
                    if isinstance(sources_value, (int, float)) and "sources_count" in finding_data:
                        # Remove wrong sources field - will try to get from state.findings
                        del finding_data["sources"]
                        logger.warning("Removed wrong sources field (was number), will try to get from state.findings",
                                     original_sources_value=sources_value)
                    else:
                        # Set to empty list to prevent errors
                        finding_data["sources"] = []
                        logger.warning("Set sources to empty list to prevent errors",
                                     original_sources_value=sources_value)
            elif isinstance(finding_data_raw, str):
                # String passed - try to find corresponding finding from state by topic/chapter_title
                # This handles cases where LLM passes topic string or chapter_title instead of full finding dict
                findings_from_context = context.get("findings", [])
                if findings_from_context:
                    finding_data_raw_normalized = finding_data_raw.strip().lower()
                    
                    # Try multiple matching strategies:
                    # 1. Exact topic match
                    # 2. Topic in finding_data_raw string (partial match)
                    # 3. finding_data_raw in topic (partial match)
                    # 4. Match by chapter_title if it contains topic
                    for f in findings_from_context:
                        if isinstance(f, dict):
                            topic = f.get("topic", "").strip().lower()
                            
                            # Match 1: Exact topic match
                            if topic == finding_data_raw_normalized:
                                finding_data = f
                                logger.info("Found finding from state by exact topic match",
                                           finding_data_raw_preview=finding_data_raw[:50],
                                           finding_topic=topic,
                                           note="String finding_data_raw was replaced with full finding dict from state")
                                break
                            
                            # Match 2: Topic in finding_data_raw (partial match - most common when LLM passes chapter_title)
                            if topic and topic in finding_data_raw_normalized:
                                finding_data = f
                                logger.info("Found finding from state by topic in finding_data_raw (partial match)",
                                           finding_data_raw_preview=finding_data_raw[:50],
                                           finding_topic=topic,
                                           note="String finding_data_raw contains topic - replaced with full finding dict")
                                break
                            
                            # Match 3: finding_data_raw in topic (reverse partial match)
                            if finding_data_raw_normalized and finding_data_raw_normalized in topic:
                                finding_data = f
                                logger.info("Found finding from state by finding_data_raw in topic (reverse partial match)",
                                           finding_data_raw_preview=finding_data_raw[:50],
                                           finding_topic=topic,
                                           note="String finding_data_raw is substring of topic - replaced with full finding dict")
                                break
                            
                            # Match 4: If chapter_title provided, check if topic matches chapter_title
                            if chapter_title:
                                chapter_title_normalized = chapter_title.strip().lower()
                                if topic == chapter_title_normalized or topic in chapter_title_normalized or chapter_title_normalized in topic:
                                    finding_data = f
                                    logger.info("Found finding from state by chapter_title match",
                                               chapter_title=chapter_title[:50],
                                               finding_topic=topic,
                                               note="Topic matches chapter_title - replaced with full finding dict")
                                    break
                        
                        if finding_data:
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
        
        # CRITICAL: First, try to get finding from current_finding in context (from queue)
        # This is the most reliable source - it's the finding supervisor is currently processing
        if not finding_data:
            current_finding_from_context = context.get("current_finding")
            if current_finding_from_context and isinstance(current_finding_from_context, dict):
                finding_data = current_finding_from_context
                logger.info("Found finding from context.current_finding (from queue)",
                           chapter_title=chapter_title,
                           finding_topic=finding_data.get("topic", "unknown"),
                           sources_count=len(finding_data.get("sources", [])),
                           note="This is the finding from supervisor queue - most reliable source")
        
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
                            # CRITICAL BUG FIX: Normalize sources in finding from state
                            # Ensure sources is always a list, not a number
                            if "sources" in finding_data:
                                sources_value = finding_data["sources"]
                                if not isinstance(sources_value, list):
                                    logger.error("CRITICAL BUG: Finding from state has sources as non-list!",
                                               sources_type=type(sources_value).__name__,
                                               sources_value=sources_value,
                                               finding_topic=topic,
                                               finding_keys=list(finding_data.keys()),
                                               note="This is a BUG - finding from state should have sources as list")
                                    # Fix: set to empty list, will try to get from other fields
                                    finding_data["sources"] = []
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
        # CRITICAL: Include ALL chapters, not just last 5 - supervisor needs to see all summaries to write better integrated chapters
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
                    "summary_preview": ch.get("summary", "")[:500] if isinstance(ch, dict) else ""  # Increased to 500 chars for better context
                }
                for ch in chapter_summaries  # ALL chapters, not just last 5 - supervisor needs full context
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
        
        # CRITICAL: Check for duplicate chapters - ONLY by finding topic
        # Each chapter is HARD-LINKED to a finding - if finding with this topic already has a chapter, don't add again
        # This is the ONLY check needed - no similarity checks, no title checks (titles can vary)
        if finding_data and isinstance(finding_data, dict) and chapter_summaries:
            finding_topic = finding_data.get("topic", "").strip().lower()
            if finding_topic:
                for ch in chapter_summaries:
                    if isinstance(ch, dict):
                        existing_topic = ch.get("topic", "").strip().lower()
                        # Exact topic match - this finding already has a chapter
                        if existing_topic and existing_topic == finding_topic:
                            logger.warning("DUPLICATE DETECTED: Chapter for this finding topic already exists - skipping",
                                         chapter_title=chapter_title,
                                         finding_topic=finding_topic,
                                         existing_chapter_number=ch.get("chapter_number"),
                                         existing_chapter_title=ch.get("chapter_title", "Unknown"),
                                         note="Finding with this topic was already processed and has a chapter")
                            return {
                                "success": False,
                                "message": f"Chapter for finding topic '{finding_topic}' already exists in draft report (Chapter {ch.get('chapter_number', '?')}: {ch.get('chapter_title', 'Unknown')}). This finding was already processed.",
                                "chapter_number": None,
                                "duplicate_reason": "topic_match",
                                "existing_chapter_number": ch.get("chapter_number"),
                                "existing_chapter_title": ch.get("chapter_title")
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
        
        # CRITICAL: Normalize finding_data.sources - ensure it's always a list
        # LLM might pass sources_count (number) instead of sources (list), or finding might have wrong structure
        if finding_data and isinstance(finding_data, dict):
            sources_raw = finding_data.get("sources")
            if sources_raw:
                if isinstance(sources_raw, list):
                    sources = sources_raw
                elif isinstance(sources_raw, (int, float)):
                    # CRITICAL BUG FIX: If sources is a number, it's actually sources_count
                    # Try to get real sources from finding_data or state.findings
                    logger.warning("finding_data.sources is a number (likely sources_count), not a list - trying to find real sources",
                                 sources_value=sources_raw,
                                 chapter_title=chapter_title,
                                 finding_keys=list(finding_data.keys()))
                    # Check if finding_data has other source-related fields
                    if "all_sources" in finding_data and isinstance(finding_data["all_sources"], list):
                        sources = finding_data["all_sources"]
                        logger.info("Found all_sources in finding_data", sources_count=len(sources))
                    else:
                        # Will try to find from state.findings below
                        sources = []
                else:
                    # Try to convert to list if possible
                    sources = list(sources_raw) if hasattr(sources_raw, '__iter__') and not isinstance(sources_raw, str) else []
                    logger.warning("finding_data.sources is not a list, converted", 
                                 original_type=type(sources_raw).__name__,
                                 sources_count=len(sources) if isinstance(sources, list) else 0)
            
            if sources:
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
                            sources_check = f.get("sources")
                            if sources_check and isinstance(sources_check, list) and len(sources_check) > 0:
                                matched_finding = f
                                logger.info("Found sources in state findings by matching topic/title",
                                           sources_count=len(sources_check),
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
                            if isinstance(f, dict):
                                sources_check = f.get("sources")
                                if sources_check and isinstance(sources_check, list) and len(sources_check) > 0:
                                    finding_topic = f.get("topic", "")
                                    # Check if this finding's topic is already in draft_report
                                    if finding_topic and finding_topic not in current_draft:
                                        matched_finding = f
                                        logger.info("Found sources in state findings (finding not yet added as chapter)",
                                                   sources_count=len(sources_check),
                                                   chapter_title=chapter_title,
                                                   finding_topic=finding_topic,
                                                   note="Using first finding with sources that hasn't been added yet")
                                        break
                    except Exception as e:
                        logger.warning("Failed to check draft_report for finding matching", error=str(e))
                
                if matched_finding and matched_finding.get("sources"):
                    sources_raw = matched_finding.get("sources", [])
                    # CRITICAL: Ensure sources is a list, not a number (sources_count)
                    if isinstance(sources_raw, list):
                        sources = sources_raw
                    elif isinstance(sources_raw, (int, float)):
                        # If sources is a number (sources_count), treat as empty list
                        logger.warning("matched_finding.sources is a number (sources_count), not a list - treating as empty",
                                     sources_value=sources_raw,
                                     chapter_title=chapter_title)
                        sources = []
                    else:
                        # Try to convert to list if possible
                        sources = list(sources_raw) if hasattr(sources_raw, '__iter__') and not isinstance(sources_raw, str) else []
        
        # Format sources section
        # CRITICAL: Deduplicate sources by URL to prevent duplicate sources in the same chapter
        # CRITICAL: Ensure sources is a list before processing
        if sources and isinstance(sources, list) and len(sources) > 0:
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
                    
                    # CRITICAL: Decode URL-encoded title and URL if needed
                    # Handle both URL-encoded strings and normal strings
                    from urllib.parse import unquote
                    try:
                        # Try to decode URL-encoded title
                        if '%' in title:
                            title_decoded = unquote(title, encoding='utf-8')
                            if title_decoded != title:
                                title = title_decoded
                                logger.debug("Decoded URL-encoded title", original=title[:50], decoded=title_decoded[:50])
                    except Exception as e:
                        logger.warning("Failed to decode title", title_preview=title[:50], error=str(e))
                    
                    # Clean title - remove any HTML entities or special characters that might break markdown
                    # Replace common problematic characters
                    title_clean = title.replace('[', '(').replace(']', ')')  # Replace brackets that break markdown links
                    title_clean = title_clean.strip()
                    
                    # Ensure title is not empty after cleaning
                    if not title_clean:
                        title_clean = "Source"
                    
                    # CRITICAL: Ensure URL is properly formatted
                    url_clean = url.strip() if url else ""
                    
                    if url_clean:
                        # Try to decode URL if it's URL-encoded
                        try:
                            if '%' in url_clean:
                                url_decoded = unquote(url_clean, encoding='utf-8')
                                if url_decoded != url_clean:
                                    url_clean = url_decoded
                                    logger.debug("Decoded URL-encoded URL", original=url_clean[:50], decoded=url_decoded[:50])
                        except Exception as e:
                            logger.warning("Failed to decode URL", url_preview=url_clean[:50], error=str(e))
                        
                        # Format as markdown link
                        sources_list.append(f"- [{title_clean}]({url_clean})")
                    else:
                        # Also check for duplicate titles if no URL
                        existing_titles = [s.split(']')[0].replace('- [', '').lower() for s in sources_list]
                        if title_clean.lower() not in existing_titles:
                            sources_list.append(f"- {title_clean}")
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
        
        # CRITICAL: Clean content from any chapter headers and sources sections that LLM might have added
        # LLM should NOT add headers or sources, but sometimes does - remove them to prevent duplication
        # Remove any "## Chapter" or "# Chapter" lines from content
        # CRITICAL: Also remove headers that match chapter_title (any number of #)
        # CRITICAL: Also remove "## Sources", "## References", or any source lists - sources are added automatically
        content_lines = content.split('\n')
        cleaned_content_lines = []
        in_sources_section = False
        chapter_title_normalized = chapter_title.strip().lower() if chapter_title else ""
        
        for i, line in enumerate(content_lines):
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
            
            # CRITICAL: Remove headers that match chapter_title (any number of # at start)
            # This catches cases like "# Title" or "## Title" where Title matches chapter_title
            if chapter_title_normalized:
                # Extract text after # symbols
                header_match = re.match(r'^(#+)\s+(.+)$', line)
                if header_match:
                    header_text = header_match.group(2).strip()
                    header_text_normalized = header_text.lower()
                    # Check if header text matches chapter_title (exact or partial match)
                    if (header_text_normalized == chapter_title_normalized or 
                        header_text_normalized in chapter_title_normalized or 
                        chapter_title_normalized in header_text_normalized):
                        logger.warning("Removed duplicate title header from content (matches chapter_title)",
                                     line=line[:100],
                                     chapter_title=chapter_title,
                                     header_text=header_text[:100],
                                     note="LLM added header that matches chapter_title - removed to prevent duplication")
                        continue
            
            # CRITICAL: Remove sources sections - they are added automatically
            # Check for "## Sources", "## References", "## Ссылки", "## Источники", etc.
            if re.match(r'^##\s+(Sources|References|Ссылки|Источники|Литература|Bibliography)', line, re.IGNORECASE):
                in_sources_section = True
                logger.warning("Removed sources section header from content",
                             line=line[:100],
                             chapter_title=chapter_title,
                             note="Sources are added automatically - removed to prevent duplication")
                continue
            
            # If we're in a sources section, skip all lines until next section or end
            if in_sources_section:
                # Check if this is a new section (starts with ##)
                if re.match(r'^##\s+', line):
                    # New section starts - stop skipping, but don't add this line (it's a new section header)
                    in_sources_section = False
                    # Don't add this line - it's likely another section header
                    continue
                # Skip all lines in sources section
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
        
        # CRITICAL: Use lock to ensure sequential chapter writing
        # This prevents concurrent writes even if multiple findings are processed
        logger.info(
            "SUPERVISOR: Acquiring draft_report write lock for sequential chapter writing",
            chapter_title=chapter_title,
            chapter_number=chapter_number,
            note="Lock ensures chapters are written sequentially in order of processing"
        )
        
        async with _draft_report_write_lock:
            logger.info(
                "SUPERVISOR: Acquired draft_report write lock - writing chapter",
                chapter_title=chapter_title,
                chapter_number=chapter_number,
                note="Lock acquired - will re-read draft_report to get latest version before writing"
            )
            
            # Re-read current draft inside lock to get latest version
            # This ensures we have the most up-to-date content even if another chapter was written
            try:
                current_locked = await agent_memory_service.file_manager.read_file(draft_file)
                logger.info(
                    "SUPERVISOR: Re-read draft_report inside lock",
                    chapter_title=chapter_title,
                    chapter_number=chapter_number,
                    current_draft_length=len(current_locked),
                    note="Got latest version of draft_report - will recalculate chapter number if needed"
                )
            except FileNotFoundError:
                current_locked = ""
                logger.info(
                    "SUPERVISOR: Draft report not found - will create new",
                    chapter_title=chapter_title,
                    chapter_number=chapter_number
                )
            
            # Re-check chapter number inside lock to prevent duplicates
            existing_chapter_numbers_locked = []
            for line in current_locked.split('\n'):
                match = re.match(chapter_pattern, line, re.IGNORECASE)
                if match:
                    try:
                        chapter_num = int(match.group(1))
                        existing_chapter_numbers_locked.append(chapter_num)
                    except (ValueError, IndexError):
                        pass
                match = re.match(single_hash_pattern, line, re.IGNORECASE)
                if match:
                    try:
                        chapter_num = int(match.group(1))
                        existing_chapter_numbers_locked.append(chapter_num)
                    except (ValueError, IndexError):
                        pass
            
            # Recalculate chapter number inside lock
            if existing_chapter_numbers_locked:
                chapter_number_locked = max(existing_chapter_numbers_locked) + 1
            else:
                chapter_number_locked = 1
            
            # Update chapter with correct number
            if chapter_number_locked != chapter_number:
                logger.info("Chapter number updated inside lock",
                           original_number=chapter_number,
                           new_number=chapter_number_locked,
                           note="Another chapter was written while waiting for lock")
                chapter = f"""

---

## Chapter {chapter_number_locked}: {chapter_title}

{cleaned_content}{sources_section}

"""
                chapter_number = chapter_number_locked
            
            # Append chapter to draft
            updated = current_locked + chapter
            logger.info(
                "SUPERVISOR: Writing chapter to draft_report (inside lock)",
                chapter_title=chapter_title,
                chapter_number=chapter_number_locked,
                updated_draft_length=len(updated),
                note="Writing chapter sequentially - lock ensures no concurrent writes"
            )
            
            await agent_memory_service.file_manager.write_file(draft_file, updated)
            
            logger.info(
                "SUPERVISOR: Chapter written successfully (releasing lock)",
                chapter_title=chapter_title,
                chapter_number=chapter_number_locked,
                note="Chapter written - lock will be released, next chapter can be written"
            )
        
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
                
                # Create chapter summary with BRIEF summary (if chapter is large, use LLM to create concise summary)
                # CRITICAL: datetime is already imported at module level, use it directly
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # CRITICAL: Create BRIEF summary for chapter_summaries
                # If chapter is large (>2000 chars), use LLM to create a concise summary (250-300 words)
                # If chapter is small, use full content as summary
                chapter_content_length = len(content) if content else 0
                brief_summary = None
                
                if chapter_content_length > 2000:
                    # Large chapter - use LLM to create brief summary
                    llm_instance = context.get("llm")
                    if llm_instance:
                        try:
                            logger.info("Creating brief chapter summary using LLM",
                                      chapter_number=chapter_number,
                                      chapter_title=chapter_title,
                                      content_length=chapter_content_length,
                                      note="Large chapter detected - using LLM to generate concise summary")
                            
                            # Get user language for summary
                            user_language = context.get("user_language", "English")
                            
                            # Create prompt for LLM to generate brief summary
                            # Use full content without truncation - LLM will handle token limits
                            summary_prompt = f"""Create a brief, concise summary of the following chapter content.

Chapter Title: {chapter_title}
Chapter Number: {chapter_number}

Chapter Content:
{content}

Requirements:
- Summary should be 250-300 words (not characters, but words)
- Focus on key points, main findings, and important information
- Write in {user_language} - the same language as the chapter content
- Be comprehensive but concise
- Do NOT include chapter title or number in the summary
- Write ONLY the summary text, no additional formatting

Summary:"""
                            
                            # Call LLM to generate summary
                            from langchain_core.messages import SystemMessage, HumanMessage
                            messages = [
                                SystemMessage(content="You are a helpful assistant that creates concise summaries of research chapters."),
                                HumanMessage(content=summary_prompt)
                            ]
                            
                            llm_response = await llm_instance.ainvoke(messages)
                            
                            # Extract summary from LLM response
                            if hasattr(llm_response, 'content'):
                                brief_summary = llm_response.content.strip()
                            elif isinstance(llm_response, str):
                                brief_summary = llm_response.strip()
                            else:
                                brief_summary = str(llm_response).strip()
                            
                            # Validate summary length (should be 250-300 words, but allow 200-400 for flexibility)
                            word_count = len(brief_summary.split())
                            if word_count < 150:
                                logger.warning("LLM-generated summary is too short",
                                            word_count=word_count,
                                            summary_preview=brief_summary[:200],
                                            note="Summary is shorter than expected, but using it anyway")
                            elif word_count > 500:
                                logger.warning("LLM-generated summary is too long, truncating",
                                            word_count=word_count,
                                            note="Summary is longer than expected, truncating to ~300 words")
                                # Truncate to approximately 300 words
                                words = brief_summary.split()
                                brief_summary = ' '.join(words[:300])
                            
                            logger.info("LLM-generated chapter summary created",
                                      chapter_number=chapter_number,
                                      summary_length=len(brief_summary),
                                      summary_word_count=len(brief_summary.split()),
                                      summary_preview=brief_summary[:200],
                                      note="Brief summary successfully generated by LLM")
                        except Exception as e:
                            logger.error("Failed to generate brief summary using LLM, using fallback",
                                      error=str(e),
                                      error_type=type(e).__name__,
                                      chapter_number=chapter_number,
                                      note="Will use finding summary or content truncation as fallback")
                            brief_summary = None
                    else:
                        logger.warning("LLM not available in context, cannot generate brief summary",
                                     chapter_number=chapter_number,
                                     note="Will use finding summary or content truncation as fallback")
                
                # Fallback: Use finding summary if available and concise
                if not brief_summary and finding_data and isinstance(finding_data, dict):
                    finding_summary = finding_data.get("summary", "")
                    if finding_summary:
                        if len(finding_summary) <= 400:  # Use if reasonably concise
                            brief_summary = finding_summary
                        else:
                            # Truncate finding summary to ~300 words
                            words = finding_summary.split()
                            brief_summary = ' '.join(words[:300])
                
                # Final fallback: Use content truncation for small chapters or if all else fails
                if not brief_summary:
                    if chapter_content_length <= 2000:
                        # Small chapter - use full content as summary
                        brief_summary = content
                    else:
                        # Large chapter but LLM failed - use smart truncation
                        first_paragraph_end = content.find('\n\n', 0, 500)
                        if first_paragraph_end > 0 and first_paragraph_end < 300:
                            brief_summary = content[:first_paragraph_end].strip()
                        else:
                            # Fallback: first 250 chars, try to end at sentence boundary
                            brief_summary = content[:250]
                            last_period = brief_summary.rfind('.')
                            if last_period > 150:
                                brief_summary = brief_summary[:last_period + 1]
                
                chapter_summary = {
                    "chapter_number": chapter_number,
                    "chapter_title": chapter_title,
                    "timestamp": timestamp,
                    "agent_id": finding_data.get("agent_id", "unknown") if finding_data and isinstance(finding_data, dict) else "unknown",
                    "topic": finding_data.get("topic", chapter_title) if finding_data and isinstance(finding_data, dict) else chapter_title,
                    "summary": brief_summary,  # BRIEF summary (250-300 chars for large chapters, full for small)
                    "key_findings": finding_data.get("key_findings", [])[:10] if finding_data and isinstance(finding_data, dict) else [],  # First 10 key findings
                    "sources_count": len(finding_data.get("sources", [])) if finding_data and isinstance(finding_data, dict) else 0,
                    "content_preview": content[:1000]  # First 1000 chars of content (for reference, not used in prompt)
                }
                
                current_metadata["chapter_summaries"].append(chapter_summary)
                
                # Update session metadata
                # CRITICAL: Ensure proper JSON serialization for JSONB field
                # SQLAlchemy handles JSONB automatically, but we need to ensure data is JSON-serializable
                from sqlalchemy import update
                from src.database.schema import ResearchSessionModel
                import json
                async with session_factory() as session:
                    # CRITICAL: Ensure all strings in metadata are properly encoded (no unicode issues)
                    # SQLAlchemy's JSONB will handle encoding, but we ensure data is clean
                    try:
                        # Test JSON serialization to catch any encoding issues early
                        json.dumps(current_metadata, ensure_ascii=False)
                    except (TypeError, ValueError) as e:
                        logger.error("Failed to serialize session_metadata to JSON",
                                   error=str(e),
                                   session_id=session_id,
                                   note="Metadata contains non-serializable data - this will cause DB errors")
                        # Clean metadata - remove non-serializable items
                        cleaned_metadata = {}
                        for key, value in current_metadata.items():
                            try:
                                json.dumps(value, ensure_ascii=False)
                                cleaned_metadata[key] = value
                            except:
                                logger.warning(f"Skipping non-serializable key in metadata: {key}")
                        current_metadata = cleaned_metadata
                    
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
        
        # CRITICAL: Log successful chapter addition with all details
        finding_topic = finding_data.get("topic", "unknown") if finding_data else "unknown"
        finding_agent_id = finding_data.get("agent_id", "unknown") if finding_data else "unknown"
        sources_count = len(sources) if isinstance(sources, list) else 0
        
        # CRITICAL: Verify task status after chapter is written
        # Task should already be 'done' (agent marked it when completing the task)
        # If task is NOT 'done', this indicates a problem - task may have been returned for rework
        # In that case, chapter should NOT have been written (supervisor should have returned early)
        agent_file_service = context.get("agent_file_service")
        pending_tasks_count = 0
        done_tasks_count = 0
        task_status_verified = False
        if agent_file_service and finding_agent_id and finding_topic:
            try:
                agent_file = await agent_file_service.read_agent_file(finding_agent_id)
                todos = agent_file.get("todos", [])
                
                # Find the task that matches the finding topic
                matching_task = None
                for todo in todos:
                    if todo.title == finding_topic:
                        matching_task = todo
                        break
                
                # CRITICAL: Verify task status - it should be 'done' if chapter was written
                # If task is 'in_progress', this means it was returned for rework, and chapter should NOT have been written
                if matching_task:
                    if matching_task.status == "done":
                        task_status_verified = True
                        logger.debug(
                            "Task status verified as 'done' after chapter was written",
                            agent_id=finding_agent_id,
                            task_title=matching_task.title,
                            note="Task is correctly in 'done' status - agent will pick next pending task"
                        )
                    elif matching_task.status == "in_progress":
                        # CRITICAL ERROR: Chapter was written but task is in_progress
                        # This should NOT happen - if task was returned for rework, chapter should NOT be written
                        logger.error(
                            "CRITICAL ERROR: Chapter was written but task is 'in_progress' - this should not happen!",
                            agent_id=finding_agent_id,
                            task_title=matching_task.title,
                            current_status=matching_task.status,
                            note="If task was returned for rework, chapter should NOT be written. This indicates a logic error in supervisor chain."
                        )
                        # Don't change status - this is an error that needs investigation
                    else:
                        # Task is in unexpected status (pending)
                        logger.warning(
                            "Task is in unexpected status after chapter was written",
                            agent_id=finding_agent_id,
                            task_title=matching_task.title,
                            current_status=matching_task.status,
                            note="Task should be 'done' after chapter is written. This may indicate a problem."
                        )
                
                # Get counts for logging
                pending_tasks_count = len([t for t in todos if t.status == "pending"])
                done_tasks_count = len([t for t in todos if t.status == "done"])
            except Exception as e:
                logger.error(
                    "Failed to verify task status after chapter was written",
                    error=str(e),
                    agent_id=finding_agent_id,
                    finding_topic=finding_topic
                )
        
        logger.info("✅ MANDATORY ACTION COMPLETED: Draft report chapter added", 
                   chapter_number=chapter_number,
                   chapter_title=chapter_title,
                   finding_topic=finding_topic,
                   finding_agent_id=finding_agent_id,
                   content_length=len(content),
                   sources_count=sources_count,
                   total_length=len(updated),
                   agent_pending_tasks=pending_tasks_count,
                   agent_done_tasks=done_tasks_count,
                   task_status_verified=task_status_verified,
                   context_used={
                       "query": bool(query),
                       "deep_search": bool(deep_search_result),
                       "clarification": bool(clarification_context),
                       "existing_chapters": len(chapter_summaries)
                   },
                   note=f"Chapter added successfully. Agent {finding_agent_id}'s task should be 'done' (agent marked it when completing). GUARANTEE: Agent will pick next pending task ({pending_tasks_count} waiting) ONLY when chapter is added (task status = 'done'). If task was returned for rework, chapter is NOT written and task becomes 'in_progress' - agent continues this task.")
        
        # Return success with context info for supervisor
        # CRITICAL: Include sources_count so supervisor knows how many sources were added
        sources_count_in_chapter = len(sources_list) if sources_list else 0
        return {
            "success": True,
            "new_length": len(updated),
            "chapter_number": chapter_number,
            "chapter_title": chapter_title,
            "sources_count": sources_count_in_chapter,  # CRITICAL: Return sources count for logging
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
    
    try:
        agent_id = args.get("agent_id")
        
        # CRITICAL: Check maximum agent count from settings
        settings = context.get("settings")
        if settings:
            max_agents = getattr(settings, "deep_research_num_agents", 3)
        else:
            # Fallback: import settings if not in context
            from src.config.settings import get_settings
            settings_obj = get_settings()
            max_agents = getattr(settings_obj, "deep_research_num_agents", 3)
        
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
        
        # CRITICAL: Validate required fields before creating task
        title = args.get("title")
        if not title or not isinstance(title, str):
            logger.error(f"Cannot create task for agent {agent_id} - title is missing or invalid",
                        title=title, title_type=type(title).__name__)
            return {
                "error": f"Task title is required and must be a non-empty string. Received: {title}",
                "agent_id": agent_id
            }
        
        objective = args.get("objective")
        if not objective or not isinstance(objective, str):
            logger.error(f"Cannot create task for agent {agent_id} - objective is missing or invalid",
                        objective=objective, objective_type=type(objective).__name__)
            return {
                "error": f"Task objective is required and must be a non-empty string. Received: {objective}",
                "agent_id": agent_id
            }
        
        # CRITICAL: Check for duplicate tasks by title before creating new one
        new_task_title = title.strip()
        if new_task_title:
            for existing_todo in current_todos:
                if existing_todo.title.strip() == new_task_title:
                    logger.warning(f"Task with title '{new_task_title}' already exists for agent {agent_id} - skipping duplicate creation",
                                 existing_status=existing_todo.status,
                                 existing_todo_id=getattr(existing_todo, "todo_id", "unknown"))
                    return {
                        "error": f"Task with title '{new_task_title}' already exists for agent {agent_id}",
                        "existing_status": existing_todo.status,
                        "suggestion": f"Use update_agent_todo to modify the existing task instead of creating a duplicate"
                    }
        
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
        
        # Create new todo (title and objective already validated above)
        new_todo = AgentTodoItem(
            reasoning=args.get("reasoning", "") or "",
            title=new_task_title,  # Use validated and stripped title
            objective=objective.strip() if objective else "",  # Use validated objective
            expected_output=args.get("expected_output", "") or "Comprehensive findings",
            sources_needed=[],
            priority=args.get("priority", "medium") or "medium",
            status="pending",
            note=args.get("guidance", "") or ""
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
                    # CRITICAL: Prevent setting status to in_progress if agent already has another in_progress task
                    elif new_status == "in_progress":
                        other_in_progress = [t for t in current_todos if t.status == "in_progress" and t.title != todo_title]
                        if other_in_progress:
                            logger.error(f"CRITICAL: Supervisor attempted to set task '{todo_title}' to in_progress, but agent {agent_id} already has {len(other_in_progress)} in_progress task(s): {[t.title for t in other_in_progress]}. Ignoring status change to prevent multiple in_progress tasks.",
                                       agent_id=agent_id,
                                       todo_title=todo_title,
                                       other_in_progress_tasks=[t.title for t in other_in_progress],
                                       note="Agent can only work on ONE task at a time. Cannot set task to in_progress when other in_progress tasks exist.")
                            # Don't update status - keep current status
                        else:
                            todo.status = new_status
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


async def return_task_to_progress_handler(args: Dict[str, Any], context: Dict[str, Any]) -> Dict:
    """
    Return a task to progress status - reject finding and ask agent to rework.
    
    CRITICAL: This is used when supervisor validates finding and determines it doesn't match the task.
    The task is returned to "in_progress" status, agent receives new guidance, and finding is NOT added to draft_report.
    Agent will continue working on the SAME task with improved instructions.
    
    CRITICAL: This should be used ONLY for SERIOUS issues - not minor problems that can be fixed in the chapter.
    """
    agent_file_service = context.get("agent_file_service")
    stream = context.get("stream")
    
    if not agent_file_service:
        return {"error": "File service not available"}
    
    try:
        agent_id = args.get("agent_id")
        todo_title = args.get("todo_title")
        motivation = args.get("motivation", "")
        instructions = args.get("instructions", "")
        
        # CRITICAL: Log detailed reasoning for rejection (for debugging and transparency)
        logger.warning(f"Supervisor rejecting finding and returning task to progress",
                      agent_id=agent_id,
                      todo_title=todo_title,
                      motivation_length=len(motivation),
                      instructions_length=len(instructions),
                      motivation_preview=motivation[:200] if motivation else "",
                      instructions_preview=instructions[:200] if instructions else "",
                      note="CRITICAL: Supervisor determined finding doesn't match task - task returned for rework. This should happen ONLY for serious issues.")
        
        # Read current agent file
        agent_file = await agent_file_service.read_agent_file(agent_id)
        current_todos = agent_file.get("todos", [])
        
        # CRITICAL: Check for duplicate tasks before processing
        # Remove duplicates by title (keep the most recent/active one)
        seen_titles = {}
        unique_todos = []
        for todo in current_todos:
            if todo.title in seen_titles:
                # Duplicate found - keep the one with more recent status or more data
                existing = seen_titles[todo.title]
                # Prefer task with "done" or "in_progress" status over "pending"
                if todo.status in ["done", "in_progress"] and existing.status == "pending":
                    unique_todos.remove(existing)
                    unique_todos.append(todo)
                    seen_titles[todo.title] = todo
                elif todo.status == existing.status:
                    # Same status - prefer one with more data (has supervisor_message, return_count, etc.)
                    if (hasattr(todo, "supervisor_message") and todo.supervisor_message) or \
                       (hasattr(todo, "return_count") and getattr(todo, "return_count", 0) > 0):
                        unique_todos.remove(existing)
                        unique_todos.append(todo)
                        seen_titles[todo.title] = todo
                    # Otherwise keep existing
                # Otherwise keep existing
            else:
                unique_todos.append(todo)
                seen_titles[todo.title] = todo
        
        # Update todos if duplicates were found
        if len(unique_todos) < len(current_todos):
            logger.warning(f"Found duplicate tasks for agent {agent_id} - removing duplicates",
                         original_count=len(current_todos),
                         unique_count=len(unique_todos),
                         duplicate_titles=[title for title, count in {t.title: sum(1 for t2 in current_todos if t2.title == t.title) for t in current_todos}.items() if count > 1])
            current_todos = unique_todos
        
        # Find the task
        task_found = False
        for todo in current_todos:
            if todo.title == todo_title:
                # CRITICAL: Can only return done tasks to progress (tasks that were just completed)
                if todo.status != "done":
                    return {
                        "error": f"Task '{todo_title}' for agent {agent_id} is not done (status: {todo.status}). Can only return done tasks to progress.",
                        "current_status": todo.status
                    }
                
                # CRITICAL: Check return_count - can only return task once (maximum 1 iteration of rework)
                if hasattr(todo, "return_count") and todo.return_count >= 1:
                    logger.warning(f"Task '{todo_title}' for agent {agent_id} has already been returned to progress once (return_count: {todo.return_count}). Cannot return again - maximum 1 iteration of rework allowed.",
                                 agent_id=agent_id,
                                 todo_title=todo_title,
                                 return_count=todo.return_count,
                                 note="Each task can be returned to progress maximum 1 time. If finding is still not acceptable after rework, supervisor should add it as chapter anyway and note issues in content.")
                    return {
                        "error": f"Task '{todo_title}' for agent {agent_id} has already been returned to progress once (return_count: {todo.return_count}). Each task can be returned maximum 1 time. If finding is still not acceptable after rework, add it as chapter anyway and note issues in content.",
                        "return_count": todo.return_count,
                        "note": "Maximum 1 iteration of rework allowed per task. After rework, supervisor should accept finding even if not perfect."
                    }
                
                # CRITICAL: Check if there are already other in_progress tasks (should be 0)
                other_in_progress = [t for t in current_todos if t.status == "in_progress" and t.title != todo_title]
                if other_in_progress:
                    return {
                        "error": f"Agent {agent_id} already has {len(other_in_progress)} in_progress task(s): {[t.title for t in other_in_progress]}. Cannot return task to progress when other tasks are in progress.",
                        "other_in_progress": [t.title for t in other_in_progress]
                    }
                
                # Return task to in_progress
                todo.status = "in_progress"
                
                # CRITICAL: Save supervisor message (motivation + instructions)
                supervisor_message = ""
                if motivation:
                    supervisor_message += f"**MOTIVATION:** {motivation}\n\n"
                if instructions:
                    supervisor_message += f"**INSTRUCTIONS:** {instructions}"
                
                if supervisor_message:
                    todo.supervisor_message = supervisor_message
                
                # Increment return_count
                if hasattr(todo, "return_count"):
                    todo.return_count += 1
                else:
                    todo.return_count = 1
                
                # CRITICAL: Grant additional steps for continuation (increase step limit)
                # Add 50% more steps (rounded up) for continuation work
                from src.workflow.research.nodes import _get_runtime_deps
                runtime_deps = _get_runtime_deps()
                settings = runtime_deps.get("settings")
                base_steps = settings.deep_research_agent_max_steps if settings else 8
                additional_steps = max(3, int(base_steps * 0.5))  # At least 3, or 50% of base
                
                if hasattr(todo, "additional_steps"):
                    todo.additional_steps += additional_steps
                else:
                    todo.additional_steps = additional_steps
                
                # Update note with supervisor feedback
                if supervisor_message:
                    todo.note = f"{todo.note or ''}\n\n**SUPERVISOR MESSAGE (REWORK REQUIRED):**\n{supervisor_message}"
                
                task_found = True
                
                # Get pending tasks count for this agent
                pending_tasks = [t for t in current_todos if t.status == "pending"]
                done_tasks = [t for t in current_todos if t.status == "done"]
                
                logger.info("⚠️ Task returned to progress - agent must continue this task",
                           agent_id=agent_id,
                           todo_title=todo_title,
                           return_count=todo.return_count,
                           additional_steps=todo.additional_steps,
                           motivation_preview=motivation[:100] if motivation else "",
                           pending_tasks_waiting=len(pending_tasks),
                           done_tasks_count=len(done_tasks),
                           note=f"CRITICAL: Task status changed from 'done' to 'in_progress'. Agent will CONTINUE this task in next cycle (not pick next pending task). {len(pending_tasks)} pending tasks will wait until this task is completed.")
                break
        
        if not task_found:
            return {"error": f"Task '{todo_title}' not found for agent {agent_id}"}
        
        # CRITICAL: Use unique todos (deduplicated) when saving
        # Write updated todos with deduplicated list
        await agent_file_service.write_agent_file(
            agent_id=agent_id,
            todos=current_todos,
            character=agent_file.get("character", ""),
            preferences=agent_file.get("preferences", "")
        )
        
        # Emit updated todos to frontend
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
            logger.info(f"Agent {agent_id} todos emitted to frontend after task returned to progress",
                       todos_count=len(todos_dict),
                       returned_task=todo_title)
        
        # CRITICAL: Remove finding from state and queue when task is returned to progress
        # Finding is rejected - it should not be processed, agent will create new finding after rework
        state = context.get("state", {})
        supervisor_queue = context.get("supervisor_queue")
        finding_topic = None
        
        # Find the finding for this task (by matching agent_id and task title)
        if state:
            findings = state.get("findings", [])
            if findings:
                # Find finding that matches this agent and task
                for finding in findings[:]:  # Copy list to iterate safely
                    if isinstance(finding, dict):
                        finding_agent_id = finding.get("agent_id", "")
                        finding_topic = finding.get("topic", "")
                        # Match by agent_id - if task was returned, its finding should be removed
                        # We can't match by task title directly, but we can match by agent_id
                        # The finding that triggered this supervisor call should be removed
                        if finding_agent_id == agent_id:
                            # Check if this finding's topic matches the task (approximate match)
                            # If task title is in finding topic or vice versa, it's likely the same
                            if todo_title.lower() in finding_topic.lower() or finding_topic.lower() in todo_title.lower():
                                findings.remove(finding)
                                state["findings"] = findings
                                logger.info("Removed finding from state after task return",
                                           agent_id=agent_id,
                                           finding_topic=finding_topic,
                                           todo_title=todo_title,
                                           note="Finding removed because task was returned to progress - agent will create new finding after rework")
                                break
        
        # CRITICAL: Remove finding from supervisor_queue if available
        # Finding was rejected - it must be removed from queue so it's not processed
        if supervisor_queue:
            try:
                queue_size = supervisor_queue.size()
                if queue_size > 0:
                    # Remove finding from queue by iterating through all events
                    # We need to find and remove the event for this agent's finding
                    removed_from_queue = False
                    temp_events = []
                    
                    # Get all events from queue
                    for _ in range(queue_size):
                        try:
                            event = supervisor_queue.queue.get_nowait()
                            temp_events.append(event)
                        except:
                            break
                    
                    # Filter out the finding for this agent and task
                    events_to_keep = []
                    for event in temp_events:
                        # Check if this event is for the returned task
                        if event.agent_id == agent_id:
                            # Check if finding topic matches task title
                            if event.result and isinstance(event.result, dict):
                                event_topic = event.result.get("topic", "")
                                # Match by topic or task_title
                                if (finding_topic and event_topic.lower() == finding_topic.lower()) or \
                                   (todo_title.lower() in event_topic.lower() or event_topic.lower() in todo_title.lower()):
                                    # This is the finding we need to remove
                                    removed_from_queue = True
                                    logger.info("Removed finding from supervisor_queue after task return",
                                               agent_id=agent_id,
                                               finding_topic=event_topic,
                                               todo_title=todo_title,
                                               note="Finding removed from queue - it was rejected and agent will create new finding after rework")
                                    continue  # Skip this event - don't put it back
                            
                            # Also check if task_title matches
                            if hasattr(event, "task_title") and event.task_title == todo_title:
                                removed_from_queue = True
                                logger.info("Removed finding from supervisor_queue after task return (matched by task_title)",
                                           agent_id=agent_id,
                                           task_title=todo_title,
                                           note="Finding removed from queue - it was rejected and agent will create new finding after rework")
                                continue  # Skip this event - don't put it back
                        
                        # Keep this event - it's not the one we need to remove
                        events_to_keep.append(event)
                    
                    # Put remaining events back in queue
                    for event in events_to_keep:
                        await supervisor_queue.queue.put(event)
                    
                    if not removed_from_queue and queue_size > 0:
                        logger.debug("Finding not found in supervisor_queue (may have already been processed)",
                                   agent_id=agent_id,
                                   finding_topic=finding_topic,
                                   todo_title=todo_title,
                                   queue_size=queue_size,
                                   note="Finding may have already been processed or wasn't in queue")
            except Exception as e:
                logger.warning("Failed to remove finding from supervisor_queue", error=str(e), exc_info=True)
        
        return {
            "success": True,
            "agent_id": agent_id,
            "todo_title": todo_title,
            "status": "in_progress",
            "message": f"Task '{todo_title}' returned to progress. Agent {agent_id} will rework with improved guidance. Finding was removed - agent will create new finding after completing the rework.",
            "finding_removed": finding_topic is not None
        }
    except Exception as e:
        logger.error("Failed to return task to progress", error=str(e))
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
            "description": "🚨🚨🚨 **ABSOLUTELY MANDATORY TOOL** - Add a new CHAPTER to the draft research report (draft_report.md) based on a finding from an agent. "
                          "**CRITICAL**: Draft report is structured by chapters - each chapter = one finding from one agent task. "
                          "**ABSOLUTELY MANDATORY**: When an agent completes a task and you receive their finding, you MUST IMMEDIATELY add it as a new chapter - NO EXCEPTIONS! "
                          "**FORBIDDEN**: Do NOT skip this step - every validated finding MUST become a chapter! "
                          "**ITERATION 2 REQUIREMENT**: In the streamlined workflow, you MUST call this tool in ITERATION 2 for EVERY finding that passes validation! "
                          "**CRITICAL**: If you skip calling this tool, the finding will be LOST and research will be INCOMPLETE! "
                          "**PRIORITY**: This is the HIGHEST PRIORITY action - call this BEFORE any other optional tools! "
                          "Write COMPREHENSIVE content based on the finding - include ALL details, facts, data, and evidence from the finding. "
                          "This file will be used to generate the final report for the user. "
                          "**CRITICAL**: Write DETAILED content based on the finding - include specific facts, dates, numbers, technical details from the finding. "
                          "DO NOT write brief summaries - write FULL, DETAILED chapter with extensive information from the finding. "
                          "Each chapter should be substantial (1000-2500 words) and cover the finding comprehensively. "
                          "**CRITICAL - NO DUPLICATION**: The 'content' parameter should EXPAND finding.summary into a comprehensive chapter, NOT just copy finding.summary. "
                          "Use finding.summary as a BASE, but add analysis, context, connections, and synthesis to create a full chapter. "
                          "**DO NOT** just copy finding.summary - expand it with your analysis and synthesis. "
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
                          "**CRITICAL SOURCES RULE - STRICTLY FORBIDDEN**: Sources are added AUTOMATICALLY at the end of each chapter with clickable links in format '- [Title](URL)'. "
                          "**ABSOLUTELY FORBIDDEN**: Do NOT write sources, references, links, '## Sources', '## References', '## Ссылки', '## Источники', or ANY source-related sections in your content - they are automatically added from finding data and will be REMOVED if you include them. "
                          "**ABSOLUTELY FORBIDDEN**: Do NOT include any source lists, reference sections, citation lists, bibliography sections, or links in the chapter content. "
                          "**ABSOLUTELY FORBIDDEN**: Do NOT add duplicate sources with the same URL - sources are automatically deduplicated. "
                          "**CRITICAL**: If you write sources in your content, they will be automatically removed - write ONLY the chapter content itself, NO sources, NO references, NO links. "
                          "Focus EXCLUSIVELY on writing the chapter content itself - sources will be added automatically at the end. "
                          "**CRITICAL: YOU SEE SUMMARY OF EACH EXISTING CHAPTER**: When you call this tool, you have access to chapter_summaries which show: "
                          "1) Chapter number and title, "
                          "2) Topic covered, "
                          "3) Summary of content (first 500 chars), "
                          "4) Key findings from that chapter. "
                          "**MANDATORY**: Use these summaries to: "
                          "* Avoid repeating information already covered in existing chapters, "
                          "* Write better integrated chapter that complements existing chapters, "
                          "* Reference related chapters when appropriate, "
                          "* Ensure smooth integration with the rest of the draft report. "
                          "**ADDITIONAL CONTEXT**: You also have access to: "
                          "1) Original user query (to understand the research goal), "
                          "2) Deep search result (initial context from deep search), "
                          "3) Clarification answers (user's additional requirements). "
                          "Use ALL this context to adapt the finding content, avoiding repetition while ensuring NO information is lost. "
                          "Write the chapter as an adapted version of the finding that fits the overall research context. "
                          "**CHAPTER TITLE**: Use chapter_title based on the finding topic. Pass the full finding data in the 'finding' parameter for chapter summary storage.",
            "args_schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Chapter content based on the finding (markdown format, comprehensive and detailed, 1000-2500 words). Use proper markdown: ### for subsections, **bold** for emphasis, *italic* for emphasis, - for lists. **CRITICAL - ABSOLUTELY FORBIDDEN**: Sources are added AUTOMATICALLY at the end of the chapter with clickable links in format '- [Title](URL)'. **ABSOLUTELY FORBIDDEN**: Do NOT write sources, references, links, '## Sources', '## References', '## Ссылки', '## Источники', or ANY source-related sections in your content - they will be automatically REMOVED if you include them. Write ONLY the chapter content itself - sources are added automatically from finding data."
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
        "return_task_to_progress": {
            "name": "return_task_to_progress",
            "description": "**CRITICAL VALIDATION TOOL**: Return a completed task to 'in_progress' status when finding doesn't match the task. "
                          "**WHEN TO USE**: If agent's finding doesn't correspond to the assigned task (wrong topic, insufficient quality, missing requirements), "
                          "call this to reject the finding and ask agent to rework. "
                          "**WHAT HAPPENS**: Task status changes from 'done' to 'in_progress', agent receives your message (motivation + instructions), "
                          "finding is NOT added to draft_report, agent continues working on the SAME task with improved instructions. "
                          "**CRITICAL LIMIT**: Each task can be returned to progress MAXIMUM 1 TIME. Check return_count before calling. "
                          "**MANDATORY**: Before adding finding to draft_report, verify it matches the task. If not, use this tool first. "
                          "**MANDATORY FIELDS**: You MUST provide both 'motivation' (why you want agent to continue) and 'instructions' (what to do).",
            "args_schema": {
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "Agent ID whose task should be returned to progress"
                    },
                    "todo_title": {
                        "type": "string",
                        "description": "Title of the completed task to return to progress"
                    },
                    "motivation": {
                        "type": "string",
                        "description": "Your motivation for asking the agent to continue work - explain why the finding needs improvement and what you want to achieve. Be specific about what's missing or wrong."
                    },
                    "instructions": {
                        "type": "string",
                        "description": "Specific instructions for the agent on what to do - what information to find, what aspects to investigate, what to improve. Be clear and actionable."
                    }
                },
                "required": ["agent_id", "todo_title", "motivation", "instructions"]
            },
            "handler": return_task_to_progress_handler
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
    supervisor_queue: Any = None,
    max_iterations: int = None  # Не используется, оставлен для совместимости
) -> Dict[str, Any]:
    """
    Обертка для backward compatibility - вызывает supervisor chain.
    
    DEPRECATED: Используйте run_supervisor_chain напрямую.
    Эта функция оставлена для совместимости со старым кодом.
    
    Args:
        state: Current research state
        llm: LLM instance
        stream: Stream generator
        supervisor_queue: Очередь файндингов
        max_iterations: Не используется (оставлен для совместимости)
        
    Returns:
        Decision dict with should_continue, replanning_needed, etc.
    """
    # Импортировать supervisor chain
    from src.workflow.research.supervisor_chain import run_supervisor_chain
    
    # Вызвать supervisor chain
    return await run_supervisor_chain(
        state=state,
        llm=llm,
        stream=stream,
        supervisor_queue=supervisor_queue
    )


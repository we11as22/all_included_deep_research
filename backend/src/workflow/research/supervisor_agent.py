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
    
    This is the supervisor's working document where the final report is assembled.
    This file will be used to generate the final report for the user.
    """
    agent_memory_service = context.get("agent_memory_service")
    
    if not agent_memory_service:
        return {"error": "Memory service not available"}
    
    try:
        content = args.get("content", "")
        section_title = args.get("section_title", "Update")
        
        # Read current draft report
        draft_file = "draft_report.md"
        try:
            current = await agent_memory_service.file_manager.read_file(draft_file)
        except FileNotFoundError:
            # Create initial draft report
            current = f"""# Research Report Draft

**Query:** {context.get('query', 'Unknown')}
**Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This is the working draft of the research report. The supervisor agent writes findings here as research progresses.

---
"""
        
        # Create structured update
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        update = f"\n\n---\n\n## {section_title} - {timestamp}\n\n{content}\n"
        
        # Append to draft
        updated = current + update
        await agent_memory_service.file_manager.write_file(draft_file, updated)
        
        logger.info("Draft report updated", section=section_title, content_length=len(content), total_length=len(updated))
        
        return {
            "success": True,
            "new_length": len(updated),
            "section": section_title
        }
    except Exception as e:
        logger.error("Failed to write draft report", error=str(e))
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
    """Create new todo for an agent."""
    agent_file_service = context.get("agent_file_service")
    
    if not agent_file_service:
        return {"error": "File service not available"}
    
    try:
        agent_id = args.get("agent_id")
        
        # Read current agent file
        agent_file = await agent_file_service.read_agent_file(agent_id)
        current_todos = agent_file.get("todos", [])
        
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
            character=agent_file.get("character", ""),
            preferences=agent_file.get("preferences", "")
        )
        
        logger.info("Created agent todo", agent_id=agent_id, title=new_todo.title)
        
        return {
            "success": True,
            "agent_id": agent_id,
            "todo_title": new_todo.title,
            "total_todos": len(current_todos)
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
                # Update fields if provided
                if "status" in args and args["status"]:
                    todo.status = args["status"]
                if "objective" in args and args["objective"]:
                    todo.objective = args["objective"]
                if "expected_output" in args and args["expected_output"]:
                    todo.expected_output = args["expected_output"]
                if "guidance" in args and args["guidance"]:
                    todo.note = args["guidance"]
                if "priority" in args and args["priority"]:
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
        
        # Get agent's notes from personal file - LIMIT to prevent context bloat
        # Only include last 10 most recent important notes
        all_notes = agent_file.get("notes", [])
        recent_notes = all_notes[-10:] if len(all_notes) > 10 else all_notes
        
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
            "recent_notes": recent_notes,  # Only recent important notes
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
                "required": []
            },
            "handler": read_main_document_handler
        },
        "write_main_document": {
            "name": "write_main_document",
            "description": "Write KEY INSIGHTS ONLY to the main research document. "
                          "CRITICAL: Only add key findings and progress updates here, NOT all items. "
                          "Items are stored in items/ directory. Main.md is for supervisor's key insights only. "
                          "Content will be added as a new section with timestamp.",
            "args_schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Key insights to add (markdown format) - only important findings, not all details"
                    },
                    "section_title": {
                        "type": "string",
                        "description": "Title for this section (e.g., 'Key Findings', 'Progress Update')"
                    }
                },
                "required": ["content", "section_title"]
            },
            "handler": write_main_document_handler
        },
        "write_draft_report": {
            "name": "write_draft_report",
            "description": "Write/append content to the draft research report (draft_report.md). "
                          "This is your working document where you assemble the final report. "
                          "Write comprehensive findings, analysis, and synthesis here. "
                          "This file will be used to generate the final report for the user. "
                          "Content will be added as a new section with timestamp.",
            "args_schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Content to add to draft report (markdown format)"
                    },
                    "section_title": {
                        "type": "string",
                        "description": "Title for this section (e.g., 'Historical Analysis', 'Technical Specifications')"
                    }
                },
                "required": ["content", "section_title"]
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
                "required": []
            },
            "handler": read_draft_report_handler
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
                "required": []
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
                          "Use this to assign new research tasks or follow-up investigations. "
                          "CRITICAL: Ensure each agent gets DIFFERENT tasks covering different aspects "
                          "(history, technical, expert views, applications, trends, comparisons, impact, challenges) "
                          "to build a complete picture. Avoid duplicate/overlapping tasks between agents.",
            "args_schema": {
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "Agent identifier (e.g., 'agent_1', 'agent_2')"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Why this task is needed"
                    },
                    "title": {
                        "type": "string",
                        "description": "Task title"
                    },
                    "objective": {
                        "type": "string",
                        "description": "What the agent should achieve"
                    },
                    "expected_output": {
                        "type": "string",
                        "description": "Expected result format"
                    },
                    "priority": {
                        "type": "string",
                        "description": "Priority: high/medium/low",
                        "enum": ["high", "medium", "low"],
                        "default": "medium"
                    },
                    "guidance": {
                        "type": "string",
                        "description": "Specific guidance on how to approach this task"
                    }
                },
                "required": ["agent_id", "title", "objective", "expected_output"]
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
                        "enum": ["pending", "in_progress", "done"]
                    },
                    "objective": {
                        "type": "string",
                        "description": "Updated objective"
                    },
                    "expected_output": {
                        "type": "string",
                        "description": "Updated expected result format"
                    },
                    "guidance": {
                        "type": "string",
                        "description": "Updated guidance on how to approach this task"
                    },
                    "priority": {
                        "type": "string",
                        "description": "Updated priority: high/medium/low",
                        "enum": ["high", "medium", "low"]
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Updated reasoning for why this task is needed"
                    }
                },
                "required": ["agent_id", "todo_title"]
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
    findings = state.get("findings", state.get("agent_findings", []))
    agent_characteristics = state.get("agent_characteristics", {})
    research_plan = state.get("research_plan", {})
    iteration = state.get("iteration", 0)
    
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
    deep_search_result_raw = state.get("deep_search_result", "")
    if isinstance(deep_search_result_raw, dict):
        deep_search_result = deep_search_result_raw.get("value", "") if isinstance(deep_search_result_raw, dict) else ""
    else:
        deep_search_result = deep_search_result_raw or ""
    
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
        stream.emit_status(f"Supervisor reviewing iteration #{iteration + 1}...", step="supervisor")
    
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
               findings_preview=findings_summary[:500] if findings_summary else "No findings")
    
    # Get clarification context if available - extract from chat_history
    clarification_context = state.get("clarification_context", "")
    if not clarification_context:
        # Extract user clarification answers from chat_history
        chat_history = state.get("chat_history", [])
        if chat_history:
            for i, msg in enumerate(chat_history):
                if msg.get("role") == "assistant" and ("clarification" in msg.get("content", "").lower() or "🔍" in msg.get("content", "")):
                    if i + 1 < len(chat_history) and chat_history[i + 1].get("role") == "user":
                        user_answer = chat_history[i + 1].get("content", "")
                        clarification_context = f"\n\n**USER CLARIFICATION ANSWERS (CRITICAL - MUST BE CONSIDERED):**\n{user_answer}\n\nThese answers refine the research scope and priorities. Use them when reviewing findings and writing the report."
                        break
        if not clarification_context:
            clarification_context = ""
    
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
    
    # Create supervisor prompt
    system_prompt = f"""You are the research supervisor coordinating a team of researcher agents.

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

**INITIAL DEEP SEARCH CONTEXT:**
{deep_search_result[:2000] if deep_search_result else "No initial deep search context available."}
{clarification_context if clarification_context else ""}

**CRITICAL CONTEXT USAGE:**
- **ALWAYS refer to the ORIGINAL USER QUERY above** - this is what the user actually asked for
- Use the initial deep search context to understand the topic and guide research direction
- **MANDATORY**: User's clarification answers (if provided) MUST be used when creating agent todos and evaluating findings
- Ensure all agent tasks and research findings directly relate to the ORIGINAL USER QUERY
- If research is going off-topic, redirect agents back to the original query

CRITICAL STRATEGY: Diversify agent tasks to build complete picture!
- Each agent should research DIFFERENT aspects/aspects of the topic
- Examples of diverse research angles:
  * Agent 1: Historical development and evolution
  * Agent 2: Technical specifications and technical details
  * Agent 3: Expert opinions, analysis, and critical perspectives
  * Agent 4: Real-world applications, case studies, and practical examples
  * Agent 5: Industry trends, current state, and future prospects
  * Agent 6: Comparative analysis with alternatives/competitors
  * Agent 7: Economic, social, or cultural impact
  * Agent 8: Challenges, limitations, and controversies
- When creating todos, ensure agents cover DIFFERENT angles - avoid overlap!
- From diverse agent findings, you will assemble a COMPLETE, comprehensive picture
- If multiple agents research the same aspect, redirect them to different angles

CRITICAL: Your agents must go DEEP, not just surface-level!
- If an agent only provides basic/general information, create a todo forcing them to dig into SPECIFIC details
- Examples of deep research: technical specifications, expert analysis, case studies, historical context, advanced features, industry trends
- Examples of shallow research: basic definitions, general overviews, simple facts
- When creating todos, explicitly instruct agents to find: technical details, expert opinions, real-world examples, advanced features, specific data

Available tools:
- read_supervisor_file: Read YOUR personal file (agents/supervisor.md) with your notes and observations
- write_supervisor_note: Write note to YOUR personal file - use this for your thoughts, observations, and notes
- read_main_document: Read current main research document (key insights only, not all items) - SHARED with all agents
- write_main_document: Add KEY INSIGHTS ONLY to main document (not all items - items stay in items/ directory) - ONLY essential shared info
- read_draft_report: Read the draft research report (draft_report.md) - your working document for final report
- write_draft_report: Write/append to draft research report (draft_report.md) - this is where you assemble the final report
- review_agent_progress: Check specific agent's progress and todos
- create_agent_todo: Assign new task to an agent (use this to force deeper research AND diversify coverage!)
- update_agent_todo: Update existing agent todo (OPTIMAL for refining tasks, changing priority, updating guidance, or modifying objectives)
- make_final_decision: Decide to continue/replan/finish

CRITICAL WORKFLOW:
1. Review agent findings - identify if they're too shallow OR if they overlap with other agents
2. **Write YOUR notes to supervisor file** - use write_supervisor_note for your personal observations and thoughts
3. **CRITICAL: Write comprehensive findings to draft_report.md** - this is your working document for the final report
   - **YOU MUST write to draft_report.md after reviewing agent findings**
   - Include: synthesis of agent findings, key discoveries, analysis, conclusions
   - This file will be used to generate the final report - it MUST be comprehensive!
   - If you don't write to draft_report.md, the final report will be empty!
4. Add only KEY INSIGHTS to main.md (not all items - items stay in items/ directory) - ONLY essential shared information
5. Check each agent's progress - ensure they cover DIFFERENT aspects
6. **CRITICAL**: 
   - If findings are basic, create todos forcing deeper research with specific instructions
   - If agents overlap, redirect them to DIFFERENT angles to build complete picture
   - Ensure comprehensive coverage: history, technical, expert views, applications, trends, comparisons, impact, challenges
   - **OPTIMAL**: Use update_agent_todo to refine existing tasks when agents need more specific instructions or when research direction changes
7. **MANDATORY: You MUST call at least ONE tool on EVERY iteration** - never return empty tool_calls!
   - If you need to review findings: call read_draft_report, read_main_document, or review_agent_progress
   - **After reviewing agent findings, you MUST call write_draft_report** to synthesize their findings
   - If you need to update documents: call write_draft_report, write_main_document, or write_supervisor_note
   - If you need to assign NEW tasks: call create_agent_todo
   - If you need to REFINE/UPDATE existing tasks: call update_agent_todo (OPTIMAL for modifying objectives, guidance, priority, or status)
   - If you're ready to finish: call make_final_decision (this is the ONLY way to finish!)
   - **CRITICAL**: Before calling make_final_decision with "finish", ensure you've written comprehensive findings to draft_report.md!
8. **Make final decision** - CRITICAL: You MUST call make_final_decision tool on EVERY review cycle!
   - This is MANDATORY - you cannot skip this tool!
   - **BEFORE deciding "finish"**: You MUST ensure draft_report.md contains comprehensive research
   - Read draft_report to check if it's complete and covers all aspects
   - If draft_report is empty or incomplete, write comprehensive findings to it FIRST, then decide "finish"
   - "finish" ONLY when: research is comprehensive AND draft_report.md is properly filled
   - "continue" if more research is needed (agents have new todos to complete)
   - "replan" if research direction needs to change
   - **YOU MUST CALL THIS TOOL** - it's the only way to finish or continue research!
9. **When finishing**: The draft_report.md will be used to generate the final report for the user - ENSURE IT'S COMPLETE!

MEMORY MANAGEMENT:
- **YOUR personal file (supervisor.md)**: Use for your notes, observations, thoughts - this is YOUR workspace
- **main.md**: ONLY essential shared information that ALL agents need to know - keep it minimal!
- **draft_report.md**: Your working document for assembling the final report
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
    
    # Extract user clarification answers from chat history if available
    clarification_context = ""
    chat_history = state.get("chat_history", [])
    if chat_history:
        # Look for user messages after clarification questions
        for i, msg in enumerate(chat_history):
            if msg.get("role") == "assistant" and ("clarification" in msg.get("content", "").lower() or "🔍" in msg.get("content", "")):
                # Check if next message is from user (user answered)
                if i + 1 < len(chat_history) and chat_history[i + 1].get("role") == "user":
                    user_answer = chat_history[i + 1].get("content", "")
                    clarification_context = f"\n\n**USER CLARIFICATION ANSWERS (CRITICAL - USE THESE TO GUIDE RESEARCH):**\n{user_answer}\n\n**MANDATORY**: Use these answers to guide research direction, create relevant agent todos, and ensure research addresses what the user actually asked for.\n"
                    logger.info("Found user clarification answers in chat history", answer_preview=user_answer[:200])
                    break
    
    # Initialize conversation
    agent_history = []
    agent_history.append({
        "role": "user",
        "content": f"""Review the latest research findings and coordinate next steps.

Current findings from agents (last 10, summarized):
{findings_summary if findings_summary else "No findings yet - agents are still researching."}

**IMPORTANT**: These findings are from your research agents. You MUST synthesize them into draft_report.md!

{notes_section}{clarification_context}

CRITICAL INSTRUCTIONS:
1. **DIVERSIFY COVERAGE**: Ensure each agent researches DIFFERENT aspects of the topic
   - Check if agents are researching overlapping areas - if so, redirect them to different angles
   - Goal: Build complete picture from diverse perspectives (history, technical, expert views, applications, trends, comparisons, impact, challenges)
   - Avoid duplicate research - each agent should contribute unique insights

2. **FORCE DEEPER RESEARCH**: If any agent provided only basic/general information, you MUST create a todo forcing them to dig deeper
   - When creating todos, specify EXACTLY what deep research is needed: technical specs, expert analysis, case studies, etc.
   - Do NOT accept surface-level findings - push agents to find specific details, data, and expert insights

3. **ASSEMBLE COMPLETE PICTURE**: From diverse agent findings, synthesize a comprehensive understanding
   - Each agent's unique angle contributes to the full picture
   - Ensure all major aspects are covered before finishing

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
   - Evaluate if findings are deep enough AND if they cover different aspects
   - Identify gaps, overlaps, or shallow research

3. **ACTIVELY write to your memory and draft (CRITICAL):**
   - **After reviewing findings, ALWAYS call write_supervisor_note** to record:
     * Your observations about the findings
     * Gaps you've identified
     * Next steps you're planning
     * Your thinking process and reasoning
   - **After reviewing findings, ALWAYS call write_draft_report** to:
     * Add new findings from agents to the draft report
     * Synthesize and analyze the findings
     * Include key discoveries, sources, and synthesis
     * Build the draft report progressively as research progresses
   - **IMPORTANT:** Don't wait until the end - write to draft_report.md continuously as agents complete tasks
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
    })
    
    # Get tools as StructuredTool objects for proper binding
    tools = SupervisorToolsRegistry.get_structured_tools({
        "state": state,
        "stream": stream,
        "agent_memory_service": agent_memory_service,
        "agent_file_service": agent_file_service,
        "supervisor_queue": supervisor_queue
    })
    logger.debug(f"Supervisor bound {len(tools)} tools", tool_names=[t.name for t in tools])
    llm_with_tools = llm.bind_tools(tools)
    
    # ReAct loop
    decision_made = False
    consecutive_empty_tool_calls = 0  # Track consecutive empty tool_calls
    
    # Track last usage of critical tools to remind supervisor if not used
    last_draft_write = -1  # Iteration when draft_report was last written
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
                    result = await SupervisorToolsRegistry.execute(
                        tool_name,
                        tool_args,
                        {
                            "agent_memory_service": agent_memory_service,
                            "agent_file_service": agent_file_service,
                            "state": state,
                            "stream": stream
                        }
                    )
                    
                    action_results.append({
                        "tool_call_id": tool_call_id,
                        "output": json.dumps(result, ensure_ascii=False)
                    })
                    
                    logger.info(f"Supervisor tool executed: {tool_name}", tool_call_id=tool_call_id)
                    
                    # Track usage of critical tools
                    if tool_name == "write_draft_report":
                        last_draft_write = react_iteration
                        logger.debug("Draft report written", iteration=react_iteration)
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
                final_decision["should_continue"] = False
                final_decision["replanning_needed"] = False
                final_decision["reasoning"] = f"Supervisor repeatedly failed to call tools ({consecutive_empty_tool_calls} times). Research completed after {react_iteration + 1} iterations."
                decision_made = True
                break
            
            # Force finish if we're near max iterations and no decision made
            iterations_without_decision = react_iteration + 1
            if (iterations_without_decision >= max_iterations - 3) and not decision_made:
                logger.warning(f"Supervisor near max iterations ({iterations_without_decision}/{max_iterations}) without decision - forcing finish")
                final_decision["should_continue"] = False
                final_decision["replanning_needed"] = False
                final_decision["reasoning"] = f"Research completed after {iterations_without_decision} supervisor iterations. All agent tasks reviewed and assigned."
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
                reminders.append("You haven't read draft_report.md recently - call read_draft_report to see current state")
            if iterations_since_memory_read >= 2:
                reminders.append("You haven't read your supervisor file recently - call read_supervisor_file to review your notes")
            if iterations_since_draft_write >= 3:
                reminders.append("You haven't written to draft_report.md in 3+ iterations - call write_draft_report to add findings")
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
            
            for result in action_results:
                agent_history.append({
                    "role": "tool",
                    "content": result["output"],
                    "tool_call_id": result["tool_call_id"]
                })
        
        except Exception as e:
            logger.error(f"Supervisor iteration {react_iteration} failed", error=str(e))
            break
    
    if not decision_made:
        logger.warning("Supervisor reached max iterations without explicit decision, stopping research", 
                     max_iterations=max_iterations, 
                     agent_history_length=len(agent_history))
        # Force finish decision
        final_decision["should_continue"] = False
        final_decision["reasoning"] = f"Supervisor reached max iterations ({max_iterations}) without making explicit decision. Forcing finish."
    
    # CRITICAL: Before finishing, ensure draft_report.md is filled with findings
    # If supervisor didn't write draft_report, we must create it from findings
    if not final_decision["should_continue"]:
        # Write final note to supervisor file about completion
        if agent_file_service:
            try:
                supervisor_file = await agent_file_service.read_agent_file("supervisor")
                existing_notes = supervisor_file.get("notes", [])
                final_note = f"Research completed. Decision: finish. Total findings: {len(findings)}. Draft report finalized."
                existing_notes.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {final_note}")
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
    
    return final_decision


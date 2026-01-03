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


class ReviewAgentProgressArgs(BaseModel):
    """Arguments for reviewing specific agent's progress."""
    agent_id: str = Field(description="Agent ID to review")


class MakeFinalDecisionArgs(BaseModel):
    """Arguments for making final research decision."""
    reasoning: str = Field(description="Analysis of current research state")
    decision: str = Field(description="Decision: continue/replan/finish")


# ==================== Supervisor Tools Handlers ====================


async def read_main_document_handler(args: Dict[str, Any], context: Dict[str, Any]) -> Dict:
    """Read main research document."""
    agent_memory_service = context.get("agent_memory_service")
    
    if not agent_memory_service:
        return {"error": "Memory service not available"}
    
    try:
        content = await agent_memory_service.read_main_file()
        max_length = args.get("max_length", 5000)
        
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
        logger.error("Failed to read main document", error=str(e))
        return {"error": str(e)}


async def write_main_document_handler(args: Dict[str, Any], context: Dict[str, Any]) -> Dict:
    """Write/update main research document."""
    agent_memory_service = context.get("agent_memory_service")
    
    if not agent_memory_service:
        return {"error": "Memory service not available"}
    
    try:
        content = args.get("content", "")
        section_title = args.get("section_title", "Update")
        
        # Read current content
        current = await agent_memory_service.read_main_file()
        
        # Create structured update
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        update = f"\n\n---\n\n## {section_title} - {timestamp}\n\n{content}\n"
        
        # Append to document
        updated = current + update
        await agent_memory_service.file_manager.write_file("main.md", updated)
        
        logger.info("Main document updated", section=section_title, content_length=len(content))
        
        return {
            "success": True,
            "new_length": len(updated),
            "section": section_title
        }
    except Exception as e:
        logger.error("Failed to write main document", error=str(e))
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
        
        # Get agent's notes
        items = await agent_memory_service.list_items() if agent_memory_service else []
        agent_notes = [item for item in items if agent_id in item.get("file_path", "")]
        
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
            "notes_count": len(agent_notes),
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
    
    logger.info("Supervisor decision", decision=decision, reasoning=reasoning[:200])
    
    return {
        "decision": decision,
        "reasoning": reasoning
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
            "description": "Write/append content to the main research document. "
                          "Use this to document findings, insights, and progress. "
                          "Content will be added as a new section with timestamp.",
            "args_schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Content to add (markdown format)"
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
        """Get tool definitions for LLM."""
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
    max_iterations: int = 10
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
    
    # Build context for supervisor
    findings_summary = "\n\n".join([
        f"**{f.get('agent_id')}** - {f.get('topic')}:\n{f.get('summary', '')}\n"
        f"Sources: {len(f.get('sources', []))}, Confidence: {f.get('confidence', 'unknown')}"
        for f in findings
    ])
    
    # Create supervisor prompt
    system_prompt = f"""You are the research supervisor coordinating a team of researcher agents.

Your role:
1. Review agent findings and update main research document
2. Identify gaps in research - ESPECIALLY superficial or basic findings
3. Create new todos for agents when needed - FORCE them to dig deeper
4. **CRITICAL**: Assign DIFFERENT tasks to different agents to cover ALL aspects of the topic
5. Decide when research is complete - only when truly comprehensive

Current query: {query}
Research plan: {research_plan.get('reasoning', '')}
Iteration: {iteration + 1}

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
- read_main_document: Read current main research document
- write_main_document: Add content to main document
- review_agent_progress: Check specific agent's progress and todos
- create_agent_todo: Assign new task to an agent (use this to force deeper research AND diversify coverage!)
- make_final_decision: Decide to continue/replan/finish

Process:
1. Review agent findings - identify if they're too shallow OR if they overlap with other agents
2. Update main document with key insights from ALL agents
3. Check each agent's progress - ensure they cover DIFFERENT aspects
4. **CRITICAL**: 
   - If findings are basic, create todos forcing deeper research with specific instructions
   - If agents overlap, redirect them to DIFFERENT angles to build complete picture
   - Ensure comprehensive coverage: history, technical, expert views, applications, trends, comparisons, impact, challenges
5. Make final decision - only finish when research is truly comprehensive AND covers all major aspects

Be thorough but efficient. Use structured reasoning. FORCE agents to dig deeper AND ensure diverse coverage!
"""
    
    # Initialize conversation
    agent_history = []
    agent_history.append({
        "role": "user",
        "content": f"""Review the latest research findings and coordinate next steps.

Current findings from agents:
{findings_summary}

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

Steps:
1. Read main document to see current progress
2. Review each agent's status - check if their findings are deep enough AND if they cover different aspects
3. Update main document with findings from ALL agents
4. **DIVERSIFY AND DEEPEN**: 
   - If agents overlap, redirect them to different angles
   - If findings are shallow, create todos with specific instructions to dig deeper
5. Make decision: continue/replan/finish (only finish when research is truly comprehensive AND covers all major aspects)
"""
    })
    
    # Get tools
    tools = SupervisorToolsRegistry.get_tool_definitions()
    llm_with_tools = llm.bind_tools(tools)
    
    # ReAct loop
    decision_made = False
    final_decision = {
        "should_continue": False,
        "replanning_needed": False,
        "gaps_identified": [],
        "iteration": iteration + 1
    }
    
    for react_iteration in range(max_iterations):
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
            response = await llm_with_tools.ainvoke(messages)
            
            # Extract tool calls - handle both dict and object formats
            tool_calls = []
            if hasattr(response, "tool_calls") and response.tool_calls:
                # Convert LangChain tool calls to dict format for consistent handling
                for tc in response.tool_calls:
                    if isinstance(tc, dict):
                        tool_calls.append(tc)
                    else:
                        # Extract from LangChain ToolCall object
                        tc_dict = {
                            "id": getattr(tc, "id", None),
                            "name": getattr(tc, "name", None),
                            "args": getattr(tc, "args", {})
                        }
                        # If id is missing, try to get it from other attributes
                        if not tc_dict["id"]:
                            # Try to get from name attribute or generate one
                            tc_dict["id"] = getattr(tc, "tool_call_id", None) or f"call_{react_iteration}_{len(tool_calls)}"
                        tool_calls.append(tc_dict)
            
            # Check for decision - handle both dict and object formats
            decision_call = None
            for tc in tool_calls:
                tool_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if tool_name == "make_final_decision":
                    decision_call = tc
                    break
            
            if decision_call or not tool_calls:
                # Extract decision
                if decision_call:
                    decision_args = decision_call.get("args", {})
                    decision_type = decision_args.get("decision", "finish")
                    
                    final_decision["should_continue"] = decision_type == "continue"
                    final_decision["replanning_needed"] = decision_type == "replan"
                    
                    logger.info("Supervisor made decision", decision=decision_type, iteration=react_iteration)
                else:
                    logger.info("Supervisor finished without explicit decision")
                
                decision_made = True
                break
            
            # Execute tools
            action_results = []
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
                    
                except Exception as e:
                    logger.error(f"Supervisor tool failed: {tool_name}", error=str(e), exc_info=True)
                    action_results.append({
                        "tool_call_id": tool_call_id,
                        "output": json.dumps({"error": str(e)})
                    })
            
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
        logger.warning("Supervisor reached max iterations without decision, stopping research")
        final_decision["should_continue"] = False
    
    return final_decision


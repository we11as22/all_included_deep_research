"""Agentic supervisor for delegating research tasks."""

from __future__ import annotations

import json
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.workflow.agentic.models import SharedResearchMemory
from src.workflow.agentic.schemas import SupervisorAction, SupervisorTasks
from src.workflow.nodes.memory_search import format_memory_context_for_prompt
from src.workflow.state import ResearchFinding
from src.utils.chat_history import format_chat_history
from src.utils.date import get_current_date
from src.memory.agent_memory_service import AgentMemoryService
from src.memory.agent_file_service import AgentFileService

logger = structlog.get_logger(__name__)


def get_supervisor_system_prompt() -> str:
    """Get supervisor system prompt with current date."""
    current_date = get_current_date()
    return f"""You are the lead research supervisor with ReAct capabilities.

Current date: {current_date}

You must respond with a single JSON object:
{{"action": "<tool_name or plan_tasks>", "args": {{...}}}}

Allowed actions:
- plan_tasks: {{"tasks": ["topic1", "topic2"], "stop": false, "reasoning": "..."}}
- create_agent: {{"agent_id": "...", "character": "...", "preferences": "...", "initial_todos": ["..."]}}
- write_to_main: {{"content": "...", "section": "Notes|Quick Reference"}}
- read_agent_file: {{"agent_id": "..."}}
- update_agent_todo: {{"agent_id": "...", "todo_title": "...", "status": "pending|in_progress|done", "note": "..."}}
- read_note: {{"file_path": "items/..."}}
- write_note: {{"title": "...", "summary": "...", "urls": ["..."], "tags": ["..."]}}
- read_main: {{}}

Rules:
- You wake up after each action by subordinate agents.
- You can read their files, update their todos, add notes to main.
- Tasks must be short, specific research directions.
- Avoid duplicating existing topics or findings.
- If coverage is sufficient, set stop to true in plan_tasks.
- Always consider the current date ({current_date}) when planning research tasks.
- You can set agent character and preferences when creating them.
"""


class AgenticSupervisor:
    """Supervisor for generating and updating research tasks."""

    def __init__(
        self,
        llm: Any,
        shared_memory: SharedResearchMemory,
        memory_context: list[Any],
        chat_history: list[dict[str, str]] | None = None,
        agent_memory_service: AgentMemoryService | None = None,
        agent_file_service: AgentFileService | None = None,
    ) -> None:
        # Apply structured output for task planning
        try:
            # Use function_calling method for better OpenAI compatibility
            self.llm_planning = llm.with_structured_output(SupervisorTasks, method="function_calling")
        except Exception:
            self.llm_planning = llm
        # Apply structured output for ReAct actions
        try:
            # Use function_calling method for better OpenAI compatibility
            self.llm_react = llm.with_structured_output(SupervisorAction, method="function_calling")
        except Exception:
            self.llm_react = llm
        self.llm = llm  # Keep original for fallback
        self.shared_memory = shared_memory
        self.memory_context = memory_context
        self.chat_history = chat_history or []
        self.agent_memory_service = agent_memory_service
        self.agent_file_service = agent_file_service

    async def initial_tasks(self, query: str, max_tasks: int = 5) -> list[str]:
        if getattr(self.llm, "_llm_type", "") == "mock-chat":
            return [query]

        prompt = self._build_prompt(
            title="Initial task planning",
            query=query,
            findings=[],
            max_tasks=max_tasks,
        )
        return await self._run_prompt(prompt, max_tasks)

    async def gap_tasks(
        self,
        query: str,
        findings: list[ResearchFinding],
        max_tasks: int = 4,
    ) -> list[str]:
        if getattr(self.llm, "_llm_type", "") == "mock-chat":
            return []

        prompt = self._build_prompt(
            title="Gap analysis",
            query=query,
            findings=findings,
            max_tasks=max_tasks,
        )
        return await self._run_prompt(prompt, max_tasks)

    def _build_prompt(
        self,
        title: str,
        query: str,
        findings: list[ResearchFinding],
        max_tasks: int,
    ) -> str:
        memory_block = format_memory_context_for_prompt(self.memory_context[:6])
        shared_notes = self.shared_memory.render_notes(limit=8)
        findings_block = _format_findings(findings)
        chat_block = format_chat_history(self.chat_history, limit=len(self.chat_history))
        current_date = get_current_date()
        
        main_content = ""
        if self.agent_memory_service:
            try:
                import asyncio
                main_content = asyncio.run(self.agent_memory_service.read_main_file())[:500]
            except Exception:
                pass

        return f"""Task: {title}
Research query: {query}
Current date: {current_date}

IMPORTANT: You must create EXACTLY {max_tasks} tasks or fewer. Do not exceed this limit.

{chat_block}

Memory context:
{memory_block}

Main file (summary):
{main_content}...

Shared notes from researchers:
{shared_notes}

Existing findings:
{findings_block}

Provide up to {max_tasks} tasks in JSON only."""

    async def _run_prompt(self, prompt: str, max_tasks: int) -> list[str]:
        try:
            # Use structured output for task planning
            response = await self.llm_planning.ainvoke(
                [SystemMessage(content=get_supervisor_system_prompt()), HumanMessage(content=prompt)]
            )
            
            # If structured output is applied, response is already SupervisorTasks
            if isinstance(response, SupervisorTasks):
                if response.stop:
                    return []
                tasks = response.tasks
            else:
                # Fallback to parsing
                content = response.content if hasattr(response, "content") else str(response)
                payload = json.loads(_extract_json(content))
                
                # Handle both old format (tasks directly) and new format (action: plan_tasks)
                if payload.get("action") == "plan_tasks":
                    args = payload.get("args", {})
                    if args.get("stop", False):
                        return []
                    tasks = args.get("tasks") or []
                else:
                    # Old format
                    if payload.get("stop", False):
                        return []
                    tasks = payload.get("tasks") or []
            
            return [task.strip() for task in tasks if task.strip()][:max_tasks]
        except Exception as exc:
            logger.warning("Supervisor task generation failed", error=str(exc))
            return []


    async def react_step(
        self,
        query: str,
        agent_id: str,
        agent_action: str,
        action_result: str,
        findings: list[ResearchFinding],
    ) -> dict[str, Any]:
        """
        ReAct step: supervisor wakes up after agent action and decides what to do.
        
        Returns:
            Dict with supervisor action and any updates
        """
        prompt = await self._build_react_prompt(
            query=query,
            agent_id=agent_id,
            agent_action=agent_action,
            action_result=action_result,
            findings=findings,
        )
        
        try:
            # Use structured output for ReAct actions
            response = await self.llm_react.ainvoke(
                [SystemMessage(content=get_supervisor_system_prompt()), HumanMessage(content=prompt)]
            )
            
            # If structured output is applied, response is already SupervisorAction
            if isinstance(response, SupervisorAction):
                action = response.action
                args = response.args
            else:
                # Fallback to parsing
                content = response.content if hasattr(response, "content") else str(response)
                payload = json.loads(_extract_json(content))
                action = payload.get("action", "plan_tasks")
                args = payload.get("args", {})
            
            # Execute supervisor action
            if action == "create_agent":
                return await self._tool_create_agent(args)
            elif action == "write_to_main":
                return await self._tool_write_to_main(args)
            elif action == "read_agent_file":
                return await self._tool_read_agent_file(args)
            elif action == "update_agent_todo":
                return await self._tool_update_agent_todo(args)
            elif action == "read_note":
                return await self._tool_read_note(args)
            elif action == "write_note":
                return await self._tool_write_note(args)
            elif action == "read_main":
                return await self._tool_read_main()
            else:
                # Default: plan_tasks
                return {"action": "plan_tasks", "tasks": args.get("tasks", [])}
        except Exception as exc:
            logger.warning("Supervisor ReAct step failed", error=str(exc))
            return {"action": "plan_tasks", "tasks": []}
    
    async def _build_react_prompt(
        self,
        query: str,
        agent_id: str,
        agent_action: str,
        action_result: str,
        findings: list[ResearchFinding],
    ) -> str:
        """Build prompt for ReAct supervisor step."""
        memory_block = format_memory_context_for_prompt(self.memory_context[:6])
        shared_notes = self.shared_memory.render_notes(limit=8)
        findings_block = _format_findings(findings)
        chat_block = format_chat_history(self.chat_history, limit=len(self.chat_history))
        current_date = get_current_date()
        
        main_content = ""
        if self.agent_memory_service:
            try:
                # Try to get main content synchronously
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        main_content = ""  # Skip if loop is running
                    else:
                        main_content = asyncio.run(self.agent_memory_service.read_main_file())
                except RuntimeError:
                    main_content = asyncio.run(self.agent_memory_service.read_main_file())
            except Exception:
                pass
        
        return f"""Supervisor ReAct Step
Research query: {query}
Current date: {current_date}

Agent {agent_id} just performed: {agent_action}
Result: {action_result[:500]}

{chat_block}

Memory context:
{memory_block}

Main file (summary):
{main_content[:500]}...

Shared notes:
{shared_notes}

Existing findings:
{findings_block}

Decide what to do next. You can:
- Create agent with character/preferences using create_agent
- Update agent's todo if needed
- Add note to main if important
- Read agent file to check progress
- Plan new tasks if needed
- Do nothing if everything is fine

Respond with JSON action."""

    async def _tool_write_to_main(self, args: dict[str, Any]) -> dict[str, Any]:
        """Write content to main.md."""
        if not self.agent_memory_service:
            return {"action": "write_to_main", "result": "Memory service not available"}
        
        content = str(args.get("content") or "")
        section = str(args.get("section") or "Notes")
        
        try:
            main_content = await self.agent_memory_service.read_main_file()
            # Add to specified section
            if f"## {section}" in main_content:
                # Append to existing section
                main_content = main_content.replace(
                    f"## {section}",
                    f"## {section}\n\n{content}\n",
                    1
                )
            else:
                # Add new section
                main_content += f"\n\n## {section}\n\n{content}\n"
            
            await self.agent_memory_service.file_manager.write_file("main.md", main_content)
            return {"action": "write_to_main", "result": f"Added to {section}"}
        except Exception as e:
            logger.warning("Failed to write to main", error=str(e))
            return {"action": "write_to_main", "result": f"Failed: {str(e)}"}

    async def _tool_read_agent_file(self, args: dict[str, Any]) -> dict[str, Any]:
        """Read agent's personal file."""
        if not self.agent_file_service:
            return {"action": "read_agent_file", "result": "Agent file service not available"}
        
        agent_id = str(args.get("agent_id") or "")
        try:
            agent_file = await self.agent_file_service.read_agent_file(agent_id)
            result = f"Agent {agent_id}:\n"
            result += f"Todos: {len(agent_file.get('todos', []))}\n"
            result += f"Notes: {len(agent_file.get('notes', []))}\n"
            return {"action": "read_agent_file", "result": result}
        except Exception as e:
            return {"action": "read_agent_file", "result": f"Failed: {str(e)}"}

    async def _tool_update_agent_todo(self, args: dict[str, Any]) -> dict[str, Any]:
        """Update agent's todo (supervisor can update any agent's todo)."""
        if not self.agent_file_service:
            return {"action": "update_agent_todo", "result": "Agent file service not available"}
        
        agent_id = str(args.get("agent_id") or "")
        todo_title = str(args.get("todo_title") or "")
        status = args.get("status")
        note = args.get("note")
        
        try:
            updated = await self.agent_file_service.update_agent_todo(agent_id, todo_title, status, note)
            return {"action": "update_agent_todo", "result": f"Updated: {updated}"}
        except Exception as e:
            return {"action": "update_agent_todo", "result": f"Failed: {str(e)}"}

    async def _tool_read_note(self, args: dict[str, Any]) -> dict[str, Any]:
        """Read a note file."""
        if not self.agent_memory_service:
            return {"action": "read_note", "result": "Memory service not available"}
        
        file_path = str(args.get("file_path") or "")
        try:
            content = await self.agent_memory_service.file_manager.read_file(file_path)
            return {"action": "read_note", "result": content[:500]}
        except Exception as e:
            return {"action": "read_note", "result": f"Failed: {str(e)}"}

    async def _tool_write_note(self, args: dict[str, Any]) -> dict[str, Any]:
        """Write a note (supervisor can write notes)."""
        if not self.agent_memory_service:
            return {"action": "write_note", "result": "Memory service not available"}
        
        from src.workflow.agentic.models import AgentNote
        note = AgentNote(
            title=str(args.get("title") or "Supervisor Note"),
            summary=str(args.get("summary") or ""),
            urls=args.get("urls") or [],
            tags=args.get("tags") or ["supervisor"],
        )
        
        try:
            file_path = await self.agent_memory_service.save_agent_note(note, "supervisor")
            return {"action": "write_note", "result": f"Saved: {file_path}"}
        except Exception as e:
            return {"action": "write_note", "result": f"Failed: {str(e)}"}

    async def _tool_read_main(self) -> dict[str, Any]:
        """Read main.md file."""
        if not self.agent_memory_service:
            return {"action": "read_main", "result": "Memory service not available"}
        
        try:
            content = await self.agent_memory_service.read_main_file()
            return {"action": "read_main", "result": content[:1000]}
        except Exception as e:
            return {"action": "read_main", "result": f"Failed: {str(e)}"}
    
    async def write_project_status(
        self,
        query: str,
        findings: list[ResearchFinding],
        active_agents: dict[str, dict[str, Any]],
    ) -> None:
        """
        Write project status to main.md after agent tasks completion.
        
        Args:
            query: Research query
            findings: Current research findings
            active_agents: Dict of agent_id -> agent status (todos, notes, etc.)
        """
        if not self.agent_memory_service:
            return
        
        try:
            from datetime import datetime, timezone
            
            status_lines = [
                f"## Project Status - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
                "",
                f"**Research Query:** {query}",
                "",
                f"**Findings Count:** {len(findings)}",
                "",
                "### Active Agents Status:",
                ""
            ]
            
            for agent_id, agent_status in active_agents.items():
                todos = agent_status.get("todos", [])
                done_count = sum(1 for t in todos if t.get("status") == "done")
                in_progress_count = sum(1 for t in todos if t.get("status") == "in_progress")
                pending_count = sum(1 for t in todos if t.get("status") == "pending")
                
                status_lines.append(f"#### {agent_id}")
                status_lines.append(f"- **Todos:** {done_count} done, {in_progress_count} in progress, {pending_count} pending")
                status_lines.append(f"- **Notes:** {len(agent_status.get('notes', []))}")
                status_lines.append("")
            
            status_lines.append("### Recent Findings:")
            status_lines.append("")
            for finding in findings[-5:]:  # Last 5 findings
                status_lines.append(f"- **{finding.topic}**: {finding.summary[:150]}...")
                status_lines.append("")
            
            status_content = "\n".join(status_lines)
            
            # Read main file
            main_content = await self.agent_memory_service.read_main_file()
            
            # Find or create "## Project Status" section
            if "## Project Status" in main_content:
                # Replace existing status section
                import re
                pattern = r"## Project Status.*?(?=\n## |\Z)"
                main_content = re.sub(pattern, status_content, main_content, flags=re.DOTALL)
            else:
                # Add status section after Overview
                if "## Overview" in main_content:
                    overview_end = main_content.find("\n## ", main_content.find("## Overview") + len("## Overview"))
                    if overview_end != -1:
                        main_content = main_content[:overview_end] + "\n\n" + status_content + "\n" + main_content[overview_end:]
                    else:
                        main_content += "\n\n" + status_content
                else:
                    main_content += "\n\n" + status_content
            
            # Write updated main file
            await self.agent_memory_service.file_manager.write_file("main.md", main_content)
            logger.info("Project status written to main.md", findings_count=len(findings), agents_count=len(active_agents))
        except Exception as e:
            logger.warning("Failed to write project status", error=str(e))

    async def _tool_create_agent(self, args: dict[str, Any]) -> dict[str, Any]:
        """Create agent with character and preferences."""
        if not self.agent_file_service:
            return {"action": "create_agent", "result": "Agent file service not available"}
        
        agent_id = str(args.get("agent_id") or "")
        character = str(args.get("character") or "")
        preferences = str(args.get("preferences") or "")
        initial_todos = args.get("initial_todos") or []
        
        if not agent_id:
            return {"action": "create_agent", "result": "No agent_id provided"}
        
        try:
            from src.workflow.agentic.models import AgentTodoItem
            todos = [AgentTodoItem(title=todo, status="pending") for todo in initial_todos] if initial_todos else []
            
            await self.agent_file_service.write_agent_file(
                agent_id=agent_id,
                todos=todos,
                character=character,
                preferences=preferences,
            )
            return {"action": "create_agent", "result": f"Agent {agent_id} created with character and preferences"}
        except Exception as e:
            return {"action": "create_agent", "result": f"Failed: {str(e)}"}


def _format_findings(findings: list[ResearchFinding]) -> str:
    if not findings:
        return "None."
    lines = []
    for finding in findings[-6:]:
        lines.append(f"- {finding.topic}: {finding.summary[:200]}")
    return "\n".join(lines)

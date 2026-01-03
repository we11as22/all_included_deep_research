"""Lead agent for agentic deep research (supervisor + coordinator)."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.search.base import SearchProvider
from src.search.scraper import WebScraper
from src.workflow.agentic.models import SharedResearchMemory
from src.workflow.agentic.researcher import AgenticResearcher
from src.workflow.agentic.schemas import SupervisorActions, SupervisorTasks, TodoUpdateSchema
from src.workflow.nodes.memory_search import format_memory_context_for_prompt
from src.workflow.state import ResearchFinding
from src.utils.chat_history import format_chat_history
from src.utils.date import get_current_date
from src.memory.agent_memory_service import AgentMemoryService
from src.memory.agent_file_service import AgentFileService
from src.utils.text import summarize_text

logger = structlog.get_logger(__name__)


def get_supervisor_system_prompt() -> str:
    """Get supervisor system prompt with current date."""
    current_date = get_current_date()
    return f"""You are the lead research supervisor with ReAct capabilities.

Current date: {current_date}

You must respond with a single JSON object:
{{"reasoning": "...", "actions": [{{"reasoning": "...", "action": "<tool_name or plan_tasks>", "args": {{...}}}}]}}

You can return multiple actions in order. Return an empty actions list when you are done and want to sleep.

Allowed actions (reasoning is always included per action, only args shown here):
- plan_tasks: args {{ "tasks": ["topic1", "topic2"], "stop": false }}
- create_agent: args {{ "agent_id": "...", "character": "...", "preferences": "...", "initial_todos": ["..."] }}
- write_to_main: args {{ "content": "...", "section": "Notes|Quick Reference" }}
- read_agent_file: args {{ "agent_id": "..." }}
- update_agent_todo: args {{ "agent_id": "...", "todo_title": "...", "reasoning": "...", "status": "pending|in_progress|done", "note": "...", "objective": "...", "expected_output": "...", "sources_needed": ["..."], "priority": "medium", "url": "..." }}
- update_agent_todos: args {{ "updates": [{{"agent_id": "...", "todo_title": "...", "reasoning": "...", "status": "pending|in_progress|done", "note": "...", "objective": "...", "expected_output": "...", "sources_needed": ["..."], "priority": "medium", "url": "..."}}] }}
- read_note: args {{ "file_path": "items/..." }}
- write_note: args {{ "title": "...", "summary": "...", "urls": ["..."], "tags": ["..."] }}
- read_main: args {{}}

CRITICAL RULES FOR DEEP RESEARCH SUPERVISION:
1. **Active Monitoring**: You wake up after each action by subordinate agents. You MUST actively monitor their progress and depth of research.

2. **Depth Assessment**: After each agent action, evaluate:
   - Are agents digging deep enough? (checking multiple sources, exploring sub-topics, verifying claims)
   - Are they updating their todos actively? (if not, update their todos for them)
   - Are they stopping too early? (if yes, add more specific tasks to push them deeper)
   - Are there gaps in coverage? (if yes, create new tasks to fill gaps)

3. **Proactive Task Generation**: For REAL deep research, you MUST:
   - Generate additional tasks when agents are not digging deep enough
   - Break down broad topics into specific sub-topics that need investigation
   - Add tasks for verification, cross-referencing, and exploring related areas
   - Create tasks for exploring different perspectives, controversies, and expert opinions
   - Add tasks for finding primary sources, not just secondary summaries
   - Generate follow-up tasks based on what agents discover (don't wait for them to ask)

4. **Todo Management**: Actively manage agent todos:
   - Read agent files regularly to check their todo progress
   - If an agent has many pending todos but isn't updating them, use update_agent_todo to push them
   - If an agent is stuck on surface-level research, add specific deep-dive todos
   - If an agent finishes too quickly, add more detailed investigation tasks
   - ACTIVELY MODIFY agent plans: if you see an agent needs to dig deeper, use update_agent_todo to add new specific tasks
   - BREAK DOWN broad todos into specific sub-tasks when agents are not being detailed enough
   - ADD verification tasks when agents make claims without sources
   - ADD cross-reference tasks when information seems incomplete
   - STRICT FORMAT: Todo updates must include reasoning, title, and any updated fields (objective, expected_output, sources_needed, priority, status, note, url)

5. **Gap Analysis**: Continuously identify research gaps:
   - What angles haven't been explored yet?
   - What sources haven't been checked?
   - What sub-topics need more investigation?
   - What claims need verification?
   - What related areas should be explored?

6. **Task Quality**: Tasks must be:
   - Short and specific research directions (<= 140 chars in title)
   - Focused on a single research action
   - Include objective + expected_output when adding new todos
   - Avoid boilerplate like "Depth:" or "Primary query:"
   - Avoid duplicating existing topics or findings (but allow different angles on same topic)

7. **Stop Condition**: Only set stop to true when:
   - All major angles have been thoroughly explored
   - Information has been verified from multiple independent sources
   - Sub-topics and related areas have been investigated
   - Primary sources have been consulted
   - Gaps have been identified and addressed
   - Research is truly comprehensive, not just "good enough"

8. **Always consider the current date ({current_date}) when planning research tasks.

9. **You can set agent character and preferences when creating them.

10. **Active Intervention**: Don't be passive - if agents aren't digging deep, actively intervene:
    - Add specific deep-dive tasks
    - Update their todos to push them deeper
    - Create new agents for specialized sub-topics if needed
    - Write notes to main.md about important findings or gaps

11. **Language Match**: Use the same language as the user's query for tasks and directives.
"""


class AgenticResearchCoordinator:
    """Run supervisor + agentic researchers with shared memory."""

    def __init__(
        self,
        llm: Any,
        search_provider: SearchProvider,
        web_scraper: WebScraper,
        memory_context: list[Any],
        chat_history: list[dict[str, str]] | None = None,
        stream: Any | None = None,
        max_rounds: int = 3,
        max_concurrent: int = 4,
        max_sources: int = 10,
        agent_memory_service: AgentMemoryService | None = None,
        agent_file_service: AgentFileService | None = None,
        emit_plan: bool = True,
    ) -> None:
        self.llm = llm
        self.llm_planning = llm.with_structured_output(SupervisorTasks, method="function_calling")
        self.llm_react = llm.with_structured_output(SupervisorActions, method="function_calling")
        self.search_provider = search_provider
        self.web_scraper = web_scraper
        self.memory_context = memory_context
        self.chat_history = chat_history or []
        self.stream = stream
        self.max_rounds = max_rounds
        self.max_concurrent = max_concurrent
        self.max_sources = max_sources
        self.agent_memory_service = agent_memory_service
        self.agent_file_service = agent_file_service
        self.emit_plan = emit_plan
        self.shared_memory = SharedResearchMemory()
        self.query = ""

    async def initial_tasks(self, query: str, max_tasks: int = 5) -> list[str]:
        if getattr(self.llm, "_llm_type", "") == "mock-chat":
            return [query]

        prompt = await self._build_prompt(
            title="Initial task planning",
            query=query,
            findings=[],
            max_tasks=max_tasks,
        )
        tasks = await self._run_prompt(prompt, max_tasks)
        tasks = _normalize_tasks(tasks, query)
        tasks = _align_tasks_with_query(tasks, query)
        return _dedupe_tasks(tasks)

    async def gap_tasks(
        self,
        query: str,
        findings: list[ResearchFinding],
        max_tasks: int = 4,
    ) -> list[str]:
        if getattr(self.llm, "_llm_type", "") == "mock-chat":
            return []

        prompt = await self._build_prompt(
            title="Gap analysis",
            query=query,
            findings=findings,
            max_tasks=max_tasks,
        )
        tasks = await self._run_prompt(prompt, max_tasks)
        tasks = _normalize_tasks(tasks, query)
        tasks = _align_tasks_with_query(tasks, query)
        return _dedupe_tasks(tasks)

    async def _build_prompt(
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
        language_hint = _language_hint(query)

        main_content = ""
        if self.agent_memory_service:
            try:
                main_content = await self.agent_memory_service.read_main_file()
                main_content = summarize_text(main_content, 12000)
            except Exception:
                main_content = ""

        return f"""Task: {title}
Research query: {query}
Current date: {current_date}
Language requirement: {language_hint} (use ONLY this language in tasks)

CRITICAL: You MUST create tasks that are DIRECTLY RELATED to the research query: "{query}"
- DO NOT create tasks about unrelated topics (e.g., if query is about "spirits/alcohols", don't create tasks about "multi-agent systems" or "tanks")
- Tasks must be SPECIFIC subtopics of the main query
- Each task should be a focused investigation angle related to: {query}
- Use the SAME language as the research query for all tasks
- Do NOT prefix tasks with the full query; embed the subject naturally in the task text

IMPORTANT: You must create EXACTLY {max_tasks} tasks or fewer. Do not exceed this limit.

CRITICAL: For DEEP RESEARCH, create SPECIFIC tasks that push agents to:
- Find primary sources and verify claims
- Cover missing angles without repeating what is already done
- Move from discovery -> evidence -> synthesis

TASK FORMAT (one line per task, max 140 chars):
- Start with a verb
- Include concrete deliverable (e.g., "find 2-3 primary sources + 3 cited facts")
- Mention source type (e.g., archives, official docs, technical manuals)
- Avoid boilerplate like "Depth:" or "Primary query:"

{chat_block}

Memory context:
{memory_block}

Main file (summary):
{main_content}

Shared notes from researchers:
{shared_notes}

Existing findings:
{findings_block}

ANALYSIS REQUIRED:
1. What angles haven't been explored yet?
2. What sources need verification?
3. What sub-topics need deeper investigation?
4. What gaps exist in current research?
5. What related areas should be explored?

Generate tasks that address these gaps and push for REAL deep research, not surface-level coverage.

Provide up to {max_tasks} tasks in JSON only with fields: reasoning, tasks, stop."""

    async def _run_prompt(self, prompt: str, max_tasks: int) -> list[str]:
        try:
            response = await self.llm_planning.ainvoke(
                [SystemMessage(content=get_supervisor_system_prompt()), HumanMessage(content=prompt)]
            )

            if not isinstance(response, SupervisorTasks):
                raise ValueError("Supervisor tasks response was not structured")

            if response.stop:
                return []

            return [task.strip() for task in response.tasks if task.strip()][:max_tasks]
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
        """ReAct step: supervisor wakes up after agent action and decides what to do."""
        prompt = await self._build_react_prompt(
            query=query,
            agent_id=agent_id,
            agent_action=agent_action,
            action_result=action_result,
            findings=findings,
        )

        try:
            response = await self.llm_react.ainvoke(
                [SystemMessage(content=get_supervisor_system_prompt()), HumanMessage(content=prompt)]
            )

            if not isinstance(response, SupervisorActions):
                raise ValueError("Supervisor action response was not structured")

            actions = response.actions or []
            if not actions:
                return {"actions": [], "tasks": [], "directives": []}

            results: list[dict[str, Any]] = []
            directives: list[dict[str, Any]] = []
            tasks: list[str] = []
            action_names: list[str] = []

            for action_item in actions:
                action = action_item.action
                args = action_item.args or {}
                action_names.append(action)
                result = await self._execute_supervisor_action(action, args)
                results.append(result)
                directives.extend(result.get("directives") or [])
                tasks.extend(result.get("tasks") or [])

            tasks = _normalize_tasks(tasks, query)
            tasks = _align_tasks_with_query(_dedupe_tasks(tasks), query)
            tasks = tasks[:2]

            return {
                "actions": action_names,
                "directives": directives,
                "tasks": tasks,
                "results": results,
            }
        except Exception as exc:
            logger.warning("Supervisor ReAct step failed", error=str(exc))
            return {"actions": [], "tasks": [], "directives": []}

    async def _execute_supervisor_action(self, action: str, args: dict[str, Any]) -> dict[str, Any]:
        if action == "create_agent":
            return await self._tool_create_agent(args)
        if action == "write_to_main":
            return await self._tool_write_to_main(args)
        if action == "read_agent_file":
            return await self._tool_read_agent_file(args)
        if action == "update_agent_todo":
            return await self._tool_update_agent_todo(args)
        if action == "update_agent_todos":
            return await self._tool_update_agent_todos(args)
        if action == "read_note":
            return await self._tool_read_note(args)
        if action == "write_note":
            return await self._tool_write_note(args)
        if action == "read_main":
            return await self._tool_read_main()
        if action == "plan_tasks":
            raw_tasks = args.get("tasks") or []
            tasks = [str(task).strip() for task in raw_tasks if str(task).strip()]
            return {"action": "plan_tasks", "tasks": tasks}
        return {"action": action, "result": "Unknown action"}

    async def _build_react_prompt(
        self,
        query: str,
        agent_id: str,
        agent_action: str,
        action_result: str,
        findings: list[ResearchFinding],
    ) -> str:
        memory_block = format_memory_context_for_prompt(self.memory_context[:6])
        shared_notes = self.shared_memory.render_notes(limit=8)
        findings_block = _format_findings(findings)
        chat_block = format_chat_history(self.chat_history, limit=len(self.chat_history))
        current_date = get_current_date()
        language_hint = _language_hint(query)

        main_content = ""
        if self.agent_memory_service:
            try:
                main_content = await self.agent_memory_service.read_main_file()
                main_content = summarize_text(main_content, 12000)
            except Exception:
                main_content = ""

        agent_file_content = ""
        if self.agent_file_service:
            try:
                agent_file = await self.agent_file_service.read_agent_file(agent_id)
                agent_file_content = _format_agent_file_snapshot(agent_file)
            except Exception:
                agent_file_content = ""

        return f"""Supervisor ReAct Step
Research query: {query}
Current date: {current_date}
Language requirement: {language_hint} (use ONLY this language in directives/tasks)

Agent {agent_id} just performed: {agent_action}
Result: {summarize_text(str(action_result), 12000)}

{chat_block}

Memory context:
{memory_block}

Main file (summary):
{main_content}

Agent {agent_id} current file (todos and progress):
{agent_file_content}

Shared notes:
{shared_notes}

Existing findings:
{findings_block}

CRITICAL EVALUATION REQUIRED:
1. Is the agent digging deep enough? (checking multiple sources, exploring sub-topics, verifying claims)
2. Is the agent updating todos actively? (if not, you MUST update their todos)
3. Is the agent stopping too early? (if yes, add more specific tasks)
4. Are there gaps in coverage? (if yes, create new tasks)
5. What additional angles need exploration?
6. What verification is needed?
7. What sub-topics should be investigated?

YOU MUST BE PROACTIVE AND ACTIVELY MANAGE PLANS:
- If agent isn't updating todos, use update_agent_todo to push them (MANDATORY)
- If agent is doing surface-level research, use update_agent_todo to add specific deep-dive tasks
- If agent finishes too quickly, use update_agent_todo to add more detailed investigation tasks
- If there are gaps, use update_agent_todo to add tasks that fill them
- BREAK DOWN broad todos into specific sub-tasks when needed
- ADD verification tasks when agents make claims without sources
- ADD cross-reference tasks when information seems incomplete
- Don't wait for agents to ask - actively guide them to deeper research
- MODIFY agent plans in real-time based on what you see in their file
- Use the SAME language as the research query for directives and tasks

Decide what to do next. You can:
- update_agent_todo: Add/modify agent's todos to push deeper research (USE THIS OFTEN)
- create_agent: Create new agent with character/preferences
- write_to_main: Add note to main if important
- read_agent_file: Check progress (already shown above, but you can read again)
- plan_tasks: Plan new high-level tasks (but prefer update_agent_todo for specific agent guidance)
- Do nothing ONLY if research is truly deep and comprehensive AND agent is actively updating todos

LIMITS:
- Add at most 2 new todo items per wake-up (keep them short and actionable)

You may return multiple actions in order. Respond with JSON containing reasoning and an actions list."""

    async def _tool_write_to_main(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self.agent_memory_service:
            return {"action": "write_to_main", "result": "Memory service not available"}

        content = str(args.get("content") or "")
        section = str(args.get("section") or "Notes")

        try:
            main_content = await self.agent_memory_service.read_main_file()
            if f"## {section}" in main_content:
                main_content = main_content.replace(
                    f"## {section}",
                    f"## {section}\n\n{content}\n",
                    1,
                )
            else:
                main_content += f"\n\n## {section}\n\n{content}\n"

            await self.agent_memory_service.file_manager.write_file("main.md", main_content)
            return {"action": "write_to_main", "result": f"Added to {section}"}
        except Exception as e:
            logger.warning("Failed to write to main", error=str(e))
            return {"action": "write_to_main", "result": f"Failed: {str(e)}"}

    async def _tool_read_agent_file(self, args: dict[str, Any]) -> dict[str, Any]:
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
        if not self.agent_file_service:
            return {"action": "update_agent_todo", "result": "Agent file service not available"}

        agent_id = str(args.get("agent_id") or "")
        todo_title = str(args.get("todo_title") or args.get("title") or "")
        try:
            update = TodoUpdateSchema.model_validate(
                {
                    "reasoning": args.get("reasoning") or "Supervisor update",
                    "title": todo_title,
                    "status": args.get("status"),
                    "note": args.get("note"),
                    "objective": args.get("objective"),
                    "expected_output": args.get("expected_output"),
                    "sources_needed": args.get("sources_needed"),
                    "priority": args.get("priority"),
                    "url": args.get("url"),
                }
            )
        except Exception as exc:
            return {"action": "update_agent_todo", "result": f"Invalid update schema: {str(exc)}"}

        try:
            updated = await self.agent_file_service.update_agent_todo(
                agent_id,
                update.title,
                update.status,
                update.note,
                update.reasoning,
                update.objective,
                update.expected_output,
                update.sources_needed,
                update.priority,
                update.url,
            )
            return {
                "action": "update_agent_todo",
                "result": f"Updated: {updated}",
                "directives": [
                    {
                        "agent_id": agent_id,
                        "reasoning": update.reasoning,
                        "title": update.title,
                        "status": update.status,
                        "note": update.note,
                        "objective": update.objective,
                        "expected_output": update.expected_output,
                        "sources_needed": update.sources_needed,
                        "priority": update.priority,
                        "url": update.url,
                    }
                ],
            }
        except Exception as e:
            return {"action": "update_agent_todo", "result": f"Failed: {str(e)}"}

    async def _tool_update_agent_todos(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self.agent_file_service:
            return {"action": "update_agent_todos", "result": "Agent file service not available"}

        updates = args.get("updates") or []
        if isinstance(updates, dict):
            updates = [updates]

        applied: list[dict[str, Any]] = []
        for item in updates:
            if not isinstance(item, dict):
                continue
            agent_id = str(item.get("agent_id") or "")
            title = str(item.get("title") or item.get("todo_title") or "").strip()
            try:
                update = TodoUpdateSchema.model_validate(
                    {
                        "reasoning": item.get("reasoning") or "Supervisor update",
                        "title": title,
                        "status": item.get("status"),
                        "note": item.get("note"),
                        "objective": item.get("objective"),
                        "expected_output": item.get("expected_output"),
                        "sources_needed": item.get("sources_needed"),
                        "priority": item.get("priority"),
                        "url": item.get("url"),
                    }
                )
            except Exception:
                continue

            try:
                await self.agent_file_service.update_agent_todo(
                    agent_id,
                    update.title,
                    update.status,
                    update.note,
                    update.reasoning,
                    update.objective,
                    update.expected_output,
                    update.sources_needed,
                    update.priority,
                    update.url,
                )
                applied.append(
                    {
                        "agent_id": agent_id,
                        "reasoning": update.reasoning,
                        "title": update.title,
                        "status": update.status,
                        "note": update.note,
                        "objective": update.objective,
                        "expected_output": update.expected_output,
                        "sources_needed": update.sources_needed,
                        "priority": update.priority,
                        "url": update.url,
                    }
                )
            except Exception:
                continue

        return {
            "action": "update_agent_todos",
            "result": f"Updated: {len(applied)}",
            "directives": applied,
        }

    async def _tool_read_note(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self.agent_memory_service:
            return {"action": "read_note", "result": "Memory service not available"}

        file_path = str(args.get("file_path") or "")
        try:
            content = await self.agent_memory_service.file_manager.read_file(file_path)
            return {"action": "read_note", "result": summarize_text(content, 12000)}
        except Exception as e:
            return {"action": "read_note", "result": f"Failed: {str(e)}"}

    async def _tool_write_note(self, args: dict[str, Any]) -> dict[str, Any]:
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
        if not self.agent_memory_service:
            return {"action": "read_main", "result": "Memory service not available"}

        try:
            content = await self.agent_memory_service.read_main_file()
            return {"action": "read_main", "result": summarize_text(content, 12000)}
        except Exception as e:
            return {"action": "read_main", "result": f"Failed: {str(e)}"}

    async def write_project_status(
        self,
        query: str,
        findings: list[ResearchFinding],
        active_agents: dict[str, dict[str, Any]],
    ) -> None:
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
                "",
            ]

            for agent_id, agent_status in active_agents.items():
                todos = agent_status.get("todos", [])
                done_count = sum(1 for t in todos if t.get("status") == "done")
                in_progress_count = sum(1 for t in todos if t.get("status") == "in_progress")
                pending_count = sum(1 for t in todos if t.get("status") == "pending")

                status_lines.append(f"#### {agent_id}")
                status_lines.append(
                    f"- **Todos:** {done_count} done, {in_progress_count} in progress, {pending_count} pending"
                )
                status_lines.append(f"- **Notes:** {len(agent_status.get('notes', []))}")
                status_lines.append("")

            status_lines.append("### Recent Findings:")
            status_lines.append("")
            for finding in findings[-5:]:
                status_lines.append(f"- **{finding.topic}**: {summarize_text(finding.summary, 8000)}")
                status_lines.append("")

            status_content = "\n".join(status_lines)

            main_content = await self.agent_memory_service.read_main_file()

            if "## Project Status" in main_content:
                import re

                pattern = r"## Project Status.*?(?=\n## |\Z)"
                main_content = re.sub(pattern, status_content, main_content, flags=re.DOTALL)
            else:
                if "## Overview" in main_content:
                    overview_end = main_content.find("\n## ", main_content.find("## Overview") + len("## Overview"))
                    if overview_end != -1:
                        main_content = main_content[:overview_end] + "\n\n" + status_content + "\n" + main_content[overview_end:]
                    else:
                        main_content += "\n\n" + status_content
                else:
                    main_content += "\n\n" + status_content

            await self.agent_memory_service.file_manager.write_file("main.md", main_content)
            logger.info("Project status written to main.md", findings_count=len(findings), agents_count=len(active_agents))
        except Exception as e:
            logger.warning("Failed to write project status", error=str(e))

    async def _tool_create_agent(self, args: dict[str, Any]) -> dict[str, Any]:
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

            todos = []
            for todo in initial_todos:
                title = str(todo).strip()
                if not title:
                    continue
                todos.append(
                    AgentTodoItem(
                        reasoning="Initial supervisor seed task",
                        title=title,
                        objective="Investigate the assigned topic",
                        expected_output="Summary with sources and key findings",
                        sources_needed=[],
                        priority="medium",
                        status="pending",
                    )
                )

            await self.agent_file_service.write_agent_file(
                agent_id=agent_id,
                todos=todos,
                character=character,
                preferences=preferences,
            )
            return {"action": "create_agent", "result": f"Agent {agent_id} created with character and preferences"}
        except Exception as e:
            return {"action": "create_agent", "result": f"Failed: {str(e)}"}

    async def run(self, query: str, seed_tasks: list[str] | None = None) -> list[ResearchFinding]:
        if not self.search_provider or not self.web_scraper:
            raise ValueError("search_provider and web_scraper are required for run()")
        self.query = query
        self._reset_session()
        findings: list[ResearchFinding] = []

        if self.agent_memory_service:
            try:
                await self.agent_memory_service.read_main_file()
            except Exception as exc:
                logger.warning("Failed to initialize main agent memory file", error=str(exc))

        if seed_tasks:
            tasks = _normalize_tasks(seed_tasks, query)
            tasks = _align_tasks_with_query(tasks, query)
            tasks = tasks[: self.max_concurrent]
        else:
            tasks = await self.initial_tasks(query, max_tasks=self.max_concurrent)

        if self.stream and tasks and self.emit_plan:
            plan_text = "Research Plan:\n\n"
            plan_text += "\n".join([f"{i+1}. {task}" for i, task in enumerate(tasks)])
            plan_text += f"\n\nTotal tasks: {len(tasks)}"
            self.stream.emit_research_plan(plan_text, tasks)

        try:
            round_id = 0
            while tasks and round_id < self.max_rounds:
                if self.stream:
                    self.stream.emit_status(f"Supervisor assigned {len(tasks)} tasks", step="supervisor")
                    self.stream.emit_search_queries(tasks, label="supervisor_tasks")

                round_findings, agent_statuses = await self._run_round(tasks, round_id, existing_findings=findings)
                findings.extend(round_findings)

                if self.agent_memory_service:
                    await self.write_project_status(
                        query=query,
                        findings=findings,
                        active_agents=agent_statuses,
                    )

                round_id += 1
                if round_id >= self.max_rounds:
                    break

                gap_tasks = await self.gap_tasks(query, findings, max_tasks=self.max_concurrent)
                tasks = gap_tasks[: self.max_concurrent] if gap_tasks else []

            return findings
        finally:
            self.shared_memory.clear()

    async def _run_round(
        self,
        tasks: list[str],
        round_id: int,
        existing_findings: list[ResearchFinding],
    ) -> tuple[list[ResearchFinding], dict[str, dict[str, Any]]]:
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def run_task(idx: int, topic: str) -> tuple[ResearchFinding | None, str, dict[str, Any]]:
            agent_id = f"agent_r{round_id}_{idx}"
            async with semaphore:
                researcher = AgenticResearcher(
                    llm=self.llm,
                    search_provider=self.search_provider,
                    web_scraper=self.web_scraper,
                    shared_memory=self.shared_memory,
                    memory_context=self.memory_context,
                    chat_history=self.chat_history,
                    stream=self.stream,
                    max_steps=6,
                    max_sources=self.max_sources,
                    agent_memory_service=self.agent_memory_service,
                    agent_file_service=self.agent_file_service,
                    supervisor=self,
                )
                assignment = self._build_assignment(topic)

                character, preferences = await self._get_agent_character_preferences(agent_id, topic)

                result = await researcher.run(
                    agent_id=agent_id,
                    topic=topic,
                    existing_findings=existing_findings,
                    assignment=assignment,
                    character=character,
                    preferences=preferences,
                )
                self._share_finding(result.finding)

                agent_status = {
                    "todos": [
                        {
                            "reasoning": todo.reasoning,
                            "title": todo.title,
                            "objective": todo.objective,
                            "expected_output": todo.expected_output,
                            "sources_needed": todo.sources_needed,
                            "priority": todo.priority,
                            "status": todo.status,
                            "note": todo.note,
                            "url": todo.url,
                        }
                        for todo in result.memory.todos
                    ],
                    "notes": [
                        {
                            "title": note.title,
                            "summary": summarize_text(note.summary, 4000),
                            "urls": note.urls,
                        }
                        for note in result.memory.notes
                    ],
                }

                if self.agent_file_service and self._should_cleanup_agent_files():
                    try:
                        await self.agent_file_service.delete_agent_file(agent_id)
                    except Exception as exc:
                        logger.warning("Failed to cleanup agent file", agent_id=agent_id, error=str(exc))

                return result.finding, agent_id, agent_status

        limited_tasks = tasks[: self.max_concurrent]

        results = await asyncio.gather(
            *[run_task(idx, topic) for idx, topic in enumerate(limited_tasks)],
            return_exceptions=True,
        )

        findings: list[ResearchFinding] = []
        agent_statuses: dict[str, dict[str, Any]] = {}

        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Agent task failed", task=tasks[idx] if idx < len(tasks) else "unknown", error=str(result))
                continue
            if result:
                finding, agent_id, agent_status = result
                if finding:
                    findings.append(finding)
                agent_statuses[agent_id] = agent_status

        return findings, agent_statuses

    def _share_finding(self, finding: ResearchFinding) -> None:
        if not finding.summary:
            return
        try:
            from src.workflow.agentic.models import AgentNote

            shared_note = AgentNote(
                title=f"Finding: {finding.topic}",
                summary=summarize_text(finding.summary, 8000),
                urls=[source.url for source in finding.sources[:3]],
                tags=["finding"],
            )
            self.shared_memory.add_note(shared_note)
        except Exception as exc:
            logger.warning("Shared note creation failed", error=str(exc))

    def _reset_session(self) -> None:
        """Reset session state - clear shared memory and reuse lead agent."""
        self.shared_memory = SharedResearchMemory()
        if self.agent_file_service:
            try:
                pass
            except Exception as e:
                logger.warning("Failed to clear agent files during reset", error=str(e))

    def _should_cleanup_agent_files(self) -> bool:
        if not self.agent_file_service:
            return False
        try:
            memory_dir = Path(self.agent_file_service.file_manager.memory_dir)
        except Exception:
            return False
        return "agent_sessions" in memory_dir.parts

    def _build_assignment(self, topic: str) -> str:
        query = self.query or "unknown"
        chat_block = format_chat_history(self.chat_history, limit=len(self.chat_history))
        return (
            f"Primary query: {query}\n"
            f"Your focus: {topic}\n"
            f"{chat_block}\n"
            "Capture high-signal evidence with citations. Share notes that help other agents."
        )

    async def _get_agent_character_preferences(self, agent_id: str, topic: str) -> tuple[str, str]:
        if not self.agent_file_service:
            return "", ""

        try:
            agent_file_content = await self.agent_file_service.read_agent_file(agent_id)
            character = agent_file_content.get("character", "")
            preferences = agent_file_content.get("preferences", "")
            return character, preferences
        except FileNotFoundError:
            return "", ""
        except Exception as e:
            logger.warning("Failed to get agent character/preferences", agent_id=agent_id, error=str(e))
            return "", ""


def _format_findings(findings: list[ResearchFinding]) -> str:
    if not findings:
        return "None."
    lines = []
    for finding in findings[-6:]:
        lines.append(f"- {finding.topic}: {summarize_text(finding.summary, 8000)}")
    return "\n".join(lines)


def _format_agent_file_snapshot(agent_file: dict[str, Any]) -> str:
    if not agent_file:
        return "None."

    lines = []
    character = agent_file.get("character") or ""
    preferences = agent_file.get("preferences") or ""
    todos = agent_file.get("todos") or []
    notes = agent_file.get("notes") or []

    if character:
        lines.append("Character:")
        lines.append(summarize_text(character, 4000))
        lines.append("")

    if preferences:
        lines.append("Preferences:")
        lines.append(summarize_text(preferences, 4000))
        lines.append("")

    lines.append("Todos:")
    if todos:
        for item in todos[:8]:
            payload = {
                "reasoning": getattr(item, "reasoning", "") or "",
                "title": getattr(item, "title", "") or "",
                "objective": getattr(item, "objective", "") or "",
                "expected_output": getattr(item, "expected_output", "") or "",
                "sources_needed": getattr(item, "sources_needed", []) or [],
                "priority": getattr(item, "priority", "medium") or "medium",
                "status": getattr(item, "status", "pending") or "pending",
                "note": getattr(item, "note", "") or "",
                "url": getattr(item, "url", "") or "",
            }
            lines.append(f"- {json.dumps(payload, ensure_ascii=False)}")
    else:
        lines.append("- None")

    if notes:
        lines.append("")
        lines.append("Notes:")
        for note in notes[-5:]:
            lines.append(f"- {summarize_text(str(note), 1600)}")

    return "\n".join(lines).strip()


def _dedupe_tasks(tasks: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for task in tasks:
        task_text = str(task).strip()
        if not task_text:
            continue
        key = task_text.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(task_text)
    return unique


def _align_tasks_with_query(tasks: list[str], query: str) -> list[str]:
    if not query:
        return tasks
    anchor_tokens = {token for token in re.findall(r"\w+", query.lower()) if len(token) >= 4}
    if not anchor_tokens:
        return tasks

    aligned: list[str] = []
    for task in tasks:
        task_text = str(task).strip()
        if not task_text:
            continue
        if any(token in task_text.lower() for token in anchor_tokens):
            aligned.append(task_text)
    return aligned or tasks


def _language_hint(text: str) -> str:
    return "match the query language"


def _normalize_tasks(tasks: list[str], query: str) -> list[str]:
    normalized = []
    for task in tasks:
        text = _normalize_task_text(task, query)
        if text:
            normalized.append(text)
    return normalized


def _normalize_task_text(task: str, query: str) -> str:
    text = str(task).strip()
    if not text:
        return ""
    text = re.sub(r"\s*Depth:.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*Primary query:.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*Your focus:.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*Chat history:.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^Task:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" -;")
    if query and text.lower().startswith(query.lower()):
        text = text[len(query):].lstrip(" :-—")
    if len(text) > 160:
        for sep in [". ", "; ", " — ", " - "]:
            idx = text.find(sep)
            if 0 < idx <= 160:
                text = text[:idx].rstrip()
                break
    if len(text) > 160:
        trimmed = text[:160].rstrip()
        if " " in trimmed:
            trimmed = trimmed.rsplit(" ", 1)[0]
        text = trimmed
    return text

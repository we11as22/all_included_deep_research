"""Agentic researcher with per-agent todo list and memory."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.search.base import SearchProvider
from src.search.scraper import WebScraper
from src.workflow.agentic.models import AgentMemory, SharedResearchMemory
from src.workflow.agentic.schemas import AgentAction
from src.workflow.nodes.memory_search import format_memory_context_for_prompt
from src.utils.chat_history import format_chat_history
from src.utils.date import get_current_date
from src.workflow.state import ResearchFinding, SourceReference
from src.memory.agent_memory_service import AgentMemoryService
from src.memory.agent_file_service import AgentFileService

logger = structlog.get_logger(__name__)


def get_agentic_system_prompt() -> str:
    """Get agentic system prompt with current date."""
    current_date = get_current_date()
    return f"""You are a research agent with tools and a personal todo list.

Current date: {current_date}

IMPORTANT: You must respond in the SAME LANGUAGE as the user's query. If the user asks in Russian, respond in Russian. If in English, respond in English. Always match the user's language.

You must respond with a single JSON object:
{{"action": "<tool_name or finish>", "args": {{...}}}}

Allowed actions:
- web_search: {{"action": "web_search", "args": {{"queries": ["..."], "max_results": 5}}}}
- scrape_urls: {{"action": "scrape_urls", "args": {{"urls": ["..."], "scroll": false}}}}
- scroll_page: {{"action": "scroll_page", "args": {{"url": "...", "scrolls": 3, "pause": 1.0}}}}
- write_note: {{"action": "write_note", "args": {{"title": "...", "summary": "...", "urls": ["..."], "tags": ["..."], "share": true}}}}
- update_note: {{"action": "update_note", "args": {{"file_path": "...", "summary": "...", "urls": ["..."]}}}}
- read_note: {{"action": "read_note", "args": {{"file_path": "items/..."}}}}
- add_todo: {{"action": "add_todo", "args": {{"items": [{{"title": "...", "note": "...", "url": "..."}}, "..."]}}}}
- update_todo: {{"action": "update_todo", "args": {{"title": "...", "status": "pending|in_progress|done", "note": "..."}}}}
- complete_todo: {{"action": "complete_todo", "args": {{"titles": ["..."]}}}}
- read_shared_notes: {{"action": "read_shared_notes", "args": {{"keyword": "...", "limit": 5}}}}
- read_agent_file: {{"action": "read_agent_file", "args": {{"agent_id": "..."}}}}
- read_main: {{"action": "read_main", "args": {{}}}}
- finish: {{}}

CRITICAL RULES FOR DEEP RESEARCH:
1. **Language Matching**: Always respond in the SAME LANGUAGE as the user's query. Match the user's language exactly.

2. **MANDATORY Todo Updates**: After EVERY tool call, you MUST update your todo list:
   - When you start working on a todo, immediately mark it as "in_progress" using update_todo
   - When you complete a todo, mark it as "done" using complete_todo
   - After discovering new information, add new todos for follow-up research using add_todo
   - After each search/scrape, review your todos and update them based on what you learned
   - NEVER skip todo updates - this is critical for deep research coordination

3. **REALLY DEEP Investigation** - This is DEEP RESEARCH mode, not surface-level search:
   - Don't stop after finding basic information - dig deeper into every interesting lead
   - For each topic, explore: historical context, current state, future trends, expert opinions, controversies, technical details, real-world applications, limitations, alternatives
   - Verify claims by checking multiple sources - don't trust single sources
   - Look for primary sources, original research papers, official documents, expert interviews
   - If you find conflicting information, investigate both sides thoroughly
   - Don't just summarize - analyze, compare, synthesize, and identify patterns
   - When you find something interesting, ask "what else should I know about this?" and add todos to explore further
   - Minimum 5-8 different search queries per topic, scraping at least 3-5 most promising sources
   - Read full articles, not just snippets - use scroll_page for dynamic content

4. **Active Plan Evolution**: Your research plan must evolve as you learn:
   - After each tool call, review what you learned and update your todos accordingly
   - If you discover a new angle or sub-topic, immediately add it as a new todo
   - If initial todos become irrelevant, update or replace them
   - Break down complex todos into smaller, actionable ones
   - Prioritize todos based on importance and dependencies

5. **Memory Sharing**: Actively read shared notes from other agents using read_shared_notes. Check what other agents have discovered and incorporate their findings into your research. Share your own high-signal notes so other agents can benefit.

6. **Todo Format**: Always use JSON format for todos. When adding todos, use: {{"action": "add_todo", "args": {{"items": [{{"title": "...", "note": "...", "url": "..."}}]}}}}. This ensures better parsing by all agents.

7. **Information Exchange**: After reading shared notes or agent files, actively update your research plan. If you discover something relevant to other agents' work, share it via write_note with share=true.

8. **Iterative Refinement**: Regularly review your todo list and the shared memory. Update your plans based on new information. Don't just follow your initial plan blindly - adapt as you learn more.

9. **Depth Indicators**: You're doing deep research correctly when:
   - You've explored multiple perspectives on the topic
   - You've verified information from multiple independent sources
   - You've identified and investigated sub-topics and related areas
   - You've found primary sources, not just secondary summaries
   - You've discovered connections between different pieces of information
   - You've identified gaps, contradictions, or areas needing more research
   - Your todos are constantly evolving and expanding as you learn

General Rules:
- Prefer web_search first if you need sources - use multiple queries from different angles.
- Use scrape_urls on the most promising links - don't just read snippets, get full content.
- Use scroll_page to load dynamic content on pages that require scrolling (e.g., infinite scroll, lazy loading).
- After scrolling, you can scrape_urls again to get the updated content.
- Write notes for anything worth reusing later, include URLs.
- Share only high-signal notes.
- Read main.md using read_main to see project status and shared knowledge.
- Read shared notes regularly to see what other agents are discovering.
- When the todo list is mostly done and you have enough evidence, use finish - but only if you've really dug deep.
- Always consider the current date ({current_date}) when evaluating information recency and relevance.
- Make research DEEP and THOROUGH - this is not a quick search, it's comprehensive investigation.
- Don't finish until you've explored at least 5-8 different angles and verified information from multiple sources.

WORKFLOW EXAMPLE FOR DEEP RESEARCH:
1. Start with broad search queries (3-5 different angles)
2. Scrape most promising sources (3-5 URLs)
3. Update todos: mark completed ones as done, add new ones for interesting leads
4. Read shared notes to see what others found
5. Follow up on interesting leads with more specific searches
6. Verify key claims by checking multiple sources
7. Add todos for sub-topics and related areas you discovered
8. Continue until you've thoroughly explored the topic from multiple angles
9. Only finish when you have comprehensive, verified information
"""


@dataclass
class AgenticResearchResult:
    """Result from agentic researcher."""

    finding: ResearchFinding
    memory: AgentMemory


class AgenticResearcher:
    """ReAct-style researcher with per-agent memory and todo list."""

    def __init__(
        self,
        llm: Any,
        search_provider: SearchProvider,
        web_scraper: WebScraper,
        shared_memory: SharedResearchMemory,
        memory_context: list[Any],
        chat_history: list[dict[str, str]] | None = None,
        stream: Any | None = None,
        max_steps: int = 6,
        max_sources: int = 8,
        agent_memory_service: AgentMemoryService | None = None,
        agent_file_service: AgentFileService | None = None,
    ) -> None:
        # Apply structured output to LLM
        if not hasattr(llm, "_structured_output") or llm._structured_output is None:
            try:
                # Use function_calling method for better OpenAI compatibility
                self.llm = llm.with_structured_output(AgentAction, method="function_calling")
            except Exception:
                # Fallback if structured output not supported
                self.llm = llm
        else:
            self.llm = llm
        self.search_provider = search_provider
        self.web_scraper = web_scraper
        self.shared_memory = shared_memory
        self.memory_context = memory_context
        self.chat_history = chat_history or []
        self.stream = stream
        self.max_steps = max_steps
        self.max_sources = max_sources
        self.agent_memory = AgentMemory()
        self.agent_memory_service = agent_memory_service
        self.agent_file_service = agent_file_service
        self.sources: list[SourceReference] = []
        self.current_topic: str | None = None

    async def run(
        self,
        agent_id: str,
        topic: str,
        existing_findings: list[ResearchFinding],
        assignment: str | None = None,
        character: str | None = None,
        preferences: str | None = None,
    ) -> AgenticResearchResult:
        self.current_topic = topic
        
        # Load or create agent's personal file
        if self.agent_file_service:
            agent_file = await self.agent_file_service.read_agent_file(agent_id)
            # Use character/preferences from file or provided
            if character:
                await self.agent_file_service.write_agent_file(agent_id, character=character)
            if preferences:
                await self.agent_file_service.write_agent_file(agent_id, preferences=preferences)
            # Load todos from file if they exist
            file_todos = agent_file.get("todos", [])
            if file_todos:
                self.agent_memory.todos = file_todos
        
        if assignment:
            self._seed_assignment(agent_id, assignment)
        
        # Only seed todos if agent file is empty
        if not self.agent_memory.todos:
            self._seed_todos(topic)
        
        # Save initial todos to agent file
        if self.agent_file_service and self.agent_memory.todos:
            await self.agent_file_service.write_agent_file(agent_id, todos=self.agent_memory.todos)
        
        last_tool_result = "None"

        if self.stream:
            self.stream.emit_research_start(agent_id, topic)
            self.stream.emit_agent_todo(agent_id, self._todo_payload())

        if getattr(self.llm, "_llm_type", "") == "mock-chat":
            note = self.agent_memory.add_note(
                title=f"Mock findings for {topic}",
                summary=f"Mock summary for {topic}.",
                urls=[],
                tags=["mock"],
            )
            self.shared_memory.add_note(note)
            if self.stream:
                self.stream.emit_agent_note(
                    agent_id,
                    {"title": note.title, "summary": note.summary, "urls": note.urls, "shared": True},
                )
                self.stream.emit_finding(agent_id, topic, note.summary, ["Mock key point."])
            return AgenticResearchResult(
                finding=ResearchFinding(
                    researcher_id=agent_id,
                    topic=topic,
                    summary=note.summary,
                    key_findings=["Mock key point."],
                    sources=[],
                    confidence="low",
                ),
                memory=self.agent_memory,
            )

        for step in range(self.max_steps):
            prompt = await self._build_prompt(
                agent_id=agent_id,
                topic=topic,
                existing_findings=existing_findings,
                last_tool_result=last_tool_result,
            )
            
            # Use structured output (already applied in __init__)
            try:
                response = await self.llm.ainvoke(
                    [SystemMessage(content=get_agentic_system_prompt()), HumanMessage(content=prompt)]
                )
                # Structured output should return AgentAction
                if isinstance(response, AgentAction):
                    action = response.action
                    args = response.args or {}
                elif hasattr(response, "action") and hasattr(response, "args"):
                    # Fallback: try to extract from response object
                    action = response.action
                    args = response.args or {}
                elif hasattr(response, "model_dump"):
                    # Pydantic model - extract fields
                    data = response.model_dump()
                    action = data.get("action", "finish")
                    args = data.get("args", {})
                    # If queries/max_results are at top level, move them to args
                    if "queries" in data and "queries" not in args:
                        args["queries"] = data["queries"]
                    if "max_results" in data and "max_results" not in args:
                        args["max_results"] = data["max_results"]
                else:
                    # Fallback to parsing text content
                    content = response.content if hasattr(response, "content") else str(response)
                    action, args = self._parse_action(content)
            except Exception as e:
                logger.warning("Structured output failed, falling back to parsing", error=str(e))
                # Try to get original LLM without structured output
                try:
                    # Get the underlying LLM if wrapped
                    original_llm = getattr(self.llm, "_llm", None) or getattr(self.llm, "bound", None) or self.llm
                    response = await original_llm.ainvoke(
                        [SystemMessage(content=get_agentic_system_prompt()), HumanMessage(content=prompt)]
                    )
                    content = response.content if hasattr(response, "content") else str(response)
                    action, args = self._parse_action(content)
                except Exception as e2:
                    logger.error("Fallback parsing also failed", error=str(e2))
                    action, args = "finish", {}
            
            if action == "finish":
                break

            last_tool_result = await self._execute_action(action, args, agent_id)

        finding = self._synthesize_finding(agent_id, topic)
        if self.stream:
            self.stream.emit_finding(agent_id, topic, finding.summary, finding.key_findings)
        return AgenticResearchResult(finding=finding, memory=self.agent_memory)

    def _seed_assignment(self, agent_id: str, assignment: str) -> None:
        note = self.agent_memory.add_note(
            title="Supervisor brief",
            summary=assignment.strip(),
            urls=[],
            tags=["assignment"],
        )
        self.shared_memory.add_note(note)
        if self.stream:
            self.stream.emit_agent_note(
                agent_id,
                {"title": note.title, "summary": note.summary, "urls": note.urls, "shared": True},
            )

    def _seed_todos(self, topic: str) -> None:
        if self.agent_memory.todos:
            return
        self.agent_memory.add_todo(f"Run broad search on {topic}")
        self.agent_memory.add_todo("Identify authoritative sources", note="Prefer official or primary sources")
        self.agent_memory.add_todo("Extract key facts and evidence")
        self.agent_memory.add_todo("Check for gaps or conflicting claims")

    async def _build_prompt(
        self,
        agent_id: str,
        topic: str,
        existing_findings: list[ResearchFinding],
        last_tool_result: str,
    ) -> str:
        memory_block = format_memory_context_for_prompt(self.memory_context[:6])
        shared_notes = self.shared_memory.render_notes(limit=6)
        chat_block = format_chat_history(self.chat_history, limit=len(self.chat_history))
        todo_block = self.agent_memory.render_todos(limit=8)
        notes_block = self.agent_memory.render_notes(limit=6)
        findings_block = _format_findings(existing_findings)
        current_date = get_current_date()

        # Load persistent memory files for agent context
        persistent_memory_block = ""
        if self.agent_memory_service:
            try:
                main_content = await self.agent_memory_service.read_main_file()
                items = await self.agent_memory_service.list_items()
                persistent_memory_block = "\n\n## Persistent Agent Memory Files\n\n"
                
                # Extract project status section if it exists
                if "## Project Status" in main_content:
                    status_start = main_content.find("## Project Status")
                    status_end = main_content.find("\n## ", status_start + len("## Project Status"))
                    if status_end == -1:
                        status_section = main_content[status_start:]
                    else:
                        status_section = main_content[status_start:status_end]
                    persistent_memory_block += f"### Current Project Status:\n{status_section}\n\n"
                
                persistent_memory_block += f"### Main Index (full content available via read_main tool):\n{main_content[:1000]}...\n\n"
                persistent_memory_block += "**Note:** Use read_main action to read the full main.md file.\n\n"
                if items:
                    persistent_memory_block += f"### Available Items ({len(items)} total, showing last 8):\n"
                    for item in items[-8:]:
                        persistent_memory_block += f"- **{item['title']}** ({item['file_path']}): {item['summary'][:100]}...\n"
                persistent_memory_block += "\nYou can reference these items and add new ones using write_note action.\n"
            except Exception as e:
                logger.warning("Failed to load persistent memory files", error=str(e), agent_id=agent_id)
        
        # Load agent's personal file
        agent_file_block = ""
        if self.agent_file_service:
            try:
                agent_file = await self.agent_file_service.read_agent_file(agent_id)
                agent_file_block = "\n\n## Your Personal Agent File\n\n"
                if agent_file.get("character"):
                    agent_file_block += f"**Character:** {agent_file['character'][:200]}\n\n"
                if agent_file.get("preferences"):
                    agent_file_block += f"**Preferences:** {agent_file['preferences'][:200]}\n\n"
            except Exception as e:
                logger.warning("Failed to load agent file", error=str(e), agent_id=agent_id)

        return f"""Agent: {agent_id}
Research topic: {topic}
Current date: {current_date}

{chat_block}

Memory context:
{memory_block}

Shared notes:
{shared_notes}
{persistent_memory_block}
{agent_file_block}
Your todo list:
{todo_block}

Your notes:
{notes_block}

Existing findings from other agents:
{findings_block}

Last tool result:
{last_tool_result}

Choose the next action (JSON only)."""

    async def _execute_action(self, action: str, args: dict[str, Any], agent_id: str) -> str:
        if action == "web_search":
            return await self._tool_web_search(args, agent_id)
        if action == "scrape_urls":
            return await self._tool_scrape_urls(args)
        if action == "scroll_page":
            return await self._tool_scroll_page(args)
        if action == "write_note":
            return await self._tool_write_note(args, agent_id)
        if action == "add_todo":
            return self._tool_add_todo(args, agent_id)
        if action == "complete_todo":
            return self._tool_complete_todo(args, agent_id)
        if action == "read_shared_notes":
            return self._tool_read_shared_notes(args)
        if action == "read_note":
            return await self._tool_read_note(args)
        if action == "update_note":
            return await self._tool_update_note(args, agent_id)
        if action == "update_todo":
            return await self._tool_update_todo(args, agent_id)
        if action == "read_agent_file":
            return await self._tool_read_agent_file(args)
        if action == "read_main":
            return await self._tool_read_main()

        return "Unknown action."

    async def _tool_web_search(self, args: dict[str, Any], agent_id: str) -> str:
        queries = args.get("queries") or []
        max_results = int(args.get("max_results") or 5)
        if isinstance(queries, str):
            queries = [queries]

        if not queries:
            return "No queries provided."

        summaries = []
        for query in queries[:3]:
            response = await self.search_provider.search(query, max_results=max_results)
            for result in response.results[:max_results]:
                self.sources.append(
                    SourceReference(
                        url=result.url,
                        title=result.title,
                        snippet=result.snippet,
                        relevance_score=result.score,
                    )
                )
                if self.stream:
                    self.stream.emit_source(agent_id, {"url": result.url, "title": result.title})
            top = response.results[:3]
            summary = "; ".join([f"{item.title} ({item.url})" for item in top])
            summaries.append(f"{query}: {summary}")

        if self.current_topic:
            self.agent_memory.complete_todo(f"Run broad search on {self.current_topic}")
        if self.stream:
            self.stream.emit_agent_todo(agent_id, self._todo_payload())
        return "Search results: " + " | ".join(summaries)

    async def _tool_scrape_urls(self, args: dict[str, Any]) -> str:
        urls = args.get("urls") or []
        scroll = args.get("scroll", False)
        if isinstance(urls, str):
            urls = [urls]
        if not urls:
            return "No urls to scrape."

        snippets = []
        for url in urls[:3]:
            try:
                content = await self.web_scraper.scrape(url, scroll=scroll)
                snippets.append(f"{content.title}: {content.content[:400]}")
            except Exception as exc:
                snippets.append(f"{url}: scrape failed ({exc})")

        return "Scraped content: " + " | ".join(snippets)

    async def _tool_scroll_page(self, args: dict[str, Any]) -> str:
        """Scroll page to load dynamic content."""
        url = str(args.get("url") or "")
        scrolls = int(args.get("scrolls") or 3)
        pause = float(args.get("pause") or 1.0)
        
        if not url:
            return "No URL provided for scrolling."
        
        # Check if Playwright is available
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return "Playwright not available. Install with: pip install playwright && playwright install chromium"
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        '--no-sandbox',
                        '--disable-setuid-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-gpu',
                        '--disable-software-rasterizer',
                        '--disable-extensions',
                    ]
                )
                context = await browser.new_context(
                    user_agent=self.web_scraper.user_agent,
                    viewport={"width": 1920, "height": 1080},
                    # Increase timeouts for slow connections
                    navigation_timeout=60000,
                )
                page = await context.new_page()
                
                logger.info("Scrolling page", url=url, scrolls=scrolls)
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                await page.wait_for_load_state("networkidle", timeout=15000)
                
                # Get initial height
                initial_height = await page.evaluate("document.body.scrollHeight")
                
                # Scroll down multiple times
                previous_height = initial_height
                total_scrolled = 0
                
                for i in range(scrolls):
                    # Scroll to bottom
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await asyncio.sleep(pause)
                    
                    # Wait for new content (increased timeout for slow connections)
                    try:
                        await page.wait_for_load_state("networkidle", timeout=10000)
                    except Exception:
                        pass
                    
                    # Check new height
                    new_height = await page.evaluate("document.body.scrollHeight")
                    
                    if new_height == previous_height:
                        logger.info("No new content after scroll", scroll_count=i + 1)
                        break
                    
                    previous_height = new_height
                    total_scrolled += 1
                
                # Get visible content in viewport after scrolling (at bottom)
                viewport_content = await page.evaluate("""
                    () => {
                        const elements = document.querySelectorAll('p, div, article, section, main, span');
                        let text = '';
                        const seen = new Set();
                        for (const el of elements) {
                            const rect = el.getBoundingClientRect();
                            // Check if element is visible in viewport (after scrolling to bottom)
                            if (rect.top >= 0 && rect.top < window.innerHeight && rect.width > 0 && rect.height > 0) {
                                const textContent = el.textContent?.trim();
                                if (textContent && textContent.length > 20 && !seen.has(textContent)) {
                                    seen.add(textContent);
                                    text += textContent + '\\n\\n';
                                }
                            }
                        }
                        return text.slice(0, 2000); // Limit to 2000 chars
                    }
                """)
                
                # Get page title
                page_title = await page.title()
                
                await browser.close()
                
                logger.info("Page scrolling completed", url=url, scrolls=total_scrolled, height_increase=new_height - initial_height)
                
                result = f"Scrolled {total_scrolled} times on '{page_title}'. Page height increased from {initial_height} to {new_height}px."
                if viewport_content:
                    result += f"\n\nViewport content (visible after scrolling):\n{viewport_content[:1000]}..."
                else:
                    result += "\n\nNo visible content captured in viewport."
                
                return result
        except Exception as e:
            logger.error("Scroll page failed", url=url, error=str(e))
            return f"Scroll failed: {str(e)}"

    async def _tool_write_note(self, args: dict[str, Any], agent_id: str) -> str:
        title = str(args.get("title") or "Note")
        summary = str(args.get("summary") or "")
        urls = args.get("urls") or []
        tags = args.get("tags") or []
        share = bool(args.get("share", True))

        note = self.agent_memory.add_note(title=title, summary=summary, urls=urls, tags=tags)
        if share:
            self.shared_memory.add_note(note)

        # Save to persistent memory if service is available
        if self.agent_memory_service:
            try:
                file_path = await self.agent_memory_service.save_agent_note(note, agent_id)
                logger.info("Agent note saved to file", file_path=file_path, agent_id=agent_id)
            except Exception as e:
                logger.warning("Failed to save agent note to file", error=str(e), agent_id=agent_id)

        if self.stream:
            self.stream.emit_agent_note(
                agent_id,
                {"title": note.title, "summary": note.summary, "urls": note.urls, "shared": share},
            )

        return f"Note saved: {title}"

    def _tool_add_todo(self, args: dict[str, Any], agent_id: str) -> str:
        items = args.get("items") or []
        added = 0
        if isinstance(items, str):
            items = [items]
        for item in items:
            if isinstance(item, str):
                self.agent_memory.add_todo(item)
                added += 1
            elif isinstance(item, dict):
                self.agent_memory.add_todo(
                    title=item.get("title", "Task"),
                    note=item.get("note"),
                    url=item.get("url"),
                )
                added += 1

        if self.stream:
            self.stream.emit_agent_todo(agent_id, self._todo_payload())
        return f"Added {added} todo items."

    def _tool_complete_todo(self, args: dict[str, Any], agent_id: str) -> str:
        titles = args.get("titles") or []
        if isinstance(titles, str):
            titles = [titles]
        completed = 0
        for title in titles:
            if self.agent_memory.complete_todo(title):
                completed += 1

        if self.stream:
            self.stream.emit_agent_todo(agent_id, self._todo_payload())
        return f"Completed {completed} todo items."

    def _tool_read_shared_notes(self, args: dict[str, Any]) -> str:
        keyword = str(args.get("keyword") or "").lower()
        limit = int(args.get("limit") or 5)
        if not keyword:
            return self.shared_memory.render_notes(limit=limit)

        matches = []
        for note in self.shared_memory.notes:
            if keyword in note.title.lower() or keyword in note.summary.lower():
                matches.append(note)
        if not matches:
            return "No shared notes matched."
        return "\n".join(
            [f"- {note.title}: {note.summary} (urls: {', '.join(note.urls[:2])})" for note in matches[:limit]]
        )

    async def _tool_read_note(self, args: dict[str, Any]) -> str:
        """Read a specific note file."""
        file_path = str(args.get("file_path") or "")
        if not file_path:
            return "No file_path provided."
        
        if not self.agent_memory_service:
            return "Memory service not available."
        
        try:
            content = await self.agent_memory_service.file_manager.read_file(file_path)
            # Extract summary for quick reference
            summary = self.agent_memory_service._extract_summary_from_content(content)
            return f"Note content:\n\n{content}\n\nSummary: {summary}"
        except FileNotFoundError:
            return f"Note file not found: {file_path}"
        except Exception as e:
            logger.warning("Failed to read note", file_path=file_path, error=str(e))
            return f"Failed to read note: {str(e)}"

    async def _tool_update_note(self, args: dict[str, Any], agent_id: str) -> str:
        """Update own note (only if created by this agent)."""
        file_path = str(args.get("file_path") or "")
        if not file_path:
            return "No file_path provided."
        
        if not self.agent_memory_service:
            return "Memory service not available."
        
        try:
            content = await self.agent_memory_service.file_manager.read_file(file_path)
            # Check if note was created by this agent
            created_by = self.agent_memory_service._extract_agent_from_content(content)
            if created_by != agent_id:
                return f"Cannot update note: created by {created_by}, not {agent_id}"
            
            # Update content
            summary = str(args.get("summary") or "")
            urls = args.get("urls") or []
            
            # Update summary section
            if summary:
                if "## Summary" in content:
                    parts = content.split("## Summary", 1)
                    next_section = parts[1].find("\n## ")
                    if next_section != -1:
                        content = parts[0] + "## Summary\n\n" + summary + "\n" + parts[1][next_section:]
                    else:
                        content = parts[0] + "## Summary\n\n" + summary + "\n"
                else:
                    content += f"\n\n## Summary\n\n{summary}\n"
            
            # Update URLs
            if urls:
                if "## Sources" in content:
                    parts = content.split("## Sources", 1)
                    next_section = parts[1].find("\n## ")
                    if next_section != -1:
                        sources_section = "## Sources\n\n" + "\n".join([f"- {url}" for url in urls]) + "\n"
                        content = parts[0] + sources_section + parts[1][next_section:]
                    else:
                        content = parts[0] + "## Sources\n\n" + "\n".join([f"- {url}" for url in urls]) + "\n"
                else:
                    content += f"\n\n## Sources\n\n" + "\n".join([f"- {url}" for url in urls]) + "\n"
            
            await self.agent_memory_service.file_manager.write_file(file_path, content)
            return f"Note updated: {file_path}"
        except Exception as e:
            logger.warning("Failed to update note", file_path=file_path, error=str(e))
            return f"Failed to update note: {str(e)}"

    async def _tool_update_todo(self, args: dict[str, Any], agent_id: str) -> str:
        """Update own todo item."""
        title = str(args.get("title") or "")
        if not title:
            return "No title provided."
        
        status = args.get("status")
        note = args.get("note")
        
        # Update in-memory todo
        updated = False
        for todo in self.agent_memory.todos:
            if todo.title == title:
                if status:
                    todo.status = status
                if note is not None:
                    todo.note = note
                updated = True
                break
        
        # Update in agent file
        if self.agent_file_service:
            await self.agent_file_service.update_agent_todo(agent_id, title, status, note)
        
        if self.stream:
            self.stream.emit_agent_todo(agent_id, self._todo_payload())
        
        if updated:
            return f"Todo updated: {title}"
        return f"Todo not found: {title}"

    async def _tool_read_agent_file(self, args: dict[str, Any]) -> str:
        """Read agent's personal file."""
        target_agent_id = str(args.get("agent_id") or "")
        if not target_agent_id:
            return "No agent_id provided."
        
        if not self.agent_file_service:
            return "Agent file service not available."
        
        try:
            agent_file = await self.agent_file_service.read_agent_file(target_agent_id)
            todos = agent_file.get("todos", [])
            notes = agent_file.get("notes", [])
            character = agent_file.get("character", "")
            preferences = agent_file.get("preferences", "")
            
            result = f"Agent file for {target_agent_id}:\n\n"
            if character:
                result += f"Character: {character}\n\n"
            if preferences:
                result += f"Preferences: {preferences}\n\n"
            result += "Todo List:\n"
            for todo in todos:
                result += f"- [{todo.status}] {todo.title}\n"
            result += "\nNotes:\n"
            for note in notes:
                result += f"- {note}\n"
            return result
        except Exception as e:
            logger.warning("Failed to read agent file", agent_id=target_agent_id, error=str(e))
            return f"Failed to read agent file: {str(e)}"

    async def _tool_read_main(self) -> str:
        """Read main.md file (all agents can read main.md)."""
        if not self.agent_memory_service:
            return "Memory service not available."
        
        try:
            content = await self.agent_memory_service.read_main_file()
            # Return full content, not truncated
            return f"Main memory file content:\n\n{content}"
        except Exception as e:
            logger.warning("Failed to read main file", error=str(e))
            return f"Failed to read main file: {str(e)}"

    def _synthesize_finding(self, agent_id: str, topic: str) -> ResearchFinding:
        notes = self.agent_memory.notes[-5:]
        if notes:
            summary = " ".join([note.summary for note in notes])[:1200]
            key_findings = [note.summary for note in notes[:3]]
            confidence = "medium" if self.sources else "low"
        else:
            summary = f"No detailed notes produced for {topic}."
            key_findings = ["No structured notes available."]
            confidence = "low"

        return ResearchFinding(
            researcher_id=agent_id,
            topic=topic,
            summary=summary,
            key_findings=key_findings,
            sources=self.sources[: self.max_sources],
            confidence=confidence,
        )

    def _todo_payload(self) -> list[dict[str, Any]]:
        """Return todos in JSON format for better parsing by all agents."""
        return [
            {
                "title": item.title,
                "status": item.status,
                "note": item.note or "",
                "url": item.url or "",
            }
            for item in self.agent_memory.todos
        ]

    def _parse_action(self, content: str) -> tuple[str, dict[str, Any]]:
        """Parse action from LLM response, handling both JSON and Python repr formats."""
        import re
        
        # Try to extract JSON first
        try:
            json_str = _extract_json(content)
            if json_str and json_str != "{}":
                payload = json.loads(json_str)
                action = str(payload.get("action", "finish"))
                args = payload.get("args") or {}
                return action, args
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Try to parse Python repr format: action='web_search' args={'queries': [...]}
        try:
            # Match action='...' or action="..."
            action_match = re.search(r"action\s*=\s*['\"]([^'\"]+)['\"]", content)
            if action_match:
                action = action_match.group(1)
                
                # Try to extract args dictionary
                args_match = re.search(r"args\s*=\s*(\{[^}]+\})", content, re.DOTALL)
                if args_match:
                    try:
                        # Try to evaluate as Python dict (safely)
                        args_str = args_match.group(1)
                        # Replace single quotes with double quotes for JSON
                        args_str = args_str.replace("'", '"')
                        args = json.loads(args_str)
                    except:
                        # Fallback: try to extract key-value pairs manually
                        args = {}
                        # Extract queries if present
                        queries_match = re.search(r"['\"]queries['\"]\s*:\s*\[([^\]]+)\]", content)
                        if queries_match:
                            queries_str = queries_match.group(1)
                            # Extract quoted strings
                            queries = re.findall(r"['\"]([^'\"]+)['\"]", queries_str)
                            args["queries"] = queries
                        
                        # Extract max_results if present
                        max_results_match = re.search(r"['\"]max_results['\"]\s*:\s*(\d+)", content)
                        if max_results_match:
                            args["max_results"] = int(max_results_match.group(1))
                        
                        # Extract urls if present
                        urls_match = re.search(r"['\"]urls['\"]\s*:\s*\[([^\]]+)\]", content)
                        if urls_match:
                            urls_str = urls_match.group(1)
                            urls = re.findall(r"['\"]([^'\"]+)['\"]", urls_str)
                            args["urls"] = urls
                        
                        # Extract other common args
                        for key in ["scroll", "url", "scrolls", "pause", "title", "summary", "file_path", "keyword", "limit", "agent_id", "status", "note", "titles", "items", "tags", "share"]:
                            if key in ["scroll", "share"]:
                                bool_match = re.search(rf"['\"]{key}['\"]\s*:\s*(True|False)", content)
                                if bool_match:
                                    args[key] = bool_match.group(1) == "True"
                            elif key in ["scrolls", "pause", "limit"]:
                                num_match = re.search(rf"['\"]{key}['\"]\s*:\s*([\d.]+)", content)
                                if num_match:
                                    args[key] = float(num_match.group(1)) if key == "pause" else int(num_match.group(1))
                            else:
                                str_match = re.search(rf"['\"]{key}['\"]\s*:\s*['\"]([^'\"]+)['\"]", content)
                                if str_match:
                                    args[key] = str_match.group(1)
                        
                        return action, args
                else:
                    return action, {}
        except Exception as e:
            logger.warning("Agent action parse failed", response=content[:200], error=str(e))
        
        logger.warning("Agent JSON parse failed, using finish", response=content[:200])
        return "finish", {}


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return "{}"
    return text[start : end + 1]


def _format_findings(findings: list[ResearchFinding]) -> str:
    if not findings:
        return "None."
    lines = []
    for finding in findings[-5:]:
        lines.append(f"- {finding.topic}: {finding.summary[:200]}")
    return "\n".join(lines)

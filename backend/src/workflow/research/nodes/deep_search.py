"""Deep search node for initial context gathering."""

import structlog
from typing import Dict, Any

from src.workflow.research.state import ResearchState
from src.workflow.research.nodes.base import ResearchNode

logger = structlog.get_logger(__name__)


class DeepSearchNode(ResearchNode):
    """Execute initial deep search to gather context.

    This node runs before clarification and planning to provide
    initial context about the research topic.
    """

    async def execute(self, state: ResearchState) -> Dict[str, Any]:
        """Execute deep search.

        Args:
            state: Current research state

        Returns:
            State updates with deep_search_result
        """
        # CRITICAL: Use original_query for deep search, not current query which might be clarification answer!
        query = state.get("original_query", state["query"])
        session_status = state.get("session_status", "active")
        chat_history = state.get("chat_history", [])

        # CRITICAL: Check if deep search should be skipped
        # Deep search should run ONLY ONCE at the beginning!

        # Method 0: CRITICAL - Check chat_history for clarification questions
        # If clarification questions exist in chat_history, this is a continuation after user answered
        # Deep search was already done in the first run, so SKIP!
        if chat_history:
            for msg in reversed(chat_history[-5:]):  # Check last 5 messages
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    # Check if this message contains clarification questions
                    if "Clarification Needed" in content or "Q1:" in content or "Q2:" in content:
                        logger.info(
                            "Skipping deep search - found clarification questions in recent chat history (continuation)",
                            session_id=state.get("session_id"),
                            session_status=session_status,
                        )
                        # Return empty to skip (deep search result already in chat_history from first run)
                        return {"deep_search_result": {"type": "override", "value": ""}}

        # Method 1: Check if deep_search_result already exists in state
        existing_result_raw = state.get("deep_search_result", "")

        # Handle both dict and string formats
        if isinstance(existing_result_raw, dict):
            existing_result = existing_result_raw.get("value", "")
        else:
            existing_result = existing_result_raw or ""

        # If result exists and is not empty, skip deep search (regardless of session_status!)
        if existing_result and existing_result.strip():
            logger.info(
                "Skipping deep search - result already exists in state",
                session_status=session_status,
                result_length=len(existing_result),
            )
            # Return override to ensure result is preserved
            return {
                "deep_search_result": {"type": "override", "value": existing_result}
            }

        # Method 2: Check database for deep search result in CURRENT session
        # This prevents double execution within same session while allowing new sessions to run
        session_id = state.get("session_id")
        if session_id and self.deps.session_manager:
            try:
                db_session = await self.deps.session_manager.get_session(session_id)
                if db_session and db_session.deep_search_result:
                    logger.info(
                        "Skipping deep search - already exists in DB for this session",
                        session_id=session_id,
                        session_status=session_status,
                        result_length=len(db_session.deep_search_result),
                    )
                    # Return empty override to skip (result already in chat_history)
                    return {
                        "deep_search_result": {"type": "override", "value": ""}
                    }
            except Exception as e:
                logger.warning("Failed to check DB for deep search result",
                              session_id=session_id,
                              error=str(e))
                # Continue with deep search if DB check fails

        # Execute deep search
        stream = self.deps.stream
        if stream:
            stream.emit_status("Starting deep search...", step="deep_search")

        search_provider = self.deps.search_provider
        scraper = self.deps.scraper
        llm = self.deps.llm

        logger.info("Running initial deep search", query=query[:100])

        # Perform web search
        search_response = await search_provider.search(query, max_results=10)

        # Extract content from top results
        contents = []
        total_results = min(len(search_response.results), 5)

        if stream:
            stream.emit_status(f"Analyzing {total_results} sources...", step="deep_search")

        for i, result in enumerate(search_response.results[:5]):
            try:
                url = result.url
                if url:
                    # Stream progress and current source to frontend
                    if stream:
                        stream.emit_status(
                            f"Analyzing source {i+1}/{total_results}: {result.title or url}",
                            step="deep_search"
                        )
                        stream.emit_source(
                            "deep_search",
                            {"url": url, "title": result.title or "Unknown"}
                        )

                    scraped = await scraper.scrape(url)
                    if scraped and scraped.content:
                        contents.append(
                            {
                                "url": url,
                                "title": scraped.title or result.title,
                                "content": scraped.content[:2000],
                            }
                        )
                        if stream:
                            stream.emit_status(
                                f"✓ Source {i+1}/{total_results} analyzed",
                                step="deep_search"
                            )
            except Exception as e:
                logger.warning(
                    "Failed to scrape URL", url=result.url, error=str(e)
                )
                if stream:
                    stream.emit_status(
                        f"⚠ Source {i+1}/{total_results} failed: {str(e)[:50]}",
                        step="deep_search"
                    )

        # Synthesize findings using LLM
        if contents:
            if stream:
                stream.emit_status(
                    f"Synthesizing insights from {len(contents)} sources...",
                    step="deep_search"
                )

            # Get user language from state for response
            user_language = state.get("user_language", "English")
            logger.info("Starting LLM synthesis",
                       user_language=user_language,
                       query=query[:50],
                       contents_count=len(contents))

            synthesis_prompt = self._build_synthesis_prompt(query, contents, user_language)

            logger.info("Calling LLM for deep search synthesis",
                       prompt_length=len(synthesis_prompt),
                       user_language=user_language)
            synthesis_result = await llm.ainvoke(synthesis_prompt)

            logger.info("LLM synthesis completed",
                       result_length=len(synthesis_result.content) if synthesis_result.content else 0,
                       user_language=user_language)
            deep_search_result = synthesis_result.content
        else:
            deep_search_result = (
                f"Initial search for '{query}' found limited results. "
                "Proceeding with general research approach."
            )

        logger.info(
            "Deep search completed", result_length=len(deep_search_result)
        )

        if stream:
            stream.emit_status("Deep search completed", step="deep_search")
            # Stream the deep search result to frontend so user can see it
            stream.emit_report_chunk(f"## 🔍 Initial Deep Search\n\n{deep_search_result}")

        # Save deep search result to DB so we can check it later
        session_id = state.get("session_id")
        if session_id and self.deps.session_manager:
            try:
                await self.deps.session_manager.save_deep_search_result(session_id, deep_search_result)
                logger.info("Saved deep search result to DB", session_id=session_id)
            except Exception as e:
                logger.warning("Failed to save deep search result to DB",
                              session_id=session_id,
                              error=str(e))

        # Return with override to ensure result is set
        return {"deep_search_result": {"type": "override", "value": deep_search_result}}

    def _build_synthesis_prompt(
        self, query: str, contents: list, user_language: str = "English"
    ) -> str:
        """Build prompt for synthesizing search results.

        Args:
            query: User query
            contents: List of scraped content
            user_language: Language for the response

        Returns:
            Synthesis prompt
        """
        contents_text = "\n\n".join(
            [
                f"Source: {c['title']}\nURL: {c['url']}\n{c['content']}"
                for c in contents
            ]
        )

        prompt = f"""Synthesize the following search results to provide initial context for the research query.

Query: {query}

Search Results:
{contents_text}

Provide a comprehensive summary (500-800 words) that:
1. Overviews the main topic
2. Highlights key aspects and subtopics
3. Identifies important context for further research
4. Notes any interesting patterns or findings

CRITICAL: Write the ENTIRE summary in {user_language}. The user's query is in {user_language}, so your response MUST be in {user_language}.

Focus on providing useful context for deeper research, not a complete answer."""

        return prompt


# Legacy function wrapper for backward compatibility
async def run_deep_search_node(state: ResearchState) -> Dict:
    """Legacy wrapper for DeepSearchNode.

    This function maintains backward compatibility with existing code
    that imports run_deep_search_node directly.

    TODO: Update imports to use DeepSearchNode class directly,
    then remove this wrapper.
    """
    # Restore runtime dependencies from context (for backward compatibility)
    from src.workflow.research.nodes import runtime_deps_context

    runtime_deps = runtime_deps_context.get()
    if not runtime_deps:
        logger.warning("Runtime dependencies not found in context")
        return {"deep_search_result": ""}

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
    node = DeepSearchNode(deps)
    return await node.execute(state)

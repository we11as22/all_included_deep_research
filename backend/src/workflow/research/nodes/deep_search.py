"""Deep search node for initial context gathering."""

import asyncio
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
        session_id = state.get("session_id")
        
        logger.info("🔍 DEEP_SEARCH NODE: Starting execution",
                   session_id=session_id,
                   query_preview=query[:100] if query else None,
                   state_keys=list(state.keys())[:10],
                   note="Deep search node called - checking if should execute or skip")
        
        # CRITICAL: Load session data DIRECTLY from DB (source of truth)
        # Don't rely on state - it may be outdated from checkpoint
        # Load once and use for all checks: session_status, clarification_answers, deep_search_result
        session_status = state.get("session_status", "active")
        clarification_answers = state.get("clarification_answers", "")
        existing_result = ""
        
        # CRITICAL: Always check DB directly for latest session data FIRST (before any other checks)
        # This ensures we have the most up-to-date information, not stale checkpoint data
        # This is the MOST RELIABLE check - DB is the source of truth
        db_check_done = False
        
        # CRITICAL: Check if session_manager is available
        session_manager_available = False
        if hasattr(self, 'deps') and self.deps and hasattr(self.deps, 'session_manager'):
            session_manager_available = self.deps.session_manager is not None
        
        logger.warning("🔍 DEEP_SEARCH: Checking session_manager availability",
                     session_id=session_id,
                     has_deps=hasattr(self, 'deps'),
                     has_session_manager_attr=hasattr(self, 'deps') and self.deps and hasattr(self.deps, 'session_manager'),
                     session_manager_available=session_manager_available,
                     session_manager_type=type(self.deps.session_manager).__name__ if session_manager_available else "None",
                     note="CRITICAL: Checking if session_manager is available in deps")
        
        if session_id and session_manager_available:
            try:
                # CRITICAL: Load from DB FIRST - this is the source of truth
                session = await self.deps.session_manager.get_session(session_id)
                if session:
                    db_check_done = True
                    # Override state values with DB values (DB is source of truth)
                    old_session_status = session_status
                    old_clarification_answers = clarification_answers
                    session_status = session.status
                    if session.clarification_answers:
                        clarification_answers = session.clarification_answers
                    if session.deep_search_result:
                        existing_result = session.deep_search_result
                        # CRITICAL: If result exists in DB, return IMMEDIATELY without any further checks
                        logger.warning("🛑 DEEP_SEARCH: RESULT FOUND IN DB - RETURNING IMMEDIATELY (HIGHEST PRIORITY)",
                                     session_id=session_id,
                                     session_status=session_status,
                                     result_length=len(existing_result),
                                     result_preview=existing_result[:200],
                                     note="CRITICAL: Result exists in DB - returning immediately WITHOUT any further processing. This prevents double execution.")
                        stream = self.deps.stream
                        if stream:
                            stream.emit_status("Deep search completed (using existing result from DB)", step="deep_search")
                        return {
                            "deep_search_result": {"type": "override", "value": existing_result}
                        }
                    
                    logger.info("✅ DEEP_SEARCH: Loaded session data from DB (source of truth)",
                               session_id=session_id,
                               session_status=session_status,
                               old_session_status=old_session_status,
                               has_clarification_answers=bool(session.clarification_answers),
                               clarification_answers_length=len(clarification_answers) if clarification_answers else 0,
                               has_deep_search_result=False,
                               note="DB values override state values - no deep_search_result in DB, will check other conditions")
                else:
                    logger.warning("⚠️ DEEP_SEARCH: Session not found in DB",
                                 session_id=session_id,
                                 note="Using state values as fallback")
            except Exception as e:
                logger.error("❌ DEEP_SEARCH: Failed to load session from DB",
                           session_id=session_id,
                           error=str(e),
                           exc_info=True,
                           note="Using state values as fallback")
        
        if not db_check_done:
            logger.error("❌ DEEP_SEARCH: DB check FAILED - session_manager not available!",
                        session_id=session_id,
                        has_deps=hasattr(self, 'deps'),
                        has_session_manager_attr=hasattr(self, 'deps') and self.deps and hasattr(self.deps, 'session_manager'),
                        session_manager_value=self.deps.session_manager if (hasattr(self, 'deps') and self.deps and hasattr(self.deps, 'session_manager')) else "N/A",
                        note="CRITICAL ERROR: Cannot check DB - will use state values (less reliable, may cause double execution!)")
        
        # CRITICAL: Deep search should run ONLY ONCE per session
        # Workflow: Запрос → deep_search → clarify → ответы → analyze → plan → execute
        # Deep search выполняется ОДИН РАЗ в начале
        # NOTE: If result was found in DB above, we already returned - no need to check again
        
        # Fallback: Check state if not found in DB (already loaded above if DB available)
        # This is only for cases where DB check failed or session_manager not available
        if not existing_result:
            existing_result_raw = state.get("deep_search_result", "")
            # Handle both dict and string formats
            if isinstance(existing_result_raw, dict):
                existing_result = existing_result_raw.get("value", "")
            else:
                existing_result = existing_result_raw or ""
            
            # If found in state, also return immediately
            if existing_result and existing_result.strip():
                logger.warning("🛑 DEEP_SEARCH: SKIPPING EXECUTION - existing result found in STATE (fallback check)",
                             session_id=session_id,
                             session_status=session_status,
                             result_length=len(existing_result),
                             result_preview=existing_result[:200],
                             note="CRITICAL: Deep search was already executed, result found in state. Returning existing result WITHOUT re-execution.")
                stream = self.deps.stream
                if stream:
                    stream.emit_status("Deep search completed (using existing result from state)", step="deep_search")
                return {
                    "deep_search_result": {"type": "override", "value": existing_result}
                }
        
        # CRITICAL CHECK 2: If user already answered clarification, skip deep_search execution
        # This is continuation after clarification - deep_search should NOT execute
        if session_status == "researching":
            logger.warning("🛑 DEEP_SEARCH: SKIPPING EXECUTION - session_status is 'researching'",
                         session_id=session_id,
                         session_status=session_status,
                         has_clarification_answers=bool(clarification_answers),
                         has_existing_result=bool(existing_result),
                         note="User answered clarification - this is continuation. Deep search should NOT execute. Returning empty result.")
            stream = self.deps.stream
            if stream:
                stream.emit_status("Skipping deep search (continuation after clarification)", step="deep_search")
            return {
                "deep_search_result": {"type": "override", "value": ""}
            }
        
        # CRITICAL CHECK 3: If clarification_answers exists, skip deep_search execution
        if clarification_answers and clarification_answers.strip():
            logger.warning("🛑 DEEP_SEARCH: SKIPPING EXECUTION - clarification_answers exists",
                         session_id=session_id,
                         session_status=session_status,
                         clarification_answers_length=len(clarification_answers),
                         clarification_answers_preview=clarification_answers[:100],
                         has_existing_result=bool(existing_result),
                         note="User answered clarification - this is continuation. Deep search should NOT execute. Returning empty result.")
            stream = self.deps.stream
            if stream:
                stream.emit_status("Skipping deep search (continuation after clarification)", step="deep_search")
            return {
                "deep_search_result": {"type": "override", "value": ""}
            }
        
        # Deep search result is empty - this is a new session, execute deep search
        # CRITICAL: Double-check that result is still not in DB (race condition protection)
        # This prevents double execution if two calls happen simultaneously
        if session_id and self.deps.session_manager:
            try:
                session = await self.deps.session_manager.get_session(session_id)
                if session and session.deep_search_result:
                    existing_result = session.deep_search_result
                    logger.warning("🛑 DEEP_SEARCH: RACE CONDITION DETECTED - result appeared in DB during execution",
                                 session_id=session_id,
                                 result_length=len(existing_result),
                                 note="Another process/thread saved result. Returning existing result to prevent double execution.")
                    stream = self.deps.stream
                    if stream:
                        stream.emit_status("Deep search completed (using existing result)", step="deep_search")
                    return {
                        "deep_search_result": {"type": "override", "value": existing_result}
                    }
            except Exception as e:
                logger.warning("Failed to double-check DB for race condition", error=str(e))
        
        logger.warning("✅ DEEP_SEARCH: EXECUTING - new session, no existing result",
                   session_id=session_id,
                   session_status=session_status,
                   query=query[:100],
                   has_existing_result=False,
                   note="CRITICAL: deep_search_result is empty - this is first run, executing deep search NOW. This should happen ONLY ONCE per session.")

        # Execute deep search
        stream = self.deps.stream
        if stream:
            stream.emit_status("Starting deep search...", step="deep_search")

        search_provider = self.deps.search_provider
        scraper = self.deps.scraper
        llm = self.deps.llm

        logger.info("Running initial deep search", query=query[:100])

        # Perform web search
        # CRITICAL: Use more results for better coverage (standalone deep_search uses balanced mode with more iterations)
        search_response = await search_provider.search(query, max_results=15)

        # Extract content from top results
        # CRITICAL: Scrape more pages for comprehensive context (standalone deep_search scrapes more)
        contents = []
        total_results = min(len(search_response.results), 10)

        if stream:
            stream.emit_status(f"Analyzing {total_results} sources...", step="deep_search")

        for i, result in enumerate(search_response.results[:10]):
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
                    # CRITICAL: Check both content and markdown - prefer markdown if available
                    # BUT: If markdown has no spaces (likely not real markdown), use plain text
                    content_to_use = None
                    if scraped:
                        # Try markdown first (better structure)
                        if hasattr(scraped, 'markdown') and scraped.markdown and scraped.markdown.strip():
                            # CRITICAL: If markdown has no spaces, it's likely not real markdown - use text instead
                            if ' ' in scraped.markdown:
                                content_to_use = scraped.markdown
                            else:
                                # Markdown has no spaces - likely not real markdown, use text
                                if hasattr(scraped, 'content') and scraped.content and scraped.content.strip():
                                    content_to_use = scraped.content
                                else:
                                    content_to_use = scraped.markdown  # Fallback to markdown even if no spaces
                        # Fallback to content
                        elif hasattr(scraped, 'content') and scraped.content and scraped.content.strip():
                            content_to_use = scraped.content
                    
                    if content_to_use:
                        contents.append(
                            {
                                "url": url,
                                "title": scraped.title if hasattr(scraped, 'title') and scraped.title else (result.title if result.title else "Unknown"),
                                "content": content_to_use[:5000],  # CRITICAL: Use more content for better synthesis (standalone uses more)
                            }
                        )
                        # Minimal logging - no huge content dumps
                        logger.debug(f"Added content to deep search synthesis",
                                   url=url[:100],  # Truncate URL
                                   content_length=len(content_to_use))
                        if stream:
                            stream.emit_status(
                                f"✓ Source {i+1}/{total_results} analyzed",
                                step="deep_search"
                            )
                    else:
                        logger.warning(f"Skipped URL - no content available",
                                     url=url,
                                     has_scraped=scraped is not None,
                                     has_content=hasattr(scraped, 'content') if scraped else False,
                                     has_markdown=hasattr(scraped, 'markdown') if scraped else False,
                                     content_length=len(scraped.content) if scraped and hasattr(scraped, 'content') else 0,
                                     markdown_length=len(scraped.markdown) if scraped and hasattr(scraped, 'markdown') else 0)
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
        # CRITICAL: Log contents before synthesis to debug empty results (minimal logging)
        logger.info("Preparing LLM synthesis",
                   contents_count=len(contents),
                   total_content_length=sum(len(c.get("content", "")) for c in contents))
        
        if contents:
            # CRITICAL: Verify contents actually have content
            valid_contents = [c for c in contents if c.get("content") and c.get("content").strip()]
            if not valid_contents:
                logger.error("CRITICAL: contents list is not empty but all items have empty content!",
                           contents_count=len(contents),
                           note="All contents have empty content - this will cause LLM to return empty result")
                # Use fallback
                deep_search_result = (
                    f"Initial deep search for '{query}' completed. "
                    f"Found {len(search_response.results)} search results, but content extraction had issues. "
                    "Proceeding with detailed research approach."
                )
            else:
                if stream:
                    stream.emit_status(
                        f"Synthesizing insights from {len(valid_contents)} sources...",
                        step="deep_search"
                    )

                # Get user language from state for response
                user_language = state.get("user_language", "English")
                logger.info("Starting LLM synthesis",
                           user_language=user_language,
                           query=query[:50],
                           valid_contents_count=len(valid_contents),
                           total_content_length=sum(len(c.get("content", "")) for c in valid_contents))

                synthesis_prompt = self._build_synthesis_prompt(query, valid_contents, user_language)

                logger.info("Calling LLM for deep search synthesis",
                           prompt_length=len(synthesis_prompt),
                           user_language=user_language,
                           prompt_preview=synthesis_prompt[:500])
                synthesis_result = await llm.ainvoke(synthesis_prompt)

                # CRITICAL: Extract content properly - handle different response formats
                synthesis_content = None
                if hasattr(synthesis_result, 'content'):
                    synthesis_content = synthesis_result.content
                elif isinstance(synthesis_result, str):
                    synthesis_content = synthesis_result
                elif hasattr(synthesis_result, 'text'):
                    synthesis_content = synthesis_result.text
                
                # CRITICAL: Log raw LLM response for debugging
                logger.info("LLM synthesis response received",
                           synthesis_result_type=type(synthesis_result).__name__,
                           content_length=len(synthesis_content) if synthesis_content else 0,
                           note="Raw LLM response received")
                
                # CRITICAL: If content is None or empty, log detailed error
                if not synthesis_content or not synthesis_content.strip():
                    logger.error("CRITICAL: LLM synthesis returned empty result!",
                               user_language=user_language,
                               synthesis_result_type=type(synthesis_result).__name__,
                               synthesis_result_repr=repr(synthesis_result)[:500],
                               has_content_attr=hasattr(synthesis_result, 'content'),
                               has_text_attr=hasattr(synthesis_result, 'text'),
                               content_attr_value=repr(getattr(synthesis_result, 'content', None))[:200] if hasattr(synthesis_result, 'content') else "N/A",
                               text_attr_value=repr(getattr(synthesis_result, 'text', None))[:200] if hasattr(synthesis_result, 'text') else "N/A",
                               valid_contents_count=len(valid_contents),
                               prompt_length=len(synthesis_prompt),
                               note="LLM returned empty result - this is a CRITICAL ERROR, not normal behavior!")
                    # Use fallback
                    deep_search_result = (
                        f"Initial deep search for '{query}' completed. "
                        f"Found {len(valid_contents)} sources with relevant information. "
                        "Proceeding with detailed research approach."
                    )
                else:
                    deep_search_result = synthesis_content
                    logger.info("LLM synthesis completed successfully",
                               result_length=len(deep_search_result),
                               user_language=user_language)
            
            # CRITICAL: Log formatting from LLM response (before any processing)
            import re
            if deep_search_result:
                raw_newline_count = deep_search_result.count('\n')
                raw_double_newline_count = deep_search_result.count('\n\n')
                has_markdown_headings = bool(re.search(r'^#{2,}\s+', deep_search_result, re.MULTILINE))
                
                logger.info(
                    "RAW LLM DEEP SEARCH RESPONSE (before any processing)",
                    result_length=len(deep_search_result),
                    newline_count=raw_newline_count,
                    double_newline_count=raw_double_newline_count,
                    has_markdown_headings=has_markdown_headings,
                    first_200_chars=repr(deep_search_result[:200]),  # Use repr to see actual \n characters
                    note="This is EXACTLY what LLM returned - no modifications yet"
                )
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
            # CRITICAL: Do NOT send deep_search_result to frontend here
            # It will be sent together with clarification as unified message
            # In workflow logic they are separate entities, but on frontend/DB they are combined
            logger.info("Deep search completed - result will be combined with clarification for frontend/DB",
                       result_length=len(deep_search_result),
                       note="Not sent separately - will be combined with clarification in clarify node")

        # CRITICAL: Save deep search result to DB IMMEDIATELY after execution (BEFORE returning)
        # This ensures that subsequent calls will find the result and skip execution
        # CRITICAL: Check if result already exists BEFORE saving (atomic check-and-save)
        # This prevents race condition where two calls both execute and both try to save
        session_id = state.get("session_id")
        
        # CRITICAL: Ensure result is not None or empty before saving
        # If result is empty, use fallback to ensure we always have something to save
        if not deep_search_result or not deep_search_result.strip():
            logger.warning("Deep search result is empty - using fallback before saving",
                         session_id=session_id,
                         original_result_length=len(deep_search_result) if deep_search_result else 0,
                         note="Result was empty - using fallback to ensure DB save succeeds")
            deep_search_result = (
                f"Initial deep search for '{query}' completed. "
                "Proceeding with detailed research approach."
            )
        
        # CRITICAL: Log session_manager availability before save attempt
        has_session_manager = hasattr(self, 'deps') and self.deps and hasattr(self.deps, 'session_manager') and self.deps.session_manager is not None
        logger.info("💾 DEEP_SEARCH: Attempting to save result to DB",
                    session_id=session_id,
                    has_session_manager=has_session_manager,
                    session_manager_type=type(self.deps.session_manager).__name__ if has_session_manager else "None",
                    result_length=len(deep_search_result) if deep_search_result else 0,
                    note="Saving deep search result to DB")
        
        if session_id and has_session_manager:
            try:
                # CRITICAL: Double-check that result is still not in DB (race condition protection)
                # Another process/thread may have saved it while we were executing
                final_check_session = await self.deps.session_manager.get_session(session_id)
                if final_check_session and final_check_session.deep_search_result:
                    existing_final_result = final_check_session.deep_search_result
                    logger.warning("🛑 DEEP_SEARCH: RACE CONDITION - result appeared in DB during execution (final check)",
                                 session_id=session_id,
                                 result_length=len(existing_final_result),
                                 our_result_length=len(deep_search_result) if deep_search_result else 0,
                                 note="CRITICAL: Another process saved result while we were executing. Returning existing result to prevent double save.")
                    # Return existing result instead of saving ours
                    return {
                        "deep_search_result": {"type": "override", "value": existing_final_result}
                    }
                
                # CRITICAL: Save immediately and wait for commit
                await self.deps.session_manager.save_deep_search_result(session_id, deep_search_result)
                
                # CRITICAL: Verify that result was saved by reading it back in separate transaction
                verification_session = await self.deps.session_manager.get_session(session_id)
                if verification_session and verification_session.deep_search_result:
                    logger.warning("💾 DEEP_SEARCH: Result saved and VERIFIED in DB (separate transaction)",
                                 session_id=session_id,
                                 result_length=len(deep_search_result) if deep_search_result else 0,
                                 verified_length=len(verification_session.deep_search_result),
                                 note="CRITICAL: Result saved and verified in separate transaction - subsequent calls will find it and skip execution")
                else:
                    logger.error("❌ DEEP_SEARCH: Result NOT found in DB after save!",
                               session_id=session_id,
                               note="CRITICAL ERROR: Result was saved but not found on verification in separate transaction!")
            except Exception as e:
                logger.error("❌ CRITICAL: Failed to save deep search result to DB",
                           error=str(e),
                           error_type=type(e).__name__,
                           session_id=session_id,
                           has_session_manager=has_session_manager,
                           note="CRITICAL ERROR: Result will NOT be saved - deep search will execute again on next call!")
        else:
            logger.error("❌ CRITICAL: Cannot save deep search result - session_manager not available!",
                        session_id=session_id,
                        has_session_manager=has_session_manager,
                        note="CRITICAL ERROR: Result will NOT be saved - deep search will execute again on next call!")

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

Provide a comprehensive, detailed summary (1500-2500 words) that:
1. Overviews the main topic
2. Highlights key aspects and subtopics
3. Identifies important context for further research
4. Notes any interesting patterns or findings

CRITICAL: Write the ENTIRE summary in {user_language}. The user's query is in {user_language}, so your response MUST be in {user_language}.

CRITICAL MARKDOWN FORMATTING REQUIREMENT:
- Your response MUST be valid markdown with proper formatting - NOT plain text!
- Use ## for main sections, ### for subsections
- Use **bold** for emphasis, *italic* for subtle emphasis
- Use proper markdown lists (- for unordered, 1. for ordered)
- CRITICAL NEWLINE FORMATTING: Use TWO newlines (\\n\\n) between paragraphs and sections for proper markdown rendering!
- Each paragraph must be separated by a blank line (two newlines: \\n\\n)!
- Sections must be separated by blank lines!
- This ensures proper markdown rendering on the frontend - without blank lines, paragraphs will merge together!

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

    # CRITICAL: Try to get session_manager from multiple sources
    # 1. First try context variable (may be lost in async LangGraph execution)
    # 2. Then try config (passed directly through LangGraph config)
    # 3. Finally try to create from session_factory if available
    
    runtime_deps = runtime_deps_context.get()
    session_manager = None
    
    # Try context variable first
    if runtime_deps:
        session_manager = runtime_deps.get("session_manager")
    
    # CRITICAL: If not in context, try to get from LangGraph config (passed through graph execution)
    # LangGraph passes config to nodes, but we need to access it differently
    # For now, try to get from context variable, and if missing, try to reconstruct from session_factory
    if not session_manager and runtime_deps:
        session_factory = runtime_deps.get("session_factory")
        if session_factory:
            # Create new SessionManager instance from factory
            from src.workflow.research.session.manager import SessionManager
            session_manager = SessionManager(session_factory)
            logger.warning("🔧 run_deep_search_node: Created SessionManager from session_factory",
                         session_id=state.get("session_id"),
                         note="Context variable lost, but reconstructed SessionManager from factory")
    
    if not runtime_deps:
        logger.error("❌ CRITICAL: Runtime dependencies not found in context")
        return {"deep_search_result": ""}

    # CRITICAL: Log session_manager availability
    logger.warning("🔍 run_deep_search_node: Checking session_manager availability",
                 has_runtime_deps=bool(runtime_deps),
                 runtime_deps_keys=list(runtime_deps.keys()) if runtime_deps else [],
                 has_session_manager_in_context="session_manager" in runtime_deps if runtime_deps else False,
                 has_session_manager_reconstructed=session_manager is not None,
                 session_manager_type=type(session_manager).__name__ if session_manager else "None",
                 note="CRITICAL: session_manager must be available for DB checks to work")

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
        session_manager=session_manager,  # CRITICAL: Use reconstructed session_manager if context lost it
        settings=runtime_deps.get("settings"),
    )
    
    # CRITICAL: Verify session_manager is in deps
    if not deps.session_manager:
        logger.error("❌ CRITICAL: session_manager is None in ResearchDependencies!",
                    session_id=state.get("session_id"),
                    runtime_deps_has_session_manager="session_manager" in runtime_deps if runtime_deps else False,
                    note="CRITICAL ERROR: DB checks will fail, deep search may execute multiple times!")

    # Execute node
    node = DeepSearchNode(deps)
    return await node.execute(state)

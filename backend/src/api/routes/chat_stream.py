"""Chat streaming endpoint with progress events."""

import asyncio
import time
from pathlib import Path
from uuid import uuid4

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.api.models.chat import ChatCompletionRequest
from src.streaming.sse import ResearchStreamingGenerator
from src.utils.pdf_generator import markdown_to_pdf
from src.memory.agent_session import create_agent_session_services, cleanup_agent_session_dir
from fastapi.responses import Response

router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = structlog.get_logger(__name__)

MAX_SESSION_REPORTS = 20
SESSION_REPORT_TTL_SECONDS = 2 * 60 * 60


def _store_session_report(app_state: object, session_id: str, report: str, query: str, mode: str) -> None:
    if not hasattr(app_state, "session_reports"):
        app_state.session_reports = {}
    if not hasattr(app_state, "session_report_order"):
        app_state.session_report_order = []

    now = time.time()
    app_state.session_reports[session_id] = {
        "report": report,
        "query": query,
        "mode": mode,
        "stored_at": now,
    }
    order = app_state.session_report_order
    if session_id in order:
        order.remove(session_id)
    order.append(session_id)
    _prune_session_reports(app_state)


def _prune_session_reports(app_state: object) -> None:
    reports = getattr(app_state, "session_reports", {})
    order = getattr(app_state, "session_report_order", [])
    now = time.time()

    expired = [
        session_id
        for session_id in list(order)
        if now - reports.get(session_id, {}).get("stored_at", now) > SESSION_REPORT_TTL_SECONDS
    ]
    for session_id in expired:
        reports.pop(session_id, None)
        if session_id in order:
            order.remove(session_id)

    while len(order) > MAX_SESSION_REPORTS:
        oldest = order.pop(0)
        reports.pop(oldest, None)


@router.post("/stream")
async def stream_chat(request: ChatCompletionRequest, app_request: Request):
    if not request.stream:
        raise HTTPException(status_code=501, detail="Only streaming responses are supported. Set 'stream=true'")

    user_messages = [msg for msg in request.messages if msg.role.value == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found in request")

    # CRITICAL: Get the ORIGINAL query (first user message before any clarification)
    # Load chat history from database if chat_id is provided
    chat_history = []
    original_query = None
    if request.chat_id:
        try:
            from src.database.schema import ChatMessageModel
            from sqlalchemy import select
            
            session_factory = app_request.app.state.session_factory
            async with session_factory() as session:
                result = await session.execute(
                    select(ChatMessageModel)
                    .where(ChatMessageModel.chat_id == request.chat_id)
                    .order_by(ChatMessageModel.created_at.asc())
                )
                db_messages = result.scalars().all()
                
                # Convert DB messages to chat history format
                for msg in db_messages:
                    chat_history.append({
                        "role": msg.role,
                        "content": msg.content
                    })
                
                # Find original query for CURRENT deep research session
                # This is the first user message in the current deep research session, before clarification
                # We need to find where the current deep research session started
                # Look for the last assistant message with clarification or deep search, then find first user message before it
                
                # First, determine if this is a deep research request
                raw_mode = request.model or "search"
                normalized = raw_mode.lower().replace("-", "_")
                is_deep_research = normalized in {"quality", "deep_research", "research"}
                
                if is_deep_research:
                    # Find the start of current deep research session
                    # Look backwards from the end to find last clarification or deep search message
                    # This helps identify the boundary of the current deep research session
                    last_clarification_or_deep_search_idx = -1
                    for i in range(len(chat_history) - 1, -1, -1):
                        msg = chat_history[i]
                        if msg.get("role") == "assistant":
                            content = msg.get("content", "").lower()
                            # Look for markers that indicate deep research session
                            if ("clarification" in content or "🔍" in content or "clarify" in content or 
                                "deep search" in content or "initial deep search" in content or
                                "research report" in content or "final report" in content):
                                last_clarification_or_deep_search_idx = i
                                logger.info("Found last deep research marker message", 
                                          message_index=i, 
                                          content_preview=content[:100])
                                break
                    
                    # Now find first user message BEFORE this marker
                    # This is the original query for current deep research session
                    # CRITICAL: This works even if there were other modes between deep research sessions
                    if last_clarification_or_deep_search_idx >= 0:
                        # Look for first user message before the marker
                        # This will be the query that started the current deep research session
                        # CRITICAL: Skip any user messages that come AFTER the marker (these are clarification answers)
                        for i in range(last_clarification_or_deep_search_idx - 1, -1, -1):
                            msg = chat_history[i]
                            if msg.get("role") == "user":
                                original_query = msg.get("content", "")
                                logger.info("Found original query for current deep research session", 
                                          original_query=original_query[:100] if original_query else None,
                                          message_index=i,
                                          before_marker_at=last_clarification_or_deep_search_idx)
                                break
                    else:
                        # No deep research markers found - this might be first deep research message in chat
                        # Or the first deep research message after other modes
                        # Find last user message (should be the query for current session)
                        # BUT: Check if last user message might be a clarification answer
                        # If chat_history has assistant messages, check if last one is clarification
                        found_clarification_before = False
                        for i in range(len(chat_history) - 1, -1, -1):
                            msg = chat_history[i]
                            if msg.get("role") == "assistant":
                                content = msg.get("content", "").lower()
                                if ("clarification" in content or "🔍" in content or "clarify" in content):
                                    found_clarification_before = True
                                    # Look for user message before this clarification
                                    for j in range(i - 1, -1, -1):
                                        prev_msg = chat_history[j]
                                        if prev_msg.get("role") == "user":
                                            original_query = prev_msg.get("content", "")
                                            logger.info("Found original query before clarification (no marker found, but found clarification pattern)", 
                                                      original_query=original_query[:100] if original_query else None,
                                                      message_index=j)
                                            break
                                    if original_query:
                                        break
                        
                        if not original_query:
                            # No clarification found - use last user message
                            for i in range(len(chat_history) - 1, -1, -1):
                                msg = chat_history[i]
                                if msg.get("role") == "user":
                                    original_query = msg.get("content", "")
                                    logger.info("Found original query (no deep research markers found, using last user message)", 
                                              original_query=original_query[:100] if original_query else None,
                                              message_index=i)
                                    break
                else:
                    # Not deep research - just use first user message
                    for msg in chat_history:
                        if msg.get("role") == "user":
                            original_query = msg.get("content", "")
                            logger.info("Found original query (not deep research, first user message)", 
                                      original_query=original_query[:100] if original_query else None)
                            break
                
                logger.info("Loaded chat history from database", 
                          chat_id=request.chat_id, 
                          messages_count=len(chat_history), 
                          original_query=original_query[:100] if original_query else None,
                          is_deep_research=is_deep_research)
        except Exception as e:
            logger.warning("Failed to load chat history from database", chat_id=request.chat_id, error=str(e))
            # Fallback to request messages
            chat_history = [{"role": msg.role.value, "content": msg.content} for msg in request.messages]
            # Determine mode first
            raw_mode = request.model or "search"
            normalized = raw_mode.lower().replace("-", "_")
            is_deep_research = normalized in {"quality", "deep_research", "research"}
            # For fallback, use first user message (should be the query)
            # CRITICAL: In fallback, request.messages might only contain clarification answer
            # So we use first user message, but log a warning
            original_query = user_messages[0].content if user_messages else None
            logger.warning("Using fallback: first user message as original query (chat_history from DB failed)", 
                      original_query=original_query[:100] if original_query else None,
                      is_deep_research=is_deep_research,
                      user_messages_count=len(user_messages))
    else:
        # Use messages from request
        chat_history = [{"role": msg.role.value, "content": msg.content} for msg in request.messages]
        # Determine mode first
        raw_mode = request.model or "search"
        normalized = raw_mode.lower().replace("-", "_")
        is_deep_research = normalized in {"quality", "deep_research", "research"}
        
        if is_deep_research and len(chat_history) > 1:
            # Find original query for current deep research session
            # Look backwards from the end to find last deep research marker message
            last_deep_research_marker_idx = -1
            for i in range(len(chat_history) - 1, -1, -1):
                msg = chat_history[i]
                if msg.get("role") == "assistant":
                    content = msg.get("content", "").lower()
                    # Look for markers that indicate deep research session
                    if ("clarification" in content or "🔍" in content or "clarify" in content or 
                        "deep search" in content or "initial deep search" in content or
                        "research report" in content or "final report" in content):
                        last_deep_research_marker_idx = i
                        break
            
            # Find first user message BEFORE this marker
            # This will be the query that started the current deep research session
            if last_deep_research_marker_idx >= 0:
                for i in range(last_deep_research_marker_idx - 1, -1, -1):
                    msg = chat_history[i]
                    if msg.get("role") == "user":
                        original_query = msg.get("content", "")
                        logger.info("Found original query for current deep research session (from request)", 
                                  original_query=original_query[:100] if original_query else None,
                                  message_index=i,
                                  before_marker_at=last_deep_research_marker_idx)
                        break
            
            if not original_query:
                # No deep research markers found or not found before it
                # CRITICAL: Don't use user_messages from request - they might be clarification answers!
                # Instead, search in chat_history for the first user message
                if len(chat_history) > 0:
                    for msg in chat_history:
                        if msg.get("role") == "user":
                            original_query = msg.get("content", "")
                            logger.warning("No markers found, using first user message from chat_history", 
                                         original_query=original_query[:100] if original_query else None)
                            break
                # Fallback to user_messages only if chat_history is empty
                if not original_query:
                    original_query = user_messages[0].content if user_messages else None
                    logger.warning("Using first user message from request (chat_history empty)", 
                                 original_query=original_query[:100] if original_query else None)
        else:
            # Not deep research or too short
            # CRITICAL: Use chat_history, not request.messages
            if len(chat_history) > 0:
                for msg in chat_history:
                    if msg.get("role") == "user":
                        original_query = msg.get("content", "")
                        logger.info("Not deep research, using first user message from chat_history", 
                                  original_query=original_query[:100] if original_query else None)
                        break
            # Fallback to user_messages only if chat_history is empty
            if not original_query:
                original_query = user_messages[0].content if user_messages else None
    
    # CRITICAL: Always use original query if found, otherwise use first user message
    # Never use last user message as it might be a clarification answer!
    # CRITICAL: For deep_research, original_query MUST come from chat_history, NOT from request.messages
    # because request.messages may only contain the clarification answer
    query = None
    if original_query:
        query = original_query
        logger.info("Using original query", query=query[:100], mode=mode if 'mode' in locals() else "unknown")
    elif is_deep_research and user_messages:
        # For deep_research, NEVER use request.messages - they may only contain clarification answer
        # Always search in chat_history for the original query
        if len(chat_history) >= 2:
            # Look for clarification pattern: assistant with clarification, then user message
            for i in range(len(chat_history) - 1, -1, -1):
                msg = chat_history[i]
                if msg.get("role") == "assistant":
                    content = msg.get("content", "").lower()
                    if ("clarification" in content or "🔍" in content or "clarify" in content):
                        # Found clarification - the query should be BEFORE this
                        # Look for user message before clarification
                        for j in range(i - 1, -1, -1):
                            prev_msg = chat_history[j]
                            if prev_msg.get("role") == "user":
                                query = prev_msg.get("content", "")
                                logger.info("Found query before clarification in chat_history", 
                                           query=query[:100], 
                                           clarification_at_index=i,
                                           query_at_index=j)
                                break
                        if query:
                            break
        
        # If still not found, use first user message from chat_history
        if not query and len(chat_history) > 0:
            for msg in chat_history:
                if msg.get("role") == "user":
                    query = msg.get("content", "")
                    logger.warning("Using first user message from chat_history (no clarification pattern found)", 
                                 query=query[:100])
                    break
        
        # Last resort: use first user message from request (should not happen if chat_history is loaded)
        if not query and user_messages:
            query = user_messages[0].content
            logger.error("CRITICAL: Using first user message from request - chat_history may be empty or incomplete!", 
                       query=query[:100],
                       chat_history_length=len(chat_history))
    elif user_messages:
        # For non-deep-research modes, try chat_history first
        if len(chat_history) > 0:
            for msg in chat_history:
                if msg.get("role") == "user":
                    query = msg.get("content", "")
                    logger.info("Using first user message from chat_history", query=query[:100])
                    break
        # Fallback to request messages
        if not query:
            query = user_messages[0].content  # Use FIRST user message, not last!
            logger.warning("Original query not found, using first user message from request", query=query[:100])
    else:
        query = ""
        logger.error("No user messages found - cannot determine query!")
    
    raw_mode = request.model or "search"
    normalized = raw_mode.lower().replace("-", "_")
    settings = app_request.app.state.settings
    
    # Limit chat history length
    if settings.chat_history_limit and len(chat_history) > settings.chat_history_limit:
        chat_history = chat_history[-settings.chat_history_limit:]

    if normalized in {"chat", "simple", "conversation"}:
        mode = "chat"
    elif normalized in {"speed"}:
        mode = "search"
    elif normalized in {"balanced"}:
        mode = "deep_search"
    elif normalized in {"quality"}:
        mode = "deep_research"
    elif normalized in {"search", "web", "web_search"}:
        mode = "search"
    elif normalized in {"deep_search", "deep"}:
        mode = "deep_search"
    elif normalized in {"deep_research", "research"}:
        mode = "deep_research"
    else:
        mode = "deep_search"

    logger.info("Chat stream request", mode=mode, query=query[:100])

    session_id = str(uuid4())
    # Pass app state to stream generator - include chat_id for DB saving
    app_state = {
        "debug_mode": bool(getattr(app_request.app.state, "settings", None) and app_request.app.state.settings.debug_mode),
        "chat_id": request.chat_id,  # CRITICAL: Pass chat_id for saving messages to DB
        "session_factory": app_request.app.state.session_factory,  # For DB access
    }
    stream_generator = ResearchStreamingGenerator(session_id=session_id, app_state=app_state)

    async def run_task():
        session_agent_dir: Path | None = None
        try:
            stream_generator.emit_init(mode=mode)
            stream_generator.emit_status("Starting chat workflow...", step="init")

            if mode == "deep_research":
                memory_root = Path(app_request.app.state.memory_manager.memory_dir)
                agent_memory_service, agent_file_service, session_agent_dir = create_agent_session_services(
                    memory_root, session_id
                )
                await agent_memory_service.read_main_file()
                # Store in both stream.app_state (for backward compatibility) and pass directly
                stream_generator.app_state["agent_memory_service"] = agent_memory_service
                stream_generator.app_state["agent_file_service"] = agent_file_service
                # Also store in a way that can be passed to graph
                stream_generator.app_state["_agent_memory_service"] = agent_memory_service
                stream_generator.app_state["_agent_file_service"] = agent_file_service

            if mode == "chat":
                # Simple chat mode - no web search
                chat_service = app_request.app.state.chat_service
                result = await chat_service.answer_simple(query, stream=stream_generator, messages=chat_history)
                
                stream_generator.emit_status("Drafting answer...", step="answer")
                for chunk in _chunk_text(result.answer, size=180):
                    stream_generator.emit_report_chunk(chunk)
                    await asyncio.sleep(0.02)
                stream_generator.emit_final_report(result.answer)
                stream_generator.emit_done()
                return
            elif mode in {"search", "deep_search"}:
                logger.info("Starting search/deep_search workflow", mode=mode, query=query[:100])
                chat_service = app_request.app.state.chat_service
                try:
                    if mode == "search":
                        logger.debug("Calling answer_web")
                        result = await chat_service.answer_web(query, stream=stream_generator, messages=chat_history)
                    else:
                        logger.debug("Calling answer_deep")
                        result = await chat_service.answer_deep(query, stream=stream_generator, messages=chat_history)
                    
                    logger.info("Search workflow completed", has_answer=bool(result.answer), answer_length=len(result.answer) if result.answer else 0)

                    # Emit final answer
                    if result.answer:
                        logger.info("Emitting final answer", answer_length=len(result.answer), mode=mode)
                        stream_generator.emit_status("Finalizing answer...", step="answer")
                        # CRITICAL: Send answer in chunks so user sees progress even if connection is slow
                        for chunk in _chunk_text(result.answer, size=180):
                            stream_generator.emit_report_chunk(chunk)
                            await asyncio.sleep(0.02)
                        # CRITICAL: emit_final_report will automatically save to DB
                        # This ensures answer is persisted even if client disconnects
                        stream_generator.emit_final_report(result.answer)
                        logger.info("Final report emitted and saved to DB", answer_length=len(result.answer), mode=mode)
                    else:
                        logger.warning("No answer generated from search workflow")
                        stream_generator.emit_error(error="No answer generated", details="Search completed but no answer was generated")
                    
                    # PDF generation only for deep_research mode
                    # Don't store reports for search/deep_search modes
                    
                    stream_generator.emit_done()
                    logger.info("Search workflow finished successfully")
                    return
                except Exception as e:
                    logger.error("Search workflow failed", error=str(e), exc_info=True)
                    stream_generator.emit_error(error=str(e), details="Search workflow encountered an error")
                    stream_generator.emit_done()
                    return

            # Use new LangGraph research system for deep_research mode
            from src.workflow.research import run_research_graph
            from src.config.modes import ResearchMode

            # Get settings and services
            settings = app_request.app.state.settings

            # Create mode config
            research_mode = ResearchMode.QUALITY
            mode_config = {
                "max_iterations": research_mode.get_max_iterations(),
                "max_concurrent": research_mode.get_max_concurrent(),
            }

            # Run new LangGraph research
            try:
                # Check if this is a continuation after clarification
                # If last assistant message has clarification and current user message exists, we're continuing
                is_continuation = False
                if len(chat_history) >= 2:
                    for i in range(len(chat_history) - 1, -1, -1):
                        msg = chat_history[i]
                        if msg.get("role") == "assistant":
                            content = msg.get("content", "").lower()
                            if "clarification" in content or "🔍" in content or "clarify" in content:
                                # Check if there's a user message after this
                                if i + 1 < len(chat_history) and chat_history[i + 1].get("role") == "user":
                                    is_continuation = True
                                    logger.info("Detected continuation after user clarification answer")
                                break
                
                final_state = await run_research_graph(
                    query=query,
                    chat_history=chat_history,
                    mode="quality",
                    llm=app_request.app.state.research_llm,
                    search_provider=app_request.app.state.chat_service.search_provider,
                    scraper=app_request.app.state.chat_service.scraper,
                    stream=stream_generator,
                    session_id=session_id,
                    mode_config=mode_config,
                    settings=settings,
                )
                
                # Check if graph stopped waiting for user (clarification needed but no answer yet)
                if isinstance(final_state, dict):
                    clarification_needed = final_state.get("clarification_needed", False)
                    clarification_waiting = final_state.get("clarification_waiting", False)
                    should_stop = final_state.get("should_stop", False)
                    
                    # If graph is still waiting for clarification and this is not a continuation
                    if (clarification_needed or clarification_waiting or should_stop) and not is_continuation:
                        # Graph stopped waiting for user - don't emit final report yet
                        logger.info("Graph stopped waiting for user clarification - not emitting final report", 
                                   clarification_needed=clarification_needed, 
                                   clarification_waiting=clarification_waiting,
                                   should_stop=should_stop)
                        # Emit status that we're waiting (but don't close stream - user needs to answer)
                        try:
                            stream_generator.emit_status("Waiting for your clarification answers...", step="clarification")
                            # Small delay to ensure events are sent
                            await asyncio.sleep(0.1)
                            stream_generator.emit_done()
                            logger.info("Clarification waiting - stream closed, waiting for user response")
                        except Exception as e:
                            logger.error("Failed to emit clarification waiting status", error=str(e), exc_info=True)
                            try:
                                stream_generator.emit_done()
                            except:
                                pass
                        return
                    
                    # If graph is still waiting even after continuation, it means research is still in progress
                    # Don't try to extract final_report yet
                    if (clarification_needed or clarification_waiting or should_stop) and is_continuation:
                        logger.info("Graph still waiting after continuation - research may still be in progress, checking for partial results")
                        # Research might still be running - check if we have any partial results
                        # But don't emit error, just log and continue
                        final_report_raw = final_state.get("final_report", "")
                        if not final_report_raw or (isinstance(final_report_raw, str) and not final_report_raw.strip()):
                            logger.info("No final report yet - research is still in progress, stream will continue")
                            # Don't emit done yet - research is still running
                            # The graph will continue and emit results via stream
                            return

                # Extract final_report - handle both dict with override and direct value
                logger.info("Extracting final report from state", state_keys=list(final_state.keys()) if isinstance(final_state, dict) else "not a dict")
                
                # Check if research is still in progress (no final_report yet)
                # This can happen when graph continues after clarification but research hasn't finished
                final_report_raw = None
                try:
                    final_report_raw = final_state.get("final_report", "") if isinstance(final_state, dict) else getattr(final_state, "final_report", "")
                except Exception as e:
                    logger.warning("Failed to extract final_report from state", error=str(e))
                    final_report_raw = ""
                
                logger.info("Final report raw", report_type=type(final_report_raw).__name__, has_value=bool(final_report_raw))
                
                # If no final_report and this is a continuation, research might still be in progress
                if not final_report_raw and is_continuation:
                    logger.info("No final_report yet after continuation - research may still be in progress")
                    # Check if research is actually still running by looking at state
                    # If there are findings or agent work in progress, research is still running
                    if isinstance(final_state, dict):
                        findings = final_state.get("findings", final_state.get("agent_findings", []))
                        iteration = final_state.get("iteration", 0)
                        should_continue = final_state.get("should_continue", False)
                        
                        if findings or iteration > 0 or should_continue:
                            logger.info("Research is still in progress - not emitting final report yet", 
                                       findings_count=len(findings) if findings else 0,
                                       iteration=iteration,
                                       should_continue=should_continue)
                            # Research is still running - don't emit done, let it continue
                            # The graph will emit results via stream as they become available
                            return
                
                if isinstance(final_report_raw, dict) and "value" in final_report_raw:
                    final_report = final_report_raw["value"]
                elif isinstance(final_report_raw, dict) and "content" in final_report_raw:
                    final_report = final_report_raw["content"]
                elif isinstance(final_report_raw, str):
                    final_report = final_report_raw
                else:
                    final_report = str(final_report_raw) if final_report_raw else ""
                
                logger.info("Final report extracted", report_length=len(final_report) if final_report else 0, preview=final_report[:200] if final_report else "empty")
                
                if final_report:
                    # Emit final report chunks and final report event
                    logger.info("Emitting final report", report_length=len(final_report))
                    stream_generator.emit_status("Finalizing report...", step="report")
                    for chunk in _chunk_text(final_report, size=200):
                        stream_generator.emit_report_chunk(chunk)
                        await asyncio.sleep(0.02)
                    stream_generator.emit_final_report(final_report)
                    logger.info("Final report emitted successfully")
                    
                    # Store final report in session for PDF generation
                    _store_session_report(app_request.app.state, session_id, final_report, query, mode)
                else:
                    logger.warning("No final report generated from research graph, creating fallback", final_state_keys=list(final_state.keys()) if isinstance(final_state, dict) else "not a dict")
                    # Try to get draft_report, main document, or findings as fallback
                    fallback_report = None
                    if isinstance(final_state, dict):
                        # Try draft_report first
                        agent_memory_service = stream_generator.app_state.get("agent_memory_service") if hasattr(stream_generator, "app_state") else None
                        if agent_memory_service:
                            try:
                                draft_report = await agent_memory_service.file_manager.read_file("draft_report.md")
                                if len(draft_report) > 500:
                                    fallback_report = draft_report
                                    logger.info("Using draft_report.md as fallback report", length=len(draft_report))
                            except Exception as e:
                                logger.warning("Could not read draft_report", error=str(e))
                        
                        # Try main document
                        if not fallback_report:
                            main_doc = final_state.get("main_document", "")
                            if main_doc and len(main_doc) > 500:
                                fallback_report = main_doc
                                logger.info("Using main document as fallback report", length=len(main_doc))
                        
                        # Try findings
                        if not fallback_report:
                            findings = final_state.get("findings", final_state.get("agent_findings", []))
                            if findings:
                                findings_text = "\n\n".join([
                                    f"## {f.get('topic', 'Unknown')}\n\n{f.get('summary', '')}\n\n"
                                    for f in findings
                                ])
                                fallback_report = f"# Research Report: {query}\n\n## Findings\n\n{findings_text}"
                                logger.info("Using findings as fallback report", findings_count=len(findings))
                    
                    if fallback_report:
                        logger.info("Emitting fallback report", report_length=len(fallback_report))
                        stream_generator.emit_status("Finalizing report...", step="report")
                        for chunk in _chunk_text(fallback_report, size=200):
                            stream_generator.emit_report_chunk(chunk)
                            await asyncio.sleep(0.02)
                        stream_generator.emit_final_report(fallback_report)
                        _store_session_report(app_request.app.state, session_id, fallback_report, query, mode)
                        logger.info("Fallback report emitted successfully")
                    else:
                        error_msg = "Research completed but no final report, draft report, main document, or findings were available"
                        logger.error(error_msg)
                        stream_generator.emit_error(error="No report generated", details=error_msg)
                    
                logger.info("Emitting done signal")
                stream_generator.emit_done()
            except Exception as e:
                logger.error("Research graph failed", error=str(e), exc_info=True)
                
                # CRITICAL: Even on error, try to send partial results to frontend
                # Extract any available results from state or draft_report
                fallback_report = None
                
                # Try to get draft_report as fallback
                agent_memory_service = stream_generator.app_state.get("agent_memory_service") if hasattr(stream_generator, "app_state") else None
                if agent_memory_service:
                    try:
                        draft_report = await agent_memory_service.file_manager.read_file("draft_report.md")
                        if len(draft_report) > 500:
                            fallback_report = draft_report
                            logger.info("Using draft_report.md as fallback after error", length=len(draft_report))
                    except Exception as e2:
                        logger.warning("Could not read draft_report after error", error=str(e2))
                
                # Try to extract partial results from final_state if available
                if not fallback_report and "final_state" in locals():
                    final_state_dict = final_state if isinstance(final_state, dict) else {}
                    final_report_raw = final_state_dict.get("final_report", "")
                    if final_report_raw:
                        fallback_report = str(final_report_raw)
                        logger.info("Using final_report from state as fallback after error")
                    else:
                        # Try findings
                        findings = final_state_dict.get("findings", final_state_dict.get("agent_findings", []))
                        if findings:
                            findings_text = "\n\n".join([
                                f"## {f.get('topic', 'Unknown')}\n\n{f.get('summary', '')}\n\n"
                                for f in findings
                            ])
                            fallback_report = f"# Research Report: {query}\n\n## Findings (Partial Results)\n\n{findings_text}\n\n---\n\n**Note:** Research encountered an error: {str(e)}"
                            logger.info("Using findings as fallback after error", findings_count=len(findings))
                
                # Emit error and fallback report if available
                if fallback_report:
                    logger.info("Emitting fallback report after error", report_length=len(fallback_report))
                    stream_generator.emit_error(error=f"Research encountered an error: {str(e)}", details="Partial results available below")
                    stream_generator.emit_status("Sending partial results...", step="error")
                    for chunk in _chunk_text(fallback_report, size=200):
                        stream_generator.emit_report_chunk(chunk)
                        await asyncio.sleep(0.02)
                    stream_generator.emit_final_report(fallback_report)
                    _store_session_report(app_request.app.state, session_id, fallback_report, query, mode)
                else:
                    # No fallback available - just emit error
                    stream_generator.emit_error(error=str(e), details="Research workflow encountered an error and no partial results are available")
                
                stream_generator.emit_done()

        except asyncio.CancelledError:
            logger.info("Chat task cancelled", session_id=session_id)
            stream_generator.emit_status("Chat cancelled by user", step="cancelled")
            stream_generator.emit_done()
            # Cleanup agent session dir on cancellation
            if session_agent_dir:
                memory_root = Path(app_request.app.state.memory_manager.memory_dir)
                cleanup_agent_session_dir(memory_root, session_agent_dir)
                logger.info("Agent session cleaned after cancellation", session_id=session_id)
        except Exception as exc:
            error_str = str(exc)
            error_details = "Chat stream failed"
            
            # Handle authentication errors specifically
            if "401" in error_str or "AuthenticationError" in error_str or "User not found" in error_str:
                error_details = (
                    "API ключ не настроен или неверен. "
                    "Пожалуйста, добавьте ваш OpenRouter API ключ в файл backend/.env: "
                    "OPENAI_API_KEY=sk-or-v1-ваш-ключ-здесь"
                )
                logger.error("Authentication error", error=error_str)
            elif "API key not configured" in error_str or "not configured" in error_str:
                error_details = (
                    "API ключ не настроен. "
                    "Пожалуйста, добавьте ваш OpenRouter API ключ в файл backend/.env: "
                    "OPENAI_API_KEY=sk-or-v1-ваш-ключ-здесь"
                )
                logger.error("API key missing", error=error_str)
            else:
                logger.error("Chat stream failed", error=error_str, exc_info=True)
            
            try:
                stream_generator.emit_error(error=error_details, details="Chat stream failed")
                stream_generator.emit_done()
            except Exception:
                pass  # Ignore errors during cleanup
            
            # Cleanup agent session dir on error
            if session_agent_dir:
                memory_root = Path(app_request.app.state.memory_manager.memory_dir)
                cleanup_agent_session_dir(memory_root, session_agent_dir)
                logger.info("Agent session cleaned after error", session_id=session_id)
        finally:
            # Remove task from active tasks
            app_request.app.state.active_tasks.pop(session_id, None)
            # Keep stream generator for a while after task completion to allow reconnection
            # Remove it after 5 minutes (for deep_research, user might reconnect)
            if hasattr(app_request.app.state, "active_streams"):
                # Schedule removal after 5 minutes
                async def remove_stream_after_delay():
                    await asyncio.sleep(300)  # 5 minutes
                    app_request.app.state.active_streams.pop(session_id, None)
                    logger.debug("Stream generator removed after delay", session_id=session_id)
                asyncio.create_task(remove_stream_after_delay())
            # DON'T cleanup session dir here - only cleanup on cancellation or error
            # This allows successful sessions to keep their files for debugging
            logger.debug("Task finished, session files preserved", session_id=session_id, preserved=session_agent_dir is not None)

    # Start background task and store it
    task = asyncio.create_task(run_task())
    app_request.app.state.active_tasks[session_id] = task
    
    # CRITICAL: Store stream generator for reconnection (especially for deep_research)
    # This allows clients to reconnect and get results even after disconnect
    if not hasattr(app_request.app.state, "active_streams"):
        app_request.app.state.active_streams = {}
    app_request.app.state.active_streams[session_id] = stream_generator

    async def watch_disconnect() -> None:
        """
        Watch for client disconnection, but DON'T cancel ANY tasks.
        ALL modes (chat, search, deep_search, deep_research) should complete even if client disconnects.
        This ensures all messages are persisted in DB and client can reconnect to get results.
        """
        try:
            while not task.done():
                if await app_request.is_disconnected():
                    # CRITICAL: For ALL modes, don't cancel - let tasks complete even if client disconnects
                    # This ensures all messages are persisted in DB and client can reconnect to get results
                    # All modes (chat, search, deep_search, deep_research) should be resilient to disconnections
                    logger.info(f"Client disconnected during {mode}, but continuing task to completion", 
                               session_id=session_id, mode=mode)
                    # Don't cancel - let the task complete
                    # Client can reconnect and get the result from DB
                    # All messages are saved via emit_final_report or emit_done
                    break
                await asyncio.sleep(0.5)
        except Exception as exc:
            logger.warning("Disconnect watcher failed", session_id=session_id, error=str(exc))

    asyncio.create_task(watch_disconnect())

    return StreamingResponse(
        stream_generator.stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-ID": session_id,
            "X-Research-Mode": mode,
        },
    )


def _chunk_text(text: str, size: int = 180) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


@router.post("/stream/{session_id}/cancel")
async def cancel_chat(session_id: str, app_request: Request):
    """Cancel an active chat session."""
    active_tasks = app_request.app.state.active_tasks
    
    if session_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Chat session not found or already completed")
    
    task = active_tasks[session_id]
    if task.done():
        active_tasks.pop(session_id, None)
        raise HTTPException(status_code=400, detail="Chat session already completed")
    
    task.cancel()
    active_tasks.pop(session_id, None)
    
    # Also remove stream generator
    if hasattr(app_request.app.state, "active_streams"):
        app_request.app.state.active_streams.pop(session_id, None)
    
    logger.info("Chat session cancelled", session_id=session_id)
    
    return {"status": "cancelled", "session_id": session_id}


@router.get("/stream/{session_id}/reconnect")
async def reconnect_stream(session_id: str, app_request: Request):
    """
    Reconnect to an active or completed stream session.
    
    This allows clients to reconnect and receive all events, even after disconnect.
    Especially useful for long-running deep_research tasks.
    """
    if not hasattr(app_request.app.state, "active_streams"):
        raise HTTPException(status_code=404, detail="No active streams found")
    
    stream_generator = app_request.app.state.active_streams.get(session_id)
    
    if not stream_generator:
        raise HTTPException(status_code=404, detail=f"Stream session {session_id} not found or expired")
    
    logger.info("Client reconnecting to stream", session_id=session_id)
    
    # Return the stream with history replay - client will receive all buffered events
    return StreamingResponse(
        stream_generator.stream(replay_history=True),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-ID": session_id,
        },
    )


@router.get("/stream/{session_id}/pdf")
async def generate_pdf(session_id: str, app_request: Request):
    """Generate PDF from final report for a completed research session."""
    if not hasattr(app_request.app.state, "session_reports"):
        raise HTTPException(status_code=404, detail="No reports found")
    
    session_reports = app_request.app.state.session_reports
    if session_id not in session_reports:
        raise HTTPException(status_code=404, detail="Report not found for this session")
    
    report_data = session_reports[session_id]
    report = report_data["report"]
    query = report_data["query"]
    
    try:
        # Generate PDF
        pdf_buffer = markdown_to_pdf(report, title=query[:80] or "Research Report")
        
        # Return PDF as response
        return Response(
            content=pdf_buffer.getvalue(),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="research_report_{session_id[:8]}.pdf"',
            },
        )
    except Exception as e:
        logger.error("PDF generation failed", error=str(e), session_id=session_id)
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")


def _collect_chat_history(messages: list, limit: int) -> list[dict[str, str]]:
    if limit <= 0:
        return []
    normalized = []
    for msg in messages:
        role = getattr(msg, "role", None)
        content = getattr(msg, "content", "") or ""
        if not content:
            continue
        role_value = role.value if hasattr(role, "value") else str(role or "user")
        if role_value == "system":
            continue
        normalized.append({"role": role_value, "content": content})
    return normalized[-limit:]

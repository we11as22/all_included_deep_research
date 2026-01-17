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

    # Get current user message
    current_user_message = user_messages[-1].content

    # Load chat history from database if chat_id is provided
    chat_history = []
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

                logger.info("Loaded chat history from database",
                          chat_id=request.chat_id,
                          messages_count=len(chat_history))
        except Exception as e:
            logger.warning("Failed to load chat history from database", chat_id=request.chat_id, error=str(e))
            # Fallback to request messages
            chat_history = [{"role": msg.role.value, "content": msg.content} for msg in request.messages]
    else:
        # Use messages from request
        chat_history = [{"role": msg.role.value, "content": msg.content} for msg in request.messages]
    
    raw_mode = request.model or "search"
    normalized = raw_mode.lower().replace("-", "_")
    settings = app_request.app.state.settings

    # Determine mode
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

    # Session management for deep_research mode with multi-chat support
    session_id = None
    query = current_user_message
    original_query = None
    is_new_session = False

    # CRITICAL DEBUG: Log mode and chat_id before SessionManager check
    logger.info("🔍 DEBUG: Checking for SessionManager invocation",
               mode=mode,
               chat_id=request.chat_id,
               current_message=current_user_message[:100])

    # Deep research mode
    if mode == "deep_research" and request.chat_id:
        logger.info("🔥 ENTERING SessionManager block!", mode=mode, chat_id=request.chat_id)
        # Use SessionManager for multi-chat session management
        from src.workflow.research.session.manager import SessionManager

        session_factory = app_request.app.state.session_factory
        session_manager = SessionManager(session_factory)

        # Get or create session for this chat
        research_session, is_new_session = await session_manager.get_or_create_session(
            chat_id=request.chat_id,
            query=current_user_message,
            mode=mode
        )

        session_id = research_session.id
        original_query = research_session.original_query

        if is_new_session:
            logger.info("Created new deep_research session",
                       session_id=session_id,
                       chat_id=request.chat_id,
                       mode=mode)
        else:
            logger.info("Resuming existing deep_research session",
                       session_id=session_id,
                       chat_id=request.chat_id,
                       status=research_session.status)

        # Use original_query from session (not current message which might be clarification answer)
        query = original_query
    elif mode != "deep_research" and request.chat_id:
        # For non-deep-research modes, check if there are active deep_research sessions to supersede
        from src.workflow.research.session.manager import SessionManager

        session_factory = app_request.app.state.session_factory
        session_manager = SessionManager(session_factory)

        # Mark any active deep_research sessions as superseded
        superseded_count = await session_manager.supersede_active_sessions(
            chat_id=request.chat_id,
            reason=f"User switched to {mode} mode"
        )

        if superseded_count > 0:
            logger.info("Superseded active deep_research sessions",
                       chat_id=request.chat_id,
                       count=superseded_count,
                       new_mode=mode)

    # Fallback: generate session_id for non-deep-research modes
    if not session_id:
        session_id = str(uuid4())

    # Limit chat history length
    if settings.chat_history_limit and len(chat_history) > settings.chat_history_limit:
        chat_history = chat_history[-settings.chat_history_limit:]

    logger.info("Chat stream request",
               mode=mode,
               query=query[:100] if query else "",
               session_id=session_id,
               is_new_session=is_new_session)
    # Pass app state to stream generator - include chat_id for DB saving
    app_state = {
        "debug_mode": bool(getattr(app_request.app.state, "settings", None) and app_request.app.state.settings.debug_mode),
        "chat_id": request.chat_id,  # CRITICAL: Pass chat_id for saving messages to DB
        "session_factory": app_request.app.state.session_factory,  # For DB access
        "embedding_provider": app_request.app.state.embedding_provider,  # CRITICAL: For generating embeddings for message search
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
                # Determine if this is a continuation based on session status (not text markers!)
                is_continuation = not is_new_session
                if is_continuation:
                    logger.info("Resuming deep_research session from checkpoint",
                               session_id=session_id,
                               status=research_session.status)
                
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
                    session_manager=session_manager,
                    session_factory=session_factory,
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
                
                # CRITICAL: Use session status from DB to determine what to do
                # This is the proper way - based on session state, not code checks
                session_status_from_db = research_session.status if research_session else None
                
                # If no final_report, check session status from DB
                if not final_report_raw:
                    # If this is a new session (not continuation), research is just starting - this is normal
                    # Don't emit error - let research proceed (deep search, clarification, etc.)
                    if not is_continuation:
                        logger.info("New session - research is starting, no final_report yet is normal",
                                   session_id=session_id,
                                   session_status=session_status_from_db)
                        # Research is just starting - don't emit error, let it proceed
                        # The graph will emit results via stream as they become available
                        return
                    
                    # If session is in "researching" status, research is still in progress
                    # Don't emit empty report - let research continue
                    if session_status_from_db == "researching":
                        logger.info("Session status is 'researching' - research is in progress, not emitting empty report",
                                   session_id=session_id,
                                   is_continuation=is_continuation)
                        # Research is still running - don't emit done, let it continue
                        # The graph will emit results via stream as they become available
                        return
                    
                    # If session is in "waiting_clarification", user needs to answer
                    # This is normal - clarification questions should be shown to user
                    if session_status_from_db == "waiting_clarification":
                        logger.info("Session is waiting for clarification - this is normal, not emitting error",
                                   session_id=session_id)
                        # Don't emit error - clarification questions should be shown to user
                        return
                    
                    # If session is in "active" status and no final_report, research may be starting
                    # This is normal for new sessions or early in research process
                    if session_status_from_db == "active":
                        logger.info("Session status is 'active' - research may be starting, not emitting error",
                                   session_id=session_id,
                                   is_continuation=is_continuation)
                        # Don't emit error - research may be in early stages
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
                    # No final_report - check if this is expected (research starting, waiting for clarification, etc.)
                    # Only emit error if research actually completed but no report was generated
                    
                    # Check if research completed (session status should be "completed" if research finished)
                    if session_status_from_db == "completed":
                        # Research completed but no final_report - try fallback
                        logger.warning("Research completed but no final_report, trying fallback sources",
                                     session_id=session_id)
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
                            # Research completed but no report available - this is an error
                            error_msg = "Research completed but no final report, draft report, main document, or findings were available"
                            logger.error(error_msg, session_id=session_id, session_status=session_status_from_db)
                            stream_generator.emit_error(error="No report generated", details=error_msg)
                    else:
                        # Research is not completed yet - this is normal, don't emit error
                        # Status is "active", "waiting_clarification", or "researching" - research is in progress
                        logger.info("No final_report but research is in progress - this is normal, not emitting error",
                                   session_id=session_id,
                                   session_status=session_status_from_db,
                                   is_continuation=is_continuation)
                        # Don't emit error - research is still running or waiting for user input
                        # The graph will emit results via stream as they become available
                    
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
    """
    Split text into chunks, trying to preserve markdown structure.
    Prefers breaking at newlines or markdown boundaries.
    """
    if len(text) <= size:
        return [text]
    
    chunks = []
    i = 0
    
    while i < len(text):
        # Try to find a good break point within the chunk size
        chunk_end = min(i + size, len(text))
        
        # If we're not at the end, try to find a better break point
        if chunk_end < len(text):
            # Prefer breaking at newlines
            newline_pos = text.rfind('\n', i, chunk_end)
            if newline_pos > i:
                chunk_end = newline_pos + 1
            else:
                # Try to break at space to avoid breaking words
                space_pos = text.rfind(' ', i, chunk_end)
                if space_pos > i:
                    chunk_end = space_pos + 1
                # Otherwise break at markdown boundaries (##, ###, etc.)
                elif chunk_end > i + 10:  # Only if we have enough space
                    md_boundary = text.rfind('##', i, chunk_end)
                    if md_boundary > i:
                        chunk_end = md_boundary
        
        chunks.append(text[i:chunk_end])
        i = chunk_end
    
    return chunks


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
        # Generate a concise title using LLM
        from langchain_openai import ChatOpenAI
        title_llm = getattr(app_request.app.state, "chat_llm", None)

        pdf_title = query[:80] or "Research Report"  # Default title

        if title_llm and len(report) > 100:
            try:
                # Extract first few paragraphs for context
                report_preview = report[:1000]
                title_prompt = f"""Generate a concise, descriptive title (max 60 characters) for this research report.

Original query: {query}

Report preview:
{report_preview}

Requirements:
- Maximum 60 characters
- Clear and descriptive
- In the same language as the query
- No quotes or special formatting

Title:"""

                title_response = await title_llm.ainvoke([{"role": "user", "content": title_prompt}])
                generated_title = title_response.content.strip().strip('"').strip("'")

                if generated_title and len(generated_title) <= 80:
                    pdf_title = generated_title
                    logger.info("Generated PDF title using LLM",
                               original_query=query[:50],
                               generated_title=pdf_title)
            except Exception as e:
                logger.warning("Failed to generate PDF title with LLM, using query", error=str(e))

        # Generate PDF
        pdf_buffer = markdown_to_pdf(report, title=pdf_title)
        
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

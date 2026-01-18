"""Socket.IO server for real-time chat streaming."""

import asyncio
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4
import socketio
import structlog

from src.streaming.socketio_stream import SocketIOStreamingGenerator
from src.memory.agent_session import create_agent_session_services, cleanup_agent_session_dir
from src.api.routes.chat_stream import _store_session_report

logger = structlog.get_logger(__name__)

# Create Socket.IO server
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
    logger=True,
    engineio_logger=False,
    ping_timeout=60,
    ping_interval=25,
)

# Store active streaming sessions
active_sessions: Dict[str, Dict[str, Any]] = {}


@sio.event
async def connect(sid: str, environ: Dict[str, Any]) -> None:
    """Handle client connection."""
    logger.info("Socket.IO client connected", sid=sid)
    await sio.emit('connected', {'sessionId': sid}, room=sid)


@sio.event
async def disconnect(sid: str) -> None:
    """Handle client disconnect."""
    logger.info("Socket.IO client disconnected", sid=sid)
    # Clean up session if exists
    if sid in active_sessions:
        session = active_sessions[sid]
        if 'task' in session and not session['task'].done():
            session['task'].cancel()
        del active_sessions[sid]


@sio.on('ping')
async def handle_ping(sid: str) -> None:
    """Handle heartbeat ping."""
    await sio.emit('pong', {}, room=sid)


@sio.on('chat:send')
async def handle_chat_send(sid: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle chat message from client."""
    try:
        chat_id = data.get('chatId')
        message = data.get('message')
        mode = data.get('mode', 'search')
        message_id = data.get('messageId')

        if not message:
            return {'error': 'Message content is required'}

        logger.info(
            "Received chat message",
            sid=sid,
            chat_id=chat_id,
            mode=mode,
            message_length=len(message)
        )

        # Import here to avoid circular imports
        from src.main import get_app_state
        app_state = get_app_state()
        # Build app_state payload for stream generator
        settings = getattr(app_state, "settings", None)
        stream_app_state: dict[str, Any] = {
            "debug_mode": bool(getattr(settings, "debug_mode", False)),
            "chat_id": chat_id,
            "session_factory": getattr(app_state, "session_factory", None),
            "settings": settings,
        }

        # CRITICAL: Handle session cancellation when mode changes from deep_research to other modes
        # If user switches mode (e.g., from deep_research to chat), cancel the active research session
        if mode not in ['deep_research', 'quality'] and chat_id:
            from src.workflow.research.session.manager import SessionManager
            session_factory = getattr(app_state, "session_factory", None)
            if session_factory:
                session_manager = SessionManager(session_factory)
                try:
                    active_session = await session_manager.get_active_session(chat_id)
                    if active_session:
                        await session_manager.update_status(active_session.id, "cancelled")
                        logger.info("🚫 Cancelled active deep_research session due to mode change",
                                   chat_id=chat_id,
                                   old_session_id=active_session.id,
                                   new_mode=mode)
                except Exception as e:
                    logger.warning("Failed to cancel active session", error=str(e))

        # CRITICAL: Use SessionManager for deep_research mode to track sessions across messages
        session_id = None
        original_query = None
        is_new_session = False
        research_session = None  # Store session for later use

        if mode in ['deep_research', 'quality'] and chat_id:
            logger.info("🔥 SocketIO: Using SessionManager for deep_research",
                       mode=mode,
                       chat_id=chat_id)
            from src.workflow.research.session.manager import SessionManager

            session_factory = getattr(app_state, "session_factory", None)
            if session_factory:
                session_manager = SessionManager(session_factory)
                research_session, is_new_session = await session_manager.get_or_create_session(
                    chat_id=chat_id,
                    query=message,
                    mode=mode
                )
                session_id = research_session.id
                original_query = research_session.original_query
                logger.info("✅ SocketIO: SessionManager result",
                           session_id=session_id,
                           is_new_session=is_new_session,
                           session_status=research_session.status,
                           original_query=original_query[:100])

        # Fallback: create new session_id for other modes
        if not session_id:
            session_id = str(uuid4())

        # Create streaming generator
        stream_generator = SocketIOStreamingGenerator(
            sid,
            sio,
            message_id=message_id,
            chat_id=chat_id,
            session_id=session_id,
            app_state=stream_app_state,
        )

        chat_service = getattr(app_state, "chat_service", None)
        if chat_service is None:
            await stream_generator.emit_error("Chat service is not initialized")
            await stream_generator.emit_done()
            return {"error": "Chat service is not initialized"}

        # Emit init event
        stream_generator.emit_init(mode)

        # Create background task for processing
        # CRITICAL: Capture research_session from outer scope for use in process_chat
        captured_research_session = research_session
        
        async def process_chat():
            session_agent_dir: Path | None = None
            # Use captured session as initial value, will be refreshed after graph execution
            current_research_session = captured_research_session
            try:
                # Load chat history if chat_id provided
                chat_history = []
                if chat_id:
                    from src.database.schema import ChatMessageModel
                    session_factory = getattr(app_state, "session_factory", None)
                    if session_factory is None:
                        raise RuntimeError("Session factory is not initialized")
                    async with session_factory() as session:
                        from sqlalchemy import select
                        result = await session.execute(
                            select(ChatMessageModel)
                            .filter(ChatMessageModel.chat_id == chat_id)
                            .order_by(ChatMessageModel.created_at)
                        )
                        messages = result.scalars().all()
                        chat_history = [
                            {'role': msg.role, 'content': msg.content}
                            for msg in messages
                        ]

                # CRITICAL: For deep_research mode, use original_query from session (already set above)
                # For other modes, determine original query from chat history
                # Initialize original_query if not already set (for non-deep-research modes)
                if 'original_query' not in locals() or original_query is None:
                    original_query = message
                
                if mode not in ['deep_research', 'quality']:
                    # Determine original query (first user message) for non-deep-research modes
                    original_query = message
                    if chat_history:
                        for msg in chat_history:
                            if msg['role'] == 'user':
                                original_query = msg['content']
                                break
                else:
                    # For deep_research mode, original_query should be already set from SessionManager above (line 126)
                    # If not set (shouldn't happen, but safety check), use current message
                    if original_query is None:
                        logger.warning("⚠️ SocketIO: original_query not set from session, using current message as fallback",
                                     session_id=session_id,
                                     current_message=message[:100])
                        original_query = message
                    else:
                        logger.info("🔍 SocketIO: Using original_query from deep_research session",
                                   original_query=original_query[:100] if original_query else None,
                                   current_message=message[:100],
                                   session_id=session_id,
                                   note="CRITICAL: original_query from session, NOT from chat_history!")

                # Process based on mode
                if mode == 'chat':
                    # Simple chat without search
                    result = await chat_service.answer_simple(
                        query=message,
                        stream=stream_generator,
                        messages=chat_history,
                    )
                    await _emit_answer(stream_generator, result.answer)
                elif mode in ['search', 'speed']:
                    # Fast web search
                    result = await chat_service.answer_web(
                        query=message,
                        stream=stream_generator,
                        messages=chat_history,
                    )
                    # CRITICAL: Log answer formatting before emitting
                    import re
                    answer = result.answer
                    newline_count = answer.count("\n") if answer else 0
                    has_newlines = "\n" in (answer or "")
                    logger.info("Web search answer received", 
                               answer_length=len(answer) if answer else 0,
                               newline_count=newline_count,
                               has_newlines=has_newlines,
                               has_markdown_headings=bool(re.search(r'^#{2,}\s+', answer, re.MULTILINE)) if answer else False)
                    await _emit_answer(stream_generator, answer)
                elif mode in ['deep_search', 'balanced']:
                    # Deep search with multiple iterations
                    result = await chat_service.answer_deep(
                        query=message,
                        stream=stream_generator,
                        messages=chat_history,
                    )
                    # CRITICAL: Log answer formatting before emitting
                    import re
                    answer = result.answer
                    newline_count = answer.count("\n") if answer else 0
                    has_newlines = "\n" in (answer or "")
                    logger.info("Deep search answer received", 
                               answer_length=len(answer) if answer else 0,
                               newline_count=newline_count,
                               has_newlines=has_newlines,
                               has_markdown_headings=bool(re.search(r'^#{2,}\s+', answer, re.MULTILINE)) if answer else False)
                    await _emit_answer(stream_generator, answer)
                elif mode in ['deep_research', 'quality']:
                    # Full research workflow using LangGraph
                    from src.workflow.research import run_research_graph
                    from src.config.modes import ResearchMode
                    from src.workflow.research.session.manager import SessionManager

                    memory_root = Path(app_state.memory_manager.memory_dir)
                    agent_memory_service, agent_file_service, session_agent_dir = create_agent_session_services(
                        memory_root, session_id
                    )
                    await agent_memory_service.read_main_file()
                    stream_generator.app_state["agent_memory_service"] = agent_memory_service
                    stream_generator.app_state["agent_file_service"] = agent_file_service
                    stream_generator.app_state["_agent_memory_service"] = agent_memory_service
                    stream_generator.app_state["_agent_file_service"] = agent_file_service

                    research_mode = ResearchMode.QUALITY
                    mode_config = {
                        "max_iterations": research_mode.get_max_iterations(),
                        "max_concurrent": research_mode.get_max_concurrent(),
                    }

                    # CRITICAL: Use original_query from SessionManager, not current message!
                    research_query = original_query if original_query else message
                    logger.info("🔥 SocketIO: Starting deep_research",
                               session_id=session_id,
                               is_new_session=is_new_session,
                               query=research_query[:100])

                    # Get SessionManager instance
                    session_factory = getattr(app_state, "session_factory", None)
                    session_manager = SessionManager(session_factory) if session_factory else None
                    
                    final_state = await run_research_graph(
                        query=research_query,
                        chat_history=chat_history,
                        mode="quality",
                        llm=app_state.research_llm,
                        search_provider=app_state.chat_service.search_provider,
                        scraper=app_state.chat_service.scraper,
                        stream=stream_generator,
                        session_id=session_id,
                        mode_config=mode_config,
                        settings=app_state.settings,
                        session_manager=session_manager,
                        session_factory=session_factory,
                    )

                    # CRITICAL: Refresh session status from DB after graph execution
                    # This ensures we have the latest status (may have changed during graph execution)
                    # Use session status from DB to determine what to do - this is the proper way
                    if session_manager and session_id:
                        try:
                            current_research_session = await session_manager.get_session(session_id)
                            logger.info("Refreshed session status from DB after graph execution",
                                       session_id=session_id,
                                       status=current_research_session.status if current_research_session else None)
                        except Exception as e:
                            logger.warning("Failed to refresh session status", error=str(e))
                            # Fallback to captured session if refresh fails
                            if not current_research_session:
                                current_research_session = captured_research_session
                    
                    # Use session status from DB to determine what to do
                    session_status_from_db = current_research_session.status if current_research_session else None
                    
                    # Check if graph stopped waiting for user clarification
                    if isinstance(final_state, dict):
                        clarification_needed = final_state.get("clarification_needed", False)
                        clarification_waiting = final_state.get("clarification_waiting", False)
                        should_stop = final_state.get("should_stop", False)

                        # If session is waiting for clarification and user hasn't answered, stop
                        if (clarification_needed or clarification_waiting or should_stop) and session_status_from_db == "waiting_clarification":
                            stream_generator.emit_status("Waiting for your clarification answers...", step="clarification")
                            await asyncio.sleep(0.1)
                            await stream_generator.emit_done()
                            return

                    # Extract final_report
                    final_report_raw = None
                    try:
                        final_report_raw = (
                            final_state.get("final_report", "")
                            if isinstance(final_state, dict)
                            else getattr(final_state, "final_report", "")
                        )
                    except Exception:
                        final_report_raw = ""

                    # CRITICAL: If no final_report, check session status from DB
                    # This is the proper way - use session state to determine continuation
                    if not final_report_raw:
                        # If this is a new session, research is just starting - this is normal
                        # Don't emit error - let research proceed (deep search, clarification, etc.)
                        if is_new_session:
                            logger.info("New session - research is starting, no final_report yet is normal",
                                       session_id=session_id,
                                       session_status=session_status_from_db)
                            # Research is just starting - don't emit error, let it proceed
                            # The graph will emit results via stream as they become available
                            await stream_generator.emit_done()
                            return
                        
                        # If session is in "researching" status, research is still in progress
                        # Don't emit empty report - let research continue
                        if session_status_from_db == "researching":
                            logger.info("Session status is 'researching' - research is in progress, not emitting empty report",
                                       session_id=session_id,
                                       is_new_session=is_new_session)
                            # Research is still running - don't emit done, let it continue
                            # The graph will emit results via stream as they become available
                            await stream_generator.emit_done()
                            return
                        
                        # If session is in "waiting_clarification", user needs to answer
                        # But if we got here and it's not a new session, user might have just answered
                        if session_status_from_db == "waiting_clarification":
                            # Check if user answered (graph should have updated status to "researching" if answered)
                            # If still "waiting_clarification", user hasn't answered yet - this is normal
                            logger.info("Session is waiting for clarification - this is normal, not emitting error",
                                       session_id=session_id)
                            # Don't emit error - clarification questions should be shown to user
                            await stream_generator.emit_done()
                            return
                        
                        # If session is in "active" status and no final_report, research may be starting
                        # This is normal for new sessions or early in research process
                        if session_status_from_db == "active":
                            logger.info("Session status is 'active' - research may be starting, not emitting error",
                                       session_id=session_id,
                                       is_new_session=is_new_session)
                            # Don't emit error - research may be in early stages
                            await stream_generator.emit_done()
                            return

                    if isinstance(final_report_raw, dict) and "value" in final_report_raw:
                        final_report = final_report_raw["value"]
                    elif isinstance(final_report_raw, dict) and "content" in final_report_raw:
                        final_report = final_report_raw["content"]
                    elif isinstance(final_report_raw, str):
                        final_report = final_report_raw
                    else:
                        final_report = str(final_report_raw) if final_report_raw else ""

                    # CRITICAL: Check for draft_report first - it's the structured research result with chapters
                    # Draft report is the primary result, final_report is generated from it
                    draft_report_available = False
                    draft_report_content = None
                    agent_memory_service = stream_generator.app_state.get("agent_memory_service") if hasattr(stream_generator, "app_state") else None
                    if agent_memory_service:
                        try:
                            draft_report_content = await agent_memory_service.file_manager.read_file("draft_report.md")
                            if len(draft_report_content) > 500:
                                draft_report_available = True
                                logger.info("Draft report available as research result",
                                           draft_length=len(draft_report_content),
                                           note="Draft report contains structured chapters and is the research result")
                        except Exception as e:
                            logger.warning("Could not read draft_report", error=str(e))
                    
                    # Priority: draft_report (structured chapters) > final_report (generated) > fallback
                    if draft_report_available and draft_report_content:
                        # CRITICAL: Draft report is the structured research result - send it as final result
                        logger.info("Sending draft_report as final research result (structured chapters)",
                                   draft_length=len(draft_report_content),
                                   session_status=session_status_from_db)
                        stream_generator.emit_status("Finalizing report...", step="report")
                        # CRITICAL: Send answer in chunks (same as deep research - 10000 chars per chunk)
                        for chunk in _chunk_text(draft_report_content, size=10000):
                            stream_generator.emit_report_chunk(chunk)
                            await asyncio.sleep(0.02)
                        stream_generator.emit_final_report(draft_report_content)
                        _store_session_report(app_state, session_id, draft_report_content, message, mode)
                    elif final_report:
                        # Fallback to generated final_report if draft_report not available
                        logger.info("Using generated final_report (draft_report not available)",
                                   final_report_length=len(final_report))
                        stream_generator.emit_status("Finalizing report...", step="report")
                        # CRITICAL: Send answer in chunks (same as deep research - 10000 chars per chunk)
                        for chunk in _chunk_text(final_report, size=10000):
                            stream_generator.emit_report_chunk(chunk)
                            await asyncio.sleep(0.02)
                        stream_generator.emit_final_report(final_report)
                        _store_session_report(app_state, session_id, final_report, message, mode)
                    elif session_status_from_db == "completed":
                        # CRITICAL: If research completed but no final_report, use draft_report as result
                        # Draft report is the structured research result with chapters
                        logger.info("Research completed - using draft_report as final result",
                                   session_id=session_id,
                                   note="Draft report contains structured chapters and is the research result")
                        agent_memory_service = stream_generator.app_state.get("agent_memory_service") if hasattr(stream_generator, "app_state") else None
                        if agent_memory_service:
                            try:
                                draft_report = await agent_memory_service.file_manager.read_file("draft_report.md")
                                if len(draft_report) > 500:
                                    logger.info("Sending draft_report as final research result",
                                               draft_length=len(draft_report),
                                               note="Draft report is the structured research result with chapters")
                                    stream_generator.emit_status("Finalizing report...", step="report")
                                    # CRITICAL: Send answer in chunks (same as deep research - 10000 chars per chunk)
                                    for chunk in _chunk_text(draft_report, size=10000):
                                        stream_generator.emit_report_chunk(chunk)
                                        await asyncio.sleep(0.02)
                                    stream_generator.emit_final_report(draft_report)
                                    _store_session_report(app_state, session_id, draft_report, message, mode)
                                else:
                                    logger.warning("Draft report too short, trying fallback",
                                                 draft_length=len(draft_report))
                                    # Fallback to findings if draft_report is too short
                                    if isinstance(final_state, dict):
                                        findings = final_state.get("findings", final_state.get("agent_findings", []))
                                        if findings:
                                            findings_text = "\n\n".join([
                                                f"## {f.get('topic', 'Unknown')}\n\n{f.get('summary', '')}\n\n"
                                                for f in findings
                                            ])
                                            fallback_report = f"# Research Report: {research_query}\n\n## Findings\n\n{findings_text}"
                                            stream_generator.emit_final_report(fallback_report)
                                            _store_session_report(app_state, session_id, fallback_report, message, mode)
                            except Exception as e:
                                logger.warning("Could not read draft_report", error=str(e))
                    else:
                        # No final_report - check if this is expected (research starting, waiting for clarification, etc.)
                        # Only emit error if research actually completed but no report was generated
                        
                        # Check if research completed (session status should be "completed" if research finished)
                        if session_status_from_db == "completed":
                            # Research completed but no final_report - try fallback
                            logger.warning("Research completed but no final_report, trying fallback sources",
                                         session_id=session_id)
                            fallback_report = None
                            agent_memory_service = stream_generator.app_state.get("agent_memory_service") if hasattr(stream_generator, "app_state") else None
                            if agent_memory_service:
                                try:
                                    draft_report = await agent_memory_service.file_manager.read_file("draft_report.md")
                                    if len(draft_report) > 500:
                                        fallback_report = draft_report
                                        logger.info("Using draft_report.md as fallback report", length=len(draft_report))
                                except Exception as e:
                                    logger.warning("Could not read draft_report", error=str(e))
                            
                            if not fallback_report and isinstance(final_state, dict):
                                findings = final_state.get("findings", final_state.get("agent_findings", []))
                                if findings:
                                    findings_text = "\n\n".join([
                                        f"## {f.get('topic', 'Unknown')}\n\n{f.get('summary', '')}\n\n"
                                        for f in findings
                                    ])
                                    fallback_report = f"# Research Report: {research_query}\n\n## Findings\n\n{findings_text}"
                                    logger.info("Using findings as fallback report", findings_count=len(findings))
                            
                            if fallback_report:
                                stream_generator.emit_status("Finalizing report...", step="report")
                                # CRITICAL: Send answer in chunks (same as deep research - 10000 chars per chunk)
                                for chunk in _chunk_text(fallback_report, size=10000):
                                    stream_generator.emit_report_chunk(chunk)
                                    await asyncio.sleep(0.02)
                                stream_generator.emit_final_report(fallback_report)
                                _store_session_report(app_state, session_id, fallback_report, message, mode)
                            else:
                                # Research completed but no report available - this is an error
                                logger.error("Research completed but no final_report and no fallback available",
                                           session_id=session_id,
                                           session_status=session_status_from_db)
                                stream_generator.emit_error("No report generated and no fallback available")
                        else:
                            # Research is not completed yet - this is normal, don't emit error
                            # Status is "active", "waiting_clarification", or "researching" - research is in progress
                            logger.info("No final_report but research is in progress - this is normal, not emitting error",
                                       session_id=session_id,
                                       session_status=session_status_from_db,
                                       is_new_session=is_new_session)
                            # Don't emit error - research is still running or waiting for user input
                            # The graph will emit results via stream as they become available
                else:
                    await stream_generator.emit_error(f"Unknown mode: {mode}")

            except asyncio.CancelledError:
                logger.info("Chat processing cancelled", sid=sid)
                await stream_generator.emit_error("Request cancelled")
            except Exception as e:
                logger.error("Error processing chat", sid=sid, error=str(e), exc_info=True)
                await stream_generator.emit_error(str(e))
            finally:
                try:
                    await stream_generator.emit_done()
                except Exception:
                    pass
                if session_agent_dir:
                    memory_root = Path(app_state.memory_manager.memory_dir)
                    cleanup_agent_session_dir(memory_root, session_agent_dir)
                # Clean up session
                if sid in active_sessions:
                    del active_sessions[sid]

        # Store task in active sessions
        task = asyncio.create_task(process_chat())
        active_sessions[sid] = {
            'task': task,
            'chat_id': chat_id,
            'mode': mode,
        }

        return {'success': True}

    except Exception as e:
        logger.error("Error in chat:send handler", sid=sid, error=str(e), exc_info=True)
        return {'error': str(e)}


@sio.on('chat:cancel')
async def handle_chat_cancel(sid: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle cancel request."""
    try:
        session_id = data.get('sessionId', sid)

        if session_id in active_sessions:
            session = active_sessions[session_id]
            if 'task' in session and not session['task'].done():
                session['task'].cancel()
                logger.info("Cancelled chat session", session_id=session_id)
                return {'success': True}

        return {'error': 'Session not found or already completed'}

    except Exception as e:
        logger.error("Error cancelling chat", sid=sid, error=str(e))
        return {'error': str(e)}


def get_sio() -> socketio.AsyncServer:
    """Get Socket.IO server instance."""
    return sio


async def _emit_answer(stream_generator: SocketIOStreamingGenerator, answer: str) -> None:
    if not answer:
        await stream_generator.emit_error("No answer generated")
        return

    # CRITICAL: Log formatting before emitting
    import re
    newline_count = answer.count("\n")
    has_newlines = "\n" in answer
    has_markdown = bool(re.search(r'^#{2,}\s+', answer, re.MULTILINE))
    logger.info("Emitting answer via SocketIO", 
               answer_length=len(answer),
               newline_count=newline_count,
               has_newlines=has_newlines,
               has_markdown_headings=has_markdown,
               answer_preview=answer[:200])

    await stream_generator.emit_status("Finalizing answer...", step="answer")
    # CRITICAL: Send answer in chunks (same as deep research - 10000 chars per chunk)
    # This preserves markdown structure and ensures smooth streaming
    # CRITICAL: _chunk_text preserves all formatting including \n
    for chunk in _chunk_text(answer, size=10000):
        await stream_generator.emit_report_chunk(chunk)
        await asyncio.sleep(0.02)
    # CRITICAL: emit_final_report preserves all formatting including \n
    await stream_generator.emit_final_report(answer)


def _chunk_text(text: str, size: int = 10000) -> list[str]:
    """
    Split text into chunks (same as deep research - 10000 chars per chunk).
    This preserves markdown structure and ensures smooth streaming.
    
    CRITICAL: Preserves all formatting including newlines - chunks are just slices of original text.
    """
    # CRITICAL: Simple slicing preserves all formatting including \n
    # No processing that could lose formatting
    chunks = [text[i : i + size] for i in range(0, len(text), size)]
    return chunks

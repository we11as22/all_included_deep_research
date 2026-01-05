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
        async def process_chat():
            session_agent_dir: Path | None = None
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

                # Determine original query (first user message)
                original_query = message
                if chat_history:
                    for msg in chat_history:
                        if msg['role'] == 'user':
                            original_query = msg['content']
                            break

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
                    await _emit_answer(stream_generator, result.answer)
                elif mode in ['deep_search', 'balanced']:
                    # Deep search with multiple iterations
                    result = await chat_service.answer_deep(
                        query=message,
                        stream=stream_generator,
                        messages=chat_history,
                    )
                    await _emit_answer(stream_generator, result.answer)
                elif mode in ['deep_research', 'quality']:
                    # Full research workflow using LangGraph
                    from src.workflow.research import run_research_graph
                    from src.config.modes import ResearchMode

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

                    final_state = await run_research_graph(
                        query=message,
                        chat_history=chat_history,
                        mode="quality",
                        llm=app_state.research_llm,
                        search_provider=app_state.chat_service.search_provider,
                        scraper=app_state.chat_service.scraper,
                        stream=stream_generator,
                        session_id=session_id,
                        mode_config=mode_config,
                        settings=app_state.settings,
                    )

                    # Check if graph stopped waiting for user clarification
                    if isinstance(final_state, dict):
                        clarification_needed = final_state.get("clarification_needed", False)
                        clarification_waiting = final_state.get("clarification_waiting", False)
                        should_stop = final_state.get("should_stop", False)

                        if clarification_needed or clarification_waiting or should_stop:
                            stream_generator.emit_status("Waiting for your clarification answers...", step="clarification")
                            await asyncio.sleep(0.1)
                            await stream_generator.emit_done()
                            return

                    final_report_raw = None
                    try:
                        final_report_raw = (
                            final_state.get("final_report", "")
                            if isinstance(final_state, dict)
                            else getattr(final_state, "final_report", "")
                        )
                    except Exception:
                        final_report_raw = ""

                    if isinstance(final_report_raw, dict) and "value" in final_report_raw:
                        final_report = final_report_raw["value"]
                    elif isinstance(final_report_raw, dict) and "content" in final_report_raw:
                        final_report = final_report_raw["content"]
                    elif isinstance(final_report_raw, str):
                        final_report = final_report_raw
                    else:
                        final_report = str(final_report_raw) if final_report_raw else ""

                    if final_report:
                        stream_generator.emit_status("Finalizing report...", step="report")
                        for chunk in _chunk_text(final_report, size=200):
                            stream_generator.emit_report_chunk(chunk)
                            await asyncio.sleep(0.02)
                        stream_generator.emit_final_report(final_report)
                        _store_session_report(app_state, session_id, final_report, message, mode)
                    else:
                        stream_generator.emit_error("No report generated")
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

    await stream_generator.emit_status("Finalizing answer...", step="answer")
    for chunk in _chunk_text(answer, size=180):
        await stream_generator.emit_report_chunk(chunk)
        await asyncio.sleep(0.02)
    await stream_generator.emit_final_report(answer)


def _chunk_text(text: str, size: int = 180) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]

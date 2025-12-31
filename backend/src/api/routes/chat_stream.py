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

    query = user_messages[-1].content
    raw_mode = request.model or "search"
    normalized = raw_mode.lower().replace("-", "_")
    settings = app_request.app.state.settings
    chat_history = _collect_chat_history(request.messages, settings.chat_history_limit)

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
    # Pass app state to stream generator
    app_state = {
        "debug_mode": bool(getattr(app_request.app.state, "settings", None) and app_request.app.state.settings.debug_mode),
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
                stream_generator.app_state["agent_memory_service"] = agent_memory_service
                stream_generator.app_state["agent_file_service"] = agent_file_service

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
                chat_service = app_request.app.state.chat_service
                if mode == "search":
                    result = await chat_service.answer_web(query, stream=stream_generator, messages=chat_history)
                else:
                    result = await chat_service.answer_deep(query, stream=stream_generator, messages=chat_history)

                # Emit final answer
                if result.answer:
                    stream_generator.emit_status("Finalizing answer...", step="answer")
                    for chunk in _chunk_text(result.answer, size=180):
                        stream_generator.emit_report_chunk(chunk)
                        await asyncio.sleep(0.02)
                    stream_generator.emit_final_report(result.answer)
                else:
                    stream_generator.emit_error(error="No answer generated", details="Search completed but no answer was generated")
                
                # Store report for PDF generation
                _store_session_report(app_request.app.state, session_id, result.answer, query, mode)
                
                stream_generator.emit_done()
                return

            workflow_factory = app_request.app.state.workflow_factory
            workflow = workflow_factory.create_workflow("quality")
            final_state = await workflow.run(
                query,
                stream=stream_generator,
                messages=chat_history,
                suppress_final_report_stream=True,
            )

            # Extract final_report - handle both dict with override and direct value
            final_report_raw = final_state.get("final_report", "") if isinstance(final_state, dict) else getattr(final_state, "final_report", "")
            if isinstance(final_report_raw, dict) and "value" in final_report_raw:
                final_report = final_report_raw["value"]
            elif isinstance(final_report_raw, str):
                final_report = final_report_raw
            else:
                final_report = str(final_report_raw) if final_report_raw else ""
            
            if final_report:
                # Emit final report chunks and final report event
                stream_generator.emit_status("Generating final report...", step="report")
                for chunk in _chunk_text(final_report, size=200):
                    stream_generator.emit_report_chunk(chunk)
                    await asyncio.sleep(0.02)
                stream_generator.emit_final_report(final_report)
                
                # Store final report in session for PDF generation
                _store_session_report(app_request.app.state, session_id, final_report, query, mode)
                
            stream_generator.emit_done()

        except asyncio.CancelledError:
            logger.info("Chat task cancelled", session_id=session_id)
            stream_generator.emit_status("Chat cancelled by user", step="cancelled")
            stream_generator.emit_done()
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
        finally:
            # Remove task from active tasks
            app_request.app.state.active_tasks.pop(session_id, None)
            if session_agent_dir:
                memory_root = Path(app_request.app.state.memory_manager.memory_dir)
                cleanup_agent_session_dir(memory_root, session_agent_dir)

    # Start background task and store it
    task = asyncio.create_task(run_task())
    app_request.app.state.active_tasks[session_id] = task

    async def watch_disconnect() -> None:
        try:
            while not task.done():
                if await app_request.is_disconnected():
                    if not task.done():
                        task.cancel()
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
    logger.info("Chat session cancelled", session_id=session_id)
    
    return {"status": "cancelled", "session_id": session_id}


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

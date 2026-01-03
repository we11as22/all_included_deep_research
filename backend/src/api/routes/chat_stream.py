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
                        logger.debug("Emitting final answer", answer_length=len(result.answer))
                        stream_generator.emit_status("Finalizing answer...", step="answer")
                        for chunk in _chunk_text(result.answer, size=180):
                            stream_generator.emit_report_chunk(chunk)
                            await asyncio.sleep(0.02)
                        stream_generator.emit_final_report(result.answer)
                        logger.debug("Final report emitted")
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
                    if clarification_needed and not is_continuation:
                        # Graph stopped waiting for user - don't emit final report yet
                        logger.info("Graph stopped waiting for user clarification - not emitting final report")
                        # Emit status that we're waiting
                        stream_generator.emit_status("Waiting for your clarification answers...", step="clarification")
                        stream_generator.emit_done()
                        return

                # Extract final_report - handle both dict with override and direct value
                logger.info("Extracting final report from state", state_keys=list(final_state.keys()) if isinstance(final_state, dict) else "not a dict")
                final_report_raw = final_state.get("final_report", "") if isinstance(final_state, dict) else getattr(final_state, "final_report", "")
                logger.info("Final report raw", report_type=type(final_report_raw).__name__, has_value=bool(final_report_raw))
                
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
                stream_generator.emit_error(error=str(e), details="Research workflow encountered an error")
                # Try to extract any partial results
                if "final_state" in locals():
                    final_report_raw = final_state.get("final_report", "") if isinstance(final_state, dict) else getattr(final_state, "final_report", "")
                    if final_report_raw:
                        final_report = str(final_report_raw)
                        stream_generator.emit_final_report(final_report)
                        _store_session_report(app_request.app.state, session_id, final_report, query, mode)
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
            # DON'T cleanup session dir here - only cleanup on cancellation or error
            # This allows successful sessions to keep their files for debugging
            logger.debug("Task finished, session files preserved", session_id=session_id, preserved=session_agent_dir is not None)

    # Start background task and store it
    task = asyncio.create_task(run_task())
    app_request.app.state.active_tasks[session_id] = task

    async def watch_disconnect() -> None:
        """
        Watch for client disconnection, but DON'T cancel deep_research tasks.
        Deep research can take a long time and should complete even if client disconnects.
        Client can reconnect and get the result.
        """
        try:
            while not task.done():
                if await app_request.is_disconnected():
                    # For deep_research mode, log but don't cancel - let it complete
                    if mode == "deep_research":
                        logger.info("Client disconnected during deep_research, but continuing task to completion", session_id=session_id)
                        # Don't cancel - let the research complete
                        # Client can reconnect and get the result
                    else:
                        # For other modes, cancel on disconnect
                        logger.warning("Client disconnected, cancelling task", session_id=session_id)
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

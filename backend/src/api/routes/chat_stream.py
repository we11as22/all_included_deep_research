"""Chat streaming endpoint with progress events."""

import asyncio
from uuid import uuid4

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.api.models.chat import ChatCompletionRequest
from src.streaming.sse import ResearchStreamingGenerator

router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = structlog.get_logger(__name__)


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

    if normalized in {"speed"}:
        mode = "search"
    elif normalized in {"balanced"}:
        mode = "deep_search"
    elif normalized in {"quality"}:
        mode = "deep_research"
    elif normalized in {"search", "simple", "web", "web_search"}:
        mode = "search"
    elif normalized in {"deep_search", "deep"}:
        mode = "deep_search"
    elif normalized in {"deep_research", "research"}:
        mode = "deep_research"
    else:
        mode = "deep_search"

    logger.info("Chat stream request", mode=mode, query=query[:100])

    session_id = str(uuid4())
    stream_generator = ResearchStreamingGenerator(session_id=session_id)

    async def run_task():
        try:
            stream_generator.emit_init(mode=mode)
            stream_generator.emit_status("Starting chat workflow...", step="init")

            if mode in {"search", "deep_search"}:
                chat_service = app_request.app.state.chat_service
                if mode == "search":
                    result = await chat_service.answer_web(query, stream=stream_generator, messages=chat_history)
                else:
                    result = await chat_service.answer_deep(query, stream=stream_generator, messages=chat_history)

                stream_generator.emit_status("Drafting answer...", step="answer")
                for chunk in _chunk_text(result.answer, size=180):
                    stream_generator.emit_report_chunk(chunk)
                    await asyncio.sleep(0.02)
                stream_generator.emit_final_report(result.answer)
                stream_generator.emit_done()
                return

            workflow_factory = app_request.app.state.workflow_factory
            workflow = workflow_factory.create_workflow("quality")
            final_state = await workflow.run(query, stream=stream_generator, messages=chat_history)

            final_report = final_state.get("final_report", "")
            if final_report:
                stream_generator.emit_status("Saving research to memory...", step="save_memory")
                try:
                    memory_manager = app_request.app.state.memory_manager
                    file_path = _generate_memory_path(query)
                    content = _format_memory_report(query, final_report)
                    await memory_manager.create_file(
                        file_path=file_path,
                        title=query[:80],
                        content=content,
                    )
                    await memory_manager.sync_file_to_db(file_path)
                except Exception as exc:
                    stream_generator.emit_error(error=str(exc), details="Memory save failed")
            stream_generator.emit_done()

        except Exception as exc:
            logger.error("Chat stream failed", error=str(exc), exc_info=True)
            stream_generator.emit_error(error=str(exc), details="Chat stream failed")
            stream_generator.emit_done()

    asyncio.create_task(run_task())

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


def _generate_memory_path(query: str) -> str:
    safe = "".join(ch for ch in query.lower() if ch.isalnum() or ch in (" ", "_", "-")).strip()
    slug = "_".join(safe.split())[:60] or "research"
    return f"conversations/{slug}.md"


def _format_memory_report(query: str, report: str) -> str:
    return f"# {query}\n\n{report}\n"


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

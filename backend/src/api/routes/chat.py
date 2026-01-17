"""OpenAI-compatible chat completion endpoint."""

import asyncio
from uuid import uuid4

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.api.models.chat import ChatCompletionRequest
from src.streaming.sse import OpenAIStreamingGenerator

router = APIRouter(prefix="/v1", tags=["chat"])
logger = structlog.get_logger(__name__)


@router.post("/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, app_request: Request):
    """
    OpenAI-compatible chat completion endpoint.

    Supports streaming mode for research workflows.
    Model field specifies chat mode: search, deep_search, deep_research (legacy: speed, balanced, quality).
    """
    if not request.stream:
        raise HTTPException(status_code=501, detail="Only streaming responses are supported. Set 'stream=true'")

    # Extract user query from messages
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

    logger.info("Chat completion request", mode=mode, query=query[:100])

    try:
        # Create streaming generator
        stream_generator = OpenAIStreamingGenerator(model=f"all-included-{mode}")
        session_id = str(uuid4())

        # Start workflow in background
        async def run_task():
            try:
                if mode in {"search", "deep_search"}:
                    chat_service = app_request.app.state.chat_service
                    if mode == "search":
                        result = await chat_service.answer_web(query, messages=chat_history)
                    else:
                        result = await chat_service.answer_deep(query, messages=chat_history)

                    # Sources already included in answer by ChatService with inline citations
                    answer = result.answer
                    # CRITICAL: Send answer in chunks (same as deep research - 10000 chars per chunk)
                    for chunk in _chunk_text(answer, size=10000):
                        stream_generator.add_chunk_from_str(chunk)
                        await asyncio.sleep(0.02)
                    stream_generator.finish()
                    return

                # deep_research mode uses full workflow
                workflow_factory = app_request.app.state.workflow_factory
                workflow = workflow_factory.create_workflow("quality")
                final_state = await workflow.run(query, messages=chat_history)
                final_report = final_state.get("final_report", "") if isinstance(final_state, dict) else getattr(final_state, "final_report", "")
                if final_report:
                    # CRITICAL: Send answer in chunks (same as deep research - 10000 chars per chunk)
                    for chunk in _chunk_text(final_report, size=10000):
                        stream_generator.add_chunk_from_str(chunk)
                        await asyncio.sleep(0.05)
                stream_generator.finish()

            except Exception as e:
                logger.error("Research workflow failed", error=str(e), exc_info=True)
                stream_generator.add_chunk_from_str(f"\n\nError: {str(e)}")
                stream_generator.finish(finish_reason="error")

        # Start background task
        asyncio.create_task(run_task())

        # Return streaming response
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

    except Exception as e:
        logger.error("Error creating chat completion", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _chunk_text(text: str, size: int = 10000) -> list[str]:
    """
    Split text into chunks (same as deep research - 10000 chars per chunk).
    This preserves markdown structure and ensures smooth streaming.
    """
    return [text[i : i + size] for i in range(0, len(text), size)]


def _append_sources(answer: str, sources: list) -> str:
    if not sources:
        return answer

    lines = [answer.strip(), "\nSources:"]
    for idx, source in enumerate(sources, 1):
        lines.append(f"[{idx}] {source.title} - {source.url}")
    return "\n".join(lines)


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

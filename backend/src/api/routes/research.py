"""Research endpoint with structured events."""

import asyncio
from pathlib import Path
from uuid import uuid4

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.api.models.research import ResearchRequest, ResearchResponse
from src.streaming.sse import ResearchStreamingGenerator
from src.memory.agent_session import create_agent_session_services, cleanup_agent_session_dir

router = APIRouter(prefix="/api", tags=["research"])
logger = structlog.get_logger(__name__)


@router.post("/research")
async def start_research(research_request: ResearchRequest, app_request: Request):
    """
    Start a research session with structured streaming events.

    Returns SSE stream with research progress updates.
    """
    logger.info(
        "Research request",
        mode=research_request.mode.value,
        query=research_request.query[:100],
        save_to_memory=research_request.save_to_memory,
    )

    session_id = str(uuid4())
    # Pass app state to stream generator
    app_state = {
        "debug_mode": bool(getattr(app_request.app.state, "settings", None) and app_request.app.state.settings.debug_mode),
    }
    stream_generator = ResearchStreamingGenerator(session_id=session_id, app_state=app_state)

    async def run_research():
        session_agent_dir: Path | None = None
        try:
            # Emit initialization
            stream_generator.emit_init(mode=research_request.mode.value)
            stream_generator.emit_status("Starting research workflow...", step="init")

            if research_request.mode.value == "quality":
                memory_root = Path(app_request.app.state.memory_manager.memory_dir)
                agent_memory_service, agent_file_service, session_agent_dir = create_agent_session_services(
                    memory_root, session_id
                )
                await agent_memory_service.read_main_file()
                stream_generator.app_state["agent_memory_service"] = agent_memory_service
                stream_generator.app_state["agent_file_service"] = agent_file_service

            # Get workflow from app state
            workflow_factory = app_request.app.state.workflow_factory
            workflow = workflow_factory.create_workflow(research_request.mode.value)

            # Run workflow with streaming
            final_state = await workflow.run(research_request.query, stream=stream_generator)

            # Memory persistence disabled; ephemeral agent memory only.
            final_report = final_state.get("final_report", "") if isinstance(final_state, dict) else getattr(final_state, "final_report", "")

            stream_generator.emit_status("Research completed!", step="done")
            stream_generator.emit_done()

        except asyncio.CancelledError:
            logger.info("Research task cancelled", session_id=session_id)
            stream_generator.emit_status("Research cancelled by user", step="cancelled")
            stream_generator.emit_done()
        except Exception as e:
            logger.error("Research workflow failed", error=str(e), exc_info=True)
            stream_generator.emit_error(error=str(e), details="Workflow execution failed")
            stream_generator.emit_done()
        finally:
            # Remove task from active tasks
            app_request.app.state.active_tasks.pop(session_id, None)
            if session_agent_dir:
                memory_root = Path(app_request.app.state.memory_manager.memory_dir)
                cleanup_agent_session_dir(memory_root, session_agent_dir)

    # Start background task and store it
    task = asyncio.create_task(run_research())
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

    # Return streaming response
    return StreamingResponse(
        stream_generator.stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-ID": session_id,
            "X-Research-Mode": research_request.mode.value,
        },
    )


def _generate_memory_path(query: str) -> str:
    safe = "".join(ch for ch in query.lower() if ch.isalnum() or ch in (" ", "_", "-")).strip()
    slug = "_".join(safe.split())[:60] or "research"
    return f"conversations/{slug}.md"


def _format_memory_report(query: str, report: str) -> str:
    return f"# {query}\n\n{report}\n"


@router.post("/research/{session_id}/cancel")
async def cancel_research(session_id: str, app_request: Request):
    """Cancel an active research session."""
    active_tasks = app_request.app.state.active_tasks
    
    if session_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Research session not found or already completed")
    
    task = active_tasks[session_id]
    if task.done():
        active_tasks.pop(session_id, None)
        raise HTTPException(status_code=400, detail="Research session already completed")
    
    task.cancel()
    active_tasks.pop(session_id, None)
    logger.info("Research session cancelled", session_id=session_id)
    
    return {"status": "cancelled", "session_id": session_id}


@router.get("/research/{session_id}", response_model=ResearchResponse)
async def get_research_status(session_id: str, app_request: Request):
    """Get research session status."""
    active_tasks = app_request.app.state.active_tasks
    
    if session_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Research session not found")
    
    task = active_tasks[session_id]
    status = "running" if not task.done() else ("completed" if task.exception() is None else "failed")
    
    return ResearchResponse(
        session_id=session_id,
        status=status,
        mode="unknown",
    )

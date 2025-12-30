"""Research endpoint with structured events."""

import asyncio
from uuid import uuid4

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.api.models.research import ResearchRequest, ResearchResponse
from src.streaming.sse import ResearchStreamingGenerator

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
    # Pass app state to stream generator for agent memory service
    app_state = {
        "agent_memory_service": app_request.app.state.agent_memory_service,
        "agent_file_service": app_request.app.state.agent_file_service,
    }
    stream_generator = ResearchStreamingGenerator(session_id=session_id, app_state=app_state)

    async def run_research():
        try:
            # Emit initialization
            stream_generator.emit_init(mode=research_request.mode.value)
            stream_generator.emit_status("Starting research workflow...", step="init")

            # Get workflow from app state
            workflow_factory = app_request.app.state.workflow_factory
            workflow = workflow_factory.create_workflow(research_request.mode.value)

            # Run workflow with streaming
            final_state = await workflow.run(research_request.query, stream=stream_generator)

            # Save to memory if requested
            final_report = final_state.get("final_report", "")
            if research_request.save_to_memory and final_report:
                stream_generator.emit_status("Saving research to memory...", step="save_memory")
                try:
                    memory_manager = app_request.app.state.memory_manager
                    file_path = _generate_memory_path(research_request.query)
                    content = _format_memory_report(research_request.query, final_report)
                    await memory_manager.create_file(
                        file_path=file_path,
                        title=research_request.query[:80],
                        content=content,
                    )
                    await memory_manager.sync_file_to_db(file_path)
                except Exception as exc:
                    stream_generator.emit_error(error=str(exc), details="Memory save failed")

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

    # Start background task and store it
    task = asyncio.create_task(run_research())
    app_request.app.state.active_tasks[session_id] = task

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

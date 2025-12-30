"""Memory system endpoints."""

import structlog
from fastapi import APIRouter, HTTPException, Request

from src.api.models.memory import (
    MemoryCreateRequest,
    MemoryFileResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    MemorySearchResult,
)

router = APIRouter(prefix="/api", tags=["memory"])
logger = structlog.get_logger(__name__)


@router.post("/memory/search", response_model=MemorySearchResponse)
async def search_memory(search_request: MemorySearchRequest, app_request: Request):
    """
    Search memory for relevant content.

    Uses hybrid search (vector + fulltext) with RRF fusion.
    """
    logger.info("Memory search request", query=search_request.query[:100])

    try:
        search_engine = app_request.app.state.search_engine

        results = await search_engine.hybrid_search(
            query=search_request.query,
            limit=search_request.limit,
            rrf_k=60,
        )

        # Filter by min score
        filtered_results = [r for r in results if r.score >= search_request.min_score]

        # Convert to response model
        search_results = [
            MemorySearchResult(
                chunk_id=r.chunk_id,
                file_path=r.file_path,
                file_title=r.file_title,
                content=r.content,
                score=r.score,
                header_path=r.header_path,
            )
            for r in filtered_results
        ]

        logger.info("Memory search completed", results_count=len(search_results))

        return MemorySearchResponse(
            query=search_request.query,
            results=search_results,
            total=len(search_results),
        )

    except Exception as e:
        logger.error("Memory search failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Memory search failed: {str(e)}")


@router.post("/memory", response_model=MemoryFileResponse)
async def create_memory_file(memory_request: MemoryCreateRequest, app_request: Request):
    """
    Create new memory file.

    Creates markdown file and syncs to database.
    """
    logger.info("Create memory file request", file_path=memory_request.file_path)

    try:
        memory_manager = app_request.app.state.memory_manager

        # Create file
        await memory_manager.create_file(
            file_path=memory_request.file_path,
            title=memory_request.title,
            content=memory_request.content,
            tags=memory_request.tags,
        )

        # Sync to database
        embedding_dimension = getattr(app_request.app.state, "embedding_dimension", 1536)
        file_id = await memory_manager.sync_file_to_db(memory_request.file_path, embedding_dimension=embedding_dimension)
        file_record = await memory_manager.get_file_by_path(memory_request.file_path)

        logger.info("Memory file created", file_path=memory_request.file_path)

        return MemoryFileResponse(
            file_id=file_id or (file_record.id if file_record else 0),
            file_path=memory_request.file_path,
            title=memory_request.title,
            chunks_count=0,
            created_at=file_record.created_at.isoformat() if file_record else "",
            updated_at=file_record.updated_at.isoformat() if file_record else "",
        )

    except Exception as e:
        logger.error("Create memory file failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create memory file: {str(e)}")


@router.get("/memory/files", response_model=list[MemoryFileResponse])
async def list_memory_files(app_request: Request):
    """List all memory files."""
    try:
        memory_manager = app_request.app.state.memory_manager
        files = await memory_manager.list_files()

        return [
            MemoryFileResponse(
                file_id=f["file_id"],
                file_path=f["file_path"],
                title=f["title"],
                chunks_count=f.get("chunks_count", 0),
                created_at=f.get("created_at", ""),
                updated_at=f.get("updated_at", ""),
            )
            for f in files
        ]

    except Exception as e:
        logger.error("List memory files failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list memory files: {str(e)}")


@router.delete("/memory/{file_path:path}")
async def delete_memory_file(file_path: str, app_request: Request):
    """Delete memory file."""
    logger.info("Delete memory file request", file_path=file_path)

    try:
        memory_manager = app_request.app.state.memory_manager
        await memory_manager.delete_file(file_path)

        logger.info("Memory file deleted", file_path=file_path)

        return {"status": "deleted", "file_path": file_path}

    except Exception as e:
        logger.error("Delete memory file failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete memory file: {str(e)}")

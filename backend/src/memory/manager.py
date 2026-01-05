"""Memory manager for multi-file markdown storage and sync."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy.orm import sessionmaker

from src.embeddings.base import EmbeddingProvider
from src.memory.chunking import MarkdownChunker
from src.memory.file_manager import FileManager
from src.memory.index_manager import IndexManager, JsonIndexManager
from src.memory.models.memory import MemoryFile
from src.memory.repository import MemoryRepository
from src.memory.sync_service import FileSyncService

logger = structlog.get_logger(__name__)


class MemoryManager:
    """Coordinates file storage, indexing, and database sync."""

    def __init__(
        self,
        memory_dir: str,
        session_factory: sessionmaker,
        embedding_provider: EmbeddingProvider,
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        embedding_batch_size: int = 100,
    ) -> None:
        self.memory_dir = Path(memory_dir)
        self.session_factory = session_factory
        self.embedding_provider = embedding_provider
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_batch_size = embedding_batch_size

        self.file_manager = FileManager(str(self.memory_dir))
        self.chunker = MarkdownChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        self._initialize_structure()

    def _initialize_structure(self) -> None:
        """Ensure base folder structure exists.
        
        CRITICAL: Do NOT create main.md or files_index.json in root memory_dir.
        These files should only be created in agent session subdirectories during deep_research.
        """
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # Only create agent_sessions directory for agent memory
        # Agent sessions are created separately via create_agent_session_services()
        # Each session has: main.md, draft_report.md, agents/, items/, files_index.json
        (self.memory_dir / "agent_sessions").mkdir(exist_ok=True)

        # DO NOT create main.md or files_index.json in root - they are created per-session
        # Initialize index managers as None - they will be created per-session if needed
        self.index_manager = None
        self.json_index_manager = None

    def _category_from_path(self, file_path: str) -> str:
        """
        Extract category from file path.
        
        For session files:
        - agents/{agent_id}.md -> agent
        - items/{item}.md -> item
        - draft_report.md -> report
        - main.md -> main
        """
        parts = file_path.split("/")
        if len(parts) > 1:
            category = parts[0]
            # Map session subdirectories
            if category == "agents":
                return "agent"
            elif category == "items":
                return "item"
            elif category == "draft_report.md" or file_path.endswith("draft_report.md"):
                return "report"
            elif category == "main.md" or file_path.endswith("main.md"):
                return "main"
        return "other"

    async def create_file(
        self,
        file_path: str,
        title: str,
        content: str,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new memory file on disk and update indexes."""
        await self.file_manager.write_file(file_path, content)
        tags = tags or []

        category = self._category_from_path(file_path)
        description = title.strip() if title.strip() else Path(file_path).stem.replace("_", " ").title()

        # Index managers are only used for deep research sessions (per-session)
        # They are None for root memory_dir - skip indexing here
        if self.index_manager is not None:
            self.index_manager.update_file_index(file_path, description, category)
            self.index_manager.touch_updated_at()
        if self.json_index_manager is not None:
            self.json_index_manager.upsert_file(
                {
                    "file_path": file_path,
                    "title": description,
                    "category": category,
                    "tags": tags,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        logger.info("memory_file_created", file_path=file_path)
        return {"file_path": file_path, "title": description}

    async def sync_file_to_db(self, file_path: str, force: bool = False, embedding_dimension: int | None = None) -> int:
        """Sync a single file to the database."""
        async with self.session_factory() as session:
            # Use database schema dimension if not provided
            if embedding_dimension is None:
                from src.database.schema import EMBEDDING_DIMENSION
                embedding_dimension = EMBEDDING_DIMENSION
            
            sync_service = FileSyncService(
                session=session,
                file_manager=self.file_manager,
                chunker=self.chunker,
                embedding_provider=self.embedding_provider,
                batch_size=self.embedding_batch_size,
                embedding_dimension=embedding_dimension,
            )
            file_id = await sync_service.sync_file(file_path=file_path, force=force)
            await session.commit()
            return file_id

    async def list_files(self) -> list[dict[str, Any]]:
        """List files from database metadata."""
        async with self.session_factory() as session:
            repository = MemoryRepository(session)
            files = await repository.list_files()
            return [self._serialize_file(file) for file in files]

    async def get_file_by_path(self, file_path: str) -> MemoryFile | None:
        """Fetch a single file metadata by path."""
        async with self.session_factory() as session:
            repository = MemoryRepository(session)
            return await repository.get_file_by_path(file_path)

    async def delete_file(self, file_path: str) -> None:
        """Delete a file from disk and database."""
        await self.file_manager.delete_file(file_path)
        # Index managers are only used for deep research sessions (per-session)
        if self.json_index_manager is not None:
            self.json_index_manager.remove_file(file_path)
        if self.index_manager is not None:
            self.index_manager.touch_updated_at()

        async with self.session_factory() as session:
            repository = MemoryRepository(session)
            existing = await repository.get_file_by_path(file_path)
            if existing:
                await repository.delete_file(existing.id)
                await session.commit()

        logger.info("memory_file_deleted", file_path=file_path)

    def _serialize_file(self, file: MemoryFile) -> dict[str, Any]:
        return {
            "file_id": file.id,
            "file_path": file.file_path,
            "title": file.title,
            "chunks_count": 0,
            "created_at": file.created_at.isoformat(),
            "updated_at": file.updated_at.isoformat(),
        }

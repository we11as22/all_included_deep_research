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
        """Ensure base folder structure and index files exist."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        for subdir in ["projects", "concepts", "conversations", "preferences", "items", "agents"]:
            (self.memory_dir / subdir).mkdir(exist_ok=True)

        main_file = self.memory_dir / "main.md"
        if not main_file.exists():
            main_file.write_text(self._default_main_content(), encoding="utf-8")
            logger.info("main_file_created", path=str(main_file))

        json_index = self.memory_dir / "files_index.json"
        if not json_index.exists():
            json_index.write_text(
                json.dumps(
                    {
                        "version": "1.0",
                        "last_updated": datetime.now(timezone.utc).isoformat(),
                        "files": [],
                    },
                    indent=2,
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )
            logger.info("json_index_created", path=str(json_index))

        self.index_manager = IndexManager(main_file)
        self.json_index_manager = JsonIndexManager(json_index)

    def _default_main_content(self) -> str:
        """Template for main.md."""
        return """# Agent Memory - Main Notes

Last Updated: 2025-01-01

## File Index

This section maintains an index of all specialized memory files with descriptions.

### Projects
<!-- Add project files here -->

### Concepts
<!-- Add concept files here -->

### Conversations
<!-- Add conversation files here -->

### Preferences
<!-- Add preference files here -->

### Other
<!-- Add other files here -->

---

## Current Goals

<!-- Active goals -->

---

## Completed Tasks

<!-- Completed tasks -->

---

## Future Plans

<!-- Long-term plans -->

---

## Plans

<!-- Detailed plans -->

---

## Recent Notes

<!-- Recent session notes -->

---

## Quick Reference

<!-- Frequently needed info -->
"""

    def _category_from_path(self, file_path: str) -> str:
        parts = file_path.split("/")
        if len(parts) > 1:
            category = parts[0]
            # Handle special case: chat_{id} -> chat (old format)
            if category.startswith("chat_"):
                return "chat"
            # Handle conversations folder: conversations/chat_{id}/... -> chat
            if category == "conversations" and len(parts) > 1:
                # Check if second part starts with chat_
                if parts[1].startswith("chat_"):
                    return "chat"
            # Remove trailing 's' for plural forms (conversations -> conversation, etc.)
            return category.rstrip("s")
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

        self.index_manager.update_file_index(file_path, description, category)
        self.index_manager.touch_updated_at()
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
        self.json_index_manager.remove_file(file_path)
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

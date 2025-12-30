"""Memory repository for PostgreSQL operations."""

from typing import Any

import asyncpg
import structlog
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.schema import MemoryChunkModel, MemoryFileModel
from src.memory.models.chunk import Chunk, ChunkCreate
from src.memory.models.memory import MemoryFile, MemoryFileCreate, MemoryFileUpdate

logger = structlog.get_logger(__name__)


class MemoryRepository:
    """Repository for memory operations with PostgreSQL."""

    def __init__(self, session: AsyncSession):
        """
        Initialize memory repository.

        Args:
            session: SQLAlchemy async session
        """
        self.session = session

    async def create_file(self, file_create: MemoryFileCreate) -> MemoryFile:
        """
        Create memory file.

        Args:
            file_create: File creation data

        Returns:
            Created memory file
        """
        db_file = MemoryFileModel(
            file_path=file_create.file_path,
            title=file_create.title,
            category=file_create.category.value,
            file_hash=file_create.file_hash,
            word_count=file_create.word_count,
            tags=file_create.tags,
            file_metadata=file_create.metadata,
        )

        self.session.add(db_file)
        await self.session.flush()
        await self.session.refresh(db_file)

        logger.info("Memory file created", file_id=db_file.id, file_path=file_create.file_path)

        # Ensure metadata is a dict, not MetaData object
        file_dict = {
            "id": db_file.id,
            "file_path": db_file.file_path,
            "title": db_file.title,
            "category": db_file.category,
            "tags": db_file.tags or [],
            "metadata": dict(db_file.file_metadata) if db_file.file_metadata else {},
            "created_at": db_file.created_at,
            "updated_at": db_file.updated_at,
            "file_hash": db_file.file_hash,
            "word_count": db_file.word_count,
        }
        return MemoryFile.model_validate(file_dict)

    async def get_file_by_id(self, file_id: int) -> MemoryFile | None:
        """Get memory file by ID."""
        result = await self.session.execute(select(MemoryFileModel).where(MemoryFileModel.id == file_id))
        db_file = result.scalar_one_or_none()

        if db_file:
            # Ensure metadata is a dict, not MetaData object
            file_dict = {
                "id": db_file.id,
                "file_path": db_file.file_path,
                "title": db_file.title,
                "category": db_file.category,
                "tags": db_file.tags or [],
                "metadata": dict(db_file.file_metadata) if db_file.file_metadata else {},
                "created_at": db_file.created_at,
                "updated_at": db_file.updated_at,
                "file_hash": db_file.file_hash,
                "word_count": db_file.word_count,
            }
            return MemoryFile.model_validate(file_dict)
        return None

    async def get_file_by_path(self, file_path: str) -> MemoryFile | None:
        """Get memory file by path."""
        result = await self.session.execute(select(MemoryFileModel).where(MemoryFileModel.file_path == file_path))
        db_file = result.scalar_one_or_none()

        if db_file:
            # Ensure metadata is a dict, not MetaData object
            file_dict = {
                "id": db_file.id,
                "file_path": db_file.file_path,
                "title": db_file.title,
                "category": db_file.category,
                "tags": db_file.tags or [],
                "metadata": dict(db_file.file_metadata) if db_file.file_metadata else {},
                "created_at": db_file.created_at,
                "updated_at": db_file.updated_at,
                "file_hash": db_file.file_hash,
                "word_count": db_file.word_count,
            }
            return MemoryFile.model_validate(file_dict)
        return None

    async def update_file(self, file_id: int, file_update: MemoryFileUpdate) -> MemoryFile | None:
        """Update memory file."""
        update_data = file_update.model_dump(exclude_unset=True)
        if "category" in update_data:
            update_data["category"] = update_data["category"].value

        stmt = update(MemoryFileModel).where(MemoryFileModel.id == file_id).values(**update_data).returning(MemoryFileModel)

        result = await self.session.execute(stmt)
        db_file = result.scalar_one_or_none()

        if db_file:
            logger.info("Memory file updated", file_id=file_id)
            # Ensure metadata is a dict, not MetaData object
            file_dict = {
                "id": db_file.id,
                "file_path": db_file.file_path,
                "title": db_file.title,
                "category": db_file.category,
                "tags": db_file.tags or [],
                "metadata": dict(db_file.file_metadata) if db_file.file_metadata else {},
                "created_at": db_file.created_at,
                "updated_at": db_file.updated_at,
                "file_hash": db_file.file_hash,
                "word_count": db_file.word_count,
            }
            return MemoryFile.model_validate(file_dict)
        return None

    async def delete_file(self, file_id: int) -> bool:
        """Delete memory file (cascades to chunks)."""
        result = await self.session.execute(delete(MemoryFileModel).where(MemoryFileModel.id == file_id))

        deleted = result.rowcount > 0
        if deleted:
            logger.info("Memory file deleted", file_id=file_id)
        return deleted

    async def list_files(
        self,
        category: str | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryFile]:
        """List memory files with filters."""
        query = select(MemoryFileModel)

        if category:
            query = query.where(MemoryFileModel.category == category)

        if tags:
            # Files must have ALL specified tags
            for tag in tags:
                query = query.where(MemoryFileModel.tags.contains([tag]))

        query = query.order_by(MemoryFileModel.updated_at.desc()).limit(limit).offset(offset)

        result = await self.session.execute(query)
        db_files = result.scalars().all()

        # Ensure metadata is a dict for each file
        files = []
        for db_file in db_files:
            file_dict = {
                "id": db_file.id,
                "file_path": db_file.file_path,
                "title": db_file.title,
                "category": db_file.category,
                "tags": db_file.tags or [],
                "metadata": dict(db_file.file_metadata) if db_file.file_metadata else {},
                "created_at": db_file.created_at,
                "updated_at": db_file.updated_at,
                "file_hash": db_file.file_hash,
                "word_count": db_file.word_count,
            }
            files.append(MemoryFile.model_validate(file_dict))
        return files

    async def insert_chunks(self, chunks: list[ChunkCreate]) -> list[int]:
        """Insert multiple chunks."""
        if not chunks:
            return []

        chunk_models = [
            MemoryChunkModel(
                file_id=chunk.file_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                content_hash=chunk.content_hash,
                embedding=chunk.embedding,
                header_path=chunk.header_path,
                section_level=chunk.section_level,
            )
            for chunk in chunks
        ]

        self.session.add_all(chunk_models)
        await self.session.flush()

        chunk_ids = [chunk.id for chunk in chunk_models]
        logger.info("Chunks inserted", count=len(chunk_ids), file_id=chunks[0].file_id)

        return chunk_ids

    async def delete_chunks_by_file(self, file_id: int) -> int:
        """Delete all chunks for a file."""
        result = await self.session.execute(delete(MemoryChunkModel).where(MemoryChunkModel.file_id == file_id))

        deleted_count = result.rowcount
        if deleted_count > 0:
            logger.info("Chunks deleted", file_id=file_id, count=deleted_count)

        return deleted_count

    async def get_file_hash(self, file_path: str) -> str | None:
        """Get file hash by path."""
        result = await self.session.execute(
            select(MemoryFileModel.file_hash).where(MemoryFileModel.file_path == file_path)
        )
        return result.scalar_one_or_none()

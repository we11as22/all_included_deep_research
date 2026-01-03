"""File synchronization service for memory system."""

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from src.embeddings.base import EmbeddingProvider
from src.memory.chunking import MarkdownChunker
from src.memory.file_manager import FileManager
from src.memory.models.chunk import ChunkCreate
from src.memory.models.memory import MemoryCategory, MemoryFileCreate, MemoryFileUpdate
from src.memory.repository import MemoryRepository

logger = structlog.get_logger(__name__)


class FileSyncService:
    """Synchronizes markdown files with database."""

    def __init__(
        self,
        session: AsyncSession,
        file_manager: FileManager,
        chunker: MarkdownChunker,
        embedding_provider: EmbeddingProvider,
        batch_size: int = 100,
        embedding_dimension: int | None = None,
    ):
        """
        Initialize file sync service.

        Args:
            session: SQLAlchemy async session
            file_manager: File manager instance
            chunker: Markdown chunker instance
            embedding_provider: Embedding provider
            batch_size: Batch size for embeddings
            embedding_dimension: Expected embedding dimension (if None, will get from provider)
        """
        self.repository = MemoryRepository(session)
        self.file_manager = file_manager
        self.chunker = chunker
        self.embedding_provider = embedding_provider
        self.batch_size = batch_size
        # Use database schema dimension if not provided
        if embedding_dimension is None:
            from src.database.schema import EMBEDDING_DIMENSION
            embedding_dimension = EMBEDDING_DIMENSION
        self.embedding_dimension = embedding_dimension

    async def sync_file(self, file_path: str, force: bool = False) -> int:
        """
        Sync single file to database.

        Args:
            file_path: Relative file path
            force: Force sync even if hash matches

        Returns:
            File ID

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        # Read file content
        content = await self.file_manager.read_file(file_path)
        file_hash = self.file_manager.compute_file_hash(content)

        # Check if sync needed
        if not force:
            db_hash = await self.repository.get_file_hash(file_path)
            if db_hash == file_hash:
                logger.info("File already synced, skipping", file_path=file_path)
                existing_file = await self.repository.get_file_by_path(file_path)
                return existing_file.id if existing_file else 0

        # Extract metadata from content
        metadata = self._extract_metadata(file_path, content)

        # Check if file exists in DB
        existing_file = await self.repository.get_file_by_path(file_path)

        if existing_file:
            # Update existing file
            update = MemoryFileUpdate(
                title=metadata["title"],
                category=metadata["category"],
                tags=metadata["tags"],
                metadata=metadata["extra"],
                file_hash=file_hash,
                word_count=self.file_manager.get_word_count(content),
            )
            await self.repository.update_file(existing_file.id, update)
            file_id = existing_file.id

            # Delete old chunks
            await self.repository.delete_chunks_by_file(file_id)
        else:
            # Create new file
            file_create = MemoryFileCreate(
                file_path=file_path,
                title=metadata["title"],
                category=metadata["category"],
                tags=metadata["tags"],
                metadata=metadata["extra"],
                content=content,
                file_hash=file_hash,
                word_count=self.file_manager.get_word_count(content),
            )
            created_file = await self.repository.create_file(file_create)
            file_id = created_file.id

        # Chunk content
        chunks = self.chunker.chunk_markdown(content, file_path)

        if not chunks:
            logger.warning("No chunks generated", file_path=file_path)
            return file_id

        # Generate embeddings in batches
        chunk_texts = [chunk["content"] for chunk in chunks]
        all_embeddings = []

        for i in range(0, len(chunk_texts), self.batch_size):
            batch = chunk_texts[i : i + self.batch_size]
            embeddings = await self.embedding_provider.embed_batch(batch)
            
            # Normalize embedding dimensions to match database schema
            # Use configured dimension from embedding provider
            db_dimension = self.embedding_dimension
            normalized_embeddings = []
            for emb in embeddings:
                emb_list = list(emb) if not isinstance(emb, list) else emb
                if len(emb_list) < db_dimension:
                    # Pad with zeros if smaller
                    emb_list = emb_list + [0.0] * (db_dimension - len(emb_list))
                elif len(emb_list) > db_dimension:
                    # Truncate if larger
                    emb_list = emb_list[:db_dimension]
                normalized_embeddings.append(emb_list)
            
            all_embeddings.extend(normalized_embeddings)

        # Create chunk objects
        chunk_creates = [
            ChunkCreate(
                file_id=file_id,
                chunk_index=chunk["chunk_index"],
                content=chunk["content"],
                content_hash=chunk["content_hash"],
                embedding=embedding,
                header_path=chunk["header_path"],
                section_level=chunk["section_level"],
            )
            for chunk, embedding in zip(chunks, all_embeddings)
        ]

        # Insert chunks
        await self.repository.insert_chunks(chunk_creates)

        logger.info("File synced", file_path=file_path, file_id=file_id, chunks_count=len(chunk_creates))

        return file_id

    async def sync_all_files(self, pattern: str = "**/*.md") -> list[int]:
        """
        Sync all files matching pattern.

        Args:
            pattern: Glob pattern

        Returns:
            List of synced file IDs
        """
        files = await self.file_manager.list_files(pattern)
        file_ids = []

        for file_path in files:
            try:
                file_id = await self.sync_file(file_path)
                file_ids.append(file_id)
            except Exception as e:
                logger.error("Failed to sync file", file_path=file_path, error=str(e))

        logger.info("Bulk sync completed", total_files=len(files), synced=len(file_ids))

        return file_ids

    def _extract_metadata(self, file_path: str, content: str) -> dict:
        """
        Extract metadata from file path and content.

        Args:
            file_path: File path
            content: File content

        Returns:
            Metadata dictionary
        """
        # Parse category from file path
        parts = file_path.split("/")
        if len(parts) > 1:
            category_str = parts[0].rstrip("s")  # Remove trailing 's'
        else:
            category_str = "other"

        # Map to MemoryCategory
        category_map = {
            "project": MemoryCategory.PROJECT,
            "concept": MemoryCategory.CONCEPT,
            "conversation": MemoryCategory.CONVERSATION,
            "preference": MemoryCategory.PREFERENCE,
            "main": MemoryCategory.MAIN,
        }
        category = category_map.get(category_str, MemoryCategory.OTHER)

        # Extract title from first header or filename
        title = parts[-1].replace(".md", "").replace("_", " ").title()
        if content.startswith("# "):
            first_line = content.split("\n")[0]
            title = first_line.lstrip("# ").strip()

        # Extract tags from frontmatter or content (simplified)
        tags = []
        if "**Tags:**" in content or "**tags:**" in content:
            # Extract tags from markdown
            for line in content.split("\n"):
                if "**Tags:**" in line or "**tags:**" in line:
                    tag_text = line.split("**")[2].strip() if "**" in line else ""
                    tags = [t.strip() for t in tag_text.split(",") if t.strip()]
                    break

        return {
            "title": title,
            "category": category,
            "tags": tags,
            "extra": {},
        }

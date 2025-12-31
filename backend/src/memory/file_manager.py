"""File manager for markdown memory files."""

import hashlib
import os
from pathlib import Path
from typing import Any

import aiofiles
import structlog

logger = structlog.get_logger(__name__)


class FileManager:
    """Manages markdown memory files on filesystem."""

    def __init__(self, memory_dir: str = "./memory_files"):
        """
        Initialize file manager.

        Args:
            memory_dir: Directory for memory files
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    async def read_file(self, file_path: str) -> str:
        """
        Read file content.

        Args:
            file_path: Relative file path

        Returns:
            File content

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        full_path = self.memory_dir / file_path

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        async with aiofiles.open(full_path, "r", encoding="utf-8") as f:
            content = await f.read()

        return content

    async def write_file(self, file_path: str, content: str) -> None:
        """
        Write content to file.

        Args:
            file_path: Relative file path
            content: File content
        """
        full_path = self.memory_dir / file_path

        # Create parent directories if needed
        full_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(full_path, "w", encoding="utf-8") as f:
            await f.write(content)

        logger.info("File written", file_path=file_path, size=len(content))

    async def delete_file(self, file_path: str) -> None:
        """
        Delete file.

        Args:
            file_path: Relative file path

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        full_path = self.memory_dir / file_path

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        full_path.unlink()
        logger.info("File deleted", file_path=file_path)

    async def file_exists(self, file_path: str) -> bool:
        """
        Check if file exists.

        Args:
            file_path: Relative file path

        Returns:
            True if file exists
        """
        full_path = self.memory_dir / file_path
        return full_path.exists()

    async def list_files(self, pattern: str = "**/*.md") -> list[str]:
        """
        List files matching pattern.

        Args:
            pattern: Glob pattern

        Returns:
            List of relative file paths
        """
        files = []
        for path in self.memory_dir.glob(pattern):
            if path.is_file():
                rel_path = path.relative_to(self.memory_dir)
                files.append(str(rel_path))

        return sorted(files)

    def compute_file_hash(self, content: str) -> str:
        """
        Compute SHA256 hash of file content.

        Args:
            content: File content

        Returns:
            SHA256 hash
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get_word_count(self, content: str) -> int:
        """
        Get word count of content.

        Args:
            content: Text content

        Returns:
            Word count
        """
        return len(content.split())

    async def move_file(self, old_path: str, new_path: str) -> None:
        """
        Move/rename file.

        Args:
            old_path: Current file path
            new_path: New file path
        """
        old_full_path = self.memory_dir / old_path
        new_full_path = self.memory_dir / new_path

        if not old_full_path.exists():
            raise FileNotFoundError(f"File not found: {old_path}")

        # Create parent directories if needed
        new_full_path.parent.mkdir(parents=True, exist_ok=True)

        old_full_path.rename(new_full_path)
        logger.info("File moved", old_path=old_path, new_path=new_path)

    async def copy_file(self, source_path: str, dest_path: str) -> None:
        """
        Copy file.

        Args:
            source_path: Source file path
            dest_path: Destination file path
        """
        content = await self.read_file(source_path)
        await self.write_file(dest_path, content)
        logger.info("File copied", source=source_path, dest=dest_path)

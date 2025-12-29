"""Markdown-aware chunking with header context preservation."""

import hashlib
from typing import Any

import structlog
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

logger = structlog.get_logger(__name__)


class MarkdownChunker:
    """Smart chunking for markdown documents preserving header context."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 200):
        """
        Initialize markdown chunker.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Split by markdown headers first
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
            ],
            strip_headers=False,  # Keep headers in content
        )

        # Then split large sections by size
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_markdown(self, content: str, file_path: str) -> list[dict[str, Any]]:
        """
        Chunk markdown content preserving structure.

        Args:
            content: Markdown content
            file_path: File path (for logging)

        Returns:
            List of chunk dictionaries with metadata
        """
        if not content.strip():
            return []

        try:
            # First split by headers
            header_splits = self.header_splitter.split_text(content)

            # Then split large sections
            all_chunks = []
            for doc in header_splits:
                # Extract header path from metadata
                header_path = []
                section_level = 0
                for i in range(1, 5):
                    header_key = f"h{i}"
                    if header_key in doc.metadata:
                        header_path.append(doc.metadata[header_key])
                        section_level = i

                # Split large sections
                if len(doc.page_content) > self.chunk_size:
                    sub_chunks = self.text_splitter.split_text(doc.page_content)
                else:
                    sub_chunks = [doc.page_content]

                # Create chunk objects
                for sub_chunk in sub_chunks:
                    if sub_chunk.strip():
                        all_chunks.append({
                            "content": sub_chunk,
                            "header_path": header_path,
                            "section_level": section_level,
                        })

            # Add chunk indices and content hashes
            result = []
            for idx, chunk in enumerate(all_chunks):
                chunk["chunk_index"] = idx
                chunk["content_hash"] = self._compute_hash(chunk["content"])
                result.append(chunk)

            logger.info(
                "Markdown chunked",
                file_path=file_path,
                total_chunks=len(result),
                avg_chunk_size=sum(len(c["content"]) for c in result) // len(result) if result else 0,
            )

            return result

        except Exception as e:
            logger.error("Failed to chunk markdown", error=str(e), file_path=file_path)
            # Fallback: simple text splitting
            return self._fallback_chunk(content)

    def _fallback_chunk(self, content: str) -> list[dict[str, Any]]:
        """
        Fallback chunking without header extraction.

        Args:
            content: Text content

        Returns:
            List of chunk dictionaries
        """
        chunks = self.text_splitter.split_text(content)
        result = []

        for idx, chunk in enumerate(chunks):
            if chunk.strip():
                result.append({
                    "content": chunk,
                    "header_path": [],
                    "section_level": 0,
                    "chunk_index": idx,
                    "content_hash": self._compute_hash(chunk),
                })

        return result

    def _compute_hash(self, content: str) -> str:
        """Compute SHA256 hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

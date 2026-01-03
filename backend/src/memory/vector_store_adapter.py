"""Vector store adapter for embedding storage and retrieval.

Abstracts away the specific vector database (Chroma, FAISS, in-memory)
to allow flexible deployment without pgvector dependency.
"""

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


# ==================== Abstract Base Class ====================


class VectorStoreAdapter(ABC):
    """Abstract base class for vector store implementations."""

    @abstractmethod
    async def add_embeddings(
        self,
        file_id: int,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        """
        Add embeddings for file chunks.

        Args:
            file_id: Database file ID
            chunks: List of chunk dicts with id, content, metadata
            embeddings: List of embedding vectors
        """
        pass

    @abstractmethod
    async def search(
        self, query_embedding: list[float], top_k: int = 10, filter_dict: dict | None = None
    ) -> list[dict[str, Any]]:
        """
        Search for similar chunks by embedding.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_dict: Optional metadata filters

        Returns:
            List of dicts with: chunk_id, file_id, content, score, metadata
        """
        pass

    @abstractmethod
    async def delete_file(self, file_id: int) -> None:
        """Delete all embeddings for a file."""
        pass

    @abstractmethod
    async def get_stats(self) -> dict[str, Any]:
        """Get vector store statistics."""
        pass


# ==================== In-Memory FAISS Implementation ====================


class FAISSAdapter(VectorStoreAdapter):
    """FAISS-based in-memory vector store.

    Fast for development/testing, but not persistent.
    Suitable for small to medium datasets (<100K embeddings).
    """

    def __init__(self, dimension: int = 1536):  # Default, but should be passed from settings
        """Initialize FAISS index."""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS not installed. Run: pip install faiss-cpu  # or faiss-gpu"
            )

        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance
        self.metadata_store: dict[int, dict] = {}  # index_id -> metadata
        self.next_id = 0
        logger.info("FAISS adapter initialized", dimension=dimension)

    async def add_embeddings(
        self,
        file_id: int,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        """Add embeddings to FAISS index."""
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have same length")

        # Convert to numpy array
        vectors = np.array(embeddings, dtype=np.float32)

        # Add to index
        start_id = self.next_id
        self.index.add(vectors)

        # Store metadata
        for i, chunk in enumerate(chunks):
            idx = start_id + i
            self.metadata_store[idx] = {
                "chunk_id": chunk["id"],
                "file_id": file_id,
                "content": chunk["content"],
                "metadata": chunk.get("metadata", {}),
            }
            self.next_id += 1

        logger.debug(
            "Added embeddings to FAISS",
            file_id=file_id,
            count=len(chunks),
            total_vectors=self.index.ntotal,
        )

    async def search(
        self, query_embedding: list[float], top_k: int = 10, filter_dict: dict | None = None
    ) -> list[dict[str, Any]]:
        """Search FAISS index."""
        if self.index.ntotal == 0:
            return []

        # Convert query to numpy
        query_vector = np.array([query_embedding], dtype=np.float32)

        # Search
        distances, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))

        # Retrieve metadata and convert distances to similarity scores
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            metadata = self.metadata_store.get(int(idx))
            if not metadata:
                continue

            # Apply filters if provided
            if filter_dict:
                match = all(
                    metadata.get("metadata", {}).get(k) == v for k, v in filter_dict.items()
                )
                if not match:
                    continue

            # Convert L2 distance to similarity (inverse)
            similarity = 1.0 / (1.0 + float(dist))

            results.append(
                {
                    "chunk_id": metadata["chunk_id"],
                    "file_id": metadata["file_id"],
                    "content": metadata["content"],
                    "score": similarity,
                    "metadata": metadata.get("metadata", {}),
                }
            )

        return results[:top_k]

    async def delete_file(self, file_id: int) -> None:
        """Delete embeddings for a file (soft delete by removing metadata)."""
        deleted_count = 0
        for idx, meta in list(self.metadata_store.items()):
            if meta["file_id"] == file_id:
                del self.metadata_store[idx]
                deleted_count += 1

        logger.debug("Deleted file embeddings from FAISS", file_id=file_id, count=deleted_count)

    async def get_stats(self) -> dict[str, Any]:
        """Get FAISS index statistics."""
        return {
            "type": "faiss",
            "dimension": self.dimension,
            "total_vectors": self.index.ntotal,
            "stored_metadata": len(self.metadata_store),
        }


# ==================== Chroma Implementation ====================


class ChromaAdapter(VectorStoreAdapter):
    """ChromaDB-based vector store.

    Persistent, supports metadata filtering, good for production.
    """

    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "embeddings"):
        """Initialize ChromaDB client."""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("ChromaDB not installed. Run: pip install chromadb")

        self.client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_directory,
            )
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)
        logger.info(
            "Chroma adapter initialized",
            persist_directory=persist_directory,
            collection=collection_name,
        )

    async def add_embeddings(
        self,
        file_id: int,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        """Add embeddings to Chroma."""
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have same length")

        # Prepare IDs (unique identifiers)
        ids = [f"{file_id}_{chunk['id']}" for chunk in chunks]

        # Prepare documents (text content)
        documents = [chunk["content"] for chunk in chunks]

        # Prepare metadatas
        metadatas = [
            {
                "file_id": file_id,
                "chunk_id": chunk["id"],
                **(chunk.get("metadata", {})),
            }
            for chunk in chunks
        ]

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.debug("Added embeddings to Chroma", file_id=file_id, count=len(chunks))

    async def search(
        self, query_embedding: list[float], top_k: int = 10, filter_dict: dict | None = None
    ) -> list[dict[str, Any]]:
        """Search Chroma collection."""
        where = filter_dict if filter_dict else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

        # Format results
        formatted = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                metadata = results["metadatas"][0][i]
                formatted.append(
                    {
                        "chunk_id": metadata.get("chunk_id"),
                        "file_id": metadata.get("file_id"),
                        "content": results["documents"][0][i],
                        "score": 1.0 - results["distances"][0][i],  # Convert distance to similarity
                        "metadata": metadata,
                    }
                )

        return formatted

    async def delete_file(self, file_id: int) -> None:
        """Delete all embeddings for a file."""
        # Query all IDs for this file
        self.collection.delete(where={"file_id": file_id})
        logger.debug("Deleted file embeddings from Chroma", file_id=file_id)

    async def get_stats(self) -> dict[str, Any]:
        """Get Chroma collection statistics."""
        return {
            "type": "chroma",
            "total_vectors": self.collection.count(),
            "collection_name": self.collection.name,
        }


# ==================== Mock Implementation (for testing) ====================


class MockVectorStoreAdapter(VectorStoreAdapter):
    """Mock vector store for testing (no actual vector search)."""

    def __init__(self):
        self.embeddings_store: dict[int, list[dict]] = {}  # file_id -> chunks
        logger.info("Mock vector store adapter initialized")

    async def add_embeddings(
        self,
        file_id: int,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        """Store chunks without actual embeddings."""
        self.embeddings_store[file_id] = chunks
        logger.debug("Added mock embeddings", file_id=file_id, count=len(chunks))

    async def search(
        self, query_embedding: list[float], top_k: int = 10, filter_dict: dict | None = None
    ) -> list[dict[str, Any]]:
        """Return dummy results."""
        all_chunks = []
        for file_id, chunks in self.embeddings_store.items():
            for chunk in chunks:
                all_chunks.append(
                    {
                        "chunk_id": chunk["id"],
                        "file_id": file_id,
                        "content": chunk["content"],
                        "score": 0.8,  # Dummy score
                        "metadata": chunk.get("metadata", {}),
                    }
                )

        return all_chunks[:top_k]

    async def delete_file(self, file_id: int) -> None:
        """Delete mock embeddings."""
        self.embeddings_store.pop(file_id, None)

    async def get_stats(self) -> dict[str, Any]:
        """Get mock stats."""
        total_chunks = sum(len(chunks) for chunks in self.embeddings_store.values())
        return {
            "type": "mock",
            "total_vectors": total_chunks,
            "files": len(self.embeddings_store),
        }


# ==================== Factory ====================


def create_vector_store(
    store_type: str,
    dimension: int = 1536,  # Default, but should be passed from settings
    persist_directory: str = "./vector_store",
    collection_name: str = "embeddings",
) -> VectorStoreAdapter:
    """
    Factory function to create vector store adapter.

    Args:
        store_type: "faiss", "chroma", or "mock"
        dimension: Embedding dimension (for FAISS)
        persist_directory: Directory for persistent storage (for Chroma)
        collection_name: Collection/index name

    Returns:
        VectorStoreAdapter instance
    """
    if store_type == "faiss":
        return FAISSAdapter(dimension=dimension)
    elif store_type == "chroma":
        return ChromaAdapter(
            persist_directory=persist_directory, collection_name=collection_name
        )
    elif store_type == "mock":
        return MockVectorStoreAdapter()
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")

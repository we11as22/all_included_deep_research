"""Mock embedding provider for offline testing."""

from __future__ import annotations

from src.embeddings.base import EmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    """Return deterministic zero embeddings."""

    def __init__(self, dimension: int = 1536) -> None:
        self.dimension = dimension

    async def embed_text(self, text: str) -> list[float]:
        return [0.0] * self.dimension

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dimension for _ in texts]

    def get_dimension(self) -> int:
        return self.dimension

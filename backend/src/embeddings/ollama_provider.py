"""Ollama embedding provider for local embeddings."""

import aiohttp
import structlog

from src.embeddings.base import EmbeddingProvider

logger = structlog.get_logger(__name__)


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama embedding provider for local models."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        """
        Initialize Ollama embedding provider.

        Args:
            base_url: Ollama server URL
            model: Model name
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._dimension: int | None = None

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for single text."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    embedding = data["embedding"]

                    # Cache dimension on first call
                    if self._dimension is None:
                        self._dimension = len(embedding)

                    return embedding
        except Exception as e:
            logger.error("Failed to generate Ollama embedding", error=str(e), model=self.model)
            raise

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for batch of texts."""
        if not texts:
            return []

        # Ollama doesn't have native batch API, so we process sequentially
        # Could parallelize with asyncio.gather but might overwhelm local server
        embeddings = []
        for text in texts:
            embedding = await self.embed_text(text)
            embeddings.append(embedding)

        return embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            # nomic-embed-text is 768 dimensions by default
            return 768
        return self._dimension

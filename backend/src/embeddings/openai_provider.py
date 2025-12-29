"""OpenAI embedding provider."""

import asyncio
from typing import Any

import structlog
from openai import AsyncOpenAI

from src.embeddings.base import EmbeddingProvider

logger = structlog.get_logger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider using text-embedding-3-small."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small", dimension: int = 1536):
        """
        Initialize OpenAI embedding provider.

        Args:
            api_key: OpenAI API key
            model: Model name
            dimension: Embedding dimension
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.dimension = dimension

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for single text."""
        try:
            response = await self.client.embeddings.create(
                input=text,
                model=self.model,
                dimensions=self.dimension if "3" in self.model else None,  # Only for v3 models
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error("Failed to generate embedding", error=str(e), model=self.model)
            raise

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for batch of texts."""
        if not texts:
            return []

        try:
            # OpenAI allows up to 2048 texts per request
            batch_size = 100
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = await self.client.embeddings.create(
                    input=batch,
                    model=self.model,
                    dimensions=self.dimension if "3" in self.model else None,
                )
                all_embeddings.extend([data.embedding for data in response.data])

                # Small delay to avoid rate limits
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)

            return all_embeddings
        except Exception as e:
            logger.error("Failed to generate batch embeddings", error=str(e), batch_size=len(texts))
            raise

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension

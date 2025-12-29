"""HuggingFace embedding provider (optional dependency)."""

import asyncio
from typing import Optional

import structlog

from src.embeddings.base import EmbeddingProvider

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = structlog.get_logger(__name__)


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """HuggingFace embedding provider using sentence-transformers."""

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        api_key: Optional[str] = None,
        use_local: bool = True,
    ):
        """
        Initialize HuggingFace embedding provider.

        Args:
            model: Model name or path
            api_key: HuggingFace API key (optional, for private models)
            use_local: Use local model (True) or API (False)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers package not installed. "
                "Install with: pip install 'sentence-transformers>=2.3.0'"
            )

        self.model_name = model
        self.api_key = api_key
        self.use_local = use_local

        if use_local:
            # Load model locally
            self.model = SentenceTransformer(model)
            self.dimension = self.model.get_sentence_embedding_dimension()
        else:
            # API-based would need different implementation
            raise NotImplementedError("API-based HuggingFace embeddings not yet implemented")

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for single text."""
        try:
            # Run in thread pool since sentence-transformers is synchronous
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(None, lambda: self.model.encode(text))
            return embedding.tolist()
        except Exception as e:
            logger.error("Failed to generate HuggingFace embedding", error=str(e), model=self.model_name)
            raise

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for batch of texts."""
        if not texts:
            return []

        try:
            # Run in thread pool
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(texts, show_progress_bar=False)
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error("Failed to generate HuggingFace batch embeddings", error=str(e), batch_size=len(texts))
            raise

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension

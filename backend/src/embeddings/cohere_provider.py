"""Cohere embedding provider (optional dependency)."""

import structlog

from src.embeddings.base import EmbeddingProvider

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

logger = structlog.get_logger(__name__)


class CohereEmbeddingProvider(EmbeddingProvider):
    """Cohere embedding provider."""

    def __init__(self, api_key: str, model: str = "embed-english-v3.0", input_type: str = "search_document"):
        """
        Initialize Cohere embedding provider.

        Args:
            api_key: Cohere API key
            model: Model name
            input_type: Input type (search_document, search_query, classification, clustering)
        """
        if not COHERE_AVAILABLE:
            raise ImportError("Cohere package not installed. Install with: pip install 'cohere>=4.47'")

        self.client = cohere.AsyncClient(api_key=api_key)
        self.model = model
        self.input_type = input_type
        self.dimension = 1024  # embed-english-v3.0 dimension

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for single text."""
        try:
            response = await self.client.embed(
                texts=[text],
                model=self.model,
                input_type=self.input_type,
            )
            return response.embeddings[0]
        except Exception as e:
            logger.error("Failed to generate Cohere embedding", error=str(e), model=self.model)
            raise

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for batch of texts."""
        if not texts:
            return []

        try:
            # Cohere allows up to 96 texts per request
            batch_size = 96
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = await self.client.embed(
                    texts=batch,
                    model=self.model,
                    input_type=self.input_type,
                )
                all_embeddings.extend(response.embeddings)

            return all_embeddings
        except Exception as e:
            logger.error("Failed to generate Cohere batch embeddings", error=str(e), batch_size=len(texts))
            raise

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension

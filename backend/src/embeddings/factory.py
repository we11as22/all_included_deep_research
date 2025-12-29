"""Factory for creating embedding providers."""

import structlog

from src.config.settings import Settings
from src.embeddings.base import EmbeddingProvider
from src.embeddings.mock_provider import MockEmbeddingProvider
from src.embeddings.openai_provider import OpenAIEmbeddingProvider
from src.embeddings.ollama_provider import OllamaEmbeddingProvider

logger = structlog.get_logger(__name__)


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """
    Create embedding provider based on settings.

    Args:
        settings: Application settings

    Returns:
        Embedding provider instance

    Raises:
        ValueError: If provider is not supported or required API key is missing
    """
    provider = settings.embedding_provider.lower()

    if provider == "openai":
        if not settings.openai_api_key:
            if settings.llm_mode == "mock":
                logger.warning("OpenAI API key missing in mock mode, using mock embeddings")
                return MockEmbeddingProvider(dimension=settings.embedding_dimension)
            raise ValueError("OpenAI API key is required for OpenAI embedding provider")

        logger.info(
            "Creating OpenAI embedding provider",
            model=settings.openai_embedding_model,
            dimension=settings.embedding_dimension,
        )
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
            dimension=settings.embedding_dimension,
        )

    elif provider == "ollama":
        logger.info(
            "Creating Ollama embedding provider",
            base_url=settings.ollama_base_url,
            model=settings.ollama_embedding_model,
        )
        return OllamaEmbeddingProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embedding_model,
        )

    elif provider == "mock":
        logger.info(
            "Creating mock embedding provider",
            dimension=settings.embedding_dimension,
        )
        return MockEmbeddingProvider(dimension=settings.embedding_dimension)

    elif provider == "cohere":
        try:
            from src.embeddings.cohere_provider import CohereEmbeddingProvider

            if not settings.cohere_api_key:
                raise ValueError("Cohere API key is required for Cohere embedding provider")

            logger.info(
                "Creating Cohere embedding provider",
                model=settings.cohere_embedding_model,
            )
            return CohereEmbeddingProvider(
                api_key=settings.cohere_api_key,
                model=settings.cohere_embedding_model,
                input_type=settings.cohere_input_type,
            )
        except ImportError:
            raise ValueError("Cohere package not installed. Install with: pip install 'cohere>=4.47'")

    elif provider == "huggingface":
        try:
            from src.embeddings.huggingface_provider import HuggingFaceEmbeddingProvider

            logger.info(
                "Creating HuggingFace embedding provider",
                model=settings.huggingface_model,
                use_local=settings.huggingface_use_local,
            )
            return HuggingFaceEmbeddingProvider(
                model=settings.huggingface_model,
                api_key=settings.huggingface_api_key,
                use_local=settings.huggingface_use_local,
            )
        except ImportError:
            raise ValueError(
                "HuggingFace packages not installed. Install with: pip install 'sentence-transformers>=2.3.0'"
            )

    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")

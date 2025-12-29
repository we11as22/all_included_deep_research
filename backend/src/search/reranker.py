"""Semantic reranking for search results."""

import structlog

from src.embeddings.base import EmbeddingProvider
from src.search.models import SearchResult

logger = structlog.get_logger(__name__)


class SemanticReranker:
    """Rerank search results using semantic similarity."""

    def __init__(self, embedding_provider: EmbeddingProvider):
        """
        Initialize semantic reranker.

        Args:
            embedding_provider: Embedding provider for similarity computation
        """
        self.embedding_provider = embedding_provider
        logger.info("SemanticReranker initialized")

    async def rerank(
        self, query: str, results: list[SearchResult], top_k: int | None = None
    ) -> list[SearchResult]:
        """
        Rerank search results by semantic similarity to query.

        Args:
            query: Search query
            results: List of search results to rerank
            top_k: Return only top K results (None = return all)

        Returns:
            Reranked list of search results
        """
        if not results:
            return []

        try:
            # Get query embedding
            query_embedding = await self.embedding_provider.embed_text(query)

            # Prepare documents for embedding
            documents = [
                f"{result.title}\n{result.snippet}" for result in results
            ]

            # Get document embeddings
            doc_embeddings = await self.embedding_provider.embed_batch(documents)

            # Compute cosine similarity scores
            scores = []
            for doc_embedding in doc_embeddings:
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                scores.append(similarity)

            # Create list of (result, score) tuples
            result_scores = list(zip(results, scores))

            # Sort by score descending
            result_scores.sort(key=lambda x: x[1], reverse=True)

            # Update scores in results
            reranked = []
            for result, score in result_scores:
                # Create a copy with updated score
                reranked_result = result.model_copy()
                reranked_result.score = score
                reranked.append(reranked_result)

            # Apply top_k limit if specified
            if top_k is not None:
                reranked = reranked[:top_k]

            logger.info(
                "Search results reranked",
                query=query,
                original_count=len(results),
                reranked_count=len(reranked),
            )

            return reranked

        except Exception as e:
            logger.error("Reranking failed, returning original order", error=str(e))
            return results[:top_k] if top_k else results

    async def rerank_documents(
        self, query: str, documents: list[str], top_k: int = 5
    ) -> list[dict[str, any]]:
        """
        Rerank text documents by semantic similarity.

        Args:
            query: Search query
            documents: List of text documents
            top_k: Number of top results to return

        Returns:
            List of dicts with 'document' and 'score' keys
        """
        if not documents:
            return []

        try:
            # Get embeddings
            query_embedding = await self.embedding_provider.embed_text(query)
            doc_embeddings = await self.embedding_provider.embed_batch(documents)

            # Compute scores
            scores = []
            for doc_embedding in doc_embeddings:
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                scores.append(similarity)

            # Create result list
            doc_scores = [
                {"document": doc, "score": score}
                for doc, score in zip(documents, scores)
            ]

            # Sort by score descending
            doc_scores.sort(key=lambda x: x["score"], reverse=True)

            # Return top K
            return doc_scores[:top_k]

        except Exception as e:
            logger.error("Document reranking failed", error=str(e))
            return [{"document": doc, "score": 0.0} for doc in documents[:top_k]]

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (0-1)
        """
        # Dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))

        # Magnitudes
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)


class HybridReranker(SemanticReranker):
    """Hybrid reranker combining semantic similarity and original scores."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        semantic_weight: float = 0.7,
        original_weight: float = 0.3,
    ):
        """
        Initialize hybrid reranker.

        Args:
            embedding_provider: Embedding provider
            semantic_weight: Weight for semantic similarity (0-1)
            original_weight: Weight for original scores (0-1)
        """
        super().__init__(embedding_provider)
        self.semantic_weight = semantic_weight
        self.original_weight = original_weight

    async def rerank(
        self, query: str, results: list[SearchResult], top_k: int | None = None
    ) -> list[SearchResult]:
        """
        Rerank using hybrid score (semantic + original).

        Args:
            query: Search query
            results: List of search results
            top_k: Return only top K results

        Returns:
            Reranked list of search results
        """
        if not results:
            return []

        try:
            # Get semantic scores from parent class
            semantic_reranked = await super().rerank(query, results, top_k=None)

            # Normalize original scores to 0-1 range
            original_scores = [r.score for r in results]
            max_original = max(original_scores) if original_scores else 1.0
            min_original = min(original_scores) if original_scores else 0.0
            score_range = max_original - min_original if max_original > min_original else 1.0

            # Compute hybrid scores
            reranked = []
            for orig_result, semantic_result in zip(results, semantic_reranked):
                # Normalize original score
                norm_original = (
                    (orig_result.score - min_original) / score_range if score_range > 0 else 0.0
                )

                # Combine scores
                hybrid_score = (
                    self.semantic_weight * semantic_result.score
                    + self.original_weight * norm_original
                )

                # Create result with hybrid score
                hybrid_result = orig_result.model_copy()
                hybrid_result.score = hybrid_score
                reranked.append(hybrid_result)

            # Sort by hybrid score
            reranked.sort(key=lambda x: x.score, reverse=True)

            # Apply top_k limit
            if top_k is not None:
                reranked = reranked[:top_k]

            logger.info(
                "Hybrid reranking completed",
                query=query,
                semantic_weight=self.semantic_weight,
                original_weight=self.original_weight,
                results_count=len(reranked),
            )

            return reranked

        except Exception as e:
            logger.error("Hybrid reranking failed", error=str(e))
            return results[:top_k] if top_k else results

"""Chat message hybrid search engine with RRF (Reciprocal Rank Fusion)."""

import asyncpg
import structlog
from dataclasses import dataclass
from typing import Any
from collections.abc import Sequence

from src.embeddings.base import EmbeddingProvider

logger = structlog.get_logger(__name__)


@dataclass
class ChatMessageSearchResult:
    """Search result for chat message."""

    message_id: int
    chat_id: str
    message_message_id: str
    role: str
    content: str
    created_at: str
    chat_title: str
    chat_updated_at: str
    score: float
    search_mode: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "message_id": self.message_id,
            "chat_id": self.chat_id,
            "message_message_id": self.message_message_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
            "chat_title": self.chat_title,
            "chat_updated_at": self.chat_updated_at,
            "score": self.score,
            "search_mode": self.search_mode,
        }


def _normalize_query(query: object) -> str:
    """Normalize query to string."""
    if isinstance(query, str):
        return query
    if query is None:
        return ""
    if hasattr(query, "tolist"):
        return ""
    if isinstance(query, (list, tuple, set, dict)):
        return ""
    if isinstance(query, Sequence):
        return ""
    return str(query)


def _format_vector_param(embedding: Sequence[float]) -> str:
    """Format embedding for pgvector input when asyncpg expects text."""
    return "[" + ", ".join(str(value) for value in embedding) + "]"


class ChatMessageSearchEngine:
    """Hybrid search for chat messages combining vector and fulltext with RRF."""

    def __init__(self, db_pool: asyncpg.Pool, embedding_provider: EmbeddingProvider, rrf_k: int = 60):
        """
        Initialize chat message search engine.

        Args:
            db_pool: AsyncPG connection pool
            embedding_provider: Embedding provider for query embedding
            rrf_k: RRF K parameter (default 60)
        """
        self.db_pool = db_pool
        self.embedding_provider = embedding_provider
        self.rrf_k = rrf_k

    async def search(
        self,
        query: str,
        limit: int = 5,
        chat_id: str | None = None,
        role_filter: str | None = None,
    ) -> list[ChatMessageSearchResult]:
        """
        Search chat messages using hybrid search (vector + fulltext).

        Args:
            query: Search query (must be a string)
            limit: Maximum results (default 5)
            chat_id: Optional filter by specific chat
            role_filter: Optional filter by role (user, assistant, system)

        Returns:
            List of search results ordered by relevance
        """
        normalized = _normalize_query(query)
        if not isinstance(query, str) or normalized != query:
            logger.warning("Query normalized for chat search", query_type=type(query).__name__)
        query = normalized

        if not query.strip():
            return []

        return await self._hybrid_search(query, limit, chat_id, role_filter)

    async def _hybrid_search(
        self,
        query: str,
        limit: int,
        chat_id: str | None,
        role_filter: str | None,
    ) -> list[ChatMessageSearchResult]:
        """Hybrid search with RRF combining vector and fulltext."""
        query = _normalize_query(query)

        # Generate query embedding (requires string input)
        query_embedding = await self.embedding_provider.embed_text(query)

        # Normalize embedding to database schema dimension (not provider dimension!)
        from src.database.schema import EMBEDDING_DIMENSION
        db_dimension = EMBEDDING_DIMENSION
        if len(query_embedding) < db_dimension:
            query_embedding = list(query_embedding) + [0.0] * (db_dimension - len(query_embedding))
        elif len(query_embedding) > db_dimension:
            query_embedding = query_embedding[:db_dimension]

        embedding_param = _format_vector_param(query_embedding)

        async with self.db_pool.acquire() as conn:
            # Build filters
            filters = []
            filter_params = []
            param_idx = 5  # Start after $1-$4 which are: embedding, query, rrf_k, limit

            if chat_id:
                filters.append(f"cm.chat_id = ${param_idx}")
                filter_params.append(chat_id)
                param_idx += 1

            if role_filter:
                filters.append(f"cm.role = ${param_idx}")
                filter_params.append(role_filter)
                param_idx += 1

            filter_clause = f"AND {' AND '.join(filters)}" if filters else ""

            # RRF Hybrid Search Query
            sql = f"""
            WITH vector_search AS (
                SELECT
                    cm.id, cm.chat_id, cm.message_id, cm.role, cm.content, cm.created_at,
                    c.title as chat_title, c.updated_at as chat_updated_at,
                    ROW_NUMBER() OVER (ORDER BY cm.embedding <=> $1::vector) AS rank
                FROM chat_messages cm
                JOIN chats c ON cm.chat_id = c.id
                WHERE cm.embedding IS NOT NULL {filter_clause}
                ORDER BY cm.embedding <=> $1::vector
                LIMIT $4
            ),
            fulltext_search AS (
                SELECT
                    cm.id, cm.chat_id, cm.message_id, cm.role, cm.content, cm.created_at,
                    c.title as chat_title, c.updated_at as chat_updated_at,
                    ROW_NUMBER() OVER (
                        ORDER BY ts_rank(to_tsvector('english', cm.content), plainto_tsquery('english', $2)) DESC
                    ) AS rank
                FROM chat_messages cm
                JOIN chats c ON cm.chat_id = c.id
                WHERE to_tsvector('english', cm.content) @@ plainto_tsquery('english', $2) {filter_clause}
                ORDER BY ts_rank(to_tsvector('english', cm.content), plainto_tsquery('english', $2)) DESC
                LIMIT $4
            ),
            combined AS (
                SELECT
                    COALESCE(v.id, f.id) AS message_id,
                    COALESCE(v.chat_id, f.chat_id) AS chat_id,
                    COALESCE(v.message_id, f.message_id) AS message_message_id,
                    COALESCE(v.role, f.role) AS role,
                    COALESCE(v.content, f.content) AS content,
                    COALESCE(v.created_at, f.created_at) AS created_at,
                    COALESCE(v.chat_title, f.chat_title) AS chat_title,
                    COALESCE(v.chat_updated_at, f.chat_updated_at) AS chat_updated_at,
                    (1.0 / ($3 + COALESCE(v.rank, 999999))) + (1.0 / ($3 + COALESCE(f.rank, 999999))) AS rrf_score
                FROM vector_search v
                FULL OUTER JOIN fulltext_search f ON v.id = f.id
            )
            SELECT * FROM combined
            ORDER BY rrf_score DESC
            LIMIT $4;
            """

            # Pass parameters explicitly: embedding (vector text), query (str), rrf_k (int), limit (int), then filter params
            query = _normalize_query(query)
            all_params = [embedding_param, query, self.rrf_k, limit] + filter_params
            rows = await conn.fetch(sql, *all_params)

            results = [
                ChatMessageSearchResult(
                    message_id=row["message_id"],
                    chat_id=row["chat_id"],
                    message_message_id=row["message_message_id"],
                    role=row["role"],
                    content=row["content"],
                    created_at=row["created_at"].isoformat() if row["created_at"] else "",
                    chat_title=row["chat_title"],
                    chat_updated_at=row["chat_updated_at"].isoformat() if row["chat_updated_at"] else "",
                    score=float(row["rrf_score"]),
                    search_mode="hybrid",
                )
                for row in rows
            ]

            logger.info("Chat message hybrid search completed", query=query, results_count=len(results))
            return results

    async def vector_search(
        self,
        query: str,
        limit: int = 5,
        chat_id: str | None = None,
        role_filter: str | None = None,
    ) -> list[ChatMessageSearchResult]:
        """Vector-only semantic search for chat messages."""
        query = _normalize_query(query)

        if not query.strip():
            return []

        query_embedding = await self.embedding_provider.embed_text(query)

        # Normalize embedding to database schema dimension (not provider dimension!)
        from src.database.schema import EMBEDDING_DIMENSION
        db_dimension = EMBEDDING_DIMENSION
        if len(query_embedding) < db_dimension:
            query_embedding = list(query_embedding) + [0.0] * (db_dimension - len(query_embedding))
        elif len(query_embedding) > db_dimension:
            query_embedding = query_embedding[:db_dimension]

        embedding_param = _format_vector_param(query_embedding)

        async with self.db_pool.acquire() as conn:
            filters = []
            filter_params = []
            param_idx = 3  # Start after $1 (embedding) and $2 (limit)

            if chat_id:
                filters.append(f"cm.chat_id = ${param_idx}")
                filter_params.append(chat_id)
                param_idx += 1

            if role_filter:
                filters.append(f"cm.role = ${param_idx}")
                filter_params.append(role_filter)
                param_idx += 1

            filter_clause = f"AND {' AND '.join(filters)}" if filters else ""

            sql = f"""
            SELECT
                cm.id as message_id,
                cm.chat_id,
                cm.message_id as message_message_id,
                cm.role,
                cm.content,
                cm.created_at,
                c.title as chat_title,
                c.updated_at as chat_updated_at,
                1 - (cm.embedding <=> $1::vector) AS similarity
            FROM chat_messages cm
            JOIN chats c ON cm.chat_id = c.id
            WHERE cm.embedding IS NOT NULL {filter_clause}
            ORDER BY cm.embedding <=> $1::vector
            LIMIT $2;
            """

            all_params = [embedding_param, limit] + filter_params
            rows = await conn.fetch(sql, *all_params)

            results = [
                ChatMessageSearchResult(
                    message_id=row["message_id"],
                    chat_id=row["chat_id"],
                    message_message_id=row["message_message_id"],
                    role=row["role"],
                    content=row["content"],
                    created_at=row["created_at"].isoformat() if row["created_at"] else "",
                    chat_title=row["chat_title"],
                    chat_updated_at=row["chat_updated_at"].isoformat() if row["chat_updated_at"] else "",
                    score=float(row["similarity"]),
                    search_mode="vector",
                )
                for row in rows
            ]

            logger.info("Chat message vector search completed", query=query, results_count=len(results))
            return results

    async def fulltext_search(
        self,
        query: str,
        limit: int = 5,
        chat_id: str | None = None,
        role_filter: str | None = None,
    ) -> list[ChatMessageSearchResult]:
        """Fulltext keyword search for chat messages."""
        query = _normalize_query(query)

        if not query.strip():
            return []

        async with self.db_pool.acquire() as conn:
            filters = []
            filter_params = []
            param_idx = 3  # Start after $1 (query) and $2 (limit)

            if chat_id:
                filters.append(f"cm.chat_id = ${param_idx}")
                filter_params.append(chat_id)
                param_idx += 1

            if role_filter:
                filters.append(f"cm.role = ${param_idx}")
                filter_params.append(role_filter)
                param_idx += 1

            filter_clause = f"AND {' AND '.join(filters)}" if filters else ""

            sql = f"""
            SELECT
                cm.id as message_id,
                cm.chat_id,
                cm.message_id as message_message_id,
                cm.role,
                cm.content,
                cm.created_at,
                c.title as chat_title,
                c.updated_at as chat_updated_at,
                ts_rank(to_tsvector('english', cm.content), plainto_tsquery('english', $1)) AS rank_score
            FROM chat_messages cm
            JOIN chats c ON cm.chat_id = c.id
            WHERE to_tsvector('english', cm.content) @@ plainto_tsquery('english', $1) {filter_clause}
            ORDER BY rank_score DESC
            LIMIT $2;
            """

            all_params = [query, limit] + filter_params
            rows = await conn.fetch(sql, *all_params)

            results = [
                ChatMessageSearchResult(
                    message_id=row["message_id"],
                    chat_id=row["chat_id"],
                    message_message_id=row["message_message_id"],
                    role=row["role"],
                    content=row["content"],
                    created_at=row["created_at"].isoformat() if row["created_at"] else "",
                    chat_title=row["chat_title"],
                    chat_updated_at=row["chat_updated_at"].isoformat() if row["chat_updated_at"] else "",
                    score=float(row["rank_score"]),
                    search_mode="fulltext",
                )
                for row in rows
            ]

            logger.info("Chat message fulltext search completed", query=query, results_count=len(results))
            return results

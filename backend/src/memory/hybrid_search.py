"""Hybrid search engine with RRF (Reciprocal Rank Fusion)."""

import asyncpg
import structlog

from src.embeddings.base import EmbeddingProvider
from src.memory.models.search import SearchMode, SearchResult

logger = structlog.get_logger(__name__)


class HybridSearchEngine:
    """Hybrid search combining vector and fulltext with RRF."""

    def __init__(self, db_pool: asyncpg.Pool, embedding_provider: EmbeddingProvider, rrf_k: int = 60):
        """
        Initialize hybrid search engine.

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
        search_mode: SearchMode = SearchMode.HYBRID,
        limit: int = 10,
        category_filter: str | None = None,
        tag_filter: list[str] | None = None,
        file_path: str | None = None,
    ) -> list[SearchResult]:
        """
        Search memory with specified mode.

        Args:
            query: Search query
            search_mode: Search mode (hybrid, vector, fulltext)
            limit: Maximum results
            category_filter: Filter by category
            tag_filter: Filter by tags (AND logic)
            file_path: Search within specific file

        Returns:
            List of search results
        """
        if search_mode == SearchMode.HYBRID:
            return await self._hybrid_search(query, limit, category_filter, tag_filter, file_path)
        elif search_mode == SearchMode.VECTOR:
            return await self._vector_search(query, limit, category_filter, tag_filter, file_path)
        elif search_mode == SearchMode.FULLTEXT:
            return await self._fulltext_search(query, limit, category_filter, tag_filter, file_path)
        else:
            raise ValueError(f"Invalid search mode: {search_mode}")

    async def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        rrf_k: int | None = None,
        category_filter: str | None = None,
        tag_filter: list[str] | None = None,
        file_path: str | None = None,
    ) -> list[SearchResult]:
        """
        Convenience wrapper for hybrid search with optional RRF tuning.

        Args:
            query: Search query
            limit: Maximum results
            rrf_k: Optional override for RRF K parameter
            category_filter: Filter by category
            tag_filter: Filter by tags (AND logic)
            file_path: Search within specific file

        Returns:
            List of search results
        """
        original_rrf = self.rrf_k
        if rrf_k is not None:
            self.rrf_k = rrf_k

        try:
            return await self._hybrid_search(query, limit, category_filter, tag_filter, file_path)
        finally:
            self.rrf_k = original_rrf

    async def _hybrid_search(
        self,
        query: str,
        limit: int,
        category_filter: str | None,
        tag_filter: list[str] | None,
        file_path: str | None,
    ) -> list[SearchResult]:
        """Hybrid search with RRF combining vector and fulltext."""
        # Generate query embedding
        query_embedding = await self.embedding_provider.embed_text(query)

        async with self.db_pool.acquire() as conn:
            # Build filters
            filters = []
            filter_params = []
            param_idx = 5  # Start after $1-$4 which are: embedding, query, rrf_k, limit

            if category_filter:
                filters.append(f"mf.category = ${param_idx}")
                filter_params.append(category_filter)
                param_idx += 1

            if tag_filter:
                for tag in tag_filter:
                    filters.append(f"${param_idx} = ANY(mf.tags)")
                    filter_params.append(tag)
                    param_idx += 1

            if file_path:
                filters.append(f"mf.file_path = ${param_idx}")
                filter_params.append(file_path)
                param_idx += 1

            filter_clause = f"AND {' AND '.join(filters)}" if filters else ""

            # RRF Hybrid Search Query
            sql = f"""
            WITH vector_search AS (
                SELECT
                    mc.id, mc.content, mc.header_path, mc.section_level,
                    mf.id as file_id, mf.file_path, mf.title, mf.category,
                    ROW_NUMBER() OVER (ORDER BY mc.embedding <=> $1::vector) AS rank
                FROM memory_chunks mc
                JOIN memory_files mf ON mc.file_id = mf.id
                WHERE mc.embedding IS NOT NULL {filter_clause}
                ORDER BY mc.embedding <=> $1::vector
                LIMIT $4
            ),
            fulltext_search AS (
                SELECT
                    mc.id, mc.content, mc.header_path, mc.section_level,
                    mf.id as file_id, mf.file_path, mf.title, mf.category,
                    ROW_NUMBER() OVER (
                        ORDER BY ts_rank(to_tsvector('english', mc.content), plainto_tsquery('english', $2)) DESC
                    ) AS rank
                FROM memory_chunks mc
                JOIN memory_files mf ON mc.file_id = mf.id
                WHERE to_tsvector('english', mc.content) @@ plainto_tsquery('english', $2) {filter_clause}
                ORDER BY ts_rank(to_tsvector('english', mc.content), plainto_tsquery('english', $2)) DESC
                LIMIT $4
            ),
            combined AS (
                SELECT
                    COALESCE(v.id, f.id) AS chunk_id,
                    COALESCE(v.file_id, f.file_id) AS file_id,
                    COALESCE(v.file_path, f.file_path) AS file_path,
                    COALESCE(v.title, f.title) AS file_title,
                    COALESCE(v.category, f.category) AS file_category,
                    COALESCE(v.content, f.content) AS content,
                    COALESCE(v.header_path, f.header_path) AS header_path,
                    COALESCE(v.section_level, f.section_level) AS section_level,
                    (1.0 / ($3 + COALESCE(v.rank, 999999))) + (1.0 / ($3 + COALESCE(f.rank, 999999))) AS rrf_score
                FROM vector_search v
                FULL OUTER JOIN fulltext_search f ON v.id = f.id
            )
            SELECT * FROM combined
            ORDER BY rrf_score DESC
            LIMIT $4;
            """

            # Pass parameters explicitly: embedding (vector), query (str), rrf_k (int), limit (int), then filter params
            # asyncpg requires vector to be passed as list[float], which it will convert to vector type
            # Build complete params list to avoid issues with *filter_params unpacking
            all_params = [query_embedding, query, self.rrf_k, limit] + filter_params
            rows = await conn.fetch(sql, *all_params)

            results = [
                SearchResult(
                    chunk_id=row["chunk_id"],
                    file_id=row["file_id"],
                    file_path=row["file_path"],
                    file_title=row["file_title"],
                    file_category=row["file_category"],
                    content=row["content"],
                    header_path=row["header_path"] or [],
                    section_level=row["section_level"],
                    score=float(row["rrf_score"]),
                    search_mode=SearchMode.HYBRID,
                )
                for row in rows
            ]

            logger.info("Hybrid search completed", query=query, results_count=len(results))
            return results

    async def _vector_search(
        self,
        query: str,
        limit: int,
        category_filter: str | None,
        tag_filter: list[str] | None,
        file_path: str | None,
    ) -> list[SearchResult]:
        """Vector-only semantic search."""
        query_embedding = await self.embedding_provider.embed_text(query)

        async with self.db_pool.acquire() as conn:
            filters = []
            filter_params = []
            param_idx = 3  # Start after $1 (embedding) and $2 (limit)

            if category_filter:
                filters.append(f"mf.category = ${param_idx}")
                filter_params.append(category_filter)
                param_idx += 1

            if tag_filter:
                for tag in tag_filter:
                    filters.append(f"${param_idx} = ANY(mf.tags)")
                    filter_params.append(tag)
                    param_idx += 1

            if file_path:
                filters.append(f"mf.file_path = ${param_idx}")
                filter_params.append(file_path)
                param_idx += 1

            filter_clause = f"AND {' AND '.join(filters)}" if filters else ""

            sql = f"""
            SELECT
                mc.id as chunk_id,
                mf.id as file_id,
                mf.file_path,
                mf.title as file_title,
                mf.category as file_category,
                mc.content,
                mc.header_path,
                mc.section_level,
                1 - (mc.embedding <=> $1::vector) AS similarity
            FROM memory_chunks mc
            JOIN memory_files mf ON mc.file_id = mf.id
            WHERE mc.embedding IS NOT NULL {filter_clause}
            ORDER BY mc.embedding <=> $1::vector
            LIMIT $2;
            """

            # Pass parameters explicitly: embedding (vector), limit (int), then filter params
            # Build complete params list to avoid issues with *filter_params unpacking
            all_params = [query_embedding, limit] + filter_params
            rows = await conn.fetch(sql, *all_params)

            results = [
                SearchResult(
                    chunk_id=row["chunk_id"],
                    file_id=row["file_id"],
                    file_path=row["file_path"],
                    file_title=row["file_title"],
                    file_category=row["file_category"],
                    content=row["content"],
                    header_path=row["header_path"] or [],
                    section_level=row["section_level"],
                    score=float(row["similarity"]),
                    search_mode=SearchMode.VECTOR,
                )
                for row in rows
            ]

            logger.info("Vector search completed", query=query, results_count=len(results))
            return results

    async def _fulltext_search(
        self,
        query: str,
        limit: int,
        category_filter: str | None,
        tag_filter: list[str] | None,
        file_path: str | None,
    ) -> list[SearchResult]:
        """Fulltext keyword search."""
        async with self.db_pool.acquire() as conn:
            filters = []
            filter_params = []
            param_idx = 3  # Start after $1 (query) and $2 (limit)

            if category_filter:
                filters.append(f"mf.category = ${param_idx}")
                filter_params.append(category_filter)
                param_idx += 1

            if tag_filter:
                for tag in tag_filter:
                    filters.append(f"${param_idx} = ANY(mf.tags)")
                    filter_params.append(tag)
                    param_idx += 1

            if file_path:
                filters.append(f"mf.file_path = ${param_idx}")
                filter_params.append(file_path)
                param_idx += 1

            filter_clause = f"AND {' AND '.join(filters)}" if filters else ""

            sql = f"""
            SELECT
                mc.id as chunk_id,
                mf.id as file_id,
                mf.file_path,
                mf.title as file_title,
                mf.category as file_category,
                mc.content,
                mc.header_path,
                mc.section_level,
                ts_rank(to_tsvector('english', mc.content), plainto_tsquery('english', $1)) AS rank_score
            FROM memory_chunks mc
            JOIN memory_files mf ON mc.file_id = mf.id
            WHERE to_tsvector('english', mc.content) @@ plainto_tsquery('english', $1) {filter_clause}
            ORDER BY rank_score DESC
            LIMIT $2;
            """

            # Pass parameters explicitly: query (str), limit (int), then filter params
            # Build complete params list to avoid issues with *filter_params unpacking
            all_params = [query, limit] + filter_params
            rows = await conn.fetch(sql, *all_params)

            results = [
                SearchResult(
                    chunk_id=row["chunk_id"],
                    file_id=row["file_id"],
                    file_path=row["file_path"],
                    file_title=row["file_title"],
                    file_category=row["file_category"],
                    content=row["content"],
                    header_path=row["header_path"] or [],
                    section_level=row["section_level"],
                    score=float(row["rank_score"]),
                    search_mode=SearchMode.FULLTEXT,
                )
                for row in rows
            ]

            logger.info("Fulltext search completed", query=query, results_count=len(results))
            return results

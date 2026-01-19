"""Service for managing research memories (notes and findings) with vector search."""

import asyncio
import hashlib
import structlog
from typing import Any, Dict, List, Optional
from datetime import datetime

from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.database.schema import ResearchMemoryModel, ResearchMemoryChunkModel, EMBEDDING_DIMENSION
from src.embeddings.base import EmbeddingProvider

logger = structlog.get_logger(__name__)


class ResearchMemoryService:
    """Service for managing research memories with vector search capabilities."""

    def __init__(
        self, 
        session_factory: Any, 
        embedding_provider: EmbeddingProvider,
        chunk_size: int = 800,
        chunk_overlap: int = 200
    ):
        """Initialize research memory service.
        
        Args:
            session_factory: Async session factory for database access
            embedding_provider: Embedding provider for generating embeddings
            chunk_size: Size of chunks for splitting notes/findings
            chunk_overlap: Overlap between chunks
        """
        self.session_factory = session_factory
        self.embedding_provider = embedding_provider
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    
    def _compute_hash(self, content: str) -> str:
        """Compute SHA256 hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
    
    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Split text into chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of chunk dictionaries with content, chunk_index, content_hash
        """
        if not text.strip():
            return []
        
        chunks = self.text_splitter.split_text(text)
        result = []
        
        for idx, chunk in enumerate(chunks):
            if chunk.strip():
                result.append({
                    "content": chunk,
                    "chunk_index": idx,
                    "content_hash": self._compute_hash(chunk),
                })
        
        return result

    async def save_note(
        self,
        session_id: str,
        agent_id: str,
        title: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Save agent note with chunked embeddings.
        
        Args:
            session_id: Research session ID
            agent_id: Agent ID who created the note
            title: Note title
            content: Note content (full text, not truncated)
            metadata: Additional metadata (urls, tags, etc.)
            
        Returns:
            ID of saved memory record
        """
        try:
            # Split content into chunks
            chunks = self._chunk_text(content)
            
            if not chunks:
                logger.warning("No chunks generated from note content", title=title[:100])
                chunks = [{"content": content, "chunk_index": 0, "content_hash": self._compute_hash(content)}]
            
            # Generate embeddings for all chunks in parallel
            chunk_texts = [chunk["content"] for chunk in chunks]
            # Include title in first chunk for better context
            if chunk_texts:
                chunk_texts[0] = f"{title}\n{chunk_texts[0]}"
            
            embedding_tasks = [self.embedding_provider.embed_text(text) for text in chunk_texts]
            chunk_embeddings = await asyncio.gather(*embedding_tasks)
            
            # Normalize embeddings to database dimension
            normalized_embeddings = []
            for emb in chunk_embeddings:
                if len(emb) < EMBEDDING_DIMENSION:
                    normalized_emb = list(emb) + [0.0] * (EMBEDDING_DIMENSION - len(emb))
                elif len(emb) > EMBEDDING_DIMENSION:
                    normalized_emb = emb[:EMBEDDING_DIMENSION]
                else:
                    normalized_emb = list(emb)
                normalized_embeddings.append(normalized_emb)
            
            # Generate embedding for full content (for backward compatibility)
            full_embedding = await self.embedding_provider.embed_text(f"{title}\n{content}")
            if len(full_embedding) < EMBEDDING_DIMENSION:
                full_embedding = list(full_embedding) + [0.0] * (EMBEDDING_DIMENSION - len(full_embedding))
            elif len(full_embedding) > EMBEDDING_DIMENSION:
                full_embedding = full_embedding[:EMBEDDING_DIMENSION]
            
            async with self.session_factory() as session:
                # Create memory record
                memory = ResearchMemoryModel(
                    session_id=session_id,
                    agent_id=agent_id,
                    memory_type="note",
                    title=title,
                    content=content,  # Full content, not truncated
                    embedding=full_embedding,  # Keep for backward compatibility
                    memory_metadata=metadata or {},
                )
                session.add(memory)
                await session.flush()  # Flush to get memory.id
                
                # Create chunk records with embeddings
                chunk_models = []
                for chunk, embedding in zip(chunks, normalized_embeddings):
                    chunk_model = ResearchMemoryChunkModel(
                        memory_id=memory.id,
                        chunk_index=chunk["chunk_index"],
                        content=chunk["content"],
                        content_hash=chunk["content_hash"],
                        embedding=embedding,
                    )
                    chunk_models.append(chunk_model)
                
                session.add_all(chunk_models)
                await session.commit()
                await session.refresh(memory)
                
                logger.info("Saved research note with chunked embeddings",
                           memory_id=memory.id,
                           session_id=session_id,
                           agent_id=agent_id,
                           title=title[:100],
                           content_length=len(content),
                           chunks_count=len(chunk_models))
                
                return memory.id
        except Exception as e:
            logger.error("Failed to save research note", error=str(e), exc_info=True)
            raise

    async def save_finding(
        self,
        session_id: str,
        agent_id: str,
        title: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Save finding summary with chunked embeddings.
        
        Args:
            session_id: Research session ID
            agent_id: Agent ID who created the finding
            title: Finding title/topic
            content: Finding summary (full text, not truncated)
            metadata: Additional metadata (sources, key_findings, etc.)
            
        Returns:
            ID of saved memory record
        """
        try:
            # Split content into chunks
            chunks = self._chunk_text(content)
            
            if not chunks:
                logger.warning("No chunks generated from finding content", title=title[:100])
                chunks = [{"content": content, "chunk_index": 0, "content_hash": self._compute_hash(content)}]
            
            # Generate embeddings for all chunks in parallel
            chunk_texts = [chunk["content"] for chunk in chunks]
            # Include title in first chunk for better context
            if chunk_texts:
                chunk_texts[0] = f"{title}\n{chunk_texts[0]}"
            
            embedding_tasks = [self.embedding_provider.embed_text(text) for text in chunk_texts]
            chunk_embeddings = await asyncio.gather(*embedding_tasks)
            
            # Normalize embeddings to database dimension
            normalized_embeddings = []
            for emb in chunk_embeddings:
                if len(emb) < EMBEDDING_DIMENSION:
                    normalized_emb = list(emb) + [0.0] * (EMBEDDING_DIMENSION - len(emb))
                elif len(emb) > EMBEDDING_DIMENSION:
                    normalized_emb = emb[:EMBEDDING_DIMENSION]
                else:
                    normalized_emb = list(emb)
                normalized_embeddings.append(normalized_emb)
            
            # Generate embedding for full content (for backward compatibility)
            full_embedding = await self.embedding_provider.embed_text(f"{title}\n{content}")
            if len(full_embedding) < EMBEDDING_DIMENSION:
                full_embedding = list(full_embedding) + [0.0] * (EMBEDDING_DIMENSION - len(full_embedding))
            elif len(full_embedding) > EMBEDDING_DIMENSION:
                full_embedding = full_embedding[:EMBEDDING_DIMENSION]
            
            async with self.session_factory() as session:
                # Create memory record
                memory = ResearchMemoryModel(
                    session_id=session_id,
                    agent_id=agent_id,
                    memory_type="finding",
                    title=title,
                    content=content,  # Full content, not truncated
                    embedding=full_embedding,  # Keep for backward compatibility
                    memory_metadata=metadata or {},
                )
                session.add(memory)
                await session.flush()  # Flush to get memory.id
                
                # Create chunk records with embeddings
                chunk_models = []
                for chunk, embedding in zip(chunks, normalized_embeddings):
                    chunk_model = ResearchMemoryChunkModel(
                        memory_id=memory.id,
                        chunk_index=chunk["chunk_index"],
                        content=chunk["content"],
                        content_hash=chunk["content_hash"],
                        embedding=embedding,
                    )
                    chunk_models.append(chunk_model)
                
                session.add_all(chunk_models)
                await session.commit()
                await session.refresh(memory)
                
                logger.info("Saved research finding with chunked embeddings",
                           memory_id=memory.id,
                           session_id=session_id,
                           agent_id=agent_id,
                           title=title[:100],
                           content_length=len(content),
                           chunks_count=len(chunk_models))
                
                return memory.id
        except Exception as e:
            logger.error("Failed to save research finding", error=str(e), exc_info=True)
            raise

    async def search_memories(
        self,
        session_id: str,
        query: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search memories using vector similarity on chunks.
        
        Searches chunks first, then groups by parent memory and returns top memories
        based on their best matching chunks.
        
        Args:
            session_id: Research session ID
            query: Search query (task description or similar)
            memory_types: List of memory types to search ('note', 'finding', or both). If None, searches both.
            limit: Maximum number of memory records to return
            
        Returns:
            List of memory records with similarity scores (based on best matching chunk)
        """
        # Early return for empty query
        if not query or not query.strip():
            logger.debug("Empty query provided to search_memories, returning empty list")
            return []
        
        try:
            # Generate embedding for query
            # Most embedding providers are fast (< 1 second), but we add timeout protection
            # to prevent blocking if there's an issue with the embedding service
            try:
                # Timeout of 10 seconds should be more than enough for embedding generation
                # Most providers complete in < 1 second
                query_embedding = await asyncio.wait_for(
                    self.embedding_provider.embed_text(query),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.error("Embedding generation timed out (>10s), returning empty results",
                           query_preview=query[:100],
                           query_length=len(query))
                return []
            
            # Normalize embedding to database dimension
            if len(query_embedding) < EMBEDDING_DIMENSION:
                query_embedding = list(query_embedding) + [0.0] * (EMBEDDING_DIMENSION - len(query_embedding))
            elif len(query_embedding) > EMBEDDING_DIMENSION:
                query_embedding = query_embedding[:EMBEDDING_DIMENSION]
            
            async with self.session_factory() as session:
                # Search chunks with vector similarity
                # CRITICAL: Use ivfflat index for fast vector search (indexed in schema)
                # We search more chunks than needed (limit * 10) to ensure we get good coverage
                # when grouping by memory
                # Use .cosine_distance() method on Vector column (not function import)
                distance_expr = ResearchMemoryChunkModel.embedding.cosine_distance(query_embedding).label('distance')
                
                stmt = select(
                    ResearchMemoryChunkModel,
                    ResearchMemoryModel,  # Join with parent memory
                    distance_expr
                ).join(
                    ResearchMemoryModel,
                    ResearchMemoryChunkModel.memory_id == ResearchMemoryModel.id
                ).where(
                    ResearchMemoryModel.session_id == session_id
                )
                
                # Filter by memory types if specified
                if memory_types:
                    stmt = stmt.where(ResearchMemoryModel.memory_type.in_(memory_types))
                
                # Order by similarity (lower distance = higher similarity)
                # Search more chunks than needed for better grouping
                stmt = stmt.order_by(distance_expr).limit(limit * 10)  # Get more chunks for grouping
                
                # Execute query - should be fast with ivfflat index
                result = await session.execute(stmt)
                rows = result.all()
                
                # Group chunks by memory_id and find best match for each memory
                memory_scores = {}  # memory_id -> (best_similarity, best_chunk, memory_obj)
                
                for row in rows:
                    chunk = row[0]
                    memory = row[1]
                    distance = row[2]
                    similarity = 1.0 - distance  # Convert distance to similarity
                    
                    memory_id = memory.id
                    
                    # Keep track of best matching chunk for each memory
                    if memory_id not in memory_scores:
                        memory_scores[memory_id] = {
                            "memory": memory,
                            "best_similarity": similarity,
                            "best_chunk": chunk,
                            "all_chunks": [chunk]
                        }
                    else:
                        # Update if this chunk is better
                        if similarity > memory_scores[memory_id]["best_similarity"]:
                            memory_scores[memory_id]["best_similarity"] = similarity
                            memory_scores[memory_id]["best_chunk"] = chunk
                        memory_scores[memory_id]["all_chunks"].append(chunk)
                
                # Sort memories by best similarity and take top N
                sorted_memories = sorted(
                    memory_scores.values(),
                    key=lambda x: x["best_similarity"],
                    reverse=True
                )[:limit]
                
                # Convert to list of dicts
                memories = []
                for mem_data in sorted_memories:
                    memory = mem_data["memory"]
                    best_chunk = mem_data["best_chunk"]
                    best_similarity = mem_data["best_similarity"]
                    
                    memories.append({
                        "id": memory.id,
                        "agent_id": memory.agent_id,
                        "memory_type": memory.memory_type,
                        "title": memory.title,
                        "content": memory.content,  # Full content, not truncated
                        "metadata": memory.memory_metadata or {},
                        "similarity": best_similarity,  # Best chunk similarity
                        "best_chunk_content": best_chunk.content,  # Most relevant chunk
                        "best_chunk_index": best_chunk.chunk_index,
                        "created_at": memory.created_at.isoformat() if memory.created_at else None,
                    })
                
                logger.info("Searched research memories via chunks",
                           session_id=session_id,
                           query_preview=query[:100],
                           query_length=len(query),
                           memory_types=memory_types,
                           chunks_searched=len(rows),
                           memories_found=len(memory_scores),
                           results_count=len(memories),
                           limit=limit,
                           has_index=True)  # ivfflat index should be present
                
                return memories
        except Exception as e:
            logger.error("Failed to search research memories", error=str(e), exc_info=True)
            return []

    async def clear_session_memories(self, session_id: str) -> int:
        """Clear all memories for a session.
        
        Args:
            session_id: Research session ID
            
        Returns:
            Number of deleted records
        """
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    delete(ResearchMemoryModel).where(
                        ResearchMemoryModel.session_id == session_id
                    )
                )
                await session.commit()
                
                deleted_count = result.rowcount
                logger.info("Cleared research memories for session",
                           session_id=session_id,
                           deleted_count=deleted_count)
                
                return deleted_count
        except Exception as e:
            logger.error("Failed to clear research memories", error=str(e), exc_info=True)
            return 0

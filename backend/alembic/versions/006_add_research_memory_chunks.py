"""Add research_memory_chunks table for chunked vector search

Revision ID: 006_research_memory_chunks
Revises: 005_research_memories
Create Date: 2026-01-19 00:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = '006_research_memory_chunks'
down_revision: Union[str, None] = '005_research_memories'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Get embedding dimension from environment or use default
    import os
    embedding_dim = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    
    # Create research_memory_chunks table
    op.create_table(
        'research_memory_chunks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('memory_id', sa.Integer(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('content_hash', sa.String(64), nullable=False),
        sa.Column('embedding', Vector(embedding_dim), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['memory_id'], ['research_memories.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index('idx_research_memory_chunks_memory_id', 'research_memory_chunks', ['memory_id'])
    
    # Create vector index for embeddings (using ivfflat)
    op.execute(f"""
        CREATE INDEX idx_research_memory_chunks_embedding ON research_memory_chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    """)


def downgrade() -> None:
    # Drop indexes
    op.execute('DROP INDEX IF EXISTS idx_research_memory_chunks_embedding')
    op.drop_index('idx_research_memory_chunks_memory_id', table_name='research_memory_chunks')
    
    # Drop table
    op.drop_table('research_memory_chunks')

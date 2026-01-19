"""Add research_memories table for vector search of notes and findings

Revision ID: 005_research_memories
Revises: 004_extend_messages
Create Date: 2026-01-18 00:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = '005_research_memories'
down_revision: Union[str, None] = '004_extend_messages'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Get embedding dimension from environment or use default
    import os
    embedding_dim = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    
    # Create research_memories table
    op.create_table(
        'research_memories',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('session_id', sa.String(64), nullable=False),
        sa.Column('agent_id', sa.String(128), nullable=True),
        sa.Column('memory_type', sa.String(32), nullable=False),  # 'note' or 'finding'
        sa.Column('title', sa.String(512), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding', Vector(embedding_dim), nullable=True),
        sa.Column('metadata', sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['session_id'], ['research_sessions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index('idx_research_memories_session_id', 'research_memories', ['session_id'])
    op.create_index('idx_research_memories_agent_id', 'research_memories', ['agent_id'])
    op.create_index('idx_research_memories_type', 'research_memories', ['memory_type'])
    op.create_index('idx_research_memories_created', 'research_memories', ['created_at'])
    
    # Create vector index for embeddings (using ivfflat)
    op.execute(f"""
        CREATE INDEX idx_research_memories_embedding ON research_memories
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    """)


def downgrade() -> None:
    # Drop indexes
    op.execute('DROP INDEX IF EXISTS idx_research_memories_embedding')
    op.drop_index('idx_research_memories_created', table_name='research_memories')
    op.drop_index('idx_research_memories_type', table_name='research_memories')
    op.drop_index('idx_research_memories_agent_id', table_name='research_memories')
    op.drop_index('idx_research_memories_session_id', table_name='research_memories')
    
    # Drop table
    op.drop_table('research_memories')

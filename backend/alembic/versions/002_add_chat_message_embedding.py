"""Add embedding and fulltext search to chat messages

Revision ID: 002_chat_embedding
Revises: 001_initial
Create Date: 2026-01-02 00:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = '002_chat_embedding'
down_revision: Union[str, None] = '001_initial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add embedding column to chat_messages
    op.add_column('chat_messages',
        sa.Column('embedding', Vector(1536), nullable=True)
    )

    # Create vector index for chat message embeddings (using ivfflat)
    op.execute("""
        CREATE INDEX idx_chat_messages_embedding ON chat_messages
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    """)

    # Create fulltext search index for chat message content
    op.execute("""
        CREATE INDEX idx_chat_messages_content_fts ON chat_messages
        USING gin(to_tsvector('english', content));
    """)


def downgrade() -> None:
    # Drop fulltext search index
    op.execute('DROP INDEX IF EXISTS idx_chat_messages_content_fts')

    # Drop vector index
    op.execute('DROP INDEX IF EXISTS idx_chat_messages_embedding')

    # Drop embedding column
    op.drop_column('chat_messages', 'embedding')

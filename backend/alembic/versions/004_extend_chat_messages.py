"""Extend chat_messages with mode, session_id, original_query

Revision ID: 004_extend_messages
Revises: 003_research_sessions
Create Date: 2026-01-12 00:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '004_extend_messages'
down_revision: Union[str, None] = '003_research_sessions'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Check if columns already exist
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing_columns = [col['name'] for col in inspector.get_columns('chat_messages')]

    # Add new columns if they don't exist
    if 'mode' not in existing_columns:
        op.add_column('chat_messages', sa.Column('mode', sa.String(16), nullable=True))
    if 'session_id' not in existing_columns:
        op.add_column('chat_messages', sa.Column('session_id', sa.String(64), nullable=True))
    if 'original_query' not in existing_columns:
        op.add_column('chat_messages', sa.Column('original_query', sa.Text(), nullable=True))

    # Create indexes for efficient querying (IF NOT EXISTS)
    op.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_mode ON chat_messages(mode)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id)")

    # Add foreign key constraint to research_sessions (nullable) if not exists
    # Check if FK already exists
    existing_fks = [fk['name'] for fk in inspector.get_foreign_keys('chat_messages')]
    if 'fk_chat_messages_session' not in existing_fks:
        op.create_foreign_key(
            'fk_chat_messages_session',
            'chat_messages',
            'research_sessions',
            ['session_id'],
            ['id'],
            ondelete='SET NULL'
        )


def downgrade() -> None:
    # Drop foreign key constraint
    op.drop_constraint('fk_chat_messages_session', 'chat_messages', type_='foreignkey')

    # Drop indexes
    op.drop_index('idx_chat_messages_session_id', table_name='chat_messages')
    op.drop_index('idx_chat_messages_mode', table_name='chat_messages')

    # Drop columns
    op.drop_column('chat_messages', 'original_query')
    op.drop_column('chat_messages', 'session_id')
    op.drop_column('chat_messages', 'mode')

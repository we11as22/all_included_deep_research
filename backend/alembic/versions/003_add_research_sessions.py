"""Add research_sessions table for deep research session management

Revision ID: 003_research_sessions
Revises: 002_chat_embedding
Create Date: 2026-01-12 00:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '003_research_sessions'
down_revision: Union[str, None] = '002_chat_embedding'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop old research_sessions table from 001 migration (it has wrong structure)
    # New structure needs chat_id, original_query, and other fields
    op.drop_table('research_sessions')

    # Create new research_sessions table with correct structure
    op.create_table(
        'research_sessions',
        sa.Column('id', sa.String(64), primary_key=True),
        sa.Column('chat_id', sa.String(64), nullable=False),
        sa.Column('original_query', sa.Text(), nullable=False),
        sa.Column('mode', sa.String(16), nullable=False),
        sa.Column('status', sa.String(32), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('deep_search_result', sa.Text(), nullable=True),
        sa.Column('clarification_answers', sa.Text(), nullable=True),
        sa.Column('draft_report', sa.Text(), nullable=True),
        sa.Column('final_report', sa.Text(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), server_default='{}', nullable=False),
        sa.ForeignKeyConstraint(['chat_id'], ['chats.id'], ondelete='CASCADE'),
    )

    # Create regular indexes
    op.create_index('idx_research_sessions_chat_id', 'research_sessions', ['chat_id'])
    op.create_index('idx_research_sessions_created', 'research_sessions', ['created_at'])
    op.create_index('idx_research_sessions_status', 'research_sessions', ['status'])

    # CRITICAL: Create unique partial index to ensure only one active session per chat
    # This enforces the business rule: 1 chat_id = max 1 active deep_research session
    op.execute("""
        CREATE UNIQUE INDEX idx_one_active_session_per_chat
        ON research_sessions(chat_id)
        WHERE status IN ('active', 'waiting_clarification', 'researching');
    """)


def downgrade() -> None:
    # Drop unique partial index
    op.execute('DROP INDEX IF EXISTS idx_one_active_session_per_chat')

    # Drop regular indexes
    op.drop_index('idx_research_sessions_status', table_name='research_sessions')
    op.drop_index('idx_research_sessions_created', table_name='research_sessions')
    op.drop_index('idx_research_sessions_chat_id', table_name='research_sessions')

    # Drop table
    op.drop_table('research_sessions')

"""Initial schema migration

Revision ID: 001_initial
Revises: 
Create Date: 2025-12-29 23:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Note: pgvector extension is created in env.py before migrations
    # Create memory_files table
    op.create_table(
        'memory_files',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('file_path', sa.String(length=512), nullable=False),
        sa.Column('title', sa.String(length=256), nullable=False),
        sa.Column('category', sa.String(length=64), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('file_hash', sa.String(length=64), nullable=False),
        sa.Column('word_count', sa.Integer(), nullable=True),
        sa.Column('tags', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('file_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_memory_files_category', 'memory_files', ['category'], unique=False)
    op.create_index('idx_memory_files_updated', 'memory_files', ['updated_at'], unique=False)
    op.create_index(op.f('ix_memory_files_id'), 'memory_files', ['id'], unique=False)
    op.create_index(op.f('ix_memory_files_file_path'), 'memory_files', ['file_path'], unique=True)

    # Create memory_chunks table
    op.create_table(
        'memory_chunks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('file_id', sa.Integer(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('content_hash', sa.String(length=64), nullable=False),
        sa.Column('embedding', Vector(1536), nullable=True),
        sa.Column('header_path', postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column('section_level', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['file_id'], ['memory_files.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_memory_chunks_file_id', 'memory_chunks', ['file_id'], unique=False)
    op.create_index(op.f('ix_memory_chunks_id'), 'memory_chunks', ['id'], unique=False)
    
    # Create vector index for embeddings (using ivfflat)
    op.execute("""
        CREATE INDEX idx_memory_chunks_embedding ON memory_chunks 
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    """)
    
    # Create fulltext search index
    op.execute("""
        CREATE INDEX idx_memory_chunks_content_fts ON memory_chunks 
        USING gin(to_tsvector('english', content));
    """)

    # Create research_sessions table
    op.create_table(
        'research_sessions',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('mode', sa.String(length=16), nullable=False),
        sa.Column('query', sa.Text(), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('final_report', sa.Text(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_research_sessions_status', 'research_sessions', ['status'], unique=False)
    op.create_index('idx_research_sessions_created', 'research_sessions', ['created_at'], unique=False)

    # Create chats table
    op.create_table(
        'chats',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('title', sa.String(length=256), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_chats_created', 'chats', ['created_at'], unique=False)
    op.create_index('idx_chats_updated', 'chats', ['updated_at'], unique=False)

    # Create chat_messages table
    op.create_table(
        'chat_messages',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('chat_id', sa.String(length=64), nullable=False),
        sa.Column('message_id', sa.String(length=64), nullable=False),
        sa.Column('role', sa.String(length=16), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['chat_id'], ['chats.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_chat_messages_chat_id', 'chat_messages', ['chat_id'], unique=False)
    op.create_index('idx_chat_messages_created', 'chat_messages', ['created_at'], unique=False)
    op.create_index(op.f('ix_chat_messages_id'), 'chat_messages', ['id'], unique=False)
    op.create_index(op.f('ix_chat_messages_message_id'), 'chat_messages', ['message_id'], unique=False)


def downgrade() -> None:
    # Note: We don't drop the vector extension as it might be used by other databases
    # op.execute('DROP EXTENSION IF EXISTS vector')
    
    op.drop_index('idx_chat_messages_created', table_name='chat_messages')
    op.drop_index('idx_chat_messages_chat_id', table_name='chat_messages')
    op.drop_index(op.f('ix_chat_messages_message_id'), table_name='chat_messages')
    op.drop_index(op.f('ix_chat_messages_id'), table_name='chat_messages')
    op.drop_table('chat_messages')
    
    op.drop_index('idx_chats_updated', table_name='chats')
    op.drop_index('idx_chats_created', table_name='chats')
    op.drop_table('chats')
    
    op.drop_index('idx_research_sessions_created', table_name='research_sessions')
    op.drop_index('idx_research_sessions_status', table_name='research_sessions')
    op.drop_table('research_sessions')
    
    op.execute('DROP INDEX IF EXISTS idx_memory_chunks_content_fts')
    op.execute('DROP INDEX IF EXISTS idx_memory_chunks_embedding')
    op.drop_index(op.f('ix_memory_chunks_id'), table_name='memory_chunks')
    op.drop_index('idx_memory_chunks_file_id', table_name='memory_chunks')
    op.drop_table('memory_chunks')
    
    op.drop_index(op.f('ix_memory_files_file_path'), table_name='memory_files')
    op.drop_index(op.f('ix_memory_files_id'), table_name='memory_files')
    op.drop_index('idx_memory_files_updated', table_name='memory_files')
    op.drop_index('idx_memory_files_category', table_name='memory_files')
    op.drop_table('memory_files')


"""Migration script from PostgreSQL to SQLite.

Usage:
    python -m scripts.migrate_to_sqlite

This script:
1. Exports chats, messages, and research sessions from PostgreSQL
2. Imports them into SQLite
3. Migrates memory files metadata (but not embeddings)
4. Embeddings are re-indexed in the new vector store

Note: Run this from the backend directory with .env configured for PostgreSQL.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

import asyncpg
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_settings
from src.database.schema import ChatModel as PGChatModel, ChatMessageModel as PGMessageModel, ResearchSessionModel as PGResearchModel
from src.database.schema_sqlite import (
    ChatModel as SQLiteChatModel,
    ChatMessageModel as SQLiteMessageModel,
    ResearchSessionModel as SQLiteResearchModel,
    MemoryFileModel as SQLiteMemoryFileModel,
    create_tables
)
from src.database.connection_sqlite import create_sync_sqlite_engine

logger = structlog.get_logger(__name__)


async def migrate_chats(pg_session: AsyncSession, sqlite_engine):
    """Migrate chats from PostgreSQL to SQLite."""
    logger.info("Migrating chats...")

    # Fetch all chats from PostgreSQL
    result = await pg_session.execute(select(PGChatModel))
    pg_chats = result.scalars().all()

    logger.info(f"Found {len(pg_chats)} chats to migrate")

    # Insert into SQLite (sync)
    from sqlalchemy.orm import Session

    with Session(sqlite_engine) as session:
        for pg_chat in pg_chats:
            sqlite_chat = SQLiteChatModel(
                id=pg_chat.id,
                title=pg_chat.title,
                created_at=pg_chat.created_at.isoformat() if pg_chat.created_at else datetime.now().isoformat(),
                updated_at=pg_chat.updated_at.isoformat() if pg_chat.updated_at else datetime.now().isoformat(),
                metadata=str(pg_chat.chat_metadata or {}),
            )
            session.merge(sqlite_chat)  # merge to handle duplicates

        session.commit()

    logger.info(f"Migrated {len(pg_chats)} chats")


async def migrate_messages(pg_session: AsyncSession, sqlite_engine):
    """Migrate chat messages from PostgreSQL to SQLite."""
    logger.info("Migrating messages...")

    # Fetch all messages from PostgreSQL
    result = await pg_session.execute(select(PGMessageModel))
    pg_messages = result.scalars().all()

    logger.info(f"Found {len(pg_messages)} messages to migrate")

    # Insert into SQLite (sync)
    from sqlalchemy.orm import Session

    with Session(sqlite_engine) as session:
        for pg_msg in pg_messages:
            sqlite_msg = SQLiteMessageModel(
                chat_id=pg_msg.chat_id,
                message_id=pg_msg.message_id,
                role=pg_msg.role,
                content=pg_msg.content,
                created_at=pg_msg.created_at.isoformat() if pg_msg.created_at else datetime.now().isoformat(),
                metadata=str(pg_msg.message_metadata or {}),
            )
            session.add(sqlite_msg)

        session.commit()

    logger.info(f"Migrated {len(pg_messages)} messages")


async def migrate_research_sessions(pg_session: AsyncSession, sqlite_engine):
    """Migrate research sessions from PostgreSQL to SQLite."""
    logger.info("Migrating research sessions...")

    # Fetch all research sessions from PostgreSQL
    result = await pg_session.execute(select(PGResearchModel))
    pg_sessions = result.scalars().all()

    logger.info(f"Found {len(pg_sessions)} research sessions to migrate")

    # Insert into SQLite (sync)
    from sqlalchemy.orm import Session

    with Session(sqlite_engine) as session:
        for pg_sess in pg_sessions:
            sqlite_sess = SQLiteResearchModel(
                id=pg_sess.id,
                mode=pg_sess.mode,
                query=pg_sess.query,
                status=pg_sess.status,
                created_at=pg_sess.created_at.isoformat() if pg_sess.created_at else datetime.now().isoformat(),
                completed_at=pg_sess.completed_at.isoformat() if pg_sess.completed_at else None,
                final_report=pg_sess.final_report,
                metadata=str(pg_sess.session_metadata or {}),
            )
            session.merge(sqlite_sess)

        session.commit()

    logger.info(f"Migrated {len(pg_sessions)} research sessions")


async def migrate_memory_files(pg_session: AsyncSession, sqlite_engine):
    """Migrate memory file metadata (embeddings will be re-indexed)."""
    logger.info("Migrating memory file metadata...")

    from src.database.schema import MemoryFileModel as PGMemoryFileModel

    # Fetch all memory files from PostgreSQL
    result = await pg_session.execute(select(PGMemoryFileModel))
    pg_files = result.scalars().all()

    logger.info(f"Found {len(pg_files)} memory files to migrate")

    # Insert into SQLite (sync)
    from sqlalchemy.orm import Session

    with Session(sqlite_engine) as session:
        for pg_file in pg_files:
            sqlite_file = SQLiteMemoryFileModel(
                file_path=pg_file.file_path,
                title=pg_file.title,
                category=pg_file.category,
                created_at=pg_file.created_at.isoformat() if pg_file.created_at else datetime.now().isoformat(),
                updated_at=pg_file.updated_at.isoformat() if pg_file.updated_at else datetime.now().isoformat(),
                file_hash=pg_file.file_hash,
                word_count=pg_file.word_count,
                tags=str(pg_file.tags or []),
                metadata=str(pg_file.file_metadata or {}),
            )
            session.add(sqlite_file)

        session.commit()

    logger.info(f"Migrated {len(pg_files)} memory files (embeddings need re-indexing)")


async def run_migration():
    """Run full migration from PostgreSQL to SQLite."""
    logger.info("Starting migration from PostgreSQL to SQLite")

    settings = get_settings()

    # Ensure PostgreSQL is configured
    if not settings.postgres_password:
        logger.error("PostgreSQL password not configured. Set POSTGRES_PASSWORD in .env")
        return

    # Create PostgreSQL connection
    pg_engine = create_async_engine(settings.database_url)
    pg_session_factory = sessionmaker(pg_engine, class_=AsyncSession, expire_on_commit=False)

    # Create SQLite engine and tables
    sqlite_engine = create_sync_sqlite_engine(settings.sqlite_db_path)
    from src.database.schema_sqlite import Base
    Base.metadata.create_all(sqlite_engine)
    logger.info(f"Created SQLite database at {settings.sqlite_db_path}")

    try:
        async with pg_session_factory() as pg_session:
            # Run migrations
            await migrate_chats(pg_session, sqlite_engine)
            await migrate_messages(pg_session, sqlite_engine)
            await migrate_research_sessions(pg_session, sqlite_engine)
            await migrate_memory_files(pg_session, sqlite_engine)

        logger.info("Migration completed successfully!")
        logger.info(f"SQLite database: {settings.sqlite_db_path}")
        logger.info("Next steps:")
        logger.info("1. Update .env to use_postgres=false")
        logger.info("2. Re-index memory embeddings with: python -m scripts.reindex_embeddings")
        logger.info("3. Test the application with SQLite")

    except Exception as e:
        logger.error("Migration failed", error=str(e), exc_info=True)
        raise
    finally:
        await pg_engine.dispose()


if __name__ == "__main__":
    asyncio.run(run_migration())

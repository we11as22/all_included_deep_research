"""SQLite database connection management with async support.

Uses aiosqlite for async operations and SQLAlchemy for ORM.
"""

import structlog
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from src.config.settings import Settings

logger = structlog.get_logger(__name__)


class SQLiteDatabaseManager:
    """Manages SQLite database connections."""

    def __init__(self, settings: Settings):
        """Initialize SQLite database manager."""
        self.settings = settings
        self.engine: AsyncEngine | None = None
        self.session_factory: async_sessionmaker | None = None
        self.db_path = Path(settings.sqlite_db_path)

    async def init_engine(self) -> None:
        """Initialize SQLAlchemy async engine for SQLite."""
        try:
            # Ensure database directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Create async engine with aiosqlite
            # Format: sqlite+aiosqlite:///path/to/db.sqlite
            db_url = f"sqlite+aiosqlite:///{self.db_path}"

            self.engine = create_async_engine(
                db_url,
                echo=self.settings.debug,
                poolclass=NullPool,  # SQLite doesn't need connection pooling
                connect_args={
                    "check_same_thread": False,  # Allow multi-threaded access
                    "timeout": 30,  # Wait up to 30s for locks
                },
            )

            # Enable foreign keys for SQLite
            @event.listens_for(self.engine.sync_engine, "connect")
            def set_sqlite_pragma(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
                cursor.execute("PRAGMA synchronous=NORMAL")  # Balance safety/speed
                cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
                cursor.execute("PRAGMA temp_store=MEMORY")  # Temp tables in RAM
                cursor.close()

            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )

            logger.info(
                "SQLite engine initialized",
                db_path=str(self.db_path),
                debug=self.settings.debug,
            )
        except Exception as e:
            logger.error("Failed to initialize SQLite engine", error=str(e))
            raise

    async def close_engine(self) -> None:
        """Close SQLAlchemy engine."""
        if self.engine:
            await self.engine.dispose()
            logger.info("SQLite engine closed")

    def get_session(self) -> AsyncSession:
        """Get SQLAlchemy async session."""
        if not self.session_factory:
            raise RuntimeError("Database engine not initialized")
        return self.session_factory()

    async def create_tables(self) -> None:
        """Create all tables in the database."""
        from src.database.schema_sqlite import Base

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created")

    async def drop_tables(self) -> None:
        """Drop all tables from the database."""
        from src.database.schema_sqlite import Base

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.info("Database tables dropped")


def create_sqlite_engine(settings: Settings) -> AsyncEngine:
    """Create SQLite AsyncEngine (factory function)."""
    db_path = Path(settings.sqlite_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db_url = f"sqlite+aiosqlite:///{db_path}"

    engine = create_async_engine(
        db_url,
        echo=settings.debug,
        poolclass=NullPool,
        connect_args={
            "check_same_thread": False,
            "timeout": 30,
        },
    )

    # Enable foreign keys and optimizations
    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=-64000")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.close()

    return engine


def create_sqlite_session_factory(engine: AsyncEngine) -> async_sessionmaker:
    """Create SQLAlchemy session factory for SQLite."""
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


async def get_db_session(
    db_manager: SQLiteDatabaseManager,
) -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database session (FastAPI dependency)."""
    async with db_manager.get_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ==================== Sync Session for Migrations ====================


def create_sync_sqlite_engine(db_path: str | Path):
    """Create synchronous SQLite engine for migrations/scripts."""
    from sqlalchemy import create_engine

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db_url = f"sqlite:///{db_path}"

    engine = create_engine(
        db_url,
        echo=False,
        connect_args={"check_same_thread": False},
    )

    # Enable foreign keys
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

    return engine

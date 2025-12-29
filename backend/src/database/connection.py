"""Database connection management with asyncpg."""

import asyncpg
import structlog
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.config.settings import Settings

logger = structlog.get_logger(__name__)


class DatabaseManager:
    """Manages database connections and pooling."""

    def __init__(self, settings: Settings):
        """Initialize database manager."""
        self.settings = settings
        self.pool: asyncpg.Pool | None = None
        self.engine: AsyncEngine | None = None
        self.session_factory: sessionmaker | None = None

    async def init_pool(self) -> None:
        """Initialize asyncpg connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.settings.postgres_host,
                port=self.settings.postgres_port,
                database=self.settings.postgres_db,
                user=self.settings.postgres_user,
                password=self.settings.postgres_password,
                min_size=2,
                max_size=self.settings.database_pool_size,
                command_timeout=60,
            )
            logger.info("Database pool initialized", pool_size=self.settings.database_pool_size)
        except Exception as e:
            logger.error("Failed to initialize database pool", error=str(e))
            raise

    async def init_engine(self) -> None:
        """Initialize SQLAlchemy async engine."""
        try:
            self.engine = create_async_engine(
                self.settings.database_url,
                echo=self.settings.debug,
                pool_size=self.settings.database_pool_size,
                max_overflow=10,
            )

            self.session_factory = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )

            logger.info("SQLAlchemy engine initialized")
        except Exception as e:
            logger.error("Failed to initialize SQLAlchemy engine", error=str(e))
            raise

    async def close_pool(self) -> None:
        """Close asyncpg connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database pool closed")

    async def close_engine(self) -> None:
        """Close SQLAlchemy engine."""
        if self.engine:
            await self.engine.dispose()
            logger.info("SQLAlchemy engine closed")

    def get_session(self) -> AsyncSession:
        """Get SQLAlchemy async session."""
        if not self.session_factory:
            raise RuntimeError("Database engine not initialized")
        return self.session_factory()


def create_database_engine(settings: Settings) -> AsyncEngine:
    """Create SQLAlchemy async engine."""
    return create_async_engine(
        settings.database_url,
        echo=settings.debug,
        pool_size=settings.database_pool_size,
        max_overflow=10,
    )


def create_session_factory(engine: AsyncEngine) -> sessionmaker:
    """Create SQLAlchemy session factory."""
    return sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


async def create_db_pool(settings: Settings) -> asyncpg.Pool:
    """Create asyncpg pool for raw SQL access (hybrid search)."""
    return await asyncpg.create_pool(
        host=settings.postgres_host,
        port=settings.postgres_port,
        database=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
        min_size=2,
        max_size=settings.database_pool_size,
        command_timeout=60,
    )


async def get_db_session(db_manager: DatabaseManager) -> AsyncSession:
    """Dependency for getting database session."""
    async with db_manager.get_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

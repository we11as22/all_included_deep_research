#!/bin/sh
set -e

cd /app

# Initialize database based on backend type
if [ "${USE_POSTGRES}" = "true" ]; then
  echo "Running PostgreSQL migrations..."
  alembic upgrade head
else
  echo "Initializing SQLite database..."
  python -c "
import asyncio
from src.database.connection_sqlite import SQLiteDatabaseManager
from src.database.schema_sqlite import Base
from src.config.settings import get_settings

async def init_db():
    settings = get_settings()
    db = SQLiteDatabaseManager(settings)
    await db.init_engine()
    await db.create_tables()
    await db.close_engine()
    print('SQLite database initialized successfully')

asyncio.run(init_db())
"
fi

echo "Starting backend server..."
exec python -m src


#!/bin/sh
set -e

echo "Starting SearXNG with uwsgi (production-ready for high load)..."

# Start SearXNG with uwsgi (production-ready, handles high load)
# Use system uwsgi with virtualenv specified in config
sudo -H -u searxng bash -c "cd /usr/local/searxng/searxng-src && export SEARXNG_SETTINGS_PATH='/etc/searxng/settings.yml' && /usr/bin/uwsgi --ini /etc/searxng/uwsgi.ini --http-socket 0.0.0.0:8080" &
SEARXNG_PID=$!

echo "Waiting for SearXNG to be ready..."
sleep 5

COUNTER=0
MAX_TRIES=30
until curl -s http://localhost:8080 > /dev/null 2>&1; do
  COUNTER=$((COUNTER+1))
  if [ $COUNTER -ge $MAX_TRIES ]; then
    echo "Warning: SearXNG health check timeout, but continuing..."
    break
  fi
  sleep 1
done

if curl -s http://localhost:8080 > /dev/null 2>&1; then
  echo "SearXNG started successfully (PID: $SEARXNG_PID)"
else
  echo "SearXNG may not be fully ready, but continuing (PID: $SEARXNG_PID)"
fi

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


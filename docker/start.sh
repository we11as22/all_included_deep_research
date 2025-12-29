#!/bin/bash

# Start All-Included Deep Research

echo "🚀 Starting All-Included Deep Research..."

# Check if root .env exists
if [ ! -f ../.env ]; then
    echo "⚠️  .env file not found in project root. Creating from .env.example..."
    cp ../.env.example ../.env
    echo "📝 Please edit .env file with your configuration before continuing."
    exit 1
fi

# Check if backend .env exists
if [ ! -f ../backend/.env ]; then
    echo "⚠️  backend/.env file not found. Creating from .env.example..."
    cp ../backend/.env.example ../backend/.env
    echo "📝 Please edit backend/.env file with your API keys before continuing."
    exit 1
fi

echo "🐳 Starting Docker containers..."
docker compose -f ../docker-compose.yml up -d

echo "⏳ Waiting for services to be ready..."
sleep 10

echo "✅ Services started!"
echo ""
echo "📍 Access points:"
echo "   - Frontend: http://localhost:3000"
echo "   - Backend API: http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"
echo ""
echo "📊 Check logs:"
echo "   docker compose -f ../docker-compose.yml logs -f"
echo ""
echo "🛑 Stop services:"
echo "   ./stop.sh"

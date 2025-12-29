#!/bin/bash

# Stop All-Included Deep Research

echo "🛑 Stopping All-Included Deep Research..."

docker compose -f ../docker-compose.yml down

echo "✅ Services stopped!"

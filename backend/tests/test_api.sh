#!/bin/bash

echo "Testing All-Included Deep Research API with Real LLM"
echo "======================================================"

# Test 1: Create chat
echo -e "\n1. Creating new chat..."
CHAT_RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/chats \
  -H "Content-Type: application/json" \
  -d '{"title": "Test Python Query"}')

CHAT_ID=$(echo $CHAT_RESPONSE | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
echo "Chat ID: $CHAT_ID"

# Test 2: Simple query (should use chat mode)
echo -e "\n2. Testing simple query (chat mode - no sources)..."
curl -X POST "http://localhost:8000/api/v1/chats/$CHAT_ID/messages" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is 2+2?",
    "stream": false
  }'

echo -e "\n\n3. Testing web search query (speed mode: 2 iterations)..."
sleep 2
curl -X POST "http://localhost:8000/api/v1/chats/$CHAT_ID/messages" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Python programming language?",
    "mode": "web_search",
    "stream": false
  }'

echo -e "\n\nDone! Check logs with: docker logs deep_research_backend 2>&1 | tail -200"

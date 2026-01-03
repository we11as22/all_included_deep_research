# All-Included Deep Research

Comprehensive deep research system with memory integration, combining the best practices from multiple open-source projects.

## ✨ Features

### 3 Chat Modes

- **Web Search** (speed): Fast query rewriting + multi-query search with citations (2 iterations)
- **Deep Search** (balanced): Quality web search with extra iterations and broader sources (6 iterations)
- **Deep Research** (quality): Full multi-agent LangGraph system with supervisor coordination (up to 25 iterations)
  - Supervisor agent (LangGraph with ReAct) coordinates 4 researcher agents
  - Each researcher is a LangGraph agent with planning/replanning
  - All agents write notes to markdown files in session folder
  - Supervisor reviews progress and assigns new tasks through queue system
  - Final report validated and streamed to frontend

### Advanced Memory System

- **Dual Storage**: Human-readable Markdown files + PostgreSQL with pgvector
- **Hybrid Search**: RRF (Reciprocal Rank Fusion) combining vector and fulltext search
- **Smart Chunking**: Markdown-aware chunking with header context preservation
- **Auto-sync**: Automatic synchronization between files and database
  - Memory files location configurable via `MEMORY_DIR` (default: `/home/asudakov/projects/memory_files`)
- **Chat Context**: Agents receive the last N chat messages (`CHAT_HISTORY_LIMIT`)
- **Agent Sessions**: Deep Research creates temporary session folders with agent memory files
  - `main.md` - supervisor's research document
  - `agents/{agent_id}.md` - each agent's todos and notes
  - `items/` - research notes with sources
  - Automatically cleaned up after research completion

### ⚙️ Flexible Configuration

- **Search Providers**: Tavily or SearXNG (self-hosted)
- **Embedding Providers**: OpenAI, Ollama (local), Cohere, or HuggingFace
- **LLM Models**: OpenAI GPT-4, Anthropic Claude, OpenRouter, and any OpenAI-compatible API
- **Fully Configurable**: All settings via `.env` files

### 🛠 Modern Tech Stack

- **Backend**: FastAPI + LangGraph workflows + Pydantic v2
- **Frontend**: Next.js 14 + React + TypeScript + Tailwind CSS
- **Database**: PostgreSQL 16 + pgvector extension
- **Real-time**: SSE streaming for live research updates
- **Deployment**: Docker Compose for easy setup

## 🚀 Quick Start

### Prerequisites

- **Docker & Docker Compose** (recommended) OR
- Python 3.11+ and Node.js 18+ for manual setup
- API keys: OpenAI (required) and Tavily (recommended)

### Option 1: Docker Compose (Recommended) ⭐

```bash
# 1. Navigate to project directory
cd /root/asudakov/projects/all_included_deep_research

# 2. Configure backend environment
cd backend
cp .env.example .env
# Edit .env and add your API keys:
# - POSTGRES_PASSWORD (set a secure password)
# - OPENAI_API_KEY (required)
# - TAVILY_API_KEY or SEARCH_PROVIDER=searxng (choose one)
# - LLM_MODE=mock to run without external LLMs (for testing)
# - SEARCH_PROVIDER=mock to run without external search (for testing)
# - CHAT_HISTORY_LIMIT to control how many recent chat messages are injected

# 3. Configure Docker environment (project root)
cd ..
cp .env.example .env
# Edit .env and set POSTGRES_PASSWORD and MEMORY_HOST_PATH if needed

# 4. Start all services
docker compose up -d

# 5. Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs

# 6. Stop services when done
docker compose down
```

**First-time setup takes 2-3 minutes** while Docker builds images and initializes the database.

**Note**: The Docker build process has been optimized - source code is now copied before package installation to ensure proper package setup.

### Option 2: Manual Setup (Development)

#### Prerequisites
- PostgreSQL 16 with pgvector extension installed
- Python 3.11+
- Node.js 18+

#### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with your settings:
# - Set POSTGRES_* variables to match your local PostgreSQL
# - Add OPENAI_API_KEY
# - Add TAVILY_API_KEY

# Run database migrations
alembic upgrade head

# Start the backend
python -m src
# Or with auto-reload: uvicorn src.api.app:app --reload
```

#### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure environment (optional)
# cp .env.local.example .env.local
# Default API URL is http://localhost:8000

# Start the frontend
npm run dev
```

#### Access the Application

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Project Structure

```
all_included_deep_research/
├── backend/              # Python FastAPI + LangGraph
│   ├── src/
│   │   ├── api/         # FastAPI endpoints
│   │   ├── workflow/    # LangGraph workflows
│   │   ├── memory/      # Memory system
│   │   ├── search/      # Search providers
│   │   ├── embeddings/  # Embedding providers
│   │   ├── streaming/   # SSE streaming
│   │   ├── database/    # Database layer
│   │   └── config/      # Configuration
│   ├── tests/
│   └── pyproject.toml
├── frontend/            # Next.js + React + TypeScript
│   ├── src/
│   │   ├── app/        # Next.js App Router
│   │   ├── components/ # React components
│   │   └── lib/        # API client, utilities
│   └── package.json
├── docker/              # Helper scripts
└── docker-compose.yml   # Full stack Compose
```

## Architecture Highlights

### LangGraph Workflows

The system uses LangGraph to orchestrate complex research workflows:

```
User Query → Memory Search → Research Brief → Supervisor
                                                  ↓
                                     Parallel Researchers (1-5)
                                                  ↓
                                     Compress Findings
                                                  ↓
                                     Final Report → Save to Memory
```

### Memory Integration

- Research context is retrieved from memory before starting
- Findings are automatically saved to memory after completion
- Hybrid search (semantic + keyword) for optimal retrieval

### Real-time Streaming

- OpenAI-compatible `/v1/chat/completions` endpoint
- SSE streaming with research/source/reasoning blocks
- SessionManager pattern for UI state management

## ⚙️ Configuration

### Required Settings

Edit `backend/.env`:

```bash
# Database (required)
POSTGRES_PASSWORD=your_secure_password

# OpenAI (required for live LLM and embeddings)
OPENAI_API_KEY=sk-your-openai-api-key

# Tavily (required for web search)
TAVILY_API_KEY=tvly-your-tavily-api-key
```

### OpenAI-Compatible APIs (OpenRouter, 302.AI, etc.)

You can use any OpenAI-compatible API by setting the base URL:

```bash
# Use OpenRouter instead of OpenAI
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=sk-or-v1-your-openrouter-key

# Optional: Custom headers for OpenRouter
OPENAI_API_HTTP_REFERER=https://github.com/your-org/your-repo
OPENAI_API_X_TITLE=Your App Name

# Use any other OpenAI-compatible API
OPENAI_BASE_URL=https://api.302.ai/v1
OPENAI_API_KEY=your-api-key
```

**Note**: For OpenRouter, default headers are automatically added if not specified. For other APIs, you may need to set custom headers if required.

### Optional Settings

```bash
# Use Anthropic Claude instead of GPT
ANTHROPIC_API_KEY=sk-ant-your-key
RESEARCH_MODEL=anthropic:claude-3-5-sonnet-20241022

# Use OpenRouter or other OpenAI-compatible APIs
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=sk-or-v1-your-key
# Configure models to use OpenRouter models:
CHAT_MODEL=openai:gpt-4o-mini
RESEARCH_MODEL=openai:gpt-4o
# Or use other models available on OpenRouter:
# RESEARCH_MODEL=openai:qwen/qwen-2.5-72b-instruct
# RESEARCH_MODEL=openai:anthropic/claude-3.5-sonnet
# See https://openrouter.ai/models for all available models

# Use local Ollama for embeddings (free!)
# Start the container with: docker compose --profile local-embeddings up -d
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Use local HuggingFace embeddings (no API key required)
EMBEDDING_PROVIDER=huggingface
HUGGINGFACE_MODEL=sentence-transformers/all-MiniLM-L6-v2
HUGGINGFACE_USE_LOCAL=true

# Offline testing without LLM/search
LLM_MODE=mock
SEARCH_PROVIDER=mock
EMBEDDING_PROVIDER=mock

# Use SearXNG instead of Tavily (self-hosted, free!)
SEARCH_PROVIDER=searxng
SEARXNG_INSTANCE_URL=http://localhost:8080

# Adjust research depth
BALANCED_MAX_ITERATIONS=10  # Default: 6
QUALITY_MAX_CONCURRENT=8    # Default: 5

# Include last N chat messages in prompts
CHAT_HISTORY_LIMIT=2

# Quality deep search tuning
DEEP_SEARCH_QUALITY_MAX_RESULTS=16
DEEP_SEARCH_QUALITY_QUERIES=6
DEEP_SEARCH_QUALITY_SCRAPE_TOP_N=8
DEEP_SEARCH_QUALITY_RERANK_TOP_K=12
DEEP_SEARCH_QUALITY_ITERATIONS=3
```

### Frontend Configuration

Edit `frontend/.env.local` (optional):

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_DEFAULT_MODE=search
```

## API Documentation

Once the backend is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Key Endpoints

- `POST /v1/chat/completions` - OpenAI-compatible chat endpoint (model: `search`/`web_search`, `deep_search`, `deep_research`)
- `POST /api/chat/stream` - Chat with progress events (SSE)
- `POST /api/research` - Start a research session
- `GET /api/memory` - List memory files
- `POST /api/memory` - Create memory file
- `GET /api/config` - Get configuration

## Development

### Running Tests

Backend:
```bash
cd backend
pytest
```

Frontend:
```bash
cd frontend
npm test
```

### Code Quality

Backend:
```bash
cd backend
ruff check .
ruff format .
mypy src
```

Frontend:
```bash
cd frontend
npm run lint
npm run type-check
```

## 🎓 Best Practices Integrated

This project combines the best features from leading open-source projects:

| Project | Feature Adopted |
|---------|----------------|
| **multifile-markdown-mcp** | Hybrid memory system with RRF (Reciprocal Rank Fusion) search |
| **open_deep_research** | Supervisor pattern with parallel researchers |
| **sgr-agent-core** | OpenAI-compatible API with SSE streaming |
| **OpenDeepSearch** | Search provider abstraction and reranking |
| **Perplexica** | Mode-based iteration limits and streaming UI blocks |

## 📊 How It Works

### Research Workflow

```
User Query → Memory Search → Research Planning
                                    ↓
                    Parallel Researchers (1-5 concurrent)
                                    ↓
                    Source Discovery & Analysis
                                    ↓
                    Findings Compression & Synthesis
                                    ↓
                    Final Report Generation
                                    ↓
                    Save to Memory (optional)
```

### Memory System

1. **Markdown Files**: Human-readable research notes (location configurable via `MEMORY_DIR`)
2. **Vector Database**: PostgreSQL + pgvector for semantic search
3. **Hybrid Search**: Combines vector similarity + fulltext search with RRF
4. **Auto-sync**: Changes to files automatically update the database

## License

MIT

## Contributing

Contributions are welcome! Please read our contributing guidelines first.

## Support

For issues and questions, please use the GitHub issue tracker.

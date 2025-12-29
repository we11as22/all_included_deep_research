# All-Included Deep Research

Comprehensive deep research system with memory integration, combining the best practices from multiple open-source projects.

## ✨ Features

### 3 Chat Modes

- **Simple Search**: Fast web lookup with citations
- **Deep Search**: Multi-query search + reranking + source summaries
- **Deep Research**: Full multi-agent report synthesis with planning and compression

### Advanced Memory System

- **Dual Storage**: Human-readable Markdown files + PostgreSQL with pgvector
- **Hybrid Search**: RRF (Reciprocal Rank Fusion) combining vector and fulltext search
- **Smart Chunking**: Markdown-aware chunking with header context preservation
- **Auto-sync**: Automatic synchronization between files and database
  - Memory files live at `/home/asudakov/projects/memory_files` by default

### ⚙️ Flexible Configuration

- **Search Providers**: Tavily or SearXNG (self-hosted)
- **Embedding Providers**: OpenAI, Ollama (local), Cohere, or HuggingFace
- **LLM Models**: OpenAI GPT-4, Anthropic Claude, and others
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
cd /home/asudakov/projects/all_included_search/all_included_deep_research

# 2. Configure backend environment
cd backend
cp .env.example .env
# Edit .env and add your API keys:
# - POSTGRES_PASSWORD (set a secure password)
# - OPENAI_API_KEY (required)
# - TAVILY_API_KEY or SEARCH_PROVIDER=searxng (choose one)
# - LLM_MODE=mock to run without external LLMs (for testing)
# - SEARCH_PROVIDER=mock to run without external search (for testing)

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

### Optional Settings

```bash
# Use Anthropic Claude instead of GPT
ANTHROPIC_API_KEY=sk-ant-your-key
RESEARCH_MODEL=anthropic:claude-3-5-sonnet-20241022

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

- `POST /v1/chat/completions` - OpenAI-compatible chat endpoint (model: `search`, `deep_search`, `deep_research`)
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

1. **Markdown Files**: Human-readable research notes in `/home/asudakov/projects/memory_files/`
2. **Vector Database**: PostgreSQL + pgvector for semantic search
3. **Hybrid Search**: Combines vector similarity + fulltext search with RRF
4. **Auto-sync**: Changes to files automatically update the database

## License

MIT

## Contributing

Contributions are welcome! Please read our contributing guidelines first.

## Support

For issues and questions, please use the GitHub issue tracker.

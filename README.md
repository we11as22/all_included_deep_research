# All-Included Deep Research

Полноценная система глубокого исследования с интеграцией памяти, объединяющая лучшие практики из множества open-source проектов.

## 📚 Документация

- **[QUICKSTART.md](QUICKSTART.md)** - Быстрый старт за 5 минут
- **[AGENT_LOGIC.md](AGENT_LOGIC.md)** - Детальное описание агентной логики всех режимов работы

## ✨ Возможности

### 4 Режима работы

- **Chat** - Простой диалог с LLM без веб-поиска
- **Web Search** (search) - Быстрый веб-поиск с расширением запросов (2 итерации)
- **Deep Search** (deep_search) - Качественный веб-поиск с глубокими итерациями (6 итераций)
- **Deep Research** (deep_research) - Полноценная мультиагентная система с координацией через супервайзера
  - Супервайзер (LangGraph с ReAct) координирует 1-5 исследовательских агентов
  - Каждый исследователь - LangGraph агент с планированием и перепланированием
  - Все агенты пишут заметки в markdown файлы в сессионной папке
  - Супервайзер просматривает прогресс и назначает новые задачи через систему очередей
  - Финальный отчет валидируется и стримится на фронтенд
  - Подробнее: [AGENT_LOGIC.md](AGENT_LOGIC.md)

### Продвинутая система памяти

- **Двойное хранилище**: Человекочитаемые Markdown файлы + PostgreSQL с pgvector
- **Гибридный поиск**: RRF (Reciprocal Rank Fusion) объединяет векторный и полнотекстовый поиск
- **Умное разбиение**: Markdown-aware chunking с сохранением контекста заголовков
- **Автосинхронизация**: Автоматическая синхронизация между файлами и базой данных
  - Расположение файлов памяти настраивается через `MEMORY_DIR` (по умолчанию: `./memory_files`)
- **Контекст чата**: Агенты получают последние N сообщений чата (`CHAT_HISTORY_LIMIT`)
- **Агентные сессии**: Deep Research создает временные папки сессий с файлами памяти агентов
  - `main.md` - главный документ исследования (ключевые инсайты)
  - `draft_report.md` - черновик финального отчета
  - `supervisor.md` - личные заметки супервайзера
  - `agents/{agent_id}.md` - файлы агентов (todos и заметки)
  - `items/` - заметки агентов с источниками
  - Автоматически сохраняются после успешного завершения исследования

### ⚙️ Гибкая конфигурация

- **Провайдеры поиска**: Tavily или SearXNG (self-hosted)
- **Провайдеры эмбеддингов**: OpenAI, Ollama (локально), Cohere, или HuggingFace
- **LLM модели**: OpenAI GPT-4, Anthropic Claude, OpenRouter, и любой OpenAI-совместимый API
- **Полностью настраиваемо**: Все настройки через `.env` файлы

### 🛠 Современный стек технологий

- **Backend**: FastAPI + LangGraph workflows + Pydantic v2
- **Frontend**: Next.js 14 + React + TypeScript + Tailwind CSS
- **База данных**: PostgreSQL 16 + pgvector extension (или SQLite для разработки)
- **Real-time**: SSE streaming для обновлений исследования в реальном времени
- **Развертывание**: Docker Compose для простой настройки

## 🚀 Быстрый старт

См. подробную инструкцию: **[QUICKSTART.md](QUICKSTART.md)**

### Требования

- **Docker & Docker Compose** (рекомендуется) ИЛИ
- Python 3.11+ и Node.js 18+ для ручной настройки
- API ключи: OpenAI (обязательно) и Tavily (рекомендуется)

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

## 📖 Подробная документация

- **[QUICKSTART.md](QUICKSTART.md)** - Пошаговая инструкция по запуску
- **[AGENT_LOGIC.md](AGENT_LOGIC.md)** - Детальное описание логики работы всех режимов:
  - Chat Mode - простой диалог
  - Web Search Mode - быстрый веб-поиск
  - Deep Search Mode - качественный веб-поиск
  - Deep Research Mode - мультиагентная система

## Структура проекта

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

## Архитектурные особенности

### LangGraph Workflows

Система использует LangGraph для оркестрации сложных исследовательских процессов:

```
User Query → Deep Search (optional) → Clarifying Questions (optional) →
Analyze Query → Plan Research → Create Agent Characteristics →
Execute Agents (parallel) → Supervisor Reviews → 
[Continue/Replan/Finish] →
Compress Findings → Generate Final Report → Return
```

Подробнее о логике работы: **[AGENT_LOGIC.md](AGENT_LOGIC.md)**

### Интеграция памяти

- Контекст исследования извлекается из памяти перед началом
- Находки автоматически сохраняются в память после завершения
- Гибридный поиск (семантический + ключевые слова) для оптимального извлечения

### Real-time Streaming

- OpenAI-совместимый endpoint `/v1/chat/completions`
- SSE streaming с блоками исследования/источников/рассуждений
- SessionManager паттерн для управления состоянием UI

## ⚙️ Конфигурация

### Обязательные настройки

Отредактируйте `backend/.env`:

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

## API Документация

После запуска backend, посетите:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Основные Endpoints

- `POST /v1/chat/completions` - OpenAI-совместимый chat endpoint (model: `chat`, `search`/`web_search`, `deep_search`, `deep_research`)
- `POST /api/chat/stream` - Chat с событиями прогресса (SSE)
- `POST /api/research` - Запустить исследовательскую сессию
- `GET /api/memory` - Список файлов памяти
- `POST /api/memory` - Создать файл памяти
- `GET /api/config` - Получить конфигурацию

## Разработка

### Запуск тестов

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

### Качество кода

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

## 🎓 Интегрированные лучшие практики

Этот проект объединяет лучшие возможности из ведущих open-source проектов:

| Проект | Интегрированная возможность |
|--------|----------------------------|
| **multifile-markdown-mcp** | Гибридная система памяти с RRF (Reciprocal Rank Fusion) поиском |
| **open_deep_research** | Паттерн супервайзера с параллельными исследователями |
| **sgr-agent-core** | OpenAI-совместимый API с SSE streaming |
| **OpenDeepSearch** | Абстракция провайдеров поиска и реранжинг |
| **Perplexica** | Лимиты итераций по режимам и streaming UI блоки |

## 📊 Как это работает

Подробное описание логики работы всех режимов: **[AGENT_LOGIC.md](AGENT_LOGIC.md)**

### Исследовательский процесс

```
User Query → Deep Search (optional) → Clarifying Questions (optional) →
Analyze Query → Plan Research → Create Agent Characteristics →
Execute Agents (parallel) → Supervisor Reviews → 
[Continue/Replan/Finish] →
Compress Findings → Generate Final Report → Return
```

### Система памяти

1. **Markdown файлы**: Человекочитаемые исследовательские заметки (расположение настраивается через `MEMORY_DIR`)
2. **Векторная база данных**: PostgreSQL + pgvector для семантического поиска
3. **Гибридный поиск**: Объединяет векторное сходство + полнотекстовый поиск с RRF
4. **Автосинхронизация**: Изменения в файлах автоматически обновляют базу данных

## Лицензия

MIT

## Поддержка

Для вопросов и проблем используйте GitHub issue tracker.

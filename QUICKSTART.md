# Quick Start Guide

## Быстрый запуск за 5 минут

### Шаг 1: Получите API ключи

Вам понадобятся:

1. **OpenAI API Key** (обязательно)
   - Зарегистрируйтесь на https://platform.openai.com
   - Создайте API ключ в разделе API Keys
   - Формат: `sk-...`

2. **Tavily API Key** (обязательно для веб-поиска)
   - Зарегистрируйтесь на https://tavily.com
   - Получите бесплатный API ключ
   - Формат: `tvly-...`

### Шаг 2: Настройте проект

```bash
# Перейдите в директорию проекта
cd /home/asudakov/projects/all_included_search/all_included_deep_research

# Настройте backend
cd backend
cp .env.example .env

# Откройте .env и добавьте ваши ключи:
# POSTGRES_PASSWORD=your_secure_password
# OPENAI_API_KEY=sk-your-key-here
# TAVILY_API_KEY=tvly-your-key-here
# LLM_MODE=mock  # для запуска без LLM
# SEARCH_PROVIDER=mock  # для запуска без внешнего поиска

# Настройте docker (в корне проекта)
cd ..
cp .env.example .env

# Откройте .env и установите пароль и путь памяти:
# POSTGRES_PASSWORD=your_secure_password
# MEMORY_HOST_PATH=/home/asudakov/projects/memory_files
```

### Шаг 3: Запустите проект

```bash
# Находясь в корне проекта
docker compose up -d
```

Подождите 2-3 минуты пока Docker соберет образы и запустит все сервисы.

### Шаг 4: Откройте в браузере

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Шаг 5: Попробуйте исследование

1. Выберите режим (Simple Search / Deep Search / Deep Research)
2. Введите ваш запрос, например:
   - "Latest developments in quantum computing"
   - "Compare React vs Vue.js for enterprise applications"
   - "Explain transformer architecture in deep learning"
3. Наблюдайте за процессом исследования в реальном времени
4. Получите подробный отчет с источниками

## Остановка проекта

```bash
docker compose down
```

## Решение проблем

### Порты заняты

Если порты 3000, 8000 или 5432 уже используются:

```bash
# Проверьте какие процессы используют порты
sudo lsof -i :3000
sudo lsof -i :8000
sudo lsof -i :5432

# Остановите конфликтующие процессы или измените порты в docker-compose.yml
```

### Docker не запускается

```bash
# Проверьте статус Docker
sudo systemctl status docker

# Запустите Docker если он остановлен
sudo systemctl start docker
```

### Ошибки миграций базы данных

```bash
# Пересоздайте базу данных
docker compose down -v  # Удалит все данные!
docker compose up -d
```

### Логи для отладки

```bash
# Все логи
docker compose logs -f

# Только backend
docker compose logs -f backend

# Только frontend
docker compose logs -f frontend
```

## Дополнительные возможности

### Использование Claude вместо GPT

Отредактируйте `backend/.env`:

```bash
ANTHROPIC_API_KEY=sk-ant-your-key
RESEARCH_MODEL=anthropic:claude-3-5-sonnet-20241022
```

### Использование локальных embeddings (Ollama)

1. Запустите Ollama контейнер: `docker compose --profile local-embeddings up -d`
2. Скачайте модель: `ollama pull nomic-embed-text` (внутри контейнера или локально)
3. Отредактируйте `backend/.env`:

```bash
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://ollama:11434
```

### Использование локальных embeddings (HuggingFace)

Отредактируйте `backend/.env`:

```bash
EMBEDDING_PROVIDER=huggingface
HUGGINGFACE_MODEL=sentence-transformers/all-MiniLM-L6-v2
HUGGINGFACE_USE_LOCAL=true
```

### Оффлайн режим без LLM/поиска

```bash
LLM_MODE=mock
SEARCH_PROVIDER=mock
EMBEDDING_PROVIDER=mock
```

### Настройка глубины исследования

Отредактируйте `backend/.env`:

```bash
# Увеличить количество итераций
BALANCED_MAX_ITERATIONS=10  # По умолчанию: 6
QUALITY_MAX_ITERATIONS=30   # По умолчанию: 25

# Увеличить количество параллельных исследователей
BALANCED_MAX_CONCURRENT=5   # По умолчанию: 3
QUALITY_MAX_CONCURRENT=8    # По умолчанию: 5
```

## Что дальше?

- Прочитайте полную документацию в [README.md](README.md)
- Изучите API в Swagger UI: http://localhost:8000/docs
- Настройте память для сохранения результатов исследований
- Интегрируйте с вашими приложениями через OpenAI-compatible API

## Поддержка

Если возникли проблемы:

1. Проверьте логи: `docker compose logs -f`
2. Убедитесь что все API ключи правильно настроены
3. Проверьте что Docker имеет достаточно ресурсов (минимум 4GB RAM)

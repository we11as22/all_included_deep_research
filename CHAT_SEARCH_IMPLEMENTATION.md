# Chat Search Implementation

## Обзор

Реализован гибридный поиск по сообщениям чатов с использованием PostgreSQL, векторных эмбеддингов и полнотекстового поиска.

## Что было сделано

### 1. Исправлена ошибка импорта (ModuleNotFoundError)

**Проблема:** `ModuleNotFoundError: No module named 'src.workflow.legacy'`

**Решение:**
- Создан файл `backend/src/workflow/legacy/__init__.py` для превращения папки в Python-модуль

### 2. Обновлена схема базы данных

**Файл:** `backend/src/database/schema.py`

**Изменения в ChatMessageModel:**
- Добавлено поле `embedding: Column(Vector(1536))` для хранения векторных представлений сообщений
- Добавлен индекс `idx_chat_messages_embedding` для векторного поиска (ivfflat)
- Добавлен комментарий о полнотекстовом индексе

### 3. Создана миграция базы данных

**Файл:** `backend/alembic/versions/002_add_chat_message_embedding.py`

**Что делает миграция:**
- Добавляет колонку `embedding` типа `vector(1536)` в таблицу `chat_messages`
- Создаёт индекс `idx_chat_messages_embedding` для векторного поиска (ivfflat с cosine similarity)
- Создаёт полнотекстовый индекс `idx_chat_messages_content_fts` для поиска по содержимому сообщений

### 4. Создан сервис гибридного поиска

**Файл:** `backend/src/chat/search.py`

**Класс:** `ChatMessageSearchEngine`

**Методы:**
- `search(query, limit, chat_id, role_filter)` - гибридный поиск (vector + fulltext с RRF)
- `vector_search()` - только векторный поиск
- `fulltext_search()` - только полнотекстовый поиск

**Возвращаемые данные:** `ChatMessageSearchResult` с информацией о:
- ID сообщения
- ID чата
- Роль (user/assistant/system)
- Содержимое сообщения
- Дата создания
- Название чата
- Оценка релевантности (score)

### 5. Обновлён API endpoint для поиска

**Файл:** `backend/src/api/routes/chats.py`

**Endpoint:** `GET /api/chats/search?q={query}&limit={limit}`

**Изменения:**
- Заменён простой ILIKE поиск на гибридный поиск
- Теперь возвращает топ-N наиболее релевантных **сообщений** (а не чатов)
- Каждое сообщение включает информацию о чате, к которому оно принадлежит
- Limit по умолчанию = 5 (топ-5 сообщений)

**Пример ответа:**
```json
{
  "messages": [
    {
      "message_id": 123,
      "chat_id": "abc-123",
      "message_message_id": "msg-456",
      "role": "user",
      "content": "How to implement search?",
      "created_at": "2026-01-02T01:00:00",
      "chat_title": "Search Implementation Discussion",
      "chat_updated_at": "2026-01-02T02:00:00",
      "score": 0.95,
      "search_mode": "hybrid"
    }
  ],
  "count": 1
}
```

### 6. Автоматическая генерация эмбеддингов

**Файл:** `backend/src/api/routes/chats.py`

**Endpoint:** `POST /api/chats/{chat_id}/messages`

**Изменения:**
- При создании нового сообщения автоматически генерируется эмбеддинг для content
- Эмбеддинг нормализуется до 1536 измерений (требование схемы БД)
- Если генерация эмбеддинга не удалась, сообщение всё равно сохраняется (без эмбеддинга)

### 7. Инициализация в приложении

**Файл:** `backend/src/api/app.py`

**Изменения:**
- Импортирован `ChatMessageSearchEngine`
- В `lifespan()` функции инициализируется `chat_message_search_engine`
- Engine добавлен в `app.state` для доступа из роутов

### 8. Обновлена конфигурация Docker

**Файл:** `docker-compose.yml`

**Изменения:**
- Изменено `USE_POSTGRES: "true"` (было `"false"`)
- Включена зависимость backend от postgres с health check

## Как использовать

### Backend API

**Поиск сообщений:**
```bash
GET http://localhost:8000/api/chats/search?q=your+search+query&limit=5
```

**Создание сообщения с автоматическим эмбеддингом:**
```bash
POST http://localhost:8000/api/chats/{chat_id}/messages
Content-Type: application/json

{
  "role": "user",
  "content": "Your message text here"
}
```

### Применение миграции

После запуска проекта миграция применится автоматически. Для ручного применения:

```bash
cd backend
alembic upgrade head
```

## Frontend интеграция (TODO)

Для реализации поиска на фронтенде необходимо:

1. **Создать поле поиска** в интерфейсе чатов
2. **Вызывать API** `/api/chats/search?q={query}`
3. **Отображать результаты** - список найденных сообщений с:
   - Превью содержимого сообщения
   - Название чата
   - Дата сообщения
   - Оценка релевантности
4. **Открывать чат при клике** на сообщение:
   - Перейти к чату с `chat_id`
   - Прокрутить до сообщения с `message_id`
   - Подсветить найденное сообщение

## Технические детали

### Гибридный поиск (RRF - Reciprocal Rank Fusion)

Алгоритм комбинирует результаты двух типов поиска:
- **Векторный поиск:** Семантическое сходство через косинусную близость эмбеддингов
- **Полнотекстовый поиск:** Поиск ключевых слов через PostgreSQL FTS

Оценка релевантности вычисляется по формуле:
```
RRF_score = (1 / (k + vector_rank)) + (1 / (k + fulltext_rank))
```
где `k = 60` (параметр RRF_K)

### Размерность эмбеддингов

- По умолчанию: **1536 измерений** (OpenAI text-embedding-ada-002)
- Поддерживается автоматическая нормализация:
  - Дополнение нулями, если эмбеддинг меньше 1536
  - Обрезка, если эмбеддинг больше 1536

### Индексы PostgreSQL

1. **Векторный индекс (ivfflat):**
   ```sql
   CREATE INDEX idx_chat_messages_embedding ON chat_messages
   USING ivfflat (embedding vector_cosine_ops)
   WITH (lists = 100);
   ```

2. **Полнотекстовый индекс (GIN):**
   ```sql
   CREATE INDEX idx_chat_messages_content_fts ON chat_messages
   USING gin(to_tsvector('english', content));
   ```

## Возможные улучшения

1. **Мультиязычность:** Поддержка русского языка в полнотекстовом поиске
   ```sql
   to_tsvector('russian', content)
   ```

2. **Фильтры:**
   - По дате создания
   - По типу сообщения (user/assistant)
   - По конкретному чату

3. **Выделение найденных фраз:** ts_headline для подсветки совпадений

4. **Пагинация:** Для больших результатов поиска

5. **Кэширование:** Кэширование часто запрашиваемых эмбеддингов

## Зависимости

- **PostgreSQL 16** с расширением **pgvector**
- **Python пакеты:**
  - `pgvector` - для работы с векторами
  - `asyncpg` - асинхронный драйвер PostgreSQL
  - `sqlalchemy` - ORM
  - Embedding provider (OpenAI, Ollama, или другой)

## Тестирование

После запуска проекта можно протестировать:

1. Создайте чат и добавьте несколько сообщений
2. Выполните поиск через API
3. Проверьте, что возвращаются релевантные сообщения

```bash
# Пример теста
curl "http://localhost:8000/api/chats/search?q=implementation&limit=5"
```

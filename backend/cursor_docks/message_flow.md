# Поток сохранения и загрузки сообщений

## Обзор

Документ описывает, как сообщения сохраняются в БД и как формируется `chat_history` для определения `original_query`.

## Сохранение сообщений

### 1. User сообщения

**Где сохраняются:**
- **Frontend** (`frontend/src/app/page.tsx`): Сохраняет user сообщение в БД через API `POST /{chat_id}/messages` ПЕРЕД отправкой запроса
- **Backend** (`backend/src/api/routes/chats.py:378`): Эндпоинт `add_message` сохраняет сообщение в таблицу `chat_messages`

**Последовательность:**
1. User вводит сообщение на фронтенде
2. Frontend вызывает `addMessage(chatId, 'user', content, messageId)` 
3. Сообщение сохраняется в БД с `role='user'`
4. Затем отправляется запрос в `/stream` с этим сообщением в `request.messages`

### 2. Assistant сообщения

**Где сохраняются:**

#### Deep Search результат:
- **Файл**: `backend/src/workflow/research/nodes.py:349`
- **Функция**: `_save_message_to_db_async` в `run_deep_search_node`
- **Когда**: После выполнения deep search, сразу после `emit_report_chunk`
- **Message ID**: `deep_search_{session_id}_{timestamp}`
- **Role**: `assistant`
- **Content**: Полный результат deep search с заголовком "## Initial Deep Search Context"

#### Clarification вопросы:
- **Файл**: `backend/src/workflow/research/nodes.py:583`
- **Функция**: `_save_message_to_db_async` в `clarify_with_user_node`
- **Когда**: После отправки clarification вопросов пользователю
- **Message ID**: `clarification_{session_id}_{timestamp}`
- **Role**: `assistant`
- **Content**: Текст с вопросами clarification

#### Final Report:
- **Файл**: `backend/src/streaming/sse.py:371`
- **Функция**: `_save_final_message_to_db` в `ResearchStreamingGenerator`
- **Когда**: После завершения исследования, при вызове `emit_final_report`
- **Message ID**: `final_report_{session_id}_{timestamp}`
- **Role**: `assistant`
- **Content**: Финальный отчет исследования

## Загрузка chat_history

### Где загружается:

**Файл**: `backend/src/api/routes/chat_stream.py:78-97`

```python
if request.chat_id:
    # Загружаем все сообщения из БД для этого chat_id
    result = await session.execute(
        select(ChatMessageModel)
        .where(ChatMessageModel.chat_id == request.chat_id)
        .order_by(ChatMessageModel.created_at.asc())  # ВАЖНО: по дате создания
    )
    db_messages = result.scalars().all()
    
    # Конвертируем в формат chat_history
    for msg in db_messages:
        chat_history.append({
            "role": msg.role,      # 'user' или 'assistant'
            "content": msg.content  # Текст сообщения
        })
```

**Важно:**
- Сообщения загружаются в порядке `created_at.asc()` - от старых к новым
- `chat_history` содержит ВСЕ сообщения из БД для данного `chat_id`
- Включает user сообщения, deep search, clarification, final report

## Определение original_query

### Логика для deep_research:

**Файл**: `backend/src/api/routes/chat_stream.py:109-175`

1. **Определение режима**: Проверяется, что `is_deep_research = True`

2. **Поиск маркера**: Ищется последнее сообщение `assistant` с маркерами:
   - "clarification" в тексте
   - "🔍" эмодзи
   - "clarify" в тексте
   - "deep search" или "initial deep search" в тексте
   - "research report" или "final report" в тексте

3. **Поиск original_query**: 
   - Если маркер найден → ищется первое `user` сообщение ПЕРЕД маркером
   - Это и есть исходный запрос пользователя

4. **Fallback**:
   - Если маркер не найден → ищется последнее `user` сообщение в `chat_history`
   - Если `chat_history` пуст → используется первое сообщение из `request.messages` (но это может быть clarification answer!)

## Проблема

### Когда пользователь отвечает на clarification:

1. **User отправляет**: "расскажи про развитие и обучение моделей qwen"
   - Сохраняется в БД как `role='user'`

2. **Система отправляет**: clarification вопросы
   - Сохраняется в БД как `role='assistant'` с маркером "clarification"

3. **User отвечает**: "всё и сразу про все технические аспекты"
   - Сохраняется в БД как `role='user'`
   - Приходит в `request.messages` как последнее сообщение

4. **Проблема**: 
   - `request.messages` содержит только ответ на clarification
   - `chat_history` из БД содержит ВСЕ сообщения, включая исходный запрос
   - Логика должна использовать `chat_history`, а не `request.messages`

## Решение

### Текущая логика (исправленная):

1. **Всегда использовать `chat_history` из БД** для определения `original_query`
2. **Никогда не использовать `request.messages`** для deep_research, если есть `chat_history`
3. **Искать маркер clarification** в `chat_history` и брать query ПЕРЕД ним

### Код:

```python
if is_deep_research:
    # Ищем маркер в chat_history
    for i in range(len(chat_history) - 1, -1, -1):
        msg = chat_history[i]
        if msg.get("role") == "assistant":
            if "clarification" in content or "🔍" in content:
                # Найден маркер - ищем query ПЕРЕД ним
                for j in range(i - 1, -1, -1):
                    prev_msg = chat_history[j]
                    if prev_msg.get("role") == "user":
                        original_query = prev_msg.get("content", "")
                        break
```

## Важные моменты

1. **Порядок сообщений в БД**: `created_at.asc()` - от старых к новым
2. **Структура chat_history**: `[{"role": "user|assistant", "content": "..."}, ...]`
3. **Маркеры deep research**: clarification, deep search, research report, final report
4. **Никогда не использовать `request.messages`** для deep_research, если есть `chat_history`

## Схема потока данных

```
1. User отправляет: "расскажи про qwen"
   ↓
   Frontend: POST /{chat_id}/messages (role='user', content="расскажи про qwen")
   ↓
   БД: Сохраняется как ChatMessageModel(role='user', content="расскажи про qwen")
   ↓
   Frontend: POST /stream (request.messages=[{role: 'user', content: "расскажи про qwen"}])
   ↓
   Backend: Загружает chat_history из БД → [{"role": "user", "content": "расскажи про qwen"}]
   ↓
   Backend: original_query = "расскажи про qwen" ✅

2. Система отправляет: clarification
   ↓
   Backend: _save_message_to_db_async(role='assistant', content="🔍 Clarification...")
   ↓
   БД: Сохраняется как ChatMessageModel(role='assistant', content="🔍 Clarification...")
   ↓
   Frontend: Показывает clarification пользователю

3. User отвечает: "всё и сразу"
   ↓
   Frontend: POST /{chat_id}/messages (role='user', content="всё и сразу")
   ↓
   БД: Сохраняется как ChatMessageModel(role='user', content="всё и сразу")
   ↓
   Frontend: POST /stream (request.messages=[{role: 'user', content: "всё и сразу"}])
   ↓
   Backend: Загружает chat_history из БД → [
     {"role": "user", "content": "расскажи про qwen"},
     {"role": "assistant", "content": "🔍 Clarification..."},
     {"role": "user", "content": "всё и сразу"}
   ]
   ↓
   Backend: Ищет маркер → находит "🔍 Clarification..." на индексе 1
   ↓
   Backend: Ищет user сообщение ПЕРЕД маркером → находит "расскажи про qwen" на индексе 0
   ↓
   Backend: original_query = "расскажи про qwen" ✅
```

## Проверка

Чтобы проверить, что логика работает правильно:

1. **Проверить логи**: Должно быть видно:
   - `Found last deep research marker message` с индексом
   - `Found original query for current deep research session` с правильным query
   - `Using original query` с правильным query

2. **Проверить chat_history**: 
   - Должен содержать все сообщения из БД
   - Должен быть в правильном порядке (по `created_at.asc()`)

3. **Проверить, что не используется request.messages**:
   - Для deep_research всегда используется `chat_history` из БД
   - `request.messages` используется только как fallback, если `chat_history` пуст

4. **Проверить порядок в БД**:
   - Сообщения должны быть в порядке: user query → assistant clarification → user answer
   - Это обеспечивается `order_by(ChatMessageModel.created_at.asc())`

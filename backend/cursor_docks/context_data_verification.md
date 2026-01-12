# Проверка передачи контекстных данных в deep research

## Данные, которые должны передаваться в supervisor

### 1. Original Query (исходный запрос пользователя)

**Откуда берется:**
- `chat_stream.py`: Определяется как `original_query` из `chat_history` ПЕРЕД маркером clarification
- Передается в `run_research_graph(query=query, ...)`
- Сохраняется в `state["query"]` через `create_initial_state(query=query, ...)`

**Где используется:**
- `supervisor_agent.py:940`: `query = state.get("query", "")`
- `supervisor_agent.py:1120`: `**ORIGINAL USER QUERY:** {query}`
- `supervisor_agent.py:1129`: `**THE ORIGINAL USER QUERY IS: "{query}"** - THIS IS THE PRIMARY TOPIC YOU MUST RESEARCH`
- Используется во всех инструкциях по созданию задач для агентов

**Проверка:**
- Логи: `Supervisor agent starting` должен показывать правильный query
- Логи: `Supervisor final context data` должен показывать query с правильной длиной
- Query должен быть исходным запросом, а не ответом на clarification

### 2. Deep Search Result

**Откуда берется:**
- `nodes.py:run_deep_search_node`: Выполняется deep search и сохраняется в `state["deep_search_result"]`
- Сохраняется как `{"type": "override", "value": result}` или просто строка

**Где используется:**
- `supervisor_agent.py:967`: `deep_search_result_raw = state.get("deep_search_result", "")`
- `supervisor_agent.py:1125`: `**INITIAL DEEP SEARCH CONTEXT (CRITICAL - USE THIS TO GUIDE RESEARCH):**\n{deep_search_result[:2000]}`
- Используется для понимания контекста темы

**Проверка:**
- Логи: `Supervisor: deep_search_result available` должен показывать длину > 0
- Логи: `Supervisor final context data` должен показывать `deep_search_result_length > 0`
- Deep search result должен содержать релевантную информацию по теме

### 3. Clarification Context (ответы пользователя на clarification)

**Откуда берется:**
- `supervisor_agent.py:1067`: Извлекается из `chat_history`
- Ищется паттерн: assistant с clarification → user с ответом
- Формируется как текст с пометкой, что это дополнительный контекст

**Где используется:**
- `supervisor_agent.py:1126`: `{clarification_context if clarification_context else clarification_fallback}`
- `supervisor_agent.py:1131`: Инструкции о том, что clarification - это дополнительный контекст
- Используется для понимания глубины/угла исследования

**Проверка:**
- Логи: `Extracted user clarification answers for supervisor` должен показывать ответ пользователя
- Логи: `Supervisor clarification context` должен показывать `has_clarification=True` если есть ответы
- Clarification context должен содержать ответы пользователя, но НЕ заменять original query

### 4. Chat History

**Откуда берется:**
- `chat_stream.py:78-97`: Загружается из БД для данного `chat_id`
- Сортируется по `created_at.asc()` - от старых к новым
- Передается в `run_research_graph(chat_history=chat_history, ...)`

**Где используется:**
- `supervisor_agent.py:1084`: Для извлечения clarification context
- `supervisor_agent.py:1100`: Для формирования `chat_history_text` (последние 2 сообщения)
- `supervisor_agent.py:1120`: Показывается в промпте как `{chat_history_text}`

**Проверка:**
- Chat history должен содержать все сообщения из БД
- Должен быть в правильном порядке (от старых к новым)
- Должен содержать исходный запрос, clarification, и ответы пользователя

## Поток данных

```
1. User: "расскажи про qwen"
   ↓
   chat_stream.py: original_query = "расскажи про qwen"
   ↓
   run_research_graph(query="расскажи про qwen", ...)
   ↓
   create_initial_state(query="расскажи про qwen", ...)
   ↓
   state["query"] = "расскажи про qwen"
   ↓
   supervisor_agent: query = state.get("query", "") = "расскажи про qwen" ✅

2. Deep Search выполняется
   ↓
   nodes.py: run_deep_search_node
   ↓
   state["deep_search_result"] = {"type": "override", "value": "..."}
   ↓
   supervisor_agent: deep_search_result = state.get("deep_search_result", "") ✅

3. Clarification отправляется
   ↓
   nodes.py: clarify_with_user_node
   ↓
   Сохраняется в БД как assistant message
   ↓
   User отвечает: "всё и сразу"
   ↓
   Сохраняется в БД как user message
   ↓
   supervisor_agent: Извлекает из chat_history
   ↓
   clarification_context = "USER CLARIFICATION: ..." ✅

4. Все данные в system_prompt:
   - ORIGINAL USER QUERY: {query} ✅
   - INITIAL DEEP SEARCH CONTEXT: {deep_search_result} ✅
   - USER CLARIFICATION: {clarification_context} ✅
```

## Проверка в логах

### Должны быть видны следующие логи:

1. **В chat_stream.py:**
   - `Found original query for current deep research session` - с правильным query
   - `Using original query` - с правильным query
   - `Chat stream request` - с правильным query

2. **В graph.py:**
   - `Starting research graph execution` - с правильным query
   - `Setting query for continuation` - если продолжение после clarification

3. **В supervisor_agent.py:**
   - `Supervisor agent starting` - с правильным query
   - `Supervisor: deep_search_result available` - с длиной > 0
   - `Extracted user clarification answers` - если есть ответы
   - `Supervisor final context data` - со всеми данными

## Потенциальные проблемы

1. **Query неправильный:**
   - Если используется ответ на clarification вместо исходного запроса
   - **Решение**: Проверить логи `Found original query` - должен быть исходный запрос

2. **Deep search result пустой:**
   - Если deep search не выполнился или не сохранился
   - **Решение**: Проверить логи `Deep search completed` и `deep_search_result available`

3. **Clarification context не извлекается:**
   - Если паттерн не найден в chat_history
   - **Решение**: Проверить логи `Extracted user clarification answers` и структуру chat_history

4. **Chat history неправильный:**
   - Если сообщения в неправильном порядке или отсутствуют
   - **Решение**: Проверить логи загрузки chat_history и порядок сообщений в БД

## Как проверить

1. Запустить deep research с clarification
2. Проверить логи Docker на наличие всех указанных логов
3. Убедиться, что:
   - Query = исходный запрос (не ответ на clarification)
   - Deep search result не пустой
   - Clarification context извлечен правильно
   - Все данные передаются в system_prompt

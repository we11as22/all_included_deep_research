# Проверка Workflow Deep Research с Сессиями

## Workflow одной сессии:

1. **Запрос** → создание сессии
2. **Deep search** → сохранение результата в сессию
3. **Clarification вопросы** → сохранение статуса в сессию
4. **Ответы пользователя** → сохранение ответов в сессию
5. **Мультиагентный research** → использование данных из сессии

## Проверка по шагам:

### 1. Запрос (создание сессии)

**Файл:** `src/api/socketio_server.py:165`
- ✅ `session_manager.get_or_create_session(chat_id, query, mode)`
- ✅ Создается сессия с `original_query`, `status="active"`, `chat_id`
- ✅ Сессия привязана к чату через `chat_id`

**Файл:** `src/workflow/research/session/manager.py:115`
- ✅ `create_session()` сохраняет `original_query` в БД
- ✅ Статус устанавливается в `"active"`

### 2. Deep Search

**Файл:** `src/workflow/research/nodes/deep_search.py:242`
- ✅ `session_manager.save_deep_search_result(session_id, deep_search_result)`
- ✅ Результат сохраняется в `research_sessions.deep_search_result`

**Файл:** `src/workflow/research/state.py:224`
- ✅ `deep_search_result` загружается из сессии в `create_initial_state()`
- ✅ Добавлен в state для использования в workflow

**Файл:** `src/workflow/research/nodes/deep_search.py:42`
- ✅ Проверка `session_status == "researching"` → пропуск deep search
- ✅ Проверка `clarification_answers` в state → пропуск deep search
- ✅ Если результат уже есть в state → возврат без выполнения

### 3. Clarification (вопросы)

**Файл:** `src/workflow/research/nodes/clarify.py:435`
- ✅ `session_manager.update_status(session_id, "waiting_clarification")`
- ✅ Статус сохраняется в БД

**Файл:** `src/workflow/research/nodes/clarify.py:53`
- ✅ Проверка `session_status == "researching"` → пропуск clarification
- ✅ Проверка `clarification_answers` из state → пропуск clarification

### 4. Ответы пользователя

**Файл:** `src/workflow/research/nodes/clarify.py:142`
- ✅ `session_manager.save_clarification_answers(session_id, last_user_message)`
- ✅ Ответы сохраняются в `research_sessions.clarification_answers`
- ✅ `session_manager.update_status(session_id, "researching")`
- ✅ Статус обновляется на `"researching"`

**Файл:** `src/workflow/research/state.py:234`
- ✅ `clarification_answers` загружается из сессии в `create_initial_state()`
- ✅ Добавлен в state для использования в workflow

**Файл:** `src/workflow/research/graph.py:598`
- ✅ При continuation `clarification_answers` загружается из initial_state (сессии)
- ✅ Передается в update_state для продолжения workflow

### 5. Мультиагентный Research

**Файл:** `src/workflow/research/nodes/analyze.py:48`
- ✅ Использует `clarification_answers` из state (из сессии)
- ✅ Fallback на chat_history только для обратной совместимости

**Файл:** `src/workflow/research/nodes/plan.py:103`
- ✅ Использует `clarification_answers` из state (из сессии)
- ✅ Fallback на chat_history только для обратной совместимости

**Файл:** `src/workflow/research/nodes/spawn_agents.py:101`
- ✅ Использует `clarification_answers` из state (из сессии)
- ✅ Fallback на chat_history только для обратной совместимости

**Файл:** `src/workflow/research/supervisor_agent.py:1944`
- ✅ Использует `clarification_answers` из state (из сессии)
- ✅ Fallback на chat_history только для обратной совместимости

## Проверка Continuation (продолжение после ответов)

**Файл:** `src/workflow/research/graph.py:550`
- ✅ При continuation загружается `session_status` из БД
- ✅ Загружается `original_query` из initial_state (сессии)
- ✅ Загружается `deep_search_result` из initial_state (сессии)
- ✅ Загружается `clarification_answers` из initial_state (сессии)
- ✅ Все данные из сессии, не из chat_history

## Состояния сессии (status)

- `"active"` - новая сессия, только что создана
- `"waiting_clarification"` - отправлены вопросы, ждем ответа
- `"researching"` - пользователь ответил, идет мультиагентный research
- `"completed"` - research завершен

## Данные в сессии

- `original_query` - оригинальный запрос пользователя (сохраняется при создании)
- `deep_search_result` - результат deep search (сохраняется после deep search)
- `clarification_answers` - ответы пользователя (сохраняются после ответа)
- `draft_report` - черновик отчета (обновляется агентами/супервайзером)
- `final_report` - финальный отчет (сохраняется при завершении)
- `status` - текущее состояние сессии

## Источник истины

**ВСЕ данные берутся из сессии БД, НЕ из chat_history:**
- ✅ `original_query` - из сессии
- ✅ `deep_search_result` - из сессии
- ✅ `clarification_answers` - из сессии
- ✅ `session_status` - из сессии
- ✅ `draft_report` - из сессии
- ✅ `final_report` - из сессии

**chat_history используется только:**
- Для обнаружения нового ответа пользователя (который потом сохраняется в сессию)
- Как fallback для обратной совместимости (если данных нет в сессии)

## Проверка переходов состояний

1. **Запрос** → `status="active"` ✅
2. **Deep search выполнен** → `deep_search_result` сохранен ✅
3. **Clarification отправлено** → `status="waiting_clarification"` ✅
4. **Пользователь ответил** → `clarification_answers` сохранен, `status="researching"` ✅
5. **Мультиагентный research** → использует данные из сессии ✅
6. **Research завершен** → `final_report` сохранен, `status="completed"` ✅

## Итог

✅ Все данные правильно сохраняются в сессию
✅ Все данные правильно загружаются из сессии
✅ Workflow использует состояния сессии, не chat_history
✅ Сессия правильно привязана к чату через `chat_id`
✅ Continuation правильно работает с данными из сессии

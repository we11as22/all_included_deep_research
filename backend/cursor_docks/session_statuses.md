# Статусы сессий Deep Research

## Обзор

Статусы сессий (`session_status`) используются для отслеживания состояния deep research workflow. Статус хранится в таблице `research_sessions` в поле `status` и управляется через `SessionManager.update_status()`.

## Все возможные статусы

### 1. `active` (Активная)
**Когда устанавливается:**
- При создании новой сессии (`SessionManager.create_session()`)
- Начальное состояние новой сессии

**Что означает:**
- Сессия только что создана
- Deep search еще не выполнен или выполняется
- Clarification еще не отправлен или отправляется
- Исследование еще не началось

**Использование:**
- Используется для определения, что сессия активна и может быть продолжена
- Входит в список активных статусов: `["active", "waiting_clarification", "researching"]`

---

### 2. `waiting_clarification` (Ожидание уточнений)
**Когда устанавливается:**
- После отправки clarification вопросов пользователю
- В `ClarifyNode.execute()` после отправки объединенного сообщения (deep_search + clarification)

**Что означает:**
- Clarification вопросы отправлены пользователю
- Система ждет ответа пользователя на вопросы
- Workflow приостановлен до получения ответа
- Graph прерывается перед `analyze_query` node

**Использование:**
- Используется для проверки, нужно ли отправлять clarification повторно
- Предотвращает повторную отправку объединенного сообщения
- Входит в список активных статусов
- Используется для определения continuation: `is_continuation = session.status in {"waiting_clarification", "researching", "active"}`

**Код:**
```python
# В clarify.py после отправки clarification
await session_manager.update_status(session_id, "waiting_clarification")
```

---

### 3. `researching` (Исследование в процессе)
**Когда устанавливается:**
- После того, как пользователь ответил на clarification вопросы
- В `ClarifyNode.execute()` когда обнаружен ответ пользователя

**Что означает:**
- Пользователь ответил на clarification вопросы
- Clarification answers сохранены в сессию
- Workflow может продолжиться с `analyze_query` node
- Multi-agent research начинается или продолжается
- Агенты выполняют задачи исследования

**Использование:**
- Используется для определения, что пользователь уже ответил на clarification
- Предотвращает повторное выполнение deep search (даже если статус изменился)
- Входит в список активных статусов
- Используется для определения continuation

**Код:**
```python
# В clarify.py после обнаружения ответа пользователя
await session_manager.update_status(session_id, "researching")
```

**Логика:**
1. Пользователь отправляет ответ на clarification
2. `ClarifyNode` обнаруживает ответ в `chat_history`
3. Статус обновляется на `"researching"`
4. `clarification_needed = False` - workflow продолжается
5. Graph переходит к `analyze_query` node

---

### 4. `completed` (Завершена)
**Когда устанавливается:**
- После успешного завершения исследования
- В `SessionManager.complete_session()` после генерации финального отчета

**Что означает:**
- Исследование полностью завершено
- Финальный отчет сгенерирован
- Сессия больше не активна
- Не может быть продолжена

**Использование:**
- Используется для определения завершенных сессий
- Не входит в список активных статусов
- Используется для фильтрации завершенных сессий

**Код:**
```python
# В session/manager.py
await session_manager.complete_session(session_id, final_report=report)
```

---

### 5. `superseded` (Заменена)
**Когда устанавливается:**
- Когда создается новая сессия для того же `chat_id`
- В `SessionManager.create_session()` для старых активных сессий

**Что означает:**
- Сессия была заменена новой сессией в том же чате
- Поддерживается ограничение: один активный сессия на `chat_id`
- Старая сессия помечается как `superseded`, новая создается со статусом `active`

**Использование:**
- Используется для поддержания целостности данных
- Не входит в список активных статусов
- Используется для автоматической очистки при переключении режимов

**Код:**
```python
# В session/manager.py при создании новой сессии
await session.execute(
    update(ResearchSessionModel)
    .where(
        ResearchSessionModel.chat_id == chat_id,
        ResearchSessionModel.status.in_(["active", "waiting_clarification", "researching"])
    )
    .values(status="superseded", completed_at=datetime.now())
)
```

---

### 6. `cancelled` (Отменена)
**Когда устанавливается:**
- Когда пользователь явно отменяет исследование
- При ручной отмене через API или UI

**Что означает:**
- Исследование было отменено пользователем
- Сессия больше не активна
- Не может быть продолжена

**Использование:**
- Используется для отслеживания отмененных исследований
- Не входит в список активных статусов

---

### 7. `expired` (Истекла)
**Когда устанавливается:**
- Автоматически для старых активных сессий (24+ часа)
- В задаче очистки или cron job

**Что означает:**
- Сессия была активна слишком долго (24+ часа)
- Автоматически помечена как истекшая
- Не входит в список активных статусов

**Использование:**
- Используется для автоматической очистки старых сессий
- Предотвращает накопление "зависших" сессий

**Код:**
```python
# В session/manager.py (если есть задача очистки)
await session.execute(
    update(ResearchSessionModel)
    .where(
        ResearchSessionModel.status.in_(["active", "waiting_clarification", "researching"]),
        ResearchSessionModel.created_at < datetime.now() - timedelta(hours=24)
    )
    .values(status="expired", completed_at=datetime.now())
)
```

---

## Активные статусы

**Активные статусы** - это статусы, при которых сессия может быть продолжена:
- `active`
- `waiting_clarification`
- `researching`

Эти статусы используются в:
- `get_active_session()` - для поиска активной сессии для `chat_id`
- `create_session()` - для пометки старых сессий как `superseded`
- `is_continuation` - для определения, является ли запрос продолжением существующей сессии

---

## Жизненный цикл статусов

### Типичный flow:

1. **Создание сессии** → `active`
   - Пользователь отправляет запрос
   - Создается новая сессия со статусом `active`

2. **Deep search выполнен** → остается `active`
   - Deep search выполняется и сохраняется в БД
   - Статус не меняется

3. **Clarification отправлен** → `waiting_clarification`
   - Clarification вопросы отправлены пользователю
   - Статус обновляется на `waiting_clarification`
   - Graph прерывается

4. **Пользователь ответил** → `researching`
   - Пользователь отправляет ответ на clarification
   - Статус обновляется на `researching`
   - Graph продолжается с `analyze_query`

5. **Исследование завершено** → `completed`
   - Финальный отчет сгенерирован
   - Статус обновляется на `completed`

### Альтернативные пути:

- **Новая сессия в том же чате** → старая сессия → `superseded`
- **Отмена пользователем** → `cancelled`
- **Истечение времени** → `expired`

---

## Использование в коде

### Проверка статуса:

```python
# В clarify.py
session_status = state.get("session_status", "active")
if session_status == "waiting_clarification":
    # Clarification уже отправлен, ждем ответа
    ...
elif session_status == "researching":
    # Пользователь уже ответил, продолжаем исследование
    ...
```

### Обновление статуса:

```python
# В clarify.py
await session_manager.update_status(session_id, "researching")
```

### Определение continuation:

```python
# В graph.py
is_continuation = session.status in {"waiting_clarification", "researching", "active"}
```

---

## Важные моменты

1. **Статус `researching` устанавливается ТОЛЬКО после ответа пользователя на clarification**
   - Это означает, что пользователь ответил на вопросы
   - Workflow может продолжиться с исследованием

2. **Deep search НЕ выполняется повторно, даже если статус изменился на `researching`**
   - Проверка DB выполняется ВСЕГДА, независимо от статуса
   - Deep search выполняется только один раз за сессию

3. **Активные статусы позволяют продолжить сессию**
   - `active`, `waiting_clarification`, `researching` - все активные
   - `completed`, `superseded`, `cancelled`, `expired` - неактивные

4. **Один активный сессия на `chat_id`**
   - При создании новой сессии старые активные помечаются как `superseded`
   - Это обеспечивает целостность данных

# Deep Research - Новая реализация с 4 режимами

## Реализованные изменения

### Фронтенд

1. **Обновлены компоненты выбора режимов**:
   - [ModeSelector.tsx](frontend/src/components/ModeSelector.tsx) - добавлен режим **Simple Chat**
   - [ModeSelectorDropdown.tsx](frontend/src/components/ModeSelectorDropdown.tsx) - обновлены описания всех 4 режимов

2. **4 режима работы**:
   - **Simple Chat** - прямой диалог с LLM без веб-поиска
   - **Web Search** - быстрый веб-поиск с multi-query expansion (balanced режим Perplexica)
   - **Deep Search** - глубокий веб-поиск с итерациями (quality режим Perplexica)
   - **Deep Research** - мультиагентная система с максимальной точностью и памятью

### Бэкенд

1. **Новые настройки** ([settings.py](backend/src/config/settings.py)):
   ```python
   deep_research_num_agents: int = 4  # Количество агентов-исследователей
   deep_research_enable_clarifying_questions: bool = True  # Уточняющие вопросы
   deep_research_run_deep_search_first: bool = True  # Deep Search перед агентами
   ```

2. **LangGraph Deep Research система** ([workflow/research/](backend/src/workflow/research/)):

   - **Новая нода**: `run_deep_search_node` - запускает Deep Search перед созданием агентов
   - **Обновлена**: `plan_research_node` - создаёт уникальные характеристики для каждого агента
   - **Обновлена**: `spawn_agents_node` - использует характеристики агентов
   - **Обновлен**: [graph.py](backend/src/workflow/research/graph.py) - добавлена deep_search нода в граф

3. **Характеристики агентов**:
   - Каждый агент получает уникальную роль (например, "Senior AI Policy Expert")
   - Экспертиза в конкретной области
   - Личность и подход к исследованию
   - Всё это используется в ReAct промптах агентов

4. **Обновлён state** ([state.py](backend/src/workflow/research/state.py)):
   ```python
   deep_search_result: str  # Результат initial deep search
   agent_characteristics: Dict[str, Dict]  # Характеристики каждого агента
   settings: Any  # Настройки приложения
   ```

5. **Обновлён researcher.py**:
   - Агенты используют свои характеристики в system prompt
   - Более персонализированный подход к исследованию

6. **SSE стриминг** ([streaming/sse.py](backend/src/streaming/sse.py)):
   - Добавлены методы: `emit_planning`, `emit_supervisor_react`
   - Обновлены методы для работы с dict параметрами
   - Все события правильно отображаются на фронтенде

7. **Интеграция** ([chat_stream.py](backend/src/api/routes/chat_stream.py)):
   - Режим `deep_research` теперь использует новый LangGraph граф
   - Передаются все необходимые параметры: settings, llm, search_provider, scraper

## Архитектура Deep Research

### Workflow
```
1. Search Memory →
2. Run Deep Search (NEW!) →
3. Plan Research (создание характеристик агентов) →
4. Spawn Agents →
5. Execute Agents (параллельно) →
6. Supervisor React →
7. (continue/replan/compress) →
8. Compress Findings →
9. Generate Report
```

### Агенты
- **Количество**: настраивается через `DEEP_RESEARCH_NUM_AGENTS` (по умолчанию 4)
- **Параллельная работа**: все агенты работают одновременно с semaphore
- **Уникальные характеристики**: каждый агент - специалист в своей области
- **ReAct формат**: планирование и перепланирование с вызовом тулзов
- **Очередь supervisor**: `SupervisorQueue` управляет одновременными вызовами

### Особенности
- ✅ Deep Search перед созданием агентов для контекста
- ✅ Создание характеристик агентов на основе задачи
- ✅ Параллельная работа агентов
- ✅ Очередь вызовов supervisor при завершении задач
- ✅ ReAct loop с web_search и scrape_url тулзами
- ✅ Supervisor проводит ревью и обновляет планы
- ✅ Агентная память через markdown файлы (уже была реализована)
- ✅ PDF генерация результата (уже была реализована)
- ✅ Очистка памяти после завершения (уже была реализована)

## Настройки окружения

Добавлены новые переменные в [.env.example](backend/.env.example):

```bash
# Deep Research Multi-Agent Settings
DEEP_RESEARCH_NUM_AGENTS=4
DEEP_RESEARCH_ENABLE_CLARIFYING_QUESTIONS=true
DEEP_RESEARCH_RUN_DEEP_SEARCH_FIRST=true
```

## Как работает каждый режим

### 1. Simple Chat
- Режим: `chat`
- Тулзы: нет
- LLM: напрямую отвечает без поиска
- Использование: быстрые вопросы, не требующие актуальной информации

### 2. Web Search
- Режим: `search`
- Тулзы: web_search (Tavily/SearXNG), scrape_url
- Итерации: 2 (speed mode)
- Использование: быстрый поиск актуальной информации

### 3. Deep Search
- Режим: `deep_search`
- Тулзы: web_search, scrape_url
- Итерации: 6 (balanced mode)
- Использование: глубокое исследование с несколькими раундами поиска

### 4. Deep Research
- Режим: `deep_research`
- Workflow: LangGraph multi-agent
- Агенты: 4 параллельных специалиста
- Итерации: до 25 (quality mode)
- Особенности:
  - Initial deep search
  - Создание характеристик агентов
  - Параллельная работа
  - Supervisor ревью
  - Максимальная точность
- Использование: комплексные исследования требующие максимальной глубины

## Запуск

```bash
cd /root/asudakov/projects/all_included_deep_research

# Создайте .env файлы
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local

# Заполните API ключи в backend/.env
# OPENAI_API_KEY, TAVILY_API_KEY и т.д.

# Запустите Docker
docker-compose up --build
```

Приложение доступно по адресу: http://localhost:3000

## Статус реализации

✅ Фронтенд с 4 режимами
✅ Deep Research с LangGraph
✅ Deep Search нода
✅ Характеристики агентов
✅ Параллельная работа агентов
✅ Supervisor queue
✅ SSE стриминг
✅ Интеграция с chat_stream
✅ Настройки в .env
✅ Docker поддержка

## Следующие шаги (опционально)

1. ❓ Уточняющие вопросы пользователю (логика есть, нужна интеграция с UI)
2. ❓ PDF генерация с кастомизацией (уже работает базово)
3. ❓ Более детальная визуализация прогресса агентов на фронте
4. ❓ Настройка количества агентов через UI

## Примечания

- Все изменения обратно совместимы
- Старые режимы (speed/balanced/quality) всё ещё работают через legacy workflows
- Новый deep_research использует LangGraph вместо legacy QualityResearchWorkflow
- Агентная память уже была реализована и работает корректно

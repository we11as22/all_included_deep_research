# Механизм передачи информации от исследователей к супервизору

## Обзор

Система использует асинхронную передачу findings от исследователей к супервизору через state и очередь.

## 1. Сбор findings агентами

### Где происходит
- **Файл**: `backend/src/workflow/research/researcher.py`
- **Функция**: `_run_researcher_agent_impl()`
- **Строки**: 805-817

### Как работает
1. Агент выполняет исследование (поиск, скрапинг)
2. В конце задачи создается `finding` объект:
   ```python
   finding = {
       "agent_id": agent_id,
       "topic": current_task.title,
       "summary": summary,  # Только реальные находки, без метаданных
       "key_findings": key_findings,  # Отфильтрованные находки
       "sources": useful_sources,  # Только источники с контентом
       "confidence": "high" if len(useful_sources) >= 5 else "medium",
       "notes": important_notes,  # Только информативные заметки
       "sources_count": len(useful_sources),
       "notes_count": len(important_notes),
       "key_findings_count": len(key_findings)
   }
   ```
3. Finding возвращается из функции агента

## 2. Сбор всех findings в execute_agents_enhanced_node

### Где происходит
- **Файл**: `backend/src/workflow/research/nodes.py`
- **Функция**: `execute_agents_enhanced_node()`
- **Строки**: 729-1070

### Как работает
1. Агенты выполняются параллельно или последовательно
2. Findings от каждого агента собираются в `all_findings`:
   ```python
   all_findings = []
   # ... выполнение агентов ...
   all_findings.append(finding)  # После завершения каждого агента
   ```
3. Findings добавляются в state через reducer:
   ```python
   return {
       "agent_findings": all_findings,
       "findings": all_findings,  # Дублируется для совместимости
       "findings_count": len(all_findings),
       "iteration": new_iteration,
       "supervisor_call_count": supervisor_call_count
   }
   ```

## 3. Передача findings супервизору

### Где происходит
- **Файл**: `backend/src/workflow/research/nodes.py`
- **Функция**: `supervisor_review_enhanced_node()`
- **Строки**: 1075-1122

### Как работает
1. Супервизор вызывается после выполнения агентов
2. Findings извлекаются из state:
   ```python
   findings = state.get("findings", state.get("agent_findings", []))
   ```
3. Findings передаются в `run_supervisor_agent()` через state

## 4. Обработка findings супервизором

### Где происходит
- **Файл**: `backend/src/workflow/research/supervisor_agent.py`
- **Функция**: `run_supervisor_agent()`
- **Строки**: 900-1689

### Как работает
1. Супервизор получает findings из state (строка 926)
2. Findings фильтруются - удаляются метаданные (строки 959-978)
3. Создается summary для промпта (строки 980-1010)
4. Summary передается в промпт супервизора (строка 1004)

## 5. Сохранение заметок агентов

### Где происходит
- **Файл**: `backend/src/workflow/research/researcher.py`
- **Строки**: 844-851

### Как работает
1. В конце задачи создается финальная заметка:
   ```python
   final_note = AgentNote(
       title=f"Task complete: {current_task.title}",
       summary=summary,
       urls=[s.get("url") for s in sources[:5] if s.get("url")],
       tags=["task_complete"]
   )
   ```
2. Заметка сохраняется через `save_agent_note()`:
   ```python
   await agent_memory_service.save_agent_note(
       final_note, 
       agent_id, 
       agent_file_service=agent_file_service  # КРИТИЧНО для добавления в файл агента
   )
   ```
3. `save_agent_note()` сохраняет заметку в:
   - `items/{timestamp}_{title}_{agent_id}.md` - файл заметки
   - `agents/{agent_id}.md` - личный файл агента (раздел Notes)

### Важно
- **КРИТИЧНО**: `agent_file_service` должен быть передан, иначе заметка не добавится в файл агента
- Заметки фильтруются - сохраняются только информативные (не метаданные)
- В файл агента добавляются только последние 20 важных заметок

## Временная последовательность

1. **Агенты выполняют задачи** → создают findings
2. **Findings собираются** в `all_findings` в `execute_agents_enhanced_node`
3. **Findings добавляются в state** через reducer
4. **Супервизор вызывается** в `supervisor_review_enhanced_node`
5. **Супервизор получает findings** из state
6. **Супервизор анализирует findings** и принимает решения
7. **Супервизор создает новые todos** для агентов через `create_agent_todo`
8. **Цикл повторяется** пока супервизор не решит завершить

## 6. Автоматическое обновление draft_report из findings

### Где происходит
- **Файл**: `backend/src/workflow/research/nodes.py`
- **Функция**: `execute_agents_enhanced_node()`
- **Строки**: После каждого цикла агентов (после строки 845)

### Как работает
1. После каждого цикла агентов проверяются новые findings
2. Если есть новые findings, draft_report автоматически обновляется:
   - Если draft_report существует (>500 символов) - новые findings добавляются в раздел "New Findings"
   - Если draft_report отсутствует или короткий - создается новый с всеми findings
3. Обновление происходит **автоматически**, независимо от действий супервизора
4. Это гарантирует, что draft_report всегда содержит актуальные findings

### Важно
- **Автоматическое обновление** происходит после каждого цикла агентов
- **Супервизор также может обновлять** draft_report через инструмент `write_draft_report` для синтеза и анализа
- Два механизма дополняют друг друга:
  - Автоматическое обновление - добавляет сырые findings
  - Обновление супервизором - добавляет синтез и анализ

## Проблемы и решения

### Проблема: Заметки не появляются в файлах агентов
**Причина**: `agent_file_service` не передавался в `save_agent_note()`
**Решение**: Исправлено - теперь `agent_file_service` передается явно

### Проблема: Findings не передаются супервизору
**Причина**: State может не содержать findings
**Решение**: Используется fallback: `state.get("findings", state.get("agent_findings", []))`

### Проблема: Много метаданных в findings
**Решение**: Findings фильтруются перед передачей супервизору (строки 959-978)

### Проблема: Draft_report не обновляется по ходу исследования
**Причина**: Обновление происходило только через инструмент супервизора или в конце
**Решение**: Добавлено автоматическое обновление draft_report после каждого цикла агентов


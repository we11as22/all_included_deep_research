# Логика составления результата Deep Research

## Обзор процесса

Результат deep research формируется в несколько этапов, собирая информацию от агентов-исследователей, супервайзера и различных источников данных.

---

## 1. Сбор Findings от агентов-исследователей

### Где собираются findings:
- **Файл**: `backend/src/workflow/research/researcher.py`
- **Функция**: `_run_researcher_agent_impl` (строки 47-1014)

### Что попадает в finding:
```python
finding = {
    "agent_id": agent_id,           # ID агента (обязательно для supervisor queue)
    "topic": current_task.title,     # Название задачи
    "summary": summary,              # Краткое резюме (только реальная информация, без метаданных)
    "key_findings": key_findings,    # Ключевые находки (отфильтрованные, без мусора)
    "sources": useful_sources,       # Источники с реальным контентом
    "confidence": "high/medium",      # Уровень уверенности
    "notes": important_notes,        # Важные заметки (отфильтрованные)
    "sources_count": len(useful_sources),
    "notes_count": len(important_notes),
    "key_findings_count": len(key_findings)
}
```

### Как findings попадают в state:
- **Файл**: `backend/src/workflow/research/nodes_legacy.py`
- **Функция**: `execute_agents_enhanced_node` (строки 1652-2666)
- Findings собираются в `all_findings` список
- Возвращаются в state как `findings: list[dict]`
- Также добавляются в `agent_findings` через reducer (Annotated[list[dict], operator.add])

---

## 2. Создание и обновление Draft Report

### Draft Report - основной источник для финального отчета

**Файл**: `backend/src/workflow/research/supervisor_agent.py`
**Функции**:
- `write_draft_report_handler` (строки 219-270) - супервайзер пишет в draft_report.md
- `update_synthesized_report_handler` (строки 273-...) - супервайзер обновляет синтезированный отчет

### Что пишется в draft_report.md:
1. **Структура**:
   ```markdown
   # Research Report Draft
   
   **Query:** {query}
   **Started:** {timestamp}
   
   ## Overview
   ...
   
   ## {section_title} - {timestamp}
   {content}
   ```

2. **Содержимое**:
   - Супервайзер синтезирует findings от агентов
   - Пишет структурированные секции с анализом
   - Обновляет по мере получения новых findings
   - Может содержать секции:
     - `SUPERVISOR SYNTHESIZED REPORT` - основной синтез
     - `RAW FINDINGS` - сырые находки агентов

### Где сохраняется:
- **В памяти**: `agent_memory_service.file_manager.write_file("draft_report.md", content)`
- **В БД**: Сохраняется в `research_sessions.draft_report` через `DraftReportService`

---

## 3. Main.md - дополнительный контекст

### Что такое main.md:
- **Файл**: `backend/src/workflow/research/nodes_legacy.py`
- **Создание**: `create_agent_characteristics_enhanced_node` (строки 1083-1150)
- **Содержимое**:
  - Research Plan (план исследования)
  - Ключевые инсайты
  - Структурированная информация о сессии

### Формат:
```markdown
# Research Session - Main Index

**Query:** {query}
**Session ID:** {session_id}

## Research Plan

- **Topic 1**: Description (Priority: high, Estimated sources: 10)
- **Topic 2**: Description (Priority: medium, Estimated sources: 5)
...
```

---

## 4. Генерация финального отчета

### Функция генерации:
- **Файл**: `backend/src/workflow/research/nodes_legacy.py`
- **Функция**: `generate_final_report_enhanced_node` (строки 2860-3295)

### Источники данных (в порядке приоритета):

#### 1. **Draft Report** (ПРИОРИТЕТ #1)
- Читается из `draft_report.md` через `agent_memory_service.file_manager.read_file("draft_report.md")`
- Это основной источник - содержит синтез супервайзера
- Если draft_report < 1000 символов или отсутствует:
  - Создается из ALL findings (без обрезки)
  - Формат:
    ```markdown
    # Research Report Draft
    
    **Query:** {query}
    **Generated:** {timestamp}
    **Total Findings:** {count}
    
    ## Executive Summary
    ...
    
    ## Detailed Findings (Complete, No Truncation)
    
    ### {topic}
    **Agent:** {agent_id}
    **Confidence:** {confidence}
    
    ### Summary
    {full_summary}
    
    ### Key Findings
    - {all_key_findings}
    
    ### Sources ({count})
    - {source titles and URLs}
    ```

#### 2. **Main Document** (Fallback #1)
- Читается через `agent_memory_service.read_main_file()`
- Используется если draft_report отсутствует или слишком короткий
- Содержит research plan и ключевые инсайты

#### 3. **Findings** (Fallback #2)
- Берется из `state.get("findings", state.get("agent_findings", []))`
- Используется если draft_report и main.md отсутствуют
- Форматируется как:
  ```markdown
  ### {topic}
  {summary}
  
  Key findings:
  - {key_finding_1}
  - {key_finding_2}
  ...
  ```

### Обработка размера draft_report:

1. **Если < 50k символов**: Используется ПОЛНОСТЬЮ (без обрезки)
2. **Если 50k-100k символов**: 
   - Первые 45k символов + детальное резюме остального (15k символов)
   - Используется `summarize_text()` для умного сжатия
3. **Если > 100k символов**:
   - Первые 80k символов + детальное резюме остального (20k символов)

### Промпт для LLM:

```python
prompt = f"""Generate a comprehensive, detailed final research report.

Query: {query}

Language Requirement:
- Write the entire report in {user_language}

Primary Source:
- Use the Draft Research Report as the PRIMARY source
- Generate a full, detailed report (aim for 3000-8000+ words total)

Draft Research Report (supervisor's working document - PRIMARY SOURCE):
{draft_report_for_prompt}

Main document (key insights):
{main_document_preview}

Additional findings (for reference if draft is incomplete):
{findings_text[:20000]}
```

### Структура финального отчета (FinalReport Pydantic model):

```python
class FinalReport(BaseModel):
    title: str
    executive_summary: str      # 400-600 слов
    sections: list[ReportSection]  # Минимум 3-7 секций, каждая 500-1200 слов
    conclusion: str              # 400-600 слов
    sources: list[str]           # Все источники
    confidence_level: str
```

### Требования к отчету:
- **Executive Summary**: 400-600 слов (не краткий!)
- **Sections**: Минимум 4-7 секций, каждая 500-1200 слов
- **Conclusion**: 400-600 слов (не краткий!)
- **Total**: 3000-8000+ слов
- **Язык**: Должен совпадать с языком пользователя
- **Источники**: ВСЕ источники из draft_report, не только несколько

### Валидация отчета:

1. **Проверка размера**: Минимум 1500 символов
2. **Проверка качества**: Через `ReportValidation` Pydantic model
3. **Если слишком короткий**: Предупреждение в логах и статусе

### Форматирование в Markdown:

```markdown
# {title}

## Executive Summary

{executive_summary}

## {section_1.title}

{section_1.content}

## {section_2.title}

{section_2.content}

...

## Conclusion

{conclusion}

---

*Confidence Level: {confidence_level}*
*Research Quality Score: {quality_score}/10*
```

---

## 5. Fallback механизмы

### Если генерация отчета провалилась:

1. **Fallback #1**: Использовать `draft_report.md` напрямую (если > 1000 символов)
2. **Fallback #2**: Использовать `main.md` + findings
3. **Fallback #3**: Использовать только findings (минимальный отчет)

### Критический fallback (если все провалилось):
```markdown
# Research Report: {query}

**Generated:** {timestamp}
**Status:** Report generation encountered errors

## Note

Research was completed but report generation failed. 
Please check draft_report.md or findings for research results.
```

---

## 6. Поток данных (Data Flow)

```
1. Агенты-исследователи
   └─> Создают findings (summary, key_findings, sources)
       └─> Добавляются в state.findings и state.agent_findings

2. Супервайзер
   └─> Получает findings от агентов
       └─> Синтезирует в draft_report.md
           └─> Пишет структурированные секции
               └─> Обновляет по мере получения новых findings

3. Генерация финального отчета
   └─> Читает draft_report.md (ПРИОРИТЕТ)
       ├─> Если отсутствует/короткий: создает из ALL findings
       └─> Читает main.md (fallback)
           └─> Использует findings (fallback)
               └─> Генерирует через LLM с structured output
                   └─> Валидирует размер и качество
                       └─> Форматирует в Markdown
                           └─> Стримит пользователю
```

---

## 7. Критические моменты

### Что НЕ должно теряться:
1. **ALL findings** - все находки агентов должны попасть в draft_report или финальный отчет
2. **ALL key_findings** - не обрезаются (раньше было `[:5]`, теперь все)
3. **ALL sources** - все источники должны быть в финальном отчете
4. **Draft report** - должен содержать полный синтез супервайзера

### Что может быть обрезано:
1. **Draft report > 50k**: Используется умное резюме (не жесткая обрезка)
2. **Findings text > 20k**: Обрезается до 20k в промпте (но draft_report - приоритет)

### Гарантии:
- **Всегда есть результат**: Даже если все провалилось, создается минимальный отчет
- **Язык сохраняется**: Отчет всегда на языке пользователя
- **Источники включены**: Все источники из findings попадают в финальный отчет

---

## 8. Логирование и отладка

### Ключевые логи:
1. `"Read draft report"` - размер draft_report
2. `"Draft report is too short"` - создание из findings
3. `"Using FULL draft_report in prompt"` - использование без обрезки
4. `"Report generated and validated"` - размер и качество финального отчета
5. `"CRITICAL: Final report is too short!"` - предупреждение о коротком отчете

### Метрики для мониторинга:
- `draft_length` - размер draft_report
- `findings_count` - количество findings
- `total_length` - размер финального отчета
- `sections` - количество секций
- `quality_score` - оценка качества (1-10)

---

## 9. Примеры

### Пример 1: Нормальный поток
1. Агенты создают 10 findings
2. Супервайзер синтезирует в draft_report.md (15k символов)
3. Генерация читает draft_report.md полностью
4. LLM создает финальный отчет (5000 слов, 5 секций)
5. Отчет стримится пользователю

### Пример 2: Draft report отсутствует
1. Агенты создают 8 findings
2. Супервайзер не создал draft_report.md
3. Генерация создает draft_report из ALL findings (10k символов)
4. LLM создает финальный отчет из созданного draft_report
5. Отчет стримится пользователю

### Пример 3: Очень большой draft_report
1. Супервайзер создал огромный draft_report (120k символов)
2. Генерация использует первые 80k + резюме остальных 20k
3. LLM создает финальный отчет на основе этого
4. Отчет стримится пользователю

---

## 10. Резюме

**Результат deep research формируется из:**
1. ✅ **Draft Report** (приоритет) - синтез супервайзера
2. ✅ **Main.md** (fallback) - план исследования
3. ✅ **Findings** (fallback) - прямые находки агентов

**Ключевые принципы:**
- Draft report - основной источник
- Все findings должны попасть в отчет
- Никакая информация не должна теряться
- Всегда есть результат (даже минимальный)
- Язык пользователя сохраняется
- Все источники включаются

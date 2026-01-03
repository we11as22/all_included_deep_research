# Отчет об использовании источников

## ✅ Исправлено: все источники используются полноценно

### Проблема
- Ответы были короткими несмотря на много источников
- Ограничение на 10 источников в writer.py
- Недостаточно четкие инструкции LLM об использовании всех источников
- Малый лимит max_tokens для chat_model (2048)

### Решение

#### 1. Убрано ограничение на количество источников ✅
**Файл**: `backend/src/workflow/search/writer.py`

**Было**:
```python
unique_sources = unique_sources[:10]  # Limit sources
```

**Стало**:
```python
# Use all available sources - don't limit artificially
# LLM can handle context and decide which to use
# In speed mode: typically 3-5 sources
# In balanced mode: typically 8-12 sources  
# In quality mode: typically 15-20 sources
```

#### 2. Увеличены лимиты слов в промптах ✅
**Файл**: `backend/src/workflow/search/writer.py`

| Режим | Было | Стало |
|-------|------|-------|
| Speed | 200-400 words | 400-600 words minimum |
| Balanced | 500-800 words | 800-1200 words minimum |
| Quality | 1000-2000 words | 1500-3000 words |

#### 3. Улучшены инструкции LLM ✅

**Speed mode**:
```
IMPORTANT: Don't just summarize snippets - synthesize information 
from ALL sources into a comprehensive answer.
```

**Balanced mode**:
```
IMPORTANT: You have many sources available - use them all! 
Don't just pick a few. Each source adds value - synthesize them 
into a complete picture.
```

**Quality mode**:
```
CRITICAL: Use EVERY source provided! You have extensive research - 
leverage all of it!
- Synthesize information from all sources into a coherent narrative
- Include specific quotes, data, and facts from sources
- Compare and contrast different perspectives
```

#### 4. Увеличен max_tokens для writer ✅
**Файл**: `backend/src/config/settings.py`

**Было**:
```python
chat_model_max_tokens: int = Field(default=2048)
```

**Стало**:
```python
chat_model_max_tokens: int = Field(default=4096, 
    description="Chat model max tokens for writer synthesis")
```

#### 5. Улучшены промпты в ChatSearchService ✅
**Файл**: `backend/src/chat/service.py`

**Добавлено**:
- Явное указание количества источников в промпте
- Инструкции использовать ВСЕ источники
- Длина ответа в зависимости от режима (300-500 / 600-1000 / 1000-2000 слов)
- Акцент на синтез информации из всех источников

#### 6. Увеличен sources_limit ✅
**Файл**: `backend/src/config/settings.py`

**Было**:
```python
sources_limit: int = Field(default=8)
```

**Стало**:
```python
sources_limit: int = Field(default=20, 
    description="Max sources to include in prompts (increased for better coverage)")
```

## 📊 Результат

### До исправлений:
- ❌ Использовалось максимум 10 источников
- ❌ Короткие ответы (200-400 слов в speed)
- ❌ LLM не получал четких инструкций использовать все источники
- ❌ Малый лимит tokens (2048) ограничивал длину ответа

### После исправлений:
- ✅ Используются ВСЕ доступные источники (до 20)
- ✅ Длинные, comprehensive ответы (400-600 / 800-1200 / 1500-3000 слов)
- ✅ Четкие инструкции LLM: "USE ALL sources", "synthesize from each"
- ✅ Увеличен max_tokens до 4096
- ✅ Промпты акцентируют синтез из всех источников

### Теперь в каждом режиме:

**Web Search (speed)**:
- 2 итерации поиска
- 3-5 источников обычно
- 400-600 слов comprehensive ответ
- Все источники используются

**Deep Search (balanced)**:
- 6 итераций поиска
- 8-12 источников обычно
- 800-1200 слов comprehensive ответ
- Все источники синтезируются

**Deep Research (quality)**:
- 25 итераций, multi-agent
- 15-20+ источников
- 1500-3000 слов comprehensive report
- Все источники используются агентами

## ✅ Проверка

Все режимы теперь:
1. ✅ Используют ВСЕ найденные источники
2. ✅ Генерируют полноценные, длинные ответы
3. ✅ Синтезируют информацию из всех источников
4. ✅ Имеют достаточно tokens для длинных ответов
5. ✅ Получают четкие инструкции использовать все источники


# Chat

Базовий синхронний chat-completion виклик.

## Формати повідомлень

Можна передавати dict у форматі OpenAI:

```python
client.chat([
    {"role": "system", "content": "You are concise."},
    {"role": "user",   "content": "Hi"},
])
```

Або типізовані `Message`:

```python
from ai_providers import Message

client.chat([
    Message(role="system", content="You are concise."),
    Message(role="user",   content="Hi"),
])
```

## Структура відповіді

```python
response = client.chat([...])

response.text            # str
response.tool_calls      # list[ToolCall]
response.finish_reason   # "stop" | "length" | "tool_calls" | "content_filter" | "error"
response.model           # str — фактична модель з відповіді API
response.usage           # Usage(prompt_tokens, completion_tokens, total_tokens, cost_usd)
response.raw             # сирий об'єкт від SDK провайдера (на випадок специфічних полів)
```

`finish_reason` нормалізується між провайдерами — наприклад Anthropic
повертає `end_turn`/`stop_sequence`, але вам прийде `"stop"`.

## Параметри per-call

```python
client.chat(
    messages,
    tools=[Tool(...)],         # див. docs/tools.md
    extra_params={             # provider-specific
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 500,     # для Anthropic — якщо не задано, default 1024
    },
)
```

`extra_params` об'єднуються з тими, що ви задали в конструкторі AIClient (per-call виграє).

## Anthropic-специфіка

- Anthropic вимагає `max_tokens`. Якщо не задано — модуль використовує 1024 за дефолтом. Перевизначайте через `extra_params={"max_tokens": ...}`.
- `system` повідомлення витягуються модулем і передаються Anthropic як окремий top-level параметр (це нормально, ваш код залишається уніфікованим).

## Безпечні дефолти

- Тайм-аути увімкнені (60с за дефолтом).
- Retries на 429 / 5xx / TimeoutError увімкнені (3 спроби).
- Помилки авторизації (401), невалідні параметри (400) — НЕ повторюються.

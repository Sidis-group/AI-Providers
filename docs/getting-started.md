# Getting Started

## Встановлення

```bash
pip install "ai-providers[openai,anthropic] @ git+https://github.com/Sidis-group/AI-Providers.git"
```

Виберіть тільки потрібні extras для свого проєкту: `openai`, `anthropic`,
`cache-disk`, `vector-chroma`, `vector-qdrant`, `all`. Якщо викличете провайдер,
чий extra не встановлений, отримаєте `ProviderNotInstalledError` з підказкою.

## Перший запит

```python
import os
from ai_providers import AIClient

client = AIClient(
    provider="openai",
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4o-mini",
)
response = client.chat([{"role": "user", "content": "Привіт!"}])
print(response.text)
print("tokens:", response.usage.total_tokens, "cost USD:", response.usage.cost_usd)
```

## Перемикання провайдера

Той самий код для Claude:

```python
client = AIClient(
    provider="anthropic",
    api_key=os.environ["ANTHROPIC_API_KEY"],
    model="claude-3-5-sonnet-20241022",
)
response = client.chat([{"role": "user", "content": "Привіт!"}])
```

## Параметри AIClient

| Параметр | Тип | Опис |
|---|---|---|
| `provider` | str | `"openai"`, `"anthropic"` або зареєстроване ім'я |
| `api_key` | str | Ключ провайдера |
| `model` | str | Назва моделі (`gpt-4o-mini`, `claude-3-5-sonnet-20241022`...) |
| `timeout` | float | Тайм-аут у секундах (default 60) |
| `max_retries` | int | Кількість повторів на rate limit / 5xx / timeout (default 3) |
| `initial_backoff` | float | Початкова пауза exp-backoff (default 1с) |
| `max_backoff` | float | Максимальна пауза (default 30с) |
| `cache` | CacheBackend | Кеш для нестрімінгових викликів |
| `cache_ttl` | int | TTL у секундах для кешованих записів |
| `middleware` | list | Список Middleware-об'єктів |
| `base_url` | str | Custom endpoint (для проксі/локальних моделей) |
| `extra_params` | dict | Дефолтні провайдер-специфічні параметри (наприклад `temperature`) |
| `extra_pricing` | dict | Перекриття таблиці цін |

`extra_params` перекриваються per-call:

```python
client = AIClient(..., extra_params={"temperature": 0.0})
client.chat(messages, extra_params={"temperature": 0.9})  # перекриває для цього виклику
```

## Подальші кроки

- [Chat](chat.md), [Streaming](streaming.md), [Async](#) (див. README)
- [Tools](tools.md), [Multimodal](multimodal.md)
- [Embeddings](embeddings.md), [Vector stores](vector-stores.md)
- [Caching](caching.md), [Middleware](middleware.md), [Error handling](error-handling.md)

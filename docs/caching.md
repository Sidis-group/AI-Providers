# Caching

Кеш зменшує кошти і затримку коли той самий запит йде повторно.

## In-memory (default)

```python
from ai_providers import AIClient, InMemoryCache

cache = InMemoryCache(default_ttl=3600)  # секунди
client = AIClient(provider="openai", api_key="...", model="gpt-4o-mini", cache=cache)
```

## Disk-backed (потрібен extra `[cache-disk]`)

```python
from ai_providers import DiskCache

client = AIClient(..., cache=DiskCache(directory="./.ai_cache", default_ttl=86400))
```

## Власний backend

Будь-який об'єкт з трьома методами:

```python
class CacheBackend(Protocol):
    def get(self, key: str) -> ChatResponse | None: ...
    def set(self, key: str, value: ChatResponse, ttl: int | None = None) -> None: ...
    def delete(self, key: str) -> None: ...
```

Приклад Redis (псевдокод; виберіть serializer на свій смак — наприклад
`dataclasses.asdict` + `json` з custom `default=str`, або власний адаптер):

```python
import json
from dataclasses import asdict
from ai_providers.types import ChatResponse, Usage, ToolCall

def _serialize(resp: ChatResponse) -> str:
    d = asdict(resp)
    d["raw"] = None  # сирий SDK об'єкт зазвичай не серіалізується
    return json.dumps(d, default=str)

def _deserialize(blob: str) -> ChatResponse:
    d = json.loads(blob)
    d["usage"] = Usage(**d["usage"])
    d["tool_calls"] = [ToolCall(**tc) for tc in d.get("tool_calls", [])]
    return ChatResponse(**d)

class RedisCache:
    def __init__(self, redis_client): self.r = redis_client

    def get(self, key):
        blob = self.r.get(key)
        return _deserialize(blob) if blob else None

    def set(self, key, value, ttl=None):
        self.r.set(key, _serialize(value), ex=ttl)

    def delete(self, key):
        self.r.delete(key)

client = AIClient(..., cache=RedisCache(my_redis))
```

## Що використовується як ключ

SHA-256 хеш від:
- `provider`
- `model`
- `messages` (нормалізовані dataclass)
- `tools`
- `extra_params` (включно з default-параметрами клієнта)

Кеш **не залежить** від `api_key`, `base_url`, `timeout`, `middleware`.

## Налаштування TTL

- `cache_ttl` у `AIClient(...)` — default TTL для всіх записів цього клієнта.
- `cache.default_ttl` (у `InMemoryCache` / `DiskCache`) — default backend'у.
- Per-call TTL контролю немає; реалізуйте у власному backend якщо треба.

## Обмеження

- **Streaming** не кешується. Кешується лише `chat()`.
- Помилки не кешуються.
- Кеш загальний для sync і async клієнтів якщо ви передасте один і той самий backend.

## Вимикання per-call

```python
client.chat(messages, use_cache=False)
```

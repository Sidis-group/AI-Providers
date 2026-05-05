# Error Handling

Уніфікована ієрархія винятків — ваш код залишається переносним між
провайдерами без `if isinstance(...)`.

## Ієрархія

```
AIProviderError
├── AuthenticationError
├── RateLimitError              # .retry_after
├── InvalidRequestError
│   └── ContextLengthError
├── ProviderAPIError            # .status_code
├── TimeoutError
├── CacheError
├── VectorStoreError
└── NotSupportedError

ProviderNotInstalledError(ImportError)   # окремо — стосується інсталяції
```

## Базовий приклад

```python
from ai_providers import (
    AIClient, AIProviderError,
    AuthenticationError, RateLimitError, ContextLengthError, InvalidRequestError,
    ProviderAPIError,
)

try:
    response = client.chat(messages)
except AuthenticationError:
    # неправильний api_key
    ...
except RateLimitError as e:
    # 429; e.retry_after — секунди
    ...
except ContextLengthError:
    # перевищено вікно контексту — скоротіть messages
    ...
except InvalidRequestError:
    # 400 — невалідні параметри/повідомлення
    ...
except ProviderAPIError as e:
    # 5xx або інші API-помилки; e.status_code
    ...
except AIProviderError:
    # будь-які інші помилки модуля
    ...
```

## Поведінка retry

Бібліотека автоматично повторює:

| Помилка             | Retry? |
|---------------------|--------|
| `RateLimitError`    | ✅     |
| `TimeoutError`      | ✅     |
| `ProviderAPIError(5xx)` | ✅ |
| `ProviderAPIError(4xx)` | ❌ |
| `AuthenticationError`   | ❌ |
| `InvalidRequestError`   | ❌ |
| `ContextLengthError`    | ❌ |
| Connection errors       | ✅ |

Налаштування у `AIClient(...)`:

```python
AIClient(
    ...,
    max_retries=3,
    initial_backoff=1.0,
    max_backoff=30.0,
)
```

`max_retries=0` повністю вимикає retry-логіку.

## ProviderNotInstalledError

```python
from ai_providers import AIClient, ProviderNotInstalledError

try:
    client = AIClient(provider="anthropic", api_key="...", model="claude-...")
except ProviderNotInstalledError as e:
    # extra не встановлений — ставимо
    print(e)
    # "Provider/backend 'anthropic' is not installed.
    #  Install with: pip install 'ai-providers[anthropic]'"
```

## Виняток із сирої SDK-помилки

Якщо потрібен доступ до оригіналу — він прив'язаний як `__cause__`:

```python
try:
    client.chat(...)
except RateLimitError as e:
    original = e.__cause__   # openai.RateLimitError або anthropic.RateLimitError
```

## Винятки vector store

```python
from ai_providers import VectorStore, ChromaStore, NotSupportedError, VectorStoreError

store = VectorStore(backend=ChromaStore(collection="x"))
try:
    store.query("q", mode="hybrid")
except NotSupportedError:
    # Chroma не вміє hybrid — fallback
    store.query("q", mode="semantic")
except VectorStoreError as e:
    # помилка backend (з'єднання, валідація)
    ...
```

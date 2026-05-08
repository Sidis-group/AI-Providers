# ai-providers

Уніфікований Python інтерфейс для AI провайдерів. Дозволяє писати код один раз і
перемикатись між провайдерами однією зміною конструктора. Підтримує chat,
streaming, tool calls, multimodal, embeddings, vector stores, кешування, retries,
middleware, та облік вартості.

**Підтримувані провайдери у v1:** OpenAI, Anthropic (Claude). Архітектура
дозволяє реєструвати власних провайдерів.

**Підтримувані vector stores у v1:** Chroma, Qdrant.

## Зміст

- [Інсталяція](#інсталяція)
- [Швидкий старт](#швидкий-старт)
- [Streaming](#streaming)
- [Async](#async)
- [Tool calling](#tool-calling)
- [Multimodal (image input)](#multimodal-image-input)
- [Embeddings](#embeddings)
- [Vector stores](#vector-stores)
- [Кешування](#кешування)
- [Middleware (логування / метрики / хуки)](#middleware-логування--метрики--хуки)
- [Обробка помилок](#обробка-помилок)
- [Облік вартості](#облік-вартості)
- [Розширення: власний провайдер](#розширення-власний-провайдер)
- [Розробка](#розробка)
- [Подальші гайди](#подальші-гайди)

## Інсталяція

Пакет ще не публікується в PyPI — встановлюється напряму з git.

Через `pip`:

```bash
pip install "ai-providers[openai,anthropic] @ git+https://github.com/Sidis-group/AI-Providers.git"
```

Через `uv`:

```bash
uv pip install "ai-providers[openai,anthropic] @ git+https://github.com/Sidis-group/AI-Providers.git"
```

Через `poetry`:

```bash
poetry add "git+https://github.com/Sidis-group/AI-Providers.git" --extras "openai anthropic"
```

### Доступні extras

Бібліотека модульна: ставите тільки те, що використовуєте.

| Extra            | Що додає                                            |
|------------------|-----------------------------------------------------|
| `openai`         | OpenAI SDK (chat, streaming, tools, embeddings)     |
| `anthropic`      | Anthropic SDK (Claude)                              |
| `cache-disk`     | Persistent кеш через `diskcache`                    |
| `vector-chroma`  | `ChromaStore` для локальних векторних баз          |
| `vector-qdrant`  | `QdrantStore` (semantic + hybrid search)            |
| `all`            | Все вище                                             |
| `dev`            | pytest, ruff, mypy для розробки самого пакету       |

```bash
pip install "ai-providers[openai,vector-qdrant] @ git+..."
```

Якщо викликати провайдер, чий extra не встановлений — отримаєте
`ProviderNotInstalledError` з підказкою, яку команду виконати.

### Оновлення до нової версії

Версіонування — [SemVer](https://semver.org/lang/uk/). Зміни описані у
[CHANGELOG.md](CHANGELOG.md). Кожен реліз отримує git-тег `vX.Y.Z`.

**Через `uv`** (рекомендовано):

```bash
# Якщо ставили без lockfile (uv pip):
uv pip install --upgrade "ai-providers[openai,anthropic] @ git+https://github.com/Sidis-group/AI-Providers.git"

# Якщо проєкт використовує uv.lock (uv add / uv sync):
uv lock --upgrade-package ai-providers
uv sync
```

**Через `pip`:**

```bash
pip install --upgrade --force-reinstall "ai-providers[openai,anthropic] @ git+https://github.com/Sidis-group/AI-Providers.git"
```

> `--force-reinstall` потрібен, якщо ви ставили без явного pin на git-commit:
> pip кешує git-залежності за version-рядком, і якщо ми забули bump — він
> побачить ту саму версію і пропустить update. У нас bump зроблено, але
> прапорець не зашкодить.

**Pin на конкретну версію** (надійніше для production):

```bash
pip install "ai-providers[openai,anthropic] @ git+https://github.com/Sidis-group/AI-Providers.git@v0.1.1"
```

Або на конкретний commit:

```bash
pip install "ai-providers[openai,anthropic] @ git+https://github.com/Sidis-group/AI-Providers.git@<sha>"
```

**Перевірити встановлену версію:**

```bash
python -c "import ai_providers; print(ai_providers.__version__)"
```

## Швидкий старт

```python
from ai_providers import AIClient

client = AIClient(
    provider="openai",
    api_key="sk-...",
    model="gpt-4o-mini",
)

response = client.chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "Привіт! Що таке Python?"},
])

print(response.text)
print("tokens:", response.usage.total_tokens)
print("cost:", response.usage.cost_usd, "USD")
```

Перемкнутись на Claude — змініть лише два параметри:

```python
client = AIClient(
    provider="anthropic",
    api_key="sk-ant-...",
    model="claude-3-5-sonnet-20241022",
)
# Решта коду той самий.
```

Конфігурацію API-ключа реалізуйте у своєму проєкті як завгодно — модуль приймає
ключ через конструктор, а звідки ви його дістаєте (env, vault, hardcoded) — не
його справа.

```python
import os
client = AIClient(provider="openai", api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini")
```

## Streaming

```python
for chunk in client.stream([{"role": "user", "content": "Розкажи коротко про Київ"}]):
    print(chunk.delta, end="", flush=True)
```

Останній чанк містить `finish_reason` та `usage` (із порахованим `cost_usd`).

## Async

```python
import asyncio
from ai_providers import AsyncAIClient

async def main():
    client = AsyncAIClient(
        provider="anthropic",
        api_key="sk-ant-...",
        model="claude-3-5-sonnet-20241022",
    )
    response = await client.chat([{"role": "user", "content": "hi"}])
    print(response.text)

    async for chunk in client.stream([{"role": "user", "content": "tell me a joke"}]):
        print(chunk.delta, end="")

asyncio.run(main())
```

## Tool calling

Той самий інтерфейс для OpenAI і Anthropic:

```python
from ai_providers import AIClient, Tool

tools = [
    Tool(
        name="get_weather",
        description="Get current weather for a city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
]

client = AIClient(provider="openai", api_key="...", model="gpt-4o")
response = client.chat([{"role": "user", "content": "What is the weather in Kyiv?"}], tools=tools)

if response.finish_reason == "tool_calls":
    for call in response.tool_calls:
        print(call.name, call.arguments)
```

## Multimodal (image input)

```python
from ai_providers import AIClient, ContentPart, Message

client = AIClient(provider="openai", api_key="...", model="gpt-4o")
response = client.chat([
    Message(role="user", content=[
        ContentPart(type="text", text="Describe this image"),
        ContentPart(type="image", image_url="https://example.com/cat.jpg"),
    ])
])
print(response.text)
```

Або base64:

```python
ContentPart(type="image", image_base64="<...>", mime_type="image/png")
```

Цей самий код працює для Claude (`provider="anthropic"`, `model="claude-3-5-sonnet-..."`).

## Embeddings

```python
from ai_providers import EmbeddingsClient

emb = EmbeddingsClient(
    provider="openai",
    api_key="...",
    model="text-embedding-3-small",
)
response = emb.embed(["hello world", "good morning"])
print(response.vectors[0][:5])  # перші 5 значень першого вектора
print(response.usage.cost_usd)
```

Anthropic **не має** власних embeddings — використовуйте OpenAI або інтегруйте
Voyage AI у власному коді.

## Vector stores

```python
from ai_providers import (
    AIClient, EmbeddingsClient, VectorStore, ChromaStore, Document,
)

emb = EmbeddingsClient(provider="openai", api_key="...", model="text-embedding-3-small")

store = VectorStore(
    backend=ChromaStore(path="./chroma_db", collection="docs"),
    embeddings=emb,  # автоматично ембедить тексти при upsert/query
)

store.upsert([
    Document(id="1", text="Python is a great programming language", metadata={"lang": "en"}),
    Document(id="2", text="Rust is fast and safe",                  metadata={"lang": "en"}),
    Document(id="3", text="Куди податися у Києві",                  metadata={"lang": "uk"}),
])

results = store.query("which language is fast?", top_k=3, where={"lang": "en"})
for r in results:
    print(f"{r.score:.3f}  {r.document.text}")
```

### Hybrid search

Qdrant підтримує hybrid (semantic + keyword) через RRF або вагований fusion:

```python
from ai_providers import QdrantStore, VectorStore

store = VectorStore(
    backend=QdrantStore(collection="docs", url="http://localhost:6333", vector_size=1536),
    embeddings=emb,
)
results = store.query("fast programming language", top_k=5, mode="hybrid", alpha=0.5)
```

Chroma не підтримує нативний hybrid — `query(mode="hybrid")` з ChromaStore кине
`NotSupportedError`. Використовуйте `mode="semantic"` або `mode="keyword"`.

Перевірити підтримку перед викликом:

```python
if store.supports("hybrid"):
    store.query("...", mode="hybrid")
```

## Кешування

Простий in-memory кеш:

```python
from ai_providers import AIClient, InMemoryCache

client = AIClient(
    provider="openai", api_key="...", model="gpt-4o-mini",
    cache=InMemoryCache(default_ttl=3600),
)
```

Persistent кеш на диску (потрібен extra `cache-disk`):

```python
from ai_providers import DiskCache  # доступне якщо встановлено [cache-disk]

client = AIClient(..., cache=DiskCache(directory="./.ai_cache"))
```

Ваш власний backend (Redis, Memcached тощо) — будь-який об'єкт, що реалізує
методи `get`, `set`, `delete`:

```python
class RedisCache:
    def __init__(self, redis): self.r = redis
    def get(self, key): ...
    def set(self, key, value, ttl=None): ...
    def delete(self, key): ...

client = AIClient(..., cache=RedisCache(my_redis))
```

Кешуються тільки нестрімінгові виклики `chat`. Streaming не кешується.

## Middleware (логування / метрики / хуки)

```python
from ai_providers import AIClient, LoggingMiddleware, MetricsMiddleware

metrics = MetricsMiddleware()
client = AIClient(
    provider="openai", api_key="...", model="gpt-4o-mini",
    middleware=[LoggingMiddleware(), metrics],
)

client.chat([{"role": "user", "content": "hi"}])
print(metrics.requests, metrics.total_tokens, metrics.total_cost_usd)
```

Власна middleware — будь-який об'єкт з методами
`on_request`, `on_response`, `on_error`:

```python
class TraceMiddleware:
    def on_request(self, ctx):  # ctx: RequestContext
        print("request:", ctx.provider, ctx.model)

    def on_response(self, ctx):  # ctx: ResponseContext
        print("response in", ctx.duration_ms, "ms")

    def on_error(self, ctx):     # ctx: ErrorContext
        print("error:", ctx.error)
```

## Обробка помилок

Усі провайдери кидають уніфіковані винятки — ваш код залишається переносним:

```python
from ai_providers import (
    AIClient, AIProviderError,
    AuthenticationError, RateLimitError, ContextLengthError, InvalidRequestError,
    ProviderAPIError, TimeoutError as AITimeoutError,
)

try:
    response = client.chat([{"role": "user", "content": "..."}])
except AuthenticationError:
    ...
except RateLimitError as e:
    print("retry after:", e.retry_after)
except ContextLengthError:
    ...
except AIProviderError:
    ...
```

## Облік вартості

Бібліотека містить таблицю цін для популярних моделей у `ai_providers.PRICING`.
`response.usage.cost_usd` обчислюється автоматично. Для невідомих моделей
поверне `None` — або передайте свою таблицю:

```python
client = AIClient(
    provider="openai",
    api_key="...",
    model="my-fine-tuned-model",
    extra_pricing={"openai": {"my-fine-tuned-model": {"input": 5.0, "output": 15.0}}},
)
```

## Розширення: власний провайдер

```python
from ai_providers import register_provider, AIClient
from ai_providers.base import BaseProvider
from ai_providers.types import ChatResponse, Usage

class MyProvider(BaseProvider):
    name = "my"
    def chat(self, messages, *, tools=None, extra_params=None):
        # ...
        return ChatResponse(text="hi", model=self.model, usage=Usage(1, 1, 2))
    def stream(self, messages, *, tools=None, extra_params=None):
        yield from []

register_provider("my", sync_cls=MyProvider)
client = AIClient(provider="my", api_key="...", model="custom")
```

Деталі — у `docs/adding-provider.md`.

## Розробка

Розробка пакету ведеться через [`uv`](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/Sidis-group/AI-Providers.git
cd ai-providers
uv venv --python 3.11
uv pip install -e ".[all,dev]"
uv run pytest
uv run ruff check
uv run mypy src
```

Тести використовують моки SDK і не вимагають справжніх API ключів.

## Подальші гайди

| Тема              | Файл                                  |
|-------------------|---------------------------------------|
| Швидкий старт     | [docs/getting-started.md](docs/getting-started.md) |
| Chat              | [docs/chat.md](docs/chat.md)         |
| Streaming         | [docs/streaming.md](docs/streaming.md) |
| Tool calling      | [docs/tools.md](docs/tools.md)       |
| Multimodal        | [docs/multimodal.md](docs/multimodal.md) |
| Embeddings        | [docs/embeddings.md](docs/embeddings.md) |
| Vector stores     | [docs/vector-stores.md](docs/vector-stores.md) |
| Caching           | [docs/caching.md](docs/caching.md)   |
| Middleware        | [docs/middleware.md](docs/middleware.md) |
| Error handling    | [docs/error-handling.md](docs/error-handling.md) |
| Adding provider   | [docs/adding-provider.md](docs/adding-provider.md) |
| Adding vector backend | [docs/adding-vector-backend.md](docs/adding-vector-backend.md) |

## Ліцензія

MIT

# Adding a Vector Store Backend

Реалізуйте `VectorStoreBackend` Protocol — це 5 методів.

## Інтерфейс

```python
from typing import Protocol
from ai_providers.types import Document, QueryResult, SearchMode

class VectorStoreBackend(Protocol):
    def upsert(self, docs: list[Document]) -> None: ...
    def query(
        self, *, vector, text, top_k, where, mode, alpha,
    ) -> list[QueryResult]: ...
    def delete(self, ids: list[str]) -> None: ...
    def count(self) -> int: ...
    def supports(self, mode: SearchMode) -> bool: ...
```

## Приклад: pgvector

```python
from ai_providers.types import Document, QueryResult
from ai_providers.exceptions import VectorStoreError, NotSupportedError

class PgVectorStore:
    def __init__(self, conn, table: str):
        self.conn = conn
        self.table = table

    def supports(self, mode):
        return mode in ("semantic",)  # розширте якщо реалізуєте hybrid через ts_vector

    def upsert(self, docs):
        if any(d.embedding is None for d in docs):
            raise VectorStoreError("PgVectorStore requires Document.embedding")
        with self.conn.cursor() as cur:
            for d in docs:
                cur.execute(
                    f"INSERT INTO {self.table} (id, text, metadata, embedding) "
                    "VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (id) DO UPDATE SET text=EXCLUDED.text, ...",
                    (d.id, d.text, d.metadata, d.embedding),
                )

    def query(self, *, vector, text, top_k, where, mode, alpha):
        if mode != "semantic":
            raise NotSupportedError("PgVectorStore supports only semantic mode")
        # ... <-> vector cosine query ...
        return [QueryResult(document=Document(...), score=...) for row in rows]

    def delete(self, ids): ...
    def count(self): ...
```

## Інтеграція

```python
from ai_providers import VectorStore, EmbeddingsClient

emb = EmbeddingsClient(provider="openai", api_key="...", model="text-embedding-3-small")
store = VectorStore(backend=PgVectorStore(conn, "docs"), embeddings=emb)
```

## Async backend

Реалізуйте `AsyncVectorStoreBackend` — той самий інтерфейс, але методи `async`:

```python
class MyAsyncBackend:
    async def upsert(self, docs): ...
    async def query(self, *, vector, text, top_k, where, mode, alpha): ...
    async def delete(self, ids): ...
    async def count(self): ...
    def supports(self, mode): ...
```

```python
from ai_providers import AsyncVectorStore, AsyncEmbeddingsClient

emb = AsyncEmbeddingsClient(...)
store = AsyncVectorStore(backend=MyAsyncBackend(...), embeddings=emb)
await store.upsert([...])
```

## Поради

- **Score normalization.** `QueryResult.score` повинен бути monotonic — більше
  означає кращий збіг. Cosine similarity (`1 - cosine_distance`) — стандартний
  вибір.

- **Metadata filtering.** Конвертуйте `where: dict` у синтаксис вашого backend.
  Складніші фільтри (`$gt`, `$or`) приймайте як nested dict.

- **Hybrid search.** Якщо backend має нативно — використайте; якщо ні — або
  реалізуйте client-side fusion (RRF / weighted), або кидайте
  `NotSupportedError`. Чесно сказати "не підтримую" — кращий UX, ніж тихо
  повертати лише semantic результати.

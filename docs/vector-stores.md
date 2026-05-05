# Vector Stores

`VectorStore` — фасад над pluggable backend з автоматичним ембедингом.

## Підтримка search-режимів у v1

| Backend       | semantic | keyword | hybrid | metadata filter |
|---------------|----------|---------|--------|------------------|
| `ChromaStore` | ✅       | ✅ (`where_document.$contains`) | ❌ → `NotSupportedError` | ✅ |
| `QdrantStore` | ✅       | ✅ (sparse vectors) | ✅ (RRF / weighted) | ✅ |

Чи підтримує конкретний backend режим — `store.supports("hybrid")`.

## Chroma (локальна, dev-friendly)

```python
from ai_providers import (
    EmbeddingsClient, VectorStore, ChromaStore, Document,
)

emb = EmbeddingsClient(provider="openai", api_key="...", model="text-embedding-3-small")

store = VectorStore(
    backend=ChromaStore(path="./chroma_db", collection="docs"),
    embeddings=emb,
)

store.upsert([
    Document(id="1", text="Python language", metadata={"lang": "en"}),
    Document(id="2", text="Rust language",   metadata={"lang": "en"}),
])

results = store.query("fast typed language", top_k=2, where={"lang": "en"})
for r in results:
    print(r.score, r.document.text)
```

`ChromaStore` ініціалізатор:

| Параметр | Опис |
|---|---|
| `collection` | Назва колекції (обов'язково) |
| `path` | Локальна директорія для PersistentClient |
| `host`, `port` | Підключення до Chroma server (HttpClient) |
| `client` | Готовий chromadb клієнт (для тестів) |
| `distance` | `"cosine"` (default), `"l2"`, `"ip"` |

Якщо не передати ні `path`, ні `host`, ні `client` — створюється EphemeralClient
у пам'яті.

## Qdrant (production, semantic + hybrid)

```python
from ai_providers import EmbeddingsClient, VectorStore, QdrantStore, Document

emb = EmbeddingsClient(provider="openai", api_key="...", model="text-embedding-3-small")

store = VectorStore(
    backend=QdrantStore(
        collection="docs",
        url="http://localhost:6333",
        api_key=None,           # для self-hosted
        vector_size=1536,       # для text-embedding-3-small
        distance="Cosine",
        hybrid_fusion="rrf",    # або "weighted"
    ),
    embeddings=emb,
)

store.upsert([
    Document(id="1", text="Python language"),
    Document(id="2", text="Rust language"),
])

results = store.query("fast language", top_k=5, mode="hybrid", alpha=0.6)
```

`QdrantStore` ініціалізатор:

| Параметр | Опис |
|---|---|
| `collection` | Назва колекції |
| `url` | URL Qdrant сервера |
| `api_key` | API key для Qdrant Cloud |
| `host`, `port` | Альтернатива до `url` |
| `path` | Локальний disk-based Qdrant |
| `client` | Готовий QdrantClient |
| `vector_size` | Розмір dense вектора (обов'язково для `upsert`) |
| `distance` | `"Cosine"` (default), `"Dot"`, `"Euclid"` |
| `hybrid_fusion` | `"rrf"` (default) або `"weighted"` |

`":memory:"` Qdrant client використовується якщо нічого з вище не передано — зручно для тестів.

> ⚠️ `QdrantStore.upsert` вимагає `Document.embedding` для кожного документа.
> Передайте `embeddings=...` у `VectorStore` — авто-обчислиться.

## Hybrid search: RRF vs weighted

- **RRF (Reciprocal Rank Fusion)** — стандартний у Qdrant, не вимагає налаштування,
  стійкий до різних масштабів score'ів.
- **weighted** — комбінує нормалізовані score'и з вагою `alpha` (semantic) і
  `1-alpha` (keyword). Передаєте у `query(..., alpha=0.7)`.

Для більшості випадків залиште дефолт RRF.

## Metadata filter

```python
store.query("query text", where={"lang": "en", "section": "intro"})
```

Формат `where`:
- Chroma — нативний Chroma filter синтаксис
- Qdrant — модуль конвертує в `must=[FieldCondition(MatchValue)]`

Для складніших фільтрів використовуйте `extra_params` або працюйте з backend напряму.

## Видалення документів

```python
store.delete(["1", "2"])
print(store.count())
```

## Async версія

```python
from ai_providers import AsyncEmbeddingsClient, AsyncVectorStore
from ai_providers.vector_stores.qdrant import QdrantStore  # backend синхронний — обгорніть або використайте sync

# AsyncVectorStore чекає на AsyncVectorStoreBackend.
# Для повністю-async варіанту використайте qdrant_client.AsyncQdrantClient напряму
# і реалізуйте AsyncVectorStoreBackend Protocol — це 4-5 методів.
```

> Готові async backend'и для Chroma/Qdrant заплановані; у v1 `VectorStore` —
> рекомендований шлях.

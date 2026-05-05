# Embeddings

```python
from ai_providers import EmbeddingsClient

emb = EmbeddingsClient(
    provider="openai",
    api_key="...",
    model="text-embedding-3-small",
)
response = emb.embed(["hello", "world"])
print(len(response.vectors))           # 2
print(len(response.vectors[0]))        # 1536 для text-embedding-3-small
print(response.usage.cost_usd)         # USD за цей виклик
```

## Async

```python
from ai_providers import AsyncEmbeddingsClient

aemb = AsyncEmbeddingsClient(provider="openai", api_key="...", model="text-embedding-3-small")
response = await aemb.embed(["hi"])
```

## Anthropic

Anthropic не надає власних embeddings. Якщо викликати —
`NotImplementedError` із підказкою на Voyage AI.

## Інтеграція з vector store

Передайте `EmbeddingsClient` як параметр `embeddings` у `VectorStore` —
ембединги робитимуться автоматично при `upsert` і `query`. Див. [vector-stores.md](vector-stores.md).

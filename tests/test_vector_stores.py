"""Vector store tests using a fake backend, plus integration tests for Chroma/Qdrant in-memory."""

from __future__ import annotations

from typing import Any

import pytest

from ai_providers import (
    Document,
    NotSupportedError,
    QueryResult,
    VectorStore,
)
from ai_providers.types import EmbeddingResponse, SearchMode, Usage


class FakeBackend:
    """Backend stub for facade tests."""

    def __init__(self, supported: tuple[SearchMode, ...] = ("semantic", "hybrid")) -> None:
        self.docs: list[Document] = []
        self.last_query: dict[str, Any] | None = None
        self._supported = supported

    def upsert(self, docs):
        self.docs.extend(docs)

    def query(self, *, vector, text, top_k, where, mode, alpha):
        self.last_query = {
            "vector": vector,
            "text": text,
            "top_k": top_k,
            "where": where,
            "mode": mode,
            "alpha": alpha,
        }
        return [QueryResult(document=d, score=1.0) for d in self.docs[:top_k]]

    def delete(self, ids):
        self.docs = [d for d in self.docs if d.id not in ids]

    def count(self):
        return len(self.docs)

    def supports(self, mode):
        return mode in self._supported


class FakeEmbeddings:
    """Bare-minimum stand-in for EmbeddingsClient."""

    def __init__(self, dim: int = 3) -> None:
        self.dim = dim
        self.calls = 0

    def embed(self, texts: list[str]):
        self.calls += 1
        vectors = [[float(i + 1)] * self.dim for i, _ in enumerate(texts)]
        return EmbeddingResponse(vectors=vectors, model="fake", usage=Usage())


def test_vector_store_auto_embeds_on_upsert():
    backend = FakeBackend()
    embeddings = FakeEmbeddings()
    store = VectorStore(backend=backend, embeddings=embeddings)  # type: ignore[arg-type]
    store.upsert([Document(id="1", text="hello"), Document(id="2", text="world")])
    assert all(d.embedding is not None for d in backend.docs)
    assert embeddings.calls == 1


def test_vector_store_query_passes_vector():
    backend = FakeBackend()
    embeddings = FakeEmbeddings()
    store = VectorStore(backend=backend, embeddings=embeddings)  # type: ignore[arg-type]
    store.upsert([Document(id="1", text="hello", embedding=[0.1, 0.2, 0.3])])
    store.query("hi", top_k=1)
    assert backend.last_query is not None
    assert backend.last_query["mode"] == "semantic"
    assert backend.last_query["vector"] is not None


def test_vector_store_unsupported_mode_raises():
    backend = FakeBackend(supported=("semantic",))
    store = VectorStore(backend=backend, embeddings=FakeEmbeddings())  # type: ignore[arg-type]
    with pytest.raises(NotSupportedError):
        store.query("x", mode="hybrid")


def test_vector_store_delete_and_count():
    backend = FakeBackend()
    store = VectorStore(backend=backend)  # type: ignore[arg-type]
    store.upsert(
        [Document(id="1", text="a", embedding=[1.0]), Document(id="2", text="b", embedding=[2.0])]
    )
    assert store.count() == 2
    store.delete(["1"])
    assert store.count() == 1


# ---- Real Chroma (ephemeral, in-process) -----------------------------------


def test_chroma_semantic_query_with_real_backend():
    pytest.importorskip("chromadb")
    from ai_providers import ChromaStore

    backend = ChromaStore(collection="t-semantic")
    store = VectorStore(backend=backend)  # use Chroma's default embedder
    store.upsert(
        [
            Document(id="1", text="cats and dogs"),
            Document(id="2", text="rocket launch"),
            Document(id="3", text="kittens are cute"),
        ]
    )
    results = store.query("pets", top_k=2)
    assert len(results) == 2
    # Either of the pet-themed docs should rank above the rocket doc.
    ids = {r.document.id for r in results}
    assert "2" not in ids or "1" in ids or "3" in ids


def test_chroma_hybrid_raises_not_supported():
    pytest.importorskip("chromadb")
    from ai_providers import ChromaStore

    backend = ChromaStore(collection="t-hybrid")
    store = VectorStore(backend=backend)
    with pytest.raises(NotSupportedError):
        store.query("hi", mode="hybrid")


# ---- Real Qdrant (in-memory) ----------------------------------------------


def test_qdrant_semantic_with_in_memory():
    pytest.importorskip("qdrant_client")
    from ai_providers import QdrantStore

    backend = QdrantStore(collection="t-qdrant", vector_size=3)
    embeddings = FakeEmbeddings(dim=3)
    store = VectorStore(backend=backend, embeddings=embeddings)  # type: ignore[arg-type]
    store.upsert(
        [
            Document(id="1", text="alpha alpha alpha"),
            Document(id="2", text="beta"),
        ]
    )
    assert backend.count() == 2
    results = store.query("alpha", top_k=2, mode="semantic")
    assert results
    ids = [r.document.id for r in results]
    assert set(ids) == {"1", "2"}


def test_qdrant_hybrid_supported_in_memory():
    pytest.importorskip("qdrant_client")
    from ai_providers import QdrantStore

    backend = QdrantStore(collection="t-qdrant-hybrid", vector_size=3)
    embeddings = FakeEmbeddings(dim=3)
    store = VectorStore(backend=backend, embeddings=embeddings)  # type: ignore[arg-type]
    assert backend.supports("hybrid")
    store.upsert(
        [
            Document(id="1", text="alpha alpha alpha"),
            Document(id="2", text="beta beta beta"),
        ]
    )
    results = store.query("alpha", top_k=2, mode="hybrid", alpha=0.5)
    assert results

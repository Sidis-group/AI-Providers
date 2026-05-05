"""Pluggable vector store backends."""

from .base import (
    AsyncVectorStore,
    AsyncVectorStoreBackend,
    VectorStore,
    VectorStoreBackend,
)

__all__ = [
    "AsyncVectorStore",
    "AsyncVectorStoreBackend",
    "VectorStore",
    "VectorStoreBackend",
]


def __getattr__(name: str):
    """Lazy access to backend classes — keeps optional deps optional."""

    if name == "ChromaStore":
        from .chroma import ChromaStore

        return ChromaStore
    if name == "QdrantStore":
        from .qdrant import QdrantStore

        return QdrantStore
    raise AttributeError(name)

"""VectorStore facade and backend Protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..exceptions import NotSupportedError
from ..types import Document, QueryResult, SearchMode

if TYPE_CHECKING:
    from ..embeddings.client import AsyncEmbeddingsClient, EmbeddingsClient


@runtime_checkable
class VectorStoreBackend(Protocol):
    """Synchronous vector store backend interface."""

    def upsert(self, docs: list[Document]) -> None: ...
    def query(
        self,
        *,
        vector: list[float] | None,
        text: str | None,
        top_k: int,
        where: dict[str, Any] | None,
        mode: SearchMode,
        alpha: float,
    ) -> list[QueryResult]: ...
    def delete(self, ids: list[str]) -> None: ...
    def count(self) -> int: ...
    def supports(self, mode: SearchMode) -> bool: ...


@runtime_checkable
class AsyncVectorStoreBackend(Protocol):
    async def upsert(self, docs: list[Document]) -> None: ...
    async def query(
        self,
        *,
        vector: list[float] | None,
        text: str | None,
        top_k: int,
        where: dict[str, Any] | None,
        mode: SearchMode,
        alpha: float,
    ) -> list[QueryResult]: ...
    async def delete(self, ids: list[str]) -> None: ...
    async def count(self) -> int: ...
    def supports(self, mode: SearchMode) -> bool: ...


class VectorStore:
    """High-level facade combining a backend with optional auto-embedding."""

    def __init__(
        self,
        backend: VectorStoreBackend,
        embeddings: EmbeddingsClient | None = None,
    ) -> None:
        self.backend = backend
        self.embeddings = embeddings

    def upsert(self, docs: list[Document]) -> None:
        # Auto-embed any document missing an embedding when an embedder is provided.
        if self.embeddings is not None:
            missing = [d for d in docs if d.embedding is None]
            if missing:
                response = self.embeddings.embed([d.text for d in missing])
                for doc, vec in zip(missing, response.vectors, strict=True):
                    doc.embedding = vec
        self.backend.upsert(docs)

    def query(
        self,
        text: str,
        *,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
        mode: SearchMode = "semantic",
        alpha: float = 0.5,
    ) -> list[QueryResult]:
        if not self.backend.supports(mode):
            raise NotSupportedError(
                f"Backend '{type(self.backend).__name__}' does not support mode={mode!r}"
            )
        vector: list[float] | None = None
        if mode in ("semantic", "hybrid") and self.embeddings is not None:
            vector = self.embeddings.embed([text]).vectors[0]
        return self.backend.query(
            vector=vector,
            text=text,
            top_k=top_k,
            where=where,
            mode=mode,
            alpha=alpha,
        )

    def delete(self, ids: list[str]) -> None:
        self.backend.delete(ids)

    def count(self) -> int:
        return self.backend.count()

    def supports(self, mode: SearchMode) -> bool:
        return self.backend.supports(mode)


class AsyncVectorStore:
    def __init__(
        self,
        backend: AsyncVectorStoreBackend,
        embeddings: AsyncEmbeddingsClient | None = None,
    ) -> None:
        self.backend = backend
        self.embeddings = embeddings

    async def upsert(self, docs: list[Document]) -> None:
        if self.embeddings is not None:
            missing = [d for d in docs if d.embedding is None]
            if missing:
                response = await self.embeddings.embed([d.text for d in missing])
                for doc, vec in zip(missing, response.vectors, strict=True):
                    doc.embedding = vec
        await self.backend.upsert(docs)

    async def query(
        self,
        text: str,
        *,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
        mode: SearchMode = "semantic",
        alpha: float = 0.5,
    ) -> list[QueryResult]:
        if not self.backend.supports(mode):
            raise NotSupportedError(
                f"Backend '{type(self.backend).__name__}' does not support mode={mode!r}"
            )
        vector: list[float] | None = None
        if mode in ("semantic", "hybrid") and self.embeddings is not None:
            vector = (await self.embeddings.embed([text])).vectors[0]
        return await self.backend.query(
            vector=vector,
            text=text,
            top_k=top_k,
            where=where,
            mode=mode,
            alpha=alpha,
        )

    async def delete(self, ids: list[str]) -> None:
        await self.backend.delete(ids)

    async def count(self) -> int:
        return await self.backend.count()

    def supports(self, mode: SearchMode) -> bool:
        return self.backend.supports(mode)

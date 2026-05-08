"""Chroma vector store backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..exceptions import NotSupportedError, ProviderNotInstalledError, VectorStoreError
from ..types import Document, QueryResult, SearchMode

if TYPE_CHECKING:
    pass

try:
    import chromadb  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - env without [vector-chroma]
    chromadb = None  # type: ignore[assignment]


# Sentinel key used to satisfy Chroma's "non-empty metadata" requirement when
# a doc has no metadata. Namespaced so it can't collide with user keys.
_SENTINEL_KEY = "__ai_providers_metadata_sentinel__"


class ChromaStore:
    """Chroma backend. Supports semantic + keyword (via where_document); no native hybrid."""

    def __init__(
        self,
        *,
        collection: str,
        path: str | None = None,
        host: str | None = None,
        port: int | None = None,
        client: Any = None,
        distance: str = "cosine",
    ) -> None:
        if chromadb is None:
            raise ProviderNotInstalledError("vector-chroma", "vector-chroma")
        if client is not None:
            self._client = client
        elif host is not None:
            self._client = chromadb.HttpClient(host=host, port=port or 8000)
        elif path is not None:
            self._client = chromadb.PersistentClient(path=path)
        else:
            self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": distance}
        )

    def supports(self, mode: SearchMode) -> bool:
        return mode in ("semantic", "keyword")

    def upsert(self, docs: list[Document]) -> None:
        if not docs:
            return
        ids = [d.id for d in docs]
        documents = [d.text for d in docs]
        # Chroma requires per-doc metadata to be a non-empty dict OR omits
        # the metadatas argument entirely. We send metadatas only if at least
        # one doc has tags; for docs without tags we pad with a sentinel
        # `{"_id": id}` so length matches and other docs keep their metadata.
        has_any_metadata = any(d.metadata for d in docs)
        metadatas: list[dict[str, Any]] | None
        if has_any_metadata:
            metadatas = [
                dict(d.metadata) if d.metadata else {_SENTINEL_KEY: True} for d in docs
            ]
        else:
            metadatas = None
        embeddings = [d.embedding for d in docs]
        embeddings_arg: Any = embeddings if all(e is not None for e in embeddings) else None
        kwargs: dict[str, Any] = {"ids": ids, "documents": documents}
        if metadatas is not None:
            kwargs["metadatas"] = metadatas
        if embeddings_arg is not None:
            kwargs["embeddings"] = embeddings_arg
        try:
            self._collection.upsert(**kwargs)
        except Exception as exc:
            raise VectorStoreError(f"Chroma upsert failed: {exc}") from exc

    def query(
        self,
        *,
        vector: list[float] | None,
        text: str | None,
        top_k: int,
        where: dict[str, Any] | None,
        mode: SearchMode,
        alpha: float,
    ) -> list[QueryResult]:
        if mode == "hybrid":
            raise NotSupportedError(
                "ChromaStore does not support hybrid search. Use mode='semantic' or 'keyword'."
            )

        kwargs: dict[str, Any] = {"n_results": top_k}
        if where:
            kwargs["where"] = where

        if mode == "keyword":
            if not text:
                raise ValueError("keyword search requires `text`")
            kwargs["query_texts"] = [text]
            kwargs["where_document"] = {"$contains": text}
        else:  # semantic
            if vector is not None:
                kwargs["query_embeddings"] = [vector]
            elif text is not None:
                kwargs["query_texts"] = [text]
            else:
                raise ValueError("semantic query requires either `vector` or `text`")

        try:
            res = self._collection.query(**kwargs)
        except Exception as exc:
            raise VectorStoreError(f"Chroma query failed: {exc}") from exc

        ids = (res.get("ids") or [[]])[0]
        documents = (res.get("documents") or [[]])[0]
        metadatas = (res.get("metadatas") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        embeddings_out = (res.get("embeddings") or [[None] * len(ids)])[0]

        results: list[QueryResult] = []
        for i, doc_id in enumerate(ids):
            distance = distances[i] if i < len(distances) else 0.0
            score = 1.0 - distance  # cosine distance → similarity
            md = metadatas[i] if i < len(metadatas) else None
            # Strip our namespaced sentinel (see upsert); preserve everything else
            # including any user key that happens to be `_id`.
            if isinstance(md, dict):
                md = {k: v for k, v in md.items() if k != _SENTINEL_KEY}
            else:
                md = {}
            doc = Document(
                id=doc_id,
                text=documents[i] if i < len(documents) else "",
                metadata=md,
                embedding=embeddings_out[i] if i < len(embeddings_out) else None,
            )
            results.append(QueryResult(document=doc, score=score))
        return results

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        try:
            self._collection.delete(ids=ids)
        except Exception as exc:
            raise VectorStoreError(f"Chroma delete failed: {exc}") from exc

    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception as exc:
            raise VectorStoreError(f"Chroma count failed: {exc}") from exc


__all__ = ["ChromaStore"]

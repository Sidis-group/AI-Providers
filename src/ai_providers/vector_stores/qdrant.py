"""Qdrant vector store backend (semantic + hybrid via sparse vectors / RRF)."""

from __future__ import annotations

import hashlib
import re
import uuid
from collections import Counter
from typing import Any

from ..exceptions import NotSupportedError, ProviderNotInstalledError, VectorStoreError
from ..types import Document, QueryResult, SearchMode

try:
    from qdrant_client import QdrantClient  # type: ignore[import-untyped]
    from qdrant_client.http import models as qm  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - env without [vector-qdrant]
    QdrantClient = None  # type: ignore[assignment]
    qm = None  # type: ignore[assignment]


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _stable_token_index(token: str) -> int:
    """Deterministic 31-bit index for a token.

    CPython randomizes ``hash(str)`` per-process (PYTHONHASHSEED), which would
    make the same token map to different sparse-vector indices in different
    processes — silently breaking keyword/hybrid retrieval across upserts and
    queries that happen in different processes (the standard Qdrant deployment
    pattern). blake2b gives us a stable hash without external deps.
    """

    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


def _bow(text: str) -> dict[int, float]:
    """Tiny built-in BoW sparse vector for keyword/hybrid search.

    For production-grade BM25 use Qdrant's FastEmbed sparse models; this default
    keeps the package zero-dep beyond qdrant-client itself.
    """

    tokens = _TOKEN_RE.findall(text.lower())
    counts = Counter(tokens)
    out: dict[int, float] = {}
    for token, count in counts.items():
        out[_stable_token_index(token)] = float(count)
    return out


def _sparse_vector(text: str) -> Any:
    bow = _bow(text)
    return qm.SparseVector(indices=list(bow.keys()), values=list(bow.values()))


def _build_filter(where: dict[str, Any] | None) -> Any | None:
    if not where:
        return None
    must: list[Any] = []
    for key, value in where.items():
        must.append(qm.FieldCondition(key=key, match=qm.MatchValue(value=value)))
    return qm.Filter(must=must)


def _normalize_id(doc_id: str) -> str | int:
    """Qdrant accepts UUIDs or unsigned ints. Coerce arbitrary strings to UUID5 if needed."""

    if doc_id.isdigit():
        return int(doc_id)
    try:
        uuid.UUID(doc_id)
        return doc_id
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))


class QdrantStore:
    """Qdrant backend. Supports semantic, keyword (sparse) and hybrid (RRF / weighted)."""

    DENSE_NAME = "dense"
    SPARSE_NAME = "sparse"

    def __init__(
        self,
        *,
        collection: str,
        url: str | None = None,
        api_key: str | None = None,
        host: str | None = None,
        port: int | None = None,
        path: str | None = None,
        client: Any = None,
        vector_size: int | None = None,
        distance: str = "Cosine",
        hybrid_fusion: str = "rrf",
    ) -> None:
        if QdrantClient is None:
            raise ProviderNotInstalledError("vector-qdrant", "vector-qdrant")
        if hybrid_fusion not in ("rrf", "weighted"):
            raise ValueError("hybrid_fusion must be 'rrf' or 'weighted'")
        self.collection = collection
        self.hybrid_fusion = hybrid_fusion
        if client is not None:
            self._client = client
        elif url is not None:
            self._client = QdrantClient(url=url, api_key=api_key)
        elif host is not None:
            self._client = QdrantClient(host=host, port=port or 6333, api_key=api_key)
        elif path is not None:
            self._client = QdrantClient(path=path)
        else:
            self._client = QdrantClient(":memory:")
        self._vector_size = vector_size
        self._distance = distance
        self._original_id_to_qdrant: dict[str, str | int] = {}

    # ------------------------------------------------------------------

    def supports(self, mode: SearchMode) -> bool:
        return mode in ("semantic", "keyword", "hybrid")

    def _ensure_collection(self, vector_size: int) -> None:
        if self._client.collection_exists(self.collection):
            return
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config={
                self.DENSE_NAME: qm.VectorParams(
                    size=vector_size, distance=getattr(qm.Distance, self._distance.upper())
                )
            },
            sparse_vectors_config={
                self.SPARSE_NAME: qm.SparseVectorParams(),
            },
        )

    def upsert(self, docs: list[Document]) -> None:
        if not docs:
            return
        if any(d.embedding is None for d in docs):
            raise VectorStoreError(
                "QdrantStore.upsert requires Document.embedding for every document. "
                "Provide an EmbeddingsClient on the VectorStore facade or pre-compute embeddings."
            )
        vector_size = self._vector_size or len(docs[0].embedding)  # type: ignore[arg-type]
        self._vector_size = vector_size
        self._ensure_collection(vector_size)

        points = []
        for d in docs:
            point_id = _normalize_id(d.id)
            self._original_id_to_qdrant[d.id] = point_id
            payload = {"_id": d.id, "text": d.text, **(d.metadata or {})}
            points.append(
                qm.PointStruct(
                    id=point_id,
                    vector={
                        self.DENSE_NAME: d.embedding,
                        self.SPARSE_NAME: _sparse_vector(d.text),
                    },
                    payload=payload,
                )
            )
        try:
            self._client.upsert(collection_name=self.collection, points=points)
        except Exception as exc:
            raise VectorStoreError(f"Qdrant upsert failed: {exc}") from exc

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
        flt = _build_filter(where)
        try:
            if mode == "semantic":
                if vector is None:
                    raise ValueError("semantic query requires a vector")
                hits = self._client.query_points(
                    collection_name=self.collection,
                    query=vector,
                    using=self.DENSE_NAME,
                    limit=top_k,
                    query_filter=flt,
                    with_payload=True,
                ).points
            elif mode == "keyword":
                if text is None:
                    raise ValueError("keyword query requires text")
                hits = self._client.query_points(
                    collection_name=self.collection,
                    query=_sparse_vector(text),
                    using=self.SPARSE_NAME,
                    limit=top_k,
                    query_filter=flt,
                    with_payload=True,
                ).points
            elif mode == "hybrid":
                if vector is None or text is None:
                    raise ValueError("hybrid query requires both vector and text")
                if self.hybrid_fusion == "rrf":
                    hits = self._client.query_points(
                        collection_name=self.collection,
                        prefetch=[
                            qm.Prefetch(
                                query=vector, using=self.DENSE_NAME, limit=top_k * 4
                            ),
                            qm.Prefetch(
                                query=_sparse_vector(text),
                                using=self.SPARSE_NAME,
                                limit=top_k * 4,
                            ),
                        ],
                        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                        limit=top_k,
                        query_filter=flt,
                        with_payload=True,
                    ).points
                else:
                    hits = self._weighted_hybrid(vector, text, flt, top_k, alpha)
            else:  # pragma: no cover - guarded by supports()
                raise NotSupportedError(f"Unsupported mode: {mode}")
        except NotSupportedError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Qdrant query failed: {exc}") from exc

        results: list[QueryResult] = []
        for hit in hits:
            payload = hit.payload or {}
            doc = Document(
                id=str(payload.get("_id", hit.id)),
                text=payload.get("text", ""),
                metadata={k: v for k, v in payload.items() if k not in ("_id", "text")},
            )
            results.append(QueryResult(document=doc, score=float(hit.score)))
        return results

    def _weighted_hybrid(
        self,
        vector: list[float],
        text: str,
        flt: Any,
        top_k: int,
        alpha: float,
    ) -> list[Any]:
        dense = self._client.query_points(
            collection_name=self.collection,
            query=vector,
            using=self.DENSE_NAME,
            limit=top_k * 4,
            query_filter=flt,
            with_payload=True,
        ).points
        sparse = self._client.query_points(
            collection_name=self.collection,
            query=_sparse_vector(text),
            using=self.SPARSE_NAME,
            limit=top_k * 4,
            query_filter=flt,
            with_payload=True,
        ).points

        def _norm(items: list[Any]) -> dict[Any, float]:
            if not items:
                return {}
            scores = [h.score for h in items]
            mn, mx = min(scores), max(scores)
            denom = (mx - mn) or 1.0
            return {h.id: (h.score - mn) / denom for h in items}

        dn = _norm(dense)
        sn = _norm(sparse)
        all_ids = set(dn) | set(sn)
        by_id = {h.id: h for h in dense + sparse}
        merged = []
        for pid in all_ids:
            score = alpha * dn.get(pid, 0.0) + (1.0 - alpha) * sn.get(pid, 0.0)
            hit = by_id[pid]
            hit.score = score
            merged.append(hit)
        merged.sort(key=lambda h: h.score, reverse=True)
        return merged[:top_k]

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        normalized = [_normalize_id(i) for i in ids]
        try:
            self._client.delete(
                collection_name=self.collection,
                points_selector=qm.PointIdsList(points=normalized),
            )
        except Exception as exc:
            raise VectorStoreError(f"Qdrant delete failed: {exc}") from exc

    def count(self) -> int:
        try:
            res = self._client.count(collection_name=self.collection, exact=True)
            return int(res.count)
        except Exception as exc:
            raise VectorStoreError(f"Qdrant count failed: {exc}") from exc


__all__ = ["QdrantStore"]

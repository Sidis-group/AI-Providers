"""In-memory cache implementation with optional TTL."""

from __future__ import annotations

import threading
import time

from ..types import ChatResponse


class InMemoryCache:
    """Thread-safe in-process cache. Not shared across processes."""

    def __init__(self, default_ttl: int | None = None) -> None:
        self._store: dict[str, tuple[ChatResponse, float | None]] = {}
        self._lock = threading.Lock()
        self.default_ttl = default_ttl

    def _expired(self, expires_at: float | None) -> bool:
        return expires_at is not None and expires_at < time.time()

    def get(self, key: str) -> ChatResponse | None:
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            value, expires_at = item
            if self._expired(expires_at):
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value: ChatResponse, ttl: int | None = None) -> None:
        ttl = ttl if ttl is not None else self.default_ttl
        expires_at = time.time() + ttl if ttl else None
        with self._lock:
            self._store[key] = (value, expires_at)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

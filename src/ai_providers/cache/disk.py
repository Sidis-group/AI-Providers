"""Disk-backed cache (optional extra: ai-providers[cache-disk])."""

from __future__ import annotations

from typing import Any

from ..exceptions import ProviderNotInstalledError
from ..types import ChatResponse

try:
    import diskcache  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - exercised when extra is missing
    diskcache = None  # type: ignore[assignment]


class DiskCache:
    """Persistent cache backed by ``diskcache``. Requires ``[cache-disk]`` extra."""

    def __init__(self, directory: str = "./.ai_providers_cache", default_ttl: int | None = None):
        if diskcache is None:
            raise ProviderNotInstalledError("disk-cache", "cache-disk")
        self._cache: Any = diskcache.Cache(directory)
        self.default_ttl = default_ttl

    def get(self, key: str) -> ChatResponse | None:
        return self._cache.get(key)

    def set(self, key: str, value: ChatResponse, ttl: int | None = None) -> None:
        ttl = ttl if ttl is not None else self.default_ttl
        if ttl is not None:
            self._cache.set(key, value, expire=ttl)
        else:
            self._cache.set(key, value)

    def delete(self, key: str) -> None:
        self._cache.delete(key)

    def clear(self) -> None:
        self._cache.clear()

    def close(self) -> None:
        self._cache.close()

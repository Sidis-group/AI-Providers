"""Cache backend Protocol and key derivation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Protocol, runtime_checkable

from ..types import ChatResponse, Message, Tool


@runtime_checkable
class CacheBackend(Protocol):
    """Pluggable cache backend interface."""

    def get(self, key: str) -> ChatResponse | None: ...
    def set(self, key: str, value: ChatResponse, ttl: int | None = None) -> None: ...
    def delete(self, key: str) -> None: ...


def _serialize(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_serialize(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    return obj


def make_cache_key(
    *,
    provider: str,
    model: str,
    messages: list[Message],
    tools: list[Tool] | None,
    extra_params: dict[str, Any] | None,
) -> str:
    """Derive a stable cache key from a request."""

    payload = {
        "provider": provider,
        "model": model,
        "messages": _serialize(messages),
        "tools": _serialize(tools) if tools else None,
        "extra_params": _serialize(extra_params) if extra_params else None,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

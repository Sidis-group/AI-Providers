"""Embedding backend interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..types import EmbeddingResponse


class BaseEmbeddingProvider(ABC):
    name: str = "base"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str,
        timeout: float = 60.0,
        base_url: str | None = None,
        extra_params: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.base_url = base_url
        self.extra_params: dict[str, Any] = dict(extra_params or {})

    @abstractmethod
    def embed(self, texts: list[str]) -> EmbeddingResponse: ...


class BaseAsyncEmbeddingProvider(ABC):
    name: str = "base"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str,
        timeout: float = 60.0,
        base_url: str | None = None,
        extra_params: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.base_url = base_url
        self.extra_params: dict[str, Any] = dict(extra_params or {})

    @abstractmethod
    async def embed(self, texts: list[str]) -> EmbeddingResponse: ...

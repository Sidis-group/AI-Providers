"""Abstract provider interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any

from .types import ChatResponse, Message, StreamChunk, Tool


class BaseProvider(ABC):
    """Abstract base for synchronous chat providers."""

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
    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[Tool] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> ChatResponse: ...

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[Tool] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> Iterator[StreamChunk]: ...

    def _merged_params(self, extra_params: dict[str, Any] | None) -> dict[str, Any]:
        merged = dict(self.extra_params)
        if extra_params:
            merged.update(extra_params)
        return merged


class BaseAsyncProvider(ABC):
    """Abstract base for asynchronous chat providers."""

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
    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[Tool] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> ChatResponse: ...

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[Tool] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]: ...

    def _merged_params(self, extra_params: dict[str, Any] | None) -> dict[str, Any]:
        merged = dict(self.extra_params)
        if extra_params:
            merged.update(extra_params)
        return merged

"""Shared fixtures and helpers for the test suite."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pytest

from ai_providers.base import BaseAsyncProvider, BaseProvider
from ai_providers.registry import register_provider
from ai_providers.types import ChatResponse, Message, StreamChunk, Tool, Usage


@dataclass
class FakeCall:
    messages: list[Message]
    tools: list[Tool] | None
    extra_params: dict[str, Any]


@dataclass
class FakeProviderConfig:
    response_text: str = "ok"
    prompt_tokens: int = 10
    completion_tokens: int = 5
    raise_factory: Any = None
    stream_chunks: list[StreamChunk] = field(default_factory=list)


class FakeSyncProvider(BaseProvider):
    name = "fake"
    config: ClassVar[FakeProviderConfig] = FakeProviderConfig()
    calls: ClassVar[list[FakeCall]] = []
    raise_count: ClassVar[int] = 0

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def chat(self, messages, *, tools=None, extra_params=None):
        FakeSyncProvider.calls.append(
            FakeCall(messages=list(messages), tools=tools, extra_params=dict(extra_params or {}))
        )
        if FakeSyncProvider.config.raise_factory is not None:
            FakeSyncProvider.raise_count += 1
            exc = FakeSyncProvider.config.raise_factory(FakeSyncProvider.raise_count)
            if exc is not None:
                raise exc
        cfg = FakeSyncProvider.config
        return ChatResponse(
            text=cfg.response_text,
            tool_calls=[],
            finish_reason="stop",
            model=self.model,
            usage=Usage(
                prompt_tokens=cfg.prompt_tokens,
                completion_tokens=cfg.completion_tokens,
                total_tokens=cfg.prompt_tokens + cfg.completion_tokens,
            ),
            raw=None,
        )

    def stream(self, messages, *, tools=None, extra_params=None) -> Iterator[StreamChunk]:
        FakeSyncProvider.calls.append(
            FakeCall(messages=list(messages), tools=tools, extra_params=dict(extra_params or {}))
        )
        if FakeSyncProvider.config.raise_factory is not None:
            FakeSyncProvider.raise_count += 1
            exc = FakeSyncProvider.config.raise_factory(FakeSyncProvider.raise_count)
            if exc is not None:
                raise exc
        yield from FakeSyncProvider.config.stream_chunks


class FakeAsyncProvider(BaseAsyncProvider):
    name = "fake"
    config: ClassVar[FakeProviderConfig] = FakeProviderConfig()
    calls: ClassVar[list[FakeCall]] = []
    raise_count: ClassVar[int] = 0

    async def chat(self, messages, *, tools=None, extra_params=None):
        FakeAsyncProvider.calls.append(
            FakeCall(messages=list(messages), tools=tools, extra_params=dict(extra_params or {}))
        )
        if FakeAsyncProvider.config.raise_factory is not None:
            FakeAsyncProvider.raise_count += 1
            exc = FakeAsyncProvider.config.raise_factory(FakeAsyncProvider.raise_count)
            if exc is not None:
                raise exc
        cfg = FakeAsyncProvider.config
        return ChatResponse(
            text=cfg.response_text,
            tool_calls=[],
            finish_reason="stop",
            model=self.model,
            usage=Usage(
                prompt_tokens=cfg.prompt_tokens,
                completion_tokens=cfg.completion_tokens,
                total_tokens=cfg.prompt_tokens + cfg.completion_tokens,
            ),
            raw=None,
        )

    async def stream(self, messages, *, tools=None, extra_params=None):
        FakeAsyncProvider.calls.append(
            FakeCall(messages=list(messages), tools=tools, extra_params=dict(extra_params or {}))
        )
        for chunk in FakeAsyncProvider.config.stream_chunks:
            yield chunk


# Register the fake provider so AIClient(provider="fake", ...) works in tests.
register_provider("fake", sync_cls=FakeSyncProvider, async_cls=FakeAsyncProvider)


@pytest.fixture(autouse=True)
def _reset_fake_provider():
    FakeSyncProvider.calls = []
    FakeSyncProvider.raise_count = 0
    FakeSyncProvider.config = FakeProviderConfig()
    FakeAsyncProvider.calls = []
    FakeAsyncProvider.raise_count = 0
    FakeAsyncProvider.config = FakeProviderConfig()
    yield

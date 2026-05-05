"""Async client tests."""

from __future__ import annotations

import pytest

from ai_providers import AsyncAIClient, InMemoryCache
from ai_providers.exceptions import AuthenticationError
from ai_providers.types import StreamChunk, Usage
from tests.conftest import FakeAsyncProvider, FakeProviderConfig

pytestmark = pytest.mark.asyncio


async def test_async_chat_basic():
    FakeAsyncProvider.config = FakeProviderConfig(response_text="hi")
    client = AsyncAIClient(provider="fake", model="m", api_key="x")
    response = await client.chat([{"role": "user", "content": "ping"}])
    assert response.text == "hi"


async def test_async_cache():
    cache = InMemoryCache()
    client = AsyncAIClient(provider="fake", model="m", api_key="x", cache=cache)
    await client.chat([{"role": "user", "content": "ping"}])
    await client.chat([{"role": "user", "content": "ping"}])
    assert len(FakeAsyncProvider.calls) == 1


async def test_async_no_retry_on_auth_error():
    FakeAsyncProvider.config = FakeProviderConfig(
        raise_factory=lambda _n: AuthenticationError("bad")
    )
    client = AsyncAIClient(
        provider="fake", model="m", api_key="x", max_retries=3, initial_backoff=0.01
    )
    with pytest.raises(AuthenticationError):
        await client.chat([{"role": "user", "content": "ping"}])
    assert FakeAsyncProvider.raise_count == 1


async def test_async_streaming():
    FakeAsyncProvider.config = FakeProviderConfig(
        stream_chunks=[
            StreamChunk(delta="a"),
            StreamChunk(delta="b"),
            StreamChunk(delta="", finish_reason="stop", usage=Usage(2, 2, 4)),
        ]
    )
    client = AsyncAIClient(provider="fake", model="m", api_key="x")
    deltas: list[str] = []
    async for c in client.stream([{"role": "user", "content": "x"}]):
        deltas.append(c.delta)
    assert "".join(deltas) == "ab"

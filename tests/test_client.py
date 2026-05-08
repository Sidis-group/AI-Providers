"""Tests for AIClient facade using a fake provider."""

from __future__ import annotations

import pytest

from ai_providers import AIClient, InMemoryCache, MetricsMiddleware
from ai_providers.exceptions import (
    AuthenticationError,
    ProviderAPIError,
    RateLimitError,
)
from ai_providers.types import StreamChunk, Usage
from tests.conftest import FakeProviderConfig, FakeSyncProvider


def test_chat_basic():
    FakeSyncProvider.config = FakeProviderConfig(
        response_text="hello", prompt_tokens=10, completion_tokens=5
    )
    client = AIClient(provider="fake", model="any", api_key="x")
    response = client.chat([{"role": "user", "content": "ping"}])
    assert response.text == "hello"
    assert response.usage.total_tokens == 15
    assert len(FakeSyncProvider.calls) == 1


def test_chat_normalizes_dict_messages_to_objects():
    client = AIClient(provider="fake", model="any", api_key="x")
    client.chat([{"role": "user", "content": "ping"}])
    msgs = FakeSyncProvider.calls[0].messages
    assert msgs[0].role == "user"
    assert msgs[0].content == "ping"


def test_cache_short_circuits_provider_call():
    cache = InMemoryCache()
    client = AIClient(provider="fake", model="m", api_key="x", cache=cache)
    client.chat([{"role": "user", "content": "ping"}])
    client.chat([{"role": "user", "content": "ping"}])
    # Provider only invoked once because second call hits cache.
    assert len(FakeSyncProvider.calls) == 1


def test_cache_misses_when_messages_change():
    cache = InMemoryCache()
    client = AIClient(provider="fake", model="m", api_key="x", cache=cache)
    client.chat([{"role": "user", "content": "a"}])
    client.chat([{"role": "user", "content": "b"}])
    assert len(FakeSyncProvider.calls) == 2


def test_middleware_invoked():
    metrics = MetricsMiddleware()
    client = AIClient(provider="fake", model="m", api_key="x", middleware=[metrics])
    client.chat([{"role": "user", "content": "ping"}])
    assert metrics.requests == 1
    assert metrics.responses == 1
    assert metrics.errors == 0


def test_middleware_on_error():
    metrics = MetricsMiddleware()

    def factory(_n: int):
        return AuthenticationError("bad key")

    FakeSyncProvider.config = FakeProviderConfig(raise_factory=factory)
    client = AIClient(provider="fake", model="m", api_key="x", middleware=[metrics], max_retries=0)
    with pytest.raises(AuthenticationError):
        client.chat([{"role": "user", "content": "ping"}])
    assert metrics.errors == 1


def test_retry_on_rate_limit_then_success():
    def factory(n: int):
        if n == 1:
            return RateLimitError("rate")
        return None

    FakeSyncProvider.config = FakeProviderConfig(raise_factory=factory)
    client = AIClient(
        provider="fake",
        model="m",
        api_key="x",
        max_retries=3,
        initial_backoff=0.01,
        max_backoff=0.05,
    )
    response = client.chat([{"role": "user", "content": "ping"}])
    assert response.text == "ok"
    assert len(FakeSyncProvider.calls) == 2  # one failed call + one successful retry


def test_no_retry_on_authentication_error():
    def factory(_n: int):
        return AuthenticationError("bad")

    FakeSyncProvider.config = FakeProviderConfig(raise_factory=factory)
    client = AIClient(
        provider="fake", model="m", api_key="x", max_retries=3, initial_backoff=0.01
    )
    with pytest.raises(AuthenticationError):
        client.chat([{"role": "user", "content": "ping"}])
    assert FakeSyncProvider.raise_count == 1  # not retried


def test_retry_on_5xx_provider_api_error():
    def factory(n: int):
        if n == 1:
            return ProviderAPIError("server boom", status_code=502)
        return None

    FakeSyncProvider.config = FakeProviderConfig(raise_factory=factory)
    client = AIClient(provider="fake", model="m", api_key="x", initial_backoff=0.01)
    response = client.chat([{"role": "user", "content": "x"}])
    assert response.text == "ok"
    assert len(FakeSyncProvider.calls) == 2


def test_no_retry_on_4xx_provider_api_error():
    def factory(_n: int):
        return ProviderAPIError("bad", status_code=404)

    FakeSyncProvider.config = FakeProviderConfig(raise_factory=factory)
    client = AIClient(provider="fake", model="m", api_key="x", initial_backoff=0.01)
    with pytest.raises(ProviderAPIError):
        client.chat([{"role": "user", "content": "x"}])
    assert FakeSyncProvider.raise_count == 1


def test_cost_computed_for_known_model():
    FakeSyncProvider.config = FakeProviderConfig(prompt_tokens=1_000_000, completion_tokens=0)
    client = AIClient(provider="fake", model="gpt-4o-mini", api_key="x")
    # Use extra_pricing because "fake" provider isn't in the price table by default.
    client.extra_pricing = {"fake": {"gpt-4o-mini": {"input": 0.15, "output": 0.6}}}
    response = client.chat([{"role": "user", "content": "x"}])
    assert response.usage.cost_usd == pytest.approx(0.15)


def test_middleware_raising_does_not_break_call():
    """A buggy middleware must not crash the client — failures are swallowed and logged."""

    class BadMiddleware:
        def on_request(self, ctx):
            raise RuntimeError("middleware boom")

        def on_response(self, ctx):
            raise RuntimeError("middleware boom")

    metrics = MetricsMiddleware()
    client = AIClient(
        provider="fake", model="m", api_key="x", middleware=[BadMiddleware(), metrics]
    )
    # Should NOT raise — the buggy middleware is isolated.
    response = client.chat([{"role": "user", "content": "ping"}])
    assert response.text == "ok"
    # Subsequent middleware still ran.
    assert metrics.requests == 1
    assert metrics.responses == 1


def test_streaming_yields_chunks():
    chunks = [
        StreamChunk(delta="he"),
        StreamChunk(delta="llo"),
        StreamChunk(delta="", finish_reason="stop", usage=Usage(3, 1, 4)),
    ]
    FakeSyncProvider.config = FakeProviderConfig(stream_chunks=chunks)
    client = AIClient(provider="fake", model="m", api_key="x")
    deltas = [c.delta for c in client.stream([{"role": "user", "content": "hi"}])]
    assert "".join(deltas) == "hello"


def test_streaming_emits_on_response_even_without_final_usage():
    """Stream that ends without `usage` in any chunk must still trigger on_response."""

    chunks = [StreamChunk(delta="hi"), StreamChunk(delta="", finish_reason="stop")]
    FakeSyncProvider.config = FakeProviderConfig(stream_chunks=chunks)
    metrics = MetricsMiddleware()
    client = AIClient(provider="fake", model="m", api_key="x", middleware=[metrics])
    list(client.stream([{"role": "user", "content": "x"}]))  # consume
    assert metrics.requests == 1
    assert metrics.responses == 1
    assert metrics.errors == 0


def test_streaming_emits_on_error_and_skips_response():
    """When the stream raises, on_error fires and on_response does NOT."""

    def factory(_n: int):
        return RuntimeError("boom mid-stream")

    FakeSyncProvider.config = FakeProviderConfig(raise_factory=factory)
    metrics = MetricsMiddleware()
    client = AIClient(
        provider="fake", model="m", api_key="x", middleware=[metrics], max_retries=0
    )
    # The fake provider raises in stream() before yielding — that's fine for this assertion.
    with pytest.raises(RuntimeError):
        list(client.stream([{"role": "user", "content": "x"}]))
    assert metrics.requests == 1
    assert metrics.errors == 1
    assert metrics.responses == 0

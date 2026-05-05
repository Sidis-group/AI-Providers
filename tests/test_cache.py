"""Cache tests."""

from __future__ import annotations

import time

from ai_providers import InMemoryCache, Message
from ai_providers.cache.base import make_cache_key
from ai_providers.types import ChatResponse, Usage


def _resp(text: str = "hello") -> ChatResponse:
    return ChatResponse(text=text, model="m", usage=Usage(1, 1, 2))


def test_inmemory_set_get_delete():
    cache = InMemoryCache()
    cache.set("a", _resp("v"))
    assert cache.get("a").text == "v"
    cache.delete("a")
    assert cache.get("a") is None


def test_inmemory_ttl_expires():
    cache = InMemoryCache(default_ttl=1)
    cache.set("a", _resp())
    assert cache.get("a") is not None
    time.sleep(1.05)
    assert cache.get("a") is None


def test_make_cache_key_stable():
    msgs = [Message(role="user", content="hi")]
    a = make_cache_key(provider="openai", model="m", messages=msgs, tools=None, extra_params=None)
    b = make_cache_key(provider="openai", model="m", messages=msgs, tools=None, extra_params=None)
    assert a == b


def test_make_cache_key_changes_with_params():
    msgs = [Message(role="user", content="hi")]
    a = make_cache_key(provider="openai", model="m", messages=msgs, tools=None, extra_params=None)
    b = make_cache_key(
        provider="openai", model="m", messages=msgs, tools=None, extra_params={"temperature": 0.7}
    )
    assert a != b


def test_make_cache_key_different_models():
    msgs = [Message(role="user", content="hi")]
    a = make_cache_key(provider="openai", model="x", messages=msgs, tools=None, extra_params=None)
    b = make_cache_key(provider="openai", model="y", messages=msgs, tools=None, extra_params=None)
    assert a != b

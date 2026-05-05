"""Tests for unified types and pricing helpers."""

from __future__ import annotations

import pytest

from ai_providers.pricing import compute_cost, lookup_price
from ai_providers.types import Message, Usage, normalize_messages


def test_normalize_messages_from_dicts():
    msgs = normalize_messages([{"role": "user", "content": "hi"}])
    assert len(msgs) == 1
    assert isinstance(msgs[0], Message)
    assert msgs[0].role == "user"
    assert msgs[0].content == "hi"


def test_normalize_messages_passthrough():
    msg = Message(role="system", content="rules")
    out = normalize_messages([msg])
    assert out[0] is msg


def test_normalize_messages_invalid_role():
    with pytest.raises(ValueError):
        normalize_messages([{"role": "bot", "content": "x"}])


def test_lookup_price_known_model():
    p = lookup_price("openai", "gpt-4o-mini")
    assert p is not None
    assert p["input"] == pytest.approx(0.15)


def test_lookup_price_unknown_model():
    assert lookup_price("openai", "no-such-model") is None


def test_compute_cost_known():
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=500_000, total_tokens=1_500_000)
    cost = compute_cost("openai", "gpt-4o-mini", usage)
    # 1M * 0.15 + 0.5M * 0.6 = 0.15 + 0.30 = 0.45
    assert cost == pytest.approx(0.45)


def test_compute_cost_extra_pricing_override():
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=0, total_tokens=1_000_000)
    extra = {"openai": {"my-model": {"input": 1.0, "output": 2.0}}}
    cost = compute_cost("openai", "my-model", usage, extra)
    assert cost == pytest.approx(1.0)


def test_compute_cost_unknown_returns_none():
    usage = Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20)
    assert compute_cost("openai", "no-such-model", usage) is None

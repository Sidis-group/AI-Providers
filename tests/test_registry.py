"""Tests for provider registry and lazy loading errors."""

from __future__ import annotations

import pytest

from ai_providers import AIClient
from ai_providers.exceptions import ProviderNotInstalledError
from ai_providers.registry import (
    get_async_provider_class,
    get_provider_class,
    known_providers,
    register_provider,
)


def test_known_providers_includes_defaults():
    names = known_providers()
    assert "openai" in names
    assert "anthropic" in names


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        get_provider_class("nope")


def test_provider_not_installed_raises_when_sdk_missing(monkeypatch):
    """If the SDK class is missing, the provider __init__ must raise ProviderNotInstalledError."""

    from ai_providers.providers import openai as openai_provider

    monkeypatch.setattr(openai_provider, "OpenAI", None)
    with pytest.raises(ProviderNotInstalledError):
        openai_provider.OpenAIProvider(api_key="x", model="gpt-4o-mini")


def test_provider_not_installed_message_format():
    err = ProviderNotInstalledError("openai", "openai")
    assert "openai" in str(err)
    assert "pip install" in str(err)


def test_register_custom_provider_then_use_it():
    from tests.conftest import FakeAsyncProvider, FakeSyncProvider

    register_provider("custom-test", sync_cls=FakeSyncProvider, async_cls=FakeAsyncProvider)
    cls = get_provider_class("custom-test")
    assert cls is FakeSyncProvider
    cls_a = get_async_provider_class("custom-test")
    assert cls_a is FakeAsyncProvider

    client = AIClient(provider="custom-test", model="m", api_key="x")
    response = client.chat([{"role": "user", "content": "ping"}])
    assert response.text == "ok"

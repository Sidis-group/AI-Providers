"""OpenAI provider conversion + exception-mapping tests using mocks."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import openai as openai_sdk
import pytest

from ai_providers import AIClient, ContentPart, Message, Tool
from ai_providers.exceptions import (
    AuthenticationError,
    ContextLengthError,
    InvalidRequestError,
    ProviderAPIError,
    RateLimitError,
)
from ai_providers.exceptions import (
    TimeoutError as AIPTimeoutError,
)
from ai_providers.providers.openai import (
    OpenAIProvider,
    _convert_messages,
    _convert_tools,
    _map_exception,
)


def _patch_openai_client(monkeypatch, mock_client):
    monkeypatch.setattr(
        "ai_providers.providers.openai.OpenAI",
        lambda **kwargs: mock_client,
    )


def _fake_completion(text: str = "hi", prompt_tokens: int = 10, completion_tokens: int = 5):
    message = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage, model="gpt-4o-mini")


def test_convert_messages_string_content():
    out = _convert_messages([Message(role="user", content="hi")])
    assert out == [{"role": "user", "content": "hi"}]


def test_convert_messages_multimodal():
    msg = Message(
        role="user",
        content=[ContentPart(type="text", text="describe"), ContentPart(type="image", image_url="https://x/y.png")],
    )
    out = _convert_messages([msg])
    assert out[0]["content"][0] == {"type": "text", "text": "describe"}
    assert out[0]["content"][1]["type"] == "image_url"


def test_convert_tools():
    tools = [Tool(name="f", description="d", parameters={"type": "object"})]
    out = _convert_tools(tools)
    assert out[0]["type"] == "function"
    assert out[0]["function"]["name"] == "f"


def test_chat_returns_unified_response(monkeypatch):
    mock = MagicMock()
    mock.chat.completions.create = MagicMock(return_value=_fake_completion("hello"))
    _patch_openai_client(monkeypatch, mock)

    client = AIClient(provider="openai", api_key="sk", model="gpt-4o-mini")
    response = client.chat([{"role": "user", "content": "hi"}])
    assert response.text == "hello"
    assert response.usage.total_tokens == 15
    assert response.usage.cost_usd is not None  # gpt-4o-mini is in pricing table


def test_chat_with_tool_calls(monkeypatch):
    tc = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="get_weather", arguments=json.dumps({"city": "Kyiv"})),
    )
    msg = SimpleNamespace(content=None, tool_calls=[tc])
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2, total_tokens=7),
        model="gpt-4o-mini",
    )
    mock = MagicMock()
    mock.chat.completions.create = MagicMock(return_value=completion)
    _patch_openai_client(monkeypatch, mock)

    client = AIClient(provider="openai", api_key="sk", model="gpt-4o-mini")
    response = client.chat(
        [Message(role="user", content="weather?")],
        tools=[Tool(name="get_weather", description="", parameters={})],
    )
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].name == "get_weather"
    assert response.tool_calls[0].arguments == {"city": "Kyiv"}


def test_streaming_yields_text(monkeypatch):
    chunk1 = SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content="he", tool_calls=None), finish_reason=None)],
        usage=None,
    )
    chunk2 = SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content="llo", tool_calls=None), finish_reason=None)],
        usage=None,
    )
    final = SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=None), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    mock = MagicMock()
    mock.chat.completions.create = MagicMock(return_value=iter([chunk1, chunk2, final]))
    _patch_openai_client(monkeypatch, mock)

    client = AIClient(provider="openai", api_key="sk", model="gpt-4o-mini")
    deltas = [c.delta for c in client.stream([{"role": "user", "content": "hi"}])]
    assert "".join(deltas) == "hello"


# ----- exception mapping ----------------------------------------------------


def _build_openai_exc(cls, message: str = "boom"):
    """Bypass strict __init__ signatures in the openai SDK."""

    obj = cls.__new__(cls)
    Exception.__init__(obj, message)
    return obj


@pytest.mark.parametrize(
    "openai_exc, expected",
    [
        (openai_sdk.AuthenticationError, AuthenticationError),
        (openai_sdk.RateLimitError, RateLimitError),
        (openai_sdk.APITimeoutError, AIPTimeoutError),
        (openai_sdk.BadRequestError, InvalidRequestError),
    ],
)
def test_exception_mapping(openai_exc, expected):
    exc = _build_openai_exc(openai_exc)
    mapped = _map_exception(exc)
    assert isinstance(mapped, expected)


def test_context_length_error_mapping():
    exc = _build_openai_exc(openai_sdk.BadRequestError, "context_length_exceeded")
    mapped = _map_exception(exc)
    assert isinstance(mapped, ContextLengthError)


def test_api_status_error_mapping():
    exc = _build_openai_exc(openai_sdk.APIStatusError, "boom")
    exc.status_code = 500
    mapped = _map_exception(exc)
    assert isinstance(mapped, ProviderAPIError)


def test_provider_call_raises_mapped_exception(monkeypatch):
    mock = MagicMock()
    mock.chat.completions.create = MagicMock(
        side_effect=_build_openai_exc(openai_sdk.AuthenticationError, "bad")
    )
    _patch_openai_client(monkeypatch, mock)

    provider = OpenAIProvider(api_key="sk", model="gpt-4o-mini")
    with pytest.raises(AuthenticationError):
        provider.chat([Message(role="user", content="x")])

"""Anthropic provider conversion + exception-mapping tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic as anthropic_sdk
import pytest

from ai_providers import AIClient, ContentPart, Message, Tool
from ai_providers.exceptions import (
    AuthenticationError,
    ContextLengthError,
    InvalidRequestError,
    RateLimitError,
)
from ai_providers.providers.anthropic import (
    AnthropicProvider,
    _convert_messages,
    _convert_tools,
    _map_exception,
    _split_system,
)


def _patch_anthropic_client(monkeypatch, mock_client):
    monkeypatch.setattr(
        "ai_providers.providers.anthropic.Anthropic",
        lambda **kwargs: mock_client,
    )


def _fake_message(text: str = "hi", input_tokens: int = 10, output_tokens: int = 5):
    block = SimpleNamespace(type="text", text=text)
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(
        content=[block], usage=usage, model="claude-3-5-sonnet-20241022", stop_reason="end_turn"
    )


def test_split_system_extracts_system_prompt():
    msgs = [
        Message(role="system", content="rules"),
        Message(role="user", content="hi"),
    ]
    system, rest = _split_system(msgs)
    assert system == "rules"
    assert len(rest) == 1
    assert rest[0].role == "user"


def test_convert_messages_skips_system():
    msgs = [
        Message(role="system", content="x"),
        Message(role="user", content="hi"),
    ]
    out = _convert_messages(msgs)
    # System is filtered out — handled at the create() layer.
    assert all(m["role"] != "system" for m in out)


def test_convert_messages_multimodal_base64():
    msg = Message(
        role="user",
        content=[
            ContentPart(type="text", text="describe"),
            ContentPart(type="image", image_base64="ZmFrZQ==", mime_type="image/png"),
        ],
    )
    out = _convert_messages([msg])
    assert out[0]["content"][1]["type"] == "image"
    assert out[0]["content"][1]["source"]["type"] == "base64"
    assert out[0]["content"][1]["source"]["data"] == "ZmFrZQ=="


def test_convert_tools_uses_input_schema():
    tools = [Tool(name="f", description="d", parameters={"type": "object"})]
    out = _convert_tools(tools)
    assert out[0]["name"] == "f"
    assert out[0]["input_schema"] == {"type": "object"}


def test_chat_returns_unified_response(monkeypatch):
    mock = MagicMock()
    mock.messages.create = MagicMock(return_value=_fake_message("hello"))
    _patch_anthropic_client(monkeypatch, mock)

    client = AIClient(provider="anthropic", api_key="sk", model="claude-3-5-sonnet-20241022")
    response = client.chat(
        [Message(role="system", content="rules"), Message(role="user", content="hi")]
    )
    assert response.text == "hello"
    assert response.finish_reason == "stop"
    # Ensure system was forwarded as separate kwarg.
    call_kwargs = mock.messages.create.call_args.kwargs
    assert call_kwargs["system"] == "rules"
    assert call_kwargs["max_tokens"] >= 1


def test_chat_with_tool_use(monkeypatch):
    tool_block = SimpleNamespace(
        type="tool_use", id="t1", name="get_weather", input={"city": "Kyiv"}
    )
    msg = SimpleNamespace(
        content=[tool_block],
        usage=SimpleNamespace(input_tokens=5, output_tokens=3),
        model="claude-3-5-sonnet-20241022",
        stop_reason="tool_use",
    )
    mock = MagicMock()
    mock.messages.create = MagicMock(return_value=msg)
    _patch_anthropic_client(monkeypatch, mock)

    client = AIClient(provider="anthropic", api_key="sk", model="claude-3-5-sonnet-20241022")
    response = client.chat(
        [Message(role="user", content="weather?")],
        tools=[Tool(name="get_weather", description="", parameters={})],
    )
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].name == "get_weather"
    assert response.tool_calls[0].arguments == {"city": "Kyiv"}


def test_streaming_text(monkeypatch):
    events = [
        SimpleNamespace(type="message_start", message=SimpleNamespace(usage=SimpleNamespace(input_tokens=4, output_tokens=0))),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="text_delta", text="he"),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="text_delta", text="llo"),
        ),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=SimpleNamespace(output_tokens=2),
        ),
        SimpleNamespace(type="message_stop"),
    ]
    mock = MagicMock()
    mock.messages.create = MagicMock(return_value=iter(events))
    _patch_anthropic_client(monkeypatch, mock)

    client = AIClient(provider="anthropic", api_key="sk", model="claude-3-5-sonnet-20241022")
    chunks = list(client.stream([Message(role="user", content="hi")]))
    text = "".join(c.delta for c in chunks)
    assert text == "hello"
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.total_tokens == 6


# ---- Exception mapping ----------------------------------------------------


def _build_exc(cls, message: str = "boom"):
    """Construct an SDK exception bypassing strict __init__ signatures."""

    obj = cls.__new__(cls)
    Exception.__init__(obj, message)
    return obj


@pytest.mark.parametrize(
    "anthropic_exc, expected",
    [
        (anthropic_sdk.AuthenticationError, AuthenticationError),
        (anthropic_sdk.RateLimitError, RateLimitError),
        (anthropic_sdk.BadRequestError, InvalidRequestError),
    ],
)
def test_exception_mapping(anthropic_exc, expected):
    mapped = _map_exception(_build_exc(anthropic_exc))
    assert isinstance(mapped, expected)


def test_context_length_error_mapping():
    exc = _build_exc(anthropic_sdk.BadRequestError, "Prompt is too long for the context window")
    mapped = _map_exception(exc)
    assert isinstance(mapped, ContextLengthError)


def test_provider_raises_mapped(monkeypatch):
    mock = MagicMock()
    mock.messages.create = MagicMock(
        side_effect=_build_exc(anthropic_sdk.AuthenticationError, "bad key")
    )
    _patch_anthropic_client(monkeypatch, mock)

    provider = AnthropicProvider(api_key="sk", model="claude-3-5-sonnet-20241022")
    with pytest.raises(AuthenticationError):
        provider.chat([Message(role="user", content="hi")])

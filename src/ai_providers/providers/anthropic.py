"""Anthropic (Claude) provider implementation (sync + async)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

from ..base import BaseAsyncProvider, BaseProvider
from ..exceptions import (
    AIProviderError,
    AuthenticationError,
    ContextLengthError,
    InvalidRequestError,
    ProviderAPIError,
    ProviderNotInstalledError,
    RateLimitError,
)
from ..exceptions import (
    TimeoutError as AIPTimeoutError,
)
from ..types import (
    ChatResponse,
    ContentPart,
    FinishReason,
    Message,
    StreamChunk,
    Tool,
    ToolCall,
    Usage,
)

try:
    import anthropic as _anthropic_sdk  # type: ignore[import-untyped]
    from anthropic import Anthropic, AsyncAnthropic  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - env without [anthropic]
    _anthropic_sdk = None  # type: ignore[assignment]
    Anthropic = None  # type: ignore[assignment]
    AsyncAnthropic = None  # type: ignore[assignment]


DEFAULT_MAX_TOKENS = 1024


# ---------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------


def _content_part_to_block(p: ContentPart) -> dict[str, Any]:
    if p.type == "text":
        return {"type": "text", "text": p.text or ""}
    if p.type == "image":
        if p.image_base64:
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": p.mime_type or "image/png",
                    "data": p.image_base64,
                },
            }
        if p.image_url:
            return {"type": "image", "source": {"type": "url", "url": p.image_url}}
        raise ValueError("ContentPart(type='image') requires image_url or image_base64")
    raise ValueError(f"Anthropic does not support content type: {p.type}")


def _convert_user_or_assistant_content(content: str | list[ContentPart]) -> Any:
    if isinstance(content, str):
        return content
    return [_content_part_to_block(p) for p in content]


def _split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
    system_chunks: list[str] = []
    rest: list[Message] = []
    for m in messages:
        if m.role == "system":
            if isinstance(m.content, str):
                system_chunks.append(m.content)
            else:
                for p in m.content:
                    if p.type == "text" and p.text:
                        system_chunks.append(p.text)
        else:
            rest.append(m)
    system = "\n\n".join(system_chunks) if system_chunks else None
    return system, rest


def _convert_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            continue  # handled separately
        if m.role == "tool":
            # tool result: must be a "user" message with tool_result block
            text = m.content if isinstance(m.content, str) else ""
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.tool_call_id or "",
                            "content": text,
                        }
                    ],
                }
            )
            continue

        if m.role == "assistant" and m.tool_calls:
            blocks: list[dict[str, Any]] = []
            if isinstance(m.content, str) and m.content:
                blocks.append({"type": "text", "text": m.content})
            elif isinstance(m.content, list):
                blocks.extend(_content_part_to_block(p) for p in m.content)
            for tc in m.tool_calls:
                blocks.append(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                )
            out.append({"role": "assistant", "content": blocks})
            continue

        out.append(
            {
                "role": m.role,
                "content": _convert_user_or_assistant_content(m.content),
            }
        )
    return out


def _convert_tools(tools: list[Tool] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        for t in tools
    ]


def _normalize_finish(reason: str | None) -> FinishReason:
    mapping: dict[str | None, FinishReason] = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }
    return mapping.get(reason, "stop")


def _to_response(message: Any, model: str) -> ChatResponse:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in getattr(message, "content", []) or []:
        block_type = getattr(block, "type", None) or (
            block.get("type") if isinstance(block, dict) else None
        )
        if block_type == "text":
            text = getattr(block, "text", None)
            if text is None and isinstance(block, dict):
                text = block.get("text", "")
            if text:
                text_parts.append(text)
        elif block_type == "tool_use":
            tc_id = getattr(block, "id", None) if not isinstance(block, dict) else block.get("id")
            tc_name = (
                getattr(block, "name", None) if not isinstance(block, dict) else block.get("name")
            )
            tc_input = (
                getattr(block, "input", None)
                if not isinstance(block, dict)
                else block.get("input")
            )
            tool_calls.append(
                ToolCall(id=tc_id or "", name=tc_name or "", arguments=dict(tc_input or {}))
            )
    raw_usage = getattr(message, "usage", None)
    usage = Usage()
    if raw_usage is not None:
        prompt = getattr(raw_usage, "input_tokens", 0) or 0
        completion = getattr(raw_usage, "output_tokens", 0) or 0
        usage = Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )
    return ChatResponse(
        text="".join(text_parts),
        tool_calls=tool_calls,
        finish_reason=_normalize_finish(getattr(message, "stop_reason", None)),
        model=getattr(message, "model", model),
        usage=usage,
        raw=message,
    )


# Streaming state machine
class _StreamState:
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.tool_buffers: dict[int, dict[str, Any]] = {}
        self.finish_reason: FinishReason | None = None

    def usage(self) -> Usage:
        return Usage(
            prompt_tokens=self.input_tokens,
            completion_tokens=self.output_tokens,
            total_tokens=self.input_tokens + self.output_tokens,
        )


def _process_stream_event(event: Any, state: _StreamState) -> StreamChunk:
    etype = getattr(event, "type", None)
    delta_text = ""
    tool_call_delta: ToolCall | None = None
    finish_reason: FinishReason | None = None
    usage: Usage | None = None

    if etype == "message_start":
        message = getattr(event, "message", None)
        if message is not None:
            u = getattr(message, "usage", None)
            if u is not None:
                state.input_tokens = getattr(u, "input_tokens", 0) or 0
                state.output_tokens = getattr(u, "output_tokens", 0) or 0
    elif etype == "content_block_start":
        block = getattr(event, "content_block", None)
        idx = getattr(event, "index", 0)
        if block is not None and getattr(block, "type", None) == "tool_use":
            state.tool_buffers[idx] = {
                "id": getattr(block, "id", "") or "",
                "name": getattr(block, "name", "") or "",
                "json": "",
            }
    elif etype == "content_block_delta":
        delta = getattr(event, "delta", None)
        idx = getattr(event, "index", 0)
        if delta is not None:
            d_type = getattr(delta, "type", None)
            if d_type == "text_delta":
                delta_text = getattr(delta, "text", "") or ""
            elif d_type == "input_json_delta":
                buf = state.tool_buffers.get(idx)
                if buf is not None:
                    buf["json"] += getattr(delta, "partial_json", "") or ""
    elif etype == "content_block_stop":
        idx = getattr(event, "index", 0)
        buf = state.tool_buffers.pop(idx, None)
        if buf is not None:
            try:
                args = json.loads(buf["json"]) if buf["json"] else {}
            except json.JSONDecodeError:
                args = {"_raw": buf["json"]}
            tool_call_delta = ToolCall(id=buf["id"], name=buf["name"], arguments=args)
    elif etype == "message_delta":
        delta = getattr(event, "delta", None)
        if delta is not None:
            stop = getattr(delta, "stop_reason", None)
            if stop is not None:
                finish_reason = _normalize_finish(stop)
                state.finish_reason = finish_reason
        u = getattr(event, "usage", None)
        if u is not None:
            new_out = getattr(u, "output_tokens", None)
            if new_out is not None:
                state.output_tokens = new_out
    elif etype == "message_stop":
        finish_reason = state.finish_reason or "stop"
        usage = state.usage()

    return StreamChunk(
        delta=delta_text,
        tool_call_delta=tool_call_delta,
        finish_reason=finish_reason,
        usage=usage,
        raw=event,
    )


# ---------------------------------------------------------------------
# Exception mapping
# ---------------------------------------------------------------------


def _map_exception(exc: BaseException) -> BaseException:
    if _anthropic_sdk is None:  # pragma: no cover
        return exc
    try:
        if isinstance(exc, _anthropic_sdk.AuthenticationError):
            return AuthenticationError(str(exc))
        if isinstance(exc, _anthropic_sdk.RateLimitError):
            retry_after = None
            response = getattr(exc, "response", None)
            if response is not None:
                headers = getattr(response, "headers", None)
                if headers is not None:
                    val = headers.get("retry-after")
                    if val is not None:
                        try:
                            retry_after = float(val)
                        except (TypeError, ValueError):
                            retry_after = None
            return RateLimitError(str(exc), retry_after=retry_after)
        if isinstance(exc, _anthropic_sdk.APITimeoutError):
            return AIPTimeoutError(str(exc))
        if isinstance(exc, _anthropic_sdk.BadRequestError):
            msg = str(exc).lower()
            if "context" in msg and ("length" in msg or "window" in msg):
                return ContextLengthError(str(exc))
            return InvalidRequestError(str(exc))
        if isinstance(exc, _anthropic_sdk.APIStatusError):
            return ProviderAPIError(str(exc), status_code=getattr(exc, "status_code", None))
        if isinstance(exc, _anthropic_sdk.APIError):
            return ProviderAPIError(str(exc))
    except AttributeError:
        pass
    return exc


def _reraise(exc: BaseException) -> None:
    mapped = _map_exception(exc)
    if mapped is exc:
        raise
    raise mapped from exc


# ---------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------


def _build_create_kwargs(
    model: str,
    messages: list[Message],
    tools: list[Tool] | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Construct Anthropic ``messages.create`` kwargs from the unified call shape."""

    system, rest = _split_system(messages)
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": extra.pop("max_tokens", DEFAULT_MAX_TOKENS),
        "messages": _convert_messages(rest),
    }
    if system is not None:
        params["system"] = system
    converted_tools = _convert_tools(tools)
    if converted_tools is not None:
        params["tools"] = converted_tools
    params.update(extra)
    return params


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if Anthropic is None:
            raise ProviderNotInstalledError("anthropic", "anthropic")
        self._client = Anthropic(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[Tool] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> ChatResponse:
        extra = self._merged_params(extra_params)
        kwargs = _build_create_kwargs(self.model, messages, tools, extra)
        try:
            message = self._client.messages.create(**kwargs)
        except AIProviderError:
            raise
        except Exception as exc:
            _reraise(exc)
            raise  # pragma: no cover
        return _to_response(message, self.model)

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[Tool] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> Iterator[StreamChunk]:
        extra = self._merged_params(extra_params)
        kwargs = _build_create_kwargs(self.model, messages, tools, extra)
        kwargs["stream"] = True
        state = _StreamState()
        try:
            stream = self._client.messages.create(**kwargs)
        except Exception as exc:
            _reraise(exc)
            raise  # pragma: no cover
        try:
            for event in stream:
                yield _process_stream_event(event, state)
        except Exception as exc:
            _reraise(exc)
            raise  # pragma: no cover


# ---------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------


class AsyncAnthropicProvider(BaseAsyncProvider):
    name = "anthropic"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if AsyncAnthropic is None:
            raise ProviderNotInstalledError("anthropic", "anthropic")
        self._client = AsyncAnthropic(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )

    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[Tool] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> ChatResponse:
        extra = self._merged_params(extra_params)
        kwargs = _build_create_kwargs(self.model, messages, tools, extra)
        try:
            message = await self._client.messages.create(**kwargs)
        except AIProviderError:
            raise
        except Exception as exc:
            _reraise(exc)
            raise  # pragma: no cover
        return _to_response(message, self.model)

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[Tool] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        extra = self._merged_params(extra_params)
        kwargs = _build_create_kwargs(self.model, messages, tools, extra)
        kwargs["stream"] = True
        state = _StreamState()
        try:
            stream = await self._client.messages.create(**kwargs)
        except Exception as exc:
            _reraise(exc)
            raise  # pragma: no cover
        try:
            async for event in stream:
                yield _process_stream_event(event, state)
        except Exception as exc:
            _reraise(exc)
            raise  # pragma: no cover


__all__ = ["AnthropicProvider", "AsyncAnthropicProvider"]

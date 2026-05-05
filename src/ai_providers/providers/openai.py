"""OpenAI provider implementation (sync + async)."""

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
    import openai as _openai_sdk  # type: ignore[import-untyped]
    from openai import AsyncOpenAI, OpenAI  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - env without [openai]
    _openai_sdk = None  # type: ignore[assignment]
    OpenAI = None  # type: ignore[assignment]
    AsyncOpenAI = None  # type: ignore[assignment]


# ---------------------------------------------------------------------
# Conversion helpers (shared between sync + async)
# ---------------------------------------------------------------------


def _convert_content(content: str | list[ContentPart]) -> Any:
    if isinstance(content, str):
        return content
    parts: list[dict[str, Any]] = []
    for p in content:
        if p.type == "text":
            parts.append({"type": "text", "text": p.text or ""})
        elif p.type == "image":
            url = p.image_url
            if url is None and p.image_base64:
                mime = p.mime_type or "image/png"
                url = f"data:{mime};base64,{p.image_base64}"
            if url is None:
                raise ValueError("ContentPart(type='image') requires image_url or image_base64")
            parts.append({"type": "image_url", "image_url": {"url": url}})
        elif p.type == "audio":
            if not p.image_base64:
                raise ValueError("Audio content currently requires base64 data")
            parts.append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": p.image_base64, "format": p.mime_type or "wav"},
                }
            )
    return parts


def _convert_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        item: dict[str, Any] = {"role": m.role, "content": _convert_content(m.content)}
        if m.name:
            item["name"] = m.name
        if m.tool_call_id:
            item["tool_call_id"] = m.tool_call_id
        if m.tool_calls:
            item["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in m.tool_calls
            ]
        out.append(item)
    return out


def _convert_tools(tools: list[Tool] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def _parse_tool_calls(raw_tool_calls: Any) -> list[ToolCall]:
    if not raw_tool_calls:
        return []
    out: list[ToolCall] = []
    for tc in raw_tool_calls:
        if isinstance(tc, dict):
            fn = tc.get("function")
            tc_id = tc.get("id", "")
        else:
            fn = getattr(tc, "function", None)
            tc_id = getattr(tc, "id", "") or ""
        if fn is None:
            continue
        if isinstance(fn, dict):
            name = fn.get("name", "")
            args_raw = fn.get("arguments", "")
        else:
            name = getattr(fn, "name", "") or ""
            args_raw = getattr(fn, "arguments", "") or ""
        try:
            if isinstance(args_raw, str):
                args = json.loads(args_raw) if args_raw else {}
            elif args_raw:
                args = dict(args_raw)
            else:
                args = {}
        except json.JSONDecodeError:
            args = {"_raw": args_raw}
        out.append(ToolCall(id=tc_id or "", name=name or "", arguments=args))
    return out


def _normalize_finish(reason: str | None) -> FinishReason:
    if reason in ("stop", "length", "tool_calls", "content_filter"):
        return reason  # type: ignore[return-value]
    if reason == "function_call":
        return "tool_calls"
    return "stop"


def _to_response(completion: Any, model: str) -> ChatResponse:
    choice = completion.choices[0]
    message = choice.message
    text = message.content or ""
    tool_calls = _parse_tool_calls(getattr(message, "tool_calls", None))
    usage = Usage()
    raw_usage = getattr(completion, "usage", None)
    if raw_usage is not None:
        usage = Usage(
            prompt_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(raw_usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(raw_usage, "total_tokens", 0) or 0,
        )
    return ChatResponse(
        text=text,
        tool_calls=tool_calls,
        finish_reason=_normalize_finish(getattr(choice, "finish_reason", None)),
        model=getattr(completion, "model", model),
        usage=usage,
        raw=completion,
    )


def _stream_chunk(event: Any) -> StreamChunk:
    delta_text = ""
    tool_call_delta: ToolCall | None = None
    finish_reason: FinishReason | None = None
    usage: Usage | None = None

    choices = getattr(event, "choices", None) or []
    if choices:
        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if delta is not None:
            delta_text = getattr(delta, "content", None) or ""
            tcs = getattr(delta, "tool_calls", None)
            if tcs:
                parsed = _parse_tool_calls(tcs)
                if parsed:
                    tool_call_delta = parsed[0]
        fr = getattr(choice, "finish_reason", None)
        if fr is not None:
            finish_reason = _normalize_finish(fr)

    raw_usage = getattr(event, "usage", None)
    if raw_usage is not None:
        usage = Usage(
            prompt_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(raw_usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(raw_usage, "total_tokens", 0) or 0,
        )

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
    if _openai_sdk is None:  # pragma: no cover
        return exc
    try:
        if isinstance(exc, _openai_sdk.AuthenticationError):
            return AuthenticationError(str(exc))
        if isinstance(exc, _openai_sdk.RateLimitError):
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
        if isinstance(exc, _openai_sdk.APITimeoutError):
            return AIPTimeoutError(str(exc))
        if isinstance(exc, _openai_sdk.BadRequestError):
            msg = str(exc).lower()
            if "context length" in msg or "context_length" in msg or "maximum context" in msg:
                return ContextLengthError(str(exc))
            return InvalidRequestError(str(exc))
        if isinstance(exc, _openai_sdk.APIStatusError):
            return ProviderAPIError(str(exc), status_code=getattr(exc, "status_code", None))
        if isinstance(exc, _openai_sdk.APIError):
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
# Sync provider
# ---------------------------------------------------------------------


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if OpenAI is None:
            raise ProviderNotInstalledError("openai", "openai")
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[Tool] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> ChatResponse:
        params = self._merged_params(extra_params)
        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=_convert_messages(messages),
                tools=_convert_tools(tools),
                stream=False,
                **params,
            )
        except AIProviderError:
            raise
        except Exception as exc:
            _reraise(exc)
            raise  # pragma: no cover
        return _to_response(completion, self.model)

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[Tool] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> Iterator[StreamChunk]:
        params = self._merged_params(extra_params)
        params.setdefault("stream_options", {"include_usage": True})
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=_convert_messages(messages),
                tools=_convert_tools(tools),
                stream=True,
                **params,
            )
        except Exception as exc:
            _reraise(exc)
            raise  # pragma: no cover

        try:
            for event in stream:
                yield _stream_chunk(event)
        except Exception as exc:
            _reraise(exc)
            raise  # pragma: no cover


# ---------------------------------------------------------------------
# Async provider
# ---------------------------------------------------------------------


class AsyncOpenAIProvider(BaseAsyncProvider):
    name = "openai"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if AsyncOpenAI is None:
            raise ProviderNotInstalledError("openai", "openai")
        self._client = AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )

    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[Tool] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> ChatResponse:
        params = self._merged_params(extra_params)
        try:
            completion = await self._client.chat.completions.create(
                model=self.model,
                messages=_convert_messages(messages),
                tools=_convert_tools(tools),
                stream=False,
                **params,
            )
        except AIProviderError:
            raise
        except Exception as exc:
            _reraise(exc)
            raise  # pragma: no cover
        return _to_response(completion, self.model)

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[Tool] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        params = self._merged_params(extra_params)
        params.setdefault("stream_options", {"include_usage": True})
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=_convert_messages(messages),
                tools=_convert_tools(tools),
                stream=True,
                **params,
            )
        except Exception as exc:
            _reraise(exc)
            raise  # pragma: no cover

        try:
            async for event in stream:
                yield _stream_chunk(event)
        except Exception as exc:
            _reraise(exc)
            raise  # pragma: no cover


__all__ = ["AsyncOpenAIProvider", "OpenAIProvider"]

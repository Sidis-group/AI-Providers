"""High-level facade: AIClient and AsyncAIClient."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from .base import BaseAsyncProvider, BaseProvider
from .cache.base import CacheBackend, make_cache_key
from .middleware import (
    ErrorContext,
    Middleware,
    RequestContext,
    ResponseContext,
    _safe_invoke,
    _Timer,
)
from .pricing import compute_cost
from .registry import get_async_provider_class, get_provider_class
from .retries import acall_with_retry, call_with_retry
from .types import ChatResponse, Message, StreamChunk, Tool, Usage, normalize_messages


def _apply_cost(
    response: ChatResponse,
    provider: str,
    model: str,
    extra_pricing: dict[str, dict[str, dict[str, float]]] | None,
) -> ChatResponse:
    if response.usage.cost_usd is None:
        response.usage.cost_usd = compute_cost(provider, model, response.usage, extra_pricing)
    return response


class AIClient:
    """Synchronous facade over a provider implementation."""

    def __init__(
        self,
        *,
        provider: str,
        api_key: str | None = None,
        model: str,
        timeout: float = 60.0,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        cache: CacheBackend | None = None,
        cache_ttl: int | None = None,
        middleware: list[Middleware] | None = None,
        base_url: str | None = None,
        extra_params: dict[str, Any] | None = None,
        extra_pricing: dict[str, dict[str, dict[str, float]]] | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.cache = cache
        self.cache_ttl = cache_ttl
        self.middleware: list[Middleware] = list(middleware or [])
        self.extra_pricing = extra_pricing

        cls = get_provider_class(provider)
        self._provider: BaseProvider = cls(
            api_key=api_key,
            model=model,
            timeout=timeout,
            base_url=base_url,
            extra_params=extra_params,
        )

    # ---- public API --------------------------------------------------

    def chat(
        self,
        messages: list[Message] | list[dict[str, Any]],
        *,
        tools: list[Tool] | None = None,
        extra_params: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> ChatResponse:
        msgs = normalize_messages(messages)
        merged_extra = self._provider._merged_params(extra_params)

        ctx = RequestContext(
            provider=self.provider, model=self.model, messages=msgs, extra_params=merged_extra
        )
        _safe_invoke(self.middleware, "on_request", ctx)
        timer = _Timer()

        if self.cache is not None and use_cache:
            key = make_cache_key(
                provider=self.provider,
                model=self.model,
                messages=msgs,
                tools=tools,
                extra_params=merged_extra,
            )
            cached = self.cache.get(key)
            if cached is not None:
                _safe_invoke(
                    self.middleware,
                    "on_response",
                    ResponseContext(
                        provider=self.provider,
                        model=self.model,
                        messages=msgs,
                        extra_params=merged_extra,
                        response=cached,
                        duration_ms=timer.ms(),
                    ),
                )
                return cached

        try:
            response = call_with_retry(
                lambda: self._provider.chat(msgs, tools=tools, extra_params=extra_params),
                max_retries=self.max_retries,
                initial_backoff=self.initial_backoff,
                max_backoff=self.max_backoff,
            )
        except BaseException as exc:
            _safe_invoke(
                self.middleware,
                "on_error",
                ErrorContext(
                    provider=self.provider,
                    model=self.model,
                    messages=msgs,
                    extra_params=merged_extra,
                    error=exc,
                    duration_ms=timer.ms(),
                ),
            )
            raise

        response = _apply_cost(response, self.provider, self.model, self.extra_pricing)

        if self.cache is not None and use_cache:
            self.cache.set(key, response, ttl=self.cache_ttl)

        _safe_invoke(
            self.middleware,
            "on_response",
            ResponseContext(
                provider=self.provider,
                model=self.model,
                messages=msgs,
                extra_params=merged_extra,
                response=response,
                duration_ms=timer.ms(),
            ),
        )
        return response

    def stream(
        self,
        messages: list[Message] | list[dict[str, Any]],
        *,
        tools: list[Tool] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> Iterator[StreamChunk]:
        """Streaming is not cached and not retried mid-stream."""

        msgs = normalize_messages(messages)
        merged_extra = self._provider._merged_params(extra_params)

        ctx = RequestContext(
            provider=self.provider, model=self.model, messages=msgs, extra_params=merged_extra
        )
        _safe_invoke(self.middleware, "on_request", ctx)
        timer = _Timer()

        last_chunk: StreamChunk | None = None
        try:
            for chunk in self._provider.stream(msgs, tools=tools, extra_params=extra_params):
                last_chunk = chunk
                yield chunk
        except BaseException as exc:
            _safe_invoke(
                self.middleware,
                "on_error",
                ErrorContext(
                    provider=self.provider,
                    model=self.model,
                    messages=msgs,
                    extra_params=merged_extra,
                    error=exc,
                    duration_ms=timer.ms(),
                ),
            )
            raise

        usage = last_chunk.usage if last_chunk is not None else None
        if usage is None:
            usage = Usage()
        if usage.cost_usd is None:
            usage.cost_usd = compute_cost(self.provider, self.model, usage, self.extra_pricing)
        response = ChatResponse(
            text="",
            tool_calls=[],
            finish_reason=(last_chunk.finish_reason if last_chunk else None) or "stop",
            model=self.model,
            usage=usage,
            raw=last_chunk.raw if last_chunk else None,
        )
        _safe_invoke(
            self.middleware,
            "on_response",
            ResponseContext(
                provider=self.provider,
                model=self.model,
                messages=msgs,
                extra_params=merged_extra,
                response=response,
                duration_ms=timer.ms(),
            ),
        )


class AsyncAIClient:
    """Async counterpart of :class:`AIClient`."""

    def __init__(
        self,
        *,
        provider: str,
        api_key: str | None = None,
        model: str,
        timeout: float = 60.0,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        cache: CacheBackend | None = None,
        cache_ttl: int | None = None,
        middleware: list[Middleware] | None = None,
        base_url: str | None = None,
        extra_params: dict[str, Any] | None = None,
        extra_pricing: dict[str, dict[str, dict[str, float]]] | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.cache = cache
        self.cache_ttl = cache_ttl
        self.middleware: list[Middleware] = list(middleware or [])
        self.extra_pricing = extra_pricing

        cls = get_async_provider_class(provider)
        self._provider: BaseAsyncProvider = cls(
            api_key=api_key,
            model=model,
            timeout=timeout,
            base_url=base_url,
            extra_params=extra_params,
        )

    async def chat(
        self,
        messages: list[Message] | list[dict[str, Any]],
        *,
        tools: list[Tool] | None = None,
        extra_params: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> ChatResponse:
        msgs = normalize_messages(messages)
        merged_extra = self._provider._merged_params(extra_params)

        ctx = RequestContext(
            provider=self.provider, model=self.model, messages=msgs, extra_params=merged_extra
        )
        _safe_invoke(self.middleware, "on_request", ctx)
        timer = _Timer()

        if self.cache is not None and use_cache:
            key = make_cache_key(
                provider=self.provider,
                model=self.model,
                messages=msgs,
                tools=tools,
                extra_params=merged_extra,
            )
            cached = self.cache.get(key)
            if cached is not None:
                _safe_invoke(
                    self.middleware,
                    "on_response",
                    ResponseContext(
                        provider=self.provider,
                        model=self.model,
                        messages=msgs,
                        extra_params=merged_extra,
                        response=cached,
                        duration_ms=timer.ms(),
                    ),
                )
                return cached

        try:
            response = await acall_with_retry(
                lambda: self._provider.chat(msgs, tools=tools, extra_params=extra_params),
                max_retries=self.max_retries,
                initial_backoff=self.initial_backoff,
                max_backoff=self.max_backoff,
            )
        except BaseException as exc:
            _safe_invoke(
                self.middleware,
                "on_error",
                ErrorContext(
                    provider=self.provider,
                    model=self.model,
                    messages=msgs,
                    extra_params=merged_extra,
                    error=exc,
                    duration_ms=timer.ms(),
                ),
            )
            raise

        response = _apply_cost(response, self.provider, self.model, self.extra_pricing)

        if self.cache is not None and use_cache:
            self.cache.set(key, response, ttl=self.cache_ttl)

        _safe_invoke(
            self.middleware,
            "on_response",
            ResponseContext(
                provider=self.provider,
                model=self.model,
                messages=msgs,
                extra_params=merged_extra,
                response=response,
                duration_ms=timer.ms(),
            ),
        )
        return response

    async def stream(
        self,
        messages: list[Message] | list[dict[str, Any]],
        *,
        tools: list[Tool] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        msgs = normalize_messages(messages)
        merged_extra = self._provider._merged_params(extra_params)

        ctx = RequestContext(
            provider=self.provider, model=self.model, messages=msgs, extra_params=merged_extra
        )
        _safe_invoke(self.middleware, "on_request", ctx)
        timer = _Timer()

        last_chunk: StreamChunk | None = None
        try:
            async for chunk in self._provider.stream(
                msgs, tools=tools, extra_params=extra_params
            ):
                last_chunk = chunk
                yield chunk
        except BaseException as exc:
            _safe_invoke(
                self.middleware,
                "on_error",
                ErrorContext(
                    provider=self.provider,
                    model=self.model,
                    messages=msgs,
                    extra_params=merged_extra,
                    error=exc,
                    duration_ms=timer.ms(),
                ),
            )
            raise

        usage = last_chunk.usage if last_chunk is not None else None
        if usage is None:
            usage = Usage()
        if usage.cost_usd is None:
            usage.cost_usd = compute_cost(self.provider, self.model, usage, self.extra_pricing)
        response = ChatResponse(
            text="",
            tool_calls=[],
            finish_reason=(last_chunk.finish_reason if last_chunk else None) or "stop",
            model=self.model,
            usage=usage,
            raw=last_chunk.raw if last_chunk else None,
        )
        _safe_invoke(
            self.middleware,
            "on_response",
            ResponseContext(
                provider=self.provider,
                model=self.model,
                messages=msgs,
                extra_params=merged_extra,
                response=response,
                duration_ms=timer.ms(),
            ),
        )


__all__ = ["AIClient", "AsyncAIClient"]

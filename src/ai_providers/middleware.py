"""Middleware Protocol and built-in implementations."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .types import ChatResponse, Message


@dataclass
class RequestContext:
    provider: str
    model: str
    messages: list[Message]
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseContext:
    provider: str
    model: str
    messages: list[Message]
    extra_params: dict[str, Any]
    response: ChatResponse
    duration_ms: float


@dataclass
class ErrorContext:
    provider: str
    model: str
    messages: list[Message]
    extra_params: dict[str, Any]
    error: BaseException
    duration_ms: float


@runtime_checkable
class Middleware(Protocol):
    """Hooks invoked by AIClient around each call."""

    def on_request(self, ctx: RequestContext) -> None: ...
    def on_response(self, ctx: ResponseContext) -> None: ...
    def on_error(self, ctx: ErrorContext) -> None: ...


class _NoopMixin:
    def on_request(self, ctx: RequestContext) -> None:  # pragma: no cover - default
        return None

    def on_response(self, ctx: ResponseContext) -> None:  # pragma: no cover - default
        return None

    def on_error(self, ctx: ErrorContext) -> None:  # pragma: no cover - default
        return None


class LoggingMiddleware(_NoopMixin):
    """Logs each call via stdlib ``logging``."""

    def __init__(self, logger: logging.Logger | None = None, level: int = logging.INFO) -> None:
        self.logger = logger or logging.getLogger("ai_providers")
        self.level = level

    def on_request(self, ctx: RequestContext) -> None:
        self.logger.log(
            self.level,
            "ai_providers request provider=%s model=%s messages=%d",
            ctx.provider,
            ctx.model,
            len(ctx.messages),
        )

    def on_response(self, ctx: ResponseContext) -> None:
        self.logger.log(
            self.level,
            "ai_providers response provider=%s model=%s tokens=%d cost_usd=%s duration_ms=%.1f",
            ctx.provider,
            ctx.model,
            ctx.response.usage.total_tokens,
            ctx.response.usage.cost_usd,
            ctx.duration_ms,
        )

    def on_error(self, ctx: ErrorContext) -> None:
        self.logger.warning(
            "ai_providers error provider=%s model=%s error=%s duration_ms=%.1f",
            ctx.provider,
            ctx.model,
            ctx.error,
            ctx.duration_ms,
        )


class MetricsMiddleware(_NoopMixin):
    """In-memory counters/durations for quick instrumentation or tests."""

    def __init__(self) -> None:
        self.requests: int = 0
        self.responses: int = 0
        self.errors: int = 0
        self.total_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self.durations_ms: list[float] = []
        self.errors_by_type: dict[str, int] = defaultdict(int)

    def on_request(self, ctx: RequestContext) -> None:
        self.requests += 1

    def on_response(self, ctx: ResponseContext) -> None:
        self.responses += 1
        self.total_tokens += ctx.response.usage.total_tokens
        if ctx.response.usage.cost_usd is not None:
            self.total_cost_usd += ctx.response.usage.cost_usd
        self.durations_ms.append(ctx.duration_ms)

    def on_error(self, ctx: ErrorContext) -> None:
        self.errors += 1
        self.errors_by_type[type(ctx.error).__name__] += 1
        self.durations_ms.append(ctx.duration_ms)


def _safe_invoke(
    middleware: list[Middleware],
    method: str,
    ctx: Any,
    logger: logging.Logger | None = None,
) -> None:
    """Call ``method`` on each middleware, swallowing per-middleware failures."""

    log = logger or logging.getLogger("ai_providers")
    for mw in middleware:
        fn = getattr(mw, method, None)
        if fn is None:
            continue
        try:
            fn(ctx)
        except Exception:
            log.exception("middleware %s.%s failed", type(mw).__name__, method)


class _Timer:
    """Tiny helper used by the client to measure call duration."""

    def __init__(self) -> None:
        self.start = time.perf_counter()

    def ms(self) -> float:
        return (time.perf_counter() - self.start) * 1000.0


__all__ = [
    "ErrorContext",
    "LoggingMiddleware",
    "MetricsMiddleware",
    "Middleware",
    "RequestContext",
    "ResponseContext",
]

"""Retry helpers built on top of tenacity."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from tenacity import (
    AsyncRetrying,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random_exponential,
)

from .exceptions import (
    AIProviderError,
    ProviderAPIError,
    RateLimitError,
)
from .exceptions import (
    TimeoutError as AIPTimeoutError,
)

T = TypeVar("T")


def _should_retry(exc: BaseException) -> bool:
    """Decide whether an exception is retryable."""

    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, AIPTimeoutError):
        return True
    if isinstance(exc, ProviderAPIError):
        # Retry only on transient server errors.
        return exc.status_code is None or exc.status_code >= 500
    if isinstance(exc, AIProviderError):
        return False
    # Network-layer exceptions: retry conservatively.
    return isinstance(exc, ConnectionError)


def _wait_strategy(initial: float, maximum: float):
    return wait_random_exponential(multiplier=initial, max=maximum) | wait_exponential(
        multiplier=initial, max=maximum
    )


def call_with_retry(
    func: Callable[[], T],
    *,
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    max_backoff: float = 30.0,
) -> T:
    """Run a sync callable with exponential-backoff retries."""

    if max_retries <= 0:
        return func()

    retryer = Retrying(
        stop=stop_after_attempt(max_retries + 1),
        wait=wait_exponential(multiplier=initial_backoff, max=max_backoff),
        retry=retry_if_exception(_should_retry),
        reraise=True,
    )
    for attempt in retryer:
        with attempt:
            return func()
    raise RuntimeError("unreachable")  # pragma: no cover


async def acall_with_retry(
    func: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    max_backoff: float = 30.0,
) -> T:
    """Run an async callable with exponential-backoff retries."""

    if max_retries <= 0:
        return await func()

    retryer = AsyncRetrying(
        stop=stop_after_attempt(max_retries + 1),
        wait=wait_exponential(multiplier=initial_backoff, max=max_backoff),
        retry=retry_if_exception(_should_retry),
        reraise=True,
    )
    async for attempt in retryer:
        with attempt:
            return await func()
    raise RuntimeError("unreachable")  # pragma: no cover


__all__ = ["acall_with_retry", "call_with_retry"]

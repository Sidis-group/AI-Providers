"""Unified exception hierarchy for ai_providers."""

from __future__ import annotations


class AIProviderError(Exception):
    """Base class for all ai_providers errors."""


class AuthenticationError(AIProviderError):
    """Raised when API credentials are invalid or missing."""


class RateLimitError(AIProviderError):
    """Raised when the provider returns a rate-limit response (HTTP 429)."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class InvalidRequestError(AIProviderError):
    """Raised when the request is malformed or rejected by the provider (HTTP 4xx)."""


class ContextLengthError(InvalidRequestError):
    """Raised when the request exceeds the model's context window."""


class ProviderAPIError(AIProviderError):
    """Raised for provider-side server errors (HTTP 5xx) or unmapped API failures."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ProviderTimeoutError(AIProviderError):
    """Raised when the provider call exceeds the configured timeout.

    .. note::
       Also exported as ``TimeoutError`` for plan compatibility, but that
       shadows :class:`builtins.TimeoutError` if used via ``from ... import *``.
       Prefer ``ProviderTimeoutError`` in new code.
    """


# Backwards-compatible alias. Importing this from ``ai_providers`` shadows
# the builtin ``TimeoutError``; if that's a concern, use the long name.
TimeoutError = ProviderTimeoutError


class CacheError(AIProviderError):
    """Raised when a cache backend fails."""


class VectorStoreError(AIProviderError):
    """Raised when a vector store backend fails."""


class NotSupportedError(AIProviderError):
    """Raised when a feature isn't supported by the current backend (e.g. hybrid search)."""


class ProviderNotInstalledError(AIProviderError, ImportError):
    """Raised when the SDK for the requested provider/backend isn't installed.

    Inherits from both ``AIProviderError`` (so ``except AIProviderError``
    catches it like any other library error) and ``ImportError`` (so legacy
    code that catches missing-module errors still works).
    """

    def __init__(self, provider: str, extra: str) -> None:
        msg = (
            f"Provider/backend '{provider}' is not installed. "
            f"Install with: pip install 'ai-providers[{extra}]'"
        )
        super().__init__(msg)
        self.provider = provider
        self.extra = extra

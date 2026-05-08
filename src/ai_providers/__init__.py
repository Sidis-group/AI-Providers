"""ai_providers — unified Python interface for AI providers.

Public API:

* :class:`AIClient` / :class:`AsyncAIClient` — chat / streaming / tool calls.
* :class:`EmbeddingsClient` / :class:`AsyncEmbeddingsClient` — embeddings.
* :class:`VectorStore` / :class:`AsyncVectorStore` — vector store facade.
* :class:`InMemoryCache` — built-in cache (other backends via extras).
* :class:`LoggingMiddleware`, :class:`MetricsMiddleware` — built-in middleware.
* :func:`register_provider` — register custom providers.
* Exception hierarchy under :class:`AIProviderError`.
* Unified types: :class:`Message`, :class:`ChatResponse`, :class:`StreamChunk`,
  :class:`Tool`, :class:`ToolCall`, :class:`ContentPart`, :class:`Usage`,
  :class:`Document`, :class:`QueryResult`.
"""

from __future__ import annotations

from .cache import InMemoryCache
from .cache.base import CacheBackend, make_cache_key
from .client import AIClient, AsyncAIClient
from .embeddings import AsyncEmbeddingsClient, EmbeddingsClient
from .exceptions import (
    AIProviderError,
    AuthenticationError,
    CacheError,
    ContextLengthError,
    InvalidRequestError,
    NotSupportedError,
    ProviderAPIError,
    ProviderNotInstalledError,
    ProviderTimeoutError,
    RateLimitError,
    TimeoutError,
    VectorStoreError,
)
from .middleware import (
    ErrorContext,
    LoggingMiddleware,
    MetricsMiddleware,
    Middleware,
    RequestContext,
    ResponseContext,
)
from .pricing import PRICING, compute_cost, lookup_price
from .registry import (
    get_async_provider_class,
    get_provider_class,
    known_providers,
    register_provider,
)
from .types import (
    ChatResponse,
    ContentPart,
    Document,
    EmbeddingResponse,
    FinishReason,
    Message,
    QueryResult,
    Role,
    SearchMode,
    StreamChunk,
    Tool,
    ToolCall,
    Usage,
)
from .vector_stores import (
    AsyncVectorStore,
    AsyncVectorStoreBackend,
    VectorStore,
    VectorStoreBackend,
)

__version__ = "0.1.1"

__all__ = [  # noqa: RUF022 - grouped by category for readability
    "__version__",
    # Clients
    "AIClient",
    "AsyncAIClient",
    "EmbeddingsClient",
    "AsyncEmbeddingsClient",
    # Vector stores
    "VectorStore",
    "AsyncVectorStore",
    "VectorStoreBackend",
    "AsyncVectorStoreBackend",
    # Cache
    "InMemoryCache",
    "CacheBackend",
    "make_cache_key",
    # Middleware
    "Middleware",
    "LoggingMiddleware",
    "MetricsMiddleware",
    "RequestContext",
    "ResponseContext",
    "ErrorContext",
    # Registry
    "register_provider",
    "get_provider_class",
    "get_async_provider_class",
    "known_providers",
    # Pricing
    "PRICING",
    "compute_cost",
    "lookup_price",
    # Types
    "Message",
    "ChatResponse",
    "StreamChunk",
    "Tool",
    "ToolCall",
    "ContentPart",
    "Usage",
    "Document",
    "QueryResult",
    "EmbeddingResponse",
    "Role",
    "FinishReason",
    "SearchMode",
    # Exceptions
    "AIProviderError",
    "AuthenticationError",
    "RateLimitError",
    "InvalidRequestError",
    "ContextLengthError",
    "ProviderAPIError",
    "TimeoutError",
    "ProviderTimeoutError",
    "CacheError",
    "VectorStoreError",
    "NotSupportedError",
    "ProviderNotInstalledError",
]


def __getattr__(name: str):
    """Expose optional vector store classes lazily."""

    if name == "ChromaStore":
        from .vector_stores.chroma import ChromaStore

        return ChromaStore
    if name == "QdrantStore":
        from .vector_stores.qdrant import QdrantStore

        return QdrantStore
    if name == "DiskCache":
        from .cache.disk import DiskCache

        return DiskCache
    raise AttributeError(name)

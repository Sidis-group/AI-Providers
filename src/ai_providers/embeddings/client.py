"""Embeddings facade similar to AIClient."""

from __future__ import annotations

from typing import Any

from ..pricing import compute_cost
from ..types import EmbeddingResponse


class EmbeddingsClient:
    """Synchronous embeddings facade."""

    def __init__(
        self,
        *,
        provider: str,
        api_key: str | None = None,
        model: str,
        timeout: float = 60.0,
        base_url: str | None = None,
        extra_params: dict[str, Any] | None = None,
        extra_pricing: dict[str, dict[str, dict[str, float]]] | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.extra_pricing = extra_pricing
        self._provider = _build_sync_provider(
            provider=provider,
            api_key=api_key,
            model=model,
            timeout=timeout,
            base_url=base_url,
            extra_params=extra_params,
        )

    def embed(self, texts: list[str]) -> EmbeddingResponse:
        response = self._provider.embed(texts)
        if response.usage.cost_usd is None:
            response.usage.cost_usd = compute_cost(
                self.provider, self.model, response.usage, self.extra_pricing
            )
        return response


class AsyncEmbeddingsClient:
    """Asynchronous embeddings facade."""

    def __init__(
        self,
        *,
        provider: str,
        api_key: str | None = None,
        model: str,
        timeout: float = 60.0,
        base_url: str | None = None,
        extra_params: dict[str, Any] | None = None,
        extra_pricing: dict[str, dict[str, dict[str, float]]] | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.extra_pricing = extra_pricing
        self._provider = _build_async_provider(
            provider=provider,
            api_key=api_key,
            model=model,
            timeout=timeout,
            base_url=base_url,
            extra_params=extra_params,
        )

    async def embed(self, texts: list[str]) -> EmbeddingResponse:
        response = await self._provider.embed(texts)
        if response.usage.cost_usd is None:
            response.usage.cost_usd = compute_cost(
                self.provider, self.model, response.usage, self.extra_pricing
            )
        return response


def _build_sync_provider(
    *,
    provider: str,
    api_key: str | None,
    model: str,
    timeout: float,
    base_url: str | None,
    extra_params: dict[str, Any] | None,
) -> Any:
    if provider == "openai":
        from .openai import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(
            api_key=api_key,
            model=model,
            timeout=timeout,
            base_url=base_url,
            extra_params=extra_params,
        )
    if provider == "anthropic":
        raise NotImplementedError(
            "Anthropic does not provide native embeddings. "
            "Use 'openai' or a Voyage AI integration."
        )
    raise ValueError(f"Unknown embeddings provider: {provider!r}")


def _build_async_provider(
    *,
    provider: str,
    api_key: str | None,
    model: str,
    timeout: float,
    base_url: str | None,
    extra_params: dict[str, Any] | None,
) -> Any:
    if provider == "openai":
        from .openai import AsyncOpenAIEmbeddingProvider

        return AsyncOpenAIEmbeddingProvider(
            api_key=api_key,
            model=model,
            timeout=timeout,
            base_url=base_url,
            extra_params=extra_params,
        )
    if provider == "anthropic":
        raise NotImplementedError(
            "Anthropic does not provide native embeddings. "
            "Use 'openai' or a Voyage AI integration."
        )
    raise ValueError(f"Unknown embeddings provider: {provider!r}")


__all__ = ["AsyncEmbeddingsClient", "EmbeddingsClient"]

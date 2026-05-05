"""OpenAI embeddings."""

from __future__ import annotations

from typing import Any

from ..exceptions import ProviderNotInstalledError
from ..types import EmbeddingResponse, Usage
from .base import BaseAsyncEmbeddingProvider, BaseEmbeddingProvider

try:
    from openai import AsyncOpenAI, OpenAI  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]
    AsyncOpenAI = None  # type: ignore[assignment]


def _to_response(api_response: Any, model: str) -> EmbeddingResponse:
    vectors = [list(d.embedding) for d in api_response.data]
    raw_usage = getattr(api_response, "usage", None)
    usage = Usage()
    if raw_usage is not None:
        prompt = getattr(raw_usage, "prompt_tokens", 0) or 0
        total = getattr(raw_usage, "total_tokens", prompt) or prompt
        usage = Usage(prompt_tokens=prompt, completion_tokens=0, total_tokens=total)
    return EmbeddingResponse(
        vectors=vectors,
        model=getattr(api_response, "model", model),
        usage=usage,
        raw=api_response,
    )


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    name = "openai"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if OpenAI is None:
            raise ProviderNotInstalledError("openai", "openai")
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def embed(self, texts: list[str]) -> EmbeddingResponse:
        api = self._client.embeddings.create(model=self.model, input=texts, **self.extra_params)
        return _to_response(api, self.model)


class AsyncOpenAIEmbeddingProvider(BaseAsyncEmbeddingProvider):
    name = "openai"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if AsyncOpenAI is None:
            raise ProviderNotInstalledError("openai", "openai")
        self._client = AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )

    async def embed(self, texts: list[str]) -> EmbeddingResponse:
        api = await self._client.embeddings.create(
            model=self.model, input=texts, **self.extra_params
        )
        return _to_response(api, self.model)

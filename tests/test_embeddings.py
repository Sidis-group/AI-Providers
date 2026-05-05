"""Embeddings tests with mocks."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ai_providers import EmbeddingsClient


def _fake_embed_response(vectors: list[list[float]], prompt_tokens: int = 5):
    data = [SimpleNamespace(embedding=v) for v in vectors]
    return SimpleNamespace(
        data=data,
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens),
        model="text-embedding-3-small",
    )


def test_openai_embeddings_returns_vectors(monkeypatch):
    mock_client = MagicMock()
    mock_client.embeddings.create = MagicMock(
        return_value=_fake_embed_response([[1.0, 2.0], [3.0, 4.0]])
    )
    monkeypatch.setattr("ai_providers.embeddings.openai.OpenAI", lambda **kw: mock_client)

    client = EmbeddingsClient(provider="openai", api_key="sk", model="text-embedding-3-small")
    response = client.embed(["a", "b"])
    assert response.vectors == [[1.0, 2.0], [3.0, 4.0]]
    assert response.usage.prompt_tokens == 5


def test_anthropic_embeddings_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        EmbeddingsClient(provider="anthropic", api_key="x", model="anything")

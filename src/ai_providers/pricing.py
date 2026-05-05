"""Token pricing tables and cost calculation."""

from __future__ import annotations

from .types import Usage

# USD per 1M tokens. Update as providers change pricing.
PRICING: dict[str, dict[str, dict[str, float]]] = {
    "openai": {
        # Chat
        "gpt-4o": {"input": 2.5, "output": 10.0},
        "gpt-4o-2024-11-20": {"input": 2.5, "output": 10.0},
        "gpt-4o-2024-08-06": {"input": 2.5, "output": 10.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},
        "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.6},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
        "o1": {"input": 15.0, "output": 60.0},
        "o1-mini": {"input": 3.0, "output": 12.0},
        "o3-mini": {"input": 1.1, "output": 4.4},
        # Embeddings (output column is unused; kept for uniform schema)
        "text-embedding-3-small": {"input": 0.02, "output": 0.0},
        "text-embedding-3-large": {"input": 0.13, "output": 0.0},
        "text-embedding-ada-002": {"input": 0.1, "output": 0.0},
    },
    "anthropic": {
        "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
        "claude-3-5-sonnet-latest": {"input": 3.0, "output": 15.0},
        "claude-3-5-haiku-20241022": {"input": 0.8, "output": 4.0},
        "claude-3-5-haiku-latest": {"input": 0.8, "output": 4.0},
        "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
        "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    },
}


def lookup_price(
    provider: str,
    model: str,
    extra_pricing: dict[str, dict[str, dict[str, float]]] | None = None,
) -> dict[str, float] | None:
    """Look up input/output USD-per-1M-token prices for a model."""

    if extra_pricing and model in extra_pricing.get(provider, {}):
        return extra_pricing[provider][model]
    return PRICING.get(provider, {}).get(model)


def compute_cost(
    provider: str,
    model: str,
    usage: Usage,
    extra_pricing: dict[str, dict[str, dict[str, float]]] | None = None,
) -> float | None:
    """Compute USD cost for a usage record. Returns ``None`` if model is unknown."""

    price = lookup_price(provider, model, extra_pricing)
    if price is None:
        return None
    input_cost = (usage.prompt_tokens / 1_000_000.0) * price.get("input", 0.0)
    output_cost = (usage.completion_tokens / 1_000_000.0) * price.get("output", 0.0)
    return round(input_cost + output_cost, 8)


__all__ = ["PRICING", "compute_cost", "lookup_price"]

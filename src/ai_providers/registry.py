"""Provider registry for plugin-style extensibility."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .exceptions import ProviderNotInstalledError

if TYPE_CHECKING:
    from .base import BaseAsyncProvider, BaseProvider

# Lazy loaders so importing ai_providers does not require any provider SDK to be installed.
_SYNC_LOADERS: dict[str, tuple[str, str, str]] = {
    # name -> (module path, class name, extra to install)
    "openai": ("ai_providers.providers.openai", "OpenAIProvider", "openai"),
    "anthropic": ("ai_providers.providers.anthropic", "AnthropicProvider", "anthropic"),
}
_ASYNC_LOADERS: dict[str, tuple[str, str, str]] = {
    "openai": ("ai_providers.providers.openai", "AsyncOpenAIProvider", "openai"),
    "anthropic": ("ai_providers.providers.anthropic", "AsyncAnthropicProvider", "anthropic"),
}

_SYNC_OVERRIDES: dict[str, type[BaseProvider]] = {}
_ASYNC_OVERRIDES: dict[str, type[BaseAsyncProvider]] = {}


def register_provider(
    name: str,
    sync_cls: type[BaseProvider] | None = None,
    async_cls: type[BaseAsyncProvider] | None = None,
) -> None:
    """Register a custom provider implementation under ``name``."""

    if sync_cls is not None:
        _SYNC_OVERRIDES[name] = sync_cls
    if async_cls is not None:
        _ASYNC_OVERRIDES[name] = async_cls


def _load(name: str, table: dict[str, tuple[str, str, str]]) -> type:
    if name not in table:
        raise ValueError(f"Unknown provider: {name!r}. Known: {sorted(table)}")
    module_path, cls_name, extra = table[name]
    try:
        module = __import__(module_path, fromlist=[cls_name])
    except ImportError as exc:
        # The SDK for this provider isn't installed.
        raise ProviderNotInstalledError(name, extra) from exc
    return getattr(module, cls_name)


def get_provider_class(name: str) -> type[BaseProvider]:
    if name in _SYNC_OVERRIDES:
        return _SYNC_OVERRIDES[name]
    return _load(name, _SYNC_LOADERS)


def get_async_provider_class(name: str) -> type[BaseAsyncProvider]:
    if name in _ASYNC_OVERRIDES:
        return _ASYNC_OVERRIDES[name]
    return _load(name, _ASYNC_LOADERS)


def known_providers() -> list[str]:
    return sorted(set(_SYNC_LOADERS) | set(_SYNC_OVERRIDES))

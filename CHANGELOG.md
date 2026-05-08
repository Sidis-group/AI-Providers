# Changelog

Усі помітні зміни в цьому проєкті документуються тут.

Формат базується на [Keep a Changelog](https://keepachangelog.com/uk/1.1.0/),
проєкт використовує [Semantic Versioning](https://semver.org/lang/uk/).

## [0.1.1] — 2026-05-05

### Fixed
- **QdrantStore**: keyword/hybrid search більше не ламається між процесами.
  Замінили `hash()` (рандомізований per-process через `PYTHONHASHSEED`) на
  детерміністичний `blake2b`. Раніше токени мапились на різні sparse-vector
  індекси у процесі upsert і процесі query, через що пошук тихо повертав
  порожні результати.
- **ChromaStore**: користувацький ключ метаданих `_id` більше не вилучається
  при query. Sentinel переіменовано з `_id` на namespaced
  `__ai_providers_metadata_sentinel__`.
- **`ProviderNotInstalledError`**: тепер наслідує і `AIProviderError`, і
  `ImportError`, тож ловиться через `except AIProviderError` як і всі інші
  винятки бібліотеки.
- **Streaming `on_response`**: завжди емітується наприкінці потоку (sync і
  async), навіть якщо провайдер не дав `usage` у фінальному chunk'у. Раніше
  middleware могла бачити `on_request` без `on_response`.

### Added
- `ProviderTimeoutError` як основна назва для таймаута. `TimeoutError`
  залишається як backward-compatible alias з docstring-попередженням про
  shadowing builtin.
- Тести: streaming on_response always-emit, Chroma mixed-metadata, Chroma
  user `_id` metadata key. Загалом 74 тести (було 70).

### Removed
- Невикористаний helper `_wait_strategy` у `retries.py`.

## [0.1.0] — 2026-05-05

### Added
- Перший реліз `ai-providers`.
- `AIClient` / `AsyncAIClient` — фасад над OpenAI + Anthropic.
- Streaming, tool calls, multimodal (image + audio), embeddings.
- Vector stores: Chroma (semantic + keyword) і Qdrant (semantic + keyword
  + hybrid через RRF / weighted fusion).
- Pluggable cache (in-memory, disk через extra), middleware
  (LoggingMiddleware, MetricsMiddleware), retries з exponential backoff.
- Уніфікована ієрархія винятків з мапінгом з SDK.
- Pricing-таблиця для 21 моделі + custom через `extra_pricing`.
- Provider registry для plugin-style розширення.
- 70 pytest-тестів.

[0.1.1]: https://github.com/Sidis-group/AI-Providers/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Sidis-group/AI-Providers/releases/tag/v0.1.0

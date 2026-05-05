# Adding a New Provider

Бібліотека підтримує plugin-style розширення. Зареєструйте свого провайдера —
далі він поводиться як `"openai"` чи `"anthropic"`.

## Крок 1. Реалізуйте `BaseProvider` (sync)

```python
from collections.abc import Iterator
from ai_providers.base import BaseProvider
from ai_providers.types import ChatResponse, Message, StreamChunk, Tool, Usage

class MyProvider(BaseProvider):
    name = "my-llm"

    def chat(self, messages, *, tools=None, extra_params=None):
        params = self._merged_params(extra_params)
        # ... виконати запит до вашого API ...
        text = "answer"
        return ChatResponse(
            text=text,
            tool_calls=[],
            finish_reason="stop",
            model=self.model,
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            raw=None,
        )

    def stream(self, messages, *, tools=None, extra_params=None) -> Iterator[StreamChunk]:
        for token in ["he", "llo"]:
            yield StreamChunk(delta=token)
        yield StreamChunk(
            delta="",
            finish_reason="stop",
            usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        )
```

## Крок 2. Async (опційно)

```python
from collections.abc import AsyncIterator
from ai_providers.base import BaseAsyncProvider

class AsyncMyProvider(BaseAsyncProvider):
    name = "my-llm"

    async def chat(self, messages, *, tools=None, extra_params=None):
        ...

    async def stream(self, messages, *, tools=None, extra_params=None) -> AsyncIterator[StreamChunk]:
        ...
```

## Крок 3. Зареєструвати

```python
from ai_providers import register_provider

register_provider("my-llm", sync_cls=MyProvider, async_cls=AsyncMyProvider)
```

Реєстрація — process-local. Зробіть це в `__init__.py` свого пакета, щоб
відбувалося одразу при імпорті.

## Крок 4. Використання

```python
from ai_providers import AIClient

client = AIClient(provider="my-llm", api_key="...", model="my-model-v1")
print(client.chat([{"role": "user", "content": "hi"}]).text)
```

## Поради

- **Маппінг винятків.** Усі провайдер-специфічні помилки конвертуйте у винятки
  з `ai_providers.exceptions` (`AuthenticationError`, `RateLimitError`, ...).
  Це робить код користувачів переносним.

- **Multimodal.** Підтримайте `ContentPart(type="text"|"image"|"audio")` де
  можливо. Якщо ваш API не підтримує — кидайте `ValueError` з ясним
  повідомленням.

- **Tool calls.** Якщо ваш API має function calling — конвертуйте `Tool` у
  ваш формат і парсіть назад у `ToolCall`.

- **Pricing.** Додайте записи у `ai_providers.PRICING` (через PR) або
  передайте `extra_pricing` користувачам:
  ```python
  AIClient(provider="my-llm", ..., extra_pricing={"my-llm": {"my-model-v1": {"input": 1.0, "output": 2.0}}})
  ```

- **OpenAI-compatible API.** Якщо ваш сервіс емулює OpenAI Chat API
  (DeepSeek, OpenRouter, локальний vLLM, тощо) — простіше використати
  `provider="openai"` з кастомним `base_url`:
  ```python
  AIClient(provider="openai", base_url="https://api.deepseek.com/v1", api_key="...", model="deepseek-chat")
  ```
  Це не вимагає писати свого провайдера.

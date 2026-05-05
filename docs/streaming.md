# Streaming

```python
for chunk in client.stream([{"role": "user", "content": "розкажи історію"}]):
    print(chunk.delta, end="", flush=True)
```

## Структура `StreamChunk`

```python
chunk.delta              # incremental text (str, "" якщо у цьому чанку немає тексту)
chunk.tool_call_delta    # ToolCall або None — приходить у момент завершення tool_use блоку
chunk.finish_reason      # "stop" | "length" | "tool_calls" | None
chunk.usage              # Usage у фінальному чанку
chunk.raw                # сирий event від SDK
```

## Async streaming

```python
async for chunk in async_client.stream([...]):
    print(chunk.delta, end="")
```

## Tool calls у streaming

Для OpenAI tool_call_delta може приходити частинами; модуль агрегує JSON-аргументи
і повертає завершений `ToolCall` у тому чанку, де блок tool_use закінчується.
Для Anthropic — той самий патерн через `content_block_delta`/`content_block_stop`.

```python
collected_tool_calls: list[ToolCall] = []
text = ""
for chunk in client.stream(messages, tools=[Tool(...)]):
    text += chunk.delta
    if chunk.tool_call_delta is not None:
        collected_tool_calls.append(chunk.tool_call_delta)
```

## Обмеження

- **Streaming не кешується** — кеш потрапляє лише на нестрімінгові `chat`.
- Retries не перезапускаються посеред стріму. Якщо стрім обірвався — кидається помилка.
- Останній чанк зазвичай містить `usage` і `finish_reason`. Для OpenAI — увімкнено `stream_options={"include_usage": True}` за дефолтом.

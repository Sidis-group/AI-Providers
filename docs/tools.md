# Tool / Function Calling

Уніфікований інтерфейс — той самий код працює для OpenAI і Anthropic.

## Визначення інструмента

```python
from ai_providers import Tool

get_weather = Tool(
    name="get_weather",
    description="Return current temperature in celsius for a city",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
        },
        "required": ["city"],
    },
)
```

`parameters` — JSON Schema. Це той самий формат, який очікують OpenAI і
Anthropic нативно (модуль конвертує їх під капотом).

## Виклик з інструментами

```python
response = client.chat(
    [{"role": "user", "content": "What's the weather in Kyiv?"}],
    tools=[get_weather],
)

if response.finish_reason == "tool_calls":
    for call in response.tool_calls:
        print(call.id, call.name, call.arguments)  # arguments — це dict
```

## Передача результату назад моделі

Після виконання інструменту відповідайте моделі додатковим повідомленням
з `role="tool"`:

```python
from ai_providers import Message

# Перший виклик: модель просить tool call
first = client.chat(messages, tools=[get_weather])

# Виконуємо інструмент локально
results = []
for call in first.tool_calls:
    if call.name == "get_weather":
        result = my_weather_api(**call.arguments)
        results.append(Message(
            role="tool",
            content=str(result),
            tool_call_id=call.id,
        ))

# Другий виклик: даємо моделі результати
final = client.chat(
    messages + [
        Message(
            role="assistant",
            content=first.text,
            tool_calls=first.tool_calls,
        )
    ] + results,
    tools=[get_weather],
)
print(final.text)
```

Модуль автоматично перетворює це у правильний формат для кожного провайдера:

- OpenAI: `{"role": "tool", "tool_call_id": "...", "content": "..."}`.
- Anthropic: `{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "...", "content": "..."}]}`.

## Streaming + tools

```python
collected_calls = []
for chunk in client.stream(messages, tools=[get_weather]):
    if chunk.delta:
        print(chunk.delta, end="")
    if chunk.tool_call_delta is not None:
        collected_calls.append(chunk.tool_call_delta)
```

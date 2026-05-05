# Middleware

Хуки навколо кожного виклику — логування, метрики, трейсинг, rate limiting тощо.

## Інтерфейс

```python
class Middleware(Protocol):
    def on_request(self, ctx: RequestContext) -> None: ...
    def on_response(self, ctx: ResponseContext) -> None: ...
    def on_error(self, ctx: ErrorContext) -> None: ...
```

Всі три методи опційні — модуль не падає якщо middleware їх не реалізовує (`Protocol` дозволяє часткову реалізацію через `getattr` та safe-invoke).

## Контексти

### `RequestContext`
```
provider: str
model: str
messages: list[Message]
extra_params: dict
```

### `ResponseContext` (extends RequestContext fields)
```
response: ChatResponse
duration_ms: float
```

### `ErrorContext` (extends RequestContext fields)
```
error: BaseException
duration_ms: float
```

## Built-in: `LoggingMiddleware`

```python
import logging
from ai_providers import LoggingMiddleware

logging.basicConfig(level=logging.INFO)
client = AIClient(..., middleware=[LoggingMiddleware()])
```

Логує: `provider`, `model`, кількість повідомлень, токени, cost, duration. Помилки — `WARNING`.

## Built-in: `MetricsMiddleware`

```python
from ai_providers import MetricsMiddleware

metrics = MetricsMiddleware()
client = AIClient(..., middleware=[metrics])

# після кількох викликів:
print(metrics.requests, metrics.responses, metrics.errors)
print(metrics.total_tokens, metrics.total_cost_usd)
print(metrics.errors_by_type)            # {"AuthenticationError": 1, ...}
print(metrics.durations_ms)              # список тривалостей
```

## Власний middleware

```python
import time

class TraceMiddleware:
    def on_request(self, ctx):
        ctx.extra_params["__trace_start"] = time.time()

    def on_response(self, ctx):
        # send to your APM:
        my_apm.record(
            name=f"ai.{ctx.provider}.{ctx.model}",
            duration_ms=ctx.duration_ms,
            tokens=ctx.response.usage.total_tokens,
            cost=ctx.response.usage.cost_usd,
        )

    def on_error(self, ctx):
        my_apm.record_error(ctx.error)

client = AIClient(..., middleware=[TraceMiddleware()])
```

## Кілька middleware

Передавайте список — викликаються по черзі. Якщо одна middleware кине виняток,
інші все одно виконаються (модуль логує і продовжує).

```python
client = AIClient(..., middleware=[LoggingMiddleware(), MetricsMiddleware(), MyTrace()])
```

# Multimodal (Image Input)

Уніфікований формат для надсилання зображень моделям, що це підтримують
(GPT-4o, Claude 3.5 Sonnet тощо).

## URL зображення

```python
from ai_providers import AIClient, ContentPart, Message

client = AIClient(provider="openai", api_key="...", model="gpt-4o")
response = client.chat([
    Message(role="user", content=[
        ContentPart(type="text", text="What's in this image?"),
        ContentPart(type="image", image_url="https://example.com/cat.jpg"),
    ])
])
print(response.text)
```

## Base64-зображення

```python
import base64

with open("cat.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

response = client.chat([
    Message(role="user", content=[
        ContentPart(type="text", text="What's in this image?"),
        ContentPart(type="image", image_base64=b64, mime_type="image/png"),
    ])
])
```

## Перемикання на Anthropic

Той самий код, тільки `provider="anthropic"`, `model="claude-3-5-sonnet-20241022"`.

Зауваження по форматах:
- Base64 image — конвертується у формат `{"type": "image", "source": {"type": "base64", ...}}` для Claude.
- URL image — конвертується у `{"type": "image", "source": {"type": "url", ...}}`.

## Аудіо (експериментально)

OpenAI підтримує аудіо-вхід для деяких моделей. Передавайте через
`ContentPart(type="audio", image_base64=..., mime_type="wav")`.

Anthropic у v1 модуля не підтримує аудіо-вхід — кине `ValueError`.

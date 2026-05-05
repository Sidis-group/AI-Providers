"""Unified data types shared across all providers and backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]
FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "error"]
SearchMode = Literal["semantic", "keyword", "hybrid"]


@dataclass
class ContentPart:
    """A single piece of multimodal content within a message."""

    type: Literal["text", "image", "audio"]
    text: str | None = None
    image_url: str | None = None
    image_base64: str | None = None
    mime_type: str | None = None


@dataclass
class ToolCall:
    """A tool/function call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Tool:
    """A tool definition exposed to the model (JSON-schema parameters)."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class Message:
    """A chat message in the unified format."""

    role: Role
    content: str | list[ContentPart]
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


@dataclass
class Usage:
    """Token usage and (optionally) computed USD cost for a single call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None


@dataclass
class ChatResponse:
    """A unified chat completion response."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: FinishReason = "stop"
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    raw: Any = None


@dataclass
class StreamChunk:
    """A unified streaming chunk."""

    delta: str = ""
    tool_call_delta: ToolCall | None = None
    finish_reason: FinishReason | None = None
    usage: Usage | None = None
    raw: Any = None


@dataclass
class EmbeddingResponse:
    """Embedding response with one vector per input."""

    vectors: list[list[float]]
    model: str
    usage: Usage = field(default_factory=Usage)
    raw: Any = None


@dataclass
class Document:
    """A document stored in (or queried against) a vector store."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class QueryResult:
    """A single ranked result from a vector store query."""

    document: Document
    score: float


def normalize_messages(
    messages: list[Message] | list[dict[str, Any]],
) -> list[Message]:
    """Convert raw dict messages (OpenAI-style) into Message objects."""

    out: list[Message] = []
    for m in messages:
        if isinstance(m, Message):
            out.append(m)
            continue
        if not isinstance(m, dict):
            raise TypeError(f"Unsupported message type: {type(m).__name__}")
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"Invalid message role: {role!r}")
        content = m.get("content", "")
        out.append(
            Message(
                role=role,  # type: ignore[arg-type]
                content=content,
                name=m.get("name"),
                tool_call_id=m.get("tool_call_id"),
                tool_calls=m.get("tool_calls"),
            )
        )
    return out

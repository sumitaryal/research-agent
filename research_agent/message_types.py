from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Message:
    content: Optional[str] = None


@dataclass
class HumanMessage(Message):
    role: str = "user"


@dataclass
class AIMessage(Message):
    role: str = "assistant"


AnyMessage = Message


def coerce_message(raw: Any) -> Message:
    """Convert common message payloads into Message instances."""
    if isinstance(raw, Message):
        return raw

    if isinstance(raw, dict):
        role = raw.get("role")
        content = raw.get("content")
        if role == "assistant":
            return AIMessage(content=content)
        if role == "user":
            return HumanMessage(content=content)
        return Message(content=content)

    if isinstance(raw, str):
        return HumanMessage(content=raw)

    raise TypeError(f"Unsupported message type: {type(raw)!r}")

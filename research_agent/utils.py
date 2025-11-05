from __future__ import annotations

from typing import Iterable, List

from .message_types import AnyMessage, coerce_message


def strip_code_fence(text: str) -> str:
    """Remove common triple-backtick code fence wrappers from model output."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if not lines:
        return stripped

    # Drop the opening fence (e.g. ```json) and closing fence if present.
    lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def ensure_messages(
    messages: Iterable[AnyMessage] | None,
) -> List[AnyMessage]:
    """Normalize inbound message payloads into Message instances."""
    if not messages:
        return []
    return [coerce_message(message) for message in messages]


__all__ = ["ensure_messages", "strip_code_fence"]

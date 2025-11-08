from __future__ import annotations

import asyncio
import random
from typing import Any, Sequence
from urllib.parse import urlsplit, urlunsplit

import httpx

from .message_types import AnyMessage, AIMessage, HumanMessage
from .research_models import Citation, SourceSegment


_USER_AGENT = "ResearchAgent/1.0"
_REQUEST_TIMEOUT = httpx.Timeout(
    connect=5.0,
    read=10.0,
    write=10.0,
    pool=None,
)
_REQUEST_LIMITS = httpx.Limits(
    max_connections=8,
    max_keepalive_connections=4,
)
_MAX_REDIRECTS = 6
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 4
_BASE_BACKOFF = 0.4
_MAX_BACKOFF = 4.0
_ALLOWED_SCHEMES = {"http", "https"}


def get_research_topic(messages: Sequence[AnyMessage]) -> str:
    """Return the most recent user request for downstream planning."""
    if not messages:
        return ""

    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            return str(content) if content is not None else ""

    fallback = messages[-1].content
    return str(fallback) if fallback is not None else ""


def get_conversation_history(messages: Sequence[AnyMessage]) -> str:
    """Format prior turns so prompts can reference conversation memory."""
    if not messages:
        return ""

    history_lines: list[str] = []
    for message in messages:
        content = (message.content or "").strip() if message.content else ""
        if not content:
            continue
        if isinstance(message, HumanMessage):
            speaker = "User"
        elif isinstance(message, AIMessage):
            speaker = "Assistant"
        else:
            speaker = getattr(message, "role", None) or "Message"
        history_lines.append(f"{speaker}: {content}")

    return "\n".join(history_lines)


def _normalize_url(url: str) -> str:
    """Return a normalized form of the supplied URL, or an empty string."""
    if not url:
        return ""

    trimmed = url.strip()
    if not trimmed:
        return ""

    parsed = urlsplit(trimmed)
    scheme = parsed.scheme.lower() if parsed.scheme else ""
    netloc = parsed.netloc.lower()

    if scheme and scheme not in _ALLOWED_SCHEMES:
        return ""

    if netloc.endswith("."):
        netloc = netloc[:-1]

    if netloc:
        host, sep, port = netloc.partition(":")
        if (scheme == "http" and port == "80") or (
            scheme == "https" and port == "443"
        ):
            netloc = host

    return urlunsplit(
        (
            scheme,
            netloc,
            parsed.path or "",
            parsed.query or "",
            "",
        )
    )


def _is_safe_redirect(original_url: str, candidate_url: str) -> bool:
    """Guard against open redirects and obviously unsafe destinations."""
    if not candidate_url:
        return False

    candidate = urlsplit(candidate_url)
    if candidate.scheme not in _ALLOWED_SCHEMES:
        return False
    if not candidate.netloc:
        return False
    if candidate.hostname and candidate.hostname.startswith("."):
        return False
    if "@" in candidate.netloc.split(":")[0]:
        return False

    original = urlsplit(original_url)
    if not original.netloc:
        return True

    if original.scheme == "https":
        return True

    return candidate.hostname == original.hostname


def _backoff_delay(attempt: int) -> float:
    """Return a jittered exponential backoff delay for the current attempt."""
    exponent = min(_BASE_BACKOFF * (2 ** (attempt - 1)), _MAX_BACKOFF)
    jitter = random.uniform(0.75, 1.25)
    return exponent * jitter


async def _resolve_grounding_url(client: httpx.AsyncClient, url: str) -> str:
    """Resolve a Vertex AI redirect URL to its final destination."""
    fallback_status = {400, 405, 501}
    normalized_original = _normalize_url(url) or url

    for method in ("HEAD", "GET"):
        last_status: int | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = await client.request(method, normalized_original)
            except httpx.HTTPError:
                if attempt == _MAX_RETRIES:
                    break
                await asyncio.sleep(_backoff_delay(attempt))
                continue

            last_status = response.status_code

            if method == "HEAD" and last_status in fallback_status:
                break

            if last_status in _RETRYABLE_STATUS:
                if attempt == _MAX_RETRIES:
                    break
                await asyncio.sleep(_backoff_delay(attempt))
                continue

            resolved = _normalize_url(str(response.url))
            if not resolved:
                continue

            if method == "HEAD" and resolved == normalized_original:
                break

            if _is_safe_redirect(normalized_original, resolved):
                return resolved

            return normalized_original

        if method == "HEAD" and last_status in fallback_status:
            continue

    return normalized_original


async def resolve_urls(
    urls_to_resolve: Sequence[Any], _identifier: int
) -> dict[str, str]:
    """Map Vertex AI URLs to their resolved destinations."""
    urls = [
        getattr(site.web, "uri", None)
        for site in urls_to_resolve
        if getattr(site, "web", None) is not None
    ]

    to_resolve: list[str] = []
    resolved_map: dict[str, str] = {}
    for url in urls:
        if url and url not in resolved_map:
            resolved_map[url] = url
            to_resolve.append(url)

    if not to_resolve:
        return resolved_map

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=_REQUEST_TIMEOUT,
        limits=_REQUEST_LIMITS,
        headers={"User-Agent": _USER_AGENT},
        max_redirects=_MAX_REDIRECTS,
    ) as client:
        tasks = [_resolve_grounding_url(client, url) for url in to_resolve]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    for original, result in zip(to_resolve, results, strict=False):
        if isinstance(result, str):
            resolved_map[original] = result or original
        else:
            continue

    return resolved_map


async def insert_citation_markers(
    text: str, citations_list: Sequence[Citation]
) -> str:
    """Insert markdown citation markers into the supplied text."""
    sorted_citations = sorted(
        citations_list,
        key=lambda c: (c.end_index, c.start_index),
        reverse=True,
    )

    modified_text = text
    for citation_info in sorted_citations:
        end_idx = citation_info.end_index
        marker_to_insert = ""
        for segment in citation_info.segments:
            if segment.value:
                marker_to_insert += f" [{segment.label}]({segment.value})"

        modified_text = (
            modified_text[:end_idx]
            + marker_to_insert
            + modified_text[end_idx:]
        )

    return modified_text


async def get_citations(
    response: Any, resolved_urls_map: dict[str, str]
) -> list[Citation]:
    """Convert Gemini grounding metadata into structured citation objects."""
    citations: list[Citation] = []

    if not response or not response.candidates:
        return citations

    candidate = response.candidates[0]
    if (
        not hasattr(candidate, "grounding_metadata")
        or not candidate.grounding_metadata
        or not hasattr(candidate.grounding_metadata, "grounding_supports")
    ):
        return citations

    for support in candidate.grounding_metadata.grounding_supports:
        if not hasattr(support, "segment") or support.segment is None:
            continue

        start_index = (
            support.segment.start_index
            if support.segment.start_index is not None
            else 0
        )

        if support.segment.end_index is None:
            continue

        segments: list[SourceSegment] = []
        if (
            hasattr(support, "grounding_chunk_indices")
            and support.grounding_chunk_indices
        ):
            for ind in support.grounding_chunk_indices:
                try:
                    chunk = candidate.grounding_metadata.grounding_chunks[ind]
                    resolved_url = (
                        resolved_urls_map.get(chunk.web.uri, None)
                        or chunk.web.uri
                    )
                    title = getattr(chunk.web, "title", "") or ""
                    if title:
                        label = title.split(".")[0]
                    else:
                        label = chunk.web.uri
                    segments.append(
                        SourceSegment(
                            label=label or "source",
                            value=resolved_url,
                        )
                    )
                except (IndexError, AttributeError, NameError):
                    pass
        citations.append(
            Citation(
                start_index=start_index,
                end_index=support.segment.end_index,
                segments=segments,
            )
        )
    return citations

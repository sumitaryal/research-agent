from __future__ import annotations

import asyncio
from typing import Any, List

from pydantic import BaseModel, ConfigDict, Field

from .message_types import AnyMessage


class SearchQueryList(BaseModel):
    query: List[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )


class Reflection(BaseModel):
    confidence: float = Field(
        description="Model-estimated confidence between 0 and 1 that the current summaries are sufficient."
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification."
    )
    follow_up_queries: List[str] = Field(
        description="A list of follow-up queries to address the knowledge gap."
    )


class SourceSegment(BaseModel):
    label: str = Field(
        description="Human readable label for the cited source."
    )
    value: str | None = Field(
        default=None,
        description="Original source URL, if available.",
    )


class Citation(BaseModel):
    start_index: int = Field(
        description="Start index of the text span covered by this citation."
    )
    end_index: int = Field(
        description="End index (exclusive) of the text span covered by this citation."
    )
    segments: List[SourceSegment] = Field(
        default_factory=list,
        description="Source segments supporting the citation.",
    )


class WebResearchResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    summary: str = Field(
        description="Model generated summary of the web research findings."
    )
    sources: List[SourceSegment] = Field(
        default_factory=list,
        description="Flattened list of cited source segments.",
    )
    citations: List[Citation] = Field(
        default_factory=list, description="Structured citation spans."
    )
    raw_text: str = Field(
        default="",
        description="Original model response text before citation markers.",
    )
    response: Any | None = Field(
        default=None,
        description="Raw response object used for citation extraction.",
        exclude=True,
    )
    resolution_task: asyncio.Task[dict[str, str]] | None = Field(
        default=None,
        description="Background task resolving Vertex AI URLs to their final destinations.",
        exclude=True,
    )


class ResearchRunResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: List[AnyMessage]
    search_query: List[str]
    web_research_result: List[str]
    sources_gathered: List[SourceSegment]
    initial_search_query_count: int
    max_research_loops: int
    research_loop_count: int
    confidence: float | None = None
    knowledge_gap: str | None = None
    follow_up_queries: List[str] | None = None

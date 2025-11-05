from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, TypedDict

from .message_types import AnyMessage
from .research_models import SourceSegment


class OverallState(TypedDict, total=False):
    messages: List[AnyMessage]
    search_query: List[str]
    web_research_result: List[str]
    sources_gathered: List[SourceSegment]
    initial_search_query_count: int | None
    max_research_loops: int | None
    research_loop_count: int | None
    reasoning_model: str | None


class ReflectionState(TypedDict, total=False):
    confidence: float
    knowledge_gap: str
    follow_up_queries: List[str]
    research_loop_count: int
    number_of_ran_queries: int


class Query(TypedDict, total=False):
    query: str
    rationale: str


class QueryGenerationState(TypedDict, total=False):
    search_query: List[Query]


class WebSearchState(TypedDict):
    search_query: str
    id: int


@dataclass(kw_only=True)
class SearchStateOutput:
    running_summary: str | None = field(default=None)

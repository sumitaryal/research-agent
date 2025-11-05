from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, TYPE_CHECKING

from google.genai import Client

from .budgeting import AdaptiveMetrics, ThinkingBudgetContext, estimate_answer_length_target
from .config import Configuration
from .research_models import Reflection, SourceSegment, WebResearchResult
from .research_utils import get_citations, insert_citation_markers

if TYPE_CHECKING:
    from .agent import ResearchAgent

logger = logging.getLogger(__name__)


@dataclass
class InflightQuery:
    """Metadata captured for each in-flight web search request."""

    query_id: int
    started_at: float


class GraphState(Enum):
    PLAN = "plan"
    SEARCH = "search"
    REFLECT = "reflect"
    DECIDE = "decide"
    SYNTHESIZE = "synthesize"


@dataclass
class GraphRunOutput:
    final_answer: str
    unique_sources: list[SourceSegment]
    executed_queries: list[str]
    web_research_summaries: list[str]
    latest_reflection: Reflection | None
    loop_count: int
    confidence: float | None
    transition_log: list[str]


class ResearchGraphRunner:
    """Small deterministic state machine orchestrating the research workflow."""

    def __init__(
        self,
        agent: "ResearchAgent",
        *,
        client: Client,
        configuration: Configuration,
        research_topic: str,
        reasoning_model_override: str | None,
        adaptive_metrics: AdaptiveMetrics,
        initial_queries: Iterable[str],
        query_budget_limit: int | None,
        max_research_loops: int,
        deadline_seconds: float | None,
        run_started_at: float,
        baseline_latency_estimate: float,
        latency_low_threshold: float,
        latency_high_threshold: float,
        metrics_log_interval: float,
        current_search_concurrency: int,
        min_search_concurrency: int,
        max_search_concurrency: int,
        current_reflection_interval: int,
        reflection_interval_min: int,
        reflection_interval_max: int,
        dynamic_budget_enabled: bool,
        confidence_threshold: float,
        no_followup_reflection_limit: int,
        novelty_window: int,
        adaptive_failure_rate_reduce_threshold: float,
        adaptive_failure_rate_increase_threshold: float,
        adaptive_reflection_failure_rate_reduce_threshold: float,
        adaptive_metrics_min_samples: int,
    ) -> None:
        self.agent = agent
        self.client = client
        self.configuration = configuration
        self.research_topic = research_topic
        self.reasoning_model_override = reasoning_model_override
        self.adaptive_metrics = adaptive_metrics
        self.query_budget_limit = query_budget_limit
        self.max_research_loops = max_research_loops
        self.deadline_seconds = deadline_seconds
        self.run_started_at = run_started_at
        self.baseline_latency_estimate = baseline_latency_estimate
        self.latency_low_threshold = latency_low_threshold
        self.latency_high_threshold = latency_high_threshold
        self.metrics_log_interval = metrics_log_interval
        self.current_search_concurrency = current_search_concurrency
        self.min_search_concurrency = min_search_concurrency
        self.max_search_concurrency = max_search_concurrency
        self.current_reflection_interval = current_reflection_interval
        self.reflection_interval_min = reflection_interval_min
        self.reflection_interval_max = reflection_interval_max
        self.dynamic_budget_enabled = dynamic_budget_enabled
        self.confidence_threshold = confidence_threshold
        self.no_followup_reflection_limit = no_followup_reflection_limit
        self.novelty_window = novelty_window
        self.adaptive_failure_rate_reduce_threshold = (
            adaptive_failure_rate_reduce_threshold
        )
        self.adaptive_failure_rate_increase_threshold = (
            adaptive_failure_rate_increase_threshold
        )
        self.adaptive_reflection_failure_rate_reduce_threshold = (
            adaptive_reflection_failure_rate_reduce_threshold
        )
        self.adaptive_metrics_min_samples = adaptive_metrics_min_samples

        self.pending_queries: deque[tuple[int, str]] = deque()
        self.query_lookup: dict[int, str] = {}
        self.next_query_id = 0
        self.total_queries_enqueued = 0

        self.executed_queries: list[str] = []
        self.web_research_summaries: list[str] = []
        self.gathered_sources: list[SourceSegment] = []
        self.completed_results: list[WebResearchResult] = []

        self.inflight_tasks: dict[
            asyncio.Task[WebResearchResult], InflightQuery
        ] = {}
        self.reflection_task: asyncio.Task[Reflection] | None = None
        self.pending_reflection_result: Reflection | None = None

        self.loop_count = 0
        self.latest_reflection: Reflection | None = None
        self.completed_since_reflection = 0
        self.satisfied = False
        self.consecutive_reflection_without_followups = 0
        self.recent_query_novelty: deque[bool] = deque(
            maxlen=self.novelty_window
        )
        self._seen_source_keys: set[tuple[str, str | None]] = set()

        self.last_metrics_log = 0.0
        self.transition_log: list[GraphState] = []
        self.state = GraphState.PLAN
        self.completed = False

        self.final_answer: str = ""
        self.unique_sources: list[SourceSegment] = []
        self._urls_resolved = False

        self._enqueue_initial_queries(initial_queries)

    async def run(self) -> GraphRunOutput:
        while not self.completed:
            await self.step()

        confidence = (
            self.latest_reflection.confidence
            if self.latest_reflection is not None
            else None
        )

        return GraphRunOutput(
            final_answer=self.final_answer,
            unique_sources=self.unique_sources,
            executed_queries=self.executed_queries,
            web_research_summaries=self.web_research_summaries,
            latest_reflection=self.latest_reflection,
            loop_count=self.loop_count,
            confidence=confidence,
            transition_log=[state.value for state in self.transition_log],
        )

    async def step(self) -> GraphState | None:
        if self.completed:
            return None

        self.transition_log.append(self.state)
        logger.info(
            "research.graph.state",
            extra={"state": self.state.value},
        )

        handler = {
            GraphState.PLAN: self._handle_plan,
            GraphState.SEARCH: self._handle_search,
            GraphState.REFLECT: self._handle_reflect,
            GraphState.DECIDE: self._handle_decide,
            GraphState.SYNTHESIZE: self._handle_synthesize,
        }[self.state]

        next_state = await handler()

        if next_state is None:
            self.completed = True
            return None

        logger.info(
            "research.graph.transition",
            extra={"from": self.state.value, "to": next_state.value},
        )
        self.state = next_state
        return next_state

    def _enqueue_initial_queries(self, initial_queries: Iterable[str]) -> None:
        for query in initial_queries:
            if not self.enqueue_query(query):
                break

    def enqueue_query(self, query: str) -> bool:
        if (
            self.query_budget_limit is not None
            and self.total_queries_enqueued >= self.query_budget_limit
        ):
            logger.info(
                "research.enqueue.skipped_budget",
                extra={
                    "query": query,
                    "enqueued": self.total_queries_enqueued,
                    "budget_limit": self.query_budget_limit,
                },
            )
            return False
        self.query_lookup[self.next_query_id] = query
        self.pending_queries.append((self.next_query_id, query))
        self.next_query_id += 1
        self.total_queries_enqueued += 1
        return True

    def average_latency_for_planning(self) -> float:
        avg = self.adaptive_metrics.average_latency
        if avg <= 0:
            return self.baseline_latency_estimate
        return max(2.5, avg)

    def should_launch_next_query(self) -> bool:
        if self.satisfied:
            return False
        if (
            self.query_budget_limit is not None
            and len(self.executed_queries) + len(self.inflight_tasks)
            >= self.query_budget_limit
        ):
            return False
        if self.deadline_seconds is not None:
            elapsed = time.monotonic() - self.run_started_at
            remaining = self.deadline_seconds - elapsed
            if remaining <= 0:
                return False
            projected = self.average_latency_for_planning() * (
                len(self.inflight_tasks) + 1
            )
            if projected >= remaining:
                return False
        if (
            self.reflection_task is not None
            and self.completed_since_reflection >= self.current_reflection_interval
        ):
            return False
        return True

    async def cancel_inflight_tasks(self) -> None:
        if not self.inflight_tasks:
            return
        active_tasks = list(self.inflight_tasks.keys())
        for task in active_tasks:
            task.cancel()
        await asyncio.gather(*active_tasks, return_exceptions=True)
        self.inflight_tasks.clear()

    def should_trigger_reflection(self) -> bool:
        if self.reflection_task is not None:
            return False
        if self.pending_reflection_result is not None:
            return False
        if not self.web_research_summaries:
            return False
        if not self.inflight_tasks and not self.pending_queries:
            return True
        return self.completed_since_reflection >= self.current_reflection_interval

    def _start_reflection_task(self) -> None:
        if self.reflection_task is not None or not self.web_research_summaries:
            return
        reflection_context = ThinkingBudgetContext(
            phase="reflection",
            research_topic=self.research_topic,
            executed_queries=len(self.executed_queries),
            pending_queries=len(self.pending_queries) + len(self.inflight_tasks),
            reflection_gap=self.completed_since_reflection,
        )
        model_name = (
            self.reasoning_model_override or self.configuration.reflection_model
        )
        self.reflection_task = asyncio.create_task(
            self.agent._reflect(
                client=self.client,
                model_name=model_name,
                research_topic=self.research_topic,
                summaries=self.web_research_summaries,
                context=reflection_context,
                dynamic_budget_enabled=self.dynamic_budget_enabled,
            )
        )

    def update_adaptive_controls(self) -> None:
        if self.adaptive_metrics.sample_size == 0:
            return
        avg_latency = self.adaptive_metrics.average_latency
        failure_rate = self.adaptive_metrics.failure_rate

        now = time.monotonic()
        if now - self.last_metrics_log >= self.metrics_log_interval:
            self.last_metrics_log = now
            logger.info(
                "research.adaptive.metrics",
                extra={
                    "avg_latency": avg_latency,
                    "failure_rate": failure_rate,
                    "sample_size": self.adaptive_metrics.sample_size,
                    "current_concurrency": self.current_search_concurrency,
                    "current_reflection_interval": self.current_reflection_interval,
                },
            )

        if self.adaptive_metrics.sample_size < self.adaptive_metrics_min_samples:
            return

        desired_concurrency = self.current_search_concurrency
        if (
            failure_rate >= self.adaptive_failure_rate_reduce_threshold
            or avg_latency >= self.latency_high_threshold
        ):
            desired_concurrency = max(
                self.min_search_concurrency, self.current_search_concurrency - 1
            )
        elif (
            failure_rate <= self.adaptive_failure_rate_increase_threshold
            and avg_latency <= self.latency_low_threshold
            and self.current_search_concurrency < self.max_search_concurrency
        ):
            desired_concurrency = min(
                self.max_search_concurrency, self.current_search_concurrency + 1
            )

        if desired_concurrency != self.current_search_concurrency:
            logger.info(
                "research.adaptive.concurrency",
                extra={
                    "previous": self.current_search_concurrency,
                    "current": desired_concurrency,
                    "avg_latency": avg_latency,
                    "failure_rate": failure_rate,
                },
            )
            self.current_search_concurrency = desired_concurrency

        desired_interval = self.current_reflection_interval
        if (
            failure_rate
            >= self.adaptive_reflection_failure_rate_reduce_threshold
            or avg_latency >= self.latency_high_threshold
        ):
            desired_interval = max(
                self.reflection_interval_min, self.current_reflection_interval - 1
            )
        elif (
            failure_rate <= self.adaptive_failure_rate_increase_threshold
            and avg_latency <= self.latency_low_threshold
            and self.current_reflection_interval < self.reflection_interval_max
        ):
            desired_interval = min(
                self.reflection_interval_max, self.current_reflection_interval + 1
            )

        if desired_interval != self.current_reflection_interval:
            logger.info(
                "research.adaptive.reflection_interval",
                extra={
                    "previous": self.current_reflection_interval,
                    "current": desired_interval,
                    "avg_latency": avg_latency,
                    "failure_rate": failure_rate,
                },
            )
            self.current_reflection_interval = desired_interval

    async def _handle_plan(self) -> GraphState:
        if self.pending_reflection_result is not None:
            return GraphState.DECIDE
        if self.reflection_task is not None:
            return GraphState.REFLECT
        if self.satisfied:
            return GraphState.SYNTHESIZE
        if self.pending_queries or self.inflight_tasks:
            return GraphState.SEARCH
        if self.web_research_summaries and self.latest_reflection is None:
            return GraphState.REFLECT
        return GraphState.SYNTHESIZE

    async def _handle_search(self) -> GraphState:
        while (self.pending_queries or self.inflight_tasks) and not self.satisfied:
            while (
                self.pending_queries
                and len(self.inflight_tasks) < self.current_search_concurrency
                and not self.satisfied
                and self.should_launch_next_query()
            ):
                await self._launch_next_query()

            if self.reflection_task is not None and self.reflection_task.done():
                return GraphState.REFLECT

            if not self.inflight_tasks:
                if self.reflection_task is not None:
                    return GraphState.REFLECT
                break

            done, _ = await asyncio.wait(
                set(self.inflight_tasks.keys()),
                return_when=asyncio.FIRST_COMPLETED,
            )

            for completed_task in done:
                self._process_completed_task(completed_task)

            self.update_adaptive_controls()

            if self.should_trigger_reflection():
                self._start_reflection_task()

        if self.reflection_task is not None and not self.pending_queries:
            return GraphState.REFLECT
        if self.pending_reflection_result is not None:
            return GraphState.DECIDE
        if self.latest_reflection is None and self.web_research_summaries:
            return GraphState.REFLECT
        return GraphState.DECIDE

    async def _launch_next_query(self) -> None:
        query_id, query = self.pending_queries.popleft()
        pending_snapshot = len(self.pending_queries) + len(self.inflight_tasks) + 1
        web_context = ThinkingBudgetContext(
            phase="web_search",
            research_topic=self.research_topic,
            executed_queries=len(self.executed_queries),
            pending_queries=pending_snapshot,
            reflection_gap=self.completed_since_reflection,
        )
        started_at = time.monotonic()
        task = asyncio.create_task(
            self.agent._web_research(
                client=self.client,
                model_name=self.configuration.query_generator_model,
                query=query,
                query_id=query_id,
                context=web_context,
                dynamic_budget_enabled=self.dynamic_budget_enabled,
            )
        )
        self.inflight_tasks[task] = InflightQuery(
            query_id=query_id, started_at=started_at
        )

    def _process_completed_task(
        self, completed_task: asyncio.Task[WebResearchResult]
    ) -> None:
        inflight_info = self.inflight_tasks.pop(completed_task)
        query_id = inflight_info.query_id
        query = self.query_lookup[query_id]
        duration = max(0.0, time.monotonic() - inflight_info.started_at)
        success = True
        try:
            research_result = completed_task.result()
        except Exception as exc:
            success = False
            logger.warning(
                "research.web_search.failed",
                extra={
                    "query": query,
                    "query_id": query_id,
                    "error": repr(exc),
                },
            )
            research_result = WebResearchResult(summary="", sources=[], citations=[])
        self.adaptive_metrics.record(duration, success)

        self.executed_queries.append(query)
        self.completed_results.append(research_result)
        self.web_research_summaries.append(research_result.summary)
        added_new_source = False
        if research_result.sources:
            for source in research_result.sources:
                source_key = (source.label, source.value)
                if source_key not in self._seen_source_keys:
                    self._seen_source_keys.add(source_key)
                    added_new_source = True
            self.gathered_sources.extend(research_result.sources)
        self.recent_query_novelty.append(added_new_source)
        self.completed_since_reflection += 1

    async def _resolve_web_results(self) -> None:
        if self._urls_resolved:
            return

        updated_summaries: list[str] = []
        updated_sources: list[SourceSegment] = []

        for result in self.completed_results:
            resolved_map: dict[str, str] = {}
            if result.resolution_task is not None:
                try:
                    resolved_map = await result.resolution_task
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "research.url_resolution.failed",
                        extra={"error": repr(exc)},
                    )
                    resolved_map = {}
                finally:
                    result.resolution_task = None

            if resolved_map and result.response is not None:
                citations = await get_citations(result.response, resolved_map)
                summary = await insert_citation_markers(
                    result.raw_text, citations
                )
                result.citations = citations
                result.summary = summary
                result.sources = [
                    segment
                    for citation in citations
                    for segment in citation.segments
                ]

            updated_summaries.append(result.summary)
            updated_sources.extend(result.sources)

        self.web_research_summaries = updated_summaries
        self.gathered_sources = updated_sources

        self._urls_resolved = True

    async def _handle_reflect(self) -> GraphState:
        if self.reflection_task is not None:
            try:
                reflection_result = await self.reflection_task
            finally:
                self.reflection_task = None
            self.pending_reflection_result = reflection_result
        elif (
            self.pending_reflection_result is None
            and self.web_research_summaries
            and self.latest_reflection is None
        ):
            reflection_context = ThinkingBudgetContext(
                phase="reflection",
                research_topic=self.research_topic,
                executed_queries=len(self.executed_queries),
                pending_queries=len(self.pending_queries) + len(self.inflight_tasks),
                reflection_gap=self.completed_since_reflection,
            )
            reflection_result = await self.agent._reflect(
                client=self.client,
                model_name=(
                    self.reasoning_model_override
                    or self.configuration.reflection_model
                ),
                research_topic=self.research_topic,
                summaries=self.web_research_summaries,
                context=reflection_context,
                dynamic_budget_enabled=self.dynamic_budget_enabled,
            )
            self.pending_reflection_result = reflection_result

        return GraphState.DECIDE

    async def _handle_decide(self) -> GraphState:
        if self.pending_reflection_result is not None:
            reflection_result = self.pending_reflection_result
            self.pending_reflection_result = None
            self.loop_count += 1
            self.latest_reflection = reflection_result
            self.completed_since_reflection = 0

            if (
                reflection_result.confidence >= self.confidence_threshold
                or self.loop_count >= self.max_research_loops
            ):
                self.satisfied = True
                self.consecutive_reflection_without_followups = 0
                await self.cancel_inflight_tasks()
                self.pending_queries.clear()
            else:
                follow_ups = list(reflection_result.follow_up_queries)
                if not follow_ups and reflection_result.knowledge_gap.strip():
                    gap_query = reflection_result.knowledge_gap.strip()
                    if gap_query:
                        follow_ups.append(f"Investigate: {gap_query}")
                new_queries = 0
                for follow_up in follow_ups:
                    if follow_up and self.enqueue_query(follow_up):
                        new_queries += 1
                if new_queries == 0:
                    logger.info(
                        "research.reflection.no_followups",
                        extra={
                            "confidence": reflection_result.confidence,
                            "knowledge_gap": reflection_result.knowledge_gap,
                        },
                    )
                    self.consecutive_reflection_without_followups += 1
                else:
                    self.consecutive_reflection_without_followups = 0

            if (
                not self.satisfied
                and self.consecutive_reflection_without_followups
                >= self.no_followup_reflection_limit
            ):
                logger.info(
                    "research.early_stop.no_followups",
                    extra={
                        "loop_count": self.loop_count,
                        "confidence": reflection_result.confidence,
                        "limit": self.no_followup_reflection_limit,
                    },
                )
                self.satisfied = True
                await self.cancel_inflight_tasks()
                self.pending_queries.clear()

        novelty_window_filled = len(self.recent_query_novelty) >= self.novelty_window
        novelty_exhausted = novelty_window_filled and not any(
            self.recent_query_novelty
        )
        if not self.satisfied and novelty_exhausted:
            logger.info(
                "research.early_stop.no_novelty",
                extra={
                    "window": self.novelty_window,
                    "executed_queries": len(self.executed_queries),
                },
            )
            self.satisfied = True
            await self.cancel_inflight_tasks()
            self.pending_queries.clear()

        if self.pending_queries or self.inflight_tasks:
            if self.satisfied:
                await self.cancel_inflight_tasks()
                self.pending_queries.clear()
                return GraphState.SYNTHESIZE
            return GraphState.PLAN

        if not self.satisfied and self.loop_count >= self.max_research_loops:
            self.satisfied = True

        if self.latest_reflection is None and self.web_research_summaries:
            return GraphState.REFLECT

        return GraphState.SYNTHESIZE

    async def _handle_synthesize(self) -> GraphState | None:
        await self._resolve_web_results()

        final_context = ThinkingBudgetContext(
            phase="final_answer",
            research_topic=self.research_topic,
            executed_queries=len(self.executed_queries),
            pending_queries=len(self.pending_queries) + len(self.inflight_tasks),
            reflection_gap=self.completed_since_reflection,
            answer_length_target=estimate_answer_length_target(
                self.research_topic,
                len(self.web_research_summaries),
                len(self.gathered_sources),
            ),
        )

        final_answer, unique_sources = await self.agent._finalize_answer(
            client=self.client,
            model_name=(
                self.reasoning_model_override or self.configuration.answer_model
            ),
            research_topic=self.research_topic,
            summaries=self.web_research_summaries,
            sources=self.gathered_sources,
            context=final_context,
            dynamic_budget_enabled=self.dynamic_budget_enabled,
        )

        self.final_answer = final_answer
        self.unique_sources = unique_sources
        return None


__all__ = [
    "GraphRunOutput",
    "GraphState",
    "InflightQuery",
    "ResearchGraphRunner",
]

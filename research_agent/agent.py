from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Iterable, Mapping

from google.genai import Client
from pydantic import ValidationError

from .config import Configuration, ConfigurationError
from .message_types import AIMessage, HumanMessage
from .prompts import (
    answer_instructions,
    get_current_date,
    query_writer_instructions,
    reflection_instructions,
    web_searcher_instructions,
)
from .research_models import (
    Reflection,
    ResearchRunResult,
    SearchQueryList,
    SourceSegment,
    WebResearchResult,
)
from .workflow_state import OverallState
from .research_utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)
from .agent_runner import ResearchGraphRunner
from .budgeting import (
    AdaptiveMetrics,
    ThinkingBudgetContext,
    adaptive_initial_query_count,
    with_thinking_config,
)
from .utils import ensure_messages, strip_code_fence

logger = logging.getLogger(__name__)


class ResearchAgent:
    """Research workflow powered directly by the Google GenAI client."""

    async def ainvoke(
        self, state: OverallState, config: Mapping[str, Any] | None = None
    ) -> ResearchRunResult:
        try:
            configuration = Configuration.from_runnable_config(config)
            configuration.validate_configuration()
        except ConfigurationError as exc:
            logger.error(
                "research.configuration.invalid",
                extra={"error": str(exc)},
            )
            raise
        client = Client(api_key=configuration.gemini_api_key)

        base_initial_query_count = (
            state.get("initial_search_query_count")
            or configuration.number_of_initial_queries
        )
        max_research_loops = state.get("max_research_loops") or (
            configuration.max_research_loops
        )
        reasoning_model_override = state.get("reasoning_model")
        base_concurrency = max(1, configuration.search_concurrency)
        min_search_concurrency = max(1, configuration.search_concurrency_min)
        max_search_concurrency = max(
            min_search_concurrency, configuration.search_concurrency_max
        )
        current_search_concurrency = max(
            min_search_concurrency,
            min(max_search_concurrency, base_concurrency),
        )

        reflection_interval_min = max(1, configuration.reflection_interval_min)
        reflection_interval_max = max(
            reflection_interval_min, configuration.reflection_interval_max
        )
        base_reflection_interval = max(
            1, configuration.reflection_check_interval
        )
        current_reflection_interval = max(
            reflection_interval_min,
            min(reflection_interval_max, base_reflection_interval),
        )

        state_deadline_raw = state.get("response_deadline_seconds")
        deadline_seconds: float | None
        if isinstance(state_deadline_raw, (int, float)):
            deadline_seconds = float(state_deadline_raw)
        elif isinstance(state_deadline_raw, str):
            try:
                deadline_seconds = float(state_deadline_raw)
            except ValueError:
                deadline_seconds = None
        else:
            deadline_seconds = None
        if deadline_seconds is None and configuration.response_deadline_seconds:
            deadline_seconds = float(configuration.response_deadline_seconds)

        query_budget_limit = configuration.per_run_query_budget
        adaptive_metrics = AdaptiveMetrics()
        latency_low_threshold = configuration.latency_low_threshold
        latency_high_threshold = configuration.latency_high_threshold
        run_started_at = time.monotonic()
        metrics_log_interval = configuration.metrics_log_interval

        messages = ensure_messages(state.get("messages"))
        research_topic = get_research_topic(messages)

        initial_query_count = adaptive_initial_query_count(
            base_initial_query_count,
            research_topic=research_topic,
            deadline_seconds=deadline_seconds,
            reflection_interval=current_reflection_interval,
            min_queries=configuration.initial_queries_min,
            max_queries=configuration.initial_queries_max,
        )

        generated_queries = await self._generate_search_queries(
            client=client,
            model_name=configuration.query_generator_model,
            research_topic=research_topic,
            initial_query_count=initial_query_count,
            dynamic_budget_enabled=configuration.dynamic_thinking_budget,
        )

        logger.info(
            "research.run.start",
            extra={
                "research_topic": research_topic,
                "initial_query_count": initial_query_count,
                "max_research_loops": max_research_loops,
                "search_concurrency": current_search_concurrency,
                "search_concurrency_min": min_search_concurrency,
                "search_concurrency_max": max_search_concurrency,
                "reflection_interval": current_reflection_interval,
                "reflection_interval_min": reflection_interval_min,
                "reflection_interval_max": reflection_interval_max,
                "response_deadline_seconds": deadline_seconds,
                "query_budget_limit": query_budget_limit,
            },
        )

        runner = ResearchGraphRunner(
            agent=self,
            client=client,
            configuration=configuration,
            research_topic=research_topic,
            reasoning_model_override=reasoning_model_override,
            adaptive_metrics=adaptive_metrics,
            initial_queries=generated_queries.query,
            query_budget_limit=query_budget_limit,
            max_research_loops=max_research_loops,
            deadline_seconds=deadline_seconds,
            run_started_at=run_started_at,
            baseline_latency_estimate=configuration.baseline_latency_estimate,
            latency_low_threshold=latency_low_threshold,
            latency_high_threshold=latency_high_threshold,
            metrics_log_interval=metrics_log_interval,
            current_search_concurrency=current_search_concurrency,
            min_search_concurrency=min_search_concurrency,
            max_search_concurrency=max_search_concurrency,
            current_reflection_interval=current_reflection_interval,
            reflection_interval_min=reflection_interval_min,
            reflection_interval_max=reflection_interval_max,
            dynamic_budget_enabled=configuration.dynamic_thinking_budget,
            confidence_threshold=configuration.confidence_threshold,
            no_followup_reflection_limit=configuration.no_followup_reflection_limit,
            novelty_window=configuration.novelty_window,
            adaptive_failure_rate_reduce_threshold=configuration.adaptive_failure_rate_reduce_threshold,
            adaptive_failure_rate_increase_threshold=configuration.adaptive_failure_rate_increase_threshold,
            adaptive_reflection_failure_rate_reduce_threshold=configuration.adaptive_reflection_failure_rate_reduce_threshold,
            adaptive_metrics_min_samples=configuration.adaptive_metrics_min_samples,
        )

        graph_output = await runner.run()

        final_answer = graph_output.final_answer
        unique_sources = graph_output.unique_sources
        executed_queries = graph_output.executed_queries
        web_research_summaries = graph_output.web_research_summaries
        latest_reflection = graph_output.latest_reflection
        loop_count = graph_output.loop_count

        response_messages = messages + [AIMessage(content=final_answer)]

        run_result = ResearchRunResult(
            messages=response_messages,
            search_query=executed_queries,
            web_research_result=web_research_summaries,
            sources_gathered=unique_sources,
            initial_search_query_count=initial_query_count,
            max_research_loops=max_research_loops,
            research_loop_count=loop_count,
            confidence=graph_output.confidence,
            knowledge_gap=(
                latest_reflection.knowledge_gap if latest_reflection else None
            ),
            follow_up_queries=(
                latest_reflection.follow_up_queries
                if latest_reflection
                else None
            ),
        )

        logger.info(
            "research.run.complete",
            extra={
                "executed_queries": len(executed_queries),
                "web_research_results": len(web_research_summaries),
                "sources_gathered": len(unique_sources),
                "research_loop_count": loop_count,
                "confidence": run_result.confidence,
                "graph_transitions": graph_output.transition_log,
            },
        )

        return run_result

    async def achat(
        self, query: str, config: Mapping[str, Any] | None = None
    ) -> ResearchRunResult:
        """Async convenience wrapper for running the agent with a single user query."""
        state: OverallState = {"messages": [HumanMessage(content=query)]}
        return await self.ainvoke(state, config)

    def chat(
        self, query: str, config: Mapping[str, Any] | None = None
    ) -> ResearchRunResult:
        """Synchronous convenience wrapper; use achat(...) when already in an event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError(
                "Cannot use chat() while an event loop is running; "
                "await achat(...) instead."
            )

        return asyncio.run(self.achat(query, config))

    def invoke(
        self, state: OverallState, config: Mapping[str, Any] | None = None
    ) -> ResearchRunResult:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError(
                "Cannot use invoke() while an event loop is running; "
                "await ainvoke(...) instead."
            )

        return asyncio.run(self.ainvoke(state, config))

    async def _generate_search_queries(
        self,
        client: Client,
        model_name: str,
        research_topic: str,
        initial_query_count: int,
        dynamic_budget_enabled: bool,
    ) -> SearchQueryList:
        formatted_prompt = query_writer_instructions.format(
            current_date=get_current_date(),
            research_topic=research_topic,
            number_queries=initial_query_count,
        )

        logger.info(
            "research.generate_queries.start",
            extra={
                "model": model_name,
                "initial_query_count": initial_query_count,
            },
        )
        budget_context = ThinkingBudgetContext(
            phase="query_generation",
            research_topic=research_topic,
            executed_queries=0,
            pending_queries=initial_query_count,
            reflection_gap=0,
        )
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=formatted_prompt,
            config=with_thinking_config(
                model_name,
                {
                    "temperature": 1.0,
                    "response_mime_type": "application/json",
                    "response_schema": SearchQueryList,
                    "max_output_tokens": 2048,
                },
                context=budget_context,
                dynamic_enabled=dynamic_budget_enabled,
            ),
        )

        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            if isinstance(parsed, SearchQueryList):
                return parsed
            try:
                return SearchQueryList.model_validate(parsed)
            except ValidationError as exc:
                logger.exception(
                    "research.generate_queries.parsed_invalid",
                    extra={"model": model_name},
                )
                raise ValueError(
                    f"Failed to validate parsed search queries response: {parsed}"
                ) from exc

        try:
            return SearchQueryList.model_validate_json(
                strip_code_fence(response.text or "")
            )
        except ValidationError as exc:
            logger.exception(
                "research.generate_queries.parse_error",
                extra={"model": model_name},
            )
            raise ValueError(
                f"Failed to parse search queries response: {response.text}"
            ) from exc

    async def _web_research(
        self,
        client: Client,
        model_name: str,
        query: str,
        query_id: int,
        context: ThinkingBudgetContext,
        dynamic_budget_enabled: bool,
    ) -> WebResearchResult:
        logger.info(
            "research.web_search.start",
            extra={"model": model_name, "query": query, "query_id": query_id},
        )
        formatted_prompt = web_searcher_instructions.format(
            current_date=get_current_date(),
            research_topic=query,
        )

        response = await client.aio.models.generate_content(
            model=model_name,
            contents=formatted_prompt,
            config=with_thinking_config(
                model_name,
                {
                    "tools": [{"google_search": {}}],
                    "temperature": 0,
                },
                context=context,
                dynamic_enabled=dynamic_budget_enabled,
            ),
        )

        grounding_chunks = []
        if (
            response.candidates
            and response.candidates[0].grounding_metadata
            and response.candidates[0].grounding_metadata.grounding_chunks
        ):
            grounding_chunks = response.candidates[
                0
            ].grounding_metadata.grounding_chunks

        raw_text = response.text or ""
        citations = await get_citations(response, {})
        modified_text = await insert_citation_markers(raw_text, citations)

        sources_gathered = [
            segment for citation in citations for segment in citation.segments
        ]

        resolution_task: asyncio.Task[dict[str, str]] | None = None
        if grounding_chunks:
            resolution_task = asyncio.create_task(
                resolve_urls(grounding_chunks, query_id)
            )

        return WebResearchResult(
            summary=modified_text,
            sources=sources_gathered,
            citations=citations,
            raw_text=raw_text,
            response=response,
            resolution_task=resolution_task,
        )

    async def _reflect(
        self,
        client: Client,
        model_name: str,
        research_topic: str,
        summaries: Iterable[str],
        context: ThinkingBudgetContext,
        dynamic_budget_enabled: bool,
    ) -> Reflection:
        summary_list = list(summaries)
        logger.info(
            "research.reflection.start",
            extra={
                "model": model_name,
                "research_topic": research_topic,
                "summary_count": len(summary_list),
            },
        )
        formatted_prompt = reflection_instructions.format(
            current_date=get_current_date(),
            research_topic=research_topic,
            summaries="\n\n---\n\n".join(summary_list),
        )

        response = await client.aio.models.generate_content(
            model=model_name,
            contents=formatted_prompt,
            config=with_thinking_config(
                model_name,
                {
                    "temperature": 1.0,
                    "response_mime_type": "application/json",
                    "response_schema": Reflection,
                },
                context=context,
                dynamic_enabled=dynamic_budget_enabled,
            ),
        )

        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            if isinstance(parsed, Reflection):
                return parsed
            try:
                return Reflection.model_validate(parsed)
            except ValidationError as exc:
                logger.exception(
                    "research.reflection.parsed_invalid",
                    extra={"model": model_name},
                )
                raise ValueError(
                    f"Failed to validate parsed reflection response: {parsed}"
                ) from exc

        try:
            return Reflection.model_validate_json(
                strip_code_fence(response.text or "")
            )
        except ValidationError as exc:
            logger.exception(
                "research.reflection.parse_error",
                extra={"model": model_name},
            )
            raise ValueError(
                f"Failed to parse reflection response: {response.text}"
            ) from exc

    async def _finalize_answer(
        self,
        client: Client,
        model_name: str,
        research_topic: str,
        summaries: Iterable[str],
        sources: Iterable[SourceSegment],
        context: ThinkingBudgetContext,
        dynamic_budget_enabled: bool,
    ) -> tuple[str, list[SourceSegment]]:
        summary_list = list(summaries)
        source_list = list(sources)
        logger.info(
            "research.finalize.start",
            extra={
                "model": model_name,
                "research_topic": research_topic,
                "summary_count": len(summary_list),
                "source_count": len(source_list),
            },
        )
        formatted_prompt = answer_instructions.format(
            current_date=get_current_date(),
            research_topic=research_topic,
            summaries="\n---\n\n".join(summary_list),
        )

        response = await client.aio.models.generate_content(
            model=model_name,
            contents=formatted_prompt,
            config=with_thinking_config(
                model_name,
                {"temperature": 0},
                context=context,
                dynamic_enabled=dynamic_budget_enabled,
            ),
        )

        content = response.text or ""
        unique_sources: list[SourceSegment] = []
        seen_identifiers: set[str] = set()

        for source in source_list:
            identifier = source.value
            if not identifier or identifier not in content:
                continue

            if identifier in seen_identifiers:
                continue
            seen_identifiers.add(identifier)
            unique_sources.append(source)

        return content, unique_sources

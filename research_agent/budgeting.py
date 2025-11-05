from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from google.genai.types import (
    GenerateContentConfig,
    GenerateContentConfigOrDict,
    ThinkingConfig,
)


@dataclass(frozen=True)
class ThinkingBudgetContext:
    """Contextual signals used when deriving adaptive thinking budgets."""

    phase: Literal[
        "query_generation", "web_search", "reflection", "final_answer"
    ]
    research_topic: str
    executed_queries: int = 0
    pending_queries: int = 0
    reflection_gap: int = 0
    answer_length_target: int | None = None


@dataclass
class AdaptiveMetrics:
    """Sliding window tracker for latency and success rates."""

    latency_samples: deque[float] = field(
        default_factory=lambda: deque(maxlen=32)
    )
    success_samples: deque[int] = field(
        default_factory=lambda: deque(maxlen=32)
    )

    def record(self, latency: float, success: bool) -> None:
        self.latency_samples.append(latency)
        self.success_samples.append(1 if success else 0)

    @property
    def sample_size(self) -> int:
        return len(self.success_samples)

    @property
    def average_latency(self) -> float:
        if not self.latency_samples:
            return 0.0
        return sum(self.latency_samples) / len(self.latency_samples)

    @property
    def failure_rate(self) -> float:
        if not self.success_samples:
            return 0.0
        failures = len(self.success_samples) - sum(self.success_samples)
        return failures / len(self.success_samples)

    @property
    def success_rate(self) -> float:
        if not self.success_samples:
            return 0.0
        return sum(self.success_samples) / len(self.success_samples)


def estimate_answer_length_target(
    research_topic: str, summary_count: int, source_count: int
) -> int:
    """Heuristic token target for the final answer synthesis stage."""
    topic_words = len(research_topic.split())
    base = 256
    if topic_words > 12:
        base += 64
    if topic_words > 24:
        base += 64
    base += min(summary_count * 72, 288)
    base += min(source_count * 18, 180)
    return max(256, min(1024, base))


def adaptive_initial_query_count(
    base_count: int,
    *,
    research_topic: str,
    deadline_seconds: float | None,
    reflection_interval: int,
    min_queries: int,
    max_queries: int,
) -> int:
    """Derive an initial query batch size that respects topic scope and deadlines."""
    topic_words = len(research_topic.split())
    topic_boost = min(topic_words // 6, 4)
    reflection_boost = max(0, reflection_interval - 1)
    desired = base_count + topic_boost + reflection_boost
    if deadline_seconds is not None:
        if deadline_seconds <= 90:
            desired -= 3
        elif deadline_seconds <= 150:
            desired -= 2
        elif deadline_seconds <= 240:
            desired -= 1
    return max(min_queries, min(max_queries, desired))


def with_thinking_config(
    model_name: str,
    config: GenerateContentConfigOrDict | Mapping[str, Any] | None,
    *,
    context: ThinkingBudgetContext | None = None,
    dynamic_enabled: bool = True,
) -> GenerateContentConfig:
    """Attach the appropriate thinking config for the specified model."""
    if isinstance(config, GenerateContentConfig):
        merged = config.model_copy()
    else:
        payload: dict[str, Any] = dict(config) if config else {}
        merged = GenerateContentConfig(**payload)
    budget = _thinking_budget_for_model(
        model_name,
        context=context,
        dynamic_enabled=dynamic_enabled,
    )
    if budget is not None:
        existing = merged.thinking_config
        if isinstance(existing, ThinkingConfig):
            thinking_config = dict(existing.model_dump(exclude_none=True))
        elif isinstance(existing, dict):
            thinking_config = dict(existing)
        else:
            thinking_config = {}
        thinking_config["thinking_budget"] = budget
        merged.thinking_config = ThinkingConfig(**thinking_config)
    return merged


def _estimate_complexity_score(context: ThinkingBudgetContext) -> float:
    """Derive a coarse complexity score from topic, query count, and reflection cadence."""
    topic_words = len(context.research_topic.split())
    topic_factor = min(topic_words / 8.0, 3.0)
    query_factor = min(context.executed_queries / 4.0, 3.0)
    reflection_factor = min(context.reflection_gap / 2.0, 2.5)
    return max(1.0, 1.0 + topic_factor + query_factor + reflection_factor)


def _thinking_budget_for_model(
    model_name: str,
    *,
    context: ThinkingBudgetContext | None,
    dynamic_enabled: bool,
) -> int | None:
    """Return the desired thinking budget for known Gemini models."""
    normalized = model_name.lower()
    pro_default = 128
    flash_default = 0

    if not dynamic_enabled or context is None:
        if "gemini-2.5-pro" in normalized:
            return pro_default
        if "gemini-2.5-flash" in normalized:
            return flash_default
        return None

    complexity = _estimate_complexity_score(context)
    pending_pressure = min(context.pending_queries / 3.0, 2.5)

    if "gemini-2.5-pro" in normalized:
        target = context.answer_length_target or 512
        if context.phase == "final_answer":
            base = 72
        else:
            base = 56
        complexity_boost = int(complexity * 14)
        length_boost = 0
        if target >= 900:
            length_boost = 40
        elif target >= 700:
            length_boost = 28
        elif target >= 500:
            length_boost = 20
        elif target >= 350:
            length_boost = 12
        reflection_adjustment = 18 if context.phase == "reflection" else 0
        budget = (
            base
            + complexity_boost
            + length_boost
            + reflection_adjustment
            + int(pending_pressure * 6)
        )
        return max(64, min(224, budget))
    if "gemini-2.5-flash" in normalized:
        if context.phase == "query_generation":
            base = 4
        elif context.phase == "web_search":
            base = 10
        else:
            base = 18
        complexity_boost = int(complexity * 4)
        reflection_boost = 6 if context.phase == "reflection" else 0
        budget = base + complexity_boost + reflection_boost + int(
            pending_pressure * 4
        )
        return max(0, min(72, budget))
    return None
__all__ = [
    "AdaptiveMetrics",
    "ThinkingBudgetContext",
    "adaptive_initial_query_count",
    "estimate_answer_length_target",
    "with_thinking_config",
]

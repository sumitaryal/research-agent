from __future__ import annotations

from typing import Any, Mapping, cast

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(ValueError):
    """Raised when the agent configuration is invalid."""


class Configuration(BaseSettings):
    """The configuration for the agent."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    query_generator_model: str = Field(
        default="gemini-2.5-flash",
        description="The name of the language model to use for the agent's query generation.",
    )

    reflection_model: str = Field(
        default="gemini-2.5-flash",
        description="The name of the language model to use for the agent's reflection.",
    )

    answer_model: str = Field(
        default="gemini-2.5-pro",
        description="The name of the language model to use for the agent's answer.",
    )

    number_of_initial_queries: int = Field(
        default=6,
        description="The number of initial search queries to generate.",
    )

    max_research_loops: int = Field(
        default=2,
        description="The maximum number of research loops to perform.",
    )
    gemini_api_key: str = Field(
        default="",
        description="The API key for the Gemini service.",
    )
    search_concurrency: int = Field(
        default=6,
        description=(
            "Maximum number of concurrent web research requests to execute."
        ),
    )
    reflection_check_interval: int = Field(
        default=3,
        description=(
            "Number of newly completed web searches before reevaluating next steps."
        ),
    )
    dynamic_thinking_budget: bool = Field(
        default=True,
        description=(
            "When true, adjust thinking budgets using run-time heuristics; fall back to fixed budgets otherwise."
        ),
    )
    search_concurrency_min: int = Field(
        default=2,
        description="Lower bound when adaptively tuning search concurrency.",
    )
    search_concurrency_max: int = Field(
        default=8,
        description="Upper bound when adaptively tuning search concurrency.",
    )
    reflection_interval_min: int = Field(
        default=2,
        description="Minimum number of completions between reflections when auto-tuning.",
    )
    reflection_interval_max: int = Field(
        default=6,
        description="Maximum number of completions between reflections when auto-tuning.",
    )
    initial_queries_min: int = Field(
        default=3,
        description="Minimum initial queries to request when planning adaptively.",
    )
    initial_queries_max: int = Field(
        default=12,
        description="Maximum initial queries to request when planning adaptively.",
    )
    response_deadline_seconds: int | None = Field(
        default=None,
        description="Soft deadline for completing the research run; used for adaptive backpressure.",
    )
    per_run_query_budget: int | None = Field(
        default=None,
        description="Optional upper bound on total query executions for a single research run.",
    )
    baseline_latency_estimate: float = Field(
        default=6.0,
        description="Fallback average latency (seconds) used before adaptive metrics are collected.",
    )
    latency_low_threshold: float = Field(
        default=4.0,
        description="Latency threshold (seconds) below which the runner may increase concurrency or reflection intervals.",
    )
    latency_high_threshold: float = Field(
        default=9.0,
        description="Latency threshold (seconds) above which the runner may decrease concurrency or reflection intervals.",
    )
    metrics_log_interval: float = Field(
        default=15.0,
        description="Minimum interval (seconds) between adaptive metrics log events.",
    )
    confidence_threshold: float = Field(
        default=0.75,
        description="Confidence level required to exit the research loop early.",
    )
    no_followup_reflection_limit: int = Field(
        default=2,
        description="Number of consecutive reflections without new follow-up queries before stopping.",
    )
    novelty_window: int = Field(
        default=4,
        description="Sliding window size for assessing novelty in gathered sources.",
    )
    adaptive_failure_rate_reduce_threshold: float = Field(
        default=0.35,
        description="Failure rate at or above which the runner reduces concurrency.",
    )
    adaptive_failure_rate_increase_threshold: float = Field(
        default=0.1,
        description="Failure rate at or below which the runner increases concurrency.",
    )
    adaptive_reflection_failure_rate_reduce_threshold: float = Field(
        default=0.3,
        description="Failure rate at or above which the runner shortens the reflection interval.",
    )
    adaptive_metrics_min_samples: int = Field(
        default=3,
        description="Minimum number of adaptive samples required before tuning heuristics apply.",
    )

    @classmethod
    def from_runnable_config(
        cls, config: Mapping[str, Any] | None = None
    ) -> "Configuration":
        """Create a Configuration instance from a mapping-style config payload."""
        configurable_raw = (
            config.get("configurable") if config and "configurable" in config else {}
        )

        if not isinstance(configurable_raw, Mapping):
            raise ConfigurationError(
                "Expected 'configurable' overrides to be a mapping of string keys."
            )

        invalid_keys = [key for key in configurable_raw.keys() if not isinstance(key, str)]
        if invalid_keys:
            raise ConfigurationError(
                f"Configurable override keys must be strings (got {invalid_keys})."
            )

        configurable = cast(Mapping[str, Any], configurable_raw)

        try:
            return cls(**configurable)
        except ValidationError as exc:
            raise ConfigurationError(
                "Failed to load agent configuration from runnable config overrides."
            ) from exc

    def validate_configuration(self) -> None:
        """Validate the configuration and raise if any values are invalid."""
        errors: list[str] = []

        if not self.gemini_api_key:
            errors.append(
                "Missing GEMINI_API_KEY. Set the environment variable or provide it via the runnable config."
            )

        if self.number_of_initial_queries <= 0:
            errors.append(
                f"number_of_initial_queries must be greater than 0 (got {self.number_of_initial_queries})."
            )

        if self.max_research_loops <= 0:
            errors.append(
                f"max_research_loops must be greater than 0 (got {self.max_research_loops})."
            )

        for field_name in (
            "search_concurrency",
            "search_concurrency_min",
            "search_concurrency_max",
            "reflection_check_interval",
            "reflection_interval_min",
            "reflection_interval_max",
            "initial_queries_min",
            "initial_queries_max",
        ):
            value = getattr(self, field_name)
            if value <= 0:
                errors.append(f"{field_name} must be greater than 0 (got {value}).")

        if self.search_concurrency_min > self.search_concurrency_max:
            errors.append(
                "search_concurrency_min cannot be greater than search_concurrency_max."
            )

        if self.initial_queries_min > self.initial_queries_max:
            errors.append(
                "initial_queries_min cannot be greater than initial_queries_max."
            )

        if self.reflection_interval_min > self.reflection_interval_max:
            errors.append(
                "reflection_interval_min cannot be greater than reflection_interval_max."
            )

        if self.latency_low_threshold <= 0:
            errors.append(
                f"latency_low_threshold must be greater than 0 (got {self.latency_low_threshold})."
            )
        if self.latency_high_threshold <= 0:
            errors.append(
                f"latency_high_threshold must be greater than 0 (got {self.latency_high_threshold})."
            )
        if self.latency_high_threshold < self.latency_low_threshold:
            errors.append("latency_high_threshold cannot be less than latency_low_threshold.")
        if self.baseline_latency_estimate <= 0:
            errors.append(
                f"baseline_latency_estimate must be greater than 0 (got {self.baseline_latency_estimate})."
            )
        if self.metrics_log_interval <= 0:
            errors.append(
                f"metrics_log_interval must be greater than 0 (got {self.metrics_log_interval})."
            )
        if not 0 <= self.confidence_threshold <= 1:
            errors.append(
                f"confidence_threshold must be between 0 and 1 (got {self.confidence_threshold})."
            )
        if self.no_followup_reflection_limit < 0:
            errors.append(
                f"no_followup_reflection_limit cannot be negative (got {self.no_followup_reflection_limit})."
            )
        if self.novelty_window <= 0:
            errors.append(
                f"novelty_window must be greater than 0 (got {self.novelty_window})."
            )
        if self.adaptive_metrics_min_samples <= 0:
            errors.append(
                f"adaptive_metrics_min_samples must be greater than 0 (got {self.adaptive_metrics_min_samples})."
            )
        for field_name in (
            "adaptive_failure_rate_reduce_threshold",
            "adaptive_failure_rate_increase_threshold",
            "adaptive_reflection_failure_rate_reduce_threshold",
        ):
            value = getattr(self, field_name)
            if not 0 <= value <= 1:
                errors.append(f"{field_name} must be between 0 and 1 (got {value}).")
        if (
            self.adaptive_failure_rate_increase_threshold
            > self.adaptive_failure_rate_reduce_threshold
        ):
            errors.append(
                "adaptive_failure_rate_increase_threshold cannot exceed adaptive_failure_rate_reduce_threshold."
            )

        if errors:
            model_summary = (
                f"query_generator_model='{self.query_generator_model}', "
                f"reflection_model='{self.reflection_model}', "
                f"answer_model='{self.answer_model}'"
            )
            formatted_errors = "\n".join(f"- {message}" for message in errors)
            raise ConfigurationError(
                f"Invalid agent configuration:\n{formatted_errors}\nModels configured: {model_summary}"
            )

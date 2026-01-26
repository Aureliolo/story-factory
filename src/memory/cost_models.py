"""Cost tracking models for generation metrics.

These models define the structure for:
- Generation metrics (tokens, time)
- Generation run costs and aggregations
- Entity type and model breakdowns
"""

import logging
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class GenerationMetrics(BaseModel):
    """Metrics from a single LLM generation call.

    Extracted from Ollama response fields:
    - prompt_eval_count: tokens in the prompt
    - eval_count: tokens generated
    """

    prompt_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Number of tokens in the prompt",
    )
    completion_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Number of tokens generated",
    )
    total_tokens: int = Field(
        default=0,
        ge=0,
        description="Total tokens (prompt + completion)",
    )
    time_seconds: float = Field(
        default=0.0,
        ge=0,
        description="Time taken for generation in seconds",
    )
    model_id: str = Field(default="", description="Model used for generation")
    agent_role: str = Field(default="", description="Agent role that generated")

    model_config = ConfigDict(use_enum_values=True)

    @property
    def tokens_per_second(self) -> float | None:
        """
        Compute generation throughput as completion tokens divided by elapsed time.

        Returns:
            float | None: Tokens per second if `time_seconds` > 0 and `completion_tokens` is set, `None` otherwise.
        """
        if self.time_seconds > 0 and self.completion_tokens:
            return self.completion_tokens / self.time_seconds
        return None


class EntityTypeCostBreakdown(BaseModel):
    """Cost breakdown for a specific entity type."""

    entity_type: str = Field(description="Entity type (character, location, etc.)")
    count: int = Field(default=0, ge=0, description="Number of entities generated")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens used")
    total_time_seconds: float = Field(default=0.0, ge=0, description="Total time spent")
    avg_iterations: float = Field(
        default=0.0,
        ge=0,
        description="Average refinement iterations",
    )
    wasted_iterations: int = Field(
        default=0,
        ge=0,
        description="Iterations that were rejected/redone",
    )

    model_config = ConfigDict(use_enum_values=True)

    @property
    def avg_tokens_per_entity(self) -> float:
        """
        Compute the average number of tokens used per entity.

        Returns:
            Average tokens per entity as a float; 0.0 if `count` is zero.
        """
        if self.count > 0:
            return self.total_tokens / self.count
        return 0.0

    @property
    def avg_time_per_entity(self) -> float:
        """
        Return the average time spent per entity.

        Returns:
            float: Average time per entity in seconds; 0.0 if `count` is zero.
        """
        if self.count > 0:
            return self.total_time_seconds / self.count
        return 0.0


class ModelCostBreakdown(BaseModel):
    """Cost breakdown for a specific model."""

    model_id: str = Field(description="Model identifier")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens used")
    total_time_seconds: float = Field(default=0.0, ge=0, description="Total time spent")
    call_count: int = Field(default=0, ge=0, description="Number of generation calls")
    avg_quality: float | None = Field(
        default=None,
        ge=0,
        le=10,
        description="Average quality score (if available)",
    )

    model_config = ConfigDict(use_enum_values=True)

    @property
    def tokens_per_second(self) -> float | None:
        """
        Compute the average tokens processed per second.

        Returns:
            float | None: Average tokens per second, or `None` if `total_time_seconds` is zero.
        """
        if self.total_time_seconds > 0:
            return self.total_tokens / self.total_time_seconds
        return None

    @property
    def avg_tokens_per_call(self) -> float:
        """
        Compute the average number of tokens used per generation call.

        Returns:
            The average number of tokens per call as a float; `0.0` if `call_count` is zero.
        """
        if self.call_count > 0:
            return self.total_tokens / self.call_count
        return 0.0


class GenerationRunCosts(BaseModel):
    """Complete cost tracking for a generation run (story or world build)."""

    run_id: str = Field(description="Unique identifier for this run")
    project_id: str = Field(description="Associated project ID")
    run_type: str = Field(description="Type: 'story_generation' or 'world_build'")
    started_at: datetime = Field(default_factory=datetime.now, description="Run start time")
    completed_at: datetime | None = Field(default=None, description="Run completion time")

    # Aggregate totals
    total_tokens: int = Field(default=0, ge=0, description="Total tokens across all calls")
    total_time_seconds: float = Field(default=0.0, ge=0, description="Total generation time")
    total_calls: int = Field(default=0, ge=0, description="Total generation calls")

    # Refinement tracking
    total_iterations: int = Field(default=0, ge=0, description="Total iterations performed")
    wasted_iterations: int = Field(
        default=0,
        ge=0,
        description="Iterations rejected/redone",
    )

    # Breakdowns
    by_entity_type: list[EntityTypeCostBreakdown] = Field(
        default_factory=list,
        description="Cost breakdown by entity type",
    )
    by_model: list[ModelCostBreakdown] = Field(
        default_factory=list,
        description="Cost breakdown by model",
    )

    model_config = ConfigDict(use_enum_values=True)

    @property
    def duration_seconds(self) -> float | None:
        """
        Return the run duration in seconds based on started_at and completed_at.

        Returns:
            float | None: Number of seconds between `started_at` and `completed_at`, or `None` if either timestamp is missing.
        """
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def efficiency_ratio(self) -> float:
        """Calculate efficiency as (useful iterations) / (total iterations).

        Returns:
            Ratio clamped to 0.0-1.0 where 1 is perfect efficiency (no wasted work).
        """
        if self.total_iterations > 0:
            useful = max(0, self.total_iterations - self.wasted_iterations)
            ratio = useful / self.total_iterations
            return max(0.0, min(1.0, ratio))
        return 1.0

    @property
    def avg_tokens_per_call(self) -> float:
        """
        Compute the average number of tokens used per generation call.

        Returns:
            float: Average tokens per call; 0.0 if `total_calls` is 0.
        """
        if self.total_calls > 0:
            return self.total_tokens / self.total_calls
        return 0.0


class CostSummary(BaseModel):
    """Summary of costs across multiple runs for display."""

    total_runs: int = Field(default=0, ge=0, description="Number of generation runs")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens across all runs")
    total_time_seconds: float = Field(default=0.0, ge=0, description="Total time across runs")
    total_iterations: int = Field(default=0, ge=0, description="Total iterations")
    total_wasted_iterations: int = Field(default=0, ge=0, description="Total wasted iterations")

    # Aggregated breakdowns
    by_entity_type: list[EntityTypeCostBreakdown] = Field(
        default_factory=list,
        description="Aggregated by entity type",
    )
    by_model: list[ModelCostBreakdown] = Field(
        default_factory=list,
        description="Aggregated by model",
    )

    # Most recent runs for display
    recent_runs: list[GenerationRunCosts] = Field(
        default_factory=list,
        description="Most recent generation runs",
    )

    model_config = ConfigDict(use_enum_values=True)

    @property
    def avg_tokens_per_run(self) -> float:
        """
        Compute the average number of tokens per run.

        Returns:
            The average tokens per run as a float; `0.0` if `total_runs` is 0.
        """
        if self.total_runs > 0:
            return self.total_tokens / self.total_runs
        return 0.0

    @property
    def overall_efficiency(self) -> float:
        """
        Compute the fraction of iterations that were not wasted across all runs.

        @returns float: A value clamped to 0.0-1.0 representing (total_iterations - total_wasted_iterations) / total_iterations; returns 1.0 if total_iterations is zero.
        """
        if self.total_iterations > 0:
            useful = max(0, self.total_iterations - self.total_wasted_iterations)
            ratio = useful / self.total_iterations
            return max(0.0, min(1.0, ratio))
        return 1.0

    def format_total_time(self) -> str:
        """Format total time as human-readable string.

        Returns:
            Formatted string like '5m 32s' or '2h 15m'.
        """
        total_secs = int(self.total_time_seconds)
        if total_secs < 60:
            return f"{total_secs}s"
        elif total_secs < 3600:
            mins = total_secs // 60
            secs = total_secs % 60
            return f"{mins}m {secs}s"
        else:
            hours = total_secs // 3600
            mins = (total_secs % 3600) // 60
            return f"{hours}h {mins}m"

    def format_total_tokens(self) -> str:
        """
        Format total_tokens into a human-readable string using K/M suffixes when appropriate.

        Returns:
            A string representing total_tokens; no suffix for values less than 1000, `'K'` for thousands with one decimal place (e.g., `1.2K`) for values from 1,000 to 999,999, and `'M'` for millions with one decimal place (e.g., `3.5M`) for values 1,000,000 and above.
        """
        if self.total_tokens < 1000:
            return str(self.total_tokens)
        elif self.total_tokens < 1_000_000:
            return f"{self.total_tokens / 1000:.1f}K"
        else:
            return f"{self.total_tokens / 1_000_000:.1f}M"

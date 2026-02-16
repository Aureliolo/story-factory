"""Shared helpers for world quality service entity modules."""

import logging
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from src.memory.world_quality import JudgeConsistencyConfig, ScoreStatistics

logger = logging.getLogger(__name__)

_RETRY_TEMP_INCREMENT = 0.15
_RETRY_TEMP_MAX = 1.5

# Shared calibration block for all judge prompts.
# Exact D_minimal variant from #246 benchmarking (rank corr 0.94, dim spread 2.7).
# Intentionally minimal: removes tier labels, justification rules, and example
# scores that anchored judges into a narrow 7.4-7.5 safe zone.
# See output/diagnostics/INVESTIGATION_246.md.
JUDGE_CALIBRATION_BLOCK = """Score each dimension 0-10 with one decimal place.
Differentiate between dimensions — scores should vary based on actual quality."""


def retry_temperature(
    config: Any, creation_retries: int, temperature_strategy: str = "escalating"
) -> float:
    """Calculate temperature for creation retries.

    Supports two strategies:
    - "escalating": increases temperature by 0.15 per retry above the configured
      creator temperature, capped at 1.5, to encourage the model to produce
      different names.
    - "stable": returns the configured creator temperature unchanged regardless
      of retries. Useful for relationship generation where escalating temperature
      produces worse JSON compliance.

    Args:
        config: RefinementConfig with creator_temperature.
        creation_retries: Number of retries so far.
        temperature_strategy: Strategy to use ("escalating" or "stable").

    Returns:
        Temperature value for the next creation attempt.
    """
    if temperature_strategy == "stable":
        logger.debug(
            "retry_temperature: stable strategy, returning base=%.2f (retries=%d ignored)",
            config.creator_temperature,
            creation_retries,
        )
        return float(config.creator_temperature)

    computed = config.creator_temperature + (creation_retries * _RETRY_TEMP_INCREMENT)
    capped = computed > _RETRY_TEMP_MAX
    result = float(min(computed, _RETRY_TEMP_MAX))
    logger.debug(
        "retry_temperature: base=%.2f, retries=%d, computed=%.2f, capped=%s",
        config.creator_temperature,
        creation_retries,
        result,
        capped,
    )
    return result


def judge_with_averaging[T: BaseModel](
    judge_fn: Callable[[], T],
    score_model_class: type[T],
    judge_config: JudgeConsistencyConfig,
) -> T:
    """Call a judge function with optional multi-call averaging.

    When multi-call is enabled, calls the judge multiple times and aggregates
    the results using ScoreStatistics for outlier detection and averaging.

    Args:
        judge_fn: Zero-argument callable that returns a score model instance.
            The caller should wrap the actual judge call with a lambda/closure.
        score_model_class: The Pydantic model class for the scores (e.g. CharacterQualityScores).
        judge_config: Configuration controlling multi-call behavior.

    Returns:
        Score model instance — either from a single call or averaged across multiple calls.
    """
    # Single call path — when consistency features are off or multi-call is disabled
    if not judge_config.enabled or not judge_config.multi_call_enabled:
        logger.debug("Multi-call averaging disabled, using single judge call")
        return judge_fn()

    call_count = judge_config.multi_call_count
    logger.debug(
        "Multi-call judge averaging enabled: making %d judge calls",
        call_count,
    )

    # Collect multiple judge results
    results: list[T] = []
    errors: list[str] = []
    for i in range(call_count):
        try:
            result = judge_fn()
            results.append(result)
            logger.debug("Judge call %d/%d succeeded", i + 1, call_count)
        except Exception as e:
            errors.append(str(e))
            logger.warning("Judge call %d/%d failed: %s", i + 1, call_count, e)

    if not results:
        # All calls failed — re-raise via a single call so the caller gets the real error
        logger.error(
            "All %d judge calls failed, falling back to single call. Errors: %s",
            call_count,
            errors,
        )
        return judge_fn()

    if len(results) == 1:
        logger.debug("Only 1 of %d judge calls succeeded, returning single result", call_count)
        return results[0]

    # Aggregate scores across all results
    return _aggregate_scores(results, score_model_class, judge_config)


def _aggregate_scores[T: BaseModel](
    results: list[T],
    score_model_class: type[T],
    judge_config: JudgeConsistencyConfig,
) -> T:
    """Aggregate multiple score model instances into a single averaged result.

    Discovers numeric fields from the Pydantic model, computes per-dimension
    statistics, applies outlier detection, and constructs the final averaged model.

    Args:
        results: List of score model instances (at least 2).
        score_model_class: The Pydantic model class to construct the result.
        judge_config: Configuration for outlier detection and strategy.

    Returns:
        A new score model instance with averaged/median dimension scores and
        combined feedback.
    """
    # Discover numeric (score) fields and string fields from the model
    numeric_fields: list[str] = []
    for field_name, field_info in score_model_class.model_fields.items():
        if field_name == "feedback":
            continue
        annotation = field_info.annotation
        if annotation is float or annotation is int:
            numeric_fields.append(field_name)

    logger.debug(
        "Aggregating %d results across %d numeric dimensions: %s",
        len(results),
        len(numeric_fields),
        numeric_fields,
    )

    # Compute per-dimension statistics
    final_scores: dict[str, float | str] = {}
    for dim in numeric_fields:
        raw_scores = [getattr(r, dim) for r in results]
        stats = ScoreStatistics.calculate(raw_scores)

        # Apply outlier detection
        if judge_config.outlier_detection:
            stats.detect_outliers(judge_config.outlier_std_threshold)

        # Choose final value based on strategy (Literal["median", "mean"])
        strategy = judge_config.outlier_strategy
        if strategy == "median":
            final_scores[dim] = stats.get_median()
        else:
            # "mean" strategy — use filtered mean (excludes outliers)
            final_scores[dim] = stats.get_filtered_mean()

        logger.debug(
            "Dimension '%s': scores=%s, mean=%.2f, std=%.2f, outliers=%s, final=%.2f (strategy=%s)",
            dim,
            raw_scores,
            stats.mean,
            stats.std,
            stats.outliers,
            final_scores[dim],
            judge_config.outlier_strategy,
        )

    # Combine feedback from all results
    feedbacks = [r.feedback for r in results if hasattr(r, "feedback") and r.feedback]
    if len(feedbacks) > 1:
        # Deduplicate feedback while preserving order: keep first occurrence of each distinct string
        combined_feedback = " | ".join(dict.fromkeys(feedbacks))
    elif feedbacks:
        combined_feedback = feedbacks[0]
    else:
        combined_feedback = ""

    # Construct the final score model
    final_scores["feedback"] = combined_feedback
    averaged = score_model_class(**final_scores)
    logger.debug(
        "Averaged %d judge calls: %s (avg=%.2f)",
        len(results),
        {k: f"{v:.1f}" for k, v in final_scores.items() if isinstance(v, float)},
        averaged.average if hasattr(averaged, "average") else 0.0,
    )
    return averaged

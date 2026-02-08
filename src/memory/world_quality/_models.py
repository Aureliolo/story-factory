"""Pydantic models for world quality scoring and refinement configuration.

These models track quality scores from the judge/refinement loop, enabling
iterative improvement of characters, locations, and relationships.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class IterationRecord(BaseModel):
    """Record of a single refinement iteration for analytics."""

    iteration: int = Field(description="1-indexed iteration number")
    entity_data: dict[str, Any] = Field(description="Entity state at this iteration")
    scores: dict[str, Any] = Field(description="Quality scores for this iteration")
    average_score: float = Field(description="Average score for quick comparison")
    feedback: str = Field(default="", description="Judge feedback for this iteration")


class RefinementHistory(BaseModel):
    """Full history of refinement iterations for an entity."""

    entity_type: str = Field(description="Type of entity (faction, item, etc.)")
    entity_name: str = Field(description="Name of the entity")
    iterations: list[IterationRecord] = Field(default_factory=list)
    best_iteration: int = Field(default=0, description="1-indexed best iteration (0 = none)")
    final_iteration: int = Field(default=0, description="Which iteration was returned")
    improvement_detected: bool = Field(default=False, description="Did iterations improve quality")
    peak_score: float = Field(default=0.0, description="Highest score achieved")
    final_score: float = Field(default=0.0, description="Score of returned entity")
    consecutive_degradations: int = Field(
        default=0, description="Count of consecutive score degradations since peak"
    )
    consecutive_plateaus: int = Field(
        default=0, description="Count of consecutive iterations with same score as peak"
    )
    failed_refinements: int = Field(
        default=0,
        description="Count of refinement attempts that raised WorldGenerationError",
    )

    def add_iteration(
        self,
        *,
        entity_data: dict[str, Any],
        scores: dict[str, Any],
        average_score: float,
        feedback: str = "",
    ) -> None:
        """Add an iteration record and update consecutive degradation tracking.

        Iteration numbers are auto-assigned sequentially (1-indexed) based on
        how many successful iterations have been recorded. This avoids mismatch
        when the loop counter advances due to creation retries that never reach
        the judge.

        Args:
            entity_data: Dict representation of the entity for this iteration.
            scores: Dict of dimension scores (e.g. {"atmosphere": 7, ...}).
            average_score: Pre-computed average across all score dimensions.
            feedback: Optional judge feedback text. Defaults to empty string.
        """
        iteration = len(self.iterations) + 1
        self.iterations.append(
            IterationRecord(
                iteration=iteration,
                entity_data=entity_data,
                scores=scores,
                average_score=average_score,
                feedback=feedback,
            )
        )
        if average_score > self.peak_score:
            self.peak_score = average_score
            self.best_iteration = iteration
            self.consecutive_degradations = 0  # Reset on new peak
            self.consecutive_plateaus = 0  # Reset on new peak
        elif average_score < self.peak_score:
            # Score degraded from peak — reset plateau streak
            self.consecutive_degradations += 1
            self.consecutive_plateaus = 0
        else:
            # Score equals peak — plateau, reset degradation streak
            self.consecutive_plateaus += 1
            self.consecutive_degradations = 0

    def get_best_entity(self) -> dict[str, Any] | None:
        """Return entity data from the best-scoring iteration."""
        if self.best_iteration == 0 or not self.iterations:
            return None
        for record in self.iterations:
            if record.iteration == self.best_iteration:
                return record.entity_data
        return None

    def should_stop_early(
        self,
        patience: int,
        min_iterations: int,
        variance_tolerance: float = 0.3,
    ) -> bool:
        """Check if early stopping criteria is met with enhanced checks.

        The enhanced early stopping considers:
        1. Minimum iterations before early stop can trigger
        2. Variance tolerance - normal score variance shouldn't trigger early stop
        3. Consecutive degradations from peak

        Args:
            patience: Number of consecutive degradations to tolerate.
            min_iterations: Minimum iterations before early stop can trigger.
            variance_tolerance: Score variance considered normal (won't trigger early stop).

        Returns:
            True if we should stop (degradation is real and persistent, not just variance).
        """
        # Must have completed minimum iterations
        if len(self.iterations) < min_iterations:
            logger.debug(
                "Early stop check: too few iterations (%d < %d)",
                len(self.iterations),
                min_iterations,
            )
            return False

        # Must have a peak to degrade from
        if self.best_iteration == 0:
            logger.debug("Early stop check: no peak established yet")
            return False

        # Check plateau: flat scores never improve, stop early
        if self.consecutive_plateaus >= patience:
            logger.debug(
                "Early stop triggered: %d consecutive plateaus at peak=%.2f "
                "(patience=%d), no improvement possible",
                self.consecutive_plateaus,
                self.peak_score,
                patience,
            )
            return True

        # Must have enough consecutive degradations
        if self.consecutive_degradations < patience:
            logger.debug(
                "Early stop check: consecutive degradations (%d) below patience (%d)",
                self.consecutive_degradations,
                patience,
            )
            return False

        # Check if the degradation is significant (not just variance)
        if len(self.iterations) >= 2 and variance_tolerance > 0:
            recent_scores = [r.average_score for r in self.iterations[-patience - 1 :]]
            if len(recent_scores) >= 2:
                # Calculate variance of recent scores
                mean_score = sum(recent_scores) / len(recent_scores)
                variance = sum((s - mean_score) ** 2 for s in recent_scores) / len(recent_scores)
                std_dev = variance**0.5

                # If degradation is within variance tolerance, it's likely just noise
                score_drop = self.peak_score - recent_scores[-1]
                if score_drop <= variance_tolerance and std_dev <= variance_tolerance:
                    logger.debug(
                        "Early stop check: degradation (%.3f) within variance tolerance (%.3f), "
                        "std_dev=%.3f, continuing",
                        score_drop,
                        variance_tolerance,
                        std_dev,
                    )
                    return False

        logger.debug(
            "Early stop triggered: %d consecutive degradations (patience=%d), "
            "peak=%.2f at iteration %d, current=%.2f",
            self.consecutive_degradations,
            patience,
            self.peak_score,
            self.best_iteration,
            self.iterations[-1].average_score if self.iterations else 0,
        )
        return True

    def analyze_improvement(self) -> dict[str, Any]:
        """Analyze whether iterations improved quality."""
        score_progression = [r.average_score for r in self.iterations]

        if len(self.iterations) < 2:
            first_score = self.iterations[0].average_score if self.iterations else 0.0
            return {
                "improved": False,
                "reason": "Not enough iterations to compare",
                "best_iteration": self.best_iteration,
                "scoring_rounds": len(self.iterations),
                "score_progression": score_progression,
                "first_score": first_score,
                "peak_score": self.peak_score,
                "final_score": first_score,
                "worsened_after_peak": False,
                "mid_loop_regression": False,
            }

        first_score = self.iterations[0].average_score
        last_score = self.iterations[-1].average_score
        self.improvement_detected = self.peak_score > first_score

        # mid_loop_regression: True if any iteration scored lower than a previous one
        mid_loop_regression = any(
            score_progression[i] < score_progression[i - 1]
            for i in range(1, len(score_progression))
        )

        return {
            "improved": self.improvement_detected,
            "first_score": first_score,
            "peak_score": self.peak_score,
            "final_score": last_score,
            "best_iteration": self.best_iteration,
            "scoring_rounds": len(self.iterations),
            "score_progression": score_progression,
            "worsened_after_peak": last_score < self.peak_score,
            "consecutive_degradations": self.consecutive_degradations,
            "mid_loop_regression": mid_loop_regression,
        }


class BaseQualityScores(BaseModel, ABC):
    """Abstract base class for all quality score models.

    Provides the common interface (feedback, average, to_dict, weak_dimensions)
    that the generic quality refinement loop requires. Each subclass defines
    its own dimension-specific fields and implements these methods.

    populate_by_name=True allows constructing with either the Python field name
    or the alias, so existing code using field names keeps working while the
    LLM JSON schema uses aliases to avoid field name collisions with entity models.
    """

    model_config = ConfigDict(populate_by_name=True)

    feedback: str = ""

    @property
    @abstractmethod
    def average(self) -> float:
        """Calculate average score across all dimensions."""

    @abstractmethod
    def to_dict(self) -> dict[str, float | str]:
        """Convert to dictionary for storage in entity attributes."""

    @abstractmethod
    def weak_dimensions(self, threshold: float = 7.0) -> list[str]:
        """
        List dimension names whose scores are below the given threshold.

        Parameters:
            threshold (float): Score cutoff; dimensions with values less than `threshold` are considered weak.

        Returns:
            list[str]: Names of dimensions with scores below `threshold`.
        """


class RefinementConfig(BaseModel):
    """Configuration for the quality refinement loop."""

    max_iterations: int = Field(default=3, ge=1, le=10, description="Max refinement iterations")
    quality_threshold: float = Field(
        default=7.5, ge=0.0, le=10.0, description="Minimum average score"
    )
    creator_temperature: float = Field(
        default=0.9, ge=0.0, le=2.0, description="Temperature for creation"
    )
    judge_temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="Temperature for judging"
    )
    refinement_temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Temperature for refinement"
    )
    early_stopping_patience: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Stop after N consecutive score degradations after peak",
    )

    # Dynamic temperature settings
    refinement_temp_start: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Starting refinement temperature"
    )
    refinement_temp_end: float = Field(
        default=0.35, ge=0.0, le=2.0, description="Ending refinement temperature"
    )
    refinement_temp_decay: Literal["linear", "exponential", "step"] = Field(
        default="linear", description="Temperature decay curve type"
    )

    # Enhanced early stopping settings
    early_stopping_min_iterations: int = Field(
        default=2, ge=1, le=10, description="Minimum scoring rounds before early stop can trigger"
    )
    early_stopping_variance_tolerance: float = Field(
        default=0.3, ge=0.0, le=2.0, description="Score variance tolerance for early stopping"
    )

    def get_refinement_temperature(
        self, iteration: int, max_iterations: int | None = None
    ) -> float:
        """Calculate the refinement temperature for a given iteration.

        Implements temperature decay to reduce variance as refinement progresses:
        - High temperature early encourages exploration
        - Low temperature later encourages refinement stability

        Args:
            iteration: Current iteration number (1-indexed).
            max_iterations: Total iterations (uses self.max_iterations if None).

        Returns:
            Temperature value for this iteration.
        """
        # Use explicit None check to avoid treating 0 as "unset"
        if max_iterations is None:
            max_iter = self.max_iterations
        else:
            max_iter = max_iterations

        # First iteration uses start temperature
        if iteration <= 1:
            return round(self.refinement_temp_start, 3)

        # Guard against max_iter <= 1 to avoid division by zero
        if max_iter <= 1:
            return round(self.refinement_temp_end, 3)

        # Last iteration uses end temperature
        if iteration >= max_iter:
            return round(self.refinement_temp_end, 3)

        # Calculate progress (0.0 to 1.0)
        progress = (iteration - 1) / (max_iter - 1)

        # Apply decay curve
        if self.refinement_temp_decay == "linear":
            # Linear interpolation
            temp = self.refinement_temp_start + progress * (
                self.refinement_temp_end - self.refinement_temp_start
            )
        elif self.refinement_temp_decay == "exponential":
            # Exponential decay (faster initial drop, then gradual)
            # Using 1 - (1 - progress)^2 gives faster temperature reduction early
            decay_factor = 1 - (1 - progress) ** 2
            temp = self.refinement_temp_start + decay_factor * (
                self.refinement_temp_end - self.refinement_temp_start
            )
        else:
            # Step function - drop at midpoint (only remaining option)
            if progress < 0.5:
                temp = self.refinement_temp_start
            else:
                temp = self.refinement_temp_end

        # Round to 3 decimal places to avoid float precision artifacts like 0.5249999999999999
        temp = round(temp, 3)

        logger.debug(
            "Dynamic temperature for iteration %d/%d: %.3f (decay=%s, progress=%.2f)",
            iteration,
            max_iter,
            temp,
            self.refinement_temp_decay,
            progress,
        )
        return temp

    @classmethod
    def from_settings(cls, settings: Any) -> RefinementConfig:
        """
        Builds a RefinementConfig from a Settings-like object by reading required fields.

        Parameters:
            settings (Any): Object exposing the following attributes (all required):
                - world_quality_max_iterations
                - world_quality_threshold
                - world_quality_creator_temp
                - world_quality_judge_temp
                - world_quality_refinement_temp
                - world_quality_early_stopping_patience
                - world_quality_refinement_temp_start
                - world_quality_refinement_temp_end
                - world_quality_refinement_temp_decay
                - world_quality_early_stopping_min_iterations
                - world_quality_early_stopping_variance_tolerance

        Returns:
            RefinementConfig: Configuration populated from the corresponding settings attributes.

        Raises:
            AttributeError: If any required attribute is missing on the settings object.
        """
        return cls(
            max_iterations=settings.world_quality_max_iterations,
            quality_threshold=settings.world_quality_threshold,
            creator_temperature=settings.world_quality_creator_temp,
            judge_temperature=settings.world_quality_judge_temp,
            refinement_temperature=settings.world_quality_refinement_temp,
            early_stopping_patience=settings.world_quality_early_stopping_patience,
            refinement_temp_start=settings.world_quality_refinement_temp_start,
            refinement_temp_end=settings.world_quality_refinement_temp_end,
            refinement_temp_decay=settings.world_quality_refinement_temp_decay,
            early_stopping_min_iterations=settings.world_quality_early_stopping_min_iterations,
            early_stopping_variance_tolerance=settings.world_quality_early_stopping_variance_tolerance,
        )


class JudgeConsistencyConfig(BaseModel):
    """Configuration for judge consistency improvements.

    Controls how the system handles score variability and outliers
    to make refinement decisions more reliable.
    """

    enabled: bool = Field(
        default=False,
        description="Whether judge consistency features are enabled (opt-in)",
    )
    multi_call_enabled: bool = Field(
        default=False,
        description="Whether to make multiple judge calls and average (expensive)",
    )
    multi_call_count: int = Field(
        default=2,
        ge=2,
        le=5,
        description="Number of judge calls to make if multi_call_enabled",
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for reliable refinement decisions",
    )
    outlier_detection: bool = Field(
        default=True,
        description="Whether to detect and handle outlier scores",
    )
    outlier_std_threshold: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
        description="Standard deviations from mean to consider outlier",
    )
    outlier_strategy: Literal["median", "mean"] = Field(
        default="median",
        description="How to handle outliers: 'median' or 'mean'",
    )

    @classmethod
    def from_settings(cls, settings: Any) -> JudgeConsistencyConfig:
        """
        Create a JudgeConsistencyConfig from a settings-like object.

        Parameters:
            settings (Any): An object exposing the following attributes:
                judge_consistency_enabled, judge_multi_call_enabled, judge_multi_call_count,
                judge_confidence_threshold, judge_outlier_detection, judge_outlier_std_threshold,
                judge_outlier_strategy.

        Returns:
            JudgeConsistencyConfig: Configuration populated from the provided settings.

        Raises:
            AttributeError: If any required attribute is missing on the settings object.
        """
        logger.debug("Building JudgeConsistencyConfig from settings")
        config = cls(
            enabled=settings.judge_consistency_enabled,
            multi_call_enabled=settings.judge_multi_call_enabled,
            multi_call_count=settings.judge_multi_call_count,
            confidence_threshold=settings.judge_confidence_threshold,
            outlier_detection=settings.judge_outlier_detection,
            outlier_std_threshold=settings.judge_outlier_std_threshold,
            outlier_strategy=settings.judge_outlier_strategy,
        )
        logger.debug(
            "JudgeConsistencyConfig created: enabled=%s, multi_call=%s, outlier_detection=%s",
            config.enabled,
            config.multi_call_enabled,
            config.outlier_detection,
        )
        return config

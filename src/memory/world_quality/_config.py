"""Configuration and statistics models for quality refinement.

Contains RefinementConfig, JudgeConsistencyConfig, and ScoreStatistics.
"""

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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
        default=2, ge=1, le=10, description="Minimum iterations before early stop can trigger"
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
            return self.refinement_temp_start

        # Guard against max_iter <= 1 to avoid division by zero
        if max_iter <= 1:
            return self.refinement_temp_end

        # Last iteration uses end temperature
        if iteration >= max_iter:
            return self.refinement_temp_end

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
        default=3,
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
    outlier_strategy: Literal["median", "mean", "retry"] = Field(
        default="median",
        description="How to handle outliers: 'median', 'mean', or 'retry'",
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


class ScoreStatistics(BaseModel):
    """Statistics for a collection of judge scores.

    Used to detect outliers and calculate confidence in refinement decisions.
    """

    scores: list[float] = Field(default_factory=list, description="Raw scores")
    mean: float = Field(default=0.0, description="Mean of scores")
    std: float = Field(default=0.0, description="Standard deviation of scores")
    confidence: float = Field(
        default=1.0,
        description="Confidence in the score (1.0 - coefficient of variation)",
    )
    outliers: list[int] = Field(
        default_factory=list,
        description="Indices of outlier scores",
    )

    @classmethod
    def calculate(cls, scores: list[float]) -> ScoreStatistics:
        """
        Create a ScoreStatistics from a list of numeric judge scores.

        Parameters:
            scores (list[float]): Collection of scores to analyze; may be empty.

        Returns:
            ScoreStatistics: instance containing the original scores, the mean, the sample standard deviation (std), and a confidence score (1 - coefficient of variation bounded to [0, 1], with special handling when mean <= 0).
        """
        logger.debug("Calculating ScoreStatistics for %d scores", len(scores))
        if not scores:
            logger.debug("Empty scores list, returning zero statistics")
            return cls(scores=[], mean=0.0, std=0.0, confidence=0.0)

        n = len(scores)
        mean = sum(scores) / n

        if n == 1:
            return cls(scores=scores, mean=mean, std=0.0, confidence=1.0)

        # Calculate standard deviation
        variance = sum((s - mean) ** 2 for s in scores) / (n - 1)
        std = variance**0.5

        # Coefficient of variation (relative std)
        # Confidence = 1.0 - CV (bounded to [0, 1])
        if mean > 0:
            cv = std / mean
            confidence = max(0.0, min(1.0, 1.0 - cv))
        else:
            confidence = 0.0 if std > 0 else 1.0

        logger.debug(
            "ScoreStatistics calculated: mean=%.3f, std=%.3f, confidence=%.3f",
            mean,
            std,
            confidence,
        )
        return cls(scores=scores, mean=mean, std=std, confidence=confidence)

    def detect_outliers(self, std_threshold: float = 2.0) -> list[int]:
        """
        Detect indices of scores that are statistical outliers based on standard deviations from the mean.

        If the sample standard deviation is zero or there are fewer than 3 scores, no outliers are reported and an empty list is returned.

        Parameters:
            std_threshold (float): Threshold in standard deviations; a score with absolute deviation from the mean
                greater than or equal to this value is considered an outlier.

        Returns:
            list[int]: Indices of scores identified as outliers.
        """
        if self.std == 0 or len(self.scores) < 3:
            logger.debug("Skipping outlier detection: std=%.3f, n=%d", self.std, len(self.scores))
            self.outliers = []  # Clear stale outliers on early return
            return []

        outliers = []
        for i, score in enumerate(self.scores):
            z_score = abs(score - self.mean) / self.std
            if z_score >= std_threshold:
                outliers.append(i)

        self.outliers = outliers
        if outliers:
            logger.debug(
                "Detected %d outliers at indices %s (threshold=%.1f std)",
                len(outliers),
                outliers,
                std_threshold,
            )
        return outliers

    def get_filtered_mean(self, exclude_indices: list[int] | None = None) -> float:
        """
        Return the mean of the stored scores excluding specified indices.

        Parameters:
            exclude_indices (list[int] | None): Indices to exclude from the calculation. If None, `self.outliers` is used.

        Returns:
            float: Mean of scores after excluding the specified indices. If no indices are excluded or exclusion yields an empty set, returns `self.mean`.
        """
        exclude = exclude_indices if exclude_indices is not None else self.outliers
        if not exclude:
            return self.mean

        filtered = [s for i, s in enumerate(self.scores) if i not in exclude]
        if not filtered:
            return self.mean

        return sum(filtered) / len(filtered)

    def get_median(self) -> float:
        """
        Return the median of the stored scores.

        If the scores list is empty, returns 0.0. For an even number of scores, returns the average of the two middle values.

        Returns:
            The median score, or 0.0 when there are no scores.
        """
        if not self.scores:
            return 0.0

        sorted_scores = sorted(self.scores)
        n = len(sorted_scores)
        mid = n // 2

        if n % 2 == 0:
            return (sorted_scores[mid - 1] + sorted_scores[mid]) / 2
        return sorted_scores[mid]

    def should_refine(
        self,
        threshold: float,
        confidence_threshold: float = 0.7,
        min_samples: int = 3,
    ) -> tuple[bool, float]:
        """
        Decides whether another refinement iteration is needed and returns the threshold used for that decision.

        If there are fewer than `min_samples` scores or the calculated confidence is at or above `confidence_threshold`, the base `threshold` is used. If confidence is below `confidence_threshold`, the method increases the threshold proportionally to uncertainty (capped at 9.5) and uses that adjusted value for the decision.

        Parameters:
            threshold (float): Base quality threshold to compare against the mean score.
            confidence_threshold (float): Confidence level above which the base threshold is considered reliable.
            min_samples (int): Minimum number of score samples required to apply confidence-based adjustment.

        Returns:
            tuple[bool, float]: `(should_refine, adjusted_threshold)` where `should_refine` is `True` if the mean score is less than the threshold applied for this decision, and `adjusted_threshold` is the actual threshold used (either `threshold` or a raised value when confidence is low).
        """
        if len(self.scores) < min_samples:
            # Not enough data for confidence-based decisions
            should_refine = self.mean < threshold
            logger.debug(
                "should_refine: insufficient samples (%d < %d), using base threshold %.2f, mean=%.3f, refine=%s",
                len(self.scores),
                min_samples,
                threshold,
                self.mean,
                should_refine,
            )
            return should_refine, threshold

        if self.confidence >= confidence_threshold:
            # High confidence - use standard threshold
            should_refine = self.mean < threshold
            logger.debug(
                "should_refine: high confidence (%.3f >= %.3f), threshold=%.2f, mean=%.3f, refine=%s",
                self.confidence,
                confidence_threshold,
                threshold,
                self.mean,
                should_refine,
            )
            return should_refine, threshold

        # Low confidence - use more conservative threshold
        # Scale up threshold proportionally to uncertainty
        uncertainty_factor = 1.0 + (1.0 - self.confidence)
        adjusted_threshold = min(threshold * uncertainty_factor, 9.5)
        should_refine = self.mean < adjusted_threshold
        logger.debug(
            "should_refine: low confidence (%.3f < %.3f), adjusted threshold %.2f -> %.2f, mean=%.3f, refine=%s",
            self.confidence,
            confidence_threshold,
            threshold,
            adjusted_threshold,
            self.mean,
            should_refine,
        )
        return should_refine, adjusted_threshold

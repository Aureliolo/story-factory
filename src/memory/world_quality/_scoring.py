"""Score statistics and analysis for world quality scoring.

Provides statistical analysis of judge scores including outlier detection,
confidence calculation, and refinement decisions.
"""

import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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

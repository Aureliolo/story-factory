"""Refinement history models for world building entities.

These models track iteration records and refinement history for the
judge/refinement loop.
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

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

    def add_iteration(
        self,
        iteration: int,
        entity_data: dict[str, Any],
        scores: dict[str, Any],
        average_score: float,
        feedback: str = "",
    ) -> None:
        """Add an iteration record and update consecutive degradation tracking."""
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
        elif average_score < self.peak_score:
            # Score degraded from peak
            self.consecutive_degradations += 1
        # Note: if score equals peak, don't reset counter (plateauing after peak)

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
        min_iterations: int = 2,
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
        if len(self.iterations) < 2:
            first_score = self.iterations[0].average_score if self.iterations else 0.0
            return {
                "improved": False,
                "reason": "Not enough iterations to compare",
                "best_iteration": self.best_iteration,
                "total_iterations": len(self.iterations),
                "score_progression": [r.average_score for r in self.iterations],
                "first_score": first_score,
                "peak_score": self.peak_score,
                "final_score": first_score,
                "worsened_after_peak": False,
            }

        first_score = self.iterations[0].average_score
        last_score = self.iterations[-1].average_score
        self.improvement_detected = self.peak_score > first_score

        return {
            "improved": self.improvement_detected,
            "first_score": first_score,
            "peak_score": self.peak_score,
            "final_score": last_score,
            "best_iteration": self.best_iteration,
            "total_iterations": len(self.iterations),
            "score_progression": [r.average_score for r in self.iterations],
            "worsened_after_peak": last_score < self.peak_score,
            "consecutive_degradations": self.consecutive_degradations,
        }

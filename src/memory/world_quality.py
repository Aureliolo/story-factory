"""Quality score models for world building entities.

These models track quality scores from the judge/refinement loop, enabling
iterative improvement of characters, locations, and relationships.
"""

from typing import Any

from pydantic import BaseModel, Field


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

    def should_stop_early(self, patience: int) -> bool:
        """Check if early stopping criteria is met.

        Args:
            patience: Number of consecutive degradations to tolerate

        Returns:
            True if we should stop (consecutive degradations >= patience after peak)
        """
        # Only trigger early stopping if:
        # 1. We have a peak (best_iteration > 0)
        # 2. Consecutive degradations >= patience
        return self.best_iteration > 0 and self.consecutive_degradations >= patience

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


class CharacterQualityScores(BaseModel):
    """Quality scores for a character (0-10 scale).

    All score fields are required - no defaults. This ensures that the LLM
    must provide explicit scores or the parsing will fail, preventing silent
    fallbacks to meaningless default values.
    """

    depth: float = Field(ge=0.0, le=10.0, description="Psychological complexity")
    goals: float = Field(ge=0.0, le=10.0, description="Clarity and story relevance")
    flaws: float = Field(ge=0.0, le=10.0, description="Meaningful vulnerabilities")
    uniqueness: float = Field(ge=0.0, le=10.0, description="Distinctiveness")
    arc_potential: float = Field(ge=0.0, le=10.0, description="Room for transformation")
    feedback: str = ""

    @property
    def average(self) -> float:
        """Calculate average score across all dimensions."""
        return (self.depth + self.goals + self.flaws + self.uniqueness + self.arc_potential) / 5.0

    def to_dict(self) -> dict[str, float | str]:
        """Convert to dictionary for storage in entity attributes."""
        return {
            "depth": self.depth,
            "goals": self.goals,
            "flaws": self.flaws,
            "uniqueness": self.uniqueness,
            "arc_potential": self.arc_potential,
            "average": self.average,
            "feedback": self.feedback,
        }

    def weak_dimensions(self, threshold: float = 7.0) -> list[str]:
        """Return list of dimensions below threshold."""
        weak = []
        if self.depth < threshold:
            weak.append("depth")
        if self.goals < threshold:
            weak.append("goals")
        if self.flaws < threshold:
            weak.append("flaws")
        if self.uniqueness < threshold:
            weak.append("uniqueness")
        if self.arc_potential < threshold:
            weak.append("arc_potential")
        return weak


class LocationQualityScores(BaseModel):
    """Quality scores for a location (0-10 scale).

    All score fields are required - no defaults.
    """

    atmosphere: float = Field(ge=0.0, le=10.0, description="Sensory richness, mood")
    significance: float = Field(ge=0.0, le=10.0, description="Plot/symbolic meaning")
    story_relevance: float = Field(ge=0.0, le=10.0, description="Theme/character links")
    distinctiveness: float = Field(ge=0.0, le=10.0, description="Memorable qualities")
    feedback: str = ""

    @property
    def average(self) -> float:
        """Calculate average score across all dimensions."""
        return (
            self.atmosphere + self.significance + self.story_relevance + self.distinctiveness
        ) / 4.0

    def to_dict(self) -> dict[str, float | str]:
        """Convert to dictionary for storage in entity attributes."""
        return {
            "atmosphere": self.atmosphere,
            "significance": self.significance,
            "story_relevance": self.story_relevance,
            "distinctiveness": self.distinctiveness,
            "average": self.average,
            "feedback": self.feedback,
        }

    def weak_dimensions(self, threshold: float = 7.0) -> list[str]:
        """Return list of dimensions below threshold."""
        weak = []
        if self.atmosphere < threshold:
            weak.append("atmosphere")
        if self.significance < threshold:
            weak.append("significance")
        if self.story_relevance < threshold:
            weak.append("story_relevance")
        if self.distinctiveness < threshold:
            weak.append("distinctiveness")
        return weak


class RelationshipQualityScores(BaseModel):
    """Quality scores for a relationship (0-10 scale).

    All score fields are required - no defaults.
    """

    tension: float = Field(ge=0.0, le=10.0, description="Conflict potential")
    dynamics: float = Field(ge=0.0, le=10.0, description="Complexity, power balance")
    story_potential: float = Field(ge=0.0, le=10.0, description="Scene opportunities")
    authenticity: float = Field(ge=0.0, le=10.0, description="Believability")
    feedback: str = ""

    @property
    def average(self) -> float:
        """Calculate average score across all dimensions."""
        return (self.tension + self.dynamics + self.story_potential + self.authenticity) / 4.0

    def to_dict(self) -> dict[str, float | str]:
        """Convert to dictionary for storage in entity attributes."""
        return {
            "tension": self.tension,
            "dynamics": self.dynamics,
            "story_potential": self.story_potential,
            "authenticity": self.authenticity,
            "average": self.average,
            "feedback": self.feedback,
        }

    def weak_dimensions(self, threshold: float = 7.0) -> list[str]:
        """Return list of dimensions below threshold."""
        weak = []
        if self.tension < threshold:
            weak.append("tension")
        if self.dynamics < threshold:
            weak.append("dynamics")
        if self.story_potential < threshold:
            weak.append("story_potential")
        if self.authenticity < threshold:
            weak.append("authenticity")
        return weak


class FactionQualityScores(BaseModel):
    """Quality scores for a faction (0-10 scale).

    All score fields are required - no defaults.
    """

    coherence: float = Field(ge=0.0, le=10.0, description="Internal consistency")
    influence: float = Field(ge=0.0, le=10.0, description="World impact")
    conflict_potential: float = Field(ge=0.0, le=10.0, description="Story conflict potential")
    distinctiveness: float = Field(ge=0.0, le=10.0, description="Memorable qualities")
    feedback: str = ""

    @property
    def average(self) -> float:
        """Calculate average score across all dimensions."""
        return (
            self.coherence + self.influence + self.conflict_potential + self.distinctiveness
        ) / 4.0

    def to_dict(self) -> dict[str, float | str]:
        """Convert to dictionary for storage in entity attributes."""
        return {
            "coherence": self.coherence,
            "influence": self.influence,
            "conflict_potential": self.conflict_potential,
            "distinctiveness": self.distinctiveness,
            "average": self.average,
            "feedback": self.feedback,
        }

    def weak_dimensions(self, threshold: float = 7.0) -> list[str]:
        """Return list of dimensions below threshold."""
        weak = []
        if self.coherence < threshold:
            weak.append("coherence")
        if self.influence < threshold:
            weak.append("influence")
        if self.conflict_potential < threshold:
            weak.append("conflict_potential")
        if self.distinctiveness < threshold:
            weak.append("distinctiveness")
        return weak


class ItemQualityScores(BaseModel):
    """Quality scores for an item (0-10 scale).

    All score fields are required - no defaults.
    """

    significance: float = Field(ge=0.0, le=10.0, description="Story importance")
    uniqueness: float = Field(ge=0.0, le=10.0, description="Distinctive qualities")
    narrative_potential: float = Field(ge=0.0, le=10.0, description="Plot opportunities")
    integration: float = Field(ge=0.0, le=10.0, description="Fits world")
    feedback: str = ""

    @property
    def average(self) -> float:
        """Calculate average score across all dimensions."""
        return (
            self.significance + self.uniqueness + self.narrative_potential + self.integration
        ) / 4.0

    def to_dict(self) -> dict[str, float | str]:
        """Convert to dictionary for storage in entity attributes."""
        return {
            "significance": self.significance,
            "uniqueness": self.uniqueness,
            "narrative_potential": self.narrative_potential,
            "integration": self.integration,
            "average": self.average,
            "feedback": self.feedback,
        }

    def weak_dimensions(self, threshold: float = 7.0) -> list[str]:
        """Return list of dimensions below threshold."""
        weak = []
        if self.significance < threshold:
            weak.append("significance")
        if self.uniqueness < threshold:
            weak.append("uniqueness")
        if self.narrative_potential < threshold:
            weak.append("narrative_potential")
        if self.integration < threshold:
            weak.append("integration")
        return weak


class ConceptQualityScores(BaseModel):
    """Quality scores for a concept (0-10 scale).

    All score fields are required - no defaults.
    """

    relevance: float = Field(ge=0.0, le=10.0, description="Theme alignment")
    depth: float = Field(ge=0.0, le=10.0, description="Philosophical richness")
    manifestation: float = Field(ge=0.0, le=10.0, description="How it appears in story")
    resonance: float = Field(ge=0.0, le=10.0, description="Emotional impact")
    feedback: str = ""

    @property
    def average(self) -> float:
        """Calculate average score across all dimensions."""
        return (self.relevance + self.depth + self.manifestation + self.resonance) / 4.0

    def to_dict(self) -> dict[str, float | str]:
        """Convert to dictionary for storage in entity attributes."""
        return {
            "relevance": self.relevance,
            "depth": self.depth,
            "manifestation": self.manifestation,
            "resonance": self.resonance,
            "average": self.average,
            "feedback": self.feedback,
        }

    def weak_dimensions(self, threshold: float = 7.0) -> list[str]:
        """Return list of dimensions below threshold."""
        weak = []
        if self.relevance < threshold:
            weak.append("relevance")
        if self.depth < threshold:
            weak.append("depth")
        if self.manifestation < threshold:
            weak.append("manifestation")
        if self.resonance < threshold:
            weak.append("resonance")
        return weak


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

    @classmethod
    def from_settings(cls, settings: Any) -> RefinementConfig:
        """Create config from Settings object.

        All settings fields are required - no default fallbacks per coding guidelines.
        Missing fields will raise AttributeError.
        """
        return cls(
            max_iterations=settings.world_quality_max_iterations,
            quality_threshold=settings.world_quality_threshold,
            creator_temperature=settings.world_quality_creator_temp,
            judge_temperature=settings.world_quality_judge_temp,
            refinement_temperature=settings.world_quality_refinement_temp,
            early_stopping_patience=settings.world_quality_early_stopping_patience,
        )


class JudgeConsistencyConfig(BaseModel):
    """Configuration for judge consistency improvements.

    Controls how the system handles score variability and outliers
    to make refinement decisions more reliable.
    """

    enabled: bool = Field(
        default=True,
        description="Whether judge consistency features are enabled",
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
    outlier_strategy: str = Field(
        default="median",
        description="How to handle outliers: 'median', 'mean', or 'retry'",
    )

    @classmethod
    def from_settings(cls, settings: Any) -> JudgeConsistencyConfig:
        """Create config from Settings object."""
        return cls(
            enabled=settings.judge_consistency_enabled,
            multi_call_enabled=settings.judge_multi_call_enabled,
            multi_call_count=settings.judge_multi_call_count,
            confidence_threshold=settings.judge_confidence_threshold,
            outlier_detection=settings.judge_outlier_detection,
            outlier_std_threshold=settings.judge_outlier_std_threshold,
            outlier_strategy=settings.judge_outlier_strategy,
        )


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
        """Calculate statistics from a list of scores.

        Args:
            scores: List of numeric scores.

        Returns:
            ScoreStatistics with calculated mean, std, and confidence.
        """
        if not scores:
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

        return cls(scores=scores, mean=mean, std=std, confidence=confidence)

    def detect_outliers(self, std_threshold: float = 2.0) -> list[int]:
        """Detect outlier scores using standard deviation method.

        Args:
            std_threshold: Number of standard deviations from mean to consider outlier.

        Returns:
            List of indices of outlier scores.
        """
        if self.std == 0 or len(self.scores) < 3:
            return []

        outliers = []
        for i, score in enumerate(self.scores):
            z_score = abs(score - self.mean) / self.std
            if z_score >= std_threshold:
                outliers.append(i)

        self.outliers = outliers
        return outliers

    def get_filtered_mean(self, exclude_indices: list[int] | None = None) -> float:
        """Get mean excluding specified indices (e.g., outliers).

        Args:
            exclude_indices: Indices to exclude. If None, uses self.outliers.

        Returns:
            Mean of non-excluded scores.
        """
        exclude = exclude_indices if exclude_indices is not None else self.outliers
        if not exclude:
            return self.mean

        filtered = [s for i, s in enumerate(self.scores) if i not in exclude]
        if not filtered:
            return self.mean

        return sum(filtered) / len(filtered)

    def get_median(self) -> float:
        """Get median score.

        Returns:
            Median of scores.
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
        """Determine if refinement should continue based on score statistics.

        Uses confidence-based decision making:
        - High confidence: use standard threshold
        - Low confidence: use more conservative (higher) threshold

        Args:
            threshold: Base quality threshold.
            confidence_threshold: Minimum confidence for reliable decisions.
            min_samples: Minimum samples needed for confidence-based decisions.

        Returns:
            Tuple of (should_refine, adjusted_threshold).
        """
        if len(self.scores) < min_samples:
            # Not enough data for confidence-based decisions
            return self.mean < threshold, threshold

        if self.confidence >= confidence_threshold:
            # High confidence - use standard threshold
            return self.mean < threshold, threshold

        # Low confidence - use more conservative threshold
        # Scale up threshold proportionally to uncertainty
        uncertainty_factor = 1.0 + (1.0 - self.confidence)
        adjusted_threshold = min(threshold * uncertainty_factor, 9.5)

        return self.mean < adjusted_threshold, adjusted_threshold

"""Database for model scoring and mode management.

This package provides the ModeDatabase class composed from focused mixins:
- ModeDatabaseBase: Core initialization and schema management
- ScoringMixin: Generation score recording and retrieval
- PerformanceMixin: Model performance aggregates
- RecommendationsMixin: Tuning recommendations
- CustomModesMixin: Custom generation modes
- AnalyticsMixin: Analytics queries
- WorldEntityMixin: World entity scoring
- PromptMetricsMixin: Prompt template metrics
- RefinementMixin: Refinement effectiveness analytics
- CostTrackingMixin: Generation cost tracking
"""

from ._analytics import AnalyticsMixin
from ._base import DEFAULT_DB_PATH, ModeDatabaseBase
from ._cost_tracking import CostTrackingMixin
from ._custom_modes import CustomModesMixin
from ._performance import PerformanceMixin
from ._prompt_metrics import PromptMetricsMixin
from ._recommendations import RecommendationsMixin
from ._refinement import RefinementMixin
from ._scoring import ScoringMixin
from ._world_entity import WorldEntityMixin


class ModeDatabase(
    ScoringMixin,
    PerformanceMixin,
    RecommendationsMixin,
    CustomModesMixin,
    AnalyticsMixin,
    WorldEntityMixin,
    PromptMetricsMixin,
    RefinementMixin,
    CostTrackingMixin,
    ModeDatabaseBase,
):
    """SQLite database for model scoring and learning data.

    Tables:
    - generation_scores: Per-generation metrics (quality, speed, implicit signals)
    - model_performance: Aggregated model performance by role and genre
    - recommendations: Tuning recommendation history
    - custom_modes: User-defined generation modes
    - world_entity_scores: World entity quality scores
    - prompt_metrics: Prompt template usage tracking
    - generation_runs: Generation cost tracking

    Composed from focused mixins for maintainability.
    """

    pass


# Re-export for backward compatibility
__all__ = [
    "DEFAULT_DB_PATH",
    "ModeDatabase",
]

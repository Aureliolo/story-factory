"""Model mode service package - manages generation modes, scoring, and adaptive learning.

This package provides the ModelModeService class composed from specialized mixins:
- ModelModeServiceBase: Core functionality and initialization
- ModesMixin: Mode management operations
- VramMixin: VRAM management operations
- ScoringMixin: Score recording operations
- QualityMixin: Quality judging operations
- LearningMixin: Learning and tuning operations
- AnalyticsMixin: Analytics operations
"""

from ._analytics import AnalyticsMixin
from ._base import ModelModeServiceBase
from ._learning import LearningMixin
from ._modes import ModesMixin
from ._quality import QualityMixin
from ._scoring import ScoringMixin
from ._vram import VramMixin


class ModelModeService(
    ModesMixin,
    VramMixin,
    ScoringMixin,
    QualityMixin,
    LearningMixin,
    AnalyticsMixin,
    ModelModeServiceBase,
):
    """Service for managing generation modes and adaptive learning.

    This service coordinates:
    - Mode selection and customization
    - VRAM-aware model loading/unloading
    - Quality scoring via LLM judge
    - Performance tracking and aggregation
    - Tuning recommendations based on historical data

    Composed from:
    - ModelModeServiceBase: Core functionality and initialization
    - ModesMixin: set_mode(), list_modes(), save_custom_mode(), get_model_for_agent()
    - VramMixin: prepare_model(), _unload_all_except()
    - ScoringMixin: record_generation(), update_quality_scores(), record_implicit_signal()
    - QualityMixin: judge_quality(), calculate_consistency_score()
    - LearningMixin: get_recommendations(), apply_recommendation(), handle_recommendations()
    - AnalyticsMixin: get_model_performance(), get_pending_recommendations(), export_scores_csv()
    """

    pass


__all__ = [
    "ModelModeService",
]

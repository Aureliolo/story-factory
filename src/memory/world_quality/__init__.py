"""Quality score models for world building entities.

This package provides quality scoring models for the judge/refinement loop:
- _models: IterationRecord, RefinementHistory for tracking refinement progress
- _scoring: Quality score models per entity type (Character, Location, etc.)
- _config: RefinementConfig, JudgeConsistencyConfig, ScoreStatistics
"""

from ._config import JudgeConsistencyConfig, RefinementConfig, ScoreStatistics
from ._models import IterationRecord, RefinementHistory
from ._scoring import (
    CharacterQualityScores,
    ConceptQualityScores,
    FactionQualityScores,
    ItemQualityScores,
    LocationQualityScores,
    RelationshipQualityScores,
)

__all__ = [
    "CharacterQualityScores",
    "ConceptQualityScores",
    "FactionQualityScores",
    "ItemQualityScores",
    "IterationRecord",
    "JudgeConsistencyConfig",
    "LocationQualityScores",
    "RefinementConfig",
    "RefinementHistory",
    "RelationshipQualityScores",
    "ScoreStatistics",
]

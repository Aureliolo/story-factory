"""World Quality package - multi-model iteration for world building quality.

Implements a generate-judge-refine loop using:
- Creator model: High temperature (0.9) for creative generation
- Judge model: Low temperature (0.1) for consistent evaluation
- Refinement: Incorporates feedback to improve entities

This package provides the WorldQualityService class composed from specialized mixins:
- WorldQualityServiceBase: Core functionality, initialization, model selection
- CharacterMixin: Character generation with quality refinement
- LocationMixin: Location generation with quality refinement
- FactionMixin: Faction generation with quality refinement
- ItemMixin: Item generation with quality refinement
- ConceptMixin: Concept generation with quality refinement
- RelationshipMixin: Relationship generation with quality refinement
- BatchMixin: Batch generation for multiple entities
- UtilsMixin: Mini descriptions, entity operations
- ValidationMixin: Entity validation and consistency checking
"""

from ._base import EntityGenerationProgress, WorldQualityServiceBase
from ._batch import BatchMixin
from ._character import CharacterMixin
from ._concept import ConceptMixin
from ._faction import FactionMixin
from ._item import ItemMixin
from ._location import LocationMixin
from ._relationship import RelationshipMixin
from ._utils import UtilsMixin
from ._validation import ValidationMixin


class WorldQualityService(
    CharacterMixin,
    LocationMixin,
    FactionMixin,
    ItemMixin,
    ConceptMixin,
    RelationshipMixin,
    BatchMixin,
    UtilsMixin,
    ValidationMixin,
    WorldQualityServiceBase,
):
    """Service for quality-controlled world entity generation.

    Uses a multi-model iteration loop:
    1. Creator generates initial entity (high temperature)
    2. Judge evaluates quality (low temperature)
    3. If below threshold, refine with feedback and repeat
    4. Return entity with quality scores

    Composed from:
    - WorldQualityServiceBase: Core functionality, initialization, model selection
    - CharacterMixin: generate_character_with_quality(), _create_character(), etc.
    - LocationMixin: generate_location_with_quality(), _create_location(), etc.
    - FactionMixin: generate_faction_with_quality(), _create_faction(), etc.
    - ItemMixin: generate_item_with_quality(), _create_item(), etc.
    - ConceptMixin: generate_concept_with_quality(), _create_concept(), etc.
    - RelationshipMixin: generate_relationship_with_quality(), _create_relationship(), etc.
    - BatchMixin: generate_*_with_quality() batch methods
    - UtilsMixin: generate_mini_description(), refine_entity(), regenerate_entity()
    - ValidationMixin: suggest_relationships_for_entity(), validate_entity_consistency()
    """

    pass


__all__ = [
    "EntityGenerationProgress",
    "WorldQualityService",
]

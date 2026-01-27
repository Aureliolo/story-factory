"""World service package - handles world/entity management.

This package provides the WorldService class composed from specialized mixins:
- WorldServiceBase: Core functionality and settings
- BuildMixin: World building operations
- ExtractionMixin: Entity extraction from text
- EntityMixin: Entity CRUD operations
- RelationshipMixin: Relationship management
- GraphMixin: Graph analysis operations
- HealthMixin: World health detection
"""

from ._base import WorldBuildOptions, WorldBuildProgress, WorldServiceBase
from ._build import BuildMixin
from ._entities import EntityMixin
from ._extraction import ExtractionMixin
from ._graph import GraphMixin
from ._health import HealthMixin
from ._relationships import RelationshipMixin


class WorldService(
    BuildMixin,
    ExtractionMixin,
    EntityMixin,
    RelationshipMixin,
    GraphMixin,
    HealthMixin,
    WorldServiceBase,
):
    """World and entity management service.

    This service handles extraction of entities from story content,
    entity CRUD operations, and relationship management.

    Composed from:
    - WorldServiceBase: Core functionality and settings
    - BuildMixin: build_world(), _generate_* methods
    - ExtractionMixin: extract_entities_from_structure(), extract_from_chapter()
    - EntityMixin: add_entity(), update_entity(), delete_entity(), search_entities(), etc.
    - RelationshipMixin: add_relationship(), delete_relationship(), get_relationships()
    - GraphMixin: find_path(), get_communities(), get_most_connected(), etc.
    - HealthMixin: find_orphan_entities(), get_world_health_metrics()
    """

    pass


__all__ = [
    "WorldBuildOptions",
    "WorldBuildProgress",
    "WorldService",
]

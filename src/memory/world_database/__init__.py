"""SQLite-backed worldbuilding database with NetworkX integration.

This package provides the WorldDatabase class composed from focused mixins:
- WorldDatabaseBase: Core initialization and connection management
- EntityMixin: Entity CRUD operations
- VersionMixin: Entity versioning operations
- RelationshipMixin: Relationship CRUD operations
- EventMixin: Event management
- GraphMixin: NetworkX graph operations
- IOMixin: Export/Import and context for agents
"""

from src.memory.entities import Entity, EntityVersion

from ._base import (
    ENTITY_UPDATE_FIELDS,
    MAX_ATTRIBUTES_DEPTH,
    MAX_ATTRIBUTES_SIZE_BYTES,
    SCHEMA_VERSION,
    VALID_ENTITY_TYPES,
    WorldDatabaseBase,
    _check_nesting_depth,
    _flatten_deep_attributes,
    _validate_and_normalize_attributes,
)
from ._entities import EntityMixin
from ._events import EventMixin
from ._graph import GraphMixin
from ._io import IOMixin
from ._relationships import RelationshipMixin
from ._versions import VersionMixin


class WorldDatabase(
    EntityMixin,
    VersionMixin,
    RelationshipMixin,
    EventMixin,
    GraphMixin,
    IOMixin,
    WorldDatabaseBase,
):
    """SQLite-backed worldbuilding database with NetworkX integration.

    Thread-safe implementation using RLock for all database operations.
    Composed from focused mixins for maintainability.
    """

    pass


__all__ = [
    "ENTITY_UPDATE_FIELDS",
    "MAX_ATTRIBUTES_DEPTH",
    "MAX_ATTRIBUTES_SIZE_BYTES",
    "SCHEMA_VERSION",
    "VALID_ENTITY_TYPES",
    "Entity",
    "EntityVersion",
    "WorldDatabase",
    "_check_nesting_depth",
    "_flatten_deep_attributes",
    "_validate_and_normalize_attributes",
]

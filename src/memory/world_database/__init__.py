"""World database package.

This package contains the WorldDatabase class for world entity management
using SQLite + NetworkX.
The module has been moved to a package structure to support future modular splitting.

Current structure:
- _database.py: Main WorldDatabase class with all methods

Future planned split (from Issue #198):
- _base.py: Schema, init, context manager
- _entities.py: Entity CRUD
- _versions.py: Entity versioning
- _relationships.py: Relationship CRUD
- _events.py: Event management
- _graph.py: NetworkX operations
- _io.py: Export/import, context for agents
"""

from src.memory.world_database._database import (
    ENTITY_UPDATE_FIELDS,
    MAX_ATTRIBUTES_DEPTH,
    MAX_ATTRIBUTES_SIZE_BYTES,
    SCHEMA_VERSION,
    VALID_ENTITY_TYPES,
    WorldDatabase,
    _check_nesting_depth,
    _flatten_deep_attributes,
    _validate_and_normalize_attributes,
)

__all__ = [
    "ENTITY_UPDATE_FIELDS",
    "MAX_ATTRIBUTES_DEPTH",
    "MAX_ATTRIBUTES_SIZE_BYTES",
    "SCHEMA_VERSION",
    "VALID_ENTITY_TYPES",
    "WorldDatabase",
    "_check_nesting_depth",
    "_flatten_deep_attributes",
    "_validate_and_normalize_attributes",
]

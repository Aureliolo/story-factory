"""SQLite-backed worldbuilding database with NetworkX integration."""

import json
import logging
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from networkx import DiGraph

if TYPE_CHECKING:
    from src.memory.world_settings import WorldSettings

from src.memory.entities import Entity, EntityVersion, EventParticipant, Relationship, WorldEvent
from src.utils.exceptions import DatabaseClosedError, RelationshipValidationError

from . import (
    _cycles,
    _embeddings,
    _entities,
    _events,
    _graph,
    _io,
    _relationships,
    _schema,
    _settings,
    _versions,
)

logger = logging.getLogger(__name__)

# Schema version for new databases and migration target for existing ones
SCHEMA_VERSION = 7

# Valid entity types (whitelist)
VALID_ENTITY_TYPES = frozenset({"character", "location", "item", "faction", "concept"})

# Allowed fields for entity updates (SQL injection prevention)
ENTITY_UPDATE_FIELDS = frozenset({"name", "description", "attributes", "type", "updated_at"})

# Attributes constraints
MAX_ATTRIBUTES_DEPTH = 3
MAX_ATTRIBUTES_SIZE_BYTES = 10 * 1024  # 10KB


def flatten_deep_attributes(
    obj: Any, max_depth: int = MAX_ATTRIBUTES_DEPTH, current_depth: int = 0
) -> Any:
    """Flatten attributes that exceed max nesting depth.

    Converts deeply nested structures to JSON string representations at the max depth
    to preserve data while meeting storage constraints.

    Args:
        obj: Object to potentially flatten.
        max_depth: Maximum nesting depth allowed.
        current_depth: Current depth in the recursion.

    Returns:
        Flattened object with nested structures converted to strings at max depth.
    """
    if current_depth >= max_depth:
        # At max depth, convert complex types to string representation
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        # Use deterministic JSON representation for nested structures
        logger.debug(f"Flattening nested {type(obj).__name__} at depth {current_depth}")
        try:
            return json.dumps(obj, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Failed to JSON-serialize %s at max depth; falling back to str(): %s",
                type(obj).__name__,
                exc,
            )
            return str(obj)

    if isinstance(obj, dict):
        return {k: flatten_deep_attributes(v, max_depth, current_depth + 1) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [flatten_deep_attributes(item, max_depth, current_depth + 1) for item in obj]
    return obj


def check_nesting_depth(obj: Any, max_depth: int, current_depth: int = 0) -> bool:
    """Check if object exceeds maximum nesting depth.

    Args:
        obj: Object to check.
        max_depth: Maximum allowed nesting depth.
        current_depth: Current depth in the recursion.

    Returns:
        True if object exceeds max depth, False otherwise.
    """
    if current_depth > max_depth:
        return True
    if isinstance(obj, dict):
        return any(check_nesting_depth(v, max_depth, current_depth + 1) for v in obj.values())
    elif isinstance(obj, list):
        return any(check_nesting_depth(item, max_depth, current_depth + 1) for item in obj)
    return False


def validate_and_normalize_attributes(
    attrs: dict[str, Any], max_depth: int = MAX_ATTRIBUTES_DEPTH
) -> dict[str, Any]:
    """Validate and normalize attributes dict.

    Flattens deeply nested structures and validates size constraints.

    Args:
        attrs: Attributes dictionary to validate.
        max_depth: Maximum nesting depth allowed.

    Returns:
        Normalized attributes dict with deep nesting flattened.

    Raises:
        ValueError: If attributes exceed size limit.
    """
    # Check if flattening is needed and flatten
    if check_nesting_depth(attrs, max_depth, current_depth=1):
        logger.warning(
            f"Attributes exceed maximum nesting depth of {max_depth}, flattening deep structures"
        )
        attrs = flatten_deep_attributes(attrs, max_depth, current_depth=1)

    # Check size (after flattening)
    attrs_json = json.dumps(attrs)
    if len(attrs_json.encode("utf-8")) > MAX_ATTRIBUTES_SIZE_BYTES:
        raise ValueError(f"Attributes exceed maximum size of {MAX_ATTRIBUTES_SIZE_BYTES // 1024}KB")

    return attrs


class WorldDatabase:
    """SQLite-backed worldbuilding database with NetworkX integration.

    Thread-safe implementation using RLock for all database operations.
    """

    def __init__(self, db_path: Path | str):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread safety lock
        self._lock = threading.RLock()

        # Connection with WAL mode for better concurrency
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._closed = False  # Initialize immediately so __del__ can always clean up
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")

        # Load sqlite-vec extension — mandatory for RAG pipeline
        from src.utils.sqlite_vec_loader import load_vec_extension

        self._vec_available: bool = load_vec_extension(self.conn)
        if not self._vec_available:
            self.conn.close()
            self._closed = True
            raise RuntimeError(
                "sqlite-vec extension failed to load. Install it with: pip install sqlite-vec"
            )

        self._init_schema()
        self._graph: DiGraph[Any] | None = None

        # Optional callback for embedding service to hook into content changes.
        # Set via attach_content_changed_callback().
        self._on_content_changed: Callable[[str, str, str], None] | None = None

        # Optional callback for embedding deletions.
        self._on_content_deleted: Callable[[str], None] | None = None

    def __del__(self) -> None:
        """Safety net for resource cleanup."""
        if hasattr(self, "_closed") and not self._closed:
            try:
                self.close()
            except Exception as e:
                # Log but don't raise during garbage collection
                logger.debug("Error during WorldDatabase cleanup in __del__: %s", e)

    def _init_schema(self) -> None:
        """Initialize database schema with versioning (delegated to _schema)."""
        _schema.init_schema(self)

    def _ensure_open(self) -> None:
        """Check that the database connection is still open.

        Raises:
            DatabaseClosedError: If the database has been closed.
        """
        if self._closed:
            raise DatabaseClosedError(f"Database connection is closed: {self.db_path}")

    def close(self) -> None:
        """Close database connection."""
        with self._lock:
            if self.conn and not self._closed:
                self.conn.close()
                self._closed = True
                logger.debug(f"Database connection closed: {self.db_path}")

    def __enter__(self) -> WorldDatabase:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False]:
        """Context manager exit - ensures connection is closed."""
        self.close()
        return False  # Don't suppress exceptions

    # =========================================================================
    # Entity CRUD Operations (delegated to _entities)
    # =========================================================================

    def add_entity(
        self,
        entity_type: str,
        name: str,
        description: str = "",
        attributes: dict[str, Any] | None = None,
    ) -> str:
        """Add a new entity to the world database.

        Args:
            entity_type: Type of entity (e.g. 'character', 'location').
            name: Display name for the entity.
            description: Optional description text.
            attributes: Optional key-value attributes dict.

        Returns:
            The generated entity ID.
        """
        return _entities.add_entity(self, entity_type, name, description, attributes)

    def get_entity(self, entity_id: str) -> Entity | None:
        """Retrieve an entity by its ID.

        Args:
            entity_id: The entity's unique identifier.

        Returns:
            Entity instance or None if not found.
        """
        return _entities.get_entity(self, entity_id)

    def get_entity_by_name(self, name: str, entity_type: str | None = None) -> Entity | None:
        """Retrieve an entity by name, optionally filtering by type.

        Args:
            name: Entity name to search for.
            entity_type: Optional type filter.

        Returns:
            Entity instance or None if not found.
        """
        return _entities.get_entity_by_name(self, name, entity_type)

    def update_entity(self, entity_id: str, **updates: Any) -> bool:
        """Update an existing entity's fields.

        Args:
            entity_id: The entity's unique identifier.
            **updates: Field names and new values to apply.

        Returns:
            True if the entity was updated, False if not found.
        """
        return _entities.update_entity(self, entity_id, **updates)

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and its associated data.

        Args:
            entity_id: The entity's unique identifier.

        Returns:
            True if the entity was deleted, False if not found.
        """
        return _entities.delete_entity(self, entity_id)

    def list_entities(self, entity_type: str | None = None) -> list[Entity]:
        """List all entities, optionally filtered by type.

        Args:
            entity_type: Optional type filter (e.g. 'character').

        Returns:
            List of Entity instances.
        """
        return _entities.list_entities(self, entity_type)

    def count_entities(self, entity_type: str | None = None) -> int:
        """Count entities, optionally filtered by type.

        Args:
            entity_type: Optional type filter.

        Returns:
            Number of matching entities.
        """
        return _entities.count_entities(self, entity_type)

    def search_entities(self, query: str, entity_type: str | None = None) -> list[Entity]:
        """Search entities by name or description text.

        Args:
            query: Search query string.
            entity_type: Optional type filter.

        Returns:
            List of matching Entity instances.
        """
        return _entities.search_entities(self, query, entity_type)

    # =========================================================================
    # Entity Versioning Operations (delegated to _versions)
    # =========================================================================

    def save_entity_version(
        self,
        entity_id: str,
        change_type: str,
        change_reason: str = "",
        quality_score: float | None = None,
    ) -> str | None:
        """Save a versioned snapshot of an entity's current state.

        Args:
            entity_id: The entity to snapshot.
            change_type: Type of change ('created', 'refined', 'edited', or 'regenerated').
            change_reason: Optional human-readable reason for the change.
            quality_score: Optional quality score at time of snapshot.

        Returns:
            The version ID, or None if the entity was not found.
        """
        return _versions.save_entity_version(
            self, entity_id, change_type, change_reason, quality_score
        )

    def get_entity_versions(self, entity_id: str, limit: int | None = None) -> list[EntityVersion]:
        """Get version history for an entity.

        Args:
            entity_id: The entity's unique identifier.
            limit: Optional maximum number of versions to return.

        Returns:
            List of EntityVersion instances, newest first.
        """
        return _versions.get_entity_versions(self, entity_id, limit)

    def revert_entity_to_version(self, entity_id: str, version_number: int) -> bool:
        """Revert an entity to a previous version.

        Args:
            entity_id: The entity's unique identifier.
            version_number: The version number to revert to.

        Returns:
            True if reverted successfully.

        Raises:
            ValueError: If the version or entity is not found.
        """
        return _versions.revert_entity_to_version(self, entity_id, version_number)

    # =========================================================================
    # Relationship CRUD Operations (delegated to _relationships)
    # =========================================================================

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        description: str = "",
        strength: float = 1.0,
        bidirectional: bool = False,
        attributes: dict[str, Any] | None = None,
        validate: bool = True,
    ) -> str:
        """Add a relationship between two entities.

        Args:
            source_id: Source entity ID.
            target_id: Target entity ID.
            relation_type: Type of relationship (e.g. 'ally_of', 'parent_of').
            description: Optional description of the relationship.
            strength: Relationship strength from 0.0 to 1.0.
            bidirectional: Whether the relationship goes both ways.
            attributes: Optional key-value attributes dict.
            validate: Whether to validate that both entities exist.

        Returns:
            The generated relationship ID.
        """
        return _relationships.add_relationship(
            self,
            source_id,
            target_id,
            relation_type,
            description,
            strength,
            bidirectional,
            attributes,
            validate,
        )

    def get_relationships(self, entity_id: str, direction: str = "both") -> list[Relationship]:
        """Get all relationships for an entity.

        Args:
            entity_id: The entity's unique identifier.
            direction: Filter direction — 'outgoing', 'incoming', or 'both'.

        Returns:
            List of Relationship instances.
        """
        return _relationships.get_relationships(self, entity_id, direction)

    def get_relationship_between(self, source_id: str, target_id: str) -> Relationship | None:
        """Get the relationship between two specific entities.

        Args:
            source_id: Source entity ID.
            target_id: Target entity ID.

        Returns:
            Relationship instance or None if no relationship exists.
        """
        return _relationships.get_relationship_between(self, source_id, target_id)

    def delete_relationship(self, rel_id: str) -> bool:
        """Delete a relationship by its ID.

        Args:
            rel_id: The relationship's unique identifier.

        Returns:
            True if the relationship was deleted, False if not found.
        """
        return _relationships.delete_relationship(self, rel_id)

    def update_relationship(
        self,
        relationship_id: str,
        relation_type: str | None = None,
        description: str | None = None,
        strength: float | None = None,
        bidirectional: bool | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> bool:
        """Update an existing relationship's fields.

        Args:
            relationship_id: The relationship's unique identifier.
            relation_type: New relation type, or None to keep current.
            description: New description, or None to keep current.
            strength: New strength value, or None to keep current.
            bidirectional: New bidirectional flag, or None to keep current.
            attributes: New attributes dict, or None to keep current.

        Returns:
            True if the relationship was updated, False if not found.
        """
        return _relationships.update_relationship(
            self, relationship_id, relation_type, description, strength, bidirectional, attributes
        )

    def list_relationships(self) -> list[Relationship]:
        """List all relationships in the world database.

        Returns:
            List of all Relationship instances.
        """
        return _relationships.list_relationships(self)

    # =========================================================================
    # Event Management (delegated to _events)
    # =========================================================================

    def add_event(
        self,
        description: str,
        participants: list[tuple[str, str]] | None = None,
        chapter_number: int | None = None,
        timestamp_in_story: str = "",
        consequences: list[str] | None = None,
        attributes: dict | None = None,
    ) -> str:
        """Add a world event.

        Args:
            description: What happened in the event.
            participants: Optional list of (entity_id, role) tuples.
            chapter_number: Optional chapter where the event occurs.
            timestamp_in_story: Optional in-story timestamp.
            consequences: Optional list of consequence descriptions.
            attributes: Optional metadata dict (e.g. quality_scores).

        Returns:
            The generated event ID.
        """
        return _events.add_event(
            self,
            description,
            participants,
            chapter_number,
            timestamp_in_story,
            consequences,
            attributes,
        )

    def get_events_for_entity(self, entity_id: str) -> list[WorldEvent]:
        """Get all events involving a specific entity.

        Args:
            entity_id: The entity's unique identifier.

        Returns:
            List of WorldEvent instances.
        """
        return _events.get_events_for_entity(self, entity_id)

    def get_events_for_chapter(self, chapter_number: int) -> list[WorldEvent]:
        """Get all events in a specific chapter.

        Args:
            chapter_number: The chapter number to filter by.

        Returns:
            List of WorldEvent instances.
        """
        return _events.get_events_for_chapter(self, chapter_number)

    def get_event_participants(self, event_id: str) -> list[EventParticipant]:
        """Get all participants of a specific event.

        Args:
            event_id: The event's unique identifier.

        Returns:
            List of EventParticipant instances.
        """
        return _events.get_event_participants(self, event_id)

    def list_events(self, limit: int | None = None) -> list[WorldEvent]:
        """List all world events.

        Args:
            limit: Optional maximum number of events to return.

        Returns:
            List of WorldEvent instances.
        """
        return _events.list_events(self, limit)

    # =========================================================================
    # NetworkX Graph Operations (delegated to _graph)
    # =========================================================================

    def _invalidate_graph(self) -> None:
        """Invalidate the cached NetworkX graph, forcing a rebuild on next access."""
        _graph.invalidate_graph(self)

    def invalidate_graph_cache(self) -> None:
        """Public API to invalidate the graph cache after external data changes."""
        _graph.invalidate_graph_cache(self)

    def _update_entity_in_graph(
        self,
        entity_id: str,
        name: str | None = None,
        entity_type: str | None = None,
        description: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Update a single entity's attributes in the cached graph without full rebuild.

        Args:
            entity_id: The entity's unique identifier.
            name: New name, or None to keep current.
            entity_type: New type, or None to keep current.
            description: New description, or None to keep current.
            attributes: New attributes dict, or None to keep current.
        """
        _graph.update_entity_in_graph(self, entity_id, name, entity_type, description, attributes)

    def get_graph(self) -> DiGraph[Any]:
        """Get the NetworkX directed graph representation of the world.

        Builds the graph from the database on first access, then caches it.

        Returns:
            NetworkX DiGraph with entities as nodes and relationships as edges.
        """
        return _graph.get_graph(self)

    def find_path(self, source_id: str, target_id: str) -> list[str]:
        """Find the shortest path between two entities in the graph.

        Args:
            source_id: Starting entity ID.
            target_id: Destination entity ID.

        Returns:
            List of entity IDs forming the path, or empty list if no path exists.
        """
        return _graph.find_path(self, source_id, target_id)

    def find_all_paths(
        self, source_id: str, target_id: str, max_length: int = 5
    ) -> list[list[str]]:
        """Find all simple paths between two entities up to a maximum length.

        Args:
            source_id: Starting entity ID.
            target_id: Destination entity ID.
            max_length: Maximum path length (number of hops).

        Returns:
            List of paths, where each path is a list of entity IDs.
        """
        return _graph.find_all_paths(self, source_id, target_id, max_length)

    def get_connected_entities(self, entity_id: str, max_depth: int = 2) -> list[Entity]:
        """Get all entities connected to a given entity within a depth limit.

        Args:
            entity_id: The entity's unique identifier.
            max_depth: Maximum traversal depth from the starting entity.

        Returns:
            List of connected Entity instances.
        """
        return _graph.get_connected_entities(self, entity_id, max_depth)

    def get_communities(self) -> list[list[str]]:
        """Detect communities (clusters) of related entities in the graph.

        Returns:
            List of communities, where each community is a list of entity IDs.
        """
        return _graph.get_communities(self)

    def get_entity_centrality(self) -> dict[str, float]:
        """Calculate centrality scores for all entities in the graph.

        Returns:
            Dict mapping entity ID to its centrality score.
        """
        return _graph.get_entity_centrality(self)

    def get_most_connected(self, limit: int = 10) -> list[tuple[Entity, int]]:
        """Get the most connected entities by relationship count.

        Args:
            limit: Maximum number of entities to return.

        Returns:
            List of (Entity, connection_count) tuples, sorted by count descending.
        """
        return _graph.get_most_connected(self, limit)

    def find_orphans(self, entity_type: str | None = None) -> list[Entity]:
        """Find entities with no relationships (orphans).

        Args:
            entity_type: Optional type filter.

        Returns:
            List of orphan Entity instances.
        """
        return _graph.find_orphans(self, entity_type)

    def find_circular_relationships(
        self,
        relation_types: list[str] | None = None,
        max_cycle_length: int = 10,
    ) -> list[list[tuple[str, str, str]]]:
        """Detect circular relationship chains in the graph.

        Args:
            relation_types: Optional list of relation types to check.
            max_cycle_length: Maximum cycle length to detect.

        Returns:
            List of cycles, where each cycle is a list of
            (source_id, relation_type, target_id) edge tuples.
        """
        return _graph.find_circular_relationships(self, relation_types, max_cycle_length)

    # =========================================================================
    # Accepted Cycles (delegated to _cycles)
    # =========================================================================

    @staticmethod
    def compute_cycle_hash(cycle_edges: list[tuple[str, str, str]]) -> str:
        """Compute a deterministic 16-char hex hash for a cycle.

        Edges are sorted lexicographically so any permutation of the same
        edge set (including rotations) produces the same hash.

        Args:
            cycle_edges: List of (source_id, relation_type, target_id) tuples.
        """
        return _cycles.compute_cycle_hash(cycle_edges)

    def accept_cycle(self, cycle_hash: str) -> None:
        """Mark a circular chain as accepted/intentional.

        Args:
            cycle_hash: Hash of the cycle (from compute_cycle_hash).

        Raises:
            ValueError: If cycle_hash is not exactly 16 hex characters.
        """
        _cycles.accept_cycle(self, cycle_hash)

    def remove_accepted_cycle(self, cycle_hash: str) -> bool:
        """Remove a previously accepted cycle.

        Args:
            cycle_hash: Hash of the cycle to un-accept.

        Returns:
            True if the cycle was removed, False if it wasn't found.

        Raises:
            ValueError: If cycle_hash is not exactly 16 hex characters.
        """
        return _cycles.remove_accepted_cycle(self, cycle_hash)

    def get_accepted_cycles(self) -> set[str]:
        """Get all accepted cycle hashes.

        Returns:
            Set of accepted cycle hash strings.
        """
        return _cycles.get_accepted_cycles(self)

    # =========================================================================
    # Export/Import (delegated to _io)
    # =========================================================================

    def export_to_json(self) -> dict[str, Any]:
        """Export the entire world database to a JSON-serializable dict.

        Returns:
            Dict containing all entities, relationships, events, and settings.
        """
        return _io.export_to_json(self)

    def import_from_json(self, data: dict[str, Any]) -> None:
        """Import world data from a JSON dict, merging into the current database.

        Args:
            data: Dict containing entities, relationships, events, and settings.
        """
        _io.import_from_json(self, data)

    # =========================================================================
    # World Settings (calendar, timeline config)
    # =========================================================================

    def get_world_settings(self) -> WorldSettings | None:
        """Get world settings including calendar configuration.

        Returns:
            WorldSettings instance or None if not configured.
        """
        return _settings.get_world_settings(self)

    @property
    def vec_available(self) -> bool:
        """Whether the sqlite-vec extension is loaded and vector search is available.

        Always True on a live instance since __init__ raises RuntimeError if
        sqlite-vec fails to load.  Retained for interface stability.
        """
        return self._vec_available

    def attach_content_changed_callback(
        self, callback: Callable[[str, str, str], None] | None
    ) -> None:
        """Register a callback to be invoked when content changes.

        The callback receives (source_id, content_type, text_to_embed) and is
        called after entity/relationship/event CRUD operations commit.

        Args:
            callback: Callable with signature (str, str, str) -> None, or None to detach.
        """
        self._on_content_changed = callback
        logger.debug("Content changed callback %s", "attached" if callback else "detached")

    def attach_content_deleted_callback(self, callback: Callable[[str], None] | None) -> None:
        """Register a callback to be invoked when content is deleted.

        The callback receives the source_id string and is called after delete operations.

        Args:
            callback: Callable with signature (str,) -> None, or None to detach.
        """
        self._on_content_deleted = callback
        logger.debug("Content deleted callback %s", "attached" if callback else "detached")

    # =========================================================================
    # Embedding Operations (delegated to _embeddings)
    # =========================================================================

    def upsert_embedding(
        self,
        source_id: str,
        content_type: str,
        text: str,
        embedding: list[float],
        model: str,
        entity_type: str = "",
        chapter_number: int | None = None,
    ) -> bool:
        """Insert or update a vector embedding for a content source.

        Args:
            source_id: Identifier for the content being embedded.
            content_type: Type of content (e.g. 'entity', 'relationship').
            text: The text that was embedded (stored for display).
            embedding: The vector embedding as a list of floats.
            model: Name of the embedding model used.
            entity_type: Optional entity type for filtering.
            chapter_number: Optional chapter number for filtering.

        Returns:
            True if the embedding was inserted/updated successfully.
        """
        return _embeddings.upsert_embedding(
            self, source_id, content_type, text, embedding, model, entity_type, chapter_number
        )

    def delete_embedding(self, source_id: str) -> bool:
        """Delete the embedding for a content source.

        Args:
            source_id: Identifier for the content whose embedding to delete.

        Returns:
            True if the embedding was deleted, False if not found.
        """
        return _embeddings.delete_embedding(self, source_id)

    def search_similar(
        self,
        query_embedding: list[float],
        k: int = 10,
        content_type: str | None = None,
        entity_type: str | None = None,
        chapter_number: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search for content with similar vector embeddings (KNN search).

        Args:
            query_embedding: The query vector to search against.
            k: Maximum number of results to return.
            content_type: Optional filter by content type.
            entity_type: Optional filter by entity type.
            chapter_number: Optional filter by chapter number.

        Returns:
            List of result dicts with source_id, distance, and metadata.
        """
        return _embeddings.search_similar(
            self, query_embedding, k, content_type, entity_type, chapter_number
        )

    def get_embedding_stats(self) -> dict[str, Any]:
        """Get statistics about stored embeddings.

        Returns:
            Dict with counts by content type, model info, and total count.
        """
        return _embeddings.get_embedding_stats(self)

    def needs_reembedding(self, current_model: str) -> bool:
        """Check if existing embeddings were created with a different model.

        Args:
            current_model: The embedding model currently configured.

        Returns:
            True if any embeddings use a different model and need regeneration.
        """
        return _embeddings.needs_reembedding(self, current_model)

    def clear_all_embeddings(self) -> None:
        """Delete all vector embeddings and their metadata."""
        _embeddings.clear_all_embeddings(self)

    def recreate_vec_table(self, dimensions: int) -> None:
        """Drop and recreate the vec_embeddings virtual table with new dimensions.

        Args:
            dimensions: Number of dimensions for the new embedding vectors.
        """
        _embeddings.recreate_vec_table(self, dimensions)

    def save_world_settings(self, settings: WorldSettings) -> None:
        """Save or update world settings.

        Args:
            settings: WorldSettings instance to save.
        """
        _settings.save_world_settings(self, settings)


# Re-export for backward compatibility
__all__ = [
    "ENTITY_UPDATE_FIELDS",
    "MAX_ATTRIBUTES_DEPTH",
    "MAX_ATTRIBUTES_SIZE_BYTES",
    "SCHEMA_VERSION",
    "VALID_ENTITY_TYPES",
    "Entity",
    "EntityVersion",
    "EventParticipant",
    "Relationship",
    "RelationshipValidationError",
    "WorldDatabase",
    "WorldEvent",
    "check_nesting_depth",
    "flatten_deep_attributes",
    "validate_and_normalize_attributes",
]

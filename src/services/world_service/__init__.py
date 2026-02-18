"""World service - handles world/entity management."""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.memory.entities import Entity, Relationship
from src.memory.story_state import StoryState
from src.memory.templates import WorldTemplate
from src.memory.world_database import WorldDatabase
from src.settings import Settings

from . import _build, _entities, _extraction, _graph, _health

if TYPE_CHECKING:
    from src.memory.world_health import WorldHealthMetrics
    from src.services import ServiceContainer

logger = logging.getLogger(__name__)

_HEALTH_CACHE_TTL_SECONDS = 30


@dataclass
class WorldBuildProgress:
    """Progress information for world building operations."""

    step: int
    total_steps: int
    message: str
    entity_type: str | None = None
    count: int = 0


@dataclass
class WorldBuildOptions:
    """Options for world building operations.

    Attributes:
        clear_existing: Whether to clear existing world data first.
        generate_calendar: Whether to generate a calendar system.
        generate_structure: Whether to generate story structure (characters, chapters).
        generate_locations: Whether to generate location entities.
        generate_factions: Whether to generate faction entities.
        generate_items: Whether to generate item entities.
        generate_concepts: Whether to generate concept entities.
        generate_relationships: Whether to generate relationships between entities.
        cancellation_event: Optional threading.Event to signal cancellation.
        world_template: Optional world template for genre-specific hints.
    """

    clear_existing: bool = False
    generate_calendar: bool = True
    generate_structure: bool = True
    generate_locations: bool = True
    generate_factions: bool = True
    generate_items: bool = True
    generate_concepts: bool = True
    generate_events: bool = True
    generate_relationships: bool = True
    cancellation_event: threading.Event | None = field(default=None, repr=False)
    world_template: WorldTemplate | None = field(default=None, repr=False)

    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self.cancellation_event is not None and self.cancellation_event.is_set()

    @classmethod
    def full(
        cls,
        cancellation_event: threading.Event | None = None,
        world_template: WorldTemplate | None = None,
    ) -> WorldBuildOptions:
        """Create options for full world build (everything, keeping existing data)."""
        return cls(
            clear_existing=False,
            generate_calendar=True,
            generate_structure=True,
            generate_locations=True,
            generate_factions=True,
            generate_items=True,
            generate_concepts=True,
            generate_events=True,
            generate_relationships=True,
            cancellation_event=cancellation_event,
            world_template=world_template,
        )

    @classmethod
    def full_rebuild(
        cls,
        cancellation_event: threading.Event | None = None,
        world_template: WorldTemplate | None = None,
    ) -> WorldBuildOptions:
        """Create options for full world rebuild (everything, clearing existing first)."""
        return cls(
            clear_existing=True,
            generate_calendar=True,
            generate_structure=True,
            generate_locations=True,
            generate_factions=True,
            generate_items=True,
            generate_concepts=True,
            generate_events=True,
            generate_relationships=True,
            cancellation_event=cancellation_event,
            world_template=world_template,
        )


class WorldService:
    """World and entity management service.

    This service handles extraction of entities from story content,
    entity CRUD operations, and relationship management.
    """

    def __init__(self, settings: Settings | None = None):
        """Initialize WorldService.

        Args:
            settings: Application settings. If None, loads from src/settings.json.
        """
        logger.debug("Initializing WorldService")
        self.settings = settings or Settings.load()

        # TTL cache for health metrics (avoids redundant recomputation on page reload)
        self._health_cache: WorldHealthMetrics | None = None
        self._health_cache_key: tuple[str, float] | None = None  # (db_path, threshold)
        self._health_cache_time: float = 0.0
        self._health_cache_lock = threading.RLock()

        logger.debug("WorldService initialized successfully")

    # ========== UNIFIED WORLD BUILDING ==========

    def build_world(
        self,
        state: StoryState,
        world_db: WorldDatabase,
        services: ServiceContainer,
        options: WorldBuildOptions | None = None,
        progress_callback: Callable[[WorldBuildProgress], None] | None = None,
    ) -> dict[str, int]:
        """Build/rebuild world with unified logic.

        This is the single entry point for all world building operations.
        Both "Build Story Structure" and "Rebuild World" use this method
        with different options.

        Args:
            state: Story state with completed brief.
            world_db: WorldDatabase to populate.
            services: ServiceContainer for accessing other services.
            options: World build options. Defaults to minimal build.
            progress_callback: Optional callback for progress updates.

        Returns:
            Dictionary with counts of generated entities by type.

        Raises:
            ValueError: If no brief exists.
            WorldGenerationError: If generation fails.
        """
        result = _build.build_world(self, state, world_db, services, options, progress_callback)
        self.invalidate_health_cache()
        return result

    # ========== ENTITY EXTRACTION ==========

    def extract_from_chapter(
        self,
        content: str,
        world_db: WorldDatabase,
        chapter_number: int,
    ) -> dict[str, int]:
        """Extract new entities and events from chapter content.

        Args:
            content: Chapter text content.
            world_db: WorldDatabase to update.
            chapter_number: The chapter number for event tracking.

        Returns:
            Dictionary with counts of extracted items.
        """
        result = _extraction.extract_from_chapter(self, content, world_db, chapter_number)
        self.invalidate_health_cache()
        return result

    # ========== ENTITY CRUD ==========

    def add_entity(
        self,
        world_db: WorldDatabase,
        entity_type: str,
        name: str,
        description: str = "",
        attributes: dict[str, Any] | None = None,
    ) -> str:
        """Add a new entity to the world.

        Args:
            world_db: WorldDatabase instance.
            entity_type: Type of entity.
            name: Entity name.
            description: Entity description.
            attributes: Optional attributes dictionary.

        Returns:
            The new entity's ID.
        """
        result = _entities.add_entity(self, world_db, entity_type, name, description, attributes)
        self.invalidate_health_cache()
        return result

    def update_entity(
        self,
        world_db: WorldDatabase,
        entity_id: str,
        name: str | None = None,
        description: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> bool:
        """Update an existing entity.

        Args:
            world_db: WorldDatabase instance.
            entity_id: Entity ID to update.
            name: New name (optional).
            description: New description (optional).
            attributes: New attributes (optional, merged with existing).

        Returns:
            True if updated, False if not found.
        """
        result = _entities.update_entity(self, world_db, entity_id, name, description, attributes)
        self.invalidate_health_cache()
        return result

    def delete_entity(self, world_db: WorldDatabase, entity_id: str) -> bool:
        """Delete an entity and its relationships.

        Args:
            world_db: WorldDatabase instance.
            entity_id: Entity ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        result = _entities.delete_entity(self, world_db, entity_id)
        self.invalidate_health_cache()
        return result

    def get_entity(self, world_db: WorldDatabase, entity_id: str) -> Entity | None:
        """Get an entity by ID.

        Args:
            world_db: WorldDatabase instance.
            entity_id: Entity ID.

        Returns:
            Entity or None if not found.
        """
        return _entities.get_entity(self, world_db, entity_id)

    def get_entity_versions(
        self, world_db: WorldDatabase, entity_id: str, limit: int | None = None
    ) -> list:
        """Get version history for an entity.

        Args:
            world_db: WorldDatabase instance.
            entity_id: Entity ID.
            limit: Maximum number of versions to return.

        Returns:
            List of EntityVersion objects, newest first.
        """
        return _entities.get_entity_versions(self, world_db, entity_id, limit)

    def revert_entity_to_version(
        self, world_db: WorldDatabase, entity_id: str, version_number: int
    ) -> bool:
        """Revert an entity to a previous version.

        Args:
            world_db: WorldDatabase instance.
            entity_id: Entity ID.
            version_number: Version number to revert to.

        Returns:
            True if reverted successfully.

        Raises:
            ValueError: If version not found or entity not found.
        """
        result = _entities.revert_entity_to_version(self, world_db, entity_id, version_number)
        self.invalidate_health_cache()
        return result

    def list_entities(
        self,
        world_db: WorldDatabase,
        entity_type: str | None = None,
    ) -> list[Entity]:
        """List all entities, optionally filtered by type.

        Args:
            world_db: WorldDatabase instance.
            entity_type: Optional type filter.

        Returns:
            List of Entity objects.
        """
        return _entities.list_entities(self, world_db, entity_type)

    def search_entities(
        self,
        world_db: WorldDatabase,
        query: str,
        entity_type: str | None = None,
    ) -> list[Entity]:
        """Search entities by name or description.

        Args:
            world_db: WorldDatabase instance.
            query: Search query.
            entity_type: Optional type filter.

        Returns:
            List of matching Entity objects.
        """
        return _entities.search_entities(self, world_db, query, entity_type)

    def find_entity_by_name(
        self,
        world_db: WorldDatabase,
        name: str,
        entity_type: str | None = None,
        fuzzy_threshold: float = 0.8,
    ) -> Entity | None:
        """Find an entity by name with fuzzy matching support.

        Attempts exact match first, then case-insensitive, then fuzzy matching
        if threshold is met.

        Args:
            world_db: WorldDatabase instance.
            name: Entity name to find.
            entity_type: Optional type filter.
            fuzzy_threshold: Similarity threshold for fuzzy matching (0.0-1.0).
                Lower values allow more lenient matching. Default 0.8.

        Returns:
            Entity if found, None otherwise.
        """
        return _entities.find_entity_by_name(self, world_db, name, entity_type, fuzzy_threshold)

    # ========== RELATIONSHIP MANAGEMENT ==========

    def add_relationship(
        self,
        world_db: WorldDatabase,
        source_id: str,
        target_id: str,
        relation_type: str,
        description: str = "",
        bidirectional: bool = False,
    ) -> str:
        """Add a relationship between entities.

        Args:
            world_db: WorldDatabase instance.
            source_id: Source entity ID.
            target_id: Target entity ID.
            relation_type: Type of relationship.
            description: Optional description.
            bidirectional: Whether relationship goes both ways.

        Returns:
            The new relationship's ID.
        """
        result = _graph.add_relationship(
            self, world_db, source_id, target_id, relation_type, description, bidirectional
        )
        self.invalidate_health_cache()
        return result

    def delete_relationship(self, world_db: WorldDatabase, relationship_id: str) -> bool:
        """Delete a relationship.

        Args:
            world_db: WorldDatabase instance.
            relationship_id: Relationship ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        result = _graph.delete_relationship(self, world_db, relationship_id)
        self.invalidate_health_cache()
        return result

    def get_relationships(
        self,
        world_db: WorldDatabase,
        entity_id: str | None = None,
    ) -> list[Relationship]:
        """Get relationships, optionally filtered by entity.

        Args:
            world_db: WorldDatabase instance.
            entity_id: Optional entity ID to filter by.

        Returns:
            List of Relationship objects.
        """
        return _graph.get_relationships(self, world_db, entity_id)

    # ========== GRAPH ANALYSIS ==========

    def find_path(
        self,
        world_db: WorldDatabase,
        source_id: str,
        target_id: str,
    ) -> list[str] | None:
        """Find shortest path between two entities.

        Args:
            world_db: WorldDatabase instance.
            source_id: Source entity ID.
            target_id: Target entity ID.

        Returns:
            List of entity IDs forming the path, or None if no path.
        """
        return _graph.find_path(self, world_db, source_id, target_id)

    def get_connected_entities(
        self,
        world_db: WorldDatabase,
        entity_id: str,
        max_depth: int = 2,
    ) -> list[Entity]:
        """Get entities connected to a given entity.

        Args:
            world_db: WorldDatabase instance.
            entity_id: Central entity ID.
            max_depth: Maximum relationship depth to traverse.

        Returns:
            List of connected Entity objects.
        """
        return _graph.get_connected_entities(self, world_db, entity_id, max_depth)

    def get_communities(self, world_db: WorldDatabase) -> list[list[str]]:
        """Detect communities/clusters in the entity graph.

        Args:
            world_db: WorldDatabase instance.

        Returns:
            List of communities, each a list of entity IDs.
        """
        return _graph.get_communities(self, world_db)

    def get_most_connected(
        self,
        world_db: WorldDatabase,
        limit: int = 10,
    ) -> list[tuple[Entity, int]]:
        """Get most connected entities (highest degree centrality).

        Args:
            world_db: WorldDatabase instance.
            limit: Maximum number to return.

        Returns:
            List of (Entity, connection_count) tuples.
        """
        return _graph.get_most_connected(self, world_db, limit)

    def get_entity_summary(self, world_db: WorldDatabase) -> dict[str, int]:
        """Get a summary count of entities by type.

        Args:
            world_db: WorldDatabase instance.

        Returns:
            Dictionary mapping entity type to count.
        """
        return _graph.get_entity_summary(self, world_db)

    # ========== PRIVATE EXTRACTION HELPERS ==========

    def _extract_locations_from_text(self, text: str) -> list[tuple[str, str]]:
        """Extract location names and descriptions from text."""
        return _extraction._extract_locations_from_text(self, text)

    @staticmethod
    def _calculate_name_similarity(name1: str, name2: str) -> float:
        """Calculate similarity between two names."""
        return _entities.calculate_name_similarity(name1, name2)

    # ========== PRIVATE BUILD HELPERS ==========
    # Delegation methods for internal build functions used by tests

    @staticmethod
    def _calculate_total_steps(
        options: WorldBuildOptions, *, generate_calendar: bool = False
    ) -> int:
        """Calculate total number of steps for progress reporting."""
        return _build._calculate_total_steps(options, generate_calendar=generate_calendar)

    @staticmethod
    def _clear_world_db(world_db: WorldDatabase) -> None:
        """Clear all entities and relationships from world database."""
        return _build._clear_world_db(world_db)

    @staticmethod
    def _extract_characters_to_world(state: StoryState, world_db: WorldDatabase) -> tuple[int, int]:
        """Extract characters and implicit relationships to world database."""
        return _build._extract_characters_to_world(state, world_db)

    def _generate_locations(
        self,
        state: StoryState,
        world_db: WorldDatabase,
        services: ServiceContainer,
        cancel_check: Callable[[], bool] | None = None,
    ) -> int:
        """Generate and add locations to world database."""
        return _build._generate_locations(
            self,
            state,
            world_db,
            services,
            cancel_check=cancel_check,
        )

    def _generate_factions(
        self,
        state: StoryState,
        world_db: WorldDatabase,
        services: ServiceContainer,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[int, int]:
        """Generate and add factions to world database.

        Returns:
            Tuple of (factions_added, implicit_relationships_added).
        """
        return _build._generate_factions(
            self,
            state,
            world_db,
            services,
            cancel_check=cancel_check,
        )

    def _generate_items(
        self,
        state: StoryState,
        world_db: WorldDatabase,
        services: ServiceContainer,
        cancel_check: Callable[[], bool] | None = None,
    ) -> int:
        """Generate and add items to world database."""
        return _build._generate_items(
            self,
            state,
            world_db,
            services,
            cancel_check=cancel_check,
        )

    def _generate_concepts(
        self,
        state: StoryState,
        world_db: WorldDatabase,
        services: ServiceContainer,
        cancel_check: Callable[[], bool] | None = None,
    ) -> int:
        """Generate and add concepts to world database."""
        return _build._generate_concepts(
            self,
            state,
            world_db,
            services,
            cancel_check=cancel_check,
        )

    def _generate_relationships(
        self,
        state: StoryState,
        world_db: WorldDatabase,
        services: ServiceContainer,
        cancel_check: Callable[[], bool] | None = None,
    ) -> int:
        """Generate and add relationships between entities."""
        return _build._generate_relationships(
            self,
            state,
            world_db,
            services,
            cancel_check=cancel_check,
        )

    def _generate_events(
        self,
        state: StoryState,
        world_db: WorldDatabase,
        services: ServiceContainer,
        cancel_check: Callable[[], bool] | None = None,
    ) -> int:
        """Generate and add world events to database."""
        return _build._generate_events(
            self,
            state,
            world_db,
            services,
            cancel_check=cancel_check,
        )

    # ========== WORLD HEALTH DETECTION ==========

    def find_orphan_entities(
        self,
        world_db: WorldDatabase,
        entity_type: str | None = None,
    ) -> list[Entity]:
        """Find entities with no relationships (orphans).

        Args:
            world_db: WorldDatabase instance.
            entity_type: Optional type filter.

        Returns:
            List of orphan entities.
        """
        return _health.find_orphan_entities(self, world_db, entity_type)

    def find_circular_relationships(
        self,
        world_db: WorldDatabase,
        relation_types: list[str] | None = None,
        max_cycle_length: int = 10,
    ) -> list[list[tuple[str, str, str]]]:
        """Find circular relationships (cycles) in the world.

        Args:
            world_db: WorldDatabase instance.
            relation_types: Optional list of relationship types to check.
            max_cycle_length: Maximum cycle length to detect.

        Returns:
            List of cycles. Each cycle is a list of (source_id, relation_type, target_id)
            tuples.
        """
        return _health.find_circular_relationships(self, world_db, relation_types, max_cycle_length)

    def get_world_health_metrics(
        self,
        world_db: WorldDatabase,
        quality_threshold: float = 6.0,
    ) -> WorldHealthMetrics:
        """Get comprehensive health metrics for a story world.

        Aggregates entity counts, orphan detection, circular relationships,
        quality scores, and computes overall health score. Results are cached
        for up to ``_HEALTH_CACHE_TTL_SECONDS`` to avoid redundant computation
        when the UI refreshes without data changes.

        Args:
            world_db: WorldDatabase instance.
            quality_threshold: Minimum quality score for "healthy" entities.

        Returns:
            WorldHealthMetrics object with all computed metrics.
        """
        cache_key = (str(world_db.db_path), quality_threshold)
        with self._health_cache_lock:
            now = time.monotonic()
            if (
                self._health_cache is not None
                and self._health_cache_key == cache_key
                and (now - self._health_cache_time) < _HEALTH_CACHE_TTL_SECONDS
            ):
                logger.debug("Health metrics cache hit (age=%.1fs)", now - self._health_cache_time)
                return self._health_cache

            logger.debug("Health metrics cache miss — recomputing")
            result = _health.get_world_health_metrics(self, world_db, quality_threshold)

            self._health_cache = result
            self._health_cache_key = cache_key
            self._health_cache_time = time.monotonic()
            return result

    def invalidate_health_cache(self) -> None:
        """Clear the cached health metrics so the next call recomputes."""
        with self._health_cache_lock:
            self._health_cache = None
            self._health_cache_key = None
            self._health_cache_time = 0.0
            logger.debug("Health metrics cache invalidated")

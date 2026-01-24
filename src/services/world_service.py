"""World service - handles world/entity management."""

from __future__ import annotations

import logging
import random
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.memory.entities import Entity, Relationship
from src.memory.story_state import StoryState
from src.memory.world_database import WorldDatabase
from src.settings import Settings
from src.utils.exceptions import GenerationCancelledError
from src.utils.validation import (
    validate_not_empty,
    validate_not_none,
    validate_positive,
    validate_type,
)

if TYPE_CHECKING:
    from src.services import ServiceContainer

logger = logging.getLogger(__name__)


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
        generate_structure: Whether to generate story structure (characters, chapters).
        generate_locations: Whether to generate location entities.
        generate_factions: Whether to generate faction entities.
        generate_items: Whether to generate item entities.
        generate_concepts: Whether to generate concept entities.
        generate_relationships: Whether to generate relationships between entities.
        cancellation_event: Optional threading.Event to signal cancellation.
    """

    clear_existing: bool = False
    generate_structure: bool = True
    generate_locations: bool = True
    generate_factions: bool = True
    generate_items: bool = True
    generate_concepts: bool = True
    generate_relationships: bool = True
    cancellation_event: threading.Event | None = field(default=None, repr=False)

    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self.cancellation_event is not None and self.cancellation_event.is_set()

    @classmethod
    def full(cls, cancellation_event: threading.Event | None = None) -> WorldBuildOptions:
        """Create options for full world build (everything, keeping existing data)."""
        return cls(
            clear_existing=False,
            generate_structure=True,
            generate_locations=True,
            generate_factions=True,
            generate_items=True,
            generate_concepts=True,
            generate_relationships=True,
            cancellation_event=cancellation_event,
        )

    @classmethod
    def full_rebuild(cls, cancellation_event: threading.Event | None = None) -> WorldBuildOptions:
        """Create options for full world rebuild (everything, clearing existing first)."""
        return cls(
            clear_existing=True,
            generate_structure=True,
            generate_locations=True,
            generate_factions=True,
            generate_items=True,
            generate_concepts=True,
            generate_relationships=True,
            cancellation_event=cancellation_event,
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
        validate_not_none(state, "state")
        validate_type(state, "state", StoryState)
        validate_not_none(world_db, "world_db")
        validate_type(world_db, "world_db", WorldDatabase)

        if not state.brief:
            raise ValueError("Cannot build world - no brief exists.")

        options = options or WorldBuildOptions.full()
        counts: dict[str, int] = {
            "characters": 0,
            "locations": 0,
            "factions": 0,
            "items": 0,
            "concepts": 0,
            "relationships": 0,
        }

        # Calculate total steps for progress reporting
        total_steps = self._calculate_total_steps(options)
        current_step = 0

        def report_progress(message: str, entity_type: str | None = None, count: int = 0) -> None:
            nonlocal current_step
            current_step += 1
            if progress_callback:
                progress_callback(
                    WorldBuildProgress(
                        step=current_step,
                        total_steps=total_steps,
                        message=message,
                        entity_type=entity_type,
                        count=count,
                    )
                )

        logger.info(f"Starting world build for project {state.id} with options: {options}")

        def check_cancelled() -> None:
            """Raise if cancellation requested."""
            if options.is_cancelled():
                logger.info("World build cancelled by user")
                raise GenerationCancelledError("Generation cancelled by user")

        # Step 1: Clear existing data if requested
        if options.clear_existing:
            check_cancelled()
            report_progress("Clearing existing world data...")
            self._clear_world_db(world_db)
            logger.info("Cleared existing world data")

        # Step 2: Generate story structure (characters, chapters) if requested
        if options.generate_structure:
            check_cancelled()
            report_progress("Generating story structure...")
            if options.clear_existing:
                # Full rebuild - clear story state and regenerate
                services.story.rebuild_world(state)
            else:
                # Initial build - build on existing state (doesn't clear)
                # Note: build_structure also extracts to world_db, but we handle that below
                orchestrator = services.story._get_orchestrator(state)
                orchestrator.build_story_structure()
                services.story._sync_state(orchestrator, state)
            counts["characters"] = len(state.characters)
            logger.info(
                f"Story structure built: {len(state.characters)} characters, "
                f"{len(state.chapters)} chapters"
            )

        # Step 3: Extract characters to world database
        check_cancelled()
        report_progress("Adding characters to world...", "character")
        char_count = self._extract_characters_to_world(state, world_db)
        counts["characters"] = char_count
        logger.info(f"Extracted {char_count} characters to world database")

        # Step 4: Generate locations if requested
        if options.generate_locations:
            check_cancelled()
            report_progress("Generating locations...", "location")
            loc_count = self._generate_locations(state, world_db, services)
            counts["locations"] = loc_count
            logger.info(f"Generated {loc_count} locations")

        # Step 5: Generate factions if requested
        if options.generate_factions:
            check_cancelled()
            report_progress("Generating factions...", "faction")
            faction_count = self._generate_factions(state, world_db, services)
            counts["factions"] = faction_count
            logger.info(f"Generated {faction_count} factions")

        # Step 6: Generate items if requested
        if options.generate_items:
            check_cancelled()
            report_progress("Generating items...", "item")
            item_count = self._generate_items(state, world_db, services)
            counts["items"] = item_count
            logger.info(f"Generated {item_count} items")

        # Step 7: Generate concepts if requested
        if options.generate_concepts:
            check_cancelled()
            report_progress("Generating concepts...", "concept")
            concept_count = self._generate_concepts(state, world_db, services)
            counts["concepts"] = concept_count
            logger.info(f"Generated {concept_count} concepts")

        # Step 8: Generate relationships if requested
        if options.generate_relationships:
            check_cancelled()
            report_progress("Generating relationships...", "relationship")
            rel_count = self._generate_relationships(state, world_db, services)
            counts["relationships"] = rel_count
            logger.info(f"Generated {rel_count} relationships")

        report_progress("World build complete!")
        logger.info(f"World build complete for project {state.id}: {counts}")
        return counts

    def _calculate_total_steps(self, options: WorldBuildOptions) -> int:
        """Calculate total number of steps for progress reporting."""
        steps = 2  # Character extraction + completion
        if options.clear_existing:
            steps += 1
        if options.generate_structure:
            steps += 1
        if options.generate_locations:
            steps += 1
        if options.generate_factions:
            steps += 1
        if options.generate_items:
            steps += 1
        if options.generate_concepts:
            steps += 1
        if options.generate_relationships:
            steps += 1
        return steps

    def _clear_world_db(self, world_db: WorldDatabase) -> None:
        """Clear all entities and relationships from world database."""
        # Delete relationships first (they reference entities)
        relationships = world_db.list_relationships()
        logger.info(f"Deleting {len(relationships)} existing relationships...")
        for rel in relationships:
            world_db.delete_relationship(rel.id)

        # Delete all entities
        entities = world_db.list_entities()
        logger.info(f"Deleting {len(entities)} existing entities...")
        for entity in entities:
            world_db.delete_entity(entity.id)

    def _extract_characters_to_world(self, state: StoryState, world_db: WorldDatabase) -> int:
        """Extract characters from story state to world database."""
        added_count = 0
        for char in state.characters:
            # Check if already exists
            existing = world_db.search_entities(char.name, entity_type="character")
            if existing:
                logger.debug(f"Character already exists: {char.name}")
                continue

            world_db.add_entity(
                entity_type="character",
                name=char.name,
                description=char.description,
                attributes={
                    "role": char.role,
                    "personality_traits": char.personality_traits,
                    "goals": char.goals,
                    "arc_notes": char.arc_notes,
                },
            )
            added_count += 1

        return added_count

    def _generate_locations(
        self,
        state: StoryState,
        world_db: WorldDatabase,
        services: ServiceContainer,
    ) -> int:
        """Generate and add locations to world database."""
        # Use project-level settings if available, otherwise fall back to global
        loc_min = state.target_locations_min or self.settings.world_gen_locations_min
        loc_max = state.target_locations_max or self.settings.world_gen_locations_max
        location_count = random.randint(loc_min, loc_max)

        locations = services.story.generate_locations(state, location_count)
        added_count = 0

        for loc in locations:
            if isinstance(loc, dict) and "name" in loc:
                world_db.add_entity(
                    entity_type="location",
                    name=loc["name"],
                    description=loc.get("description", ""),
                    attributes={"significance": loc.get("significance", "")},
                )
                added_count += 1
            else:
                logger.warning(f"Skipping invalid location: {loc}")

        return added_count

    def _generate_factions(
        self,
        state: StoryState,
        world_db: WorldDatabase,
        services: ServiceContainer,
    ) -> int:
        """Generate and add factions to world database."""
        all_entities = world_db.list_entities()
        all_existing_names = [e.name for e in all_entities]
        existing_locations = [e.name for e in all_entities if e.type == "location"]

        # Use project-level settings if available, otherwise fall back to global
        fac_min = state.target_factions_min or self.settings.world_gen_factions_min
        fac_max = state.target_factions_max or self.settings.world_gen_factions_max
        faction_count = random.randint(fac_min, fac_max)

        faction_results = services.world_quality.generate_factions_with_quality(
            state, all_existing_names, faction_count, existing_locations
        )
        added_count = 0

        for faction, faction_scores in faction_results:
            if isinstance(faction, dict) and "name" in faction:
                faction_entity_id = world_db.add_entity(
                    entity_type="faction",
                    name=faction["name"],
                    description=faction.get("description", ""),
                    attributes={
                        "leader": faction.get("leader", ""),
                        "goals": faction.get("goals", []),
                        "values": faction.get("values", []),
                        "base_location": faction.get("base_location", ""),
                        "quality_scores": faction_scores.to_dict(),
                    },
                )
                added_count += 1

                # Create relationship to base location if it exists
                base_loc = faction.get("base_location", "")
                if base_loc:
                    location_entity = next(
                        (e for e in all_entities if e.name == base_loc and e.type == "location"),
                        None,
                    )
                    if location_entity:
                        world_db.add_relationship(
                            source_id=faction_entity_id,
                            target_id=location_entity.id,
                            relation_type="based_in",
                            description=f"{faction['name']} is headquartered in {base_loc}",
                        )
            else:
                logger.warning(f"Skipping invalid faction: {faction}")

        return added_count

    def _generate_items(
        self,
        state: StoryState,
        world_db: WorldDatabase,
        services: ServiceContainer,
    ) -> int:
        """Generate and add items to world database."""
        all_existing_names = [e.name for e in world_db.list_entities()]

        # Use project-level settings if available, otherwise fall back to global
        item_min = state.target_items_min or self.settings.world_gen_items_min
        item_max = state.target_items_max or self.settings.world_gen_items_max
        item_count = random.randint(item_min, item_max)

        item_results = services.world_quality.generate_items_with_quality(
            state, all_existing_names, item_count
        )
        added_count = 0

        for item, item_scores in item_results:
            if isinstance(item, dict) and "name" in item:
                world_db.add_entity(
                    entity_type="item",
                    name=item["name"],
                    description=item.get("description", ""),
                    attributes={
                        "significance": item.get("significance", ""),
                        "owner": item.get("owner", ""),
                        "location": item.get("location", ""),
                        "quality_scores": item_scores.to_dict(),
                    },
                )
                added_count += 1
            else:
                logger.warning(f"Skipping invalid item: {item}")

        return added_count

    def _generate_concepts(
        self,
        state: StoryState,
        world_db: WorldDatabase,
        services: ServiceContainer,
    ) -> int:
        """Generate and add concepts to world database."""
        all_existing_names = [e.name for e in world_db.list_entities()]

        # Use project-level settings if available, otherwise fall back to global
        concept_min = state.target_concepts_min or self.settings.world_gen_concepts_min
        concept_max = state.target_concepts_max or self.settings.world_gen_concepts_max
        concept_count = random.randint(concept_min, concept_max)

        concept_results = services.world_quality.generate_concepts_with_quality(
            state, all_existing_names, concept_count
        )
        added_count = 0

        for concept, concept_scores in concept_results:
            if isinstance(concept, dict) and "name" in concept:
                world_db.add_entity(
                    entity_type="concept",
                    name=concept["name"],
                    description=concept.get("description", ""),
                    attributes={
                        "type": concept.get("type", ""),
                        "importance": concept.get("importance", ""),
                        "quality_scores": concept_scores.to_dict(),
                    },
                )
                added_count += 1
            else:
                logger.warning(f"Skipping invalid concept: {concept}")

        return added_count

    def _generate_relationships(
        self,
        state: StoryState,
        world_db: WorldDatabase,
        services: ServiceContainer,
    ) -> int:
        """Generate and add relationships between entities."""
        all_entities = world_db.list_entities()
        entity_names = [e.name for e in all_entities]
        existing_rels = [(r.source_id, r.target_id) for r in world_db.list_relationships()]

        rel_count = random.randint(
            self.settings.world_gen_relationships_min,
            self.settings.world_gen_relationships_max,
        )

        relationships = services.story.generate_relationships(
            state, entity_names, existing_rels, rel_count
        )
        added_count = 0

        for rel in relationships:
            if isinstance(rel, dict) and "source" in rel and "target" in rel:
                # Find source and target entities
                source_entity = next((e for e in all_entities if e.name == rel["source"]), None)
                target_entity = next((e for e in all_entities if e.name == rel["target"]), None)

                if source_entity and target_entity:
                    world_db.add_relationship(
                        source_id=source_entity.id,
                        target_id=target_entity.id,
                        relation_type=rel.get("relation_type", "related_to"),
                        description=rel.get("description", ""),
                    )
                    added_count += 1
                else:
                    logger.warning(
                        f"Could not find entities for relationship: "
                        f"{rel.get('source')} -> {rel.get('target')}"
                    )
            else:
                logger.warning(f"Skipping invalid relationship: {rel}")

        return added_count

    # ========== ENTITY EXTRACTION ==========

    def extract_entities_from_structure(self, state: StoryState, world_db: WorldDatabase) -> int:
        """Extract characters and locations from story structure to world database.

        Args:
            state: Story state with characters and world description.
            world_db: WorldDatabase to populate.

        Returns:
            Number of entities extracted.
        """
        validate_not_none(state, "state")
        validate_type(state, "state", StoryState)
        validate_not_none(world_db, "world_db")
        validate_type(world_db, "world_db", WorldDatabase)
        logger.debug(
            f"extract_entities_from_structure called: project_id={state.id}, characters={len(state.characters)}"
        )
        count = 0

        try:
            # Extract characters
            for char in state.characters:
                existing = world_db.search_entities(char.name, entity_type="character")
                if existing:
                    continue

                attributes = {
                    "role": char.role,
                    "personality_traits": char.personality_traits,
                    "goals": char.goals,
                    "arc_notes": char.arc_notes,
                }

                entity_id = world_db.add_entity(
                    entity_type="character",
                    name=char.name,
                    description=char.description,
                    attributes=attributes,
                )
                count += 1

                # Add relationships from character data
                for related_name, relationship in char.relationships.items():
                    related_entities = world_db.search_entities(
                        related_name, entity_type="character"
                    )
                    if related_entities:
                        world_db.add_relationship(
                            source_id=entity_id,
                            target_id=related_entities[0].id,
                            relation_type=relationship,
                        )

            # Extract locations from world description
            if state.world_description:
                locations = self._extract_locations_from_text(state.world_description)
                for loc_name, loc_desc in locations:
                    existing = world_db.search_entities(loc_name, entity_type="location")
                    if existing:
                        continue

                    world_db.add_entity(
                        entity_type="location",
                        name=loc_name,
                        description=loc_desc,
                    )
                    count += 1

            logger.info(f"Extracted {count} entities from story structure for project {state.id}")
            return count
        except Exception as e:
            logger.error(f"Failed to extract entities for project {state.id}: {e}", exc_info=True)
            raise

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
        validate_not_empty(content, "content")
        validate_not_none(world_db, "world_db")
        validate_type(world_db, "world_db", WorldDatabase)
        validate_positive(chapter_number, "chapter_number")
        logger.debug(
            f"extract_from_chapter called: chapter={chapter_number}, content_length={len(content)}"
        )
        counts = {
            "entities": 0,
            "relationships": 0,
            "events": 0,
        }

        try:
            # Extract potential new locations mentioned
            locations = self._extract_locations_from_text(content)
            for loc_name, loc_desc in locations:
                existing = world_db.search_entities(loc_name, entity_type="location")
                if not existing:
                    world_db.add_entity(
                        entity_type="location",
                        name=loc_name,
                        description=loc_desc,
                    )
                    counts["entities"] += 1

            # Extract items mentioned
            items = self._extract_items_from_text(content)
            for item_name, item_desc in items:
                existing = world_db.search_entities(item_name, entity_type="item")
                if not existing:
                    world_db.add_entity(
                        entity_type="item",
                        name=item_name,
                        description=item_desc,
                    )
                    counts["entities"] += 1

            # Extract key events
            events = self._extract_events_from_text(content, chapter_number)
            for event_desc in events:
                world_db.add_event(
                    description=event_desc,
                    chapter_number=chapter_number,
                )
                counts["events"] += 1

            logger.info(
                f"Chapter {chapter_number}: extracted {counts['entities']} entities, "
                f"{counts['events']} events"
            )
            return counts
        except Exception as e:
            logger.error(f"Failed to extract from chapter {chapter_number}: {e}", exc_info=True)
            raise

    def _extract_locations_from_text(self, text: str) -> list[tuple[str, str]]:
        """Extract location names and descriptions from text.

        Uses heuristics to find capitalized place names.

        Args:
            text: Text to analyze.

        Returns:
            List of (name, description) tuples.
        """
        locations = []

        # Pattern for "the [Place Name]" or "[Place Name]"
        # Look for capitalized multi-word names that might be places
        patterns = [
            r"(?:in|at|to|from|near|within)\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:forest|castle|city|town|village|mountain|river|valley|kingdom|empire|realm)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) > 2 and match not in ["The", "And", "But", "For"]:
                    # Get context around the match for description
                    idx = text.find(match)
                    if idx >= 0:
                        start = max(0, idx - 50)
                        end = min(len(text), idx + len(match) + 100)
                        context = text[start:end].strip()
                        locations.append((match, context))

        # Deduplicate
        seen = set()
        unique_locations = []
        for name, desc in locations:
            if name.lower() not in seen:
                seen.add(name.lower())
                unique_locations.append((name, desc))

        return unique_locations[: self.settings.entity_extract_locations_max]

    def _extract_items_from_text(self, text: str) -> list[tuple[str, str]]:
        """Extract significant items from text.

        Args:
            text: Text to analyze.

        Returns:
            List of (name, description) tuples.
        """
        items = []

        # Pattern for "the [Item]" with descriptive adjectives
        patterns = [
            r"(?:the|a|an)\s+((?:ancient|magical|enchanted|cursed|sacred|golden|silver)\s+[a-z]+)",
            r"([A-Z][a-z]+(?:'s)?\s+(?:sword|ring|amulet|staff|crown|book|scroll|key|orb|gem))",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) > 3:
                    idx = text.find(match)
                    if idx >= 0:
                        start = max(0, idx - 30)
                        end = min(len(text), idx + len(match) + 80)
                        context = text[start:end].strip()
                        items.append((match.title(), context))

        # Deduplicate
        seen = set()
        unique_items = []
        for name, desc in items:
            if name.lower() not in seen:
                seen.add(name.lower())
                unique_items.append((name, desc))

        return unique_items[: self.settings.entity_extract_items_max]

    def _extract_events_from_text(self, text: str, chapter_number: int) -> list[str]:
        """Extract key events from chapter text.

        Args:
            text: Chapter text content.
            chapter_number: Chapter number.

        Returns:
            List of event descriptions.
        """
        events = []

        # Split into sentences and look for action-heavy ones
        sentences = re.split(r"[.!?]+", text)

        action_verbs = [
            "discovered",
            "found",
            "killed",
            "defeated",
            "escaped",
            "revealed",
            "betrayed",
            "married",
            "died",
            "born",
            "destroyed",
            "created",
            "saved",
            "captured",
            "freed",
            "declared",
            "attacked",
            "defended",
            "won",
            "lost",
        ]

        for sentence in sentences:
            sentence = sentence.strip()
            if (
                len(sentence) < self.settings.event_sentence_min_length
                or len(sentence) > self.settings.event_sentence_max_length
            ):
                continue

            # Check if sentence contains significant action
            sentence_lower = sentence.lower()
            for verb in action_verbs:
                if verb in sentence_lower:
                    events.append(sentence)
                    break

        return events[: self.settings.entity_extract_events_max]

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
        logger.info(f"Adding {entity_type} entity: {name}")
        entity_id = world_db.add_entity(
            entity_type=entity_type,
            name=name,
            description=description,
            attributes=attributes or {},
        )
        logger.debug(f"Created entity {entity_id}")
        return entity_id

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
        logger.info(f"Updating entity {entity_id}" + (f" -> {name}" if name else ""))
        result = world_db.update_entity(
            entity_id=entity_id,
            name=name,
            description=description,
            attributes=attributes,
        )
        if result:
            logger.debug(f"Entity {entity_id} updated successfully")
        else:
            logger.warning(f"Entity {entity_id} not found for update")
        return result

    def delete_entity(self, world_db: WorldDatabase, entity_id: str) -> bool:
        """Delete an entity and its relationships.

        Args:
            world_db: WorldDatabase instance.
            entity_id: Entity ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        logger.info(f"Deleting entity {entity_id}")
        result = world_db.delete_entity(entity_id)
        if result:
            logger.debug(f"Entity {entity_id} deleted")
        else:
            logger.warning(f"Entity {entity_id} not found for deletion")
        return result

    def get_entity(self, world_db: WorldDatabase, entity_id: str) -> Entity | None:
        """Get an entity by ID.

        Args:
            world_db: WorldDatabase instance.
            entity_id: Entity ID.

        Returns:
            Entity or None if not found.
        """
        return world_db.get_entity(entity_id)

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
        logger.debug(f"list_entities called: entity_type={entity_type}")
        entities = world_db.list_entities(entity_type=entity_type)
        logger.debug(
            f"Found {len(entities)} entities" + (f" of type {entity_type}" if entity_type else "")
        )
        return entities

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
        logger.debug(f"search_entities called: query={query}, entity_type={entity_type}")
        results = world_db.search_entities(query, entity_type=entity_type)
        logger.debug(f"Found {len(results)} entities matching '{query}'")
        return results

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
        logger.info(f"Adding relationship: {source_id} -> {target_id} ({relation_type})")
        rel_id = world_db.add_relationship(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            description=description,
            bidirectional=bidirectional,
        )
        logger.debug(f"Created relationship {rel_id}")
        return rel_id

    def delete_relationship(self, world_db: WorldDatabase, relationship_id: str) -> bool:
        """Delete a relationship.

        Args:
            world_db: WorldDatabase instance.
            relationship_id: Relationship ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        logger.info(f"Deleting relationship {relationship_id}")
        result = world_db.delete_relationship(relationship_id)
        if result:
            logger.debug(f"Relationship {relationship_id} deleted")
        else:
            logger.warning(f"Relationship {relationship_id} not found")
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
        if entity_id:
            return world_db.get_relationships(entity_id)
        return world_db.list_relationships()

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
        return world_db.find_path(source_id, target_id)

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
        return world_db.get_connected_entities(entity_id, max_depth=max_depth)

    def get_communities(self, world_db: WorldDatabase) -> list[list[str]]:
        """Detect communities/clusters in the entity graph.

        Args:
            world_db: WorldDatabase instance.

        Returns:
            List of communities, each a list of entity IDs.
        """
        return world_db.get_communities()

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
        return world_db.get_most_connected(limit=limit)

    # ========== CONTEXT FOR AGENTS ==========

    def get_context_for_agents(
        self,
        world_db: WorldDatabase,
        max_entities: int = 50,
    ) -> dict[str, Any]:
        """Get world context formatted for AI agents.

        Args:
            world_db: WorldDatabase instance.
            max_entities: Maximum entities per type to include.

        Returns:
            Dictionary with world context (characters, locations, items, etc.).
        """
        return world_db.get_context_for_agents(max_entities=max_entities)

    def get_entity_summary(self, world_db: WorldDatabase) -> dict[str, int]:
        """Get a summary count of entities by type.

        Args:
            world_db: WorldDatabase instance.

        Returns:
            Dictionary mapping entity type to count.
        """
        logger.debug("get_entity_summary called")
        summary = {
            "character": world_db.count_entities("character"),
            "location": world_db.count_entities("location"),
            "item": world_db.count_entities("item"),
            "faction": world_db.count_entities("faction"),
            "concept": world_db.count_entities("concept"),
            "relationships": len(world_db.list_relationships()),
        }
        logger.debug(f"Entity summary: {summary}")
        return summary

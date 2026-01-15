"""World service - handles world/entity management."""

import logging
import re
from typing import Any

from memory.entities import Entity, Relationship
from memory.story_state import StoryState
from memory.world_database import WorldDatabase

logger = logging.getLogger(__name__)


class WorldService:
    """World and entity management service.

    This service handles extraction of entities from story content,
    entity CRUD operations, and relationship management.
    """

    # ========== ENTITY EXTRACTION ==========

    def extract_entities_from_structure(self, state: StoryState, world_db: WorldDatabase) -> int:
        """Extract characters and locations from story structure to world database.

        Args:
            state: Story state with characters and world description.
            world_db: WorldDatabase to populate.

        Returns:
            Number of entities extracted.
        """
        count = 0

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
                related_entities = world_db.search_entities(related_name, entity_type="character")
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

        logger.info(f"Extracted {count} entities from story structure")
        return count

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
        counts = {
            "entities": 0,
            "relationships": 0,
            "events": 0,
        }

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

        return unique_locations[:10]  # Limit to prevent over-extraction

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

        return unique_items[:5]  # Limit

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
            if len(sentence) < 20 or len(sentence) > 200:
                continue

            # Check if sentence contains significant action
            sentence_lower = sentence.lower()
            for verb in action_verbs:
                if verb in sentence_lower:
                    events.append(sentence)
                    break

        return events[:5]  # Limit to most significant events

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
        return world_db.add_entity(
            entity_type=entity_type,
            name=name,
            description=description,
            attributes=attributes or {},
        )

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
        return world_db.update_entity(
            entity_id=entity_id,
            name=name,
            description=description,
            attributes=attributes,
        )

    def delete_entity(self, world_db: WorldDatabase, entity_id: str) -> bool:
        """Delete an entity and its relationships.

        Args:
            world_db: WorldDatabase instance.
            entity_id: Entity ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        return world_db.delete_entity(entity_id)

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
        return world_db.list_entities(entity_type=entity_type)

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
        return world_db.search_entities(query, entity_type=entity_type)

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
        return world_db.add_relationship(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            description=description,
            bidirectional=bidirectional,
        )

    def delete_relationship(self, world_db: WorldDatabase, relationship_id: str) -> bool:
        """Delete a relationship.

        Args:
            world_db: WorldDatabase instance.
            relationship_id: Relationship ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        return world_db.delete_relationship(relationship_id)

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
        return {
            "character": world_db.count_entities("character"),
            "location": world_db.count_entities("location"),
            "item": world_db.count_entities("item"),
            "faction": world_db.count_entities("faction"),
            "concept": world_db.count_entities("concept"),
            "relationships": len(world_db.list_relationships()),
        }

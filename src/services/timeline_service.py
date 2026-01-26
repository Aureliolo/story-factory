"""Timeline service - aggregates entity and event data for timeline visualization.

This service handles:
- Timestamp parsing from raw strings
- Entity lifecycle extraction
- Timeline data aggregation for vis.js
"""

import logging
from typing import TYPE_CHECKING, Any

from src.memory.entities import Entity, WorldEvent
from src.memory.timeline_types import (
    EntityLifecycle,
    StoryTimestamp,
    TimelineItem,
    extract_lifecycle_from_attributes,
    parse_timestamp,
)
from src.settings import Settings
from src.utils.constants import get_entity_color
from src.utils.validation import validate_not_none, validate_type

if TYPE_CHECKING:
    from src.memory.world_database import WorldDatabase

logger = logging.getLogger(__name__)


class TimelineService:
    """Service for timeline data aggregation and visualization.

    This service coordinates:
    - Parsing timestamp strings from entities and events
    - Extracting entity lifecycle information
    - Aggregating data for vis.js Timeline component
    """

    def __init__(self, settings: Settings):
        """
        Create a TimelineService configured with the provided application settings.

        Parameters:
            settings (Settings): Application settings used to configure timeline behavior and dependencies.
        """
        logger.debug("Initializing TimelineService")
        self.settings = settings
        logger.debug("TimelineService initialized successfully")

    def get_timeline_items(
        self,
        world_db: WorldDatabase,
        entity_types: list[str] | None = None,
        include_events: bool = True,
    ) -> list[TimelineItem]:
        """
        Aggregate entities and events from the world database into a list of timeline items.

        Parameters:
            world_db (WorldDatabase): Source database to read entities and events from.
            entity_types (list[str] | None): If provided, include only entities whose type is in this list.
            include_events (bool): If True, include world events as timeline items.

        Returns:
            list[TimelineItem]: Timeline items sorted by each item's start timestamp.
        """
        validate_not_none(world_db, "world_db")
        logger.debug(
            f"get_timeline_items: entity_types={entity_types}, include_events={include_events}"
        )

        items: list[TimelineItem] = []

        # Get entities
        entities = world_db.list_entities()
        for entity in entities:
            if entity_types and entity.type not in entity_types:
                continue

            item = self._entity_to_timeline_item(entity)
            if item:
                items.append(item)

        # Get events if requested
        if include_events:
            events = world_db.list_events()
            for event in events:
                item = self._event_to_timeline_item(event)
                if item:
                    items.append(item)

        # Sort by start timestamp
        items.sort(key=lambda x: x.start.sort_key)

        logger.info(f"Generated {len(items)} timeline items")
        return items

    def _entity_to_timeline_item(self, entity: Entity) -> TimelineItem | None:
        """
        Convert an Entity into a TimelineItem for timeline visualization.

        Uses lifecycle timestamps (birth or first_appearance for start; death or last_appearance for end) when present; otherwise uses the entity's created_at as a relative start timestamp. The returned TimelineItem contains id, entity_id, label, type, start, end, color, description, and group suitable for timeline rendering.

        Parameters:
            entity (Entity): The entity to convert; must have a created_at timestamp for relative ordering when explicit lifecycle dates are absent.

        Returns:
            TimelineItem: A timeline representation of the entity.
        """
        validate_not_none(entity, "entity")

        # Try to extract lifecycle from attributes
        lifecycle = extract_lifecycle_from_attributes(entity.attributes)

        start: StoryTimestamp | None = None
        end: StoryTimestamp | None = None

        if lifecycle:
            # Use lifecycle dates if available
            if lifecycle.birth:
                start = lifecycle.birth
            elif lifecycle.first_appearance:
                start = lifecycle.first_appearance

            if lifecycle.death:
                end = lifecycle.death
            elif lifecycle.last_appearance:
                end = lifecycle.last_appearance

        # Fall back to created_at if no lifecycle data
        if not start:
            # Create a relative timestamp based on creation order
            # This ensures entities without dates still appear on timeline
            start = StoryTimestamp(
                raw_text=f"Added: {entity.created_at.strftime('%Y-%m-%d')}",
                relative_order=int(entity.created_at.timestamp()),
            )

        color = get_entity_color(entity.type)

        item = TimelineItem(
            id=f"entity-{entity.id}",
            entity_id=entity.id,
            event_id=None,
            label=entity.name,
            item_type=entity.type,
            start=start,
            end=end,
            color=color,
            description=entity.description,
            group=entity.type,
        )

        logger.debug(f"Created timeline item for entity {entity.name}: {item.id}")
        return item

    def _event_to_timeline_item(self, event: WorldEvent) -> TimelineItem | None:
        """
        Create a TimelineItem representing a world event for timeline display.

        The start timestamp is taken from `event.timestamp_in_story` when present, otherwise from `event.chapter_number`, and falls back to `event.created_at` if neither is available.

        Returns:
            TimelineItem representing the event for the timeline.
        """
        validate_not_none(event, "event")

        # Parse timestamp from the event
        if event.timestamp_in_story:
            start = parse_timestamp(event.timestamp_in_story)
        elif event.chapter_number is not None:
            start = StoryTimestamp(
                relative_order=event.chapter_number,
                raw_text=f"Chapter {event.chapter_number}",
            )
        else:
            # Use creation time as fallback
            start = StoryTimestamp(
                raw_text=f"Added: {event.created_at.strftime('%Y-%m-%d')}",
                relative_order=int(event.created_at.timestamp()),
            )

        # Events are point-in-time (no end)
        item = TimelineItem(
            id=f"event-{event.id}",
            entity_id=None,
            event_id=event.id,
            label=event.description[:50] + ("..." if len(event.description) > 50 else ""),
            item_type="event",
            start=start,
            end=None,  # Events are points, not ranges
            color="#FF5722",  # Orange for events
            description=event.description,
            group="event",
        )

        logger.debug(f"Created timeline item for event {event.id}")
        return item

    def get_entity_lifecycle(self, entity: Entity) -> EntityLifecycle | None:
        """
        Return the entity's lifecycle extracted from its attributes.

        Returns:
            EntityLifecycle or None: Lifecycle data parsed from the entity's attributes, or `None` if no lifecycle information is present.
        """
        validate_not_none(entity, "entity")
        validate_type(entity, "entity", Entity)
        return extract_lifecycle_from_attributes(entity.attributes)

    def update_entity_lifecycle(
        self,
        world_db: WorldDatabase,
        entity_id: str,
        lifecycle: EntityLifecycle,
    ) -> bool:
        """
        Update an entity's lifecycle data stored in the entity's attributes under the "lifecycle" key.

        Parameters:
                world_db (WorldDatabase): Database instance used to retrieve and update the entity.
                entity_id (str): ID of the entity to update.
                lifecycle (EntityLifecycle): Lifecycle information to persist (birth, death, first_appearance, last_appearance).

        Returns:
                bool: True if the entity was found and updated successfully, False otherwise.
        """
        validate_not_none(world_db, "world_db")
        validate_not_none(entity_id, "entity_id")
        validate_not_none(lifecycle, "lifecycle")

        # Get current entity
        entity = world_db.get_entity(entity_id)
        if not entity:
            logger.warning(f"Entity not found for lifecycle update: {entity_id}")
            return False

        # Build lifecycle dict for storage
        lifecycle_data: dict = {}
        if lifecycle.birth:
            lifecycle_data["birth"] = lifecycle.birth.model_dump()
        if lifecycle.death:
            lifecycle_data["death"] = lifecycle.death.model_dump()
        if lifecycle.first_appearance:
            lifecycle_data["first_appearance"] = lifecycle.first_appearance.model_dump()
        if lifecycle.last_appearance:
            lifecycle_data["last_appearance"] = lifecycle.last_appearance.model_dump()

        # Update entity attributes
        new_attributes = dict(entity.attributes)
        new_attributes["lifecycle"] = lifecycle_data

        result = world_db.update_entity(entity_id, attributes=new_attributes)
        if result:
            logger.info(f"Updated lifecycle for entity {entity_id}")
        else:
            logger.warning(f"Failed to update lifecycle for entity {entity_id}")

        return result

    def get_timeline_groups(self, items: list[TimelineItem]) -> list[dict]:
        """
        Builds unique group definitions from timeline items for use by the timeline renderer.

        Parameters:
            items (list[TimelineItem]): Timeline items from which group ids and titles are derived.

        Returns:
            list[dict]: A list of group objects with keys:
                - `id`: group identifier,
                - `content`: title-cased label for display,
                - `style`: inline CSS color style for the group.
        """
        groups_seen = set()
        groups = []

        for item in items:
            if item.group and item.group not in groups_seen:
                groups_seen.add(item.group)
                groups.append(
                    {
                        "id": item.group,
                        "content": item.group.title(),
                        "style": f"color: {get_entity_color(item.group)}",
                    }
                )

        # Sort groups by predefined order
        group_order = ["character", "faction", "location", "item", "concept", "event"]
        groups.sort(key=lambda g: group_order.index(g["id"]) if g["id"] in group_order else 999)

        logger.debug(f"Generated {len(groups)} timeline groups")
        return groups

    def get_timeline_data_for_visjs(
        self,
        world_db: WorldDatabase,
        entity_types: list[str] | None = None,
        include_events: bool = True,
    ) -> dict:
        """
        Generate data structured for use with the vis.js Timeline component.

        Parameters:
            world_db (WorldDatabase): Database providing entities and events to include in the timeline.
            entity_types (list[str] | None): Optional list of entity type names to include; include all types if None.

        Returns:
            dict: A mapping with keys "items" (list of vis.js item dicts) and "groups" (list of vis.js group dicts). Items without temporal information are omitted.
        """
        validate_not_none(world_db, "world_db")

        items = self.get_timeline_items(world_db, entity_types, include_events)
        groups = self.get_timeline_groups(items)

        # Convert items to vis.js format
        visjs_items: list[dict[str, Any]] = []
        for item in items:
            vis_item: dict[str, Any] = {
                "id": item.id,
                "content": item.label,
                "group": item.group,
                "title": item.description,  # Tooltip
                "style": f"background-color: {item.color}22; border-color: {item.color}",
            }

            # Add start/end based on whether it's a range or point
            if item.start.year is not None:
                vis_item["start"] = f"{item.start.year:04d}-01-01"
                if item.end and item.end.year is not None:
                    vis_item["end"] = f"{item.end.year:04d}-12-31"
                    vis_item["type"] = "range"
                else:
                    vis_item["type"] = "point"
            elif item.start.relative_order is not None:
                # Use relative order as pseudo-date (epoch-based)
                vis_item["start"] = item.start.relative_order * 1000  # milliseconds
                if item.end and item.end.relative_order is not None:
                    vis_item["end"] = item.end.relative_order * 1000
                    vis_item["type"] = "range"
                else:
                    vis_item["type"] = "point"
            else:
                continue  # Skip items with no temporal data

            visjs_items.append(vis_item)

        logger.info(f"Generated vis.js data: {len(visjs_items)} items, {len(groups)} groups")

        return {
            "items": visjs_items,
            "groups": groups,
        }

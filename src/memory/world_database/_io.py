"""Export/Import and context operations mixin for WorldDatabase."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._base import WorldDatabaseBase

    _IOMixinBase = WorldDatabaseBase
else:
    _IOMixinBase = object

logger = logging.getLogger(__name__)


class IOMixin(_IOMixinBase):
    """Mixin providing export/import and context operations."""

    def get_context_for_agents(self, max_entities: int = 50) -> dict[str, Any]:
        """Get compressed world context for AI agents.

        Args:
            max_entities: Maximum entities to include per type

        Returns:
            Dict with world context
        """
        return {
            "characters": [
                {"name": e.name, "description": e.description, "attributes": e.attributes}
                for e in self.list_entities("character")[:20]
            ],
            "locations": [
                {"name": e.name, "description": e.description, "attributes": e.attributes}
                for e in self.list_entities("location")[:15]
            ],
            "items": [
                {"name": e.name, "description": e.description}
                for e in self.list_entities("item")[:10]
            ],
            "factions": [
                {"name": e.name, "description": e.description}
                for e in self.list_entities("faction")[:10]
            ],
            "key_relationships": self._get_important_relationships(limit=30),
            "recent_events": [
                {"description": e.description, "chapter": e.chapter_number}
                for e in self.list_events(limit=20)
            ],
            "entity_counts": {
                "characters": self.count_entities("character"),
                "locations": self.count_entities("location"),
                "items": self.count_entities("item"),
                "factions": self.count_entities("faction"),
                "concepts": self.count_entities("concept"),
            },
        }

    def _get_important_relationships(self, limit: int = 30) -> list[dict[str, Any]]:
        """Get most important relationships for context.

        Args:
            limit: Maximum relationships to return

        Returns:
            List of relationship dicts
        """
        relationships = self.list_relationships()
        # Sort by strength descending
        relationships.sort(key=lambda r: r.strength, reverse=True)

        result = []
        for rel in relationships[:limit]:
            source = self.get_entity(rel.source_id)
            target = self.get_entity(rel.target_id)
            if source and target:
                result.append(
                    {
                        "from": source.name,
                        "to": target.name,
                        "type": rel.relation_type,
                        "description": rel.description,
                    }
                )
        return result

    def export_to_json(self) -> dict[str, Any]:
        """Export entire database to JSON.

        Returns:
            Dict with all data
        """
        return {
            "entities": [e.model_dump(mode="json") for e in self.list_entities()],
            "relationships": [r.model_dump(mode="json") for r in self.list_relationships()],
            "events": [
                {
                    **e.model_dump(mode="json"),
                    "participants": [
                        {"entity_id": p.entity_id, "role": p.role}
                        for p in self.get_event_participants(e.id)
                    ],
                }
                for e in self.list_events()
            ],
        }

    def import_from_json(self, data: dict[str, Any]) -> None:
        """Import data from JSON export.

        Args:
            data: Previously exported data
        """
        with self._lock:
            cursor = self.conn.cursor()

            # Clear existing data
            cursor.execute("DELETE FROM event_participants")
            cursor.execute("DELETE FROM events")
            cursor.execute("DELETE FROM relationships")
            cursor.execute("DELETE FROM entities")

            # Import entities
            for entity_data in data.get("entities", []):
                cursor.execute(
                    """
                    INSERT INTO entities (id, type, name, description, attributes, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entity_data["id"],
                        entity_data["type"],
                        entity_data["name"],
                        entity_data.get("description", ""),
                        json.dumps(entity_data.get("attributes", {})),
                        entity_data.get("created_at", datetime.now().isoformat()),
                        entity_data.get("updated_at", datetime.now().isoformat()),
                    ),
                )

            # Import relationships
            for rel_data in data.get("relationships", []):
                cursor.execute(
                    """
                    INSERT INTO relationships
                    (id, source_id, target_id, relation_type, description, strength, bidirectional, attributes, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rel_data["id"],
                        rel_data["source_id"],
                        rel_data["target_id"],
                        rel_data["relation_type"],
                        rel_data.get("description", ""),
                        rel_data.get("strength", 1.0),
                        1 if rel_data.get("bidirectional", False) else 0,
                        json.dumps(rel_data.get("attributes", {})),
                        rel_data.get("created_at", datetime.now().isoformat()),
                    ),
                )

            # Import events
            for event_data in data.get("events", []):
                cursor.execute(
                    """
                    INSERT INTO events (id, description, chapter_number, timestamp_in_story, consequences, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_data["id"],
                        event_data["description"],
                        event_data.get("chapter_number"),
                        event_data.get("timestamp_in_story", ""),
                        json.dumps(event_data.get("consequences", [])),
                        event_data.get("created_at", datetime.now().isoformat()),
                    ),
                )

                # Import participants
                for participant in event_data.get("participants", []):
                    cursor.execute(
                        "INSERT INTO event_participants (event_id, entity_id, role) VALUES (?, ?, ?)",
                        (event_data["id"], participant["entity_id"], participant["role"]),
                    )

            self.conn.commit()
            self._invalidate_graph()

        logger.info(
            f"Imported {len(data.get('entities', []))} entities, "
            f"{len(data.get('relationships', []))} relationships, "
            f"{len(data.get('events', []))} events"
        )

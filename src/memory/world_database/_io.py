"""Import/export operations for WorldDatabase."""

import json
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def export_to_json(db) -> dict[str, Any]:
    """Export entire database to JSON.

    Args:
        db: WorldDatabase instance.

    Returns:
        Dict with all data
    """
    with db._lock:
        db._ensure_open()
        return {
            "entities": [e.model_dump(mode="json") for e in db.list_entities()],
            "relationships": [r.model_dump(mode="json") for r in db.list_relationships()],
            "events": [
                {
                    **e.model_dump(mode="json"),
                    "participants": [
                        {"entity_id": p.entity_id, "role": p.role}
                        for p in db.get_event_participants(e.id)
                    ],
                }
                for e in db.list_events()
            ],
        }


def import_from_json(db, data: dict[str, Any]) -> None:
    """Import data from JSON export.

    Args:
        db: WorldDatabase instance.
        data: Previously exported data
    """
    with db._lock:
        db._ensure_open()
        cursor = db.conn.cursor()

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

        db.conn.commit()
        from . import _graph

        _graph.invalidate_graph(db)

    logger.info(
        f"Imported {len(data.get('entities', []))} entities, "
        f"{len(data.get('relationships', []))} relationships, "
        f"{len(data.get('events', []))} events"
    )

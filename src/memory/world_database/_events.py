"""Event management operations for WorldDatabase."""

import json
import logging
import uuid
from datetime import datetime

from src.memory.entities import EventParticipant, WorldEvent

logger = logging.getLogger(__name__)


def add_event(
    db,
    description: str,
    participants: list[tuple[str, str]] | None = None,
    chapter_number: int | None = None,
    timestamp_in_story: str = "",
    consequences: list[str] | None = None,
) -> str:
    """Add an event.

    Args:
        db: WorldDatabase instance.
        description: Event description
        participants: List of (entity_id, role) tuples
        chapter_number: Chapter where event occurs
        timestamp_in_story: In-world timing
        consequences: List of consequences

    Returns:
        Event ID
    """
    event_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    consequences_json = json.dumps(consequences or [])

    with db._lock:
        cursor = db.conn.cursor()
        cursor.execute(
            """
            INSERT INTO events (id, description, chapter_number, timestamp_in_story, consequences, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (event_id, description, chapter_number, timestamp_in_story, consequences_json, now),
        )

        # Add participants
        if participants:
            for entity_id, role in participants:
                cursor.execute(
                    "INSERT INTO event_participants (event_id, entity_id, role) VALUES (?, ?, ?)",
                    (event_id, entity_id, role),
                )

        db.conn.commit()

    logger.debug(f"Added event: {description[:50]}...")
    return event_id


def get_events_for_entity(db, entity_id: str) -> list[WorldEvent]:
    """Get all events involving an entity.

    Args:
        db: WorldDatabase instance.
        entity_id: Entity ID

    Returns:
        List of events
    """
    with db._lock:
        cursor = db.conn.cursor()
        cursor.execute(
            """
            SELECT e.* FROM events e
            JOIN event_participants ep ON e.id = ep.event_id
            WHERE ep.entity_id = ?
            ORDER BY e.chapter_number, e.created_at
            """,
            (entity_id,),
        )
        return [row_to_event(row) for row in cursor.fetchall()]


def get_events_for_chapter(db, chapter_number: int) -> list[WorldEvent]:
    """Get all events for a chapter.

    Args:
        db: WorldDatabase instance.
        chapter_number: Chapter number

    Returns:
        List of events
    """
    with db._lock:
        cursor = db.conn.cursor()
        cursor.execute(
            "SELECT * FROM events WHERE chapter_number = ? ORDER BY created_at",
            (chapter_number,),
        )
        return [row_to_event(row) for row in cursor.fetchall()]


def get_event_participants(db, event_id: str) -> list[EventParticipant]:
    """Get participants for an event.

    Args:
        db: WorldDatabase instance.
        event_id: Event ID

    Returns:
        List of event participants
    """
    with db._lock:
        cursor = db.conn.cursor()
        cursor.execute("SELECT * FROM event_participants WHERE event_id = ?", (event_id,))
        return [
            EventParticipant(
                event_id=row["event_id"],
                entity_id=row["entity_id"],
                role=row["role"],
            )
            for row in cursor.fetchall()
        ]


def list_events(db, limit: int | None = None) -> list[WorldEvent]:
    """List all events.

    Args:
        db: WorldDatabase instance.
        limit: Optional limit on number of events

    Returns:
        List of events
    """
    # Guard against invalid limit values
    if limit is not None and limit <= 0:
        logger.debug(f"list_events called with limit={limit}, returning empty list")
        return []

    with db._lock:
        cursor = db.conn.cursor()
        if limit is not None:
            cursor.execute(
                "SELECT * FROM events ORDER BY chapter_number, created_at LIMIT ?",
                (limit,),
            )
        else:
            cursor.execute("SELECT * FROM events ORDER BY chapter_number, created_at")
        return [row_to_event(row) for row in cursor.fetchall()]


def row_to_event(row) -> WorldEvent:
    """Convert a database row to a WorldEvent."""
    return WorldEvent(
        id=row["id"],
        description=row["description"],
        chapter_number=row["chapter_number"],
        timestamp_in_story=row["timestamp_in_story"],
        consequences=json.loads(row["consequences"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )

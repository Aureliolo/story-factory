"""Event management mixin for WorldDatabase."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from src.memory.entities import EventParticipant, WorldEvent

if TYPE_CHECKING:
    from ._base import WorldDatabaseBase

    _EventMixinBase = WorldDatabaseBase
else:
    _EventMixinBase = object

logger = logging.getLogger(__name__)


class EventMixin(_EventMixinBase):
    """Mixin providing event management operations."""

    def add_event(
        self,
        description: str,
        participants: list[tuple[str, str]] | None = None,
        chapter_number: int | None = None,
        timestamp_in_story: str = "",
        consequences: list[str] | None = None,
    ) -> str:
        """Add an event.

        Args:
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

        with self._lock:
            cursor = self.conn.cursor()
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

            self.conn.commit()

        logger.debug(f"Added event: {description[:50]}...")
        return event_id

    def get_events_for_entity(self, entity_id: str) -> list[WorldEvent]:
        """Get all events involving an entity.

        Args:
            entity_id: Entity ID

        Returns:
            List of events
        """
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT e.* FROM events e
                JOIN event_participants ep ON e.id = ep.event_id
                WHERE ep.entity_id = ?
                ORDER BY e.chapter_number, e.created_at
                """,
                (entity_id,),
            )
            return [self._row_to_event(row) for row in cursor.fetchall()]

    def get_events_for_chapter(self, chapter_number: int) -> list[WorldEvent]:
        """Get all events for a chapter.

        Args:
            chapter_number: Chapter number

        Returns:
            List of events
        """
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT * FROM events WHERE chapter_number = ? ORDER BY created_at",
                (chapter_number,),
            )
            return [self._row_to_event(row) for row in cursor.fetchall()]

    def get_event_participants(self, event_id: str) -> list[EventParticipant]:
        """Get participants for an event.

        Args:
            event_id: Event ID

        Returns:
            List of event participants
        """
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM event_participants WHERE event_id = ?", (event_id,))
            return [
                EventParticipant(
                    event_id=row["event_id"],
                    entity_id=row["entity_id"],
                    role=row["role"],
                )
                for row in cursor.fetchall()
            ]

    def list_events(self, limit: int | None = None) -> list[WorldEvent]:
        """List all events.

        Args:
            limit: Optional limit on number of events

        Returns:
            List of events
        """
        with self._lock:
            cursor = self.conn.cursor()
            if limit:
                cursor.execute(
                    "SELECT * FROM events ORDER BY chapter_number, created_at LIMIT ?",
                    (limit,),
                )
            else:
                cursor.execute("SELECT * FROM events ORDER BY chapter_number, created_at")
            return [self._row_to_event(row) for row in cursor.fetchall()]

    def _row_to_event(self, row) -> WorldEvent:
        """Convert a database row to a WorldEvent."""
        return WorldEvent(
            id=row["id"],
            description=row["description"],
            chapter_number=row["chapter_number"],
            timestamp_in_story=row["timestamp_in_story"],
            consequences=json.loads(row["consequences"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

"""Entity models for worldbuilding database."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """A world entity (character, location, item, faction, concept)."""

    id: str
    type: str  # character, location, item, faction, concept
    name: str
    description: str = ""
    attributes: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class Relationship(BaseModel):
    """A relationship between two entities."""

    id: str
    source_id: str
    target_id: str
    relation_type: str  # knows, loves, hates, located_in, owns, member_of, etc.
    description: str = ""
    strength: float = 1.0  # Relationship strength/importance (0.0-1.0)
    bidirectional: bool = False
    attributes: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class WorldEvent(BaseModel):
    """A significant event in the story world."""

    id: str
    description: str
    chapter_number: int | None = None
    timestamp_in_story: str = ""  # In-world timing (e.g., "Day 3", "Year 1042")
    consequences: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


class EventParticipant(BaseModel):
    """Links an entity to an event with a role."""

    event_id: str
    entity_id: str
    role: str  # actor, location, affected, witness

"""Tests for world service event helper functions."""

import json
import logging
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.memory.entities import WorldEvent
from src.memory.story_state import EventParticipantEntry, WorldEventCreation
from src.services.world_service._event_helpers import (
    _extract_lifecycle_temporal,
    _parse_lifecycle_sub,
    build_event_entity_context,
    build_event_timestamp,
    resolve_event_participants,
)


class TestParseLifecycleSub:
    """Tests for _parse_lifecycle_sub."""

    def test_dict_passthrough(self):
        """Test that a dict is returned as-is."""
        d = {"year": 1200}
        assert _parse_lifecycle_sub(d) is d

    def test_json_string_parsed(self):
        """Test that a JSON string is parsed back to a dict."""
        assert _parse_lifecycle_sub('{"year": 1200}') == {"year": 1200}

    def test_non_dict_json_returns_none(self):
        """Test that a JSON string of a non-dict returns None."""
        assert _parse_lifecycle_sub("[1, 2, 3]") is None

    def test_invalid_json_string_returns_none(self):
        """Test that an invalid JSON string returns None."""
        assert _parse_lifecycle_sub("not json") is None

    def test_none_returns_none(self):
        """Test that None returns None."""
        assert _parse_lifecycle_sub(None) is None

    def test_int_returns_none(self):
        """Test that an int returns None."""
        assert _parse_lifecycle_sub(42) is None


class TestExtractLifecycleTemporal:
    """Tests for _extract_lifecycle_temporal."""

    def test_no_lifecycle_key(self):
        """Test with attributes that have no lifecycle."""
        assert _extract_lifecycle_temporal({}) == []

    def test_lifecycle_not_dict(self):
        """Test when lifecycle value is not a dict."""
        assert _extract_lifecycle_temporal({"lifecycle": "invalid"}) == []

    def test_birth_year_extracted(self):
        """Test extraction of birth year from lifecycle."""
        attrs = {"lifecycle": {"birth": {"year": 1200}}}
        result = _extract_lifecycle_temporal(attrs)
        assert "birth_year=1200" in result

    def test_death_year_extracted(self):
        """Test extraction of death year from lifecycle."""
        attrs = {"lifecycle": {"death": {"year": 1260}}}
        result = _extract_lifecycle_temporal(attrs)
        assert "death_year=1260" in result

    def test_founding_year_extracted(self):
        """Test extraction of founding year from lifecycle."""
        attrs = {"lifecycle": {"founding": {"year": 800}}}
        result = _extract_lifecycle_temporal(attrs)
        assert "founding_year=800" in result

    def test_dissolution_year_extracted(self):
        """Test extraction of dissolution year from lifecycle."""
        attrs = {"lifecycle": {"dissolution": {"year": 1500}}}
        result = _extract_lifecycle_temporal(attrs)
        assert "dissolution_year=1500" in result

    def test_creation_year_extracted(self):
        """Test extraction of creation year from lifecycle."""
        attrs = {"lifecycle": {"creation": {"year": 950}}}
        result = _extract_lifecycle_temporal(attrs)
        assert "creation_year=950" in result

    def test_multiple_temporal_fields(self):
        """Test extraction of multiple lifecycle temporal fields."""
        attrs = {"lifecycle": {"birth": {"year": 1200}, "death": {"year": 1260}}}
        result = _extract_lifecycle_temporal(attrs)
        assert len(result) == 2
        assert "birth_year=1200" in result
        assert "death_year=1260" in result

    def test_missing_year_in_sub_dict(self):
        """Test when lifecycle sub-dict exists but has no year."""
        attrs = {"lifecycle": {"birth": {"era": "Dark Age"}}}
        result = _extract_lifecycle_temporal(attrs)
        assert result == []

    def test_none_lifecycle(self):
        """Test when lifecycle is None."""
        assert _extract_lifecycle_temporal({"lifecycle": None}) == []

    def test_json_string_sub_values(self):
        """Test extraction when sub-values are JSON strings (DB flattened)."""
        attrs = {
            "lifecycle": {
                "birth": json.dumps({"year": 1200}),
                "death": json.dumps({"year": 1260}),
            }
        }
        result = _extract_lifecycle_temporal(attrs)
        assert len(result) == 2
        assert "birth_year=1200" in result
        assert "death_year=1260" in result

    def test_invalid_json_string_sub_value_skipped(self):
        """Test that invalid JSON string sub-values are skipped gracefully."""
        attrs = {"lifecycle": {"birth": "not valid json"}}
        result = _extract_lifecycle_temporal(attrs)
        assert result == []


class TestBuildEventEntityContext:
    """Tests for build_event_entity_context."""

    def test_empty_database(self, caplog):
        """Test with no entities or relationships logs warning."""
        world_db = MagicMock()
        world_db.list_entities.return_value = []
        world_db.list_relationships.return_value = []

        with caplog.at_level(logging.WARNING, logger="src.services.world_service._event_helpers"):
            result = build_event_entity_context(world_db)

        assert result == "No entities yet."
        assert "no entities or relationships found" in caplog.text

    def test_entities_with_lifecycle_temporal(self):
        """Test with entities that have lifecycle temporal data."""
        entity = MagicMock()
        entity.name = "Aragorn"
        entity.type = "character"
        entity.attributes = {"lifecycle": {"birth": {"year": 2931}}}

        world_db = MagicMock()
        world_db.list_entities.return_value = [entity]
        world_db.list_relationships.return_value = []

        result = build_event_entity_context(world_db)

        assert "ENTITIES:" in result
        assert "Aragorn (character)" in result
        assert "birth_year=2931" in result

    def test_entities_with_flattened_lifecycle(self):
        """Test with entities that have DB-flattened lifecycle strings."""
        entity = MagicMock()
        entity.name = "Aragorn"
        entity.type = "character"
        entity.attributes = {
            "lifecycle": {
                "birth": json.dumps({"year": 2931}),
            }
        }

        world_db = MagicMock()
        world_db.list_entities.return_value = [entity]
        world_db.list_relationships.return_value = []

        result = build_event_entity_context(world_db)

        assert "ENTITIES:" in result
        assert "Aragorn (character)" in result
        assert "birth_year=2931" in result

    def test_entities_without_lifecycle(self):
        """Test with entities that have no lifecycle data."""
        entity = MagicMock()
        entity.name = "Aragorn"
        entity.type = "character"
        entity.attributes = {"some_other_key": "value"}

        world_db = MagicMock()
        world_db.list_entities.return_value = [entity]
        world_db.list_relationships.return_value = []

        result = build_event_entity_context(world_db)

        assert "ENTITIES:" in result
        assert "Aragorn (character)" in result

    def test_entities_with_relationships(self):
        """Test with entities and valid relationships."""
        entity1 = MagicMock()
        entity1.id = "e1"
        entity1.name = "Frodo"
        entity1.type = "character"
        entity1.attributes = {}

        entity2 = MagicMock()
        entity2.id = "e2"
        entity2.name = "The Shire"
        entity2.type = "location"
        entity2.attributes = {}

        rel = MagicMock()
        rel.source_id = "e1"
        rel.target_id = "e2"
        rel.relation_type = "lives_in"

        world_db = MagicMock()
        world_db.list_entities.return_value = [entity1, entity2]
        world_db.list_relationships.return_value = [rel]

        result = build_event_entity_context(world_db)

        assert "ENTITIES:" in result
        assert "RELATIONSHIPS:" in result
        assert "Frodo -[lives_in]-> The Shire" in result

    def test_dangling_relationship_warns(self, caplog):
        """Test that dangling relationships log a warning."""
        entity = MagicMock()
        entity.id = "e1"
        entity.name = "Gandalf"
        entity.type = "character"
        entity.attributes = {}

        rel = MagicMock()
        rel.source_id = "e1"
        rel.target_id = "missing_id"
        rel.relation_type = "knows"

        world_db = MagicMock()
        world_db.list_entities.return_value = [entity]
        world_db.list_relationships.return_value = [rel]

        with caplog.at_level(logging.WARNING, logger="src.services.world_service._event_helpers"):
            result = build_event_entity_context(world_db)

        assert "dangling reference" in caplog.text
        assert "RELATIONSHIPS:" not in result

    def test_entity_without_attributes(self):
        """Test entity with None attributes."""
        entity = MagicMock()
        entity.name = "Mordor"
        entity.type = "location"
        entity.attributes = None

        world_db = MagicMock()
        world_db.list_entities.return_value = [entity]
        world_db.list_relationships.return_value = []

        result = build_event_entity_context(world_db)

        assert "Mordor (location)" in result


class TestBuildEventTimestamp:
    """Tests for build_event_timestamp."""

    def test_full_timestamp(self):
        """Test with year, month, and era_name."""
        event = {"year": 1200, "month": 3, "era_name": "Dark Age"}

        result = build_event_timestamp(event)

        assert result == "Year 1200, Month 3, Dark Age"

    def test_year_only(self):
        """Test with year only."""
        event = {"year": 500}

        result = build_event_timestamp(event)

        assert result == "Year 500"

    def test_empty_event(self):
        """Test with no temporal fields."""
        result = build_event_timestamp({})

        assert result == ""


class TestResolveEventParticipants:
    """Tests for resolve_event_participants."""

    def test_dict_participants(self):
        """Test resolving dict-format participants."""
        entity = MagicMock()
        entity.id = "e1"
        entity.name = "Gandalf"

        event = {
            "participants": [
                {"entity_name": "Gandalf", "role": "instigator"},
            ],
        }

        result = resolve_event_participants(event, [entity])

        assert result == [("e1", "instigator")]

    def test_string_participant_warns(self, caplog):
        """Test that string participants log a warning."""
        entity = MagicMock()
        entity.id = "e1"
        entity.name = "Gandalf"

        event = {"participants": ["Gandalf"]}

        with caplog.at_level(logging.WARNING, logger="src.services.world_service._event_helpers"):
            result = resolve_event_participants(event, [entity])

        assert "Unexpected participant format" in caplog.text
        assert result == [("e1", "affected")]

    def test_unresolved_participant_warns(self, caplog):
        """Test participant that doesn't match any entity logs warning."""
        event = {
            "participants": [{"entity_name": "Nobody", "role": "bystander"}],
        }

        with caplog.at_level(logging.WARNING, logger="src.services.world_service._event_helpers"):
            result = resolve_event_participants(event, [])

        assert result == []
        assert "Could not resolve event participant 'Nobody'" in caplog.text

    def test_no_participants_key(self):
        """Test event with no participants key."""
        result = resolve_event_participants({}, [])

        assert result == []


class TestEventModels:
    """Tests for EventParticipantEntry and WorldEventCreation Pydantic models."""

    def test_event_participant_entry_valid(self):
        """Test creating a valid EventParticipantEntry."""
        entry = EventParticipantEntry(entity_name="Hero", role="actor")
        assert entry.entity_name == "Hero"
        assert entry.role == "actor"

    def test_event_participant_entry_default_role(self):
        """Test that role defaults to 'affected'."""
        entry = EventParticipantEntry(entity_name="Hero")
        assert entry.role == "affected"

    def test_event_participant_entry_rejects_empty_name(self):
        """Test that empty entity_name raises ValidationError."""
        with pytest.raises(ValidationError):
            EventParticipantEntry(entity_name="")

    def test_world_event_creation_valid(self):
        """Test creating a valid WorldEventCreation."""
        event = WorldEventCreation(description="A battle")
        assert event.description == "A battle"

    def test_world_event_creation_rejects_empty_description(self):
        """Test that empty description raises ValidationError."""
        with pytest.raises(ValidationError):
            WorldEventCreation(description="")

    def test_world_event_creation_month_range(self):
        """Test month field validation — valid at 14, invalid at 0 and 15."""
        # Valid: month=14 (max for custom calendars)
        event = WorldEventCreation(description="A feast", month=14)
        assert event.month == 14

        # Invalid: month=0
        with pytest.raises(ValidationError):
            WorldEventCreation(description="A feast", month=0)

        # Invalid: month=15
        with pytest.raises(ValidationError):
            WorldEventCreation(description="A feast", month=15)

    def test_world_event_creation_defaults(self):
        """Test that participants and consequences default to empty lists."""
        event = WorldEventCreation(description="A quiet day")
        assert event.participants == []
        assert event.consequences == []


class TestWorldEventModel:
    """Tests for the WorldEvent entity model."""

    def test_world_event_rejects_empty_description(self):
        """Test that empty description raises ValidationError."""
        with pytest.raises(ValidationError):
            WorldEvent(id="e1", description="")

    def test_world_event_valid(self):
        """Test creating a valid WorldEvent."""
        event = WorldEvent(id="e1", description="Battle")
        assert event.id == "e1"
        assert event.description == "Battle"

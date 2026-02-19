"""Tests for world service event helper functions."""

import logging
from unittest.mock import MagicMock

from src.services.world_service._event_helpers import (
    _extract_lifecycle_temporal,
    build_event_entity_context,
    build_event_timestamp,
    resolve_event_participants,
)


class TestBuildEventEntityContext:
    """Tests for build_event_entity_context."""

    def test_empty_database(self):
        """Test with no entities or relationships."""
        world_db = MagicMock()
        world_db.list_entities.return_value = []
        world_db.list_relationships.return_value = []

        result = build_event_entity_context(world_db)

        assert result == "No entities yet."

    def test_entities_only(self):
        """Test with entities but no relationships."""
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

    def test_includes_lifecycle_temporal(self):
        """Test that entity context includes lifecycle temporal data."""
        entity = MagicMock()
        entity.id = "e1"
        entity.name = "Elara"
        entity.type = "character"
        entity.attributes = {
            "lifecycle": {
                "birth": {"year": 1200, "era_name": "Golden Age"},
                "death": {"year": 1280},
            },
        }

        world_db = MagicMock()
        world_db.list_entities.return_value = [entity]
        world_db.list_relationships.return_value = []

        result = build_event_entity_context(world_db)

        assert "Elara (character)" in result
        assert "birth_year=1200" in result
        assert "era=Golden Age" in result
        assert "death_year=1280" in result


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

    def test_month_only(self):
        """Test with month but no year and no era_name."""
        event = {"month": 3}

        result = build_event_timestamp(event)

        assert result == "Month 3"

    def test_era_only(self):
        """Test with era_name only, no year and no month."""
        event = {"era_name": "Dark Age"}

        result = build_event_timestamp(event)

        assert result == "Dark Age"


class TestExtractLifecycleTemporal:
    """Tests for _extract_lifecycle_temporal."""

    def test_with_lifecycle_data(self):
        """Test extraction from a realistic lifecycle dict."""
        attrs = {
            "lifecycle": {
                "birth": {"year": 1200, "era_name": "Golden Age"},
                "founding_year": 800,
                "death": {"year": 1260},
            },
        }

        result = _extract_lifecycle_temporal(attrs)

        assert "birth_year=1200" in result
        assert "era=Golden Age" in result
        assert "death_year=1260" in result
        assert "founding_year=800" in result

    def test_empty_attrs(self):
        """Test with empty attributes dict returns empty list."""
        result = _extract_lifecycle_temporal({})

        assert result == []

    def test_no_lifecycle_key(self):
        """Test with attributes that have no lifecycle key returns empty list."""
        result = _extract_lifecycle_temporal({"some_attr": "value"})

        assert result == []

    def test_destruction_year(self):
        """Test extraction of destruction_year field."""
        attrs = {"lifecycle": {"destruction_year": 1500}}

        result = _extract_lifecycle_temporal(attrs)

        assert "destruction_year=1500" in result

    def test_temporal_notes(self):
        """Test extraction of temporal_notes field."""
        attrs = {"lifecycle": {"temporal_notes": "Founded during the great migration"}}

        result = _extract_lifecycle_temporal(attrs)

        assert "notes=Founded during the great migration" in result


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

        with caplog.at_level(logging.DEBUG, logger="src.services.world_service._event_helpers"):
            result = resolve_event_participants(event, [entity])

        assert "Participant provided as plain string" in caplog.text
        assert result == [("e1", "affected")]

    def test_unresolved_participant(self):
        """Test participant that doesn't match any entity."""
        event = {
            "participants": [{"entity_name": "Nobody", "role": "bystander"}],
        }

        result = resolve_event_participants(event, [])

        assert result == []

    def test_no_participants_key(self):
        """Test event with no participants key."""
        result = resolve_event_participants({}, [])

        assert result == []

    def test_empty_entity_name(self):
        """Test that a participant with an empty entity_name is silently skipped."""
        entity = MagicMock()
        entity.id = "e1"
        entity.name = "Gandalf"

        event = {
            "participants": [{"entity_name": "", "role": "actor"}],
        }

        result = resolve_event_participants(event, [entity])

        assert result == []

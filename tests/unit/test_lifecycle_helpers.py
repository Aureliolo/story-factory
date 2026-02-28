"""Tests for lifecycle attribute builders."""

from src.memory.story_state import Character
from src.services.world_service._lifecycle_helpers import (
    build_character_lifecycle,
    build_entity_lifecycle,
)


class TestBuildCharacterLifecycle:
    """Tests for build_character_lifecycle()."""

    def test_full_temporal_data(self):
        """Character with all temporal fields produces complete lifecycle dict."""
        char = Character(
            name="Gandalf",
            role="supporting",
            description="A wizard",
            birth_year=100,
            death_year=200,
            birth_era="First Age",
            death_era="Third Age",
            temporal_notes="Some notes",
        )
        result = build_character_lifecycle(char)
        assert result == {
            "lifecycle": {
                "birth": {"year": 100, "era_name": "First Age"},
                "death": {"year": 200, "era_name": "Third Age"},
                "temporal_notes": "Some notes",
            }
        }

    def test_no_temporal_data(self):
        """Character with all temporal fields at defaults returns empty dict."""
        char = Character(
            name="Test",
            role="protagonist",
            description="A test character",
        )
        result = build_character_lifecycle(char)
        assert result == {}

    def test_birth_year_only(self):
        """Character with only birth_year returns lifecycle with birth year."""
        char = Character(
            name="Test",
            role="protagonist",
            description="A test character",
            birth_year=50,
        )
        result = build_character_lifecycle(char)
        assert result == {"lifecycle": {"birth": {"year": 50}}}

    def test_temporal_notes_only(self):
        """Character with only temporal_notes returns lifecycle with notes."""
        char = Character(
            name="Test",
            role="protagonist",
            description="A test character",
            temporal_notes="Active during war",
        )
        result = build_character_lifecycle(char)
        assert result == {"lifecycle": {"temporal_notes": "Active during war"}}

    def test_birth_era_only(self):
        """Character with only birth_era returns lifecycle with era in birth."""
        char = Character(
            name="Test",
            role="protagonist",
            description="A test character",
            birth_era="Second Age",
        )
        result = build_character_lifecycle(char)
        assert result == {"lifecycle": {"birth": {"era_name": "Second Age"}}}

    def test_death_year_only(self):
        """Character with only death_year returns lifecycle with death."""
        char = Character(
            name="Test",
            role="protagonist",
            description="A test character",
            death_year=500,
        )
        result = build_character_lifecycle(char)
        assert result == {"lifecycle": {"death": {"year": 500}}}

    def test_death_era_only(self):
        """Character with only death_era returns lifecycle with death era."""
        char = Character(
            name="Test",
            role="protagonist",
            description="A test character",
            death_era="Final Age",
        )
        result = build_character_lifecycle(char)
        assert result == {"lifecycle": {"death": {"era_name": "Final Age"}}}

    def test_death_year_and_era(self):
        """Character with death_year and death_era produces both in death dict."""
        char = Character(
            name="Test",
            role="protagonist",
            description="A test character",
            death_year=500,
            death_era="Dark Age",
        )
        result = build_character_lifecycle(char)
        assert result == {"lifecycle": {"death": {"year": 500, "era_name": "Dark Age"}}}

    def test_negative_death_year_treated_as_alive(self):
        """Character with negative death_year has death omitted (LLM sentinel)."""
        char = Character(
            name="Test",
            role="protagonist",
            description="A test character",
            death_year=-999,
        )
        result = build_character_lifecycle(char)
        # Negative death_year is rejected as an LLM sentinel — no lifecycle data remains
        assert result == {}

    def test_negative_death_year_with_era_keeps_era(self):
        """Character with negative death_year but valid era keeps era in death dict."""
        char = Character(
            name="Test",
            role="protagonist",
            description="A test character",
            death_year=-1,
            death_era="Final Age",
        )
        result = build_character_lifecycle(char)
        # -1 death_year is rejected, but death_era is preserved
        assert result == {"lifecycle": {"death": {"era_name": "Final Age"}}}

    def test_negative_death_year_with_birth_data(self):
        """Character with negative death_year still has valid birth data."""
        char = Character(
            name="Test",
            role="protagonist",
            description="A test character",
            birth_year=100,
            death_year=-50,
        )
        result = build_character_lifecycle(char)
        assert result == {"lifecycle": {"birth": {"year": 100}}}

    def test_birth_year_and_era(self):
        """Character with birth_year and birth_era produces both in birth dict."""
        char = Character(
            name="Test",
            role="protagonist",
            description="A test character",
            birth_year=300,
            birth_era="Third Age",
        )
        result = build_character_lifecycle(char)
        assert result == {"lifecycle": {"birth": {"year": 300, "era_name": "Third Age"}}}


class TestBuildEntityLifecycle:
    """Tests for build_entity_lifecycle()."""

    def test_location_full_data(self):
        """Location with full temporal data produces lifecycle dict."""
        entity = {
            "name": "Rivendell",
            "founding_year": 500,
            "destruction_year": 800,
            "founding_era": "Golden Age",
            "temporal_notes": "Ancient elven city",
        }
        result = build_entity_lifecycle(entity, "location")
        assert "lifecycle" in result
        lifecycle = result["lifecycle"]
        assert lifecycle["founding_year"] == 500
        assert lifecycle["destruction_year"] == 800
        assert lifecycle["birth"] == {"year": 500, "era_name": "Golden Age"}
        assert lifecycle["temporal_notes"] == "Ancient elven city"

    def test_location_no_data(self):
        """Location with no temporal fields returns empty dict."""
        entity = {"name": "Empty Place"}
        result = build_entity_lifecycle(entity, "location")
        assert result == {}

    def test_faction_with_data(self):
        """Faction with temporal data produces lifecycle dict."""
        entity = {
            "name": "Silver Order",
            "founding_year": 100,
            "dissolution_year": 300,
            "founding_era": "Silver Age",
        }
        result = build_entity_lifecycle(entity, "faction")
        assert "lifecycle" in result
        lifecycle = result["lifecycle"]
        assert lifecycle["founding_year"] == 100
        assert lifecycle["destruction_year"] == 300
        assert lifecycle["birth"] == {"year": 100, "era_name": "Silver Age"}

    def test_faction_no_data(self):
        """Faction with no temporal fields returns empty dict."""
        entity = {"name": "Empty Faction"}
        result = build_entity_lifecycle(entity, "faction")
        assert result == {}

    def test_faction_with_temporal_notes(self):
        """Faction with temporal_notes includes them in lifecycle."""
        entity = {
            "name": "Temporal Faction",
            "founding_year": 50,
            "temporal_notes": "Founded during the great schism",
        }
        result = build_entity_lifecycle(entity, "faction")
        assert "lifecycle" in result
        assert result["lifecycle"]["temporal_notes"] == "Founded during the great schism"

    def test_item_with_data(self):
        """Item with temporal data produces lifecycle dict with birth."""
        entity = {
            "name": "Ancient Sword",
            "creation_year": 250,
            "creation_era": "Iron Age",
        }
        result = build_entity_lifecycle(entity, "item")
        assert "lifecycle" in result
        lifecycle = result["lifecycle"]
        assert lifecycle["birth"] == {"year": 250, "era_name": "Iron Age"}

    def test_item_no_data(self):
        """Item with no temporal fields returns empty dict."""
        entity = {"name": "Plain Ring"}
        result = build_entity_lifecycle(entity, "item")
        assert result == {}

    def test_concept_with_data(self):
        """Concept with temporal data produces lifecycle dict with birth."""
        entity = {
            "name": "The Awakening",
            "emergence_year": 1,
            "emergence_era": "Dawn",
        }
        result = build_entity_lifecycle(entity, "concept")
        assert "lifecycle" in result
        lifecycle = result["lifecycle"]
        assert lifecycle["birth"] == {"year": 1, "era_name": "Dawn"}

    def test_concept_no_data(self):
        """Concept with no temporal fields returns empty dict."""
        entity = {"name": "Abstract Idea"}
        result = build_entity_lifecycle(entity, "concept")
        assert result == {}

    def test_unknown_entity_type(self):
        """Unknown entity type returns empty dict."""
        entity = {"name": "Mystery", "founding_year": 100}
        result = build_entity_lifecycle(entity, "unknown")  # type: ignore[arg-type]
        assert result == {}

    def test_item_temporal_notes_only(self):
        """Item with only temporal_notes returns lifecycle with notes."""
        entity = {
            "name": "Artifact",
            "temporal_notes": "Lost to time",
        }
        result = build_entity_lifecycle(entity, "item")
        assert result == {"lifecycle": {"temporal_notes": "Lost to time"}}

    def test_concept_temporal_notes_only(self):
        """Concept with only temporal_notes returns lifecycle with notes."""
        entity = {
            "name": "Ancient Lore",
            "temporal_notes": "Forgotten wisdom",
        }
        result = build_entity_lifecycle(entity, "concept")
        assert result == {"lifecycle": {"temporal_notes": "Forgotten wisdom"}}

    def test_location_founding_without_era(self):
        """Location with founding_year but no era has birth with year only."""
        entity = {
            "name": "Old Town",
            "founding_year": 400,
        }
        result = build_entity_lifecycle(entity, "location")
        assert "lifecycle" in result
        lifecycle = result["lifecycle"]
        assert lifecycle["founding_year"] == 400
        assert lifecycle["birth"] == {"year": 400}

    def test_item_creation_without_era(self):
        """Item with creation_year but no era has birth without era_name."""
        entity = {
            "name": "Simple Blade",
            "creation_year": 150,
        }
        result = build_entity_lifecycle(entity, "item")
        assert "lifecycle" in result
        lifecycle = result["lifecycle"]
        assert lifecycle["birth"] == {"year": 150}

    def test_location_era_only(self):
        """Location with founding_era but no year produces birth with era only."""
        entity = {
            "name": "Misty Valley",
            "founding_era": "Dawn Era",
        }
        result = build_entity_lifecycle(entity, "location")
        assert "lifecycle" in result
        assert result["lifecycle"]["birth"] == {"era_name": "Dawn Era"}

    def test_faction_founding_without_era(self):
        """Faction with founding_year but no era has birth with year only."""
        entity = {
            "name": "Iron Guard",
            "founding_year": 200,
        }
        result = build_entity_lifecycle(entity, "faction")
        assert "lifecycle" in result
        assert result["lifecycle"]["birth"] == {"year": 200}

    def test_item_era_only(self):
        """Item with creation_era but no year produces birth with era only."""
        entity = {
            "name": "Mysterious Gem",
            "creation_era": "Ancient Era",
        }
        result = build_entity_lifecycle(entity, "item")
        assert "lifecycle" in result
        assert result["lifecycle"]["birth"] == {"era_name": "Ancient Era"}

    def test_concept_era_only(self):
        """Concept with emergence_era but no year produces birth with era only."""
        entity = {
            "name": "Lost Magic",
            "emergence_era": "Forgotten Age",
        }
        result = build_entity_lifecycle(entity, "concept")
        assert "lifecycle" in result
        assert result["lifecycle"]["birth"] == {"era_name": "Forgotten Age"}

    def test_concept_emergence_without_era(self):
        """Concept with emergence_year but no era has birth with year only."""
        entity = {
            "name": "Raw Power",
            "emergence_year": 75,
        }
        result = build_entity_lifecycle(entity, "concept")
        assert "lifecycle" in result
        assert result["lifecycle"]["birth"] == {"year": 75}

    def test_location_destruction_only(self):
        """Location with only destruction_year produces lifecycle."""
        entity = {
            "name": "Fallen City",
            "destruction_year": 600,
        }
        result = build_entity_lifecycle(entity, "location")
        assert "lifecycle" in result
        assert result["lifecycle"]["destruction_year"] == 600

    def test_faction_dissolution_only(self):
        """Faction with only dissolution_year produces lifecycle."""
        entity = {
            "name": "Fallen Guild",
            "dissolution_year": 900,
        }
        result = build_entity_lifecycle(entity, "faction")
        assert "lifecycle" in result
        lifecycle = result["lifecycle"]
        assert lifecycle["destruction_year"] == 900

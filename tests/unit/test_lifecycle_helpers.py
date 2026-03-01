"""Tests for lifecycle attribute builders."""

import pytest

from src.memory.story_state import Character
from src.memory.world_calendar import HistoricalEra, WorldCalendar
from src.services.world_service._lifecycle_helpers import (
    _is_destruction_sentinel,
    _resolve_era_name,
    _validate_and_resolve_era,
    build_character_lifecycle,
    build_entity_lifecycle,
)


@pytest.fixture
def multi_era_calendar() -> WorldCalendar:
    """Calendar with three eras for testing era resolution."""
    return WorldCalendar(
        current_era_name="Third Age",
        era_abbreviation="TA",
        era_start_year=1,
        current_story_year=500,
        eras=[
            HistoricalEra(name="First Age", start_year=1, end_year=200),
            HistoricalEra(name="Second Age", start_year=201, end_year=400),
            HistoricalEra(name="Third Age", start_year=401, end_year=None),
        ],
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
        """Character with negative death_year but valid era keeps era in death dict.

        The Character model validator rejects sentinel death_year values (< 0)
        at construction, setting death_year to None.  The era survives because
        the lifecycle helper only guards on death_year being negative — which
        it never is by the time it reaches the helper.
        """
        char = Character(
            name="Test",
            role="protagonist",
            description="A test character",
            death_year=-1,
            death_era="Final Age",
        )
        result = build_character_lifecycle(char)
        # death_year=-1 is rejected by the Character validator → None at model level,
        # so the lifecycle helper sees death_year=None + death_era="Final Age" → era preserved
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


class TestResolveEraName:
    """Tests for _resolve_era_name()."""

    def test_year_in_first_era(self, multi_era_calendar: WorldCalendar):
        """Year within first era resolves to its name."""
        assert _resolve_era_name(100, multi_era_calendar) == "First Age"

    def test_year_in_second_era(self, multi_era_calendar: WorldCalendar):
        """Year within second era resolves to its name."""
        assert _resolve_era_name(300, multi_era_calendar) == "Second Age"

    def test_year_in_ongoing_era(self, multi_era_calendar: WorldCalendar):
        """Year within ongoing (open-ended) era resolves to its name."""
        assert _resolve_era_name(500, multi_era_calendar) == "Third Age"

    def test_year_at_era_boundary(self, multi_era_calendar: WorldCalendar):
        """Year at exact era boundary resolves to that era."""
        assert _resolve_era_name(200, multi_era_calendar) == "First Age"
        assert _resolve_era_name(201, multi_era_calendar) == "Second Age"

    def test_year_before_all_eras(self, multi_era_calendar: WorldCalendar):
        """Year before all defined eras returns None."""
        # All eras start at year 1, so year -5 is before everything
        # But since First Age starts at 1, year 0 would be before
        # Actually HistoricalEra allows any start_year, year -5 < 1
        assert _resolve_era_name(-5, multi_era_calendar) is None


class TestValidateAndResolveEra:
    """Tests for _validate_and_resolve_era()."""

    def test_no_calendar_returns_claimed_era(self):
        """Without calendar, returns the claimed era unchanged."""
        result = _validate_and_resolve_era(100, "Golden Age", None, "test entity", "founding_year")
        assert result == "Golden Age"

    def test_no_year_returns_claimed_era(self, multi_era_calendar: WorldCalendar):
        """Without a year, returns the claimed era unchanged even with calendar."""
        result = _validate_and_resolve_era(
            None, "Golden Age", multi_era_calendar, "test entity", "founding_year"
        )
        assert result == "Golden Age"

    def test_matching_era_preserved(self, multi_era_calendar: WorldCalendar):
        """Claimed era that matches calendar is preserved."""
        result = _validate_and_resolve_era(
            100, "First Age", multi_era_calendar, "test entity", "founding_year"
        )
        assert result == "First Age"

    def test_mismatched_era_auto_resolved(self, multi_era_calendar: WorldCalendar):
        """Claimed era that mismatches calendar is auto-resolved to correct era."""
        # Year 100 is in First Age, but claimed as "Wrong Age"
        result = _validate_and_resolve_era(
            100, "Wrong Age", multi_era_calendar, "test entity", "founding_year"
        )
        assert result == "First Age"

    def test_no_era_auto_resolved_from_calendar(self, multi_era_calendar: WorldCalendar):
        """Missing era is auto-resolved from calendar when year is provided."""
        result = _validate_and_resolve_era(
            300, None, multi_era_calendar, "test entity", "founding_year"
        )
        assert result == "Second Age"

    def test_year_outside_eras_keeps_claimed(self, multi_era_calendar: WorldCalendar):
        """Year outside all eras keeps claimed era when resolution fails."""
        result = _validate_and_resolve_era(
            -5, "Ancient Era", multi_era_calendar, "test entity", "founding_year"
        )
        assert result == "Ancient Era"

    def test_year_outside_eras_no_claimed_returns_none(self, multi_era_calendar: WorldCalendar):
        """Year outside all eras with no claimed era returns None."""
        result = _validate_and_resolve_era(
            -5, None, multi_era_calendar, "test entity", "founding_year"
        )
        assert result is None


class TestIsDestructionSentinel:
    """Tests for _is_destruction_sentinel()."""

    def test_zero_is_sentinel(self):
        """Integer 0 is a destruction sentinel."""
        assert _is_destruction_sentinel(0) is True

    def test_nonzero_int_not_sentinel(self):
        """Non-zero integers are not sentinels."""
        assert _is_destruction_sentinel(500) is False
        assert _is_destruction_sentinel(-1) is False

    def test_none_not_sentinel(self):
        """None is not a sentinel."""
        assert _is_destruction_sentinel(None) is False

    def test_string_zero_not_sentinel(self):
        """String "0" is not a sentinel (must be int)."""
        assert _is_destruction_sentinel("0") is False

    def test_float_zero_not_sentinel(self):
        """Float 0.0 is not a sentinel (must be int)."""
        assert _is_destruction_sentinel(0.0) is False

    def test_bool_false_not_sentinel(self):
        """Boolean False is not a sentinel; only integer year 0 is a sentinel."""
        assert _is_destruction_sentinel(False) is False


class TestBuildEntityLifecycleWithCalendar:
    """Tests for build_entity_lifecycle() with calendar era validation."""

    def test_location_era_auto_resolved(self, multi_era_calendar: WorldCalendar):
        """Location without era gets era auto-resolved from calendar."""
        entity = {"name": "Old City", "founding_year": 150}
        result = build_entity_lifecycle(entity, "location", calendar=multi_era_calendar)
        assert result["lifecycle"]["birth"] == {"year": 150, "era_name": "First Age"}

    def test_location_mismatched_era_corrected(self, multi_era_calendar: WorldCalendar):
        """Location with wrong era gets auto-corrected from calendar."""
        entity = {"name": "Citadel", "founding_year": 150, "founding_era": "Third Age"}
        result = build_entity_lifecycle(entity, "location", calendar=multi_era_calendar)
        # Year 150 is in First Age, not Third Age — should be auto-corrected
        assert result["lifecycle"]["birth"] == {"year": 150, "era_name": "First Age"}

    def test_location_correct_era_preserved(self, multi_era_calendar: WorldCalendar):
        """Location with correct era keeps it unchanged."""
        entity = {"name": "Tower", "founding_year": 150, "founding_era": "First Age"}
        result = build_entity_lifecycle(entity, "location", calendar=multi_era_calendar)
        assert result["lifecycle"]["birth"] == {"year": 150, "era_name": "First Age"}

    def test_location_destruction_year_zero_dropped(self):
        """Location with destruction_year=0 has it dropped as sentinel."""
        entity = {"name": "Standing City", "founding_year": 100, "destruction_year": 0}
        result = build_entity_lifecycle(entity, "location")
        lifecycle = result["lifecycle"]
        assert "destruction_year" not in lifecycle
        assert lifecycle["founding_year"] == 100

    def test_location_destruction_year_nonzero_preserved(self):
        """Location with non-zero destruction_year keeps it."""
        entity = {"name": "Fallen City", "founding_year": 100, "destruction_year": 500}
        result = build_entity_lifecycle(entity, "location")
        assert result["lifecycle"]["destruction_year"] == 500

    def test_faction_era_auto_resolved(self, multi_era_calendar: WorldCalendar):
        """Faction without era gets era auto-resolved from calendar."""
        entity = {"name": "Old Guard", "founding_year": 350}
        result = build_entity_lifecycle(entity, "faction", calendar=multi_era_calendar)
        assert result["lifecycle"]["birth"] == {"year": 350, "era_name": "Second Age"}

    def test_faction_dissolution_year_zero_dropped(self):
        """Faction with dissolution_year=0 has it dropped as sentinel."""
        entity = {"name": "Active Guild", "founding_year": 100, "dissolution_year": 0}
        result = build_entity_lifecycle(entity, "faction")
        lifecycle = result["lifecycle"]
        assert "destruction_year" not in lifecycle
        assert lifecycle["founding_year"] == 100

    def test_faction_dissolution_year_nonzero_preserved(self):
        """Faction with non-zero dissolution_year keeps it."""
        entity = {"name": "Dead Guild", "founding_year": 100, "dissolution_year": 300}
        result = build_entity_lifecycle(entity, "faction")
        assert result["lifecycle"]["destruction_year"] == 300

    def test_item_era_auto_resolved(self, multi_era_calendar: WorldCalendar):
        """Item without era gets era auto-resolved from calendar."""
        entity = {"name": "Ancient Blade", "creation_year": 50}
        result = build_entity_lifecycle(entity, "item", calendar=multi_era_calendar)
        assert result["lifecycle"]["birth"] == {"year": 50, "era_name": "First Age"}

    def test_item_mismatched_era_corrected(self, multi_era_calendar: WorldCalendar):
        """Item with wrong era gets auto-corrected from calendar."""
        entity = {"name": "Relic", "creation_year": 250, "creation_era": "First Age"}
        result = build_entity_lifecycle(entity, "item", calendar=multi_era_calendar)
        assert result["lifecycle"]["birth"] == {"year": 250, "era_name": "Second Age"}

    def test_concept_era_auto_resolved(self, multi_era_calendar: WorldCalendar):
        """Concept without era gets era auto-resolved from calendar."""
        entity = {"name": "Old Magic", "emergence_year": 450}
        result = build_entity_lifecycle(entity, "concept", calendar=multi_era_calendar)
        assert result["lifecycle"]["birth"] == {"year": 450, "era_name": "Third Age"}

    def test_concept_mismatched_era_corrected(self, multi_era_calendar: WorldCalendar):
        """Concept with wrong era gets auto-corrected from calendar."""
        entity = {"name": "Arcane Lore", "emergence_year": 450, "emergence_era": "First Age"}
        result = build_entity_lifecycle(entity, "concept", calendar=multi_era_calendar)
        assert result["lifecycle"]["birth"] == {"year": 450, "era_name": "Third Age"}

    def test_no_calendar_no_era_change(self):
        """Without calendar, era fields remain as-is."""
        entity = {"name": "Town", "founding_year": 100, "founding_era": "Whatever"}
        result = build_entity_lifecycle(entity, "location", calendar=None)
        assert result["lifecycle"]["birth"] == {"year": 100, "era_name": "Whatever"}

    def test_era_only_no_year_no_calendar_lookup(self, multi_era_calendar: WorldCalendar):
        """Era without year skips calendar validation (no year to look up)."""
        entity = {"name": "Mystic Place", "founding_era": "Fictional Era"}
        result = build_entity_lifecycle(entity, "location", calendar=multi_era_calendar)
        # No year to validate against, so the claimed era is kept
        assert result["lifecycle"]["birth"] == {"era_name": "Fictional Era"}

    def test_destruction_zero_only_field_returns_empty(self):
        """Location with only destruction_year=0 returns empty dict (sentinel dropped)."""
        entity = {"name": "Standing", "destruction_year": 0}
        result = build_entity_lifecycle(entity, "location")
        assert result == {}


class TestBuildCharacterLifecycleWithCalendar:
    """Tests for build_character_lifecycle() with calendar era auto-resolution."""

    def test_birth_era_auto_resolved(self, multi_era_calendar: WorldCalendar):
        """Character with birth_year but no birth_era gets era auto-resolved."""
        char = Character(
            name="Hero",
            role="protagonist",
            description="A hero",
            birth_year=100,
        )
        result = build_character_lifecycle(char, calendar=multi_era_calendar)
        assert result["lifecycle"]["birth"] == {"year": 100, "era_name": "First Age"}

    def test_death_era_auto_resolved(self, multi_era_calendar: WorldCalendar):
        """Character with death_year but no death_era gets era auto-resolved."""
        char = Character(
            name="Fallen",
            role="supporting",
            description="A fallen hero",
            death_year=350,
        )
        result = build_character_lifecycle(char, calendar=multi_era_calendar)
        assert result["lifecycle"]["death"] == {"year": 350, "era_name": "Second Age"}

    def test_both_eras_auto_resolved(self, multi_era_calendar: WorldCalendar):
        """Character with years but no eras gets both eras auto-resolved."""
        char = Character(
            name="Elder",
            role="supporting",
            description="An ancient being",
            birth_year=50,
            death_year=450,
        )
        result = build_character_lifecycle(char, calendar=multi_era_calendar)
        assert result["lifecycle"]["birth"] == {"year": 50, "era_name": "First Age"}
        assert result["lifecycle"]["death"] == {"year": 450, "era_name": "Third Age"}

    def test_correct_eras_preserved(self, multi_era_calendar: WorldCalendar):
        """Character with correct eras keeps them unchanged."""
        char = Character(
            name="Sage",
            role="supporting",
            description="A wise sage",
            birth_year=100,
            birth_era="First Age",
            death_year=300,
            death_era="Second Age",
        )
        result = build_character_lifecycle(char, calendar=multi_era_calendar)
        assert result["lifecycle"]["birth"] == {"year": 100, "era_name": "First Age"}
        assert result["lifecycle"]["death"] == {"year": 300, "era_name": "Second Age"}

    def test_mismatched_birth_era_corrected(self, multi_era_calendar: WorldCalendar):
        """Character with wrong birth_era gets auto-corrected."""
        char = Character(
            name="Warrior",
            role="protagonist",
            description="A warrior",
            birth_year=100,
            birth_era="Third Age",  # Wrong: year 100 is in First Age
        )
        result = build_character_lifecycle(char, calendar=multi_era_calendar)
        assert result["lifecycle"]["birth"] == {"year": 100, "era_name": "First Age"}

    def test_mismatched_death_era_corrected(self, multi_era_calendar: WorldCalendar):
        """Character with wrong death_era gets auto-corrected."""
        char = Character(
            name="Ghost",
            role="supporting",
            description="A ghost",
            death_year=300,
            death_era="First Age",  # Wrong: year 300 is in Second Age
        )
        result = build_character_lifecycle(char, calendar=multi_era_calendar)
        assert result["lifecycle"]["death"] == {"year": 300, "era_name": "Second Age"}

    def test_no_calendar_no_auto_resolve(self):
        """Without calendar, birth/death eras are not auto-resolved."""
        char = Character(
            name="Simple",
            role="protagonist",
            description="A simple character",
            birth_year=100,
            death_year=500,
        )
        result = build_character_lifecycle(char, calendar=None)
        assert result["lifecycle"]["birth"] == {"year": 100}
        assert result["lifecycle"]["death"] == {"year": 500}

    def test_era_only_no_year_preserved_with_calendar(self, multi_era_calendar: WorldCalendar):
        """Character with era but no year keeps era unchanged (no year to validate)."""
        char = Character(
            name="Mystery",
            role="supporting",
            description="A mysterious figure",
            birth_era="Some Era",
        )
        result = build_character_lifecycle(char, calendar=multi_era_calendar)
        assert result["lifecycle"]["birth"] == {"era_name": "Some Era"}

    def test_negative_death_year_still_skipped_with_calendar(
        self, multi_era_calendar: WorldCalendar
    ):
        """Negative death_year sentinel still works with calendar present."""
        char = Character(
            name="Immortal",
            role="protagonist",
            description="An immortal being",
            birth_year=100,
            death_year=-999,
        )
        result = build_character_lifecycle(char, calendar=multi_era_calendar)
        # death_year=-999 is rejected by Character validator → None,
        # so only birth data remains; birth_year=100 gets era auto-resolved
        assert result["lifecycle"]["birth"] == {"year": 100, "era_name": "First Age"}
        assert "death" not in result["lifecycle"]

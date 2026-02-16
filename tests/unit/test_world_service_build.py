"""Tests for WorldService.build_world and related methods."""

from unittest.mock import MagicMock

import pytest

from src.memory.story_state import (
    Chapter,
    Character,
    PlotOutline,
    PlotPoint,
    StoryBrief,
    StoryState,
)
from src.memory.templates import WorldTemplate
from src.memory.world_database import WorldDatabase
from src.services.world_service import (
    WorldBuildOptions,
    WorldBuildProgress,
    WorldService,
)
from src.settings import Settings


class TestWorldBuildOptions:
    """Tests for WorldBuildOptions dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        options = WorldBuildOptions()
        assert options.clear_existing is False
        assert options.generate_structure is True
        assert options.generate_locations is True
        assert options.generate_factions is True
        assert options.generate_items is True
        assert options.generate_concepts is True
        assert options.generate_relationships is True

    def test_full_factory(self):
        """Test full() factory method."""
        options = WorldBuildOptions.full()
        assert options.clear_existing is False
        assert options.generate_structure is True
        assert options.generate_locations is True
        assert options.generate_factions is True
        assert options.generate_items is True
        assert options.generate_concepts is True
        assert options.generate_relationships is True

    def test_full_rebuild_factory(self):
        """Test full_rebuild() factory method."""
        options = WorldBuildOptions.full_rebuild()
        assert options.clear_existing is True
        assert options.generate_structure is True
        assert options.generate_locations is True
        assert options.generate_factions is True
        assert options.generate_items is True
        assert options.generate_concepts is True
        assert options.generate_relationships is True

    def test_custom_options(self):
        """Test custom option combinations."""
        options = WorldBuildOptions(
            clear_existing=True,
            generate_structure=False,
            generate_locations=True,
            generate_factions=False,
            generate_items=True,
            generate_concepts=False,
            generate_relationships=True,
        )
        assert options.clear_existing is True
        assert options.generate_structure is False
        assert options.generate_locations is True
        assert options.generate_factions is False
        assert options.generate_items is True
        assert options.generate_concepts is False
        assert options.generate_relationships is True


class TestWorldBuildProgress:
    """Tests for WorldBuildProgress dataclass."""

    def test_required_fields(self):
        """Test required fields are set correctly."""
        progress = WorldBuildProgress(
            step=1,
            total_steps=5,
            message="Testing...",
        )
        assert progress.step == 1
        assert progress.total_steps == 5
        assert progress.message == "Testing..."
        assert progress.entity_type is None
        assert progress.count == 0

    def test_optional_fields(self):
        """Test optional fields are set correctly."""
        progress = WorldBuildProgress(
            step=2,
            total_steps=10,
            message="Generating characters",
            entity_type="character",
            count=5,
        )
        assert progress.step == 2
        assert progress.total_steps == 10
        assert progress.message == "Generating characters"
        assert progress.entity_type == "character"
        assert progress.count == 5


@pytest.fixture
def settings():
    """Create test settings.

    Uses default Settings() to avoid dependency on local settings.json file.
    """
    return Settings()


@pytest.fixture
def world_service(settings):
    """Create WorldService instance."""
    return WorldService(settings)


@pytest.fixture
def mock_world_db(tmp_path):
    """Create a mock WorldDatabase."""
    db_path = tmp_path / "test_world.db"
    db = WorldDatabase(str(db_path))
    yield db
    db.close()


@pytest.fixture
def sample_story_state():
    """Create a sample story state with brief."""
    state = StoryState(id="test-project")
    state.brief = StoryBrief(
        premise="A hero's journey",
        genre="fantasy",
        tone="epic",
        target_length="short_story",
        themes=["courage"],
        language="English",
        setting_time="Medieval era",
        setting_place="A fantasy kingdom",
        content_rating="none",
    )
    state.world_description = "A magical realm"
    state.characters = [
        Character(
            name="Hero",
            role="protagonist",
            description="The main character",
            personality_traits=["brave"],
            goals=["save the world"],
            arc_notes="Grows stronger",
            relationships={},
        ),
        Character(
            name="Mentor",
            role="supporting",
            description="Wise guide",
            personality_traits=["wise"],
            goals=["guide the hero"],
            arc_notes="Sacrifices self",
            relationships={"Hero": "mentor"},
        ),
    ]
    state.chapters = [
        Chapter(
            number=1,
            title="Beginning",
            outline="Scene 1: The hero awakens and begins the journey.",
        ),
    ]
    return state


@pytest.fixture
def mock_services():
    """Create mock ServiceContainer."""
    services = MagicMock()
    services.story.rebuild_world = MagicMock()
    services.world_quality.generate_locations_with_quality = MagicMock(return_value=[])
    services.world_quality.generate_factions_with_quality = MagicMock(return_value=[])
    services.world_quality.generate_items_with_quality = MagicMock(return_value=[])
    services.world_quality.generate_concepts_with_quality = MagicMock(return_value=[])
    services.world_quality.generate_relationships_with_quality = MagicMock(return_value=[])

    # Quality review pass-through: return characters/chapters unchanged with mock scores
    def _review_characters(characters, state, cancel_check=None):
        """Pass characters through review unchanged."""
        return [(char, MagicMock()) for char in characters]

    def _review_chapters(chapters, state, cancel_check=None):
        """Pass chapters through review unchanged."""
        return [(ch, MagicMock()) for ch in chapters]

    services.world_quality.review_characters_batch = MagicMock(side_effect=_review_characters)
    services.world_quality.review_chapters_batch = MagicMock(side_effect=_review_chapters)
    return services


class TestCalculateTotalSteps:
    """Tests for _calculate_total_steps method."""

    def test_full_options(self, world_service):
        """Test step count for full options."""
        options = WorldBuildOptions.full()
        # Base (2: character extraction + completion)
        # + structure (1) + quality review (3: characters, plot, chapters)
        # + locations (1) + factions (1)
        # + items (1) + concepts (1) + relationships (1) = 11
        assert world_service._calculate_total_steps(options) == 11

    def test_full_rebuild_options(self, world_service):
        """Test step count for full rebuild options."""
        options = WorldBuildOptions.full_rebuild()
        # Base (2: character extraction + completion)
        # + clear (1) + structure (1) + quality review (3: characters, plot, chapters)
        # + locations (1) + factions (1)
        # + items (1) + concepts (1) + relationships (1) = 12
        assert world_service._calculate_total_steps(options) == 12

    def test_custom_options(self, world_service):
        """Test step count for custom options."""
        options = WorldBuildOptions(
            clear_existing=True,
            generate_structure=False,
            generate_locations=True,
            generate_factions=False,
            generate_items=False,
            generate_concepts=True,
            generate_relationships=False,
        )
        # Clear (1) + characters (1) + locations (1) + concepts (1) + completion (1) = 5
        assert world_service._calculate_total_steps(options) == 5


class TestClearWorldDb:
    """Tests for _clear_world_db method."""

    def test_clears_relationships_first(self, world_service, mock_world_db):
        """Test relationships are deleted before entities."""
        # Add some entities and relationships
        id1 = mock_world_db.add_entity("character", "Char1", "Desc1")
        id2 = mock_world_db.add_entity("character", "Char2", "Desc2")
        mock_world_db.add_relationship(id1, id2, "knows")

        assert len(mock_world_db.list_relationships()) == 1
        assert mock_world_db.count_entities() == 2

        world_service._clear_world_db(mock_world_db)

        assert len(mock_world_db.list_relationships()) == 0
        assert mock_world_db.count_entities() == 0

    def test_handles_empty_db(self, world_service, mock_world_db):
        """Test handles empty database gracefully."""
        assert mock_world_db.count_entities() == 0
        world_service._clear_world_db(mock_world_db)
        assert mock_world_db.count_entities() == 0


class TestExtractCharactersToWorld:
    """Tests for _extract_characters_to_world method."""

    def test_extracts_all_characters(self, world_service, mock_world_db, sample_story_state):
        """Test extracts all characters from story state."""
        char_count, _rel_count = world_service._extract_characters_to_world(
            sample_story_state, mock_world_db
        )

        assert char_count == 2
        entities = mock_world_db.list_entities(entity_type="character")
        assert len(entities) == 2
        names = [e.name for e in entities]
        assert "Hero" in names
        assert "Mentor" in names

    def test_skips_existing_characters(self, world_service, mock_world_db, sample_story_state):
        """Test skips characters that already exist."""
        # Pre-add one character
        mock_world_db.add_entity("character", "Hero", "Pre-existing")

        char_count, _rel_count = world_service._extract_characters_to_world(
            sample_story_state, mock_world_db
        )

        # Only Mentor should be added
        assert char_count == 1
        entities = mock_world_db.list_entities(entity_type="character")
        assert len(entities) == 2

    def test_stores_character_attributes(self, world_service, mock_world_db, sample_story_state):
        """Test character attributes are stored correctly."""
        world_service._extract_characters_to_world(sample_story_state, mock_world_db)

        hero = mock_world_db.search_entities("Hero", entity_type="character")[0]
        assert hero.attributes["role"] == "protagonist"
        assert "brave" in hero.attributes["personality_traits"]
        assert "save the world" in hero.attributes["goals"]

    def test_handles_no_characters(self, world_service, mock_world_db):
        """Test handles state with no characters."""
        state = StoryState(id="empty")
        state.brief = StoryBrief(
            premise="Test",
            genre="test",
            tone="test",
            target_length="short_story",
            themes=[],
            language="English",
            setting_time="Modern",
            setting_place="City",
            content_rating="none",
        )
        state.characters = []

        char_count, rel_count = world_service._extract_characters_to_world(state, mock_world_db)
        assert char_count == 0
        assert rel_count == 0

    def test_creates_implicit_relationships(self, world_service, mock_world_db, sample_story_state):
        """Test creates implicit relationships from Character.relationships."""
        char_count, rel_count = world_service._extract_characters_to_world(
            sample_story_state, mock_world_db
        )

        assert char_count == 2
        # Mentor has relationships={"Hero": "mentor"}
        assert rel_count == 1

        # Verify the relationship exists in the database
        mentor = mock_world_db.search_entities("Mentor", entity_type="character")[0]
        hero = mock_world_db.search_entities("Hero", entity_type="character")[0]
        rels = mock_world_db.get_relationships(mentor.id)
        assert len(rels) == 1
        assert rels[0].target_id == hero.id
        assert rels[0].relation_type == "mentor"

    def test_skips_implicit_relationships_for_existing_characters(
        self, world_service, mock_world_db, sample_story_state
    ):
        """Test that incremental builds don't duplicate implicit relationships."""
        # First run: both characters are new, relationships created
        char_count, rel_count = world_service._extract_characters_to_world(
            sample_story_state, mock_world_db
        )
        assert char_count == 2
        assert rel_count == 1

        # Second run: both characters exist, no new relationships
        char_count, rel_count = world_service._extract_characters_to_world(
            sample_story_state, mock_world_db
        )
        assert char_count == 0
        assert rel_count == 0

        # Verify only 1 relationship exists (no duplicate)
        mentor = mock_world_db.search_entities("Mentor", entity_type="character")[0]
        rels = mock_world_db.get_relationships(mentor.id)
        assert len(rels) == 1

    def test_skips_relationships_for_unknown_targets(self, world_service, mock_world_db):
        """Test skips relationships when target character is not in the character list."""
        state = StoryState(id="test-unknown-target")
        state.brief = StoryBrief(
            premise="Test",
            genre="test",
            tone="test",
            target_length="short_story",
            themes=[],
            language="English",
            setting_time="Modern",
            setting_place="City",
            content_rating="none",
        )
        state.characters = [
            Character(
                name="Alice",
                role="protagonist",
                description="Main character",
                relationships={"Unknown": "friend"},
            ),
        ]

        char_count, rel_count = world_service._extract_characters_to_world(state, mock_world_db)

        assert char_count == 1
        assert rel_count == 0


class TestGenerateLocations:
    """Tests for _generate_locations method."""

    def test_generates_locations_via_quality_service(
        self, world_service, mock_world_db, sample_story_state, mock_services
    ):
        """Test generates locations through quality refinement service."""
        mock_quality_scores = MagicMock()
        mock_quality_scores.to_dict.return_value = {"atmosphere": 8.0}

        mock_services.world_quality.generate_locations_with_quality.return_value = [
            (
                {"name": "Castle", "description": "A grand castle", "significance": "Home base"},
                mock_quality_scores,
            ),
            (
                {"name": "Forest", "description": "Dark woods", "significance": "Danger zone"},
                mock_quality_scores,
            ),
        ]

        count = world_service._generate_locations(sample_story_state, mock_world_db, mock_services)

        assert count == 2
        locations = mock_world_db.list_entities(entity_type="location")
        assert len(locations) == 2
        names = [loc.name for loc in locations]
        assert "Castle" in names
        assert "Forest" in names
        # Verify quality service was called, not story service
        mock_services.world_quality.generate_locations_with_quality.assert_called_once()

    def test_stores_quality_scores_in_attributes(
        self, world_service, mock_world_db, sample_story_state, mock_services
    ):
        """Test that quality scores are stored in entity attributes."""
        mock_quality_scores = MagicMock()
        mock_quality_scores.to_dict.return_value = {"atmosphere": 9.0, "average": 8.5}

        mock_services.world_quality.generate_locations_with_quality.return_value = [
            (
                {"name": "Castle", "description": "A castle", "significance": "Important"},
                mock_quality_scores,
            ),
        ]

        world_service._generate_locations(sample_story_state, mock_world_db, mock_services)

        locations = mock_world_db.list_entities(entity_type="location")
        assert len(locations) == 1
        assert locations[0].attributes["quality_scores"] == {"atmosphere": 9.0, "average": 8.5}

    def test_skips_locations_without_name(
        self, world_service, mock_world_db, sample_story_state, mock_services, caplog
    ):
        """Test skips locations without name field."""
        import logging

        mock_quality_scores = MagicMock()
        mock_quality_scores.to_dict.return_value = {}

        mock_services.world_quality.generate_locations_with_quality.return_value = [
            ({"name": "", "description": "No name"}, mock_quality_scores),
            ({"description": "Missing name entirely"}, mock_quality_scores),
        ]

        with caplog.at_level(logging.WARNING):
            count = world_service._generate_locations(
                sample_story_state, mock_world_db, mock_services
            )

        assert count == 0

    def test_handles_empty_response(
        self, world_service, mock_world_db, sample_story_state, mock_services
    ):
        """Test handles empty location list."""
        mock_services.world_quality.generate_locations_with_quality.return_value = []

        count = world_service._generate_locations(sample_story_state, mock_world_db, mock_services)

        assert count == 0

    def test_passes_location_names_only(
        self, world_service, mock_world_db, sample_story_state, mock_services
    ):
        """Test that only location-type names are passed for duplicate checking."""
        # Add entities of different types
        mock_world_db.add_entity("character", "Hero", "The hero")
        mock_world_db.add_entity("location", "Castle", "A castle")
        mock_world_db.add_entity("faction", "Guild", "A guild")

        mock_services.world_quality.generate_locations_with_quality.return_value = []

        world_service._generate_locations(sample_story_state, mock_world_db, mock_services)

        # Verify only location names were passed
        call_args = mock_services.world_quality.generate_locations_with_quality.call_args
        existing_names = call_args[0][1]  # Second positional arg
        assert "Castle" in existing_names
        assert "Hero" not in existing_names
        assert "Guild" not in existing_names


class TestGenerateFactions:
    """Tests for _generate_factions method."""

    def test_generates_factions(
        self, world_service, mock_world_db, sample_story_state, mock_services
    ):
        """Test generates and adds factions."""
        mock_quality_scores = MagicMock()
        mock_quality_scores.to_dict.return_value = {"score": 0.8}

        mock_services.world_quality.generate_factions_with_quality.return_value = [
            ({"name": "Guild", "description": "A guild", "leader": "Master"}, mock_quality_scores),
        ]

        faction_count, implicit_count = world_service._generate_factions(
            sample_story_state, mock_world_db, mock_services
        )

        assert faction_count == 1
        assert implicit_count == 0
        factions = mock_world_db.list_entities(entity_type="faction")
        assert len(factions) == 1
        assert factions[0].name == "Guild"

    def test_creates_base_location_relationship(
        self, world_service, mock_world_db, sample_story_state, mock_services
    ):
        """Test creates relationship to base location if it exists."""
        # First add a location
        mock_world_db.add_entity("location", "Castle", "A castle")

        mock_quality_scores = MagicMock()
        mock_quality_scores.to_dict.return_value = {}

        mock_services.world_quality.generate_factions_with_quality.return_value = [
            (
                {"name": "Knights", "description": "Noble knights", "base_location": "Castle"},
                mock_quality_scores,
            ),
        ]

        _faction_count, implicit_count = world_service._generate_factions(
            sample_story_state, mock_world_db, mock_services
        )

        # Check relationship was created and counted
        relationships = mock_world_db.list_relationships()
        assert len(relationships) == 1
        assert relationships[0].relation_type == "based_in"
        assert implicit_count == 1

    def test_skips_invalid_factions(
        self, world_service, mock_world_db, sample_story_state, mock_services, caplog
    ):
        """Test skips factions without name field and logs warning."""
        import logging

        mock_quality_scores = MagicMock()
        mock_quality_scores.to_dict.return_value = {}

        mock_services.world_quality.generate_factions_with_quality.return_value = [
            ({"description": "Missing name"}, mock_quality_scores),  # Invalid
            ("just a string", mock_quality_scores),  # Invalid
        ]

        with caplog.at_level(logging.WARNING):
            faction_count, implicit_count = world_service._generate_factions(
                sample_story_state, mock_world_db, mock_services
            )

        assert faction_count == 0
        assert implicit_count == 0
        assert "Skipping invalid faction" in caplog.text


class TestGenerateItems:
    """Tests for _generate_items method."""

    def test_generates_items(self, world_service, mock_world_db, sample_story_state, mock_services):
        """Test generates and adds items."""
        mock_quality_scores = MagicMock()
        mock_quality_scores.to_dict.return_value = {"quality": 0.9}

        mock_services.world_quality.generate_items_with_quality.return_value = [
            (
                {"name": "Sword", "description": "A magic sword", "significance": "Key weapon"},
                mock_quality_scores,
            ),
        ]

        count = world_service._generate_items(sample_story_state, mock_world_db, mock_services)

        assert count == 1
        items = mock_world_db.list_entities(entity_type="item")
        assert len(items) == 1
        assert items[0].name == "Sword"

    def test_skips_invalid_items(
        self, world_service, mock_world_db, sample_story_state, mock_services, caplog
    ):
        """Test skips items without name field and logs warning."""
        import logging

        mock_quality_scores = MagicMock()
        mock_quality_scores.to_dict.return_value = {}

        mock_services.world_quality.generate_items_with_quality.return_value = [
            ({"description": "Missing name"}, mock_quality_scores),  # Invalid
            ("just a string", mock_quality_scores),  # Invalid
        ]

        with caplog.at_level(logging.WARNING):
            count = world_service._generate_items(sample_story_state, mock_world_db, mock_services)

        assert count == 0
        assert "Skipping invalid item" in caplog.text


class TestGenerateConcepts:
    """Tests for _generate_concepts method."""

    def test_generates_concepts(
        self, world_service, mock_world_db, sample_story_state, mock_services
    ):
        """Test generates and adds concepts."""
        mock_quality_scores = MagicMock()
        mock_quality_scores.to_dict.return_value = {}

        mock_services.world_quality.generate_concepts_with_quality.return_value = [
            (
                {"name": "Magic", "description": "The force of magic", "type": "power"},
                mock_quality_scores,
            ),
        ]

        count = world_service._generate_concepts(sample_story_state, mock_world_db, mock_services)

        assert count == 1
        concepts = mock_world_db.list_entities(entity_type="concept")
        assert len(concepts) == 1
        assert concepts[0].name == "Magic"

    def test_skips_invalid_concepts(
        self, world_service, mock_world_db, sample_story_state, mock_services, caplog
    ):
        """Test skips concepts without name field and logs warning."""
        import logging

        mock_quality_scores = MagicMock()
        mock_quality_scores.to_dict.return_value = {}

        mock_services.world_quality.generate_concepts_with_quality.return_value = [
            ({"description": "Missing name"}, mock_quality_scores),  # Invalid
            ("just a string", mock_quality_scores),  # Invalid
        ]

        with caplog.at_level(logging.WARNING):
            count = world_service._generate_concepts(
                sample_story_state, mock_world_db, mock_services
            )

        assert count == 0
        assert "Skipping invalid concept" in caplog.text


class TestGenerateRelationships:
    """Tests for _generate_relationships method."""

    def test_generates_relationships(
        self, world_service, mock_world_db, sample_story_state, mock_services
    ):
        """Test generates and adds relationships via quality service."""
        # Add some entities first
        mock_world_db.add_entity("character", "Hero", "The hero")
        mock_world_db.add_entity("character", "Villain", "The villain")

        mock_quality_scores = MagicMock()
        mock_quality_scores.average = 8.0

        mock_services.world_quality.generate_relationships_with_quality.return_value = [
            (
                {
                    "source": "Hero",
                    "target": "Villain",
                    "relation_type": "enemies",
                    "description": "Mortal enemies",
                },
                mock_quality_scores,
            ),
        ]

        count = world_service._generate_relationships(
            sample_story_state, mock_world_db, mock_services
        )

        assert count == 1
        relationships = mock_world_db.list_relationships()
        assert len(relationships) == 1
        assert relationships[0].relation_type == "enemies"
        # Verify quality service was called, not story service
        mock_services.world_quality.generate_relationships_with_quality.assert_called_once()

    def test_skips_invalid_relationships(
        self, world_service, mock_world_db, sample_story_state, mock_services
    ):
        """Test skips relationships with missing entities."""
        mock_world_db.add_entity("character", "Hero", "The hero")
        # Note: Villain is NOT added

        mock_quality_scores = MagicMock()
        mock_quality_scores.average = 7.0

        mock_services.world_quality.generate_relationships_with_quality.return_value = [
            (
                {
                    "source": "Hero",
                    "target": "Villain",
                    "relation_type": "enemies",
                },
                mock_quality_scores,
            ),
        ]

        count = world_service._generate_relationships(
            sample_story_state, mock_world_db, mock_services
        )

        assert count == 0

    def test_skips_malformed_relationships(
        self, world_service, mock_world_db, sample_story_state, mock_services, caplog
    ):
        """Test skips relationships that are not proper dicts and logs warning."""
        import logging

        mock_quality_scores = MagicMock()
        mock_quality_scores.average = 5.0

        mock_services.world_quality.generate_relationships_with_quality.return_value = [
            ("just a string", mock_quality_scores),  # Invalid - not a dict
            (
                {"missing": "required_fields"},
                mock_quality_scores,
            ),  # Invalid - missing source/target
        ]

        with caplog.at_level(logging.WARNING):
            count = world_service._generate_relationships(
                sample_story_state, mock_world_db, mock_services
            )

        assert count == 0
        assert "Skipping invalid relationship" in caplog.text

    def test_passes_cancel_check_to_quality_service(
        self, world_service, mock_world_db, sample_story_state, mock_services
    ):
        """Test that cancel_check is passed to generate_relationships_with_quality."""
        mock_services.world_quality.generate_relationships_with_quality.return_value = []

        def my_cancel_check():
            """Stub cancel check that never cancels."""
            return False

        world_service._generate_relationships(
            sample_story_state, mock_world_db, mock_services, cancel_check=my_cancel_check
        )

        call_kwargs = mock_services.world_quality.generate_relationships_with_quality.call_args
        assert call_kwargs.kwargs["cancel_check"] is my_cancel_check


class TestBuildWorld:
    """Tests for build_world method."""

    def test_validates_inputs(self, world_service, mock_world_db, mock_services):
        """Test validates required inputs."""
        with pytest.raises(ValueError, match="state"):
            world_service.build_world(None, mock_world_db, mock_services)

    def test_requires_brief(self, world_service, mock_world_db, mock_services):
        """Test requires story brief to exist."""
        state = StoryState(id="no-brief")
        state.brief = None

        with pytest.raises(ValueError, match="no brief"):
            world_service.build_world(state, mock_world_db, mock_services)

    def test_full_build_keeps_existing(
        self, world_service, mock_world_db, sample_story_state, mock_services
    ):
        """Test full build keeps existing data and generates new."""
        # Pre-populate database with an existing character
        mock_world_db.add_entity("character", "ExistingChar", "Should be kept")

        # Setup mocks for generation
        mock_quality_scores = MagicMock()
        mock_quality_scores.to_dict.return_value = {}

        mock_services.world_quality.generate_locations_with_quality.return_value = []
        mock_services.world_quality.generate_factions_with_quality.return_value = []
        mock_services.world_quality.generate_items_with_quality.return_value = []
        mock_services.world_quality.generate_concepts_with_quality.return_value = []
        mock_services.world_quality.generate_relationships_with_quality.return_value = []

        counts = world_service.build_world(
            sample_story_state,
            mock_world_db,
            mock_services,
            WorldBuildOptions.full(),
        )

        # Should have extracted characters (existing + new from state)
        assert counts["characters"] >= 2
        # Existing character should still be there
        existing = mock_world_db.search_entities("ExistingChar")
        assert len(existing) > 0

    def test_full_rebuild_clears_first(
        self, world_service, mock_world_db, sample_story_state, mock_services
    ):
        """Test full rebuild clears database first."""
        # Pre-populate database
        mock_world_db.add_entity("character", "OldChar", "Should be deleted")

        world_service.build_world(
            sample_story_state,
            mock_world_db,
            mock_services,
            WorldBuildOptions.full_rebuild(),
        )

        # Old character should be gone
        old_chars = mock_world_db.search_entities("OldChar")
        assert len(old_chars) == 0

    def test_progress_callback(
        self, world_service, mock_world_db, sample_story_state, mock_services
    ):
        """Test progress callback is called."""
        progress_updates = []

        def on_progress(progress):
            """Collect progress updates into a list for later assertions."""
            progress_updates.append(progress)

        # Setup mocks for generation
        mock_quality_scores = MagicMock()
        mock_quality_scores.to_dict.return_value = {}

        mock_services.world_quality.generate_locations_with_quality.return_value = []
        mock_services.world_quality.generate_factions_with_quality.return_value = []
        mock_services.world_quality.generate_items_with_quality.return_value = []
        mock_services.world_quality.generate_concepts_with_quality.return_value = []
        mock_services.world_quality.generate_relationships_with_quality.return_value = []

        world_service.build_world(
            sample_story_state,
            mock_world_db,
            mock_services,
            WorldBuildOptions.full(),
            on_progress,
        )

        # Should have received progress updates
        assert len(progress_updates) > 0
        # Last update should be completion
        assert "complete" in progress_updates[-1].message.lower()

    def test_returns_counts(self, world_service, mock_world_db, sample_story_state, mock_services):
        """Test returns accurate entity counts."""
        mock_quality_scores = MagicMock()
        mock_quality_scores.to_dict.return_value = {}

        mock_services.world_quality.generate_locations_with_quality.return_value = [
            ({"name": "Loc1", "description": "A place"}, mock_quality_scores),
            ({"name": "Loc2", "description": "Another place"}, mock_quality_scores),
        ]
        mock_services.world_quality.generate_factions_with_quality.return_value = [
            ({"name": "Faction1", "description": "A group"}, mock_quality_scores),
        ]
        mock_services.world_quality.generate_items_with_quality.return_value = []
        mock_services.world_quality.generate_concepts_with_quality.return_value = []
        mock_services.world_quality.generate_relationships_with_quality.return_value = []

        counts = world_service.build_world(
            sample_story_state,
            mock_world_db,
            mock_services,
            WorldBuildOptions.full_rebuild(),
        )

        assert counts["characters"] == 2
        assert counts["locations"] == 2
        assert counts["factions"] == 1
        assert counts["items"] == 0
        assert counts["concepts"] == 0
        assert counts["relationships"] == 0

    def test_sets_world_template_id_on_state(
        self, world_service, mock_world_db, sample_story_state, mock_services
    ):
        """Test that world_template_id is set on state when template is provided."""
        # Create a world template
        template = WorldTemplate(
            id="test_fantasy",
            name="Test Fantasy",
            description="A test world template",
            is_builtin=False,
            genre="fantasy",
        )

        # Setup mocks
        mock_services.world_quality.generate_locations_with_quality.return_value = []
        mock_services.world_quality.generate_factions_with_quality.return_value = []
        mock_services.world_quality.generate_items_with_quality.return_value = []
        mock_services.world_quality.generate_concepts_with_quality.return_value = []
        mock_services.world_quality.generate_relationships_with_quality.return_value = []

        options = WorldBuildOptions(world_template=template)

        # Initially world_template_id should be None
        assert sample_story_state.world_template_id is None

        world_service.build_world(
            sample_story_state,
            mock_world_db,
            mock_services,
            options,
        )

        # After build, world_template_id should be set
        assert sample_story_state.world_template_id == "test_fantasy"

    def test_plot_quality_review_runs_when_plot_summary_exists(
        self, world_service, mock_world_db, sample_story_state, mock_services
    ):
        """Test that plot quality review updates the story state when plot_summary is present."""
        sample_story_state.plot_summary = "A hero's journey"
        sample_story_state.plot_points = [PlotPoint(description="The call", chapter=1)]

        refined_plot = PlotOutline(
            plot_summary="Refined journey",
            plot_points=[PlotPoint(description="The refined call", chapter=1)],
        )
        mock_plot_scores = MagicMock()
        mock_plot_scores.average = 8.5

        mock_services.world_quality.review_plot_quality = MagicMock(
            return_value=(refined_plot, mock_plot_scores, 2)
        )
        mock_services.world_quality.generate_locations_with_quality.return_value = []
        mock_services.world_quality.generate_factions_with_quality.return_value = []
        mock_services.world_quality.generate_items_with_quality.return_value = []
        mock_services.world_quality.generate_concepts_with_quality.return_value = []
        mock_services.world_quality.generate_relationships_with_quality.return_value = []

        world_service.build_world(
            sample_story_state,
            mock_world_db,
            mock_services,
            WorldBuildOptions.full(),
        )

        assert sample_story_state.plot_summary == "Refined journey"
        assert len(sample_story_state.plot_points) == 1
        assert sample_story_state.plot_points[0].description == "The refined call"
        mock_services.world_quality.review_plot_quality.assert_called_once()


class TestWorldBuildCancellation:
    """Tests for world build cancellation."""

    def test_build_world_cancellation_raises_error(
        self, world_service, sample_story_state, mock_world_db, mock_services
    ):
        """Test that cancellation event triggers GenerationCancelledError."""
        import threading

        from src.utils.exceptions import GenerationCancelledError

        # Create a cancellation event that's already set
        cancel_event = threading.Event()
        cancel_event.set()  # Pre-set to trigger immediate cancellation

        options = WorldBuildOptions(
            clear_existing=True,  # This step checks cancellation first
            generate_structure=True,
            generate_locations=True,
            generate_factions=True,
            generate_items=True,
            generate_concepts=True,
            generate_relationships=True,
            cancellation_event=cancel_event,
        )

        with pytest.raises(GenerationCancelledError, match="Generation cancelled"):
            world_service.build_world(
                sample_story_state,
                mock_world_db,
                mock_services,
                options,
            )

    def test_cancel_check_passed_to_factions(
        self, world_service, sample_story_state, mock_world_db, mock_services
    ):
        """Test that cancel_check is passed to generate_factions_with_quality."""
        import threading

        cancel_event = threading.Event()

        mock_services.world_quality.generate_locations_with_quality.return_value = []
        mock_services.world_quality.generate_factions_with_quality.return_value = []
        mock_services.world_quality.generate_items_with_quality.return_value = []
        mock_services.world_quality.generate_concepts_with_quality.return_value = []
        mock_services.world_quality.generate_relationships_with_quality.return_value = []

        options = WorldBuildOptions.full(cancellation_event=cancel_event)

        world_service.build_world(
            sample_story_state,
            mock_world_db,
            mock_services,
            options,
        )

        mock_services.world_quality.generate_factions_with_quality.assert_called_once()
        call_kwargs = mock_services.world_quality.generate_factions_with_quality.call_args
        assert call_kwargs.kwargs["cancel_check"] is not None
        # Verify it's the is_cancelled method bound to our options
        assert call_kwargs.kwargs["cancel_check"]() is False
        cancel_event.set()
        assert call_kwargs.kwargs["cancel_check"]() is True

    def test_cancel_check_passed_to_items(
        self, world_service, sample_story_state, mock_world_db, mock_services
    ):
        """Test that cancel_check is passed to generate_items_with_quality."""
        import threading

        cancel_event = threading.Event()

        mock_services.world_quality.generate_locations_with_quality.return_value = []
        mock_services.world_quality.generate_factions_with_quality.return_value = []
        mock_services.world_quality.generate_items_with_quality.return_value = []
        mock_services.world_quality.generate_concepts_with_quality.return_value = []
        mock_services.world_quality.generate_relationships_with_quality.return_value = []

        options = WorldBuildOptions.full(cancellation_event=cancel_event)

        world_service.build_world(
            sample_story_state,
            mock_world_db,
            mock_services,
            options,
        )

        mock_services.world_quality.generate_items_with_quality.assert_called_once()
        call_kwargs = mock_services.world_quality.generate_items_with_quality.call_args
        assert call_kwargs.kwargs["cancel_check"] is not None

    def test_cancel_check_passed_to_concepts(
        self, world_service, sample_story_state, mock_world_db, mock_services
    ):
        """Test that cancel_check is passed to generate_concepts_with_quality."""
        import threading

        cancel_event = threading.Event()

        mock_services.world_quality.generate_locations_with_quality.return_value = []
        mock_services.world_quality.generate_factions_with_quality.return_value = []
        mock_services.world_quality.generate_items_with_quality.return_value = []
        mock_services.world_quality.generate_concepts_with_quality.return_value = []
        mock_services.world_quality.generate_relationships_with_quality.return_value = []

        options = WorldBuildOptions.full(cancellation_event=cancel_event)

        world_service.build_world(
            sample_story_state,
            mock_world_db,
            mock_services,
            options,
        )

        mock_services.world_quality.generate_concepts_with_quality.assert_called_once()
        call_kwargs = mock_services.world_quality.generate_concepts_with_quality.call_args
        assert call_kwargs.kwargs["cancel_check"] is not None

    def test_location_cancel_check_stops_processing(
        self, world_service, sample_story_state, mock_world_db, mock_services
    ):
        """Test that cancel_check in location processing stops adding entities."""
        from src.memory.world_quality import LocationQualityScores

        call_count = 0

        def cancel_after_first():
            """Cancel after being called once (first location processed)."""
            nonlocal call_count
            call_count += 1
            return call_count > 1

        mock_scores = LocationQualityScores(
            atmosphere=8.0,
            significance=7.5,
            story_relevance=8.0,
            distinctiveness=7.0,
            feedback="Good",
        )
        mock_services.world_quality.generate_locations_with_quality.return_value = [
            ({"name": "Loc1", "description": "Place 1", "significance": "Important"}, mock_scores),
            ({"name": "Loc2", "description": "Place 2", "significance": "Minor"}, mock_scores),
            ({"name": "Loc3", "description": "Place 3", "significance": "Key"}, mock_scores),
        ]

        count = world_service._generate_locations(
            sample_story_state,
            mock_world_db,
            mock_services,
            cancel_check=cancel_after_first,
        )

        # Should have processed only the first location before cancel kicked in
        assert count == 1
        locations = mock_world_db.list_entities(entity_type="location")
        assert len(locations) == 1
        assert locations[0].name == "Loc1"

    def test_relationship_cancel_check_stops_processing(
        self, world_service, sample_story_state, mock_world_db, mock_services
    ):
        """Test that cancel_check in relationship processing stops adding entities."""
        mock_world_db.add_entity("character", "Hero", "The hero")
        mock_world_db.add_entity("character", "Villain", "The villain")
        mock_world_db.add_entity("character", "Mentor", "The mentor")

        mock_quality_scores = MagicMock()
        mock_quality_scores.average = 7.0

        mock_services.world_quality.generate_relationships_with_quality.return_value = [
            (
                {"source": "Hero", "target": "Villain", "relation_type": "enemies"},
                mock_quality_scores,
            ),
            (
                {"source": "Hero", "target": "Mentor", "relation_type": "allies"},
                mock_quality_scores,
            ),
        ]

        # Cancel immediately
        count = world_service._generate_relationships(
            sample_story_state,
            mock_world_db,
            mock_services,
            cancel_check=lambda: True,
        )

        assert count == 0


class TestPerTypeNameFiltering:
    """Tests for Issue 3: per-type entity name filtering in build functions."""

    def test_generate_factions_passes_faction_names_only(
        self, world_service, sample_story_state, mock_world_db, mock_services
    ):
        """Test _generate_factions passes only faction names, not all entity names."""
        # Add entities of different types
        mock_world_db.add_entity("location", "Dark Forest", "A dark forest")
        mock_world_db.add_entity("character", "Hero", "The hero")
        mock_world_db.add_entity("faction", "Knights", "A faction")

        mock_services.world_quality.generate_factions_with_quality.return_value = []

        world_service._generate_factions(sample_story_state, mock_world_db, mock_services)

        call_args = mock_services.world_quality.generate_factions_with_quality.call_args
        faction_names_arg = call_args[0][1]  # Second positional arg = existing_names
        # Should only contain faction names, not location or character names
        assert "Knights" in faction_names_arg
        assert "Dark Forest" not in faction_names_arg
        assert "Hero" not in faction_names_arg

    def test_generate_items_passes_item_names_only(
        self, world_service, sample_story_state, mock_world_db, mock_services
    ):
        """Test _generate_items passes only item names, not all entity names."""
        mock_world_db.add_entity("location", "Castle", "A castle")
        mock_world_db.add_entity("item", "Magic Sword", "A sword")
        mock_world_db.add_entity("concept", "Honor", "A concept")

        mock_services.world_quality.generate_items_with_quality.return_value = []

        world_service._generate_items(sample_story_state, mock_world_db, mock_services)

        call_args = mock_services.world_quality.generate_items_with_quality.call_args
        item_names_arg = call_args[0][1]  # Second positional arg
        assert "Magic Sword" in item_names_arg
        assert "Castle" not in item_names_arg
        assert "Honor" not in item_names_arg

    def test_generate_concepts_passes_concept_names_only(
        self, world_service, sample_story_state, mock_world_db, mock_services
    ):
        """Test _generate_concepts passes only concept names, not all entity names."""
        mock_world_db.add_entity("character", "Wizard", "A wizard")
        mock_world_db.add_entity("concept", "The Echo", "A concept about echoes")
        mock_world_db.add_entity("faction", "Mages Guild", "A faction")

        mock_services.world_quality.generate_concepts_with_quality.return_value = []

        world_service._generate_concepts(sample_story_state, mock_world_db, mock_services)

        call_args = mock_services.world_quality.generate_concepts_with_quality.call_args
        concept_names_arg = call_args[0][1]  # Second positional arg
        assert "The Echo" in concept_names_arg
        assert "Wizard" not in concept_names_arg
        assert "Mages Guild" not in concept_names_arg

    def test_generate_locations_passes_location_names_only(
        self, world_service, sample_story_state, mock_world_db, mock_services
    ):
        """Test _generate_locations passes only location names, not all entity names."""
        mock_world_db.add_entity("location", "Dark Forest", "A dark forest")
        mock_world_db.add_entity("character", "Hero", "The hero")
        mock_world_db.add_entity("item", "Magic Ring", "A ring")

        mock_services.world_quality.generate_locations_with_quality.return_value = []

        world_service._generate_locations(sample_story_state, mock_world_db, mock_services)

        call_args = mock_services.world_quality.generate_locations_with_quality.call_args
        location_names_arg = call_args[0][1]  # Second positional arg
        assert "Dark Forest" in location_names_arg
        assert "Hero" not in location_names_arg
        assert "Magic Ring" not in location_names_arg


class TestLocationQualityRefinement:
    """Tests for Issue 6: locations use quality refinement in world build."""

    def test_generate_locations_calls_quality_service(
        self, world_service, sample_story_state, mock_world_db, mock_services
    ):
        """Test _generate_locations uses world_quality service, not story service."""
        mock_services.world_quality.generate_locations_with_quality.return_value = []

        world_service._generate_locations(sample_story_state, mock_world_db, mock_services)

        mock_services.world_quality.generate_locations_with_quality.assert_called_once()

    def test_generate_locations_stores_quality_scores(
        self, world_service, sample_story_state, mock_world_db, mock_services
    ):
        """Test _generate_locations stores quality scores in entity attributes."""
        from src.memory.world_quality import LocationQualityScores

        mock_scores = LocationQualityScores(
            atmosphere=8.0,
            significance=7.5,
            story_relevance=8.0,
            distinctiveness=7.0,
            feedback="Good location",
        )
        mock_services.world_quality.generate_locations_with_quality.return_value = [
            (
                {"name": "Castle", "description": "A grand castle", "significance": "Key"},
                mock_scores,
            ),
        ]

        count = world_service._generate_locations(sample_story_state, mock_world_db, mock_services)

        assert count == 1
        locations = mock_world_db.list_entities(entity_type="location")
        assert len(locations) == 1
        assert locations[0].name == "Castle"

    def test_generate_locations_passes_cancel_check(
        self, world_service, sample_story_state, mock_world_db, mock_services
    ):
        """Test _generate_locations passes cancel_check to quality service."""
        mock_services.world_quality.generate_locations_with_quality.return_value = []

        def my_cancel_check():
            """Stub cancel check that never cancels."""
            return False

        world_service._generate_locations(
            sample_story_state, mock_world_db, mock_services, cancel_check=my_cancel_check
        )

        call_kwargs = mock_services.world_quality.generate_locations_with_quality.call_args
        assert call_kwargs.kwargs["cancel_check"] is my_cancel_check


class TestFuzzyEntityNameMatching:
    """Tests for fuzzy entity name matching in relationship generation."""

    @pytest.fixture
    def mock_rel_scores(self):
        """Create mock quality scores for relationship tests."""
        scores = MagicMock()
        scores.average = 7.5
        return scores

    def test_exact_match(
        self, world_service, sample_story_state, mock_world_db, mock_services, mock_rel_scores
    ):
        """Test exact name match still works."""
        mock_world_db.add_entity("character", "Kai Chen", "A hacker")
        mock_world_db.add_entity("concept", "Echoes of the Network", "A concept")

        mock_services.world_quality.generate_relationships_with_quality.return_value = [
            (
                {
                    "source": "Kai Chen",
                    "target": "Echoes of the Network",
                    "relation_type": "explores",
                },
                mock_rel_scores,
            ),
        ]

        count = world_service._generate_relationships(
            sample_story_state, mock_world_db, mock_services
        )
        assert count == 1

    def test_fuzzy_match_the_prefix(
        self, world_service, sample_story_state, mock_world_db, mock_services, mock_rel_scores
    ):
        """Test fuzzy match handles 'The' prefix added by LLM."""
        mock_world_db.add_entity("character", "Kai Chen", "A hacker")
        mock_world_db.add_entity("concept", "Echoes of the Network", "A concept")

        # LLM added "The" prefix that doesn't exist in the actual entity name
        mock_services.world_quality.generate_relationships_with_quality.return_value = [
            (
                {
                    "source": "The Echoes of the Network",
                    "target": "Kai Chen",
                    "relation_type": "influences",
                },
                mock_rel_scores,
            ),
        ]

        count = world_service._generate_relationships(
            sample_story_state, mock_world_db, mock_services
        )
        assert count == 1
        rels = mock_world_db.list_relationships()
        assert len(rels) == 1

    def test_fuzzy_match_case_insensitive(
        self, world_service, sample_story_state, mock_world_db, mock_services, mock_rel_scores
    ):
        """Test fuzzy match handles case differences."""
        mock_world_db.add_entity("faction", "The Synchroflux", "A faction")
        mock_world_db.add_entity("character", "Kai Chen", "A hacker")

        mock_services.world_quality.generate_relationships_with_quality.return_value = [
            (
                {
                    "source": "the synchroflux",
                    "target": "Kai Chen",
                    "relation_type": "recruits",
                },
                mock_rel_scores,
            ),
        ]

        count = world_service._generate_relationships(
            sample_story_state, mock_world_db, mock_services
        )
        assert count == 1

    def test_no_match_still_warns(
        self, world_service, sample_story_state, mock_world_db, mock_services, mock_rel_scores
    ):
        """Test that completely wrong names still produce a warning."""
        mock_world_db.add_entity("character", "Kai Chen", "A hacker")

        mock_services.world_quality.generate_relationships_with_quality.return_value = [
            (
                {
                    "source": "Nonexistent Entity",
                    "target": "Kai Chen",
                    "relation_type": "knows",
                },
                mock_rel_scores,
            ),
        ]

        count = world_service._generate_relationships(
            sample_story_state, mock_world_db, mock_services
        )
        assert count == 0

    def test_fuzzy_match_a_prefix(
        self, world_service, sample_story_state, mock_world_db, mock_services, mock_rel_scores
    ):
        """Test fuzzy match handles 'A' prefix added by LLM."""
        mock_world_db.add_entity("item", "Cursed Blade", "A sword")
        mock_world_db.add_entity("character", "Kai Chen", "A hacker")

        mock_services.world_quality.generate_relationships_with_quality.return_value = [
            (
                {
                    "source": "A Cursed Blade",
                    "target": "Kai Chen",
                    "relation_type": "wields",
                },
                mock_rel_scores,
            ),
        ]

        count = world_service._generate_relationships(
            sample_story_state, mock_world_db, mock_services
        )
        assert count == 1

    def test_fuzzy_match_an_prefix(
        self, world_service, sample_story_state, mock_world_db, mock_services, mock_rel_scores
    ):
        """Test fuzzy match handles 'An' prefix added by LLM."""
        mock_world_db.add_entity("concept", "Ancient Promise", "A concept")
        mock_world_db.add_entity("character", "Kai Chen", "A hacker")

        mock_services.world_quality.generate_relationships_with_quality.return_value = [
            (
                {
                    "source": "An Ancient Promise",
                    "target": "Kai Chen",
                    "relation_type": "binds",
                },
                mock_rel_scores,
            ),
        ]

        count = world_service._generate_relationships(
            sample_story_state, mock_world_db, mock_services
        )
        assert count == 1

    def test_fuzzy_match_whitespace_normalization(
        self, world_service, sample_story_state, mock_world_db, mock_services, mock_rel_scores
    ):
        """Test fuzzy match handles extra whitespace in LLM-generated names."""
        mock_world_db.add_entity("location", "Dark Forest", "A forest")
        mock_world_db.add_entity("character", "Kai Chen", "A hacker")

        mock_services.world_quality.generate_relationships_with_quality.return_value = [
            (
                {
                    "source": "  Dark   Forest  ",
                    "target": "Kai Chen",
                    "relation_type": "explores",
                },
                mock_rel_scores,
            ),
        ]

        count = world_service._generate_relationships(
            sample_story_state, mock_world_db, mock_services
        )
        assert count == 1


class TestFindEntityByNameAmbiguity:
    """Tests for _find_entity_by_name ambiguity guard."""

    def test_exact_match_returns_entity(self):
        """Exact name match should return the entity directly."""
        from src.services.world_service._build import _find_entity_by_name

        e1 = MagicMock()
        e1.name = "Dark Forest"
        result = _find_entity_by_name([e1], "Dark Forest")
        assert result is e1

    def test_single_fuzzy_match_returns_entity(self):
        """Single fuzzy match (e.g. article difference) should return the entity."""
        from src.services.world_service._build import _find_entity_by_name

        e1 = MagicMock()
        e1.name = "The Dark Forest"
        result = _find_entity_by_name([e1], "Dark Forest")
        assert result is e1

    def test_no_match_returns_none(self):
        """No match should return None."""
        from src.services.world_service._build import _find_entity_by_name

        e1 = MagicMock()
        e1.name = "Bright Meadow"
        result = _find_entity_by_name([e1], "Dark Forest")
        assert result is None

    def test_ambiguous_fuzzy_match_returns_none(self, caplog):
        """Multiple fuzzy matches should return None and log a warning."""
        import logging

        from src.services.world_service._build import _find_entity_by_name

        e1 = MagicMock()
        e1.name = "The Guild"
        e2 = MagicMock()
        e2.name = "A Guild"

        with caplog.at_level(logging.WARNING):
            result = _find_entity_by_name([e1, e2], "Guild")

        assert result is None
        assert "Ambiguous fuzzy match" in caplog.text


class TestImplicitRelationshipTracking:
    """Tests for implicit relationship logging and counting."""

    def test_faction_generation_logs_implicit_relationships(
        self, world_service, mock_world_db, sample_story_state, mock_services, caplog
    ):
        """Test faction generation logs each implicit based_in relationship."""
        import logging

        # Add a location first
        mock_world_db.add_entity("location", "Castle", "A grand castle")

        mock_quality_scores = MagicMock()
        mock_quality_scores.to_dict.return_value = {}

        mock_services.world_quality.generate_factions_with_quality.return_value = [
            (
                {"name": "Knights", "description": "Noble knights", "base_location": "Castle"},
                mock_quality_scores,
            ),
        ]

        with caplog.at_level(logging.INFO):
            faction_count, implicit_count = world_service._generate_factions(
                sample_story_state, mock_world_db, mock_services
            )

        assert faction_count == 1
        assert implicit_count == 1
        assert "Created implicit 'based_in' relationship: Knights -> Castle" in caplog.text
        assert "1 implicit based_in relationship(s)" in caplog.text

    def test_faction_generation_counts_multiple_implicit_relationships(
        self, world_service, mock_world_db, sample_story_state, mock_services
    ):
        """Test faction generation correctly counts multiple implicit relationships."""
        mock_world_db.add_entity("location", "Castle", "A castle")
        mock_world_db.add_entity("location", "Forest", "A forest")

        mock_quality_scores = MagicMock()
        mock_quality_scores.to_dict.return_value = {}

        mock_services.world_quality.generate_factions_with_quality.return_value = [
            (
                {"name": "Knights", "description": "Noble", "base_location": "Castle"},
                mock_quality_scores,
            ),
            (
                {"name": "Rangers", "description": "Scouts", "base_location": "Forest"},
                mock_quality_scores,
            ),
            (
                {"name": "Mages", "description": "Wizards", "base_location": "Tower"},
                mock_quality_scores,
            ),
        ]

        faction_count, implicit_count = world_service._generate_factions(
            sample_story_state, mock_world_db, mock_services
        )

        assert faction_count == 3
        # Only Castle and Forest exist as locations; Tower doesn't
        assert implicit_count == 2

    def test_faction_generation_no_implicit_when_no_base_location(
        self, world_service, mock_world_db, sample_story_state, mock_services
    ):
        """Test no implicit relationships when factions have no base_location."""
        mock_quality_scores = MagicMock()
        mock_quality_scores.to_dict.return_value = {}

        mock_services.world_quality.generate_factions_with_quality.return_value = [
            (
                {"name": "Wanderers", "description": "Nomadic", "base_location": ""},
                mock_quality_scores,
            ),
        ]

        faction_count, implicit_count = world_service._generate_factions(
            sample_story_state, mock_world_db, mock_services
        )

        assert faction_count == 1
        assert implicit_count == 0

    def test_build_summary_includes_implicit_relationships(
        self, world_service, mock_world_db, sample_story_state, mock_services, caplog
    ):
        """Test build summary includes implicit relationship count."""
        import logging

        mock_world_db.add_entity("location", "Castle", "A castle")

        mock_quality_scores = MagicMock()
        mock_quality_scores.to_dict.return_value = {}

        mock_services.world_quality.generate_locations_with_quality.return_value = []
        mock_services.world_quality.generate_factions_with_quality.return_value = [
            (
                {"name": "Knights", "description": "Noble", "base_location": "Castle"},
                mock_quality_scores,
            ),
        ]
        mock_services.world_quality.generate_items_with_quality.return_value = []
        mock_services.world_quality.generate_concepts_with_quality.return_value = []
        mock_services.world_quality.generate_relationships_with_quality.return_value = []

        with caplog.at_level(logging.INFO):
            counts = world_service.build_world(
                sample_story_state,
                mock_world_db,
                mock_services,
                WorldBuildOptions.full(),
            )

        # 1 from character extraction (Mentor->Hero) + 1 from faction (Knights->Castle)
        assert counts["implicit_relationships"] == 2
        assert "0 explicit + 2 implicit" in caplog.text

    def test_build_counts_dict_has_implicit_relationships_key(
        self, world_service, mock_world_db, sample_story_state, mock_services
    ):
        """Test build_world returns counts dict with implicit_relationships key."""
        mock_services.world_quality.generate_locations_with_quality.return_value = []
        mock_services.world_quality.generate_factions_with_quality.return_value = []
        mock_services.world_quality.generate_items_with_quality.return_value = []
        mock_services.world_quality.generate_concepts_with_quality.return_value = []
        mock_services.world_quality.generate_relationships_with_quality.return_value = []

        counts = world_service.build_world(
            sample_story_state,
            mock_world_db,
            mock_services,
            WorldBuildOptions.full(),
        )

        assert "implicit_relationships" in counts
        # 1 from character extraction (Mentor->Hero), 0 from factions
        assert counts["implicit_relationships"] == 1


class TestRelationshipQualityRefinement:
    """Tests for relationship generation using quality refinement pipeline."""

    def test_generate_relationships_uses_quality_service(
        self, world_service, mock_world_db, sample_story_state, mock_services
    ):
        """Test _generate_relationships uses world_quality service, not story service."""
        mock_services.world_quality.generate_relationships_with_quality.return_value = []

        world_service._generate_relationships(sample_story_state, mock_world_db, mock_services)

        mock_services.world_quality.generate_relationships_with_quality.assert_called_once()


class TestRecoverOrphans:
    """Tests for _recover_orphans function."""

    def test_recover_orphans_no_orphans_returns_zero(
        self, world_service, mock_world_db, sample_story_state, mock_services
    ):
        """Test returns 0 when there are no orphan entities."""
        from src.services.world_service._build import _recover_orphans

        # Add 2 entities with a relationship between them
        id1 = mock_world_db.add_entity("character", "Alice", "A brave adventurer")
        id2 = mock_world_db.add_entity("character", "Bob", "A wise mage")
        mock_world_db.add_relationship(id1, id2, "allies")

        result = _recover_orphans(world_service, sample_story_state, mock_world_db, mock_services)

        assert result == 0

    def test_recover_orphans_generates_relationship(
        self, world_service, mock_world_db, sample_story_state, mock_services
    ):
        """Test generates a relationship for an orphan entity."""
        from src.services.world_service._build import _recover_orphans

        # Add 3 entities; only first two are connected, third is orphan
        id1 = mock_world_db.add_entity("character", "Alice", "A brave adventurer")
        id2 = mock_world_db.add_entity("character", "Bob", "A wise mage")
        mock_world_db.add_entity("character", "Charlie", "A lonely wanderer")
        mock_world_db.add_relationship(id1, id2, "allies")

        # Mock generate_relationship_with_quality to return a relationship
        # connecting the orphan to one of the connected entities
        mock_scores = MagicMock()
        mock_scores.average = 7.5
        mock_services.world_quality.generate_relationship_with_quality = MagicMock(
            return_value=(
                {
                    "source": "Charlie",
                    "target": "Alice",
                    "relation_type": "friends",
                    "description": "Charlie befriends Alice on the road",
                },
                mock_scores,
                2,
            )
        )

        result = _recover_orphans(world_service, sample_story_state, mock_world_db, mock_services)

        assert result == 1
        # Verify Charlie now has a relationship
        charlie = mock_world_db.search_entities("Charlie", entity_type="character")[0]
        rels = mock_world_db.get_relationships(charlie.id)
        assert len(rels) == 1
        assert rels[0].relation_type == "friends"

    def test_recover_orphans_handles_generation_failure(
        self, world_service, mock_world_db, sample_story_state, mock_services
    ):
        """Test returns 0 when relationship generation raises an error."""
        from src.services.world_service._build import _recover_orphans
        from src.utils.exceptions import WorldGenerationError

        # Add 2 entities with no relationships (both are orphans)
        mock_world_db.add_entity("character", "Alice", "A brave adventurer")
        mock_world_db.add_entity("character", "Bob", "A wise mage")

        # Mock generate_relationship_with_quality to raise WorldGenerationError
        mock_services.world_quality.generate_relationship_with_quality = MagicMock(
            side_effect=WorldGenerationError("test failure")
        )

        result = _recover_orphans(world_service, sample_story_state, mock_world_db, mock_services)

        assert result == 0

    def test_recover_orphans_cancel_check_stops_iteration(
        self, world_service, mock_world_db, sample_story_state, mock_services, caplog
    ):
        """Test cancel_check breaks the recovery loop after first iteration."""
        import logging

        from src.services.world_service._build import _recover_orphans

        # Add 3 orphan entities (no relationships) so attempts > 1
        mock_world_db.add_entity("character", "Alice", "A brave adventurer")
        mock_world_db.add_entity("character", "Bob", "A wise mage")
        mock_world_db.add_entity("character", "Charlie", "A lonely wanderer")

        call_count = 0

        def cancel_after_first():
            """Return True after the first call to cancel."""
            nonlocal call_count
            call_count += 1
            return call_count > 1

        mock_scores = MagicMock()
        mock_scores.average = 7.5
        mock_services.world_quality.generate_relationship_with_quality = MagicMock(
            return_value=(
                {
                    "source": "Alice",
                    "target": "Bob",
                    "relation_type": "friends",
                    "description": "They are friends",
                },
                mock_scores,
                1,
            )
        )

        with caplog.at_level(logging.INFO):
            result = _recover_orphans(
                world_service,
                sample_story_state,
                mock_world_db,
                mock_services,
                cancel_check=cancel_after_first,
            )

        # First iteration succeeds (cancel_check returns False), then second
        # iteration is cancelled (cancel_check returns True) before generating
        assert result == 1
        assert "Orphan recovery cancelled after" in caplog.text

    def test_recover_orphans_empty_relationship_continues(
        self, world_service, mock_world_db, sample_story_state, mock_services, caplog
    ):
        """Test empty or missing source/target in generated relationship logs warning and continues."""
        import logging

        from src.services.world_service._build import _recover_orphans

        # Add 2 orphan entities (no relationships)
        mock_world_db.add_entity("character", "Alice", "A brave adventurer")
        mock_world_db.add_entity("character", "Bob", "A wise mage")

        mock_scores = MagicMock()
        mock_scores.average = 5.0

        # Return an empty dict (no "source" or "target" keys)
        mock_services.world_quality.generate_relationship_with_quality = MagicMock(
            return_value=({}, mock_scores, 1)
        )

        with caplog.at_level(logging.WARNING):
            result = _recover_orphans(
                world_service, sample_story_state, mock_world_db, mock_services
            )

        assert result == 0
        assert "empty relationship generated" in caplog.text

    def test_recover_orphans_unresolvable_entities_logs_warning(
        self, world_service, mock_world_db, sample_story_state, mock_services, caplog
    ):
        """Test unresolvable entity names in generated relationship logs warning."""
        import logging

        from src.services.world_service._build import _recover_orphans

        # Add 2 orphan entities (no relationships)
        mock_world_db.add_entity("character", "Alice", "A brave adventurer")
        mock_world_db.add_entity("character", "Bob", "A wise mage")

        mock_scores = MagicMock()
        mock_scores.average = 7.0

        # Return a relationship with names that don't exist in the world db
        mock_services.world_quality.generate_relationship_with_quality = MagicMock(
            return_value=(
                {
                    "source": "Nonexistent Hero",
                    "target": "Unknown Villain",
                    "relation_type": "enemies",
                    "description": "They are enemies",
                },
                mock_scores,
                1,
            )
        )

        with caplog.at_level(logging.WARNING):
            result = _recover_orphans(
                world_service, sample_story_state, mock_world_db, mock_services
            )

        assert result == 0
        assert "could not resolve entities for" in caplog.text


class TestBuildWorldOrphanRecovery:
    """Tests for orphan recovery integration in build_world."""

    def test_build_world_orphan_recovery_adds_to_relationship_count(
        self, world_service, mock_world_db, sample_story_state, mock_services
    ):
        """Test build_world adds orphan recovery count to relationships total."""
        from unittest.mock import patch

        mock_services.world_quality.generate_locations_with_quality.return_value = []
        mock_services.world_quality.generate_factions_with_quality.return_value = []
        mock_services.world_quality.generate_items_with_quality.return_value = []
        mock_services.world_quality.generate_concepts_with_quality.return_value = []
        mock_services.world_quality.generate_relationships_with_quality.return_value = []

        # Patch _recover_orphans to return a positive count
        with patch(
            "src.services.world_service._build._recover_orphans",
            return_value=3,
        ):
            counts = world_service.build_world(
                sample_story_state,
                mock_world_db,
                mock_services,
                WorldBuildOptions.full(),
            )

        # Relationships should include the orphan recovery count
        assert counts["relationships"] == 3


class TestHealthScoreCircularPenalty:
    """Tests for the split circular penalty in health score calculation."""

    def test_health_score_hierarchical_cycle_heavy_penalty(self):
        """Test hierarchical cycles receive a heavy penalty of 5 per cycle."""
        from src.memory.world_health import WorldHealthMetrics

        metrics = WorldHealthMetrics(
            total_entities=10,
            total_relationships=15,
            hierarchical_circular_count=2,
            mutual_circular_count=0,
            average_quality=8.0,
            relationship_density=1.5,
        )
        score = metrics.calculate_health_score()
        # hierarchical_penalty = 2 * 5 = 10
        # structural = 100 - 10 + 10 (density bonus) = 100.0
        # quality = 8.0 * 10 = 80.0
        # final = 100.0 * 0.6 + 80.0 * 0.4 = 60.0 + 32.0 = 92.0
        assert score == 92.0

    def test_health_score_mutual_cycle_light_penalty(self):
        """Test mutual cycles receive a light penalty of 1 per cycle."""
        from src.memory.world_health import WorldHealthMetrics

        metrics = WorldHealthMetrics(
            total_entities=10,
            total_relationships=15,
            hierarchical_circular_count=0,
            mutual_circular_count=3,
            average_quality=8.0,
            relationship_density=1.5,
        )
        score = metrics.calculate_health_score()
        # mutual_penalty = 3 * 1 = 3
        # structural = 100 - 3 + 10 = 107 -> capped at 100.0
        # quality = 8.0 * 10 = 80.0
        # final = 100.0 * 0.6 + 80.0 * 0.4 = 92.0
        assert score == 92.0

    def test_health_score_mixed_cycles(self):
        """Test combined hierarchical and mutual cycle penalties."""
        from src.memory.world_health import WorldHealthMetrics

        metrics = WorldHealthMetrics(
            total_entities=10,
            total_relationships=15,
            hierarchical_circular_count=1,
            mutual_circular_count=2,
            average_quality=8.0,
            relationship_density=1.5,
        )
        score = metrics.calculate_health_score()
        # hierarchical_penalty = 1 * 5 = 5, mutual_penalty = 2 * 1 = 2, total = 7
        # structural = 100 - 7 + 10 = 103 -> capped at 100.0
        # quality = 8.0 * 10 = 80.0
        # final = 100.0 * 0.6 + 80.0 * 0.4 = 92.0
        assert score == 92.0

    def test_health_score_many_hierarchical_cycles_capped(self):
        """Test hierarchical penalty is capped at 25."""
        from src.memory.world_health import WorldHealthMetrics

        metrics = WorldHealthMetrics(
            total_entities=10,
            total_relationships=15,
            hierarchical_circular_count=10,
            mutual_circular_count=0,
            average_quality=8.0,
            relationship_density=1.5,
        )
        score = metrics.calculate_health_score()
        # hierarchical_penalty = min(10 * 5, 25) = 25
        # structural = 100 - 25 + 10 = 85.0
        # quality = 8.0 * 10 = 80.0
        # final = 85.0 * 0.6 + 80.0 * 0.4 = 51.0 + 32.0 = 83.0
        assert score == 83.0

    def test_health_score_no_cycles_no_penalty(self):
        """Test no circular penalty when there are no cycles."""
        from src.memory.world_health import WorldHealthMetrics

        metrics = WorldHealthMetrics(
            total_entities=10,
            total_relationships=15,
            hierarchical_circular_count=0,
            mutual_circular_count=0,
            average_quality=8.0,
            relationship_density=1.5,
        )
        score = metrics.calculate_health_score()
        # no circular penalty
        # structural = 100 + 10 = 110 -> capped at 100.0
        # quality = 8.0 * 10 = 80.0
        # final = 100.0 * 0.6 + 80.0 * 0.4 = 92.0
        assert score == 92.0

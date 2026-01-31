"""Tests for WorldService.build_world and related methods."""

from unittest.mock import MagicMock

import pytest

from src.memory.story_state import Chapter, Character, StoryBrief, StoryState
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
    services.story.generate_relationships = MagicMock(return_value=[])
    services.world_quality.generate_locations_with_quality = MagicMock(return_value=[])
    services.world_quality.generate_factions_with_quality = MagicMock(return_value=[])
    services.world_quality.generate_items_with_quality = MagicMock(return_value=[])
    services.world_quality.generate_concepts_with_quality = MagicMock(return_value=[])
    return services


class TestCalculateTotalSteps:
    """Tests for _calculate_total_steps method."""

    def test_full_options(self, world_service):
        """Test step count for full options."""
        options = WorldBuildOptions.full()
        # Base (2: character extraction + completion)
        # + structure (1) + locations (1) + factions (1)
        # + items (1) + concepts (1) + relationships (1) = 8
        assert world_service._calculate_total_steps(options) == 8

    def test_full_rebuild_options(self, world_service):
        """Test step count for full rebuild options."""
        options = WorldBuildOptions.full_rebuild()
        # Base (2: character extraction + completion)
        # + clear (1) + structure (1) + locations (1) + factions (1)
        # + items (1) + concepts (1) + relationships (1) = 9
        assert world_service._calculate_total_steps(options) == 9

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
        count = world_service._extract_characters_to_world(sample_story_state, mock_world_db)

        assert count == 2
        entities = mock_world_db.list_entities(entity_type="character")
        assert len(entities) == 2
        names = [e.name for e in entities]
        assert "Hero" in names
        assert "Mentor" in names

    def test_skips_existing_characters(self, world_service, mock_world_db, sample_story_state):
        """Test skips characters that already exist."""
        # Pre-add one character
        mock_world_db.add_entity("character", "Hero", "Pre-existing")

        count = world_service._extract_characters_to_world(sample_story_state, mock_world_db)

        # Only Mentor should be added
        assert count == 1
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

        count = world_service._extract_characters_to_world(state, mock_world_db)
        assert count == 0


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

        count = world_service._generate_factions(sample_story_state, mock_world_db, mock_services)

        assert count == 1
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

        world_service._generate_factions(sample_story_state, mock_world_db, mock_services)

        # Check relationship was created
        relationships = mock_world_db.list_relationships()
        assert len(relationships) == 1
        assert relationships[0].relation_type == "based_in"

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
            count = world_service._generate_factions(
                sample_story_state, mock_world_db, mock_services
            )

        assert count == 0
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
        """Test generates and adds relationships."""
        # Add some entities first
        mock_world_db.add_entity("character", "Hero", "The hero")
        mock_world_db.add_entity("character", "Villain", "The villain")

        mock_services.story.generate_relationships.return_value = [
            {
                "source": "Hero",
                "target": "Villain",
                "relation_type": "enemies",
                "description": "Mortal enemies",
            },
        ]

        count = world_service._generate_relationships(
            sample_story_state, mock_world_db, mock_services
        )

        assert count == 1
        relationships = mock_world_db.list_relationships()
        assert len(relationships) == 1
        assert relationships[0].relation_type == "enemies"

    def test_skips_invalid_relationships(
        self, world_service, mock_world_db, sample_story_state, mock_services
    ):
        """Test skips relationships with missing entities."""
        mock_world_db.add_entity("character", "Hero", "The hero")
        # Note: Villain is NOT added

        mock_services.story.generate_relationships.return_value = [
            {
                "source": "Hero",
                "target": "Villain",
                "relation_type": "enemies",
            },  # Villain doesn't exist
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

        mock_services.story.generate_relationships.return_value = [
            "just a string",  # Invalid - not a dict
            {"missing": "required_fields"},  # Invalid - missing source/target
        ]

        with caplog.at_level(logging.WARNING):
            count = world_service._generate_relationships(
                sample_story_state, mock_world_db, mock_services
            )

        assert count == 0
        assert "Skipping invalid relationship" in caplog.text


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
        mock_services.story.generate_relationships.return_value = []

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
        mock_services.story.generate_relationships.return_value = []

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
        mock_services.story.generate_relationships.return_value = []

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
        mock_services.story.generate_relationships.return_value = []

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
        mock_services.story.generate_relationships.return_value = []

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
        mock_services.story.generate_relationships.return_value = []

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
        mock_services.story.generate_relationships.return_value = []

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

        mock_services.story.generate_relationships.return_value = [
            {"source": "Hero", "target": "Villain", "relation_type": "enemies"},
            {"source": "Hero", "target": "Mentor", "relation_type": "allies"},
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

    def test_exact_match(self, world_service, sample_story_state, mock_world_db, mock_services):
        """Test exact name match still works."""
        mock_world_db.add_entity("character", "Kai Chen", "A hacker")
        mock_world_db.add_entity("concept", "Echoes of the Network", "A concept")

        mock_services.story.generate_relationships.return_value = [
            {
                "source": "Kai Chen",
                "target": "Echoes of the Network",
                "relation_type": "explores",
            },
        ]

        count = world_service._generate_relationships(
            sample_story_state, mock_world_db, mock_services
        )
        assert count == 1

    def test_fuzzy_match_the_prefix(
        self, world_service, sample_story_state, mock_world_db, mock_services
    ):
        """Test fuzzy match handles 'The' prefix added by LLM."""
        mock_world_db.add_entity("character", "Kai Chen", "A hacker")
        mock_world_db.add_entity("concept", "Echoes of the Network", "A concept")

        # LLM added "The" prefix that doesn't exist in the actual entity name
        mock_services.story.generate_relationships.return_value = [
            {
                "source": "The Echoes of the Network",
                "target": "Kai Chen",
                "relation_type": "influences",
            },
        ]

        count = world_service._generate_relationships(
            sample_story_state, mock_world_db, mock_services
        )
        assert count == 1
        rels = mock_world_db.list_relationships()
        assert len(rels) == 1

    def test_fuzzy_match_case_insensitive(
        self, world_service, sample_story_state, mock_world_db, mock_services
    ):
        """Test fuzzy match handles case differences."""
        mock_world_db.add_entity("faction", "The Synchroflux", "A faction")
        mock_world_db.add_entity("character", "Kai Chen", "A hacker")

        mock_services.story.generate_relationships.return_value = [
            {
                "source": "the synchroflux",
                "target": "Kai Chen",
                "relation_type": "recruits",
            },
        ]

        count = world_service._generate_relationships(
            sample_story_state, mock_world_db, mock_services
        )
        assert count == 1

    def test_no_match_still_warns(
        self, world_service, sample_story_state, mock_world_db, mock_services
    ):
        """Test that completely wrong names still produce a warning."""
        mock_world_db.add_entity("character", "Kai Chen", "A hacker")

        mock_services.story.generate_relationships.return_value = [
            {
                "source": "Nonexistent Entity",
                "target": "Kai Chen",
                "relation_type": "knows",
            },
        ]

        count = world_service._generate_relationships(
            sample_story_state, mock_world_db, mock_services
        )
        assert count == 0

    def test_fuzzy_match_a_prefix(
        self, world_service, sample_story_state, mock_world_db, mock_services
    ):
        """Test fuzzy match handles 'A' prefix added by LLM."""
        mock_world_db.add_entity("item", "Cursed Blade", "A sword")
        mock_world_db.add_entity("character", "Kai Chen", "A hacker")

        mock_services.story.generate_relationships.return_value = [
            {
                "source": "A Cursed Blade",
                "target": "Kai Chen",
                "relation_type": "wields",
            },
        ]

        count = world_service._generate_relationships(
            sample_story_state, mock_world_db, mock_services
        )
        assert count == 1

    def test_fuzzy_match_an_prefix(
        self, world_service, sample_story_state, mock_world_db, mock_services
    ):
        """Test fuzzy match handles 'An' prefix added by LLM."""
        mock_world_db.add_entity("concept", "Ancient Promise", "A concept")
        mock_world_db.add_entity("character", "Kai Chen", "A hacker")

        mock_services.story.generate_relationships.return_value = [
            {
                "source": "An Ancient Promise",
                "target": "Kai Chen",
                "relation_type": "binds",
            },
        ]

        count = world_service._generate_relationships(
            sample_story_state, mock_world_db, mock_services
        )
        assert count == 1

    def test_fuzzy_match_whitespace_normalization(
        self, world_service, sample_story_state, mock_world_db, mock_services
    ):
        """Test fuzzy match handles extra whitespace in LLM-generated names."""
        mock_world_db.add_entity("location", "Dark Forest", "A forest")
        mock_world_db.add_entity("character", "Kai Chen", "A hacker")

        mock_services.story.generate_relationships.return_value = [
            {
                "source": "  Dark   Forest  ",
                "target": "Kai Chen",
                "relation_type": "explores",
            },
        ]

        count = world_service._generate_relationships(
            sample_story_state, mock_world_db, mock_services
        )
        assert count == 1

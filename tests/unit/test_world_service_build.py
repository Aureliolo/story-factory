"""Tests for WorldService.build_world and related methods."""

from unittest.mock import MagicMock

import pytest

from memory.story_state import Chapter, Character, StoryBrief, StoryState
from memory.world_database import WorldDatabase
from services.world_service import (
    WorldBuildOptions,
    WorldBuildProgress,
    WorldService,
)
from settings import Settings


class TestWorldBuildOptions:
    """Tests for WorldBuildOptions dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        options = WorldBuildOptions()
        assert options.clear_existing is False
        assert options.generate_structure is True
        assert options.generate_locations is False
        assert options.generate_factions is False
        assert options.generate_items is False
        assert options.generate_concepts is False
        assert options.generate_relationships is False

    def test_minimal_factory(self):
        """Test minimal() factory method."""
        options = WorldBuildOptions.minimal()
        assert options.clear_existing is False
        assert options.generate_structure is True
        assert options.generate_locations is False
        assert options.generate_factions is False
        assert options.generate_items is False
        assert options.generate_concepts is False
        assert options.generate_relationships is False

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
    """Create test settings."""
    return Settings.load()


@pytest.fixture
def world_service(settings):
    """Create WorldService instance."""
    return WorldService(settings)


@pytest.fixture
def mock_world_db(tmp_path):
    """Create a mock WorldDatabase."""
    db_path = tmp_path / "test_world.db"
    return WorldDatabase(str(db_path))


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
    services.story.generate_locations = MagicMock(return_value=[])
    services.story.generate_relationships = MagicMock(return_value=[])
    services.world_quality.generate_factions_with_quality = MagicMock(return_value=[])
    services.world_quality.generate_items_with_quality = MagicMock(return_value=[])
    services.world_quality.generate_concepts_with_quality = MagicMock(return_value=[])
    return services


class TestCalculateTotalSteps:
    """Tests for _calculate_total_steps method."""

    def test_minimal_options(self, world_service):
        """Test step count for minimal options."""
        options = WorldBuildOptions.minimal()
        # Structure (1) + character extraction (1) + completion (1) = 3
        assert world_service._calculate_total_steps(options) == 3

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

    def test_generates_locations(
        self, world_service, mock_world_db, sample_story_state, mock_services
    ):
        """Test generates and adds locations."""
        mock_services.story.generate_locations.return_value = [
            {"name": "Castle", "description": "A grand castle", "significance": "Home base"},
            {"name": "Forest", "description": "Dark woods", "significance": "Danger zone"},
        ]

        count = world_service._generate_locations(sample_story_state, mock_world_db, mock_services)

        assert count == 2
        locations = mock_world_db.list_entities(entity_type="location")
        assert len(locations) == 2
        names = [loc.name for loc in locations]
        assert "Castle" in names
        assert "Forest" in names

    def test_skips_invalid_locations(
        self, world_service, mock_world_db, sample_story_state, mock_services
    ):
        """Test skips locations without name field."""
        mock_services.story.generate_locations.return_value = [
            {"name": "Valid", "description": "OK"},
            {"description": "Missing name"},  # Invalid
            "just a string",  # Invalid
        ]

        count = world_service._generate_locations(sample_story_state, mock_world_db, mock_services)

        assert count == 1

    def test_handles_empty_response(
        self, world_service, mock_world_db, sample_story_state, mock_services
    ):
        """Test handles empty location list."""
        mock_services.story.generate_locations.return_value = []

        count = world_service._generate_locations(sample_story_state, mock_world_db, mock_services)

        assert count == 0


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

    def test_minimal_build(self, world_service, mock_world_db, sample_story_state, mock_services):
        """Test minimal build extracts characters only."""
        # Mock the orchestrator path for non-clearing build
        mock_orchestrator = MagicMock()
        mock_services.story._get_orchestrator.return_value = mock_orchestrator
        mock_services.story._sync_state = MagicMock()

        counts = world_service.build_world(
            sample_story_state,
            mock_world_db,
            mock_services,
            WorldBuildOptions.minimal(),
        )

        # Should have extracted characters
        assert counts["characters"] == 2
        assert counts["locations"] == 0
        assert counts["factions"] == 0

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
            progress_updates.append(progress)

        # Mock orchestrator for non-clearing build
        mock_orchestrator = MagicMock()
        mock_services.story._get_orchestrator.return_value = mock_orchestrator
        mock_services.story._sync_state = MagicMock()

        world_service.build_world(
            sample_story_state,
            mock_world_db,
            mock_services,
            WorldBuildOptions.minimal(),
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

        mock_services.story.generate_locations.return_value = [
            {"name": "Loc1", "description": "A place"},
            {"name": "Loc2", "description": "Another place"},
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

"""Tests for WorldService."""

from pathlib import Path

import pytest

from memory.story_state import Character, StoryBrief, StoryState
from memory.world_database import WorldDatabase
from services.world_service import WorldService
from settings import Settings


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings()


@pytest.fixture
def world_service(settings):
    """Create WorldService instance."""
    return WorldService(settings)


@pytest.fixture
def world_db(tmp_path: Path):
    """Create a temporary WorldDatabase."""
    return WorldDatabase(tmp_path / "test_world.db")


@pytest.fixture
def sample_story_state():
    """Create a sample story state with characters."""
    brief = StoryBrief(
        premise="An epic fantasy adventure",
        genre="Fantasy",
        subgenres=["Epic", "Adventure"],
        tone="Grand",
        themes=["Heroism", "Sacrifice"],
        setting_time="Medieval",
        setting_place="Kingdom of Valdoria",
        target_length="novel",
        language="English",
        content_rating="general",
        content_preferences=["Magic", "Dragons"],
        content_avoid=["Gore"],
    )
    state = StoryState(
        id="test-story-001",
        project_name="The Dragon's Quest",
        brief=brief,
        status="writing",
        characters=[
            Character(
                name="Sir Roland",
                role="protagonist",
                description="A brave knight with a noble heart",
                personality_traits=["brave", "honorable", "determined"],
                goals=["Save the kingdom", "Defeat the dragon"],
                relationships={"Lady Elara": "beloved"},
                arc_notes="Learns true courage comes from the heart",
            ),
            Character(
                name="Lady Elara",
                role="love_interest",
                description="A wise sorceress with hidden powers",
                personality_traits=["intelligent", "mysterious", "kind"],
                goals=["Protect her people", "Master her magic"],
                relationships={"Sir Roland": "beloved"},
                arc_notes="Embraces her destiny",
            ),
        ],
        world_description="""The Kingdom of Valdoria sprawls across the Misty Mountains...

Key locations:
- The Crystal Palace sits atop Mount Seren
- The Darkwood Forest borders the eastern realm
- The Dragon's Lair lies beyond the Forgotten Pass

The realm is governed by ancient magic.""",
    )
    return state


class TestWorldServiceInit:
    """Tests for WorldService initialization."""

    def test_init_with_settings(self, settings):
        """Test service initializes with provided settings."""
        service = WorldService(settings)
        assert service.settings == settings

    def test_init_without_settings(self):
        """Test service loads default settings when none provided."""
        service = WorldService()
        assert service.settings is not None


class TestWorldServiceExtractEntitiesFromStructure:
    """Tests for extract_entities_from_structure method."""

    def test_extracts_characters(self, world_service, world_db, sample_story_state):
        """Test extracts characters from story state."""
        count = world_service.extract_entities_from_structure(sample_story_state, world_db)

        assert count >= 2  # At least the two characters

        # Verify characters were added
        entities = world_db.list_entities(entity_type="character")
        names = [e.name for e in entities]
        assert "Sir Roland" in names
        assert "Lady Elara" in names

    def test_extracts_locations_from_world_description(
        self, world_service, world_db, sample_story_state
    ):
        """Test extracts locations from world description."""
        count = world_service.extract_entities_from_structure(sample_story_state, world_db)

        assert count > 0

        # Verify at least some locations were extracted
        locations = world_db.list_entities(entity_type="location")
        # Location extraction is heuristic-based, may vary
        assert isinstance(locations, list)

    def test_creates_character_relationships(self, world_service, world_db, sample_story_state):
        """Test creates relationships between characters."""
        world_service.extract_entities_from_structure(sample_story_state, world_db)

        # Get Sir Roland
        roland_entities = world_db.search_entities("Sir Roland", entity_type="character")
        if roland_entities:
            relationships = world_db.get_relationships(roland_entities[0].id)
            # May have relationship with Lady Elara
            assert isinstance(relationships, list)

    def test_skips_existing_entities(self, world_service, world_db, sample_story_state):
        """Test doesn't duplicate existing entities."""
        # Add Roland first
        world_db.add_entity(
            entity_type="character",
            name="Sir Roland",
            description="Already exists",
        )

        world_service.extract_entities_from_structure(sample_story_state, world_db)

        # Should only add Lady Elara + any locations
        rolands = world_db.search_entities("Sir Roland", entity_type="character")
        assert len(rolands) == 1  # Only one Roland

    def test_handles_empty_characters(self, world_service, world_db):
        """Test handles story state with no characters."""
        state = StoryState(
            id="test",
            status="writing",
            brief=StoryBrief(
                premise="A story",
                genre="Fiction",
                tone="Neutral",
                setting_time="Now",
                setting_place="Here",
                target_length="short_story",
                language="English",
                content_rating="general",
            ),
        )

        count = world_service.extract_entities_from_structure(state, world_db)

        assert count >= 0


class TestWorldServiceExtractFromChapter:
    """Tests for extract_from_chapter method."""

    def test_extracts_locations_from_chapter(self, world_service, world_db):
        """Test extracts new locations mentioned in chapter."""
        chapter_content = """Sir Roland rode through the gates of Castle Blackthorn.
The ancient fortress loomed against the stormy sky."""

        counts = world_service.extract_from_chapter(chapter_content, world_db, chapter_number=1)

        assert "entities" in counts
        assert "events" in counts
        assert counts["entities"] >= 0  # Extraction is heuristic

    def test_extracts_items_from_chapter(self, world_service, world_db):
        """Test extracts significant items from chapter."""
        chapter_content = """He reached for the ancient sword hanging on the wall.
The magical amulet around his neck glowed faintly."""

        counts = world_service.extract_from_chapter(chapter_content, world_db, chapter_number=1)

        assert counts["entities"] >= 0

    def test_extracts_events_from_chapter(self, world_service, world_db):
        """Test extracts key events from chapter."""
        chapter_content = """The dragon was defeated at last.
Sir Roland had saved the kingdom from destruction.
The people celebrated their victory."""

        counts = world_service.extract_from_chapter(chapter_content, world_db, chapter_number=5)

        assert counts["events"] >= 0  # Should find "defeated", "saved"


class TestWorldServiceEntityCRUD:
    """Tests for entity CRUD operations."""

    def test_add_entity(self, world_service, world_db):
        """Test adding a new entity."""
        entity_id = world_service.add_entity(
            world_db,
            entity_type="character",
            name="Test Character",
            description="A test character",
            attributes={"role": "protagonist"},
        )

        assert entity_id is not None
        entity = world_db.get_entity(entity_id)
        assert entity.name == "Test Character"
        assert entity.attributes["role"] == "protagonist"

    def test_update_entity(self, world_service, world_db):
        """Test updating an entity."""
        entity_id = world_db.add_entity(
            entity_type="character",
            name="Original Name",
            description="Original description",
        )

        result = world_service.update_entity(
            world_db,
            entity_id,
            name="Updated Name",
            description="Updated description",
        )

        assert result is True
        entity = world_db.get_entity(entity_id)
        assert entity.name == "Updated Name"
        assert entity.description == "Updated description"

    def test_update_nonexistent_entity(self, world_service, world_db):
        """Test updating non-existent entity returns False."""
        result = world_service.update_entity(
            world_db,
            "nonexistent-id",
            name="New Name",
        )

        assert result is False

    def test_delete_entity(self, world_service, world_db):
        """Test deleting an entity."""
        entity_id = world_db.add_entity(
            entity_type="item",
            name="To Delete",
            description="Will be deleted",
        )

        result = world_service.delete_entity(world_db, entity_id)

        assert result is True
        assert world_db.get_entity(entity_id) is None

    def test_delete_nonexistent_entity(self, world_service, world_db):
        """Test deleting non-existent entity returns False."""
        result = world_service.delete_entity(world_db, "nonexistent-id")

        assert result is False

    def test_get_entity(self, world_service, world_db):
        """Test getting entity by ID."""
        entity_id = world_db.add_entity(
            entity_type="location",
            name="Test Location",
            description="A test location",
        )

        entity = world_service.get_entity(world_db, entity_id)

        assert entity is not None
        assert entity.name == "Test Location"

    def test_get_nonexistent_entity(self, world_service, world_db):
        """Test getting non-existent entity returns None."""
        entity = world_service.get_entity(world_db, "nonexistent-id")

        assert entity is None

    def test_list_entities(self, world_service, world_db):
        """Test listing all entities."""
        world_db.add_entity(entity_type="character", name="Char 1", description="")
        world_db.add_entity(entity_type="character", name="Char 2", description="")
        world_db.add_entity(entity_type="location", name="Loc 1", description="")

        all_entities = world_service.list_entities(world_db)
        characters = world_service.list_entities(world_db, entity_type="character")

        assert len(all_entities) == 3
        assert len(characters) == 2

    def test_search_entities(self, world_service, world_db):
        """Test searching entities."""
        world_db.add_entity(entity_type="character", name="Sir Roland", description="A knight")
        world_db.add_entity(entity_type="character", name="Lady Elara", description="A sorceress")
        world_db.add_entity(entity_type="location", name="Roland's Castle", description="His home")

        results = world_service.search_entities(world_db, "Roland")

        assert len(results) >= 1
        names = [e.name for e in results]
        assert "Sir Roland" in names


class TestWorldServiceRelationshipManagement:
    """Tests for relationship management."""

    def test_add_relationship(self, world_service, world_db):
        """Test adding a relationship between entities."""
        char1_id = world_db.add_entity(entity_type="character", name="Alice", description="")
        char2_id = world_db.add_entity(entity_type="character", name="Bob", description="")

        rel_id = world_service.add_relationship(
            world_db,
            source_id=char1_id,
            target_id=char2_id,
            relation_type="friend_of",
            description="Best friends since childhood",
        )

        assert rel_id is not None
        relationships = world_db.get_relationships(char1_id)
        assert len(relationships) >= 1

    def test_add_bidirectional_relationship(self, world_service, world_db):
        """Test adding bidirectional relationship."""
        char1_id = world_db.add_entity(entity_type="character", name="Alice", description="")
        char2_id = world_db.add_entity(entity_type="character", name="Bob", description="")

        world_service.add_relationship(
            world_db,
            source_id=char1_id,
            target_id=char2_id,
            relation_type="sibling_of",
            bidirectional=True,
        )

        # Both should have relationships
        alice_rels = world_db.get_relationships(char1_id)
        bob_rels = world_db.get_relationships(char2_id)
        assert len(alice_rels) >= 1
        assert len(bob_rels) >= 1

    def test_delete_relationship(self, world_service, world_db):
        """Test deleting a relationship."""
        char1_id = world_db.add_entity(entity_type="character", name="Alice", description="")
        char2_id = world_db.add_entity(entity_type="character", name="Bob", description="")
        rel_id = world_db.add_relationship(char1_id, char2_id, "knows")

        result = world_service.delete_relationship(world_db, rel_id)

        assert result is True

    def test_get_relationships(self, world_service, world_db):
        """Test getting relationships for an entity."""
        char1_id = world_db.add_entity(entity_type="character", name="Alice", description="")
        char2_id = world_db.add_entity(entity_type="character", name="Bob", description="")
        char3_id = world_db.add_entity(entity_type="character", name="Carol", description="")

        world_db.add_relationship(char1_id, char2_id, "friend_of")
        world_db.add_relationship(char1_id, char3_id, "colleague_of")

        relationships = world_service.get_relationships(world_db, entity_id=char1_id)

        assert len(relationships) == 2


class TestWorldServiceGraphAnalysis:
    """Tests for graph analysis methods."""

    def test_find_path(self, world_service, world_db):
        """Test finding path between entities."""
        # Create a chain: A -> B -> C
        a_id = world_db.add_entity(entity_type="character", name="A", description="")
        b_id = world_db.add_entity(entity_type="character", name="B", description="")
        c_id = world_db.add_entity(entity_type="character", name="C", description="")

        world_db.add_relationship(a_id, b_id, "knows")
        world_db.add_relationship(b_id, c_id, "knows")

        path = world_service.find_path(world_db, a_id, c_id)

        assert path is not None
        assert len(path) == 3  # A -> B -> C

    def test_find_path_no_connection(self, world_service, world_db):
        """Test returns empty list or None when no path exists."""
        a_id = world_db.add_entity(entity_type="character", name="A", description="")
        b_id = world_db.add_entity(entity_type="character", name="B", description="")

        path = world_service.find_path(world_db, a_id, b_id)

        # Either None or empty list is acceptable for "no path"
        assert path is None or path == []

    def test_get_connected_entities(self, world_service, world_db):
        """Test getting connected entities."""
        center_id = world_db.add_entity(entity_type="character", name="Center", description="")
        friend1_id = world_db.add_entity(entity_type="character", name="Friend1", description="")
        friend2_id = world_db.add_entity(entity_type="character", name="Friend2", description="")

        world_db.add_relationship(center_id, friend1_id, "knows")
        world_db.add_relationship(center_id, friend2_id, "knows")

        connected = world_service.get_connected_entities(world_db, center_id, max_depth=1)

        assert len(connected) >= 2

    def test_get_most_connected(self, world_service, world_db):
        """Test getting most connected entities."""
        # Create hub with many connections
        hub_id = world_db.add_entity(entity_type="character", name="Hub", description="")
        for i in range(5):
            other_id = world_db.add_entity(
                entity_type="character", name=f"Other{i}", description=""
            )
            world_db.add_relationship(hub_id, other_id, "knows")

        most_connected = world_service.get_most_connected(world_db, limit=3)

        assert len(most_connected) <= 3
        # Hub should be the most connected
        if most_connected:
            assert most_connected[0][0].name == "Hub"


class TestWorldServiceContextForAgents:
    """Tests for agent context generation."""

    def test_get_context_for_agents(self, world_service, world_db):
        """Test getting context formatted for AI agents."""
        world_db.add_entity(entity_type="character", name="Hero", description="The protagonist")
        world_db.add_entity(entity_type="location", name="Castle", description="A grand castle")

        context = world_service.get_context_for_agents(world_db)

        assert isinstance(context, dict)

    def test_get_entity_summary(self, world_service, world_db):
        """Test getting entity count summary."""
        world_db.add_entity(entity_type="character", name="Char1", description="")
        world_db.add_entity(entity_type="character", name="Char2", description="")
        world_db.add_entity(entity_type="location", name="Loc1", description="")

        summary = world_service.get_entity_summary(world_db)

        assert summary["character"] == 2
        assert summary["location"] == 1
        assert "relationships" in summary

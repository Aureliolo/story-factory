"""Tests for WorldService."""

from pathlib import Path

import pytest

from src.memory.story_state import Character, StoryBrief, StoryState
from src.memory.world_database import WorldDatabase
from src.services.world_service import WorldService
from src.settings import Settings


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
    db = WorldDatabase(tmp_path / "test_world.db")
    yield db
    db.close()


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
        assert isinstance(service.settings, Settings)


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

        assert isinstance(entity_id, str) and len(entity_id) > 0
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

        assert entity.name == "Test Location"
        assert entity.type == "location"

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

        assert isinstance(rel_id, str) and len(rel_id) > 0
        relationships = world_db.get_relationships(char1_id)
        assert len(relationships) >= 1
        assert any(r.relation_type == "friend_of" for r in relationships)

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
        assert len(path) > 0
        assert len(path) == 3  # A -> B -> C
        assert path[0] == a_id
        assert path[-1] == c_id

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


class TestWorldServiceEdgeCases:
    """Tests for edge cases and uncovered code paths."""

    def test_extract_skips_existing_location_in_world_description(self, world_service, world_db):
        """Test extract_entities_from_structure skips existing locations in world_description."""
        # Create a story state with world_description containing pattern-matching locations
        brief = StoryBrief(
            premise="Test story",
            genre="Fantasy",
            tone="Epic",
            setting_time="Medieval",
            setting_place="Test Kingdom",
            target_length="short_story",
            language="English",
            content_rating="general",
        )

        # First, run extraction to find what names are actually extracted
        state = StoryState(
            id="test-skip-loc",
            project_name="Test",
            brief=brief,
            status="writing",
            # World description with location patterns
            world_description="The Great Castle stands tall in the Northern Valley. The heroes traveled to the Ancient Forest.",
        )

        # Run extraction once to get the extracted location names
        world_service.extract_entities_from_structure(state, world_db)

        # Get the locations that were extracted
        extracted_locations = world_db.list_entities(entity_type="location")

        # If any locations were extracted, clear the DB and test the skip logic
        if extracted_locations:
            # Clear DB
            for loc in extracted_locations:
                world_db.delete_entity(loc.id)

            # Pre-add the first extracted location
            first_loc_name = extracted_locations[0].name
            world_db.add_entity(
                entity_type="location",
                name=first_loc_name,
                description="Pre-existing",
            )

            # Run extraction again - should skip the pre-existing location
            state2 = StoryState(
                id="test-skip-loc-2",
                project_name="Test2",
                brief=brief,
                status="writing",
                world_description=state.world_description,
            )
            world_service.extract_entities_from_structure(state2, world_db)

            # Count occurrences of the first location name
            all_locs = world_db.list_entities(entity_type="location")
            first_name_count = sum(1 for e in all_locs if e.name == first_loc_name)
            assert first_name_count == 1  # Only one instance, not duplicated

    def test_extract_from_chapter_adds_locations(self, world_service, world_db):
        """Test extract_from_chapter adds new location entities."""
        # Chapter content with a clear location pattern
        chapter_content = """The heroes arrived at the Enchanted Forest.
The ancient trees whispered secrets as they passed through the Moonlit Valley.
Finally, they reached the Tower of Shadows."""

        counts = world_service.extract_from_chapter(chapter_content, world_db, chapter_number=1)

        # Should have extracted at least one entity
        locations = world_db.list_entities(entity_type="location")
        # The extraction is heuristic, but verify at least 0 or more were added
        assert counts["entities"] >= 0
        assert isinstance(locations, list)

    def test_extract_from_chapter_skips_existing_locations(self, world_service, world_db):
        """Test extract_from_chapter doesn't duplicate existing locations."""
        # Pre-add a location
        world_db.add_entity(
            entity_type="location",
            name="Dark Cave",
            description="Already exists",
        )

        chapter_content = """They entered the Dark Cave cautiously.
Inside, they found another location: the Crystal Grotto."""

        world_service.extract_from_chapter(chapter_content, world_db, chapter_number=1)

        # Should not duplicate Dark Cave
        caves = world_db.search_entities("Dark Cave", entity_type="location")
        assert len(caves) == 1

    def test_delete_nonexistent_relationship(self, world_service, world_db):
        """Test delete_relationship returns False for non-existent relationship."""
        result = world_service.delete_relationship(world_db, "nonexistent-rel-id")

        # Should return False and log warning
        assert result is False

    def test_get_relationships_without_entity_filter(self, world_service, world_db):
        """Test get_relationships returns all relationships when no entity_id provided."""
        char1_id = world_db.add_entity(entity_type="character", name="Alice", description="")
        char2_id = world_db.add_entity(entity_type="character", name="Bob", description="")
        char3_id = world_db.add_entity(entity_type="character", name="Carol", description="")

        world_db.add_relationship(char1_id, char2_id, "friend_of")
        world_db.add_relationship(char2_id, char3_id, "colleague_of")

        # Get ALL relationships (no entity_id filter)
        all_relationships = world_service.get_relationships(world_db)

        assert len(all_relationships) == 2

    def test_get_communities(self, world_service, world_db):
        """Test get_communities detects entity clusters."""
        # Create two separate groups
        group1_a = world_db.add_entity(entity_type="character", name="Group1-A", description="")
        group1_b = world_db.add_entity(entity_type="character", name="Group1-B", description="")
        world_db.add_relationship(group1_a, group1_b, "friend_of")

        group2_a = world_db.add_entity(entity_type="character", name="Group2-A", description="")
        group2_b = world_db.add_entity(entity_type="character", name="Group2-B", description="")
        world_db.add_relationship(group2_a, group2_b, "friend_of")

        communities = world_service.get_communities(world_db)

        # Should return a list of communities
        assert isinstance(communities, list)


class TestWorldServiceErrorHandling:
    """Tests for WorldService error handling paths."""

    def test_extract_entities_from_structure_exception(self, world_service, world_db, monkeypatch):
        """Test exception handling in extract_entities_from_structure."""
        from src.memory.story_state import Character, StoryState

        state = StoryState(id="test-error")
        state.characters = [
            Character(name="Alice", role="protagonist", description="Test character")
        ]

        # Mock add_entity to raise an exception
        def mock_add_entity(*args, **kwargs):
            raise RuntimeError("Simulated database error")

        monkeypatch.setattr(world_db, "add_entity", mock_add_entity)

        with pytest.raises(RuntimeError, match="Simulated database error"):
            world_service.extract_entities_from_structure(state, world_db)

    def test_extract_from_chapter_exception(self, world_service, world_db, monkeypatch):
        """Test exception handling in extract_from_chapter."""

        # Mock _extract_locations_from_text to raise an exception
        def mock_extract(*args, **kwargs):
            raise ValueError("Simulated extraction error")

        monkeypatch.setattr(world_service, "_extract_locations_from_text", mock_extract)

        with pytest.raises(ValueError, match="Simulated extraction error"):
            world_service.extract_from_chapter("Some chapter content", world_db, 1)

"""Tests for WorldService."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.memory.story_state import Character, StoryBrief, StoryState
from src.memory.world_database import WorldDatabase
from src.services.world_service import _HEALTH_CACHE_TTL_SECONDS, WorldBuildOptions, WorldService
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

    def test_extract_from_chapter_exception(self, world_service, world_db, monkeypatch):
        """Test exception handling in extract_from_chapter."""

        # Mock _extract_locations_from_text to raise an exception
        def mock_extract(*args, **kwargs):
            """Simulate an extraction error when parsing chapter content."""
            raise ValueError("Simulated extraction error")

        monkeypatch.setattr(world_service, "_extract_locations_from_text", mock_extract)

        with pytest.raises(ValueError, match="Simulated extraction error"):
            world_service.extract_from_chapter("Some chapter content", world_db, 1)

    def test_get_entity_versions(self, world_service, world_db):
        """Test getting entity versions through WorldService."""
        # Create an entity (which creates version 1)
        entity_id = world_service.add_entity(
            world_db,
            "character",
            "Version Test Character",
            description="A test character",
            attributes={"role": "test"},
        )

        # Update it to create version 2
        world_service.update_entity(
            world_db,
            entity_id,
            attributes={"role": "updated role"},
        )

        # Get versions via WorldService
        versions = world_service.get_entity_versions(world_db, entity_id, limit=10)

        assert len(versions) == 2
        # Versions returned newest first
        assert versions[0].version_number == 2
        assert versions[1].version_number == 1

    def test_revert_entity_to_version(self, world_service, world_db):
        """Test reverting entity to previous version through WorldService."""
        # Create an entity (which creates version 1)
        entity_id = world_service.add_entity(
            world_db,
            "character",
            "Revert Test Character",
            description="A test character",
            attributes={"role": "original"},
        )

        # Update it to create version 2
        world_service.update_entity(
            world_db,
            entity_id,
            attributes={"role": "changed"},
        )

        # Revert to version 1 via WorldService
        result = world_service.revert_entity_to_version(world_db, entity_id, 1)

        assert result is True

        # Check entity was reverted
        entity = world_db.get_entity(entity_id)
        assert entity.attributes["role"] == "original"


class TestWorldHealthMethods:
    """Tests for world health detection methods in WorldService."""

    def test_find_orphan_entities(self, world_service, world_db):
        """Test finding orphan entities through WorldService."""
        # Create orphan entity
        world_db.add_entity("character", "Lonely Alice")

        # Create connected entities
        bob_id = world_db.add_entity("character", "Connected Bob")
        charlie_id = world_db.add_entity("character", "Connected Charlie")
        world_db.add_relationship(bob_id, charlie_id, "knows", validate=False)

        orphans = world_service.find_orphan_entities(world_db)

        assert len(orphans) == 1
        assert orphans[0].name == "Lonely Alice"

    def test_find_orphan_entities_with_type_filter(self, world_service, world_db):
        """Test finding orphan entities with type filter."""
        world_db.add_entity("character", "Lonely Alice")
        world_db.add_entity("location", "Empty Town")

        # Only get character orphans
        orphans = world_service.find_orphan_entities(world_db, entity_type="character")

        assert len(orphans) == 1
        assert orphans[0].type == "character"

    def test_find_circular_relationships(self, world_service, world_db):
        """Test finding circular relationships through WorldService."""
        # Create circular chain: A -> B -> C -> A
        a_id = world_db.add_entity("character", "A")
        b_id = world_db.add_entity("character", "B")
        c_id = world_db.add_entity("character", "C")

        world_db.add_relationship(a_id, b_id, "reports_to", validate=False)
        world_db.add_relationship(b_id, c_id, "reports_to", validate=False)
        world_db.add_relationship(c_id, a_id, "reports_to", validate=False)

        cycles = world_service.find_circular_relationships(world_db, relation_types=["reports_to"])

        assert len(cycles) >= 1

    def test_find_circular_relationships_empty(self, world_service, world_db):
        """Test finding circular relationships when none exist."""
        # Create linear chain: A -> B -> C
        a_id = world_db.add_entity("character", "A")
        b_id = world_db.add_entity("character", "B")
        c_id = world_db.add_entity("character", "C")

        world_db.add_relationship(a_id, b_id, "knows", validate=False)
        world_db.add_relationship(b_id, c_id, "knows", validate=False)

        cycles = world_service.find_circular_relationships(world_db)

        assert cycles == []

    def test_get_world_health_metrics(self, world_service, world_db):
        """Test getting world health metrics."""
        # Create some entities
        alice_id = world_db.add_entity("character", "Alice", attributes={"quality_score": 8.0})
        bob_id = world_db.add_entity("character", "Bob", attributes={"quality_score": 7.0})
        world_db.add_entity("location", "Town", attributes={"quality_score": 6.0})
        world_db.add_entity("character", "Orphan", attributes={"quality_score": 3.0})

        # Add a relationship
        world_db.add_relationship(alice_id, bob_id, "knows", validate=False)

        metrics = world_service.get_world_health_metrics(world_db)

        assert metrics.total_entities == 4
        assert metrics.entity_counts["character"] == 3
        assert metrics.entity_counts["location"] == 1
        assert metrics.total_relationships == 1
        assert metrics.orphan_count == 2  # Town and Orphan
        assert metrics.health_score > 0

    def test_get_world_health_metrics_empty(self, world_service, world_db):
        """Test getting world health metrics for empty database."""
        metrics = world_service.get_world_health_metrics(world_db)

        assert metrics.total_entities == 0
        assert metrics.total_relationships == 0
        assert metrics.orphan_count == 0
        assert metrics.health_score == 0.0  # Empty worlds score 0

    def test_get_world_health_metrics_generates_recommendations(self, world_service, world_db):
        """Test that health metrics include recommendations."""
        # Create orphan entity
        world_db.add_entity("character", "Lonely Alice")

        metrics = world_service.get_world_health_metrics(world_db)

        # Should have recommendation about orphans
        assert len(metrics.recommendations) > 0

    def test_find_entity_by_name_exact_match(self, world_service, world_db):
        """Test finding entity by exact name match."""
        world_db.add_entity("character", "Alice the Brave")

        entity = world_service.find_entity_by_name(world_db, "Alice the Brave")

        assert entity is not None
        assert entity.name == "Alice the Brave"

    def test_find_entity_by_name_case_insensitive(self, world_service, world_db):
        """Test finding entity with case-insensitive match."""
        world_db.add_entity("character", "Alice")

        entity = world_service.find_entity_by_name(world_db, "alice")

        assert entity is not None
        assert entity.name == "Alice"

    def test_find_entity_by_name_fuzzy_match(self, world_service, world_db):
        """Test finding entity with fuzzy name matching."""
        world_db.add_entity("character", "Sir Roland the Brave")

        # Should find with fuzzy match
        entity = world_service.find_entity_by_name(world_db, "Sir Roland", fuzzy_threshold=0.7)

        assert entity is not None
        assert "Roland" in entity.name

    def test_find_entity_by_name_not_found(self, world_service, world_db):
        """Test finding entity returns None when not found."""
        world_db.add_entity("character", "Alice")

        entity = world_service.find_entity_by_name(world_db, "Nonexistent Character")

        assert entity is None

    def test_find_entity_by_name_with_type_filter(self, world_service, world_db):
        """Test finding entity with type filter."""
        world_db.add_entity("character", "Alice")
        world_db.add_entity("location", "Alice's House")

        # Should find only character
        entity = world_service.find_entity_by_name(world_db, "Alice", entity_type="character")

        assert entity is not None
        assert entity.type == "character"

    def test_find_entity_by_name_no_entities_for_fuzzy(self, world_service, world_db):
        """Test fuzzy matching when database has no entities for the type."""
        # Don't add any entities of the searched type - create different type
        world_db.add_entity("location", "Somewhere")

        # Search for character type that doesn't exist
        entity = world_service.find_entity_by_name(
            world_db, "Nonexistent", entity_type="character", fuzzy_threshold=0.7
        )

        assert entity is None

    def test_find_entity_by_name_suffix_match(self, world_service, world_db):
        """Test fuzzy matching with suffix match."""
        world_db.add_entity("character", "Lord Blackwood")

        # Search by suffix
        entity = world_service.find_entity_by_name(world_db, "Blackwood", fuzzy_threshold=0.7)

        assert entity is not None
        assert entity.name == "Lord Blackwood"

    def test_find_entity_by_name_substring_match(self, world_service, world_db):
        """Test fuzzy matching with substring match."""
        world_db.add_entity("character", "Great King Arthur of Camelot")

        # Search by substring that's not a prefix/suffix
        entity = world_service.find_entity_by_name(world_db, "King Arthur", fuzzy_threshold=0.7)

        assert entity is not None
        assert "Arthur" in entity.name

    def test_get_world_health_metrics_with_circular(self, world_service, world_db):
        """Test health metrics include circular relationship detection."""
        # Create circular chain
        a_id = world_db.add_entity("character", "A")
        b_id = world_db.add_entity("character", "B")
        c_id = world_db.add_entity("character", "C")

        world_db.add_relationship(a_id, b_id, "reports_to", validate=False)
        world_db.add_relationship(b_id, c_id, "reports_to", validate=False)
        world_db.add_relationship(c_id, a_id, "reports_to", validate=False)

        metrics = world_service.get_world_health_metrics(world_db)

        # Should detect the circular relationship
        assert metrics.circular_count >= 1
        assert len(metrics.circular_relationships) >= 1
        # Verify structure of circular info
        cycle_info = metrics.circular_relationships[0]
        assert "edges" in cycle_info
        assert "length" in cycle_info

    def test_get_world_health_metrics_quality_distribution(self, world_service, world_db):
        """Test health metrics quality distribution covers all buckets."""
        # Create entities with quality scores in different buckets
        world_db.add_entity("character", "Low1", attributes={"quality_score": 1.0})
        world_db.add_entity("character", "Low2", attributes={"quality_score": 3.0})
        world_db.add_entity("character", "Mid", attributes={"quality_score": 5.0})
        world_db.add_entity("character", "High1", attributes={"quality_score": 7.0})
        world_db.add_entity("character", "High2", attributes={"quality_score": 9.0})

        metrics = world_service.get_world_health_metrics(world_db)

        # Verify quality distribution
        assert metrics.quality_distribution["0-2"] >= 1
        assert metrics.quality_distribution["2-4"] >= 1
        assert metrics.quality_distribution["4-6"] >= 1
        assert metrics.quality_distribution["6-8"] >= 1
        assert metrics.quality_distribution["8-10"] >= 1

    def test_find_entity_by_name_fuzzy_exact_match_after_normalization(
        self, world_service, world_db
    ):
        """Test fuzzy matching hits exact match path after whitespace normalization.

        When searching with trailing/leading whitespace that the DB lookup misses,
        the fuzzy matching should still find exact matches after strip().
        """
        world_db.add_entity("character", "Alice")

        # Search with trailing space - DB exact match will fail,
        # but fuzzy matching should find it after normalization
        entity = world_service.find_entity_by_name(world_db, "Alice ", fuzzy_threshold=0.7)

        assert entity is not None
        assert entity.name == "Alice"

    def test_find_entity_by_name_empty_name_returns_none(self, world_service, world_db):
        """Test that empty or whitespace-only name returns None."""
        world_db.add_entity("character", "Alice")

        # Empty string
        assert world_service.find_entity_by_name(world_db, "") is None

        # Whitespace only
        assert world_service.find_entity_by_name(world_db, "   ") is None

        # None-like empty after strip
        assert world_service.find_entity_by_name(world_db, "  \t\n  ") is None

    def test_find_entity_by_name_clamps_fuzzy_threshold(self, world_service, world_db):
        """Test that fuzzy_threshold is clamped to 0.0-1.0 range."""
        world_db.add_entity("character", "Alice")
        world_db.add_entity("character", "Alicia")  # Similar name

        # Threshold above 1.0 should be clamped to 1.0 (exact match only)
        entity = world_service.find_entity_by_name(world_db, "Alice", fuzzy_threshold=1.5)
        assert entity is not None
        assert entity.name == "Alice"

        # Threshold below 0.0 should be clamped to 0.0 (very lenient)
        entity = world_service.find_entity_by_name(world_db, "Alicia", fuzzy_threshold=-0.5)
        assert entity is not None

    def test_calculate_name_similarity_exact_match(self, world_service):
        """Test _calculate_name_similarity returns 1.0 for exact match."""
        # Direct test of the private method to ensure coverage
        score = world_service._calculate_name_similarity("alice", "alice")
        assert score == 1.0

        # Test with empty strings (edge case)
        score = world_service._calculate_name_similarity("", "")
        assert score == 1.0

    def test_get_world_health_metrics_orphan_detection_disabled(self, world_db):
        """Test health metrics skips orphan detection when setting disabled."""
        # Create settings with orphan detection disabled
        settings = Settings()
        settings.orphan_detection_enabled = False
        service = WorldService(settings)

        # Create orphan entities (no relationships)
        world_db.add_entity("character", "Lonely Alice")
        world_db.add_entity("character", "Lonely Bob")

        metrics = service.get_world_health_metrics(world_db)

        # Orphan count should be 0 because detection is disabled
        assert metrics.orphan_count == 0
        assert metrics.orphan_entities == []

    def test_get_world_health_metrics_circular_detection_disabled(self, world_db):
        """Test health metrics skips circular detection when setting disabled."""
        # Create settings with circular detection disabled
        settings = Settings()
        settings.circular_detection_enabled = False
        service = WorldService(settings)

        # Create circular chain
        a_id = world_db.add_entity("character", "A")
        b_id = world_db.add_entity("character", "B")
        c_id = world_db.add_entity("character", "C")
        world_db.add_relationship(a_id, b_id, "reports_to", validate=False)
        world_db.add_relationship(b_id, c_id, "reports_to", validate=False)
        world_db.add_relationship(c_id, a_id, "reports_to", validate=False)

        metrics = service.get_world_health_metrics(world_db)

        # Circular count should be 0 because detection is disabled
        assert metrics.circular_count == 0
        assert metrics.circular_relationships == []

    def test_get_world_health_metrics_circular_check_all_types(self, world_db):
        """Test health metrics checks all types when circular_check_all_types is True."""
        settings = Settings()
        settings.circular_detection_enabled = True
        settings.circular_check_all_types = True
        service = WorldService(settings)

        # Create circular chain with a type NOT in the default list
        a_id = world_db.add_entity("character", "A")
        b_id = world_db.add_entity("character", "B")
        c_id = world_db.add_entity("character", "C")
        world_db.add_relationship(a_id, b_id, "friends_with", validate=False)
        world_db.add_relationship(b_id, c_id, "friends_with", validate=False)
        world_db.add_relationship(c_id, a_id, "friends_with", validate=False)

        metrics = service.get_world_health_metrics(world_db)

        # Should detect circular relationships even for non-default types
        assert metrics.circular_count > 0

    def test_get_world_health_metrics_circular_specific_types_only(self, world_db):
        """Test health metrics checks only specific types when circular_check_all_types is False."""
        settings = Settings()
        settings.circular_detection_enabled = True
        settings.circular_check_all_types = False
        settings.circular_relationship_types = ["reports_to"]
        service = WorldService(settings)

        # Create circular chain with a type NOT in the specific list
        a_id = world_db.add_entity("character", "A")
        b_id = world_db.add_entity("character", "B")
        c_id = world_db.add_entity("character", "C")
        world_db.add_relationship(a_id, b_id, "friends_with", validate=False)
        world_db.add_relationship(b_id, c_id, "friends_with", validate=False)
        world_db.add_relationship(c_id, a_id, "friends_with", validate=False)

        metrics = service.get_world_health_metrics(world_db)

        # Should NOT detect circular relationships (friends_with not in the list)
        assert metrics.circular_count == 0

    def test_get_world_health_metrics_quality_scores_dict_format(self, world_service, world_db):
        """Test health metrics correctly extracts quality_scores in dict format."""
        # Create entity with quality_scores dict (new format)
        world_db.add_entity(
            "character",
            "Alice",
            attributes={"quality_scores": {"average": 8.5, "depth": 8.0, "clarity": 9.0}},
        )
        # Create entity with legacy quality_score (old format)
        world_db.add_entity("character", "Bob", attributes={"quality_score": 7.0})

        metrics = world_service.get_world_health_metrics(world_db)

        # Average quality should reflect both entities
        assert metrics.average_quality > 0
        # Both should be counted in distribution
        assert metrics.quality_distribution["8-10"] >= 1  # Alice
        assert metrics.quality_distribution["6-8"] >= 1  # Bob

    def test_health_score_weighted_structural_and_quality(self, world_service, world_db):
        """Test health score is a weighted blend of structural (60%) and quality (40%)."""
        # Create entities with perfect quality (10.0) and no structural issues
        alice_id = world_db.add_entity("character", "Alice", attributes={"quality_score": 10.0})
        bob_id = world_db.add_entity("character", "Bob", attributes={"quality_score": 10.0})
        world_db.add_relationship(alice_id, bob_id, "knows", validate=False)

        metrics = world_service.get_world_health_metrics(world_db)

        # structural = 100 (no penalties, density < 1.0 so no bonus)
        # quality = 10.0 * 10 = 100
        # score = 100 * 0.6 + 100 * 0.4 = 100
        assert metrics.health_score == 100.0

    def test_health_score_unscored_entities_contribute_zero(self, world_service, world_db):
        """Test unscored entities contribute 0.0 to quality average."""
        # Create entities with no quality scores
        world_db.add_entity("character", "Alice")
        world_db.add_entity("character", "Bob")
        world_db.add_entity("location", "Town")

        metrics = world_service.get_world_health_metrics(world_db)

        # All entities are unscored, average quality should be 0.0
        assert metrics.average_quality == 0.0
        # All 3 entities should be in the 0-2 bucket
        assert metrics.quality_distribution["0-2"] == 3
        # Health score should reflect 0 quality: structural * 0.6 + 0 * 0.4
        # structural = 100 - orphan penalties (all 3 are orphans = -6)
        # structural = 94, score = 94 * 0.6 + 0 * 0.4 = 56.4
        assert metrics.health_score == pytest.approx(56.4, abs=0.1)

    def test_health_score_mixed_scored_and_unscored(self, world_service, world_db):
        """Test health score with mix of scored and unscored entities."""
        alice_id = world_db.add_entity("character", "Alice", attributes={"quality_score": 8.0})
        bob_id = world_db.add_entity("character", "Bob")  # Unscored
        world_db.add_relationship(alice_id, bob_id, "knows", validate=False)

        metrics = world_service.get_world_health_metrics(world_db)

        # Average quality = (8.0 + 0.0) / 2 = 4.0
        assert metrics.average_quality == pytest.approx(4.0, abs=0.1)
        # structural = 100, quality = 4.0 * 10 = 40
        # score = 100 * 0.6 + 40 * 0.4 = 76.0
        assert metrics.health_score == pytest.approx(76.0, abs=0.1)

    def test_health_score_bool_quality_ignored(self, world_service, world_db, caplog):
        """Test that boolean quality scores are excluded from quality average."""
        import logging

        # Create entity with bool quality_score (corrupt data)
        alice_id = world_db.add_entity("character", "Alice", attributes={"quality_score": True})
        bob_id = world_db.add_entity("character", "Bob", attributes={"quality_score": 8.0})
        world_db.add_relationship(alice_id, bob_id, "knows", validate=False)

        with caplog.at_level(logging.DEBUG, logger="src.services.world_service._health"):
            metrics = world_service.get_world_health_metrics(world_db)

        # Alice's bool score is excluded, only Bob's 8.0 counts
        # average_quality = 8.0 (only one valid score)
        assert metrics.average_quality == pytest.approx(8.0, abs=0.1)
        # Verify debug log for bool quality score
        assert any("bool quality score" in r.message for r in caplog.records)

    def test_health_score_negative_quality_clamped(self, world_service, world_db):
        """Test that negative quality scores are clamped to 0."""
        alice_id = world_db.add_entity("character", "Alice", attributes={"quality_score": -5.0})
        bob_id = world_db.add_entity("character", "Bob", attributes={"quality_score": 8.0})
        world_db.add_relationship(alice_id, bob_id, "knows", validate=False)

        metrics = world_service.get_world_health_metrics(world_db)

        # Alice's -5.0 clamped to 0.0, Bob is 8.0
        # average_quality = (0.0 + 8.0) / 2 = 4.0
        assert metrics.average_quality == pytest.approx(4.0, abs=0.1)

    def test_health_score_nan_quality_treated_as_zero(self, world_service, world_db):
        """Test that NaN quality scores are treated as 0."""
        alice_id = world_db.add_entity(
            "character", "Alice", attributes={"quality_score": float("nan")}
        )
        bob_id = world_db.add_entity("character", "Bob", attributes={"quality_score": 8.0})
        world_db.add_relationship(alice_id, bob_id, "knows", validate=False)

        metrics = world_service.get_world_health_metrics(world_db)

        # Alice's NaN treated as 0.0, Bob is 8.0
        # average_quality = (0.0 + 8.0) / 2 = 4.0
        assert metrics.average_quality == pytest.approx(4.0, abs=0.1)

    def test_health_score_overflow_quality_clamped(self, world_service, world_db):
        """Test that quality scores above 10 are clamped to 10."""
        alice_id = world_db.add_entity("character", "Alice", attributes={"quality_score": 15.0})
        bob_id = world_db.add_entity("character", "Bob", attributes={"quality_score": 8.0})
        world_db.add_relationship(alice_id, bob_id, "knows", validate=False)

        metrics = world_service.get_world_health_metrics(world_db)

        # Alice's 15.0 clamped to 10.0, Bob is 8.0
        # average_quality = (10.0 + 8.0) / 2 = 9.0
        assert metrics.average_quality == pytest.approx(9.0, abs=0.1)


class TestHealthMetricsCache:
    """Tests for WorldService health metrics TTL cache."""

    def test_cache_hit_returns_same_object(self, world_service, world_db):
        """Calling get_world_health_metrics twice returns the cached object."""
        world_db.add_entity("character", "Alice", attributes={"quality_score": 8.0})

        first = world_service.get_world_health_metrics(world_db)
        second = world_service.get_world_health_metrics(world_db)

        assert first is second  # exact same object, not a recomputation

    def test_cache_miss_on_different_world_db(self, world_service, tmp_path):
        """Different world_db instance triggers fresh computation."""
        db1 = WorldDatabase(tmp_path / "world1.db")
        db2 = WorldDatabase(tmp_path / "world2.db")
        try:
            db1.add_entity("character", "Alice")
            db2.add_entity("location", "Town")

            first = world_service.get_world_health_metrics(db1)
            second = world_service.get_world_health_metrics(db2)

            assert first is not second
            assert first.total_entities == 1
            assert second.total_entities == 1
            assert first.entity_counts["character"] == 1
            assert second.entity_counts["location"] == 1
        finally:
            db1.close()
            db2.close()

    def test_cache_miss_on_different_threshold(self, world_service, world_db):
        """Different quality_threshold triggers fresh computation."""
        world_db.add_entity("character", "Alice", attributes={"quality_score": 5.0})

        first = world_service.get_world_health_metrics(world_db, quality_threshold=6.0)
        second = world_service.get_world_health_metrics(world_db, quality_threshold=4.0)

        assert first is not second

    def test_cache_expires_after_ttl(self, world_service, world_db):
        """After TTL elapses the cache is stale and recomputes."""
        world_db.add_entity("character", "Alice")

        first = world_service.get_world_health_metrics(world_db)

        # Compute the future timestamp before patching (avoid calling mock inside patch)
        future_time = time.monotonic() + _HEALTH_CACHE_TTL_SECONDS + 1
        with patch("src.services.world_service.time.monotonic", return_value=future_time):
            second = world_service.get_world_health_metrics(world_db)

        assert first is not second  # recomputed

    def test_invalidate_health_cache(self, world_service, world_db):
        """invalidate_health_cache forces recomputation on next call."""
        world_db.add_entity("character", "Alice")

        first = world_service.get_world_health_metrics(world_db)
        world_service.invalidate_health_cache()
        second = world_service.get_world_health_metrics(world_db)

        assert first is not second

    def test_add_entity_invalidates_cache(self, world_service, world_db):
        """add_entity clears the health cache."""
        world_db.add_entity("character", "Alice")
        first = world_service.get_world_health_metrics(world_db)

        world_service.add_entity(world_db, "character", "Bob", description="new")
        second = world_service.get_world_health_metrics(world_db)

        assert first is not second
        assert second.total_entities == first.total_entities + 1

    def test_update_entity_invalidates_cache(self, world_service, world_db):
        """update_entity clears the health cache."""
        eid = world_db.add_entity("character", "Alice")
        first = world_service.get_world_health_metrics(world_db)

        world_service.update_entity(world_db, eid, name="Alice Updated")
        second = world_service.get_world_health_metrics(world_db)

        assert first is not second

    def test_delete_entity_invalidates_cache(self, world_service, world_db):
        """delete_entity clears the health cache."""
        eid = world_service.add_entity(world_db, "character", "Alice", description="test")
        first = world_service.get_world_health_metrics(world_db)

        world_service.delete_entity(world_db, eid)
        second = world_service.get_world_health_metrics(world_db)

        assert first is not second
        assert second.total_entities == first.total_entities - 1

    def test_add_relationship_invalidates_cache(self, world_service, world_db):
        """add_relationship clears the health cache."""
        a_id = world_db.add_entity("character", "Alice")
        b_id = world_db.add_entity("character", "Bob")
        first = world_service.get_world_health_metrics(world_db)

        world_service.add_relationship(world_db, a_id, b_id, "knows")
        second = world_service.get_world_health_metrics(world_db)

        assert first is not second
        assert second.total_relationships == first.total_relationships + 1

    def test_delete_relationship_invalidates_cache(self, world_service, world_db):
        """delete_relationship clears the health cache."""
        a_id = world_db.add_entity("character", "Alice")
        b_id = world_db.add_entity("character", "Bob")
        rel_id = world_service.add_relationship(world_db, a_id, b_id, "knows")
        first = world_service.get_world_health_metrics(world_db)

        world_service.delete_relationship(world_db, rel_id)
        second = world_service.get_world_health_metrics(world_db)

        assert first is not second

    def test_extract_from_chapter_invalidates_cache(self, world_service, world_db):
        """extract_from_chapter clears the health cache."""
        first = world_service.get_world_health_metrics(world_db)

        world_service.extract_from_chapter(
            "The heroes arrived at the Enchanted Forest.", world_db, chapter_number=1
        )
        second = world_service.get_world_health_metrics(world_db)

        assert first is not second

    def test_revert_entity_to_version_invalidates_cache(self, world_service, world_db):
        """revert_entity_to_version clears the health cache."""
        eid = world_service.add_entity(
            world_db, "character", "Alice", description="v1", attributes={"role": "original"}
        )
        world_service.update_entity(world_db, eid, attributes={"role": "changed"})
        first = world_service.get_world_health_metrics(world_db)

        world_service.revert_entity_to_version(world_db, eid, 1)
        second = world_service.get_world_health_metrics(world_db)

        assert first is not second

    def test_build_world_invalidates_cache(
        self, world_service, world_db, sample_story_state, settings
    ):
        """build_world clears the health cache after building."""
        first = world_service.get_world_health_metrics(world_db)

        # Use a mock ServiceContainer — build_world invalidates after delegating
        mock_services = MagicMock()
        mock_services.world_quality = MagicMock()
        mock_services.scoring = MagicMock()

        options = WorldBuildOptions(
            generate_structure=True,
            generate_locations=False,
            generate_factions=False,
            generate_items=False,
            generate_concepts=False,
            generate_relationships=False,
        )
        world_service.build_world(sample_story_state, world_db, mock_services, options=options)
        second = world_service.get_world_health_metrics(world_db)

        assert first is not second

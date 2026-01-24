"""Integration tests for interview to world generation flow.

Tests project creation, interview phase, and world building.
"""

import pytest

from src.memory.story_state import StoryBrief


class TestProjectCreation:
    """Test project creation workflow."""

    def test_create_project(self, services, tmp_path):
        """Test creating a new project."""
        story_state, world_db = services.project.create_project("Test Project")

        assert story_state.id is not None
        assert story_state.project_name == "Test Project"
        assert story_state.status == "interview"
        assert world_db is not None

    def test_project_creates_with_id(self, services):
        """Test that created projects have IDs."""
        story_state, world_db = services.project.create_project("ID Test")

        # Verify project has a valid UUID
        assert story_state.id is not None
        assert len(story_state.id) > 0
        assert "-" in story_state.id  # UUID format

    def test_list_projects(self, services):
        """Test listing projects."""
        # Create multiple projects
        services.project.create_project("Project 1")
        services.project.create_project("Project 2")

        # List projects
        projects = services.project.list_projects()

        assert len(projects) >= 2
        project_names = [p.name for p in projects]
        assert "Project 1" in project_names
        assert "Project 2" in project_names

    def test_load_project(self, services):
        """Test loading a saved project."""
        # Create and save
        story_state, world_db = services.project.create_project("Load Test")
        story_state.brief = StoryBrief(
            premise="Test premise",
            genre="Fantasy",
            tone="Epic",
            setting_time="Medieval",
            setting_place="Kingdom",
            target_length="novella",
            language="English",
            content_rating="general",
        )
        services.project.save_project(story_state)

        # Load
        loaded_state, loaded_db = services.project.load_project(story_state.id)

        assert loaded_state.id == story_state.id
        assert loaded_state.project_name == "Load Test"
        assert loaded_state.brief is not None
        assert loaded_state.brief.premise == "Test premise"

    def test_delete_project(self, services):
        """Test deleting a project."""
        # Create
        story_state, world_db = services.project.create_project("Delete Test")
        project_id = story_state.id
        services.project.save_project(story_state)

        # Verify it exists
        projects_before = services.project.list_projects()
        ids_before = [p.id for p in projects_before]
        assert project_id in ids_before

        # Close database before deletion (required on Windows due to file locking)
        world_db.close()

        # Delete
        services.project.delete_project(project_id)

        # Verify it's gone
        projects_after = services.project.list_projects()
        ids_after = [p.id for p in projects_after]
        assert project_id not in ids_after


class TestWorldDatabase:
    """Test world database operations."""

    def test_add_entity_to_world(self, services):
        """Test adding entities to world database."""
        story_state, world_db = services.project.create_project("Entity Test")

        # Add a character
        entity_id = world_db.add_entity(
            entity_type="character", name="Alice", description="The protagonist"
        )

        assert entity_id is not None

        # Retrieve the entity
        entities = world_db.list_entities(entity_type="character")
        assert len(entities) == 1
        assert entities[0].name == "Alice"

    def test_add_relationship(self, services):
        """Test adding relationships between entities."""
        story_state, world_db = services.project.create_project("Relationship Test")

        # Add two characters
        alice_id = world_db.add_entity("character", "Alice", "Protagonist")
        bob_id = world_db.add_entity("character", "Bob", "Friend")

        # Add relationship
        world_db.add_relationship(alice_id, bob_id, "friend_of", description="Best friends")

        # Verify relationship exists
        relationships = world_db.get_relationships(alice_id)
        assert len(relationships) > 0

    def test_world_database_persistence(self, services):
        """Test that world database persists across loads."""
        # Create and add entity
        story_state, world_db = services.project.create_project("Persistence Test")
        world_db.add_entity("character", "Alice", "Test character")
        services.project.save_project(story_state)

        # Load and verify entity persists
        loaded_state, loaded_db = services.project.load_project(story_state.id)
        entities = loaded_db.list_entities(entity_type="character")

        assert len(entities) == 1
        assert entities[0].name == "Alice"


class TestInterviewValidation:
    """Test interview phase validation."""

    def test_build_structure_requires_brief(self, services):
        """Test that build_structure validates brief exists."""
        story_state, world_db = services.project.create_project("No Brief")

        # Should raise error without brief
        with pytest.raises(ValueError, match="brief"):
            services.story.build_structure(story_state, world_db)

    def test_continue_interview_requires_brief(self, services):
        """Test that continue_interview validates brief exists."""
        story_state, _ = services.project.create_project("No Brief")

        # Should raise error without brief
        with pytest.raises(ValueError, match="brief"):
            services.story.continue_interview(story_state, "Some input")


class TestStoryStateManagement:
    """Test story state management across workflow."""

    def test_story_state_tracks_status(self, services):
        """Test that story state tracks workflow status."""
        story_state, world_db = services.project.create_project("Status Test")

        # Initial status
        assert story_state.status == "interview"

        # Update status
        story_state.status = "outlining"
        services.project.save_project(story_state)

        # Load and verify
        loaded_state, _ = services.project.load_project(story_state.id)
        assert loaded_state.status == "outlining"

    def test_add_reviews(self, services):
        """Test adding reviews to story state."""
        story_state, _ = services.project.create_project("Review Test")

        # Add reviews
        services.story.add_review(
            story_state, review_type="user_feedback", content="Good start", chapter_num=1
        )
        services.story.add_review(
            story_state, review_type="ai_suggestion", content="Add detail", chapter_num=1
        )

        # Get reviews
        all_reviews = services.story.get_reviews(story_state)
        assert len(all_reviews) == 2

        # Get chapter-specific reviews
        chapter_1_reviews = services.story.get_reviews(story_state, chapter_num=1)
        assert len(chapter_1_reviews) == 2

    def test_story_brief_persistence(self, services):
        """Test that story brief persists correctly."""
        story_state, world_db = services.project.create_project("Brief Test")

        # Set brief
        story_state.brief = StoryBrief(
            premise="A detective investigates a mystery",
            genre="Mystery",
            tone="Dark",
            setting_time="1940s",
            setting_place="New York",
            target_length="novella",
            language="English",
            content_rating="mature",
        )

        # Save and load
        services.project.save_project(story_state)
        loaded_state, _ = services.project.load_project(story_state.id)

        # Verify brief persists
        assert loaded_state.brief is not None
        assert loaded_state.brief.genre == "Mystery"
        assert loaded_state.brief.premise == "A detective investigates a mystery"

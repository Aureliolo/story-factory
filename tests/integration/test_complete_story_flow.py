"""Integration tests for complete end-to-end story generation flow.

Tests the full workflow combining all phases:
- Project creation
- Interview/brief setting
- World building
- Writing
- Export
"""

from unittest.mock import patch

import pytest

from memory.story_state import Chapter, Character, StoryBrief
from services import ServiceContainer
from settings import Settings


class TestCompleteWorkflow:
    """Test complete end-to-end workflows."""

    @pytest.fixture
    def services(self, tmp_path, mock_ollama_for_agents):
        """Create service container."""
        stories_dir = tmp_path / "stories"
        worlds_dir = tmp_path / "worlds"
        stories_dir.mkdir(parents=True, exist_ok=True)
        worlds_dir.mkdir(parents=True, exist_ok=True)

        with patch("settings.STORIES_DIR", stories_dir), patch("settings.WORLDS_DIR", worlds_dir):
            settings = Settings()
            yield ServiceContainer(settings)

    def test_short_story_complete_workflow(self, services, tmp_path):
        """Test complete workflow for short story."""
        # Step 1: Create project
        story_state, world_db = services.project.create_project("Short Story Complete")
        assert story_state is not None

        # Step 2: Set brief (simulating interview completion)
        story_state.brief = StoryBrief(
            premise="A scientist discovers something unexpected",
            genre="Science Fiction",
            tone="Wonder",
            setting_time="Near Future",
            setting_place="Research Lab",
            target_length="short_story",
            language="English",
            content_rating="general",
        )

        # Step 3: Add world elements (simulating world generation)
        world_db.add_entity("character", "Dr. Sarah", "Protagonist scientist")
        world_db.add_entity("location", "Lab", "High-tech research facility")

        # Step 4: Create story structure
        story_state.characters = [
            Character(
                name="Dr. Sarah",
                role="protagonist",
                description="Brilliant scientist",
                personality_traits=["curious", "determined"],
                goals=["Make a discovery"],
            )
        ]

        story_state.chapters = [
            Chapter(
                number=1,
                title="The Discovery",
                outline="Sarah makes the discovery",
                content="Dr. Sarah stared at the results in disbelief. This changed everything.",
                status="final",
            )
        ]

        story_state.status = "complete"

        # Step 5: Export
        markdown = services.export.to_markdown(story_state)
        assert len(markdown) > 0
        assert "Sarah" in markdown

        # Step 6: Save
        services.project.save_project(story_state)

        # Step 7: Verify persistence
        loaded_state, loaded_db = services.project.load_project(story_state.id)
        assert loaded_state.brief.premise == story_state.brief.premise
        assert len(loaded_state.chapters) == 1

    def test_novella_complete_workflow(self, services, tmp_path):
        """Test complete workflow for novella."""
        # Create project
        story_state, world_db = services.project.create_project("Novella Complete")

        # Set brief for novella
        story_state.brief = StoryBrief(
            premise="A detective solves a complex mystery",
            genre="Mystery",
            tone="Suspenseful",
            setting_time="1940s",
            setting_place="New York",
            target_length="novella",
            language="English",
            content_rating="mature",
        )

        # Add multiple characters
        story_state.characters = [
            Character(
                name="Detective Mike",
                role="protagonist",
                description="Hard-boiled detective",
                personality_traits=["cynical", "determined"],
                goals=["Solve the case"],
            ),
            Character(
                name="Jane Doe",
                role="antagonist",
                description="Mysterious villain",
                personality_traits=["cunning", "ruthless"],
                goals=["Escape justice"],
            ),
        ]

        # Add multiple chapters
        story_state.chapters = [
            Chapter(
                number=1,
                title="The Case",
                outline="Introduction",
                content="Chapter 1 content",
                status="final",
            ),
            Chapter(
                number=2,
                title="Investigation",
                outline="Clues",
                content="Chapter 2 content",
                status="final",
            ),
            Chapter(
                number=3,
                title="Resolution",
                outline="Conclusion",
                content="Chapter 3 content",
                status="final",
            ),
        ]

        story_state.status = "complete"

        # Export and verify
        markdown = services.export.to_markdown(story_state)
        assert "Chapter 1" in markdown
        assert "Chapter 2" in markdown
        assert "Chapter 3" in markdown

        # Save and reload
        services.project.save_project(story_state)
        loaded_state, _ = services.project.load_project(story_state.id)
        assert len(loaded_state.chapters) == 3
        assert len(loaded_state.characters) == 2


class TestWorkflowPersistence:
    """Test state persistence across workflow phases."""

    @pytest.fixture
    def services(self, tmp_path, mock_ollama_for_agents):
        """Create service container."""
        stories_dir = tmp_path / "stories"
        worlds_dir = tmp_path / "worlds"
        stories_dir.mkdir(parents=True, exist_ok=True)
        worlds_dir.mkdir(parents=True, exist_ok=True)

        with patch("settings.STORIES_DIR", stories_dir), patch("settings.WORLDS_DIR", worlds_dir):
            settings = Settings()
            yield ServiceContainer(settings)

    def test_persist_after_each_phase(self, services):
        """Test saving and loading after each workflow phase."""
        # Phase 1: Create project
        story_state, world_db = services.project.create_project("Persistence Test")
        project_id = story_state.id
        services.project.save_project(story_state)

        # Reload and verify
        story_state, world_db = services.project.load_project(project_id)
        assert story_state.status == "interview"

        # Phase 2: Add brief
        story_state.brief = StoryBrief(
            premise="Test story",
            genre="Fiction",
            tone="Neutral",
            setting_time="Now",
            setting_place="Here",
            target_length="short_story",
            language="English",
            content_rating="general",
        )
        story_state.status = "outlining"
        services.project.save_project(story_state)

        # Reload and verify
        story_state, world_db = services.project.load_project(project_id)
        assert story_state.brief is not None
        assert story_state.status == "outlining"

        # Phase 3: Add structure
        story_state.chapters = [Chapter(number=1, title="Test", outline="Test outline")]
        story_state.status = "writing"
        services.project.save_project(story_state)

        # Reload and verify
        story_state, world_db = services.project.load_project(project_id)
        assert len(story_state.chapters) == 1
        assert story_state.status == "writing"

        # Phase 4: Add content
        story_state.chapters[0].content = "Test content"
        story_state.status = "complete"
        services.project.save_project(story_state)

        # Final reload and verify
        story_state, world_db = services.project.load_project(project_id)
        assert story_state.chapters[0].content == "Test content"
        assert story_state.status == "complete"

    def test_world_database_syncs_with_story(self, services):
        """Test that world database stays in sync with story state."""
        # Create project
        story_state, world_db = services.project.create_project("Sync Test")

        # Add character to story state
        story_state.characters = [
            Character(
                name="Alice",
                role="protagonist",
                description="Main character",
                personality_traits=["brave"],
                goals=["Win"],
            )
        ]

        # Manually add to world database (simulating extraction)
        world_db.add_entity("character", "Alice", "Main character")

        # Save both
        services.project.save_project(story_state)

        # Reload and verify both have the character
        loaded_state, loaded_db = services.project.load_project(story_state.id)
        assert len(loaded_state.characters) == 1
        assert loaded_state.characters[0].name == "Alice"

        entities = loaded_db.list_entities(entity_type="character")
        assert len(entities) == 1
        assert entities[0].name == "Alice"


class TestProjectLifecycle:
    """Test complete project lifecycle operations."""

    @pytest.fixture
    def services(self, tmp_path, mock_ollama_for_agents):
        """Create service container."""
        stories_dir = tmp_path / "stories"
        worlds_dir = tmp_path / "worlds"
        stories_dir.mkdir(parents=True, exist_ok=True)
        worlds_dir.mkdir(parents=True, exist_ok=True)

        with patch("settings.STORIES_DIR", stories_dir), patch("settings.WORLDS_DIR", worlds_dir):
            settings = Settings()
            yield ServiceContainer(settings)

    def test_full_project_lifecycle(self, services):
        """Test create, modify, save, load, delete lifecycle."""
        # Create
        story_state, world_db = services.project.create_project("Lifecycle Test")
        project_id = story_state.id

        # Modify
        story_state.brief = StoryBrief(
            premise="Test",
            genre="Fiction",
            tone="Neutral",
            setting_time="Now",
            setting_place="Here",
            target_length="short_story",
            language="English",
            content_rating="general",
        )

        # Save
        services.project.save_project(story_state)

        # List and verify it exists
        projects = services.project.list_projects()
        project_ids = [p.id for p in projects]
        assert project_id in project_ids

        # Load
        loaded_state, loaded_db = services.project.load_project(project_id)
        assert loaded_state.brief is not None

        # Delete
        services.project.delete_project(project_id)

        # Verify deleted
        projects_after = services.project.list_projects()
        project_ids_after = [p.id for p in projects_after]
        assert project_id not in project_ids_after

    def test_multiple_projects_isolation(self, services):
        """Test that multiple projects are isolated from each other."""
        # Create two projects
        story1, world1 = services.project.create_project("Project 1")
        story2, world2 = services.project.create_project("Project 2")

        # Add different data to each
        story1.brief = StoryBrief(
            premise="Story 1",
            genre="Mystery",
            tone="Dark",
            setting_time="Past",
            setting_place="London",
            target_length="short_story",
            language="English",
            content_rating="general",
        )

        story2.brief = StoryBrief(
            premise="Story 2",
            genre="Fantasy",
            tone="Epic",
            setting_time="Medieval",
            setting_place="Kingdom",
            target_length="novella",
            language="English",
            content_rating="general",
        )

        # Save both
        services.project.save_project(story1)
        services.project.save_project(story2)

        # Load and verify they're different
        loaded1, _ = services.project.load_project(story1.id)
        loaded2, _ = services.project.load_project(story2.id)

        assert loaded1.brief.premise == "Story 1"
        assert loaded2.brief.premise == "Story 2"
        assert loaded1.brief.genre == "Mystery"
        assert loaded2.brief.genre == "Fantasy"


class TestErrorRecovery:
    """Test error recovery in complete workflows."""

    @pytest.fixture
    def services(self, tmp_path, mock_ollama_for_agents):
        """Create service container."""
        stories_dir = tmp_path / "stories"
        worlds_dir = tmp_path / "worlds"
        stories_dir.mkdir(parents=True, exist_ok=True)
        worlds_dir.mkdir(parents=True, exist_ok=True)

        with patch("settings.STORIES_DIR", stories_dir), patch("settings.WORLDS_DIR", worlds_dir):
            settings = Settings()
            yield ServiceContainer(settings)

    def test_load_nonexistent_project(self, services):
        """Test loading a project that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            services.project.load_project("nonexistent-id")

    def test_delete_nonexistent_project(self, services):
        """Test deleting a project that doesn't exist."""
        # Should not raise an error, just silently succeed or log
        services.project.delete_project("nonexistent-id")

    def test_export_with_missing_data(self, services):
        """Test exporting story with minimal data."""
        story_state, _ = services.project.create_project("Minimal Export")

        # Export without brief or chapters
        markdown = services.export.to_markdown(story_state)

        # Should still produce something
        assert markdown is not None
        assert len(markdown) > 0

    def test_validation_prevents_invalid_operations(self, services):
        """Test that validation prevents invalid workflow operations."""
        story_state, world_db = services.project.create_project("Validation Test")

        # Try to build structure without brief
        with pytest.raises(ValueError, match="brief"):
            services.story.build_structure(story_state, world_db)

        # Try to continue interview without brief
        with pytest.raises(ValueError, match="brief"):
            services.story.continue_interview(story_state, "Some input")

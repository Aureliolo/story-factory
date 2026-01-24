"""Pytest configuration for NiceGUI component tests.

These tests use NiceGUI's User fixture for fast, lightweight testing
of UI components without requiring a browser.

Note: The pytest_plugins for NiceGUI is registered in the root conftest.py.
Note: Ollama mocking is handled by the autouse fixture in the root conftest.py
      which uses the shared mock from tests/shared/mock_ollama.py.
"""

from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def test_settings():
    """Create test settings for component tests.

    Uses the default Ollama URL which is accepted by the shared mock.
    """
    from settings import Settings

    return Settings()


@pytest.fixture
def test_services(test_settings, tmp_path):
    """Create test service container with mocked services and isolated directories.

    IMPORTANT: Must patch STORIES_DIR/WORLDS_DIR to prevent tests from
    writing to the real output directory.
    """
    from services import ServiceContainer

    stories_dir = tmp_path / "stories"
    worlds_dir = tmp_path / "worlds"
    stories_dir.mkdir(parents=True, exist_ok=True)
    worlds_dir.mkdir(parents=True, exist_ok=True)

    # Patch at ALL locations where these constants are imported
    with (
        patch("settings.STORIES_DIR", stories_dir),
        patch("settings.WORLDS_DIR", worlds_dir),
        patch("services.project_service.STORIES_DIR", stories_dir),
        patch("services.project_service.WORLDS_DIR", worlds_dir),
        patch("services.backup_service.STORIES_DIR", stories_dir),
        patch("services.backup_service.WORLDS_DIR", worlds_dir),
        patch("services.export_service.STORIES_DIR", stories_dir),
    ):
        yield ServiceContainer(test_settings)


@pytest.fixture
def test_world_db(tmp_path: Path):
    """Create a test WorldDatabase with sample data."""
    from memory.world_database import WorldDatabase

    db = WorldDatabase(tmp_path / "test_world.db")

    # Add sample entities
    char_id = db.add_entity(
        entity_type="character",
        name="Test Character",
        description="A test character for component tests",
        attributes={"role": "protagonist"},
    )

    loc_id = db.add_entity(
        entity_type="location",
        name="Test Location",
        description="A test location",
    )

    db.add_relationship(char_id, loc_id, "located_in")

    yield db
    db.close()


@pytest.fixture
def test_story_state():
    """Create a test StoryState."""
    from memory.story_state import StoryBrief, StoryState

    brief = StoryBrief(
        premise="A test story premise",
        genre="Fantasy",
        subgenres=["Adventure"],
        tone="Epic",
        themes=["Testing"],
        setting_time="Present",
        setting_place="Test Land",
        target_length="short_story",
        language="English",
        content_rating="none",
        content_preferences=[],
        content_avoid=[],
    )

    return StoryState(
        id="test-story-001",
        project_name="Test Story",
        brief=brief,
        status="interview",
    )


@pytest.fixture
def test_app_state(test_story_state, test_world_db):
    """Create a test AppState with project loaded."""
    from ui.state import AppState

    state = AppState()
    state.set_project(test_story_state.id, test_story_state, test_world_db)
    return state

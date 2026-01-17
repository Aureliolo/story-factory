"""Pytest configuration for NiceGUI component tests.

These tests use NiceGUI's User fixture for fast, lightweight testing
of UI components without requiring a browser.

Note: The pytest_plugins for NiceGUI is registered in the root conftest.py.
"""

from pathlib import Path
from unittest.mock import patch

import pytest


class MockOllamaClient:
    """Mock Ollama client that returns realistic responses."""

    def __init__(self, host=None, timeout=None):
        self.host = host
        self.timeout = timeout

    def list(self):
        """Return a list of mock models."""

        class Models:
            models = [
                type("Model", (), {"model": "huihui_ai/qwen3-abliterated:8b"})(),
                type("Model", (), {"model": "llama3.2:8b"})(),
            ]

        return Models()

    def generate(self, model, prompt, options=None):
        """Return a mock LLM response."""

        class Response:
            # Return a realistic interview response
            response = """Thank you for sharing your story idea! I have a few questions:

1. What genre are you most interested in? (Fantasy, Sci-Fi, Mystery, etc.)
2. What tone should the story have? (Dark, Light-hearted, Epic, etc.)
3. Any specific themes you want to explore?

Please share more details about your vision!"""

        return Response()

    def chat(self, model, messages, options=None):
        """Return a mock chat response as dict (matches Ollama API)."""
        return {
            "message": {
                "content": """Thank you for sharing your story idea! I have a few questions:

1. What genre are you most interested in? (Fantasy, Sci-Fi, Mystery, etc.)
2. What tone should the story have? (Dark, Light-hearted, Epic, etc.)
3. Any specific themes you want to explore?

Please share more details about your vision!"""
            }
        }

    def pull(self, model, stream=False):
        """Mock model pulling."""
        if stream:
            yield {"status": "pulling", "completed": 50, "total": 100}
            yield {"status": "success", "completed": 100, "total": 100}
        return {"status": "success"}

    def delete(self, model):
        """Mock model deletion."""
        return {"status": "success"}


@pytest.fixture(autouse=True)
def mock_ollama():
    """Automatically mock Ollama for all component tests."""
    with patch("ollama.Client", MockOllamaClient):
        yield MockOllamaClient


@pytest.fixture
def test_settings():
    """Create test settings with mocked Ollama URL."""
    from settings import Settings

    settings = Settings()
    settings.ollama_url = "http://mock:11434"
    return settings


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

    return db


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


@pytest.fixture
def mock_ollama_client(monkeypatch):
    """Mock Ollama client to prevent actual LLM calls."""

    class MockClient:
        def __init__(self, host=None, timeout=None):
            self.host = host
            self.timeout = timeout

        def list(self):
            class Models:
                models = []

            return Models()

        def generate(self, model, prompt, options=None):
            class Response:
                response = '{"test": "mock response"}'

            return Response()

    monkeypatch.setattr("ollama.Client", MockClient)
    return MockClient

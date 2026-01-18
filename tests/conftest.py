"""Pytest fixtures for Story Factory tests."""

from pathlib import Path

import pytest

from memory.story_state import StoryBrief, StoryState
from memory.world_database import WorldDatabase
from settings import Settings

# Enable NiceGUI testing plugin for component tests
pytest_plugins = ["nicegui.testing.user_plugin"]


@pytest.fixture(autouse=True)
def clear_settings_cache_per_test():
    """Clear Settings cache before each test to ensure isolation.

    This is autouse because caching can cause test pollution when tests
    modify settings or patch SETTINGS_FILE to different paths.
    """
    Settings.clear_cache()
    yield
    Settings.clear_cache()


@pytest.fixture(scope="session")
def cached_settings() -> Settings:
    """Create settings once per test session for performance.

    Using session scope avoids repeated Settings.load() calls which take ~0.3s each.
    Tests that need fresh settings should create them directly.

    Returns:
        Cached settings instance.
    """
    return Settings()


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    """Create settings with temporary directories.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Settings configured for testing.
    """
    settings = Settings()
    # Override paths to use temp directories
    return settings


@pytest.fixture
def orchestrator_temp_dir(tmp_path: Path, monkeypatch):
    """Set up temp directory for orchestrator tests.

    This replaces the duplicate use_temp_dir fixtures across test classes.

    Args:
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        Path to the stories directory.
    """
    stories_dir = tmp_path / "stories"
    stories_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("workflows.orchestrator.STORIES_DIR", stories_dir)
    return stories_dir


@pytest.fixture
def fast_orchestrator(cached_settings: Settings, orchestrator_temp_dir: Path):
    """Create a StoryOrchestrator with cached settings for fast tests.

    Uses session-cached settings to avoid repeated Settings.load() calls.

    Args:
        cached_settings: Session-scoped settings fixture.
        orchestrator_temp_dir: Temp directory for story files.

    Returns:
        StoryOrchestrator configured for testing.
    """
    from workflows.orchestrator import StoryOrchestrator

    return StoryOrchestrator(settings=cached_settings)


@pytest.fixture
def sample_world_db(tmp_path: Path) -> WorldDatabase:
    """Create a sample WorldDatabase with test data.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        WorldDatabase with sample entities and relationships.
    """
    db = WorldDatabase(tmp_path / "test_world.db")

    # Add characters
    alice_id = db.add_entity(
        entity_type="character",
        name="Alice",
        description="The protagonist, a curious young woman",
        attributes={"role": "protagonist", "age": 25},
    )

    bob_id = db.add_entity(
        entity_type="character",
        name="Bob",
        description="Alice's loyal friend",
        attributes={"role": "supporting", "age": 27},
    )

    villain_id = db.add_entity(
        entity_type="character",
        name="The Dark Lord",
        description="The main antagonist",
        attributes={"role": "antagonist"},
    )

    # Add locations
    db.add_entity(
        entity_type="location",
        name="Enchanted Forest",
        description="A mysterious forest full of magic",
    )

    castle_id = db.add_entity(
        entity_type="location",
        name="Dark Castle",
        description="The villain's lair",
    )

    # Add relationships
    db.add_relationship(alice_id, bob_id, "knows", description="Best friends")
    db.add_relationship(alice_id, villain_id, "enemy_of")
    db.add_relationship(villain_id, castle_id, "located_in")

    return db


@pytest.fixture
def sample_story_state() -> StoryState:
    """Create a sample StoryState for testing.

    Returns:
        StoryState with sample data.
    """
    brief = StoryBrief(
        premise="A young woman discovers she has magical powers",
        genre="Fantasy",
        subgenres=["Adventure", "Coming of Age"],
        tone="Epic",
        themes=["Self-discovery", "Good vs Evil"],
        setting_time="Medieval",
        setting_place="Kingdom of Eldoria",
        target_length="novella",
        language="English",
        content_rating="none",
        content_preferences=["Magic", "Adventure"],
        content_avoid=["Gore"],
    )

    state = StoryState(
        id="test-story-001",
        project_name="Test Story",
        brief=brief,
        status="interview",
    )

    return state


@pytest.fixture
def sample_story_with_chapters(sample_story_state: StoryState) -> StoryState:
    """Create a StoryState with chapters.

    Args:
        sample_story_state: Base story state.

    Returns:
        StoryState with chapters added.
    """
    from memory.story_state import Chapter, Character

    state = sample_story_state

    # Add characters
    state.characters = [
        Character(
            name="Alice",
            role="protagonist",
            description="A curious young woman",
            personality_traits=["brave", "curious"],
            goals=["Discover her powers", "Save the kingdom"],
        ),
        Character(
            name="Bob",
            role="supporting",
            description="Alice's loyal friend",
            personality_traits=["loyal", "wise"],
            goals=["Help Alice"],
        ),
    ]

    # Add chapters
    state.chapters = [
        Chapter(
            number=1,
            title="The Discovery",
            outline="Alice discovers her magical abilities",
            content="",
            status="pending",
        ),
        Chapter(
            number=2,
            title="The Journey Begins",
            outline="Alice and Bob set out on their adventure",
            content="",
            status="pending",
        ),
        Chapter(
            number=3,
            title="The Dark Castle",
            outline="They reach the villain's lair",
            content="",
            status="pending",
        ),
    ]

    state.status = "writing"

    return state


@pytest.fixture
def mock_ollama(monkeypatch):
    """Mock Ollama client for testing without a real server.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Yields:
        Mock response generator.
    """

    class MockOllamaClient:
        def __init__(self, host=None, timeout=None):
            self.host = host
            self.timeout = timeout

        def list(self):
            class Models:
                models = [
                    type("Model", (), {"model": "test-model:latest"})(),
                ]

            return Models()

        def generate(self, model, prompt, options=None):
            class Response:
                response = "Mock response from AI"

            return Response()

        def pull(self, model, stream=False):
            if stream:
                yield {"status": "pulling", "completed": 50, "total": 100}
                yield {"status": "success", "completed": 100, "total": 100}
            return {"status": "success"}

        def delete(self, model):
            return {"status": "success"}

    monkeypatch.setattr("ollama.Client", MockOllamaClient)

    return MockOllamaClient


@pytest.fixture
def clean_output_dirs(tmp_path: Path):
    """Ensure clean output directories for testing.

    Args:
        tmp_path: Pytest temporary path fixture.

    Yields:
        Tuple of (stories_dir, worlds_dir).
    """
    stories_dir = tmp_path / "stories"
    worlds_dir = tmp_path / "worlds"

    stories_dir.mkdir(parents=True, exist_ok=True)
    worlds_dir.mkdir(parents=True, exist_ok=True)

    yield stories_dir, worlds_dir

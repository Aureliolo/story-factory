"""Pytest fixtures for Story Factory tests."""

import logging
from collections.abc import Generator
from pathlib import Path

import pytest

from memory.story_state import StoryBrief, StoryState
from memory.world_database import WorldDatabase
from settings import Settings

# Enable NiceGUI testing plugin for component tests
pytest_plugins = ["nicegui.testing.user_plugin"]


@pytest.fixture(autouse=True, scope="function")
def cleanup_production_log_handlers():
    """Remove file handlers pointing to the production log after each test.

    This fixture runs after each test and removes any file handlers
    that point to the production log file (logs/story_factory.log).
    This ensures tests don't accidentally leave handlers that write to
    the production log, while still allowing logging tests to work.
    """
    yield

    # After the test, clean up any handlers pointing to production log
    root_logger = logging.getLogger()
    production_log_name = "story_factory.log"

    handlers_to_remove = []
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            # Check if this handler points to the production log
            if hasattr(handler, "baseFilename") and production_log_name in handler.baseFilename:
                handlers_to_remove.append(handler)

    for handler in handlers_to_remove:
        handler.close()
        root_logger.removeHandler(handler)


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
def sample_world_db(tmp_path: Path) -> Generator[WorldDatabase]:
    """Create a sample WorldDatabase with test data.

    Args:
        tmp_path: Pytest temporary path fixture.

    Yields:
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

    yield db
    db.close()


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


@pytest.fixture(autouse=True)
def mock_ollama_globally(monkeypatch):
    """Mock Ollama client and system calls to prevent real connections.

    AUTOUSE: All tests automatically mock Ollama to prevent:
    - Real ollama.Client connections
    - `ollama list` subprocess calls
    - `nvidia-smi` subprocess calls

    Individual tests can override with their own patches using `with patch()`.
    The patch context manager takes precedence over monkeypatch.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import subprocess
    from unittest.mock import MagicMock

    # Create a mock client class that returns safe defaults
    class MockOllamaClient:
        def __init__(self, host=None, timeout=None):
            self.host = host
            self.timeout = timeout

        def list(self):
            mock_model = MagicMock()
            mock_model.model = "test-model:latest"
            mock_response = MagicMock()
            mock_response.models = [mock_model]
            return mock_response

        def generate(self, model=None, prompt=None, options=None, **kwargs):
            return {"response": "Mock response from AI"}

        def chat(self, model=None, messages=None, options=None, **kwargs):
            return {
                "message": {"content": "Mock AI response", "role": "assistant"},
                "done": True,
            }

        def pull(self, model, stream=False):
            if stream:

                def gen():
                    yield {"status": "pulling", "completed": 50, "total": 100}
                    yield {"status": "success", "completed": 100, "total": 100}

                return gen()
            return {"status": "success"}

        def delete(self, model):
            return {"status": "success"}

    # Mock ollama.Client at the ollama module level
    monkeypatch.setattr("ollama.Client", MockOllamaClient)

    # Mock subprocess.run to intercept `ollama list` and `nvidia-smi` commands
    original_subprocess_run = subprocess.run

    def mock_subprocess_run(cmd, *args, **kwargs):
        """Intercept subprocess calls to ollama and nvidia-smi."""
        cmd_str = cmd[0] if isinstance(cmd, list) else cmd

        if "ollama" in cmd_str:
            # Mock `ollama list` output
            class MockOllamaResult:
                stdout = "NAME                    ID      SIZE    MODIFIED\ntest-model:latest    abc123  4.1 GB  2 days ago\n"
                stderr = ""
                returncode = 0

            return MockOllamaResult()

        if "nvidia-smi" in cmd_str:
            # Mock nvidia-smi for VRAM detection
            class MockNvidiaSmiResult:
                stdout = "8192"  # 8GB VRAM
                stderr = ""
                returncode = 0

            return MockNvidiaSmiResult()

        # Pass through other subprocess calls
        return original_subprocess_run(cmd, *args, **kwargs)

    monkeypatch.setattr("subprocess.run", mock_subprocess_run)


@pytest.fixture
def mock_ollama():
    """Deprecated: Use mock_ollama_globally (autouse) instead.

    This fixture is kept for backwards compatibility with tests that
    explicitly request it. The mock_ollama_globally fixture now handles
    all Ollama mocking automatically.

    Returns:
        The mocked ollama.Client class (for inspection if needed).
    """
    import ollama

    return ollama.Client  # Returns the already-mocked Client


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

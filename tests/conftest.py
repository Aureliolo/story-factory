"""Pytest fixtures for Story Factory tests."""

import logging
from collections.abc import Generator
from pathlib import Path

import pytest

from src.memory.story_state import StoryBrief, StoryState
from src.memory.world_database import WorldDatabase
from src.settings import Settings

# Enable NiceGUI testing plugin for component tests
pytest_plugins = ["nicegui.testing.user_plugin"]


@pytest.fixture(autouse=True, scope="function")
def cleanup_production_log_handlers():
    """Remove file handlers pointing to the production log after each test.

    This fixture runs after each test and removes any file handlers
    that point to the production log file (output/logs/story_factory.log).
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


@pytest.fixture(autouse=True)
def clear_prompt_registry_cache_per_test():
    """Clear prompt registry cache before each test to ensure isolation.

    The prompt registry is a module-level singleton that caches templates.
    This can cause test pollution if tests modify settings.prompt_templates_dir.
    """
    import src.agents.base as base_module

    base_module._prompt_registry = None
    yield
    base_module._prompt_registry = None


@pytest.fixture(autouse=True)
def isolate_mode_database(tmp_path, monkeypatch):
    """Redirect ModeDatabase to temp directory to prevent test data in production DB.

    The ModeDatabase has a DEFAULT_DB_PATH that points to output/model_scores.db.
    Without this fixture, any test that creates a ModelModeService or ServiceContainer
    without explicitly providing a db_path would write to the real analytics database.

    This autouse fixture ensures ALL tests use an isolated temp database by default.
    """
    import src.memory.mode_database as mode_database_module

    test_db_path = tmp_path / "test_model_scores.db"
    monkeypatch.setattr(mode_database_module, "DEFAULT_DB_PATH", test_db_path)
    yield


@pytest.fixture(autouse=True)
def isolate_project_directories(tmp_path, monkeypatch):
    """Redirect STORIES_DIR and WORLDS_DIR to temp directory.

    The ProjectService creates directories at STORIES_DIR and WORLDS_DIR during __init__.
    Without this fixture, tests that create ServiceContainer without explicit patches
    would create directories in the real output/ folder.

    This autouse fixture patches at ALL import locations to ensure complete isolation.
    Note: Directories are NOT created here - ProjectService._ensure_directories() creates
    them with exist_ok=True, and tests that need dirs before service init create their own.
    """
    import src.services.backup_service as backup_service_module
    import src.services.orchestrator._persistence as orchestrator_persistence_module
    import src.services.project_service as project_service_module
    import src.services.world_template_service as world_template_service_module
    import src.settings as settings_module

    # Also need to patch export service types if it exists
    try:
        import src.services.export_service._types as export_types_module

        has_export_types = True
    except ImportError:
        has_export_types = False

    stories_dir = tmp_path / "stories"
    worlds_dir = tmp_path / "worlds"
    # Don't create directories here - let tests or ProjectService create them
    # This avoids conflicts with tests that create their own temp directories

    # Patch at all import locations
    monkeypatch.setattr(settings_module, "STORIES_DIR", stories_dir)
    monkeypatch.setattr(settings_module, "WORLDS_DIR", worlds_dir)
    monkeypatch.setattr(project_service_module, "STORIES_DIR", stories_dir)
    monkeypatch.setattr(project_service_module, "WORLDS_DIR", worlds_dir)
    monkeypatch.setattr(backup_service_module, "STORIES_DIR", stories_dir)
    monkeypatch.setattr(backup_service_module, "WORLDS_DIR", worlds_dir)
    monkeypatch.setattr(orchestrator_persistence_module, "STORIES_DIR", stories_dir)
    monkeypatch.setattr(world_template_service_module, "STORIES_DIR", stories_dir)

    if has_export_types:
        monkeypatch.setattr(export_types_module, "STORIES_DIR", stories_dir)

    yield


@pytest.fixture(scope="session")
def cached_settings() -> Settings:
    """Create default settings once per test session for performance.

    Using session scope avoids repeated Settings.load() calls which take ~0.3s each.
    Tests that need fresh settings should create them directly.

    IMPORTANT: This fixture creates a default Settings() instance without loading
    from the settings.json file to ensure test isolation.

    Returns:
        Cached default settings instance.
    """
    # Create default settings WITHOUT loading from file
    # This ensures tests don't depend on local settings.json content
    return Settings()


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    """Create default settings with temporary directories.

    IMPORTANT: This fixture creates a default Settings() instance without loading
    from the settings.json file to ensure test isolation.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Default settings configured for testing.
    """
    # Create default settings WITHOUT loading from file
    # This ensures tests don't depend on local settings.json content
    settings = Settings()
    # Run validation to apply migrations (e.g. populate world_quality_thresholds)
    settings.validate()
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
    monkeypatch.setattr("src.services.orchestrator.STORIES_DIR", stories_dir)
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
    from src.services.orchestrator import StoryOrchestrator

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
    """
    Create a StoryState populated with sample characters and chapters.

    Parameters:
        sample_story_state (StoryState): Base story state to augment.

    Returns:
        StoryState: The same StoryState instance with two sample characters, three sample chapters, and `status` set to "writing".
    """
    from src.memory.story_state import Chapter, Character

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
    """
    Autouse pytest fixture that installs global mocks for the Ollama client and related system calls for all tests.

    Prevents real ollama.Client connections and subprocess calls for `ollama list` and `nvidia-smi`. Tests may override these mocks using `patch()`; patch context managers take precedence over monkeypatch. Uses the shared test utilities to provide a consistent mock implementation across the test suite.
    """
    from tests.shared.mock_ollama import setup_ollama_mocks

    setup_ollama_mocks(monkeypatch)


@pytest.fixture
def mock_ollama():
    """Deprecated: Use mock_ollama_globally (autouse) instead.

    This fixture is kept for backwards compatibility with tests that
    explicitly request it. The mock_ollama_globally fixture now handles
    all Ollama mocking automatically.

    Returns:
        The mocked ollama.Client class (for inspection if needed).
    """
    import warnings

    import ollama

    warnings.warn(
        "mock_ollama fixture is deprecated; mock_ollama_globally (autouse) "
        "now handles all Ollama mocking automatically",
        DeprecationWarning,
        stacklevel=2,
    )

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

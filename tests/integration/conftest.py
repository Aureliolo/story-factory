"""Shared fixtures and mocks for integration tests."""

from unittest.mock import MagicMock, patch

import pytest

from src.services import ServiceContainer
from src.settings import Settings


class MockModel:
    """Mock model object returned by ollama.list().

    The real ollama library returns objects with a .model attribute,
    not dictionaries with a 'name' key.
    """

    def __init__(self, model: str):
        """Initialize mock model with model name."""
        self.model = model


class MockListResponse:
    """Mock response from ollama.Client().list().

    The real ollama library returns an object with a .models attribute
    containing a list of model objects.
    """

    def __init__(self, models: list[str]):
        """Initialize mock list response with model names."""
        self.models = [MockModel(m) for m in models]


class MockChatResponse:
    """Mock response from ollama.Client().chat().

    Deprecated for streaming: Use MockStreamChunk from tests.shared.mock_ollama instead.
    Kept for backward compatibility with non-streaming integration tests.
    """

    def __init__(self, content: str):
        """Initialize mock chat response with content."""
        self.message = {"content": content}

    def __getitem__(self, key):
        """Support dictionary-style access."""
        if key == "message":
            return self.message
        raise KeyError(key)


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client for integration tests.

    Provides a mock that returns a list of models matching the format
    expected by services/model_service.py (objects with .models and .model attributes).
    """
    with patch("src.services.model_service.ollama") as mock_ollama:
        mock_client = MagicMock()
        mock_client.list.return_value = MockListResponse(["model-a:latest", "model-b:7b"])
        mock_ollama.Client.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_ollama_for_agents():
    """Mock Ollama for agent operations.

    Provides comprehensive mocking for all agent interactions.
    Returns a mock client configured to work with agents.
    """
    with patch("src.agents.base.ollama") as mock_ollama:
        mock_client = MagicMock()

        # Default chat response — returns stream-compatible iterator
        # since base.py uses stream=True + consume_stream()
        def default_chat(*args, **kwargs):
            """Return a default mock streaming chat response for agent tests."""
            from tests.shared.mock_ollama import MockStreamChunk

            return iter(
                [
                    MockStreamChunk(
                        content="Default AI response",
                        done=True,
                        prompt_eval_count=10,
                        eval_count=5,
                    ),
                ]
            )

        mock_client.chat.side_effect = default_chat
        mock_ollama.Client.return_value = mock_client

        yield mock_client


@pytest.fixture
def services(tmp_path, mock_ollama_for_agents):
    """Create service container with patched directories.

    This fixture is shared across all integration test classes to avoid
    code duplication and ensure consistent service setup.

    IMPORTANT: Must patch STORIES_DIR/WORLDS_DIR at ALL locations where they're
    imported, not just in settings module. Python's `from X import Y` binds
    the name at import time, so patching settings.Y won't affect already-imported
    references.
    """
    stories_dir = tmp_path / "stories"
    worlds_dir = tmp_path / "worlds"
    stories_dir.mkdir(parents=True, exist_ok=True)
    worlds_dir.mkdir(parents=True, exist_ok=True)

    # Patch at ALL locations where these constants are imported
    # Note: ModeDatabase is automatically isolated by conftest.py's isolate_mode_database fixture
    with (
        patch("src.settings.STORIES_DIR", stories_dir),
        patch("src.settings.WORLDS_DIR", worlds_dir),
        patch("src.services.project_service.STORIES_DIR", stories_dir),
        patch("src.services.project_service.WORLDS_DIR", worlds_dir),
        patch("src.services.backup_service.STORIES_DIR", stories_dir),
        patch("src.services.backup_service.WORLDS_DIR", worlds_dir),
        patch("src.services.export_service._types.STORIES_DIR", stories_dir),
    ):
        settings = Settings()
        yield ServiceContainer(settings)

"""Shared fixtures and mocks for integration tests."""

from unittest.mock import MagicMock, patch

import pytest


class MockModel:
    """Mock model object returned by ollama.list().

    The real ollama library returns objects with a .model attribute,
    not dictionaries with a 'name' key.
    """

    def __init__(self, model: str):
        self.model = model


class MockListResponse:
    """Mock response from ollama.Client().list().

    The real ollama library returns an object with a .models attribute
    containing a list of model objects.
    """

    def __init__(self, models: list[str]):
        self.models = [MockModel(m) for m in models]


class MockChatResponse:
    """Mock response from ollama.Client().chat()."""

    def __init__(self, content: str):
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
    with patch("services.model_service.ollama") as mock_ollama:
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
    with patch("agents.base.ollama") as mock_ollama:
        mock_client = MagicMock()

        # Default chat response
        def default_chat(*args, **kwargs):
            return MockChatResponse("Default AI response")

        mock_client.chat.side_effect = default_chat
        mock_ollama.Client.return_value = mock_client

        yield mock_client

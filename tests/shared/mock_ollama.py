"""Shared Ollama mock utilities for Story Factory tests.

This module consolidates all Ollama mocking to ensure consistent behavior
across unit and component tests. All tests should use this shared mock
rather than defining their own MockOllamaClient implementations.

Usage:
    # In conftest.py (autouse fixture handles this automatically)
    from tests.shared.mock_ollama import MockOllamaClient, setup_ollama_mocks

    # For manual mocking in specific tests
    from tests.shared.mock_ollama import MockOllamaClient
    with patch("ollama.Client", MockOllamaClient):
        ...
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)

# Type alias to avoid shadowing by MockOllamaClient.list method
MessageList = list

# Use a model that exists in RECOMMENDED_MODELS
# This model has tags ["continuity", "interviewer", "suggestion"] in RECOMMENDED_MODELS
# The setup_ollama_mocks function patches get_model_tags to add all roles
TEST_MODEL = "huihui_ai/dolphin3-abliterated:8b"

# All role tags that agents need for auto-selection
ALL_ROLE_TAGS = [
    "interviewer",
    "architect",
    "writer",
    "editor",
    "continuity",
    "validator",
    "suggestion",
    "quality",
]

# Valid Ollama hosts that the mock accepts (default Ollama endpoints)
VALID_HOSTS = [
    None,
    "http://localhost:11434",
    "localhost:11434",
    "http://127.0.0.1:11434",
]


class MockResponse:
    """Mock response object supporting both attribute and dict access.

    This supports both access patterns used in the codebase:
    - Object access: response.response
    - Dict access: response["response"]
    """

    def __init__(self, response_text: str):
        """Initialize with the response text.

        Args:
            response_text: The text content of the response.
        """
        self.response = response_text

    def __getitem__(self, key: str):
        """Support dict-style access for compatibility."""
        if key == "response":
            return self.response
        return None


class MockModel:
    """Mock model object supporting both attribute and dict access.

    Supports both access patterns:
    - Object access: model.model
    - Dict access: model["name"]
    """

    def __init__(self, model_name: str):
        """Initialize with the model name.

        Args:
            model_name: The model identifier.
        """
        self.model = model_name

    def __getitem__(self, key: str):
        """Support dict-style access for compatibility."""
        if key == "name":
            return self.model
        return None


class MockListResponse:
    """Mock response for ollama.Client.list().

    Supports both access patterns:
    - Object access: response.models, model.model
    - Dict access: response.get("models", []), model["name"]
    """

    def __init__(self, models: list[str] | None = None):
        """Initialize with a list of model names.

        Args:
            models: List of model names. Defaults to [TEST_MODEL].
        """
        if models is None:
            models = [TEST_MODEL]
        self.models = [MockModel(name) for name in models]

    def get(self, key: str, default=None):
        """Support dict-like access for code that uses .get()."""
        if key == "models":
            return [{"name": m.model} for m in self.models]
        return default


class MockOllamaClient:
    """Mock Ollama client that returns safe defaults without real connections.

    This mock:
    - Prevents real Ollama connections during tests
    - Supports both dict and object access patterns used in the codebase
    - Raises ConnectionError for invalid hosts (to test error handling)
    - Returns consistent mock responses for all API methods
    """

    def __init__(self, host: str | None = None, timeout: float | None = None):
        """Initialize the mock Ollama client.

        Args:
            host: Host address (validated against VALID_HOSTS).
            timeout: Request timeout in seconds.

        Raises:
            ConnectionError: If host is not in VALID_HOSTS.
        """
        self.host = host
        self.timeout = timeout

        # Simulate connection failure for invalid hosts
        if host is not None and host not in VALID_HOSTS:
            raise ConnectionError(f"Failed to connect to {host}")

    def list(self) -> MockListResponse:
        """Return a mock list of installed models.

        Returns:
            MockListResponse with TEST_MODEL.
        """
        return MockListResponse([TEST_MODEL])

    def generate(
        self,
        model: str | None = None,
        prompt: str | None = None,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> MockResponse:
        """Return a mock generate response.

        Args:
            model: Model identifier.
            prompt: The prompt text.
            options: Generation options.
            **kwargs: Additional arguments.

        Returns:
            MockResponse with default text.
        """
        return MockResponse("Mock response from AI")

    def chat(
        self,
        model: str | None = None,
        messages: MessageList[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return a mock chat response.

        Args:
            model: Model identifier.
            messages: Conversation messages.
            options: Generation options.
            **kwargs: Additional arguments.

        Returns:
            Dict with message content matching Ollama API format.
        """
        return {
            "message": {"content": "Mock AI response", "role": "assistant"},
            "done": True,
        }

    def pull(self, model: str, stream: bool = False):
        """Mock model pulling.

        Args:
            model: Model to pull.
            stream: Whether to stream progress.

        Returns:
            Iterator of progress dicts if stream=True, else summary dict.
        """
        if stream:
            return iter(
                [
                    {"status": "pulling", "completed": 50, "total": 100},
                    {"status": "success", "completed": 100, "total": 100},
                ]
            )
        return {"status": "success"}

    def delete(self, model: str) -> dict[str, str]:
        """Mock model deletion.

        Args:
            model: Model to delete.

        Returns:
            Success status dict.
        """
        return {"status": "success"}


def create_mock_subprocess_run(original_run):
    """Create a mock subprocess.run that intercepts ollama and nvidia-smi commands.

    Args:
        original_run: The original subprocess.run function.

    Returns:
        A mock function that intercepts specific commands.
    """

    def mock_subprocess_run(cmd, *args, **kwargs):
        """Intercept subprocess.run for ollama and nvidia-smi commands."""
        # Get the executable name robustly
        if isinstance(cmd, list):
            cmd_str = cmd[0] if cmd else ""
        else:
            cmd_str = cmd.split()[0] if cmd else ""

        # Extract base name (handle full paths like /usr/bin/ollama)
        cmd_base = os.path.basename(cmd_str)

        if cmd_base == "ollama":

            class MockOllamaResult:
                stdout = (
                    f"NAME                              ID      SIZE    MODIFIED\n"
                    f"{TEST_MODEL}    abc123  8.0 GB  2 days ago\n"
                )
                stderr = ""
                returncode = 0

            return MockOllamaResult()

        if cmd_base == "nvidia-smi":

            class MockNvidiaSmiResult:
                stdout = "8192"  # 8GB VRAM
                stderr = ""
                returncode = 0

            return MockNvidiaSmiResult()

        # Pass through other subprocess calls
        return original_run(cmd, *args, **kwargs)

    return mock_subprocess_run


def setup_ollama_mocks(monkeypatch):
    """Set up all Ollama-related mocks for a test.

    This function sets up:
    1. Mock ollama.Client class
    2. Mock subprocess.run for ollama list and nvidia-smi
    3. Mock Settings.get_model_tags to return all roles for TEST_MODEL

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import subprocess

    # Mock ollama.Client
    monkeypatch.setattr("ollama.Client", MockOllamaClient)

    # Mock subprocess.run for ollama and nvidia-smi commands
    original_subprocess_run = subprocess.run
    mock_run = create_mock_subprocess_run(original_subprocess_run)
    monkeypatch.setattr("subprocess.run", mock_run)

    # Mock Settings.get_model_tags to return all roles for TEST_MODEL
    # But only if TEST_MODEL is the only installed model (to not interfere
    # with tests that set up specific model selection scenarios)
    try:
        import settings as settings_module

        original_get_model_tags = settings_module.Settings.get_model_tags

        def mock_get_model_tags(self, model_id: str) -> list[str]:
            """Return all role tags for TEST_MODEL when it's the only installed model.

            IMPORTANT: We look up get_installed_models_with_sizes dynamically from
            the settings module rather than importing it, because tests may patch
            it after this fixture runs.
            """
            if model_id == TEST_MODEL:
                try:
                    # Dynamic lookup to respect test patches
                    get_installed = getattr(
                        settings_module, "get_installed_models_with_sizes", None
                    )
                    if get_installed is None:
                        return ALL_ROLE_TAGS

                    installed = get_installed()
                    # Only apply all tags if TEST_MODEL is the ONLY installed model
                    # or if no models are installed (fallback behavior)
                    if not installed or (len(installed) == 1 and TEST_MODEL in installed):
                        return ALL_ROLE_TAGS
                except Exception:
                    # If we can't check, apply all tags as a safe default
                    return ALL_ROLE_TAGS
            return original_get_model_tags(self, model_id)

        monkeypatch.setattr(settings_module.Settings, "get_model_tags", mock_get_model_tags)
    except ImportError:
        logger.debug("Could not import Settings for get_model_tags mock")

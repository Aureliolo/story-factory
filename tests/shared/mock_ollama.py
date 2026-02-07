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

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)

# Type alias to avoid shadowing by MockOllamaClient.list method
MessageList = list

# Use a fake model name for tests — never use real model IDs from RECOMMENDED_MODELS.
# The setup_ollama_mocks function patches get_model_tags to return all role tags
# for this test model, so agents can auto-select it for any role.
TEST_MODEL = "test-model:8b"

# All role tags that agents need for auto-selection
ALL_ROLE_TAGS = [
    "interviewer",
    "architect",
    "writer",
    "editor",
    "continuity",
    "suggestion",
    "quality",
    "judge",
]


class MockResponse:
    """Mock response object supporting both attribute and dict access.

    This supports both access patterns used in the codebase:
    - Object access: response.response
    - Dict access: response["response"]
    """

    def __init__(self, response_text: str):
        """
        Initialize the mock response object storing the given text as the `response` attribute and accessible via `response["response"]`.

        Parameters:
            response_text (str): Text content to store as the mock response.
        """
        self.response = response_text

    def __getitem__(self, key: str):
        """
        Provide dict-style access to the mock response object.

        Parameters:
            key (str): The dictionary key to retrieve; supports "response".

        Returns:
            The stored response string when `key` is "response", `None` otherwise.
        """
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
        """
        Create a MockModel that represents a model identifier.

        Parameters:
            model_name (str): The model identifier to expose as the `model` attribute and via dict-style access under the `"name"` key.
        """
        self.model = model_name

    def __getitem__(self, key: str):
        """
        Provide dict-style access to the wrapped model name.

        Parameters:
            key (str): The dictionary key to retrieve; only "name" is supported.

        Returns:
            The model name (str) when `key` is "name", `None` otherwise.
        """
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
        """
        Create a MockListResponse containing MockModel instances for each provided model name.

        Parameters:
            models (list[str] | None): Model names to include; if None, defaults to [TEST_MODEL].
        """
        if models is None:
            models = [TEST_MODEL]
        self.models = [MockModel(name) for name in models]

    def get(self, key: str, default=None):
        """
        Provide dict-like `.get()` access for MockListResponse.

        When `key` is "models", return a list of dictionaries each containing a `"name"` key mapped to the model name; otherwise return `default`.

        Parameters:
            key (str): The dictionary key to retrieve.
            default: Value to return if `key` is not "models".

        Returns:
            list[dict] | Any: A list of `{"name": model_name}` dictionaries when `key == "models"`, otherwise `default`.
        """
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
            host: Host address.
            timeout: Request timeout in seconds.
        """
        self.host = host
        self.timeout = timeout

    def list(self) -> MockListResponse:
        """
        Provide a mock list of installed models.

        Returns:
            MockListResponse: A response containing TEST_MODEL as the sole installed model.
        """
        return MockListResponse([TEST_MODEL])

    def generate(
        self,
        model: str | None = None,
        prompt: str | None = None,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> MockResponse:
        """
        Produce a mock generation response for the given model and prompt.

        Returns:
            MockResponse: MockResponse containing the text "Mock response from AI".
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
        """
        Simulate pulling a model from Ollama for tests.

        Parameters:
            model (str): Identifier of the model to pull.
            stream (bool): If True, return an iterator of progress updates; if False, return a final summary.

        Returns:
            Iterator[dict]: When `stream` is True, yields progress dictionaries with keys `status`, `completed`, and `total`.
            dict: When `stream` is False, a summary dictionary (e.g., `{"status": "success"}`).
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
    """
    Return a callable suitable for monkeypatching that intercepts 'ollama' and 'nvidia-smi' subprocess calls and returns predefined mock results.

    Parameters:
        original_run (callable): The real subprocess.run function to delegate non-intercepted commands to.

    Returns:
        mock_subprocess_run (callable): A replacement for subprocess.run that returns canned outputs for 'ollama' and 'nvidia-smi' and calls `original_run` for all other commands.
    """

    def mock_subprocess_run(cmd, *args, **kwargs):
        """
        Intercept specific subprocess commands used in tests and return canned results for them, otherwise delegate to the original subprocess runner.

        Returns:
            A subprocess.run-compatible result object with `stdout`, `stderr`, and `returncode`.
            - For `ollama`: `stdout` contains a sample model listing that includes `TEST_MODEL`.
            - For `nvidia-smi`: `stdout` contains GPU memory in megabytes (e.g., "8192").
            - For other commands: the result returned by the original `subprocess.run`.
        """
        # Get the executable name robustly
        if isinstance(cmd, list):
            cmd_str = cmd[0] if cmd else ""
        else:
            cmd_str = cmd.split()[0] if cmd else ""

        # Extract base name (handle full paths like /usr/bin/ollama)
        cmd_base = os.path.basename(cmd_str)

        if cmd_base == "ollama":

            class MockOllamaResult:
                """Mock subprocess result for ollama CLI commands."""

                stdout = (
                    f"NAME                              ID      SIZE    MODIFIED\n"
                    f"{TEST_MODEL}    abc123  8.0 GB  2 days ago\n"
                )
                stderr = ""
                returncode = 0

            return MockOllamaResult()

        if cmd_base == "nvidia-smi":

            class MockNvidiaSmiResult:
                """Mock subprocess result for nvidia-smi commands."""

                stdout = "8192"  # 8GB VRAM
                stderr = ""
                returncode = 0

            return MockNvidiaSmiResult()

        # Pass through other subprocess calls
        return original_run(cmd, *args, **kwargs)

    return mock_subprocess_run


def setup_ollama_mocks(monkeypatch):
    """
    Configure shared Ollama-related test doubles on the provided pytest monkeypatch.

    Sets patched implementations for:
    - ollama.Client to a MockOllamaClient,
    - subprocess.run to intercept `ollama` and `nvidia-smi` calls with deterministic responses,
    - settings.Settings.get_model_tags to return all role tags for TEST_MODEL when that model is the only (or no) installed model; gracefully skips this patch if the settings module is unavailable.
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
        import src.settings as settings_module

        original_get_model_tags = settings_module.Settings.get_model_tags

        def mock_get_model_tags(self, model_id: str) -> list[str]:
            """
            Provide role tags for a model identifier, with special-case behavior for the test model.

            Parameters:
                model_id (str): The model identifier to query; used to detect the special-cased TEST_MODEL.

            Returns:
                list[str]: A list of role tag strings. If `model_id` equals `TEST_MODEL` and either no models are installed or `TEST_MODEL` is the only installed model — or if the installed-models lookup cannot be performed — returns `ALL_ROLE_TAGS`. Otherwise returns the tags provided by the settings module for the given model.
            """
            if model_id == TEST_MODEL:
                try:
                    # Dynamic lookup to respect test patches.
                    # Tests may patch either src.settings.get_installed_models_with_sizes
                    # or src.settings._settings.get_installed_models_with_sizes, so we
                    # check BOTH locations and if either shows non-default models,
                    # defer to original tag lookup.
                    import sys

                    installed_results = []

                    # Check __init__.py namespace (patched by most tests)
                    fn_init = getattr(settings_module, "get_installed_models_with_sizes", None)
                    if fn_init is not None:
                        try:
                            installed_results.append(fn_init())
                        except Exception:
                            pass

                    # Check _settings submodule namespace (patched by settings tests)
                    _settings_mod = sys.modules.get("src.settings._settings")
                    if _settings_mod:
                        fn_sub = getattr(_settings_mod, "get_installed_models_with_sizes", None)
                        if fn_sub is not None and fn_sub is not fn_init:
                            try:
                                installed_results.append(fn_sub())
                            except Exception:
                                pass

                    if not installed_results:
                        return ALL_ROLE_TAGS

                    # If ANY source shows non-default models, don't apply all tags
                    for installed in installed_results:
                        if installed and not (len(installed) == 1 and TEST_MODEL in installed):
                            return original_get_model_tags(self, model_id)

                    # All sources show default (only TEST_MODEL or empty)
                    return ALL_ROLE_TAGS
                except Exception:
                    # If we can't check, apply all tags as a safe default
                    return ALL_ROLE_TAGS
            return original_get_model_tags(self, model_id)

        monkeypatch.setattr(settings_module.Settings, "get_model_tags", mock_get_model_tags)
    except ImportError:
        logger.debug("Could not import Settings for get_model_tags mock")

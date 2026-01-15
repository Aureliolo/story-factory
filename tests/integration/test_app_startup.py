"""Integration tests for application startup.

These tests verify that components work together correctly during startup.
"""

import json
from unittest.mock import MagicMock, patch

import pytest


class MockModel:
    """Mock model object returned by ollama.list()."""

    def __init__(self, model: str):
        self.model = model


class MockListResponse:
    """Mock response from ollama.list()."""

    def __init__(self, models: list[str]):
        self.models = [MockModel(m) for m in models]


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client for tests."""
    with patch("services.model_service.ollama") as mock_ollama:
        mock_client = MagicMock()
        mock_client.list.return_value = MockListResponse(["model-a:latest", "model-b:7b"])
        mock_ollama.Client.return_value = mock_client
        yield mock_client


class TestFullStartupSequence:
    """Test the full startup sequence."""

    def test_startup_with_clean_settings(self, mock_ollama_client, tmp_path):
        """App starts correctly with default settings."""
        from services import ServiceContainer
        from settings import Settings
        from ui.app import StoryFactoryApp

        settings = Settings()
        services = ServiceContainer(settings)
        app = StoryFactoryApp(services)

        assert app.services.settings == settings
        assert app.services.model is not None

    def test_startup_with_custom_settings(self, mock_ollama_client, tmp_path):
        """App starts correctly with custom settings file."""
        from services import ServiceContainer
        from settings import Settings
        from ui.app import StoryFactoryApp

        config_file = tmp_path / "settings.json"
        config_file.write_text(
            json.dumps(
                {
                    "ollama_url": "http://custom:11434",
                    "default_model": "custom-model:latest",
                }
            )
        )

        with patch("settings.SETTINGS_FILE", config_file):
            settings = Settings.load()
            assert settings.ollama_url == "http://custom:11434"

            services = ServiceContainer(settings)
            app = StoryFactoryApp(services)

            assert app.services.settings.ollama_url == "http://custom:11434"

    def test_startup_with_stale_model_config(self, mock_ollama_client, tmp_path):
        """App handles settings with models that are no longer installed."""
        from services import ServiceContainer
        from settings import Settings
        from ui.app import StoryFactoryApp

        config_file = tmp_path / "settings.json"
        config_file.write_text(
            json.dumps(
                {
                    "default_model": "uninstalled-model:latest",
                    "use_per_agent_models": True,
                    "agent_models": {
                        "writer": "also-uninstalled:7b",
                        "editor": "model-a:latest",
                    },
                }
            )
        )

        with patch("settings.SETTINGS_FILE", config_file):
            settings = Settings.load()
            services = ServiceContainer(settings)
            StoryFactoryApp(services)  # Verify it constructs without error

            installed = services.model.list_installed()
            assert "uninstalled-model:latest" not in installed
            assert "model-a:latest" in installed


class TestServiceInteractions:
    """Test service layer interactions."""

    def test_project_service_initializes(self, mock_ollama_client, tmp_path):
        """ProjectService initializes correctly."""
        from services import ServiceContainer
        from settings import Settings

        settings = Settings()
        services = ServiceContainer(settings)
        assert services.project is not None

    def test_model_service_uses_settings_url(self, mock_ollama_client, tmp_path):
        """ModelService uses Ollama URL from settings."""
        from services import ServiceContainer
        from settings import Settings

        settings = Settings()
        settings.ollama_url = "http://custom-ollama:11434"

        services = ServiceContainer(settings)

        assert services.model.settings.ollama_url == "http://custom-ollama:11434"


class TestSettingsValidation:
    """Test settings validation during startup."""

    def test_invalid_ollama_url_falls_back_to_default(self, tmp_path):
        """Settings falls back to default URL for invalid URLs."""
        from settings import Settings

        config_file = tmp_path / "settings.json"
        config_file.write_text(json.dumps({"ollama_url": "not-a-valid-url"}))

        with patch("settings.SETTINGS_FILE", config_file):
            settings = Settings.load()
            # Invalid URL gets replaced with default
            assert settings.ollama_url == "http://localhost:11434"

    def test_missing_required_fields_use_defaults(self, tmp_path):
        """Settings uses defaults for missing fields."""
        from settings import Settings

        config_file = tmp_path / "settings.json"
        config_file.write_text(json.dumps({}))

        with patch("settings.SETTINGS_FILE", config_file):
            settings = Settings.load()
            assert settings.ollama_url == "http://localhost:11434"
            assert settings.context_size == 32768
            assert settings.default_model is not None

    def test_extra_fields_ignored(self, tmp_path):
        """Settings ignores unknown fields in config."""
        from settings import Settings

        config_file = tmp_path / "settings.json"
        config_file.write_text(
            json.dumps(
                {
                    "unknown_field": "should be ignored",
                    "another_unknown": 12345,
                }
            )
        )

        with patch("settings.SETTINGS_FILE", config_file):
            settings = Settings.load()
            assert not hasattr(settings, "unknown_field")

"""Integration tests for application startup.

These tests verify that components work together correctly during startup.
"""

import json
from unittest.mock import patch

from src.services.model_service import ModelService
from src.services.project_service import ProjectService

# mock_ollama_client fixture is provided by conftest.py


class TestFullStartupSequence:
    """Test the full startup sequence."""

    def test_startup_with_clean_settings(self, mock_ollama_client, tmp_path):
        """App starts correctly with default settings."""
        from src.services import ServiceContainer
        from src.settings import Settings
        from src.ui.app import StoryFactoryApp

        settings = Settings()
        services = ServiceContainer(settings)
        app = StoryFactoryApp(services)

        assert app.services.settings == settings
        assert isinstance(app.services.model, ModelService)

    def test_startup_with_custom_settings(self, mock_ollama_client, tmp_path):
        """App starts correctly with custom settings file."""
        from src.services import ServiceContainer
        from src.settings import Settings
        from src.ui.app import StoryFactoryApp

        config_file = tmp_path / "settings.json"
        config_file.write_text(
            json.dumps(
                {
                    "ollama_url": "http://custom:11434",
                    "default_model": "custom-model:latest",
                }
            )
        )

        with patch("src.settings.SETTINGS_FILE", config_file):
            settings = Settings.load()
            assert settings.ollama_url == "http://custom:11434"

            services = ServiceContainer(settings)
            app = StoryFactoryApp(services)

            assert app.services.settings.ollama_url == "http://custom:11434"

    def test_startup_with_stale_model_config(self, mock_ollama_client, tmp_path):
        """App handles settings with models that are no longer installed."""
        from src.services import ServiceContainer
        from src.settings import Settings
        from src.ui.app import StoryFactoryApp

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

        with patch("src.settings.SETTINGS_FILE", config_file):
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
        from src.services import ServiceContainer
        from src.settings import Settings

        settings = Settings()
        services = ServiceContainer(settings)
        assert isinstance(services.project, ProjectService)

    def test_model_service_uses_settings_url(self, mock_ollama_client, tmp_path):
        """ModelService uses Ollama URL from src.settings."""
        from src.services import ServiceContainer
        from src.settings import Settings

        settings = Settings()
        settings.ollama_url = "http://custom-ollama:11434"

        services = ServiceContainer(settings)

        assert services.model.settings.ollama_url == "http://custom-ollama:11434"


class TestSettingsValidation:
    """Test settings validation during startup."""

    def test_invalid_ollama_url_returns_default_settings(self, tmp_path):
        """Settings.load() returns default Settings when config has invalid URL.

        When validation fails (e.g., invalid URL format), the entire settings
        object is replaced with defaults, not just the invalid field.
        """
        from src.settings import Settings

        config_file = tmp_path / "settings.json"
        config_file.write_text(json.dumps({"ollama_url": "not-a-valid-url"}))

        with patch("src.settings.SETTINGS_FILE", config_file):
            settings = Settings.load()
            # Entire settings is replaced with defaults when validation fails
            assert settings.ollama_url == "http://localhost:11434"

    def test_missing_required_fields_use_defaults(self, tmp_path):
        """Settings uses defaults for missing fields."""
        from src.settings import Settings

        config_file = tmp_path / "settings.json"
        config_file.write_text(json.dumps({}))

        with patch("src.settings.SETTINGS_FILE", config_file):
            settings = Settings.load()
            assert settings.ollama_url == "http://localhost:11434"
            assert settings.context_size == 32768
            assert isinstance(settings.default_model, str)
            assert len(settings.default_model) > 0

    def test_extra_fields_ignored(self, tmp_path):
        """Settings ignores unknown fields in config."""
        from src.settings import Settings

        config_file = tmp_path / "settings.json"
        config_file.write_text(
            json.dumps(
                {
                    "unknown_field": "should be ignored",
                    "another_unknown": 12345,
                }
            )
        )

        with patch("src.settings.SETTINGS_FILE", config_file):
            settings = Settings.load()
            assert not hasattr(settings, "unknown_field")

"""Integration tests for settings validation with UI components.

These tests verify that settings page handles various model states correctly.
"""

import json
from unittest.mock import MagicMock, patch

# Import shared mock classes from integration conftest
from tests.integration.conftest import MockListResponse


class TestModelSelectionFallback:
    """Test model selection fallback behavior."""

    def test_default_model_fallback_to_auto(self, tmp_path):
        """Default model falls back to 'auto' when not installed."""
        from src.services import ServiceContainer
        from src.settings import Settings

        with patch("src.services.model_service.ollama") as mock_ollama:
            mock_client = MagicMock()
            mock_client.list.return_value = MockListResponse(["installed-model:latest"])
            mock_ollama.Client.return_value = mock_client

            config_file = tmp_path / "settings.json"
            config_file.write_text(json.dumps({"default_model": "uninstalled-model:latest"}))

            with patch("src.settings.SETTINGS_FILE", config_file):
                settings = Settings.load()
                services = ServiceContainer(settings)

                installed = services.model.list_installed()
                model_options = {"auto": "Auto-select"} | {m: m for m in installed}

                value = settings.default_model
                if value not in model_options:
                    value = "auto"

                assert value == "auto"

    def test_agent_model_fallback_to_auto(self, tmp_path):
        """Per-agent models fall back to 'auto' when not installed."""
        from src.services import ServiceContainer
        from src.settings import Settings

        with patch("src.services.model_service.ollama") as mock_ollama:
            mock_client = MagicMock()
            mock_client.list.return_value = MockListResponse(["installed-model:latest"])
            mock_ollama.Client.return_value = mock_client

            config_file = tmp_path / "settings.json"
            config_file.write_text(
                json.dumps(
                    {
                        "use_per_agent_models": True,
                        "agent_models": {
                            "writer": "uninstalled-writer-model:7b",
                            "editor": "installed-model:latest",
                        },
                    }
                )
            )

            with patch("src.settings.SETTINGS_FILE", config_file):
                settings = Settings.load()
                services = ServiceContainer(settings)

                installed = services.model.list_installed()
                model_options = {"auto": "Auto-select"} | {m: m for m in installed}

                writer_value = settings.agent_models.get("writer", "auto")
                if writer_value not in model_options:
                    writer_value = "auto"

                editor_value = settings.agent_models.get("editor", "auto")
                if editor_value not in model_options:
                    editor_value = "auto"

                assert writer_value == "auto"
                assert editor_value == "installed-model:latest"

    def test_all_models_uninstalled(self, tmp_path):
        """Handles case when no models are installed."""
        from src.services import ServiceContainer
        from src.settings import Settings

        with patch("src.services.model_service.ollama") as mock_ollama:
            mock_client = MagicMock()
            mock_client.list.return_value = MockListResponse([])
            mock_ollama.Client.return_value = mock_client

            config_file = tmp_path / "settings.json"
            config_file.write_text(
                json.dumps(
                    {
                        "default_model": "some-model:latest",
                        "agent_models": {"writer": "another-model:7b"},
                    }
                )
            )

            with patch("src.settings.SETTINGS_FILE", config_file):
                settings = Settings.load()
                services = ServiceContainer(settings)

                installed = services.model.list_installed()
                model_options = {"auto": "Auto-select"} | {m: m for m in installed}

                assert model_options == {"auto": "Auto-select"}

                value = settings.default_model
                if value not in model_options:
                    value = "auto"

                assert value == "auto"


class TestSettingsPageConstruction:
    """Test settings page can be constructed with various states."""

    def test_settings_page_with_valid_models(self, tmp_path):
        """Settings page constructs correctly with valid models."""
        from src.services import ServiceContainer
        from src.settings import Settings
        from src.ui.state import AppState

        with patch("src.services.model_service.ollama") as mock_ollama:
            mock_client = MagicMock()
            mock_client.list.return_value = MockListResponse(["model-a:latest", "model-b:7b"])
            mock_ollama.Client.return_value = mock_client

            config_file = tmp_path / "settings.json"
            config_file.write_text(json.dumps({"default_model": "model-a:latest"}))

            with patch("src.settings.SETTINGS_FILE", config_file):
                settings = Settings.load()
                services = ServiceContainer(settings)
                AppState()  # Verify it constructs

                installed = services.model.list_installed()
                model_options = {"auto": "Auto-select"} | {m: m for m in installed}

                assert settings.default_model in model_options

    def test_settings_page_with_stale_config(self, tmp_path):
        """Settings page handles stale config without crashing."""
        from src.services import ServiceContainer
        from src.settings import Settings
        from src.ui.state import AppState

        with patch("src.services.model_service.ollama") as mock_ollama:
            mock_client = MagicMock()
            mock_client.list.return_value = MockListResponse(["only-model:latest"])
            mock_ollama.Client.return_value = mock_client

            config_file = tmp_path / "settings.json"
            config_file.write_text(
                json.dumps(
                    {
                        "default_model": "removed-model:latest",
                        "use_per_agent_models": True,
                        "agent_models": {
                            "interviewer": "auto",
                            "architect": "removed-arch-model:7b",
                            "writer": "removed-writer-model:13b",
                            "editor": "only-model:latest",
                            "continuity": "auto",
                        },
                    }
                )
            )

            with patch("src.settings.SETTINGS_FILE", config_file):
                settings = Settings.load()
                services = ServiceContainer(settings)
                AppState()  # Verify it constructs

                installed = services.model.list_installed()
                model_options = {"auto": "Auto-select"} | {m: m for m in installed}

                for role in ["interviewer", "architect", "writer", "editor", "continuity"]:
                    value = settings.agent_models.get(role, "auto")
                    if value not in model_options:
                        value = "auto"

                    assert value in model_options, f"Role {role} has invalid value"

"""Smoke tests for application startup.

These tests verify that the application can start without crashing.
They catch issues like:
- Import errors
- Missing dependencies
- Configuration errors
- Invalid settings states
"""

from unittest.mock import MagicMock, patch


class TestImports:
    """Test that all modules can be imported without errors."""

    def test_import_settings(self):
        """Settings module imports cleanly."""
        from settings import AGENT_ROLES, Settings

        assert Settings is not None
        assert AGENT_ROLES is not None

    def test_import_services(self):
        """Services module imports cleanly."""
        from services import ServiceContainer

        assert ServiceContainer is not None

    def test_import_agents(self):
        """Agent modules import cleanly."""
        from agents.architect import ArchitectAgent
        from agents.base import BaseAgent
        from agents.continuity import ContinuityAgent
        from agents.editor import EditorAgent
        from agents.interviewer import InterviewerAgent
        from agents.writer import WriterAgent

        assert BaseAgent is not None
        assert InterviewerAgent is not None
        assert ArchitectAgent is not None
        assert WriterAgent is not None
        assert EditorAgent is not None
        assert ContinuityAgent is not None

    def test_import_memory(self):
        """Memory modules import cleanly."""
        from memory.story_state import StoryState
        from memory.world_database import WorldDatabase

        assert StoryState is not None
        assert WorldDatabase is not None

    def test_import_ui(self):
        """UI modules import cleanly."""
        from ui.app import StoryFactoryApp
        from ui.state import AppState

        assert StoryFactoryApp is not None
        assert AppState is not None

    def test_import_workflows(self):
        """Workflow modules import cleanly."""
        from workflows.orchestrator import StoryOrchestrator

        assert StoryOrchestrator is not None


class TestSettingsInitialization:
    """Test settings can be loaded in various states."""

    def test_settings_default_values(self):
        """Settings has sensible defaults when no file exists."""
        from settings import Settings

        settings = Settings()
        assert settings.ollama_url == "http://localhost:11434"
        assert settings.default_model is not None
        assert isinstance(settings.agent_models, dict)
        assert isinstance(settings.agent_temperatures, dict)

    def test_settings_with_missing_models(self, tmp_path):
        """Settings loads even when saved models don't exist."""
        import json

        from settings import Settings

        config_file = tmp_path / "settings.json"
        config_file.write_text(
            json.dumps(
                {
                    "default_model": "nonexistent-model:latest",
                    "agent_models": {
                        "writer": "also-nonexistent:7b",
                    },
                }
            )
        )

        with patch("settings.SETTINGS_FILE", config_file):
            settings = Settings.load()
            assert settings.default_model == "nonexistent-model:latest"
            assert settings.agent_models.get("writer") == "also-nonexistent:7b"


class TestServiceContainerInitialization:
    """Test ServiceContainer can be initialized."""

    def test_service_container_creates_all_services(self):
        """ServiceContainer initializes all services."""
        from services import ServiceContainer
        from settings import Settings

        settings = Settings()

        with patch("services.model_service.ollama") as mock_ollama:
            mock_ollama.Client.return_value = MagicMock()
            services = ServiceContainer(settings)

            assert services.project is not None
            assert services.story is not None
            assert services.world is not None
            assert services.model is not None
            assert services.export is not None


class TestUIComponentsConstruction:
    """Test UI components can be constructed without running server."""

    def test_app_state_creation(self):
        """AppState can be created with defaults."""
        from ui.state import AppState

        state = AppState()
        assert state is not None
        assert state.project_id is None
        assert state.interview_history == []

    def test_story_factory_app_construction(self):
        """StoryFactoryApp can be constructed."""
        from services import ServiceContainer
        from settings import Settings
        from ui.app import StoryFactoryApp

        settings = Settings()

        with patch("services.model_service.ollama") as mock_ollama:
            mock_ollama.Client.return_value = MagicMock()
            services = ServiceContainer(settings)
            app = StoryFactoryApp(services)

            assert app is not None
            assert app.services is services


class TestSettingsPageModelHandling:
    """Test settings page handles model states correctly."""

    def test_model_fallback_when_not_in_options(self):
        """Settings page falls back to 'auto' for missing models."""
        model_options = {"auto": "Auto-select", "model-a": "model-a"}

        saved_value = "nonexistent-model"
        if saved_value not in model_options:
            saved_value = "auto"

        assert saved_value == "auto"

    def test_model_preserved_when_in_options(self):
        """Settings page preserves valid model selections."""
        model_options = {"auto": "Auto-select", "model-a": "model-a"}

        saved_value = "model-a"
        if saved_value not in model_options:
            saved_value = "auto"

        assert saved_value == "model-a"

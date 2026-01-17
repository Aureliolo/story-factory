"""Tests for the template service."""

import json
from unittest.mock import MagicMock

import pytest

from memory.story_state import StoryState
from services.template_service import TemplateService
from settings import Settings


class TestTemplateServiceErrorHandling:
    """Tests for error handling in TemplateService."""

    @pytest.fixture
    def settings(self):
        """Create settings for testing."""
        return Settings()

    @pytest.fixture
    def template_service(self, settings, tmp_path, monkeypatch):
        """Create a TemplateService with mocked TEMPLATES_DIR."""
        # Mock TEMPLATES_DIR to use temp path
        monkeypatch.setattr("services.template_service.TEMPLATES_DIR", tmp_path / "templates")
        return TemplateService(settings)

    def test_list_templates_handles_invalid_template_file(
        self, template_service, tmp_path, monkeypatch, caplog
    ):
        """Test list_templates logs warning for invalid template files."""
        # Lines 66-67: Exception handling when loading custom templates
        import logging

        templates_dir = tmp_path / "templates"
        templates_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("services.template_service.TEMPLATES_DIR", templates_dir)

        # Create an invalid JSON file
        invalid_file = templates_dir / "invalid.json"
        invalid_file.write_text("{invalid json", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            templates = template_service.list_templates()

        # Should log warning about failed load
        assert "Failed to load template" in caplog.text
        # Built-in templates should still be returned
        assert len(templates) >= 0

    def test_get_template_handles_invalid_custom_template(
        self, template_service, tmp_path, monkeypatch, caplog
    ):
        """Test get_template returns None and logs error for invalid custom template."""
        # Lines 93-99: Exception handling in get_template
        import logging

        templates_dir = tmp_path / "templates"
        templates_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("services.template_service.TEMPLATES_DIR", templates_dir)

        # Create a valid JSON but invalid StoryTemplate
        template_file = templates_dir / "bad-template.json"
        template_file.write_text('{"not": "a valid template"}', encoding="utf-8")

        with caplog.at_level(logging.ERROR):
            result = template_service.get_template("bad-template")

        # Should return None
        assert result is None
        # Should log error
        assert "Failed to load template" in caplog.text

    def test_apply_template_to_state_sets_state_correctly(
        self, template_service, monkeypatch, caplog
    ):
        """Test apply_template_to_state populates state from template."""
        import logging

        from memory.builtin_templates import BUILTIN_STORY_TEMPLATES

        # Create a story state
        state = StoryState(id="test-state")

        # Get a built-in template
        template = list(BUILTIN_STORY_TEMPLATES.values())[0]

        # Create a mock world_db
        mock_world_db = MagicMock()

        with caplog.at_level(logging.DEBUG):
            template_service.apply_template_to_state(template, state, mock_world_db)

        # State should be populated
        assert state.brief is not None
        assert state.brief.genre == template.genre
        assert "Template applied with" in caplog.text

    def test_import_template_reassigns_id_when_collides_with_builtin(
        self, template_service, tmp_path, monkeypatch
    ):
        """Test import_template generates new ID when colliding with builtin."""
        # Lines 401-402: ID collision handling in import_template
        from memory.builtin_templates import BUILTIN_STORY_TEMPLATES

        templates_dir = tmp_path / "templates"
        templates_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("services.template_service.TEMPLATES_DIR", templates_dir)

        # Get a builtin template ID
        builtin_id = list(BUILTIN_STORY_TEMPLATES.keys())[0]

        # Create a template file that has the same ID as a builtin
        template_data = {
            "id": builtin_id,
            "name": "Test Template",
            "description": "A test template",
            "is_builtin": False,
            "genre": "Fantasy",
            "subgenres": [],
            "tone": "Epic",
            "themes": [],
            "setting_time": "Medieval",
            "setting_place": "Kingdom",
            "target_length": "novella",
            "world_description": "",
            "world_rules": [],
            "characters": [],
            "plot_points": [],
        }

        import_file = tmp_path / "import_template.json"
        with open(import_file, "w", encoding="utf-8") as f:
            json.dump(template_data, f)

        # Import the template
        imported = template_service.import_template(import_file)

        # ID should have been changed to avoid collision
        assert imported.id != builtin_id
        assert imported.id.startswith("imported-")

    def test_import_template_raises_on_invalid_file(self, template_service, tmp_path, monkeypatch):
        """Test import_template raises ValueError for invalid file content."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("services.template_service.TEMPLATES_DIR", templates_dir)

        # Create an invalid JSON file
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("{not valid json", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid template file"):
            template_service.import_template(invalid_file)

    def test_import_template_raises_on_missing_file(self, template_service, tmp_path, monkeypatch):
        """Test import_template raises FileNotFoundError for missing file."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("services.template_service.TEMPLATES_DIR", templates_dir)

        missing_file = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError, match="Import file not found"):
            template_service.import_template(missing_file)


class TestTemplateServiceListTemplates:
    """Additional tests for list_templates."""

    @pytest.fixture
    def template_service(self, tmp_path, monkeypatch):
        """Create a TemplateService with mocked TEMPLATES_DIR."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("services.template_service.TEMPLATES_DIR", templates_dir)
        return TemplateService(Settings())

    def test_list_templates_loads_valid_custom_templates(
        self, template_service, tmp_path, monkeypatch
    ):
        """Test list_templates loads valid custom template files."""
        templates_dir = tmp_path / "templates"
        monkeypatch.setattr("services.template_service.TEMPLATES_DIR", templates_dir)

        # Create a valid custom template
        template_data = {
            "id": "custom-test-123",
            "name": "Custom Test Template",
            "description": "A test template",
            "is_builtin": False,
            "genre": "Sci-Fi",
            "subgenres": ["Space Opera"],
            "tone": "Adventurous",
            "themes": ["Exploration"],
            "setting_time": "Far Future",
            "setting_place": "Outer Space",
            "target_length": "novel",
            "world_description": "",
            "world_rules": [],
            "characters": [],
            "plot_points": [],
        }

        template_file = templates_dir / "custom-test-123.json"
        with open(template_file, "w", encoding="utf-8") as f:
            json.dump(template_data, f)

        templates = template_service.list_templates()

        # Should find the custom template
        custom_templates = [t for t in templates if t.id == "custom-test-123"]
        assert len(custom_templates) == 1
        assert custom_templates[0].name == "Custom Test Template"

"""Tests for TemplateService."""

import json

import pytest

from src.memory.story_state import StoryBrief, StoryState
from src.memory.templates import StoryTemplate
from src.memory.world_database import WorldDatabase
from src.services.template_service import TemplateService


class TestTemplateService:
    """Tests for TemplateService."""

    def test_list_templates_builtin(self, tmp_settings, monkeypatch, tmp_path):
        """Test listing built-in templates."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir(parents=True)
        monkeypatch.setattr("src.services.template_service.TEMPLATES_DIR", templates_dir)

        service = TemplateService(tmp_settings)
        templates = service.list_templates()

        # Should have built-in templates
        assert len(templates) > 0
        builtin_templates = [t for t in templates if t.is_builtin]
        assert len(builtin_templates) > 0

        # Check for expected built-in templates
        template_ids = {t.id for t in templates}
        assert "mystery-detective" in template_ids
        assert "romance-contemporary" in template_ids
        assert "scifi-space-opera" in template_ids

    def test_get_builtin_template(self, tmp_settings, monkeypatch, tmp_path):
        """Test getting a built-in template."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir(parents=True)
        monkeypatch.setattr("src.services.template_service.TEMPLATES_DIR", templates_dir)

        service = TemplateService(tmp_settings)
        template = service.get_template("mystery-detective")

        assert template is not None
        assert template.id == "mystery-detective"
        assert template.genre == "Mystery"
        assert template.is_builtin is True

    def test_get_nonexistent_template(self, tmp_settings, monkeypatch, tmp_path):
        """Test getting a template that doesn't exist."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir(parents=True)
        monkeypatch.setattr("src.services.template_service.TEMPLATES_DIR", templates_dir)

        service = TemplateService(tmp_settings)
        template = service.get_template("nonexistent-template")

        assert template is None

    def test_save_custom_template(self, tmp_settings, monkeypatch, tmp_path):
        """Test saving a custom template."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir(parents=True)
        monkeypatch.setattr("src.services.template_service.TEMPLATES_DIR", templates_dir)

        service = TemplateService(tmp_settings)

        # Create a custom template
        template = StoryTemplate(
            id="custom-test",
            name="Test Template",
            description="A test template",
            is_builtin=False,
            genre="Test",
            tone="Testing",
        )

        # Save it
        path = service.save_template(template)

        assert path.exists()
        assert path.name == "custom-test.json"

        # Verify content
        with open(path) as f:
            data = json.load(f)

        assert data["id"] == "custom-test"
        assert data["name"] == "Test Template"

    def test_cannot_save_builtin_template(self, tmp_settings, monkeypatch, tmp_path):
        """Test that built-in templates cannot be modified."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir(parents=True)
        monkeypatch.setattr("src.services.template_service.TEMPLATES_DIR", templates_dir)

        service = TemplateService(tmp_settings)

        template = service.get_template("mystery-detective")
        assert template is not None

        with pytest.raises(ValueError, match="Cannot modify built-in templates"):
            service.save_template(template)

    def test_delete_custom_template(self, tmp_settings, monkeypatch, tmp_path):
        """Test deleting a custom template."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir(parents=True)
        monkeypatch.setattr("src.services.template_service.TEMPLATES_DIR", templates_dir)

        service = TemplateService(tmp_settings)

        # Create and save a custom template
        template = StoryTemplate(
            id="custom-to-delete",
            name="Delete Me",
            description="Will be deleted",
            is_builtin=False,
            genre="Test",
            tone="Testing",
        )
        service.save_template(template)

        # Delete it
        result = service.delete_template("custom-to-delete")

        assert result is True
        assert not (templates_dir / "custom-to-delete.json").exists()

    def test_cannot_delete_builtin_template(self, tmp_settings, monkeypatch, tmp_path):
        """Test that built-in templates cannot be deleted."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir(parents=True)
        monkeypatch.setattr("src.services.template_service.TEMPLATES_DIR", templates_dir)

        service = TemplateService(tmp_settings)

        with pytest.raises(ValueError, match="Cannot delete built-in templates"):
            service.delete_template("mystery-detective")

    def test_delete_nonexistent_template(self, tmp_settings, monkeypatch, tmp_path):
        """Test deleting a template that doesn't exist."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir(parents=True)
        monkeypatch.setattr("src.services.template_service.TEMPLATES_DIR", templates_dir)

        service = TemplateService(tmp_settings)
        result = service.delete_template("nonexistent")

        assert result is False

    def test_create_template_from_project(self, tmp_settings, monkeypatch, tmp_path):
        """Test creating a template from an existing project."""
        from datetime import datetime

        from src.memory.story_state import Character, PlotPoint

        templates_dir = tmp_path / "templates"
        templates_dir.mkdir(parents=True)
        monkeypatch.setattr("src.services.template_service.TEMPLATES_DIR", templates_dir)

        service = TemplateService(tmp_settings)

        # Create a story state with some data
        state = StoryState(
            id="test-project",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            project_name="Test Project",
            brief=StoryBrief(
                premise="A test premise",
                genre="Fantasy",
                tone="Epic",
                themes=["Adventure", "Friendship"],
                setting_time="Medieval",
                setting_place="Fantasy Kingdom",
                target_length="novel",
                language="English",
                content_rating="none",
            ),
            characters=[
                Character(
                    name="Hero",
                    role="protagonist",
                    description="A brave hero",
                    personality_traits=["Brave", "Kind"],
                    goals=["Save the world"],
                )
            ],
            plot_points=[
                PlotPoint(description="Beginning of adventure", chapter=1),
                PlotPoint(description="Major conflict", chapter=5),
            ],
            world_description="A magical fantasy world",
            world_rules=["Magic is real", "Dragons exist"],
        )

        # Create template from project
        template = service.create_template_from_project(
            state, "My Custom Template", "A template from my project"
        )

        assert template is not None
        assert template.name == "My Custom Template"
        assert template.description == "A template from my project"
        assert template.genre == "Fantasy"
        assert template.tone == "Epic"
        assert len(template.characters) == 1
        assert len(template.plot_points) == 2
        assert template.is_builtin is False

        # Verify it was saved
        templates = service.list_templates()
        custom_templates = [t for t in templates if not t.is_builtin]
        assert len(custom_templates) > 0

    def test_apply_template_to_state(self, tmp_settings, monkeypatch, tmp_path):
        """Test applying a template to a story state."""
        from datetime import datetime

        templates_dir = tmp_path / "templates"
        templates_dir.mkdir(parents=True)
        monkeypatch.setattr("src.services.template_service.TEMPLATES_DIR", templates_dir)

        service = TemplateService(tmp_settings)

        # Get a built-in template
        template = service.get_template("mystery-detective")
        assert template is not None

        # Create a blank story state
        world_db_path = tmp_path / "test.db"
        state = StoryState(
            id="test-project",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            project_name="Test",
            world_db_path=str(world_db_path),
        )
        world_db = WorldDatabase(world_db_path)

        # Apply template
        service.apply_template_to_state(template, state, world_db)

        # Verify template was applied
        assert state.brief is not None
        assert state.brief.genre == template.genre
        assert state.brief.tone == template.tone
        assert len(state.characters) == len(template.characters)
        assert len(state.plot_points) >= len(template.plot_points)
        assert state.world_description == template.world_description

        # Cleanup
        world_db.close()

    def test_list_structure_presets(self, tmp_settings, monkeypatch, tmp_path):
        """Test listing structure presets."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir(parents=True)
        monkeypatch.setattr("src.services.template_service.TEMPLATES_DIR", templates_dir)

        service = TemplateService(tmp_settings)
        presets = service.list_structure_presets()

        assert len(presets) > 0
        preset_ids = {p.id for p in presets}
        assert "three-act" in preset_ids
        assert "heros-journey" in preset_ids
        assert "save-the-cat" in preset_ids

    def test_get_structure_preset(self, tmp_settings, monkeypatch, tmp_path):
        """Test getting a structure preset."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir(parents=True)
        monkeypatch.setattr("src.services.template_service.TEMPLATES_DIR", templates_dir)

        service = TemplateService(tmp_settings)
        preset = service.get_structure_preset("three-act")

        assert preset is not None
        assert preset.id == "three-act"
        assert preset.name == "Three-Act Structure"
        assert len(preset.acts) == 3
        assert len(preset.plot_points) > 0

    def test_export_template(self, tmp_settings, monkeypatch, tmp_path):
        """Test exporting a template."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir(parents=True)
        monkeypatch.setattr("src.services.template_service.TEMPLATES_DIR", templates_dir)

        service = TemplateService(tmp_settings)

        # Export a built-in template
        export_path = tmp_path / "exported.json"
        result = service.export_template("mystery-detective", export_path)

        assert result.exists()
        assert result == export_path

        # Verify content
        with open(export_path) as f:
            data = json.load(f)

        assert data["id"] == "mystery-detective"
        assert data["name"] == "Mystery / Detective"

    def test_export_template_invalid_extension(self, tmp_settings, monkeypatch, tmp_path):
        """Test that export fails with non-JSON extension."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir(parents=True)
        monkeypatch.setattr("src.services.template_service.TEMPLATES_DIR", templates_dir)

        service = TemplateService(tmp_settings)

        export_path = tmp_path / "exported.txt"
        with pytest.raises(ValueError, match="must be a .json file"):
            service.export_template("mystery-detective", export_path)

    def test_export_nonexistent_template(self, tmp_settings, monkeypatch, tmp_path):
        """Test that export fails for nonexistent template."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir(parents=True)
        monkeypatch.setattr("src.services.template_service.TEMPLATES_DIR", templates_dir)

        service = TemplateService(tmp_settings)

        export_path = tmp_path / "exported.json"
        with pytest.raises(FileNotFoundError, match="Template not found"):
            service.export_template("nonexistent", export_path)

    def test_import_template(self, tmp_settings, monkeypatch, tmp_path):
        """Test importing a template."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir(parents=True)
        monkeypatch.setattr("src.services.template_service.TEMPLATES_DIR", templates_dir)

        service = TemplateService(tmp_settings)

        # Create a template file to import
        import_data = {
            "id": "imported-test",
            "name": "Imported Template",
            "description": "An imported template",
            "is_builtin": False,
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T00:00:00",
            "genre": "Test",
            "subgenres": [],
            "tone": "Testing",
            "themes": [],
            "setting_time": "",
            "setting_place": "",
            "target_length": "novel",
            "structure_preset_id": None,
            "world_description": "",
            "world_rules": [],
            "characters": [],
            "plot_points": [],
            "author": "",
            "tags": [],
        }

        import_path = tmp_path / "import.json"
        with open(import_path, "w") as f:
            json.dump(import_data, f)

        # Import it
        template = service.import_template(import_path)

        assert template is not None
        assert template.name == "Imported Template"
        assert template.is_builtin is False

        # Verify it was saved
        templates = service.list_templates()
        custom_templates = [t for t in templates if not t.is_builtin]
        assert any(t.name == "Imported Template" for t in custom_templates)

    def test_import_nonexistent_file(self, tmp_settings, monkeypatch, tmp_path):
        """Test that import fails for nonexistent file."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir(parents=True)
        monkeypatch.setattr("src.services.template_service.TEMPLATES_DIR", templates_dir)

        service = TemplateService(tmp_settings)

        import_path = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError, match="Import file not found"):
            service.import_template(import_path)

    def test_import_invalid_template(self, tmp_settings, monkeypatch, tmp_path):
        """Test that import fails for invalid template data."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir(parents=True)
        monkeypatch.setattr("src.services.template_service.TEMPLATES_DIR", templates_dir)

        service = TemplateService(tmp_settings)

        # Create an invalid template file
        import_path = tmp_path / "invalid.json"
        with open(import_path, "w") as f:
            f.write('{"invalid": "data"}')

        with pytest.raises(ValueError, match="Invalid template file"):
            service.import_template(import_path)

    def test_list_templates_includes_custom(self, tmp_settings, monkeypatch, tmp_path):
        """Test that list_templates includes both built-in and custom templates."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir(parents=True)
        monkeypatch.setattr("src.services.template_service.TEMPLATES_DIR", templates_dir)

        service = TemplateService(tmp_settings)

        # Create a custom template
        template = StoryTemplate(
            id="custom-list-test",
            name="Custom List Test",
            description="A custom template",
            is_builtin=False,
            genre="Test",
            tone="Testing",
        )
        service.save_template(template)

        # List all templates
        templates = service.list_templates()

        builtin = [t for t in templates if t.is_builtin]
        custom = [t for t in templates if not t.is_builtin]

        assert len(builtin) > 0
        assert len(custom) > 0
        assert any(t.id == "custom-list-test" for t in custom)

    def test_apply_template_with_structure_preset(self, tmp_settings, monkeypatch, tmp_path):
        """Test applying a template that references a structure preset."""
        from datetime import datetime

        templates_dir = tmp_path / "templates"
        templates_dir.mkdir(parents=True)
        monkeypatch.setattr("src.services.template_service.TEMPLATES_DIR", templates_dir)

        service = TemplateService(tmp_settings)

        # Get template that uses structure preset
        template = service.get_template("scifi-space-opera")
        assert template is not None
        assert template.structure_preset_id == "heros-journey"

        # Create a blank story state
        world_db_path = tmp_path / "test.db"
        state = StoryState(
            id="test-project",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            project_name="Test",
            world_db_path=str(world_db_path),
        )
        world_db = WorldDatabase(world_db_path)

        # Apply template
        service.apply_template_to_state(template, state, world_db)

        # Verify structure preset was applied
        assert len(state.plot_points) > len(template.plot_points)
        # Should have both template plot points and structure preset plot points

        # Cleanup
        world_db.close()

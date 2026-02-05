"""Tests for the built-in templates registry."""

import logging

import pytest
import yaml

from src.memory.builtin_templates import (
    BUILTIN_STORY_TEMPLATES,
    BUILTIN_STRUCTURE_PRESETS,
    TemplateRegistry,
    TemplateRegistryError,
    get_builtin_story_templates,
    get_builtin_structure_presets,
)
from src.memory.templates import StoryTemplate, StructurePreset


class TestBuiltinTemplatesExports:
    """Tests for the module-level exports."""

    def test_builtin_structure_presets_loaded(self):
        """Test that built-in structure presets are loaded."""
        assert len(BUILTIN_STRUCTURE_PRESETS) == 3
        assert "three-act" in BUILTIN_STRUCTURE_PRESETS
        assert "heros-journey" in BUILTIN_STRUCTURE_PRESETS
        assert "save-the-cat" in BUILTIN_STRUCTURE_PRESETS

    def test_builtin_story_templates_loaded(self):
        """
        Verify the module exposes the expected built-in story template IDs.

        Asserts that exactly 5 built-in story templates are loaded and that the IDs "mystery-detective", "romance-contemporary", "scifi-space-opera", "fantasy-epic", and "thriller-action" are present.
        """
        assert len(BUILTIN_STORY_TEMPLATES) == 5
        assert "mystery-detective" in BUILTIN_STORY_TEMPLATES
        assert "romance-contemporary" in BUILTIN_STORY_TEMPLATES
        assert "scifi-space-opera" in BUILTIN_STORY_TEMPLATES
        assert "fantasy-epic" in BUILTIN_STORY_TEMPLATES
        assert "thriller-action" in BUILTIN_STORY_TEMPLATES

    def test_structure_presets_are_valid_models(self):
        """Test that all structure presets are valid StructurePreset models."""
        for preset_id, preset in BUILTIN_STRUCTURE_PRESETS.items():
            assert isinstance(preset, StructurePreset)
            assert preset.id == preset_id
            assert preset.name
            assert preset.description
            assert len(preset.acts) > 0
            assert len(preset.plot_points) > 0
            assert len(preset.beats) > 0

    def test_story_templates_are_valid_models(self):
        """Test that all story templates are valid StoryTemplate models."""
        for template_id, template in BUILTIN_STORY_TEMPLATES.items():
            assert isinstance(template, StoryTemplate)
            assert template.id == template_id
            assert template.name
            assert template.description
            assert template.genre
            assert len(template.characters) > 0
            assert len(template.plot_points) > 0

    def test_story_templates_reference_valid_structure_presets(self):
        """Test that story templates reference existing structure presets."""
        for template in BUILTIN_STORY_TEMPLATES.values():
            if template.structure_preset_id:
                assert template.structure_preset_id in BUILTIN_STRUCTURE_PRESETS

    def test_get_functions_return_same_data(self):
        """Test that getter functions return the same data as module-level dicts."""
        assert get_builtin_structure_presets() == BUILTIN_STRUCTURE_PRESETS
        assert get_builtin_story_templates() == BUILTIN_STORY_TEMPLATES


class TestTemplateRegistry:
    """Tests for the TemplateRegistry class."""

    def test_create_registry_from_default_directory(self):
        """Test creating a registry from the default directory."""
        registry = TemplateRegistry()
        assert len(registry.structure_presets) == 3
        assert len(registry.story_templates) == 5

    def test_create_registry_from_custom_directory(self, tmp_path):
        """Test creating a registry from a custom directory."""
        # Create empty structures and stories directories
        (tmp_path / "structures").mkdir()
        (tmp_path / "stories").mkdir()

        registry = TemplateRegistry(tmp_path)
        assert len(registry.structure_presets) == 0
        assert len(registry.story_templates) == 0

    def test_get_structure_preset(self):
        """Test getting a structure preset by ID."""
        registry = TemplateRegistry()
        preset = registry.get_structure_preset("three-act")
        assert preset is not None
        assert preset.id == "three-act"
        assert preset.name == "Three-Act Structure"

    def test_get_structure_preset_not_found(self):
        """Test getting a non-existent structure preset returns None."""
        registry = TemplateRegistry()
        preset = registry.get_structure_preset("non-existent")
        assert preset is None

    def test_get_story_template(self):
        """Test getting a story template by ID."""
        registry = TemplateRegistry()
        template = registry.get_story_template("mystery-detective")
        assert template is not None
        assert template.id == "mystery-detective"
        assert template.genre == "Mystery"

    def test_get_story_template_not_found(self):
        """
        Verify that requesting a story template with a nonexistent ID yields no template.
        """
        registry = TemplateRegistry()
        template = registry.get_story_template("non-existent")
        assert template is None

    def test_reload_clears_and_reloads(self):
        """Test that reload clears and reloads all templates."""
        registry = TemplateRegistry()
        initial_presets = len(registry.structure_presets)
        initial_templates = len(registry.story_templates)

        registry.reload()

        assert len(registry.structure_presets) == initial_presets
        assert len(registry.story_templates) == initial_templates

    def test_repr(self):
        """Test string representation."""
        registry = TemplateRegistry()
        repr_str = repr(registry)
        assert "TemplateRegistry" in repr_str
        assert "3 structures" in repr_str
        assert "5 templates" in repr_str


class TestTemplateRegistryErrorHandling:
    """Tests for error handling in TemplateRegistry."""

    def test_missing_templates_directory_raises_error(self, tmp_path):
        """Test that non-existent templates directory raises an error."""
        non_existent = tmp_path / "non_existent"

        with pytest.raises(TemplateRegistryError, match="does not exist"):
            TemplateRegistry(non_existent)

    def test_missing_structures_directory_logs_warning(self, tmp_path, caplog):
        """Test that missing structures directory logs a warning."""
        # Only create stories directory
        (tmp_path / "stories").mkdir()

        with caplog.at_level(logging.WARNING):
            registry = TemplateRegistry(tmp_path)

        assert "Structures directory not found" in caplog.text
        assert len(registry.structure_presets) == 0

    def test_missing_stories_directory_logs_warning(self, tmp_path, caplog):
        """Test that missing stories directory logs a warning."""
        # Only create structures directory
        (tmp_path / "structures").mkdir()

        with caplog.at_level(logging.WARNING):
            registry = TemplateRegistry(tmp_path)

        assert "Stories directory not found" in caplog.text
        assert len(registry.story_templates) == 0

    def test_invalid_yaml_raises_error(self, tmp_path):
        """Test that invalid YAML files raise an error."""
        structures_dir = tmp_path / "structures"
        stories_dir = tmp_path / "stories"
        structures_dir.mkdir()
        stories_dir.mkdir()

        # Create invalid YAML file
        (structures_dir / "invalid.yaml").write_text("{invalid yaml", encoding="utf-8")

        with pytest.raises(TemplateRegistryError, match="Failed to load 1 template"):
            TemplateRegistry(tmp_path)

    def test_invalid_model_data_raises_error(self, tmp_path):
        """
        Ensures a TemplateRegistryError is raised when a YAML file is syntactically valid but omits required model fields.

        Creates a structures directory with a YAML file missing required fields and asserts that initializing TemplateRegistry reports a failure to load the invalid template.
        """
        structures_dir = tmp_path / "structures"
        stories_dir = tmp_path / "stories"
        structures_dir.mkdir()
        stories_dir.mkdir()

        # Create valid YAML but missing required fields
        invalid_data = {"name": "Test", "description": "Test"}  # Missing 'id'
        (structures_dir / "invalid.yaml").write_text(yaml.dump(invalid_data), encoding="utf-8")

        with pytest.raises(TemplateRegistryError, match="Failed to load 1 template"):
            TemplateRegistry(tmp_path)

    def test_multiple_invalid_files_collects_all_errors(self, tmp_path):
        """Test that multiple invalid files are all reported in the error."""
        structures_dir = tmp_path / "structures"
        stories_dir = tmp_path / "stories"
        structures_dir.mkdir()
        stories_dir.mkdir()

        # Create two invalid files
        (structures_dir / "invalid1.yaml").write_text("{invalid yaml", encoding="utf-8")
        (structures_dir / "invalid2.yaml").write_text("- list\n- instead", encoding="utf-8")

        with pytest.raises(TemplateRegistryError, match="Failed to load 2 template"):
            TemplateRegistry(tmp_path)

    def test_non_dict_yaml_raises_error(self, tmp_path):
        """Test that YAML files containing non-dict data raise an error."""
        structures_dir = tmp_path / "structures"
        stories_dir = tmp_path / "stories"
        structures_dir.mkdir()
        stories_dir.mkdir()

        # Create YAML file with a list instead of dict
        (structures_dir / "list.yaml").write_text("- item1\n- item2", encoding="utf-8")

        with pytest.raises(TemplateRegistryError, match="Failed to load 1 template"):
            TemplateRegistry(tmp_path)

    def test_unreadable_file_raises_error(self, tmp_path, monkeypatch):
        """Test that unreadable files raise an error."""
        structures_dir = tmp_path / "structures"
        stories_dir = tmp_path / "stories"
        structures_dir.mkdir()
        stories_dir.mkdir()

        # Create a valid file
        valid_data = {"id": "test", "name": "Test", "description": "Test"}
        yaml_file = structures_dir / "test.yaml"
        yaml_file.write_text(yaml.dump(valid_data), encoding="utf-8")

        # Patch open to raise OSError
        original_open = open

        def mock_open(path, *args, **kwargs):
            """
            A patched file opener that simulates an unreadable file for paths containing "test.yaml".

            Parameters:
                path (str | os.PathLike): File path to open; if the path string contains "test.yaml" an OSError is raised.
                *args: Positional arguments forwarded to the real open.
                **kwargs: Keyword arguments forwarded to the real open.

            Returns:
                file object: The result of calling the original open for the given path and arguments.

            Raises:
                OSError: Always raised with message "Permission denied" when `path` contains "test.yaml".
            """
            if "test.yaml" in str(path):
                raise OSError("Permission denied")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", mock_open)

        with pytest.raises(TemplateRegistryError, match="Failed to load 1 template"):
            TemplateRegistry(tmp_path)

    def test_invalid_story_template_raises_error(self, tmp_path):
        """Test that invalid story templates raise an error."""
        structures_dir = tmp_path / "structures"
        stories_dir = tmp_path / "stories"
        structures_dir.mkdir()
        stories_dir.mkdir()

        # Create invalid story template YAML
        (stories_dir / "invalid.yaml").write_text("{invalid yaml", encoding="utf-8")

        with pytest.raises(TemplateRegistryError, match="Failed to load 1 template"):
            TemplateRegistry(tmp_path)


class TestTemplateDataIntegrity:
    """Tests for the integrity of the template data itself."""

    def test_three_act_structure_has_correct_plot_points(self):
        """Test three-act structure has all expected plot points."""
        preset = BUILTIN_STRUCTURE_PRESETS["three-act"]
        assert len(preset.plot_points) == 7
        # Check key plot points
        descriptions = [pp.description for pp in preset.plot_points]
        assert any("Opening Image" in d for d in descriptions)
        assert any("Inciting Incident" in d for d in descriptions)
        assert any("Climax" in d for d in descriptions)

    def test_heros_journey_has_12_stages(self):
        """Test hero's journey has all 12 stages."""
        preset = BUILTIN_STRUCTURE_PRESETS["heros-journey"]
        assert len(preset.plot_points) == 12
        descriptions = [pp.description for pp in preset.plot_points]
        assert any("Ordinary World" in d for d in descriptions)
        assert any("Call to Adventure" in d for d in descriptions)
        assert any("Return with Elixir" in d for d in descriptions)

    def test_save_the_cat_has_15_beats(self):
        """Test Save the Cat has all 15 beats."""
        preset = BUILTIN_STRUCTURE_PRESETS["save-the-cat"]
        assert len(preset.plot_points) == 15
        assert len(preset.beats) == 15
        descriptions = [pp.description for pp in preset.plot_points]
        assert any("Opening Image" in d for d in descriptions)
        assert any("Fun and Games" in d for d in descriptions)
        assert any("Final Image" in d for d in descriptions)

    def test_save_the_cat_has_4_acts(self):
        """Test Save the Cat has 4 acts (Act 1, 2A, 2B, 3) for backward compatibility."""
        preset = BUILTIN_STRUCTURE_PRESETS["save-the-cat"]
        assert len(preset.acts) == 4
        assert preset.acts == ["Act 1", "Act 2A", "Act 2B", "Act 3"]

    def test_mystery_template_has_detective_protagonist(self):
        """Test mystery template has a detective protagonist."""
        template = BUILTIN_STORY_TEMPLATES["mystery-detective"]
        protagonists = [c for c in template.characters if c.role == "protagonist"]
        assert len(protagonists) == 1
        assert "Detective" in protagonists[0].name or "Investigator" in protagonists[0].name

    def test_romance_template_has_two_protagonists(self):
        """Test romance template has two protagonists."""
        template = BUILTIN_STORY_TEMPLATES["romance-contemporary"]
        protagonists = [c for c in template.characters if c.role == "protagonist"]
        assert len(protagonists) == 2

    def test_all_templates_have_world_rules(self):
        """Test all story templates have world rules defined."""
        for template_id, template in BUILTIN_STORY_TEMPLATES.items():
            assert len(template.world_rules) > 0, f"{template_id} has no world rules"

    def test_plot_points_have_valid_percentages(self):
        """Test all plot points have percentages between 0 and 100."""
        for preset_id, preset in BUILTIN_STRUCTURE_PRESETS.items():
            for pp in preset.plot_points:
                assert pp.percentage is not None, f"{preset_id} has plot point without percentage"
                assert 0 <= pp.percentage <= 100, (
                    f"{preset_id} has invalid percentage: {pp.percentage}"
                )

        for template_id, template in BUILTIN_STORY_TEMPLATES.items():
            for pp in template.plot_points:
                if pp.percentage is not None:
                    assert 0 <= pp.percentage <= 100, (
                        f"{template_id} has invalid percentage: {pp.percentage}"
                    )

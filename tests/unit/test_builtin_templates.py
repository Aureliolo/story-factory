"""Tests for the built-in templates registry."""

import logging

import yaml

from src.memory.builtin_templates import (
    BUILTIN_STORY_TEMPLATES,
    BUILTIN_STRUCTURE_PRESETS,
    TemplateRegistry,
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
        """Test that built-in story templates are loaded."""
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
        """Test getting a non-existent story template returns None."""
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

    def test_invalid_yaml_logs_error(self, tmp_path, caplog):
        """Test that invalid YAML files log an error."""
        structures_dir = tmp_path / "structures"
        stories_dir = tmp_path / "stories"
        structures_dir.mkdir()
        stories_dir.mkdir()

        # Create invalid YAML file
        (structures_dir / "invalid.yaml").write_text("{invalid yaml", encoding="utf-8")

        with caplog.at_level(logging.ERROR):
            registry = TemplateRegistry(tmp_path)

        assert "Failed to load structure preset" in caplog.text
        assert len(registry.structure_presets) == 0

    def test_invalid_model_data_logs_error(self, tmp_path, caplog):
        """Test that valid YAML with invalid model data logs an error."""
        structures_dir = tmp_path / "structures"
        stories_dir = tmp_path / "stories"
        structures_dir.mkdir()
        stories_dir.mkdir()

        # Create valid YAML but missing required fields
        invalid_data = {"name": "Test", "description": "Test"}  # Missing 'id'
        (structures_dir / "invalid.yaml").write_text(yaml.dump(invalid_data), encoding="utf-8")

        with caplog.at_level(logging.ERROR):
            registry = TemplateRegistry(tmp_path)

        assert "Failed to load structure preset" in caplog.text
        assert len(registry.structure_presets) == 0

    def test_loads_valid_files_despite_invalid_ones(self, tmp_path, caplog):
        """Test that valid files are loaded even when some are invalid."""
        structures_dir = tmp_path / "structures"
        stories_dir = tmp_path / "stories"
        structures_dir.mkdir()
        stories_dir.mkdir()

        # Create one invalid and one valid file
        (structures_dir / "invalid.yaml").write_text("{invalid yaml", encoding="utf-8")
        valid_data = {
            "id": "valid-preset",
            "name": "Valid Preset",
            "description": "A valid preset",
            "acts": ["Act 1"],
            "plot_points": [],
            "beats": [],
        }
        (structures_dir / "valid.yaml").write_text(yaml.dump(valid_data), encoding="utf-8")

        with caplog.at_level(logging.ERROR):
            registry = TemplateRegistry(tmp_path)

        # Valid file should still load
        assert len(registry.structure_presets) == 1
        assert "valid-preset" in registry.structure_presets


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

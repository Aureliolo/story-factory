"""Registry for loading built-in templates from YAML files."""

import logging
from pathlib import Path
from typing import Any

import yaml

from src.memory.templates import (
    CharacterTemplate,
    PlotPointTemplate,
    StoryTemplate,
    StructurePreset,
)
from src.utils.exceptions import StoryFactoryError

logger = logging.getLogger(__name__)

# Templates directory (same directory as this file)
_TEMPLATES_DIR = Path(__file__).parent


class TemplateRegistryError(StoryFactoryError):
    """Raised when template loading fails."""


class TemplateRegistry:
    """Registry that loads built-in templates from YAML files.

    The registry provides:
    - Automatic discovery and loading of YAML templates
    - Template lookup by ID
    - Pydantic validation on load

    Templates are organized in directories by type:
    ```
    src/memory/builtin_templates/
    ├── structures/
    │   ├── three-act.yaml
    │   ├── heros-journey.yaml
    │   └── save-the-cat.yaml
    └── stories/
        ├── mystery-detective.yaml
        ├── romance-contemporary.yaml
        └── ...
    ```
    """

    def __init__(self, templates_dir: Path | str | None = None):
        """Initialize registry and load all templates.

        Args:
            templates_dir: Directory containing template YAML files.
                          Defaults to src/memory/builtin_templates.
        """
        self.templates_dir = Path(templates_dir) if templates_dir else _TEMPLATES_DIR
        self._structure_presets: dict[str, StructurePreset] = {}
        self._story_templates: dict[str, StoryTemplate] = {}
        self._load_all_templates()

    def _load_yaml_file(self, filepath: Path) -> dict[str, Any]:
        """Load and parse a YAML file.

        Args:
            filepath: Path to the YAML file.

        Returns:
            Parsed YAML data as dictionary.

        Raises:
            TemplateRegistryError: If loading or parsing fails.
        """
        try:
            with open(filepath, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if not isinstance(data, dict):
                    raise TemplateRegistryError(
                        f"Expected dict in {filepath}, got {type(data).__name__}"
                    )
                return data
        except yaml.YAMLError as e:
            raise TemplateRegistryError(f"Invalid YAML in {filepath}: {e}") from e
        except OSError as e:
            raise TemplateRegistryError(f"Failed to read {filepath}: {e}") from e

    def _convert_plot_points(self, data: dict[str, Any]) -> None:
        """Convert plot_points from dicts to PlotPointTemplate in place.

        Args:
            data: Template data dict that may contain plot_points.
        """
        if "plot_points" in data:
            data["plot_points"] = [
                PlotPointTemplate(**pp) if isinstance(pp, dict) else pp
                for pp in data["plot_points"]
            ]

    def _load_structure_preset(self, filepath: Path) -> None:
        """Load a structure preset from YAML.

        Args:
            filepath: Path to the structure preset YAML file.
        """
        data = self._load_yaml_file(filepath)
        self._convert_plot_points(data)

        preset = StructurePreset.model_validate(data)
        self._structure_presets[preset.id] = preset
        logger.debug("Loaded structure preset: %s", preset.id)

    def _load_story_template(self, filepath: Path) -> None:
        """Load a story template from YAML.

        Args:
            filepath: Path to the story template YAML file.
        """
        data = self._load_yaml_file(filepath)

        # Convert characters from dicts to CharacterTemplate
        if "characters" in data:
            data["characters"] = [
                CharacterTemplate(**char) if isinstance(char, dict) else char
                for char in data["characters"]
            ]

        self._convert_plot_points(data)

        template = StoryTemplate.model_validate(data)
        self._story_templates[template.id] = template
        logger.debug("Loaded story template: %s", template.id)

    def _load_all_templates(self) -> None:
        """Load all templates from the templates directory."""
        structures_dir = self.templates_dir / "structures"
        stories_dir = self.templates_dir / "stories"

        # Load structure presets
        if structures_dir.exists():
            yaml_files = sorted(structures_dir.glob("*.yaml"))
            logger.info("Loading %d structure presets from %s", len(yaml_files), structures_dir)

            for yaml_file in yaml_files:
                try:
                    self._load_structure_preset(yaml_file)
                except (TemplateRegistryError, ValueError) as e:
                    logger.error("Failed to load structure preset %s: %s", yaml_file, e)
        else:
            logger.warning("Structures directory not found: %s", structures_dir)

        # Load story templates
        if stories_dir.exists():
            yaml_files = sorted(stories_dir.glob("*.yaml"))
            logger.info("Loading %d story templates from %s", len(yaml_files), stories_dir)

            for yaml_file in yaml_files:
                try:
                    self._load_story_template(yaml_file)
                except (TemplateRegistryError, ValueError) as e:
                    logger.error("Failed to load story template %s: %s", yaml_file, e)
        else:
            logger.warning("Stories directory not found: %s", stories_dir)

        logger.info(
            "Loaded %d structure presets and %d story templates",
            len(self._structure_presets),
            len(self._story_templates),
        )

    @property
    def structure_presets(self) -> dict[str, StructurePreset]:
        """Get all loaded structure presets."""
        return self._structure_presets

    @property
    def story_templates(self) -> dict[str, StoryTemplate]:
        """Get all loaded story templates."""
        return self._story_templates

    def get_structure_preset(self, preset_id: str) -> StructurePreset | None:
        """Get a structure preset by ID.

        Args:
            preset_id: The preset ID.

        Returns:
            StructurePreset if found, None otherwise.
        """
        preset = self._structure_presets.get(preset_id)
        if preset is None:
            logger.debug("Structure preset not found: %s", preset_id)
        return preset

    def get_story_template(self, template_id: str) -> StoryTemplate | None:
        """Get a story template by ID.

        Args:
            template_id: The template ID.

        Returns:
            StoryTemplate if found, None otherwise.
        """
        template = self._story_templates.get(template_id)
        if template is None:
            logger.debug("Story template not found: %s", template_id)
        return template

    def reload(self) -> None:
        """Reload all templates from disk."""
        logger.info("Reloading all templates")
        self._structure_presets.clear()
        self._story_templates.clear()
        self._load_all_templates()

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"TemplateRegistry({len(self._structure_presets)} structures, "
            f"{len(self._story_templates)} templates from {self.templates_dir})"
        )


# Module-level singleton instance - loaded once at import time
_registry: TemplateRegistry | None = None


def _get_registry() -> TemplateRegistry:
    """Get or create the singleton registry instance."""
    global _registry
    if _registry is None:
        _registry = TemplateRegistry()
    return _registry


def get_builtin_structure_presets() -> dict[str, StructurePreset]:
    """Get all built-in structure presets."""
    return _get_registry().structure_presets


def get_builtin_story_templates() -> dict[str, StoryTemplate]:
    """Get all built-in story templates."""
    return _get_registry().story_templates

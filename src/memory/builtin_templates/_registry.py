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
        """
        Create a TemplateRegistry and load built-in templates from the given directory.

        Parameters:
            templates_dir (Path | str | None): Directory containing template YAML files. If omitted, uses the package's built-in templates directory.

        Raises:
            TemplateRegistryError: If the resolved templates directory does not exist.
        """
        self.templates_dir = Path(templates_dir) if templates_dir else _TEMPLATES_DIR
        if not self.templates_dir.exists():
            raise TemplateRegistryError(f"Templates directory does not exist: {self.templates_dir}")
        logger.info("Initializing TemplateRegistry from %s", self.templates_dir)
        self._structure_presets: dict[str, StructurePreset] = {}
        self._story_templates: dict[str, StoryTemplate] = {}
        self._load_all_templates()

    def _load_yaml_file(self, filepath: Path) -> dict[str, Any]:
        """
        Load and parse a YAML file into a dictionary.

        Parameters:
            filepath (Path): Path to the YAML file to read.

        Returns:
            dict[str, Any]: Parsed YAML content as a dictionary.

        Raises:
            TemplateRegistryError: If the file cannot be read, the YAML is invalid, or the parsed content is not a mapping.
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
        """
        Convert any dict entries in the `plot_points` list into `PlotPointTemplate` instances.

        If `data` contains a "plot_points" key, each entry that is a dict will be replaced by `PlotPointTemplate(**entry)`; entries that are already objects are left unchanged. The transformation modifies `data` in place.

        Parameters:
            data (dict[str, Any]): Template data that may include a "plot_points" list.
        """
        if "plot_points" in data:
            data["plot_points"] = [
                PlotPointTemplate(**pp) if isinstance(pp, dict) else pp
                for pp in data["plot_points"]
            ]

    def _load_structure_preset(self, filepath: Path) -> None:
        """
        Load and validate a structure preset from a YAML file and register it in the registry under its `id`.

        Parameters:
            filepath (Path): Path to the structure preset YAML file to load.

        Raises:
            TemplateRegistryError: If the YAML cannot be read/parsed or if validation of the preset fails.
        """
        data = self._load_yaml_file(filepath)
        self._convert_plot_points(data)

        preset = StructurePreset.model_validate(data)
        self._structure_presets[preset.id] = preset
        logger.debug("Loaded structure preset: %s", preset.id)

    def _load_story_template(self, filepath: Path) -> None:
        """
        Load and validate a story template from a YAML file and store it in the registry's template cache.

        Parameters:
            filepath (Path): Path to the YAML file containing the story template. The validated template is stored in the registry keyed by its `id`.
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
        """
        Load and validate all built-in templates from the registry's templates directory and populate internal caches.

        Searches for YAML files under "<templates_dir>/structures" and "<templates_dir>/stories", validates each file as a StructurePreset or StoryTemplate respectively, and stores successful results in the registry's in-memory caches. If a subdirectory is missing, a warning is logged. If any file fails to load or validate, a TemplateRegistryError is raised listing the failing files.

        Raises:
            TemplateRegistryError: If one or more templates fail to load or validate.
        """
        structures_dir = self.templates_dir / "structures"
        stories_dir = self.templates_dir / "stories"
        errors: list[str] = []

        # Load structure presets
        if structures_dir.exists():
            yaml_files = sorted(structures_dir.glob("*.yaml"))
            logger.info("Loading %d structure presets from %s", len(yaml_files), structures_dir)

            for yaml_file in yaml_files:
                try:
                    self._load_structure_preset(yaml_file)
                except (TemplateRegistryError, ValueError) as e:
                    errors.append(f"structure preset {yaml_file.name}: {e}")
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
                    errors.append(f"story template {yaml_file.name}: {e}")
        else:
            logger.warning("Stories directory not found: %s", stories_dir)

        # Fail fast if any templates failed to load
        if errors:
            error_msg = f"Failed to load {len(errors)} template(s):\n" + "\n".join(
                f"  - {e}" for e in errors
            )
            logger.error(error_msg)
            raise TemplateRegistryError(error_msg)

        logger.info(
            "Loaded %d structure presets and %d story templates",
            len(self._structure_presets),
            len(self._story_templates),
        )

    @property
    def structure_presets(self) -> dict[str, StructurePreset]:
        """
        Provide access to built-in structure presets keyed by preset ID.

        Returns:
            dict[str, StructurePreset]: Mapping from preset ID to its StructurePreset instance.
        """
        return self._structure_presets

    @property
    def story_templates(self) -> dict[str, StoryTemplate]:
        """
        Return the mapping of all loaded story templates by ID.

        Returns:
            dict[str, StoryTemplate]: Mapping from template ID to its validated StoryTemplate.
        """
        return self._story_templates

    def get_structure_preset(self, preset_id: str) -> StructurePreset | None:
        """
        Retrieve a structure preset by its identifier.

        Parameters:
            preset_id (str): The structure preset identifier to look up.

        Returns:
            StructurePreset | None: The matching StructurePreset, or None if no preset exists with that ID.
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
        """
        Reload the registry by clearing cached templates and repopulating them from disk.

        This clears the in-memory structure preset and story template caches and then loads all templates found in the configured templates directory.
        """
        logger.info("Reloading all templates")
        self._structure_presets.clear()
        self._story_templates.clear()
        self._load_all_templates()

    def __repr__(self) -> str:
        """
        Represent the registry with counts of loaded structure presets and story templates and the templates directory.

        Returns:
            A string containing the number of loaded structure presets, the number of loaded story templates, and the templates directory path.
        """
        return (
            f"TemplateRegistry({len(self._structure_presets)} structures, "
            f"{len(self._story_templates)} templates from {self.templates_dir})"
        )


# Module-level singleton instance - initialized eagerly at import time for thread safety
_registry: TemplateRegistry = TemplateRegistry()


def _get_registry() -> TemplateRegistry:
    """
    Return the module-level singleton registry instance.

    Returns:
        The singleton TemplateRegistry instance.
    """
    return _registry


def get_builtin_structure_presets() -> dict[str, StructurePreset]:
    """
    Retrieve the mapping of built-in structure presets keyed by preset ID.

    Returns:
        dict[str, StructurePreset]: Mapping from preset ID to its StructurePreset instance.
    """
    return _get_registry().structure_presets


def get_builtin_story_templates() -> dict[str, StoryTemplate]:
    """
    Get built-in story templates indexed by their IDs.

    Returns:
        dict[str, StoryTemplate]: Mapping from template ID to the corresponding StoryTemplate instance.
    """
    return _get_registry().story_templates

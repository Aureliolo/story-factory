"""Template service - handles template CRUD operations and application."""

import json
import logging
from pathlib import Path

from src.memory.builtin_templates import (
    BUILTIN_STORY_TEMPLATES,
    BUILTIN_STRUCTURE_PRESETS,
)  # Now loaded from YAML files
from src.memory.story_state import Character, PlotPoint, StoryBrief, StoryState
from src.memory.templates import StoryTemplate, StructurePreset
from src.memory.world_database import WorldDatabase
from src.settings import Settings

logger = logging.getLogger(__name__)

# Templates directory (go up from services/ to src/ to project root, then into output/)
TEMPLATES_DIR = Path(__file__).parent.parent.parent / "output" / "templates"


class TemplateService:
    """Service for managing story templates and structure presets.

    This service handles:
    - Loading built-in templates
    - Creating custom templates from existing projects
    - Applying templates to new projects
    - Importing/exporting templates
    - Listing available templates
    """

    def __init__(self, settings: Settings):
        """Initialize template service.

        Args:
            settings: Application settings.
        """
        self.settings = settings
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure templates directory exists."""
        TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

    def list_templates(self) -> list[StoryTemplate]:
        """List all available templates (built-in + custom).

        Returns:
            List of StoryTemplate objects.
        """
        logger.debug("Listing all templates")
        templates: list[StoryTemplate] = []

        # Add built-in templates
        templates.extend(BUILTIN_STORY_TEMPLATES.values())

        # Add custom templates
        if TEMPLATES_DIR.exists():
            for filepath in TEMPLATES_DIR.glob("*.json"):
                try:
                    with open(filepath, encoding="utf-8") as f:
                        data = json.load(f)
                    template = StoryTemplate.model_validate(data)
                    templates.append(template)
                except Exception as e:
                    logger.warning(f"Failed to load template {filepath}: {e}")

        # Sort: built-in first, then by name
        templates.sort(key=lambda t: (not t.is_builtin, t.name))
        logger.debug(
            "Found %d templates (%d built-in)", len(templates), len(BUILTIN_STORY_TEMPLATES)
        )
        return templates

    def get_template(self, template_id: str) -> StoryTemplate | None:
        """Get a specific template by ID.

        Args:
            template_id: The template ID.

        Returns:
            StoryTemplate if found, None otherwise.
        """
        logger.debug("Getting template: %s", template_id)
        # Check built-in templates first
        if template_id in BUILTIN_STORY_TEMPLATES:
            return BUILTIN_STORY_TEMPLATES[template_id]

        # Check custom templates
        template_path = TEMPLATES_DIR / f"{template_id}.json"
        if template_path.exists():
            try:
                with open(template_path, encoding="utf-8") as f:
                    data = json.load(f)
                return StoryTemplate.model_validate(data)
            except Exception as e:
                logger.error(f"Failed to load template {template_id}: {e}")
                return None

        return None

    def save_template(self, template: StoryTemplate) -> Path:
        """Save a custom template to disk.

        Args:
            template: The template to save.

        Returns:
            Path where template was saved.

        Raises:
            ValueError: If trying to save over a built-in template.
        """
        if template.is_builtin:
            raise ValueError("Cannot modify built-in templates")

        output_path = TEMPLATES_DIR / f"{template.id}.json"
        template_data = template.model_dump(mode="json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(template_data, f, indent=2, default=str)

        logger.info(f"Saved template: {template.id}")
        return output_path

    def delete_template(self, template_id: str) -> bool:
        """Delete a custom template.

        Args:
            template_id: The template ID to delete.

        Returns:
            True if deleted, False if not found.

        Raises:
            ValueError: If trying to delete a built-in template.
        """
        if template_id in BUILTIN_STORY_TEMPLATES:
            raise ValueError("Cannot delete built-in templates")

        template_path = TEMPLATES_DIR / f"{template_id}.json"
        if template_path.exists():
            template_path.unlink()
            logger.info(f"Deleted template: {template_id}")
            return True

        return False

    def create_template_from_project(
        self, state: StoryState, template_name: str, template_description: str
    ) -> StoryTemplate:
        """Create a custom template from an existing project.

        Args:
            state: The story state to create template from.
            template_name: Name for the template.
            template_description: Description of the template.

        Returns:
            The created StoryTemplate.
        """
        import uuid
        from datetime import datetime

        from src.memory.templates import CharacterTemplate, PlotPointTemplate

        template_id = f"custom-{uuid.uuid4()}"

        # Convert characters to character templates
        char_templates = [
            CharacterTemplate(
                name=char.name,
                role=char.role,
                description=char.description,
                personality_traits=char.personality_traits.copy(),
                goals=char.goals.copy(),
                arc_notes=char.arc_notes,
            )
            for char in state.characters
        ]

        # Convert plot points to plot point templates
        plot_templates = [
            PlotPointTemplate(
                description=pp.description,
                act=None,  # Will be set if structure preset used
                percentage=None,
            )
            for pp in state.plot_points
        ]

        # Create template from state
        template = StoryTemplate(
            id=template_id,
            name=template_name,
            description=template_description,
            is_builtin=False,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            genre=state.brief.genre if state.brief else "General",
            subgenres=state.brief.subgenres.copy() if state.brief else [],
            tone=state.brief.tone if state.brief else "",
            themes=state.brief.themes.copy() if state.brief else [],
            setting_time=state.brief.setting_time if state.brief else "",
            setting_place=state.brief.setting_place if state.brief else "",
            target_length=state.brief.target_length if state.brief else "novel",
            structure_preset_id=None,
            world_description=state.world_description,
            world_rules=state.world_rules.copy(),
            characters=char_templates,
            plot_points=plot_templates,
            author="User",
            tags=[],
        )

        # Save the template
        self.save_template(template)
        logger.info(f"Created template from project: {template_id}")
        return template

    def apply_template_to_state(
        self, template: StoryTemplate, state: StoryState, world_db: WorldDatabase
    ) -> None:
        """Apply a template to a story state.

        Args:
            template: The template to apply.
            state: The story state to populate.
            world_db: The world database to populate.
        """
        logger.info(f"Applying template {template.id} to project {state.id}")

        # Apply story brief
        state.brief = StoryBrief(
            premise="",  # User will fill this in during interview
            genre=template.genre,
            subgenres=template.subgenres.copy(),
            tone=template.tone,
            themes=template.themes.copy(),
            setting_time=template.setting_time,
            setting_place=template.setting_place,
            target_length=template.target_length,
            language="English",
            content_rating="none",
        )

        # Apply world building
        state.world_description = template.world_description
        state.world_rules = template.world_rules.copy()

        # Apply characters
        state.characters = [
            Character(
                name=ct.name,
                role=ct.role,
                description=ct.description,
                personality_traits=ct.personality_traits.copy(),
                goals=ct.goals.copy(),
                arc_notes=ct.arc_notes,
            )
            for ct in template.characters
        ]

        # Apply plot points
        state.plot_points = [
            PlotPoint(
                description=pt.description,
                chapter=None,
                completed=False,
            )
            for pt in template.plot_points
        ]

        # If template references a structure preset, apply it
        if template.structure_preset_id:
            preset = self.get_structure_preset(template.structure_preset_id)
            if preset:
                self._apply_structure_preset(preset, state)

        # Log template application summary
        logger.debug(
            f"Template applied with {len(state.characters)} characters and {len(state.plot_points)} plot points"
        )
        logger.info("Successfully applied template to project")

    def _apply_structure_preset(self, preset: StructurePreset, state: StoryState) -> None:
        """Apply a structure preset to enhance plot points.

        Args:
            preset: The structure preset to apply.
            state: The story state to modify.
        """
        logger.debug(f"Applying structure preset: {preset.name}")

        # Add structure-specific plot points if not already present
        existing_descriptions = {pp.description for pp in state.plot_points}

        for template_point in preset.plot_points:
            if template_point.description not in existing_descriptions:
                state.plot_points.append(
                    PlotPoint(
                        description=template_point.description,
                        chapter=None,
                        completed=False,
                    )
                )

        # Add structure notes to world rules if not present
        structure_note = f"Story follows {preset.name} structure"
        if structure_note not in state.world_rules:
            state.world_rules.append(structure_note)

    def get_structure_preset(self, preset_id: str) -> StructurePreset | None:
        """Get a structure preset by ID.

        Args:
            preset_id: The preset ID.

        Returns:
            StructurePreset if found, None otherwise.
        """
        logger.debug("Getting structure preset: %s", preset_id)
        return BUILTIN_STRUCTURE_PRESETS.get(preset_id)

    def list_structure_presets(self) -> list[StructurePreset]:
        """List all available structure presets.

        Returns:
            List of StructurePreset objects.
        """
        logger.debug("Listing %d structure presets", len(BUILTIN_STRUCTURE_PRESETS))
        return list(BUILTIN_STRUCTURE_PRESETS.values())

    def export_template(self, template_id: str, export_path: Path) -> Path:
        """Export a template to a file for sharing.

        Args:
            template_id: The template ID to export.
            export_path: Path to export to (must be .json).

        Returns:
            Path where template was exported.

        Raises:
            FileNotFoundError: If template not found.
            ValueError: If export_path is not .json.
        """
        if export_path.suffix != ".json":
            raise ValueError("Export path must be a .json file")

        template = self.get_template(template_id)
        if not template:
            raise FileNotFoundError(f"Template not found: {template_id}")

        template_data = template.model_dump(mode="json")
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(template_data, f, indent=2, default=str)

        logger.info(f"Exported template {template_id} to {export_path}")
        return export_path

    def import_template(self, import_path: Path) -> StoryTemplate:
        """Import a template from a file.

        Args:
            import_path: Path to the template file.

        Returns:
            The imported StoryTemplate.

        Raises:
            FileNotFoundError: If import file doesn't exist.
            ValueError: If file is invalid.
        """
        if not import_path.exists():
            raise FileNotFoundError(f"Import file not found: {import_path}")

        try:
            with open(import_path, encoding="utf-8") as f:
                data = json.load(f)

            template = StoryTemplate.model_validate(data)

            # Mark as custom (not built-in)
            template.is_builtin = False

            # Ensure unique ID
            import uuid

            if template.id in BUILTIN_STORY_TEMPLATES:
                template.id = f"imported-{uuid.uuid4()}"

            # Save to custom templates
            self.save_template(template)

            logger.info(f"Imported template: {template.id}")
            return template

        except Exception as e:
            logger.error(f"Failed to import template from {import_path}: {e}")
            raise ValueError(f"Invalid template file: {e}") from e

"""World template service for managing and applying world templates."""

import json
import logging
from pathlib import Path

from src.memory.builtin_world_templates import BUILTIN_WORLD_TEMPLATES, get_world_template
from src.memory.templates import WorldTemplate
from src.settings import Settings

logger = logging.getLogger(__name__)

# Path for user-created templates
USER_TEMPLATES_FILE = Path(__file__).parent.parent / "data" / "user_world_templates.json"


class WorldTemplateService:
    """Service for managing world templates.

    Provides CRUD operations for world templates and formatting for prompt injection.
    """

    def __init__(self, settings: Settings | None = None):
        """Initialize WorldTemplateService.

        Args:
            settings: Application settings. If None, loads from src/settings.json.
        """
        logger.debug("Initializing WorldTemplateService")
        self.settings = settings or Settings.load()
        self._user_templates: dict[str, WorldTemplate] = {}
        self._load_user_templates()
        logger.debug("WorldTemplateService initialized successfully")

    def _load_user_templates(self) -> None:
        """Load user-created templates from disk."""
        if USER_TEMPLATES_FILE.exists():
            try:
                with open(USER_TEMPLATES_FILE) as f:
                    data = json.load(f)
                for template_data in data:
                    template = WorldTemplate(**template_data)
                    self._user_templates[template.id] = template
                logger.info(f"Loaded {len(self._user_templates)} user world templates")
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to load user world templates: {e}")

    def _save_user_templates(self) -> None:
        """Save user-created templates to disk."""
        USER_TEMPLATES_FILE.parent.mkdir(parents=True, exist_ok=True)
        templates_data = [t.model_dump() for t in self._user_templates.values()]
        with open(USER_TEMPLATES_FILE, "w") as f:
            json.dump(templates_data, f, indent=2, default=str)
        logger.debug(f"Saved {len(self._user_templates)} user world templates")

    def list_templates(self, include_builtin: bool = True) -> list[WorldTemplate]:
        """List all available world templates.

        Args:
            include_builtin: Whether to include built-in templates.

        Returns:
            List of world templates.
        """
        templates: list[WorldTemplate] = []

        if include_builtin:
            templates.extend(BUILTIN_WORLD_TEMPLATES.values())

        templates.extend(self._user_templates.values())

        logger.debug(f"Listed {len(templates)} world templates")
        return templates

    def get_template(self, template_id: str) -> WorldTemplate | None:
        """Get a world template by ID.

        Args:
            template_id: The template ID to look up.

        Returns:
            The world template if found, None otherwise.
        """
        # Check user templates first (allows override)
        if template_id in self._user_templates:
            logger.debug(f"Retrieved user world template: {template_id}")
            return self._user_templates[template_id]

        # Fall back to built-in
        template = get_world_template(template_id)
        if template:
            logger.debug(f"Retrieved builtin world template: {template_id}")
        return template

    def save_template(self, template: WorldTemplate) -> None:
        """Save a user-created world template.

        Args:
            template: The template to save.

        Raises:
            ValueError: If trying to overwrite a built-in template.
        """
        if template.id in BUILTIN_WORLD_TEMPLATES:
            raise ValueError(f"Cannot overwrite built-in template: {template.id}")

        template.is_builtin = False
        self._user_templates[template.id] = template
        self._save_user_templates()
        logger.info(f"Saved user world template: {template.id}")

    def delete_template(self, template_id: str) -> bool:
        """Delete a user-created world template.

        Args:
            template_id: The template ID to delete.

        Returns:
            True if deleted, False if not found or built-in.
        """
        if template_id in BUILTIN_WORLD_TEMPLATES:
            logger.warning(f"Cannot delete built-in template: {template_id}")
            return False

        if template_id in self._user_templates:
            del self._user_templates[template_id]
            self._save_user_templates()
            logger.info(f"Deleted user world template: {template_id}")
            return True

        logger.warning(f"Template not found: {template_id}")
        return False

    def format_hints_for_prompt(self, template: WorldTemplate) -> str:
        """Format a world template into prompt guidance text.

        This generates custom instructions for entity generation based on the template.

        Args:
            template: The world template to format.

        Returns:
            Formatted string for prompt injection.
        """
        lines = [
            f"WORLD TEMPLATE: {template.name}",
            f"Genre: {template.genre}",
            f"Atmosphere: {template.atmosphere}",
            "",
        ]

        hints = template.entity_hints

        if hints.character_roles:
            roles = ", ".join(hints.character_roles)
            lines.append(f"Suggested Character Roles: {roles}")

        if hints.location_types:
            locs = ", ".join(hints.location_types)
            lines.append(f"Suggested Location Types: {locs}")

        if hints.faction_types:
            facs = ", ".join(hints.faction_types)
            lines.append(f"Suggested Faction Types: {facs}")

        if hints.item_types:
            items = ", ".join(hints.item_types)
            lines.append(f"Suggested Item Types: {items}")

        if hints.concept_types:
            concepts = ", ".join(hints.concept_types)
            lines.append(f"Suggested Concepts: {concepts}")

        if template.relationship_patterns:
            rels = ", ".join(template.relationship_patterns)
            lines.append(f"\nCommon Relationship Types: {rels}")

        if template.naming_style:
            lines.append(f"\nNaming Style: {template.naming_style}")

        if template.recommended_counts:
            lines.append("\nRecommended Entity Counts:")
            for entity_type, (min_count, max_count) in template.recommended_counts.items():
                lines.append(f"  - {entity_type}: {min_count}-{max_count}")

        guidance = "\n".join(lines)
        logger.debug(f"Formatted world template guidance for {template.id} ({len(guidance)} chars)")
        return guidance

    def get_template_for_genre(self, genre: str) -> WorldTemplate | None:
        """Find a template that matches a genre.

        Args:
            genre: The genre to match (e.g., "fantasy", "science_fiction").

        Returns:
            A matching template or None.
        """
        genre_lower = genre.lower()

        # Check built-in templates
        for template in BUILTIN_WORLD_TEMPLATES.values():
            if template.genre.lower() == genre_lower:
                logger.debug(f"Found template {template.id} for genre {genre}")
                return template
            # Check tags as well
            if any(tag.lower() == genre_lower for tag in template.tags):
                logger.debug(f"Found template {template.id} for genre {genre} via tags")
                return template

        # Check user templates
        for template in self._user_templates.values():
            if template.genre.lower() == genre_lower:
                return template
            if any(tag.lower() == genre_lower for tag in template.tags):
                return template

        logger.debug(f"No template found for genre: {genre}")
        return None

    def get_recommended_counts(
        self, template: WorldTemplate, entity_type: str
    ) -> tuple[int, int] | None:
        """Get recommended entity counts from a template.

        Args:
            template: The world template.
            entity_type: The entity type (e.g., "characters", "locations").

        Returns:
            Tuple of (min, max) counts or None if not specified.
        """
        return template.recommended_counts.get(entity_type)

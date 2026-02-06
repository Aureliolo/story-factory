"""Settings page - world validation, circular types, calendar, and relationship minimums."""

import logging
from typing import TYPE_CHECKING

from nicegui import ui

from src.memory.conflict_types import RELATION_CONFLICT_MAPPING

if TYPE_CHECKING:
    from src.ui.pages.settings import SettingsPage

logger = logging.getLogger(__name__)


def _get_circular_type_options(current_types: list[str]) -> dict[str, str]:
    """Get options for circular relationship type multi-select.

    Merges known relationship types from RELATION_CONFLICT_MAPPING with any
    existing user-configured types to prevent data loss.

    Args:
        current_types: Currently configured circular relationship types.

    Returns:
        Dictionary mapping relationship type to display label.
    """
    # Merge known types from RELATION_CONFLICT_MAPPING with user-configured types
    all_types = set(RELATION_CONFLICT_MAPPING.keys()) | set(current_types)
    return {rel_type: rel_type.replace("_", " ").title() for rel_type in sorted(all_types)}


def build_validation_rules_section(page: SettingsPage) -> None:
    """Build validation rules settings card.

    Args:
        page: The SettingsPage instance.
    """
    with ui.card().classes("w-full"):
        page._section_header(
            "Validation Rules",
            "fact_check",
            "Configure validation, orphan detection, and fuzzy matching thresholds.",
        )

        with ui.column().classes("w-full gap-3"):
            with ui.row().classes("flex-wrap gap-x-4 gap-y-2"):
                page._relationship_validation_switch = ui.switch(
                    "Validate on Creation",
                    value=page.settings.relationship_validation_enabled,
                ).tooltip("Validate source/target entities exist")

                page._orphan_detection_switch = ui.switch(
                    "Orphan Detection",
                    value=page.settings.orphan_detection_enabled,
                ).tooltip("Find entities without relationships")

                page._circular_detection_switch = ui.switch(
                    "Circular Detection",
                    value=page.settings.circular_detection_enabled,
                ).tooltip("Find relationship loops")

            ui.separator().classes("my-2")

            with ui.row().classes("items-center gap-3 flex-wrap"):
                with ui.column().classes("gap-1"):
                    ui.label("Fuzzy Threshold").classes("text-xs text-gray-500")
                    page._fuzzy_threshold_input = page._build_number_input(
                        value=page.settings.fuzzy_match_threshold,
                        min_val=0.5,
                        max_val=1.0,
                        step=0.05,
                        tooltip_text="Similarity for name matching (0.5-1.0)",
                        width="w-16",
                    )

                with ui.column().classes("gap-1"):
                    ui.label("Max Relationships").classes("text-xs text-gray-500")
                    page._max_relationships_input = page._build_number_input(
                        value=page.settings.max_relationships_per_entity,
                        min_val=1,
                        max_val=50,
                        step=1,
                        tooltip_text="Per entity for suggestions (1-50)",
                        width="w-16",
                    )

    logger.debug("Validation rules section built")


def build_circular_types_section(page: SettingsPage) -> None:
    """Build circular relationship types settings card.

    Args:
        page: The SettingsPage instance.
    """
    with ui.card().classes("w-full"):
        page._section_header(
            "Circular Types",
            "loop",
            "Select relationship types where cycles would be illogical.",
        )

        def _toggle_check_all(e) -> None:
            page._circular_types_select.set_visibility(not e.value)

        page._circular_check_all_checkbox = ui.checkbox(
            "Check all types",
            value=page.settings.circular_check_all_types,
            on_change=_toggle_check_all,
        ).tooltip("When checked, all relationship types are checked for circularity")

        # Note: Values are extracted during save_to_settings() to preserve undo/redo
        page._circular_types_select = (
            ui.select(
                options=_get_circular_type_options(page.settings.circular_relationship_types),
                value=page.settings.circular_relationship_types,
                multiple=True,
            )
            .props("outlined dense use-chips")
            .classes("w-full")
            .tooltip("Select types where cycles would be illogical")
        )
        page._circular_types_select.set_visibility(not page.settings.circular_check_all_types)

    logger.debug("Circular types section built")


def build_calendar_temporal_section(page: SettingsPage) -> None:
    """Build calendar and temporal validation settings card.

    Args:
        page: The SettingsPage instance.
    """
    with ui.card().classes("w-full"):
        page._section_header(
            "Calendar & Temporal",
            "calendar_month",
            "Configure automatic calendar generation and timeline consistency validation.",
        )

        with ui.row().classes("flex-wrap gap-x-4 gap-y-2"):
            page._generate_calendar_switch = ui.switch(
                "Auto-Generate Calendar",
                value=page.settings.generate_calendar_on_world_build,
            ).tooltip("Create calendar during world build")

            page._temporal_validation_switch = ui.switch(
                "Temporal Validation",
                value=page.settings.validate_temporal_consistency,
            ).tooltip("Validate timeline consistency")

    logger.debug("Calendar & temporal section built")


def build_relationship_minimums_section(page: SettingsPage) -> None:
    """Build relationship minimums settings card.

    Args:
        page: The SettingsPage instance.
    """
    with ui.card().classes("w-full"):
        page._section_header(
            "Relationship Minimums",
            "tune",
            "Minimum relationships required per entity type and role.",
        )

        # Container for relationship minimum inputs - can be rebuilt on settings restore
        page._relationship_min_container = ui.column().classes("w-full")
        _build_relationship_minimums_inputs(page)

    logger.debug("Relationship minimums section built")


def _build_relationship_minimums_inputs(page: SettingsPage) -> None:
    """Build number inputs for relationship minimums using clear and rebuild pattern.

    Creates editable number inputs for each entity type and role combination.
    This function clears and rebuilds the container, ensuring UI stays in sync
    with settings even when structure changes.

    Args:
        page: The SettingsPage instance.
    """
    logger.debug(
        "Building relationship minimums inputs for %d entity types",
        len(page.settings.relationship_minimums),
    )
    page._relationship_min_container.clear()
    page._relationship_min_inputs: dict[str, dict[str, ui.number]] = {}  # type: ignore[misc]

    with page._relationship_min_container:
        for entity_type, roles in page.settings.relationship_minimums.items():
            page._relationship_min_inputs[entity_type] = {}
            with ui.row().classes("items-center gap-2 mb-2"):
                ui.icon("folder", size="xs").classes("text-blue-500")
                ui.label(f"{entity_type}:").classes("text-xs font-medium w-20")
                for role, count in roles.items():
                    with ui.column().classes("gap-0"):
                        ui.label(role).classes("text-[10px] text-gray-500")
                        num_input = (
                            ui.number(value=count, min=0, max=10, step=1)
                            .props("outlined dense")
                            .classes("w-14")
                            .tooltip(f"Min relationships for {entity_type}/{role}")
                        )
                        page._relationship_min_inputs[entity_type][role] = num_input


def save_to_settings(page: SettingsPage) -> None:
    """Extract world validation settings from UI and save to settings.

    Args:
        page: The SettingsPage instance.
    """
    settings = page.settings

    # Relationship validation settings
    if hasattr(page, "_relationship_validation_switch"):
        settings.relationship_validation_enabled = page._relationship_validation_switch.value
    if hasattr(page, "_orphan_detection_switch"):
        settings.orphan_detection_enabled = page._orphan_detection_switch.value
    if hasattr(page, "_circular_detection_switch"):
        settings.circular_detection_enabled = page._circular_detection_switch.value
    if hasattr(page, "_fuzzy_threshold_input"):
        settings.fuzzy_match_threshold = float(page._fuzzy_threshold_input.value)
    if hasattr(page, "_max_relationships_input"):
        settings.max_relationships_per_entity = int(page._max_relationships_input.value)
    if hasattr(page, "_circular_check_all_checkbox"):
        settings.circular_check_all_types = page._circular_check_all_checkbox.value
    if hasattr(page, "_circular_types_select"):
        settings.circular_relationship_types = list(page._circular_types_select.value)

    # Relationship minimums (extract from number inputs)
    if hasattr(page, "_relationship_min_inputs"):
        for entity_type, roles in page._relationship_min_inputs.items():
            if entity_type not in settings.relationship_minimums:
                raise ValueError(f"Unknown relationship minimums entity type: {entity_type}")
            for role, num_input in roles.items():
                if role not in settings.relationship_minimums[entity_type]:
                    raise ValueError(f"Unknown relationship minimums role: {entity_type}/{role}")
                if num_input.value is None:
                    raise ValueError(f"Relationship minimum required for {entity_type}/{role}")
                settings.relationship_minimums[entity_type][role] = int(num_input.value)

    # Calendar and temporal validation settings
    if hasattr(page, "_generate_calendar_switch"):
        settings.generate_calendar_on_world_build = page._generate_calendar_switch.value
    if hasattr(page, "_temporal_validation_switch"):
        settings.validate_temporal_consistency = page._temporal_validation_switch.value

    logger.debug("World validation settings saved")


def refresh_from_settings(page: SettingsPage) -> None:
    """Refresh world validation UI elements from current settings values.

    Args:
        page: The SettingsPage instance.
    """
    settings = page.settings

    # Relationship validation settings
    if hasattr(page, "_relationship_validation_switch"):
        page._relationship_validation_switch.value = settings.relationship_validation_enabled
    if hasattr(page, "_orphan_detection_switch"):
        page._orphan_detection_switch.value = settings.orphan_detection_enabled
    if hasattr(page, "_circular_detection_switch"):
        page._circular_detection_switch.value = settings.circular_detection_enabled
    if hasattr(page, "_fuzzy_threshold_input"):
        page._fuzzy_threshold_input.value = settings.fuzzy_match_threshold
    if hasattr(page, "_max_relationships_input"):
        page._max_relationships_input.value = settings.max_relationships_per_entity

    # Rebuild relationship minimums using clear-and-rebuild pattern
    if hasattr(page, "_relationship_min_container"):
        logger.debug("Rebuilding relationship minimums inputs from settings")
        _build_relationship_minimums_inputs(page)

    # Update circular check-all checkbox and types multi-select from settings
    if hasattr(page, "_circular_check_all_checkbox"):
        page._circular_check_all_checkbox.value = settings.circular_check_all_types
    if hasattr(page, "_circular_types_select"):
        logger.debug("Updating circular types select from settings")
        # Update options to include any types from restored settings
        page._circular_types_select.options = _get_circular_type_options(
            settings.circular_relationship_types
        )
        page._circular_types_select.value = list(settings.circular_relationship_types)
        page._circular_types_select.set_visibility(not settings.circular_check_all_types)

    # Calendar and temporal validation settings
    if hasattr(page, "_generate_calendar_switch"):
        page._generate_calendar_switch.value = settings.generate_calendar_on_world_build
    if hasattr(page, "_temporal_validation_switch"):
        page._temporal_validation_switch.value = settings.validate_temporal_consistency

    logger.debug("World validation UI refreshed from settings")

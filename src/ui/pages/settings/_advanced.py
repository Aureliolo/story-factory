"""Settings page - advanced LLM, world gen, story structure, data integrity, and relationship sections."""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from nicegui import ui

from src.settings import REFINEMENT_TEMP_DECAY_CURVES

if TYPE_CHECKING:
    from src.ui.pages.settings import SettingsPage

logger = logging.getLogger(__name__)


def build_world_gen_section(page: SettingsPage) -> None:
    """Build world generation settings section.

    Args:
        page: The SettingsPage instance.
    """
    with ui.card().classes("w-full h-full"):
        page._section_header(
            "World Generation",
            "public",
            "Configure entity counts for world building. "
            "Actual counts are randomized between min and max values.",
        )

        # Store inputs for saving
        page._world_gen_inputs: dict[str, tuple[ui.number, ui.number]] = {}  # type: ignore[misc]

        entity_configs = [
            ("characters", "Characters", "people", 1, 20),
            ("locations", "Locations", "place", 1, 15),
            ("factions", "Factions", "groups", 0, 10),
            ("items", "Items", "inventory", 0, 15),
            ("concepts", "Concepts", "lightbulb", 0, 10),
            ("relationships", "Relationships", "share", 1, 40),
        ]

        # Table-style layout with headers
        with ui.element("div").classes("w-full"):
            # Header row
            with ui.row().classes("items-center gap-2 mb-2 text-xs text-gray-500"):
                ui.element("div").classes("w-28")  # Spacer for label column
                ui.label("Min").classes("w-14 text-center")
                ui.label("Max").classes("w-14 text-center")

            # Entity rows
            for key, label, icon, abs_min, abs_max in entity_configs:
                min_attr = f"world_gen_{key}_min"
                max_attr = f"world_gen_{key}_max"
                current_min = getattr(page.settings, min_attr)
                current_max = getattr(page.settings, max_attr)

                with ui.row().classes("items-center gap-2"):
                    ui.icon(icon, size="xs").classes("text-gray-500 w-5")
                    ui.label(label).classes("text-sm w-24")
                    min_input = (
                        ui.number(value=current_min, min=abs_min, max=abs_max, step=1)
                        .props("outlined dense")
                        .classes("w-14")
                    )
                    max_input = (
                        ui.number(value=current_max, min=abs_min, max=abs_max, step=1)
                        .props("outlined dense")
                        .classes("w-14")
                    )

                    page._world_gen_inputs[key] = (min_input, max_input)

        # Quality refinement settings (subsection)
        ui.separator().classes("my-3")
        with ui.row().classes("items-center gap-2 mb-2"):
            ui.icon("auto_fix_high", size="xs").classes("text-gray-500")
            ui.label("Quality Refinement").classes("text-sm font-medium")

        # Quality threshold and iterations in a row
        with ui.row().classes("items-center gap-4"):
            with ui.column().classes("gap-1"):
                ui.label("Threshold").classes("text-xs text-gray-500")
                page._quality_threshold_input = (
                    ui.number(
                        value=page.settings.world_quality_threshold,
                        min=0.0,
                        max=10.0,
                        step=0.5,
                    )
                    .props("outlined dense")
                    .classes("w-16")
                    .tooltip("Minimum quality score (0-10) to accept entity")
                )

            with ui.column().classes("gap-1"):
                ui.label("Max Iter.").classes("text-xs text-gray-500")
                page._quality_max_iterations_input = (
                    ui.number(
                        value=page.settings.world_quality_max_iterations,
                        min=1,
                        max=10,
                        step=1,
                    )
                    .props("outlined dense")
                    .classes("w-16")
                    .tooltip("Maximum refinement iterations per entity")
                )

            with ui.column().classes("gap-1"):
                ui.label("Patience").classes("text-xs text-gray-500")
                page._quality_patience_input = (
                    ui.number(
                        value=page.settings.world_quality_early_stopping_patience,
                        min=1,
                        max=10,
                        step=1,
                    )
                    .props("outlined dense")
                    .classes("w-16")
                    .tooltip(
                        "Stop early after N consecutive score degradations. "
                        "Saves compute when quality isn't improving."
                    )
                )

    logger.debug("World generation section built")


def build_story_structure_section(page: SettingsPage) -> None:
    """Build story structure settings section.

    Args:
        page: The SettingsPage instance.
    """
    with ui.card().classes("w-full h-full"):
        page._section_header(
            "Story Structure",
            "menu_book",
            "Default chapter counts for different story lengths. "
            "Projects can override these values individually.",
        )

        # Store inputs for saving
        page._chapter_inputs: dict[str, ui.number] = {}  # type: ignore[misc]

        length_configs = [
            ("short_story", "Short Story", "Quick reads", 1, 5),
            ("novella", "Novella", "Medium length", 3, 15),
            ("novel", "Novel", "Full length", 10, 50),
        ]

        with ui.column().classes("w-full gap-4"):
            for key, label, hint, min_val, max_val in length_configs:
                attr_name = f"chapters_{key}"
                current_val = getattr(page.settings, attr_name)

                with ui.row().classes("w-full items-center gap-3"):
                    with ui.column().classes("flex-grow"):
                        ui.label(label).classes("text-sm font-medium")
                        ui.label(hint).classes("text-xs text-gray-500")

                    page._chapter_inputs[key] = (
                        ui.number(
                            value=current_val,
                            min=min_val,
                            max=max_val,
                            step=1,
                        )
                        .props("outlined dense")
                        .classes("w-20")
                    )

            ui.separator().classes("my-2")

            # Info about per-project overrides
            with ui.row().classes("items-center gap-2"):
                ui.icon("info", size="xs").classes("text-blue-500")
                ui.label("Individual projects can override these in 'Generation Settings'").classes(
                    "text-xs text-gray-500 dark:text-gray-400"
                )

    logger.debug("Story structure section built")


def build_data_integrity_section(page: SettingsPage) -> None:
    """Build data integrity settings section.

    Args:
        page: The SettingsPage instance.
    """
    with ui.card().classes("w-full h-full"):
        page._section_header(
            "Data Integrity",
            "verified_user",
            "Configure entity versioning and backup verification settings.",
        )

        with ui.column().classes("w-full gap-4"):
            # Entity version retention
            with ui.row().classes("w-full items-center gap-3"):
                with ui.column().classes("flex-grow"):
                    ui.label("Version History Limit").classes("text-sm font-medium")
                    ui.label("Versions kept per entity").classes("text-xs text-gray-500")

                page._entity_version_retention_input = (
                    ui.number(
                        value=page.settings.entity_version_retention,
                        min=1,
                        max=100,
                        step=1,
                    )
                    .props("outlined dense")
                    .classes("w-20")
                    .tooltip("Number of version history entries to keep per entity (1-100)")
                )

            ui.separator().classes("my-2")

            # Backup verification toggle
            with ui.row().classes("w-full items-center gap-3"):
                with ui.column().classes("flex-grow"):
                    ui.label("Verify Backups on Restore").classes("text-sm font-medium")
                    ui.label("Check integrity before restoring").classes("text-xs text-gray-500")

                page._backup_verify_on_restore_switch = ui.switch(
                    value=page.settings.backup_verify_on_restore,
                ).tooltip(
                    "When enabled, backups are verified for integrity "
                    "(checksums, file completeness, SQLite validity) before restoration"
                )

            # Info about what verification checks
            with ui.expansion("Verification Checks", icon="info").classes("w-full"):
                checks = [
                    ("check_circle", "File completeness"),
                    ("fingerprint", "SHA-256 checksums"),
                    ("storage", "SQLite integrity"),
                    ("code", "JSON validity"),
                    ("update", "Version compatibility"),
                ]
                for icon, label in checks:
                    with ui.row().classes("items-center gap-2"):
                        ui.icon(icon, size="xs").classes("text-green-500")
                        ui.label(label).classes("text-xs")

    logger.debug("Data integrity section built")


def build_advanced_llm_section(page: SettingsPage) -> None:
    """Build advanced LLM settings section (WP1/WP2 settings).

    Args:
        page: The SettingsPage instance.
    """
    with ui.card().classes("w-full h-full"):
        page._section_header(
            "Advanced LLM Settings",
            "tune",
            "Configure circuit breaker, retry strategies, duplicate detection, "
            "dynamic temperature, and early stopping behavior.",
        )

        with ui.expansion("Circuit Breaker", icon="security").classes("w-full"):
            with ui.column().classes("w-full gap-3 p-2"):
                with ui.row().classes("w-full items-center gap-3"):
                    with ui.column().classes("flex-grow"):
                        ui.label("Enabled").classes("text-sm font-medium")
                        ui.label("Protect against cascading LLM failures").classes(
                            "text-xs text-gray-500"
                        )
                    page._circuit_breaker_enabled_switch = ui.switch(
                        value=page.settings.circuit_breaker_enabled,
                    ).tooltip("When enabled, temporarily stops requests after repeated failures")

                # Settings visible when enabled
                with (
                    ui.element("div")
                    .classes("w-full")
                    .bind_visibility_from(page._circuit_breaker_enabled_switch, "value")
                ):
                    with ui.row().classes("items-center gap-4 flex-wrap"):
                        with ui.column().classes("gap-1"):
                            ui.label("Failure Threshold").classes("text-xs text-gray-500")
                            page._circuit_breaker_failure_threshold_input = (
                                page._build_number_input(
                                    value=page.settings.circuit_breaker_failure_threshold,
                                    min_val=1,
                                    max_val=20,
                                    step=1,
                                    tooltip_text="Failures before opening circuit (1-20)",
                                )
                            )

                        with ui.column().classes("gap-1"):
                            ui.label("Success Threshold").classes("text-xs text-gray-500")
                            page._circuit_breaker_success_threshold_input = (
                                page._build_number_input(
                                    value=page.settings.circuit_breaker_success_threshold,
                                    min_val=1,
                                    max_val=10,
                                    step=1,
                                    tooltip_text="Successes to close from half-open (1-10)",
                                )
                            )

                        with ui.column().classes("gap-1"):
                            ui.label("Timeout (s)").classes("text-xs text-gray-500")
                            page._circuit_breaker_timeout_input = page._build_number_input(
                                value=page.settings.circuit_breaker_timeout,
                                min_val=10,
                                max_val=600,
                                step=10,
                                tooltip_text="Seconds before half-open (10-600)",
                            )

        with ui.expansion("Retry Strategy", icon="replay").classes("w-full"):
            with ui.column().classes("w-full gap-3 p-2"):
                with ui.row().classes("items-center gap-4 flex-wrap"):
                    with ui.column().classes("gap-1"):
                        ui.label("Temp Increase").classes("text-xs text-gray-500")
                        page._retry_temp_increase_input = page._build_number_input(
                            value=page.settings.retry_temp_increase,
                            min_val=0.0,
                            max_val=1.0,
                            step=0.05,
                            tooltip_text="Temperature increase on retry attempt 2+ (0.0-1.0)",
                        )

                    with ui.column().classes("gap-1"):
                        ui.label("Simplify on Attempt").classes("text-xs text-gray-500")
                        page._retry_simplify_on_attempt_input = page._build_number_input(
                            value=page.settings.retry_simplify_on_attempt,
                            min_val=2,
                            max_val=10,
                            step=1,
                            tooltip_text="Attempt number to start simplifying prompts (2-10)",
                        )

        with ui.expansion("Duplicate Detection", icon="content_copy").classes("w-full"):
            with ui.column().classes("w-full gap-3 p-2"):
                with ui.row().classes("w-full items-center gap-3"):
                    with ui.column().classes("flex-grow"):
                        ui.label("Semantic Duplicate Detection").classes("text-sm font-medium")
                        ui.label("Use embeddings to detect similar content").classes(
                            "text-xs text-gray-500"
                        )
                    page._semantic_duplicate_enabled_switch = ui.switch(
                        value=page.settings.semantic_duplicate_enabled,
                    ).tooltip("Opt-in: detect duplicates using embedding similarity")

                # Settings visible when enabled
                with (
                    ui.element("div")
                    .classes("w-full")
                    .bind_visibility_from(page._semantic_duplicate_enabled_switch, "value")
                ):
                    with ui.row().classes("items-center gap-4 flex-wrap"):
                        with ui.column().classes("gap-1"):
                            ui.label("Similarity Threshold").classes("text-xs text-gray-500")
                            page._semantic_duplicate_threshold_input = page._build_number_input(
                                value=page.settings.semantic_duplicate_threshold,
                                min_val=0.5,
                                max_val=1.0,
                                step=0.05,
                                tooltip_text="Cosine similarity threshold (0.5-1.0)",
                            )

                        with ui.column().classes("gap-1 flex-grow"):
                            ui.label("Embedding Model").classes("text-xs text-gray-500")
                            # Get installed models for embedding selection
                            installed_models = page.services.model.list_installed()
                            embedding_options = {m: m for m in installed_models}
                            # Add current value if not in list
                            if page.settings.embedding_model not in embedding_options:
                                embedding_options[page.settings.embedding_model] = (
                                    page.settings.embedding_model
                                )
                            page._embedding_model_select = (
                                ui.select(
                                    options=embedding_options,
                                    value=page.settings.embedding_model,
                                )
                                .props("outlined dense")
                                .classes("w-full")
                                .tooltip("Model for generating embeddings")
                            )

        with ui.expansion("Refinement Temperature", icon="thermostat").classes("w-full"):
            with ui.column().classes("w-full gap-3 p-2"):
                ui.label("Dynamic temperature decay during quality refinement iterations").classes(
                    "text-xs text-gray-500 mb-2"
                )

                with ui.row().classes("items-center gap-4 flex-wrap"):
                    with ui.column().classes("gap-1"):
                        ui.label("Start Temp").classes("text-xs text-gray-500")
                        page._refinement_temp_start_input = page._build_number_input(
                            value=page.settings.world_quality_refinement_temp_start,
                            min_val=0.0,
                            max_val=2.0,
                            step=0.05,
                            tooltip_text="Starting temperature (0.0-2.0)",
                        )

                    with ui.column().classes("gap-1"):
                        ui.label("End Temp").classes("text-xs text-gray-500")
                        page._refinement_temp_end_input = page._build_number_input(
                            value=page.settings.world_quality_refinement_temp_end,
                            min_val=0.0,
                            max_val=2.0,
                            step=0.05,
                            tooltip_text="Ending temperature (0.0-2.0)",
                        )

                    with ui.column().classes("gap-1"):
                        ui.label("Decay Curve").classes("text-xs text-gray-500")
                        page._refinement_temp_decay_select = (
                            ui.select(
                                options=REFINEMENT_TEMP_DECAY_CURVES,
                                value=page.settings.world_quality_refinement_temp_decay,
                            )
                            .props("outlined dense")
                            .classes("w-32")
                            .tooltip("How temperature decreases over iterations")
                        )

        with ui.expansion("Early Stopping", icon="stop_circle").classes("w-full"):
            with ui.column().classes("w-full gap-3 p-2"):
                ui.label("Stop refinement early when quality improvements plateau").classes(
                    "text-xs text-gray-500 mb-2"
                )

                with ui.row().classes("items-center gap-4 flex-wrap"):
                    with ui.column().classes("gap-1"):
                        ui.label("Min Iterations").classes("text-xs text-gray-500")
                        page._early_stopping_min_iterations_input = page._build_number_input(
                            value=page.settings.world_quality_early_stopping_min_iterations,
                            min_val=1,
                            max_val=10,
                            step=1,
                            tooltip_text="Minimum iterations before early stop (1-10)",
                        )

                    with ui.column().classes("gap-1"):
                        ui.label("Variance Tolerance").classes("text-xs text-gray-500")
                        page._early_stopping_variance_tolerance_input = page._build_number_input(
                            value=page.settings.world_quality_early_stopping_variance_tolerance,
                            min_val=0.0,
                            max_val=2.0,
                            step=0.05,
                            tooltip_text="Score variance tolerance for plateau detection (0.0-2.0)",
                        )

    logger.debug("Advanced LLM section built")


def _make_remove_handler(page: SettingsPage, rel_type: str) -> Callable[[], None]:
    """Create a handler to remove a relationship type.

    Args:
        page: The SettingsPage instance.
        rel_type: The relationship type to remove.

    Returns:
        A callable that removes the type when invoked.
    """

    def remove() -> None:
        """Remove the relationship type and rebuild chips."""
        if rel_type in page.settings.circular_relationship_types:
            logger.debug("Removing circular relationship type: %s", rel_type)
            page.settings.circular_relationship_types.remove(rel_type)
            _build_circular_type_chips(page)

    return remove


def _build_circular_type_chips(page: SettingsPage) -> None:
    """Build chips for circular relationship types.

    Creates visual chip elements for each relationship type that can be removed
    by clicking the X button.

    Args:
        page: The SettingsPage instance.
    """
    logger.debug(
        "Building circular type chips for %d types",
        len(page.settings.circular_relationship_types),
    )
    page._circular_types_container.clear()
    with page._circular_types_container:
        for rel_type in page.settings.circular_relationship_types:
            with ui.element("div").classes(
                "flex items-center gap-1 px-2 py-1 bg-blue-100 dark:bg-blue-900 "
                "rounded-full text-xs"
            ):
                ui.label(rel_type).classes("text-blue-700 dark:text-blue-200")
                ui.button(icon="close", on_click=_make_remove_handler(page, rel_type)).props(
                    "flat dense round size=xs"
                ).classes("text-blue-500 dark:text-blue-300")


def build_relationship_validation_section(page: SettingsPage) -> None:
    """Build relationship validation and world health settings section.

    Args:
        page: The SettingsPage instance.
    """
    with ui.card().classes("w-full h-full"):
        page._section_header(
            "Relationship Validation",
            "fact_check",
            "Configure validation, orphan detection, and circular relationship checking.",
        )

        with ui.column().classes("w-full gap-4"):
            # Validation toggle
            with ui.row().classes("w-full items-center gap-3"):
                with ui.column().classes("flex-grow"):
                    ui.label("Validate on Creation").classes("text-sm font-medium")
                    ui.label("Check entity existence").classes("text-xs text-gray-500")

                page._relationship_validation_switch = ui.switch(
                    value=page.settings.relationship_validation_enabled,
                ).tooltip("Validate source/target entities exist when creating relationships")

            # Orphan detection toggle
            with ui.row().classes("w-full items-center gap-3"):
                with ui.column().classes("flex-grow"):
                    ui.label("Orphan Detection").classes("text-sm font-medium")
                    ui.label("Find entities without relationships").classes("text-xs text-gray-500")

                page._orphan_detection_switch = ui.switch(
                    value=page.settings.orphan_detection_enabled,
                ).tooltip("Enable detection of orphan entities in world health checks")

            # Circular detection toggle
            with ui.row().classes("w-full items-center gap-3"):
                with ui.column().classes("flex-grow"):
                    ui.label("Circular Detection").classes("text-sm font-medium")
                    ui.label("Find relationship loops").classes("text-xs text-gray-500")

                page._circular_detection_switch = ui.switch(
                    value=page.settings.circular_detection_enabled,
                ).tooltip("Enable detection of circular relationship chains")

            ui.separator().classes("my-2")

            # Fuzzy match threshold
            with ui.row().classes("w-full items-center gap-3"):
                with ui.column().classes("flex-grow"):
                    ui.label("Fuzzy Match Threshold").classes("text-sm font-medium")
                    ui.label("Similarity for name matching").classes("text-xs text-gray-500")

                page._fuzzy_threshold_input = (
                    ui.number(
                        value=page.settings.fuzzy_match_threshold,
                        min=0.5,
                        max=1.0,
                        step=0.05,
                    )
                    .props("outlined dense")
                    .classes("w-20")
                    .tooltip("Minimum similarity (0.5-1.0) for fuzzy entity name matching")
                )

            # Max relationships per entity
            with ui.row().classes("w-full items-center gap-3"):
                with ui.column().classes("flex-grow"):
                    ui.label("Max Relationships").classes("text-sm font-medium")
                    ui.label("Per entity for suggestions").classes("text-xs text-gray-500")

                page._max_relationships_input = (
                    ui.number(
                        value=page.settings.max_relationships_per_entity,
                        min=1,
                        max=50,
                        step=1,
                    )
                    .props("outlined dense")
                    .classes("w-20")
                    .tooltip("Maximum relationships to suggest per entity (1-50)")
                )

            # Circular relationship types (chip-based editor)
            with ui.column().classes("w-full gap-1"):
                ui.label("Circular Check Types").classes("text-sm font-medium")
                ui.label("Relationship types to check for cycles").classes("text-xs text-gray-500")

                # Container for chips - will be refreshed when types change
                page._circular_types_container = ui.row().classes("flex-wrap gap-1 mt-1")
                _build_circular_type_chips(page)

                # Add new type input
                with ui.row().classes("items-center gap-2 mt-2"):
                    page._new_circular_type_input = (
                        ui.input(placeholder="Add type...").props("outlined dense").classes("w-32")
                    )

                    def add_circular_type() -> None:
                        """Add a new circular relationship type from the input field."""
                        new_type = page._new_circular_type_input.value.strip().lower()
                        if not new_type:
                            logger.debug("Skipped adding circular relationship type: empty input")
                            return
                        if new_type in page.settings.circular_relationship_types:
                            logger.debug("Circular relationship type already present: %s", new_type)
                            page._new_circular_type_input.value = ""
                            return
                        page.settings.circular_relationship_types.append(new_type)
                        logger.info("Added circular relationship type: %s", new_type)
                        page._new_circular_type_input.value = ""
                        _build_circular_type_chips(page)

                    ui.button(icon="add", on_click=add_circular_type).props(
                        "flat dense round"
                    ).classes("text-blue-500")

            ui.separator().classes("my-2")

            # Calendar and Temporal Validation subsection
            with ui.row().classes("items-center gap-2 mb-2"):
                ui.icon("calendar_month", size="xs").classes("text-gray-500")
                ui.label("Calendar & Temporal").classes("text-sm font-medium")

            # Generate calendar on world build
            with ui.row().classes("w-full items-center gap-3"):
                with ui.column().classes("flex-grow"):
                    ui.label("Auto-Generate Calendar").classes("text-sm font-medium")
                    ui.label("Create calendar during world build").classes("text-xs text-gray-500")

                page._generate_calendar_switch = ui.switch(
                    value=page.settings.generate_calendar_on_world_build,
                ).tooltip(
                    "Automatically generate a fictional calendar system when building world structure"
                )

            # Temporal consistency validation
            with ui.row().classes("w-full items-center gap-3"):
                with ui.column().classes("flex-grow"):
                    ui.label("Temporal Validation").classes("text-sm font-medium")
                    ui.label("Validate timeline consistency").classes("text-xs text-gray-500")

                page._temporal_validation_switch = ui.switch(
                    value=page.settings.validate_temporal_consistency,
                ).tooltip(
                    "Validate that entity timelines are consistent "
                    "(e.g., characters born before factions they join)"
                )

            ui.separator().classes("my-2")

            # Relationship minimums (editable) - uses clear and rebuild pattern
            with ui.expansion("Relationship Minimums", icon="tune", value=False).classes("w-full"):
                ui.label("Minimum relationships per entity type/role:").classes(
                    "text-xs text-gray-500 mb-2"
                )
                # Container for relationship minimum inputs - can be rebuilt on settings restore
                page._relationship_min_container = ui.column().classes("w-full")
                _build_relationship_minimums_inputs(page)

    logger.debug("Relationship validation section built")


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
    """Extract advanced settings from UI and save to settings.

    Handles world generation, story structure, data integrity, advanced LLM,
    and relationship validation settings.

    Args:
        page: The SettingsPage instance.
    """
    settings = page.settings

    # World generation settings
    for key, (min_input, max_input) in page._world_gen_inputs.items():
        min_val = int(min_input.value)
        max_val = int(max_input.value)
        # Ensure min <= max
        if min_val > max_val:
            max_val = min_val
        setattr(settings, f"world_gen_{key}_min", min_val)
        setattr(settings, f"world_gen_{key}_max", max_val)

    # Quality refinement settings
    if hasattr(page, "_quality_threshold_input"):
        settings.world_quality_threshold = float(page._quality_threshold_input.value)
    if hasattr(page, "_quality_max_iterations_input"):
        settings.world_quality_max_iterations = int(page._quality_max_iterations_input.value)
    if hasattr(page, "_quality_patience_input"):
        settings.world_quality_early_stopping_patience = int(page._quality_patience_input.value)

    # Story structure settings (chapter counts)
    if hasattr(page, "_chapter_inputs"):
        for key, input_widget in page._chapter_inputs.items():
            setattr(settings, f"chapters_{key}", int(input_widget.value))

    # Data integrity settings
    if hasattr(page, "_entity_version_retention_input"):
        settings.entity_version_retention = int(page._entity_version_retention_input.value)
    if hasattr(page, "_backup_verify_on_restore_switch"):
        settings.backup_verify_on_restore = page._backup_verify_on_restore_switch.value

    # Advanced LLM settings (WP1/WP2)
    advanced_llm_settings_map = [
        ("_circuit_breaker_enabled_switch", "circuit_breaker_enabled", None),
        ("_circuit_breaker_failure_threshold_input", "circuit_breaker_failure_threshold", int),
        ("_circuit_breaker_success_threshold_input", "circuit_breaker_success_threshold", int),
        ("_circuit_breaker_timeout_input", "circuit_breaker_timeout", float),
        ("_retry_temp_increase_input", "retry_temp_increase", float),
        ("_retry_simplify_on_attempt_input", "retry_simplify_on_attempt", int),
        ("_semantic_duplicate_enabled_switch", "semantic_duplicate_enabled", None),
        ("_semantic_duplicate_threshold_input", "semantic_duplicate_threshold", float),
        ("_embedding_model_select", "embedding_model", None),
        ("_refinement_temp_start_input", "world_quality_refinement_temp_start", float),
        ("_refinement_temp_end_input", "world_quality_refinement_temp_end", float),
        ("_refinement_temp_decay_select", "world_quality_refinement_temp_decay", None),
        (
            "_early_stopping_min_iterations_input",
            "world_quality_early_stopping_min_iterations",
            int,
        ),
        (
            "_early_stopping_variance_tolerance_input",
            "world_quality_early_stopping_variance_tolerance",
            float,
        ),
    ]

    for ui_attr, setting_attr, type_conv in advanced_llm_settings_map:
        if hasattr(page, ui_attr):
            ui_element = getattr(page, ui_attr)
            value = ui_element.value
            if type_conv:
                value = type_conv(value)
            setattr(settings, setting_attr, value)

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
    # circular_relationship_types is modified directly by the chip UI, no extraction needed

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

    logger.debug("Advanced settings saved")


def refresh_from_settings(page: SettingsPage) -> None:
    """Refresh advanced UI elements from current settings values.

    Handles world generation, story structure, data integrity, advanced LLM,
    and relationship validation settings. Uses clear-and-rebuild for dynamic
    elements like relationship minimums and circular type chips.

    Args:
        page: The SettingsPage instance.
    """
    settings = page.settings

    # World generation settings
    if hasattr(page, "_world_gen_inputs"):
        for key, (min_input, max_input) in page._world_gen_inputs.items():
            min_attr = f"world_gen_{key}_min"
            max_attr = f"world_gen_{key}_max"
            if hasattr(settings, min_attr):
                min_input.value = getattr(settings, min_attr)
                max_input.value = getattr(settings, max_attr)

    # Quality refinement settings
    if hasattr(page, "_quality_threshold_input"):
        page._quality_threshold_input.value = settings.world_quality_threshold
    if hasattr(page, "_quality_max_iterations_input"):
        page._quality_max_iterations_input.value = settings.world_quality_max_iterations
    if hasattr(page, "_quality_patience_input"):
        page._quality_patience_input.value = settings.world_quality_early_stopping_patience

    # Data integrity settings
    if hasattr(page, "_entity_version_retention_input"):
        page._entity_version_retention_input.value = settings.entity_version_retention
    if hasattr(page, "_backup_verify_on_restore_switch"):
        page._backup_verify_on_restore_switch.value = settings.backup_verify_on_restore

    # Advanced LLM settings (WP1/WP2)
    advanced_llm_ui_map = [
        ("_circuit_breaker_enabled_switch", "circuit_breaker_enabled"),
        ("_circuit_breaker_failure_threshold_input", "circuit_breaker_failure_threshold"),
        ("_circuit_breaker_success_threshold_input", "circuit_breaker_success_threshold"),
        ("_circuit_breaker_timeout_input", "circuit_breaker_timeout"),
        ("_retry_temp_increase_input", "retry_temp_increase"),
        ("_retry_simplify_on_attempt_input", "retry_simplify_on_attempt"),
        ("_semantic_duplicate_enabled_switch", "semantic_duplicate_enabled"),
        ("_semantic_duplicate_threshold_input", "semantic_duplicate_threshold"),
        ("_embedding_model_select", "embedding_model"),
        ("_refinement_temp_start_input", "world_quality_refinement_temp_start"),
        ("_refinement_temp_end_input", "world_quality_refinement_temp_end"),
        ("_refinement_temp_decay_select", "world_quality_refinement_temp_decay"),
        ("_early_stopping_min_iterations_input", "world_quality_early_stopping_min_iterations"),
        (
            "_early_stopping_variance_tolerance_input",
            "world_quality_early_stopping_variance_tolerance",
        ),
    ]

    for ui_attr, setting_attr in advanced_llm_ui_map:
        if hasattr(page, ui_attr):
            getattr(page, ui_attr).value = getattr(settings, setting_attr)

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

    # Rebuild circular type chips from settings
    if hasattr(page, "_circular_types_container"):
        logger.debug("Rebuilding circular type chips from settings")
        _build_circular_type_chips(page)

    # Calendar and temporal validation settings
    if hasattr(page, "_generate_calendar_switch"):
        page._generate_calendar_switch.value = settings.generate_calendar_on_world_build
    if hasattr(page, "_temporal_validation_switch"):
        page._temporal_validation_switch.value = settings.validate_temporal_consistency

    logger.debug("Advanced UI refreshed from settings")

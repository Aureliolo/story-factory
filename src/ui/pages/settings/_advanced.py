"""Settings page - advanced LLM, world gen, story structure, data integrity, and relationship sections."""

import logging
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

            # Circular relationship types (editable)
            with ui.row().classes("w-full items-center gap-3"):
                with ui.column().classes("flex-grow"):
                    ui.label("Circular Check Types").classes("text-sm font-medium")
                    ui.label("Comma-separated relationship types").classes("text-xs text-gray-500")

                page._circular_types_input = (
                    ui.input(
                        value=", ".join(page.settings.circular_relationship_types),
                    )
                    .props("outlined dense")
                    .classes("w-64")
                    .tooltip(
                        "Relationship types to check for cycles (e.g., owns, reports_to, parent_of)"
                    )
                )

            ui.separator().classes("my-2")

            # Relationship minimums info (read-only display)
            with ui.expansion("Relationship Minimums", icon="info", value=False).classes("w-full"):
                ui.label("Minimum relationships per entity type/role:").classes(
                    "text-xs text-gray-500 mb-2"
                )
                for entity_type, roles in page.settings.relationship_minimums.items():
                    with ui.row().classes("items-center gap-2 mb-1"):
                        ui.icon("folder", size="xs").classes("text-blue-500")
                        ui.label(f"{entity_type}:").classes("text-xs font-medium w-20")
                        roles_str = ", ".join(f"{r}={c}" for r, c in roles.items())
                        ui.label(roles_str).classes("text-xs text-gray-600")
                ui.label("Edit settings.json directly to customize minimums").classes(
                    "text-xs text-gray-400 italic mt-2"
                )

    logger.debug("Relationship validation section built")

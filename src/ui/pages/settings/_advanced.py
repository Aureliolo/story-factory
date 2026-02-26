"""Settings page - advanced LLM, world gen, story structure, and data integrity sections."""

import logging
from typing import TYPE_CHECKING

from nicegui import ui

from src.settings import REFINEMENT_TEMP_DECAY_CURVES

if TYPE_CHECKING:
    from src.ui.pages.settings import SettingsPage

logger = logging.getLogger(__name__)


def _subsection_header(title: str, icon: str) -> None:
    """Build a compact subsection header with icon.

    Args:
        title: Text label for the subsection header.
        icon: Name of the Material icon to display.
    """
    with ui.row().classes("items-center gap-2 mb-2"):
        ui.icon(icon, size="xs").classes("text-gray-500")
        ui.label(title).classes("text-sm font-medium")


def build_world_gen_section(page: SettingsPage) -> None:
    """Build world generation settings section.

    Args:
        page: The SettingsPage instance.
    """
    with ui.card().classes("w-full"):
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
            ("events", "Events", "event", 1, 15),
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
        _subsection_header("Quality Refinement", "auto_fix_high")

        # Per-entity quality thresholds
        ui.label("Quality Thresholds (per entity type)").classes("text-xs text-gray-500 mb-1")
        page._quality_threshold_inputs: dict[str, ui.number] = {}  # type: ignore[misc]
        primary_threshold_configs = [
            ("character", "Character", "people"),
            ("location", "Location", "place"),
            ("faction", "Faction", "groups"),
            ("item", "Item", "inventory"),
            ("concept", "Concept", "lightbulb"),
            ("event", "Event", "event"),
            ("calendar", "Calendar", "calendar_month"),
        ]
        secondary_threshold_configs = [
            ("relationship", "Relationship", "link"),
            ("plot", "Plot", "auto_stories"),
            ("chapter", "Chapter", "menu_book"),
        ]
        with ui.row().classes("items-center gap-3 flex-wrap mb-3"):
            for key, label, icon in primary_threshold_configs:
                threshold_val = page.settings.world_quality_thresholds[key]
                with ui.column().classes("gap-1"):
                    with ui.row().classes("items-center gap-1"):
                        ui.icon(icon, size="xs").classes("text-gray-500")
                        ui.label(label).classes("text-xs text-gray-500")
                    page._quality_threshold_inputs[key] = (
                        ui.number(
                            value=threshold_val,
                            min=0.0,
                            max=10.0,
                            step=0.5,
                        )
                        .props("outlined dense")
                        .classes("w-16")
                        .tooltip(f"Min quality score (0-10) for {label.lower()} entities")
                    )

        ui.label("Secondary").classes("text-xs text-gray-400 mt-1")
        with ui.row().classes("items-center gap-3 flex-wrap mb-3"):
            for key, label, icon in secondary_threshold_configs:
                threshold_val = page.settings.world_quality_thresholds[key]
                with ui.column().classes("gap-1"):
                    with ui.row().classes("items-center gap-1"):
                        ui.icon(icon, size="xs").classes("text-gray-400")
                        ui.label(label).classes("text-xs text-gray-400")
                    page._quality_threshold_inputs[key] = (
                        ui.number(
                            value=threshold_val,
                            min=0.0,
                            max=10.0,
                            step=0.5,
                        )
                        .props("outlined dense")
                        .classes("w-16")
                        .tooltip(f"Min quality score (0-10) for {label.lower()} entities")
                    )

        # Max iterations and patience in a row
        with ui.row().classes("items-center gap-4"):
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
    with ui.card().classes("w-full"):
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
                    "text-xs text-gray-400"
                )

    logger.debug("Story structure section built")


def build_data_integrity_section(page: SettingsPage) -> None:
    """Build data integrity settings section.

    Args:
        page: The SettingsPage instance.
    """
    with ui.card().classes("w-full"):
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

            # Verification checks info (inline instead of expansion)
            ui.separator().classes("my-2")
            _subsection_header("Verification Checks", "info")
            checks = [
                ("check_circle", "File completeness"),
                ("fingerprint", "SHA-256 checksums"),
                ("storage", "SQLite integrity"),
                ("code", "JSON validity"),
                ("update", "Version compatibility"),
            ]
            with ui.row().classes("flex-wrap gap-x-4 gap-y-1"):
                for icon, label in checks:
                    with ui.row().classes("items-center gap-1"):
                        ui.icon(icon, size="xs").classes("text-green-500")
                        ui.label(label).classes("text-xs")

    logger.debug("Data integrity section built")


def build_circuit_breaker_section(page: SettingsPage) -> None:
    """Build circuit breaker settings card.

    Args:
        page: The SettingsPage instance.
    """
    with ui.card().classes("w-full"):
        page._section_header(
            "Circuit Breaker",
            "security",
            "Protect against cascading LLM failures by opening the circuit after repeated errors.",
        )

        with ui.row().classes("items-center gap-3 mb-3"):
            page._circuit_breaker_enabled_switch = ui.switch(
                "Enabled",
                value=page.settings.circuit_breaker_enabled,
            ).tooltip("Protect against cascading LLM failures")

        with ui.element("div").bind_visibility_from(page._circuit_breaker_enabled_switch, "value"):
            with ui.row().classes("items-center gap-3 flex-wrap"):
                with ui.column().classes("gap-1"):
                    ui.label("Failures").classes("text-xs text-gray-500")
                    page._circuit_breaker_failure_threshold_input = page._build_number_input(
                        value=page.settings.circuit_breaker_failure_threshold,
                        min_val=1,
                        max_val=20,
                        step=1,
                        tooltip_text="Failures before opening circuit (1-20)",
                        width="w-16",
                    )

                with ui.column().classes("gap-1"):
                    ui.label("Successes").classes("text-xs text-gray-500")
                    page._circuit_breaker_success_threshold_input = page._build_number_input(
                        value=page.settings.circuit_breaker_success_threshold,
                        min_val=1,
                        max_val=10,
                        step=1,
                        tooltip_text="Successes to close from half-open (1-10)",
                        width="w-16",
                    )

                with ui.column().classes("gap-1"):
                    ui.label("Timeout").classes("text-xs text-gray-500")
                    page._circuit_breaker_timeout_input = page._build_number_input(
                        value=page.settings.circuit_breaker_timeout,
                        min_val=10,
                        max_val=600,
                        step=10,
                        tooltip_text="Seconds before half-open (10-600)",
                        width="w-16",
                    )

        ui.separator().classes("my-3")
        _subsection_header("Request Queue", "queue")

        with ui.row().classes("items-center gap-3 flex-wrap"):
            with ui.column().classes("gap-1"):
                ui.label("Max Concurrent").classes("text-xs text-gray-500")
                page._llm_max_concurrent_requests_input = page._build_number_input(
                    value=page.settings.llm_max_concurrent_requests,
                    min_val=1,
                    max_val=10,
                    step=1,
                    tooltip_text="Max concurrent LLM requests (1-10). "
                    "Higher values speed up parallel generation but use more VRAM.",
                    width="w-16",
                )

            with ui.column().classes("gap-1"):
                ui.label("Queue Timeout").classes("text-xs text-gray-500")
                page._llm_semaphore_timeout_input = page._build_number_input(
                    value=page.settings.llm_semaphore_timeout,
                    min_val=30,
                    max_val=600,
                    step=30,
                    tooltip_text="Seconds to wait for LLM queue slot (30-600)",
                    width="w-16",
                )

        ui.separator().classes("my-3")
        _subsection_header("Small Model Timeout Cap", "timer")

        with ui.row().classes("items-center gap-3 flex-wrap"):
            with ui.column().classes("gap-1"):
                ui.label("Size Threshold (GB)").classes("text-xs text-gray-500")
                page._small_model_size_threshold_input = page._build_number_input(
                    value=page.settings.small_model_size_threshold_gb,
                    min_val=1.0,
                    max_val=20.0,
                    step=0.5,
                    tooltip_text="Models under this size (GB) get capped timeout (1-20)",
                    width="w-16",
                )

            with ui.column().classes("gap-1"):
                ui.label("Timeout Cap (s)").classes("text-xs text-gray-500")
                page._small_model_timeout_cap_input = page._build_number_input(
                    value=page.settings.small_model_timeout_cap,
                    min_val=10.0,
                    max_val=300.0,
                    step=5.0,
                    tooltip_text="Max timeout in seconds for small models (10-300)",
                    width="w-16",
                )

    logger.debug("Circuit breaker section built")


def build_retry_strategy_section(page: SettingsPage) -> None:
    """Build retry strategy settings card.

    Args:
        page: The SettingsPage instance.
    """
    with ui.card().classes("w-full"):
        page._section_header(
            "Retry Strategy",
            "replay",
            "Configure how failed LLM requests are retried with adjusted parameters.",
        )

        with ui.row().classes("items-center gap-3 flex-wrap"):
            with ui.column().classes("gap-1"):
                ui.label("Temp Increase").classes("text-xs text-gray-500")
                page._retry_temp_increase_input = page._build_number_input(
                    value=page.settings.retry_temp_increase,
                    min_val=0.0,
                    max_val=1.0,
                    step=0.05,
                    tooltip_text="Temperature increase on retry (0.0-1.0)",
                    width="w-16",
                )

            with ui.column().classes("gap-1"):
                ui.label("Simplify At").classes("text-xs text-gray-500")
                page._retry_simplify_on_attempt_input = page._build_number_input(
                    value=page.settings.retry_simplify_on_attempt,
                    min_val=2,
                    max_val=10,
                    step=1,
                    tooltip_text="Attempt to start simplifying prompts (2-10)",
                    width="w-16",
                )

    logger.debug("Retry strategy section built")


def build_duplicate_detection_section(page: SettingsPage) -> None:
    """Build duplicate detection settings card.

    Args:
        page: The SettingsPage instance.
    """
    with ui.card().classes("w-full"):
        page._section_header(
            "Duplicate Detection",
            "content_copy",
            "Use semantic embeddings to detect and prevent similar content generation.",
        )

        with ui.row().classes("items-center gap-3 mb-3"):
            page._semantic_duplicate_enabled_switch = ui.switch(
                "Semantic Detection",
                value=page.settings.semantic_duplicate_enabled,
            ).tooltip("Use embeddings to detect similar content")

        with ui.element("div").bind_visibility_from(
            page._semantic_duplicate_enabled_switch, "value"
        ):
            with ui.row().classes("items-center gap-3 flex-wrap"):
                with ui.column().classes("gap-1"):
                    ui.label("Threshold").classes("text-xs text-gray-500")
                    page._semantic_duplicate_threshold_input = page._build_number_input(
                        value=page.settings.semantic_duplicate_threshold,
                        min_val=0.5,
                        max_val=1.0,
                        step=0.05,
                        tooltip_text="Cosine similarity threshold (0.5-1.0)",
                        width="w-16",
                    )

                with ui.column().classes("gap-1"):
                    ui.label("Embedding Model").classes("text-xs text-gray-500")
                    # Use cached installed models from parent build()
                    installed_models = page._cached_installed_models
                    embedding_options: dict[str, str] = {"": "(Select a model)"}
                    embedding_options.update({m: m for m in installed_models})
                    current_value = page.settings.embedding_model
                    if current_value and current_value not in embedding_options:
                        embedding_options[current_value] = current_value
                    page._embedding_model_select = (
                        ui.select(
                            options=embedding_options,
                            value=current_value,
                        )
                        .props("outlined dense")
                        .classes("w-48")
                        .tooltip("Model for generating embeddings")
                    )

    logger.debug("Duplicate detection section built")


def build_rag_context_section(page: SettingsPage) -> None:
    """Build RAG (smart context retrieval) settings card.

    Args:
        page: The SettingsPage instance.
    """
    with ui.card().classes("w-full"):
        page._section_header(
            "Smart Context (RAG)",
            "psychology",
            "Use vector similarity search to retrieve relevant world context "
            "for every LLM call. Requires an embedding model.",
        )

        with ui.row().classes("items-center gap-3 mb-3"):
            page._rag_context_enabled_switch = ui.switch(
                "Enable RAG Context",
                value=page.settings.rag_context_enabled,
            ).tooltip(
                "When enabled, agents receive semantically relevant world "
                "context instead of a fixed slice of entities"
            )

        with ui.element("div").bind_visibility_from(page._rag_context_enabled_switch, "value"):
            with ui.row().classes("items-center gap-3 flex-wrap mb-3"):
                with ui.column().classes("gap-1"):
                    ui.label("Max Tokens").classes("text-xs text-gray-500")
                    page._rag_context_max_tokens_input = page._build_number_input(
                        value=page.settings.rag_context_max_tokens,
                        min_val=100,
                        max_val=16000,
                        step=100,
                        tooltip_text="Maximum tokens of context per LLM call (100-16000)",
                        width="w-20",
                    )

                with ui.column().classes("gap-1"):
                    ui.label("Max Items").classes("text-xs text-gray-500")
                    page._rag_context_max_items_input = page._build_number_input(
                        value=page.settings.rag_context_max_items,
                        min_val=1,
                        max_val=100,
                        step=5,
                        tooltip_text="Maximum items to retrieve per query (1-100)",
                        width="w-16",
                    )

                with ui.column().classes("gap-1"):
                    ui.label("Threshold").classes("text-xs text-gray-500")
                    page._rag_context_threshold_input = page._build_number_input(
                        value=page.settings.rag_context_similarity_threshold,
                        min_val=0.0,
                        max_val=1.0,
                        step=0.05,
                        tooltip_text="Min relevance score to include an item (0.0-1.0)",
                        width="w-16",
                    )

            ui.separator().classes("my-2")
            _subsection_header("Graph Expansion", "hub")

            with ui.row().classes("items-center gap-3 mb-3"):
                page._rag_graph_expansion_switch = ui.switch(
                    "Expand with neighbors",
                    value=page.settings.rag_context_graph_expansion,
                ).tooltip("Include related entities from the world graph")

            with ui.element("div").bind_visibility_from(page._rag_graph_expansion_switch, "value"):
                with ui.row().classes("items-center gap-3"):
                    with ui.column().classes("gap-1"):
                        ui.label("Depth").classes("text-xs text-gray-500")
                        page._rag_graph_depth_input = page._build_number_input(
                            value=page.settings.rag_context_graph_depth,
                            min_val=1,
                            max_val=3,
                            step=1,
                            tooltip_text="Graph neighbor expansion depth (1-3)",
                            width="w-16",
                        )

    logger.debug("RAG context section built")


def build_refinement_stopping_section(page: SettingsPage) -> None:
    """Build refinement temperature and early stopping settings card.

    Args:
        page: The SettingsPage instance.
    """
    with ui.card().classes("w-full"):
        page._section_header(
            "Refinement & Stopping",
            "thermostat",
            "Configure dynamic temperature decay and early stopping for quality refinement.",
        )

        _subsection_header("Temperature Decay", "thermostat")
        with ui.row().classes("items-center gap-3 flex-wrap mb-4"):
            with ui.column().classes("gap-1"):
                ui.label("Start").classes("text-xs text-gray-500")
                page._refinement_temp_start_input = page._build_number_input(
                    value=page.settings.world_quality_refinement_temp_start,
                    min_val=0.0,
                    max_val=2.0,
                    step=0.05,
                    tooltip_text="Starting temperature (0.0-2.0)",
                    width="w-16",
                )

            with ui.column().classes("gap-1"):
                ui.label("End").classes("text-xs text-gray-500")
                page._refinement_temp_end_input = page._build_number_input(
                    value=page.settings.world_quality_refinement_temp_end,
                    min_val=0.0,
                    max_val=2.0,
                    step=0.05,
                    tooltip_text="Ending temperature (0.0-2.0)",
                    width="w-16",
                )

            with ui.column().classes("gap-1"):
                ui.label("Decay").classes("text-xs text-gray-500")
                page._refinement_temp_decay_select = (
                    ui.select(
                        options=REFINEMENT_TEMP_DECAY_CURVES,
                        value=page.settings.world_quality_refinement_temp_decay,
                    )
                    .props("outlined dense")
                    .classes("w-24")
                    .tooltip("How temperature decreases over iterations")
                )

        ui.separator().classes("my-2")

        _subsection_header("Early Stopping", "stop_circle")
        ui.label("Stop refinement when quality improvements plateau").classes(
            "text-xs text-gray-500 mb-2"
        )

        with ui.row().classes("items-center gap-3 flex-wrap"):
            with ui.column().classes("gap-1"):
                ui.label("Min Iter.").classes("text-xs text-gray-500")
                page._early_stopping_min_iterations_input = page._build_number_input(
                    value=page.settings.world_quality_early_stopping_min_iterations,
                    min_val=1,
                    max_val=10,
                    step=1,
                    tooltip_text="Minimum iterations before early stop (1-10)",
                    width="w-16",
                )

            with ui.column().classes("gap-1"):
                ui.label("Variance").classes("text-xs text-gray-500")
                page._early_stopping_variance_tolerance_input = page._build_number_input(
                    value=page.settings.world_quality_early_stopping_variance_tolerance,
                    min_val=0.0,
                    max_val=2.0,
                    step=0.05,
                    tooltip_text="Score variance tolerance for plateau detection (0.0-2.0)",
                    width="w-16",
                )

            with ui.column().classes("gap-1"):
                ui.label("Plateau Tol.").classes("text-xs text-gray-500")
                page._score_plateau_tolerance_input = page._build_number_input(
                    value=page.settings.world_quality_score_plateau_tolerance,
                    min_val=0.0,
                    max_val=1.0,
                    step=0.05,
                    tooltip_text="Max score difference for plateau early-stop (0.0-1.0)",
                    width="w-16",
                )

    logger.debug("Refinement & stopping section built")


def build_judge_consistency_section(page: SettingsPage) -> None:
    """Build judge consistency settings card.

    Controls multi-call averaging and outlier detection for the quality judge,
    which reduces score variance from noisy local models.

    Args:
        page: The SettingsPage instance.
    """
    with ui.card().classes("w-full"):
        page._section_header(
            "Judge Consistency",
            "balance",
            "Improve scoring reliability by calling the judge multiple times "
            "and averaging results with outlier detection.",
        )

        with ui.row().classes("items-center gap-3 mb-3"):
            page._judge_consistency_switch = ui.switch(
                "Enable Judge Consistency",
                value=page.settings.judge_consistency_enabled,
            ).tooltip(
                "Master switch for judge consistency features. "
                "When disabled, all settings below are ignored."
            )

        with ui.element("div").bind_visibility_from(page._judge_consistency_switch, "value"):
            with ui.row().classes("items-center gap-3 mb-3"):
                page._judge_multi_call_switch = ui.switch(
                    "Multi-Call Averaging",
                    value=page.settings.judge_multi_call_enabled,
                ).tooltip(
                    "Call the judge model multiple times per entity and average scores. "
                    "Disabled by default — testing shows near-zero variance at low temperature. "
                    "Enable only if using a small or highly variable judge model."
                )

            with ui.element("div").bind_visibility_from(page._judge_multi_call_switch, "value"):
                with ui.row().classes("items-center gap-3 flex-wrap mb-3"):
                    with ui.column().classes("gap-1"):
                        ui.label("Calls").classes("text-xs text-gray-500")
                        page._judge_multi_call_count_input = page._build_number_input(
                            value=page.settings.judge_multi_call_count,
                            min_val=2,
                            max_val=5,
                            step=1,
                            tooltip_text="Number of judge calls per entity (2-5)",
                            width="w-16",
                        )

                    with ui.column().classes("gap-1"):
                        ui.label("Confidence").classes("text-xs text-gray-500")
                        page._judge_confidence_threshold_input = page._build_number_input(
                            value=page.settings.judge_confidence_threshold,
                            min_val=0.0,
                            max_val=1.0,
                            step=0.05,
                            tooltip_text="Min confidence for reliable decisions (0.0-1.0)",
                            width="w-16",
                        )

                ui.separator().classes("my-2")

                _subsection_header("Outlier Detection", "filter_alt")
                with ui.row().classes("items-center gap-3 mb-3"):
                    page._judge_outlier_detection_switch = ui.switch(
                        "Detect Outliers",
                        value=page.settings.judge_outlier_detection,
                    ).tooltip("Remove statistical outlier scores before averaging")

                with ui.element("div").bind_visibility_from(
                    page._judge_outlier_detection_switch, "value"
                ):
                    with ui.row().classes("items-center gap-3 flex-wrap"):
                        with ui.column().classes("gap-1"):
                            ui.label("Std Threshold").classes("text-xs text-gray-500")
                            page._judge_outlier_std_threshold_input = page._build_number_input(
                                value=page.settings.judge_outlier_std_threshold,
                                min_val=1.0,
                                max_val=4.0,
                                step=0.5,
                                tooltip_text="Standard deviations for outlier detection (1.0-4.0)",
                                width="w-16",
                            )

                        with ui.column().classes("gap-1"):
                            ui.label("Strategy").classes("text-xs text-gray-500")
                            page._judge_outlier_strategy_select = (
                                ui.select(
                                    options={
                                        "median": "Median",
                                        "mean": "Mean",
                                    },
                                    value=page.settings.judge_outlier_strategy,
                                )
                                .props("outlined dense")
                                .classes("w-24")
                                .tooltip("How to aggregate scores after outlier removal")
                            )

    logger.debug("Judge consistency section built")


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

    # Quality refinement settings — per-entity thresholds
    if hasattr(page, "_quality_threshold_inputs"):
        for entity_type, input_widget in page._quality_threshold_inputs.items():
            settings.world_quality_thresholds[entity_type] = float(input_widget.value)
        # Keep legacy single threshold in sync — older code paths and migration
        # use world_quality_threshold as the seed value for new per-entity keys.
        # max() ensures it stays representative of the strictest configured threshold.
        settings.world_quality_threshold = max(settings.world_quality_thresholds.values())
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
        (
            "_score_plateau_tolerance_input",
            "world_quality_score_plateau_tolerance",
            float,
        ),
        # Judge consistency settings
        ("_judge_consistency_switch", "judge_consistency_enabled", None),
        ("_judge_multi_call_switch", "judge_multi_call_enabled", None),
        ("_judge_multi_call_count_input", "judge_multi_call_count", int),
        ("_judge_confidence_threshold_input", "judge_confidence_threshold", float),
        ("_judge_outlier_detection_switch", "judge_outlier_detection", None),
        ("_judge_outlier_std_threshold_input", "judge_outlier_std_threshold", float),
        ("_judge_outlier_strategy_select", "judge_outlier_strategy", None),
        ("_llm_max_concurrent_requests_input", "llm_max_concurrent_requests", int),
        ("_llm_semaphore_timeout_input", "llm_semaphore_timeout", int),
        # Small model timeout cap
        ("_small_model_size_threshold_input", "small_model_size_threshold_gb", float),
        ("_small_model_timeout_cap_input", "small_model_timeout_cap", float),
        # RAG context settings
        ("_rag_context_enabled_switch", "rag_context_enabled", None),
        ("_rag_context_max_tokens_input", "rag_context_max_tokens", int),
        ("_rag_context_max_items_input", "rag_context_max_items", int),
        ("_rag_context_threshold_input", "rag_context_similarity_threshold", float),
        ("_rag_graph_expansion_switch", "rag_context_graph_expansion", None),
        ("_rag_graph_depth_input", "rag_context_graph_depth", int),
    ]

    for ui_attr, setting_attr, type_conv in advanced_llm_settings_map:
        if hasattr(page, ui_attr):
            ui_element = getattr(page, ui_attr)
            value = ui_element.value
            if type_conv:
                value = type_conv(value)
            setattr(settings, setting_attr, value)

    logger.debug("Advanced settings saved")


def refresh_from_settings(page: SettingsPage) -> None:
    """Refresh advanced UI elements from current settings values.

    Handles world generation, story structure, data integrity, advanced LLM,
    and relationship validation settings. Uses clear-and-rebuild for dynamic
    elements like relationship minimums.

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

    # Quality refinement settings — per-entity thresholds
    if hasattr(page, "_quality_threshold_inputs"):
        for entity_type, input_widget in page._quality_threshold_inputs.items():
            input_widget.value = settings.world_quality_thresholds[entity_type]
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
        ("_score_plateau_tolerance_input", "world_quality_score_plateau_tolerance"),
        # Judge consistency settings
        ("_judge_consistency_switch", "judge_consistency_enabled"),
        ("_judge_multi_call_switch", "judge_multi_call_enabled"),
        ("_judge_multi_call_count_input", "judge_multi_call_count"),
        ("_judge_confidence_threshold_input", "judge_confidence_threshold"),
        ("_judge_outlier_detection_switch", "judge_outlier_detection"),
        ("_judge_outlier_std_threshold_input", "judge_outlier_std_threshold"),
        ("_judge_outlier_strategy_select", "judge_outlier_strategy"),
        ("_llm_max_concurrent_requests_input", "llm_max_concurrent_requests"),
        ("_llm_semaphore_timeout_input", "llm_semaphore_timeout"),
        # Small model timeout cap
        ("_small_model_size_threshold_input", "small_model_size_threshold_gb"),
        ("_small_model_timeout_cap_input", "small_model_timeout_cap"),
        # RAG context settings
        ("_rag_context_enabled_switch", "rag_context_enabled"),
        ("_rag_context_max_tokens_input", "rag_context_max_tokens"),
        ("_rag_context_max_items_input", "rag_context_max_items"),
        ("_rag_context_threshold_input", "rag_context_similarity_threshold"),
        ("_rag_graph_expansion_switch", "rag_context_graph_expansion"),
        ("_rag_graph_depth_input", "rag_context_graph_depth"),
    ]

    for ui_attr, setting_attr in advanced_llm_ui_map:
        if hasattr(page, ui_attr):
            getattr(page, ui_attr).value = getattr(settings, setting_attr)

    logger.debug("Advanced UI refreshed from settings")

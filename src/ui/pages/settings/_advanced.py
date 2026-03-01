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
        page._world_gen_inputs.clear()

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

        ui.separator().classes("my-3")
        _subsection_header("Streaming Timeouts", "stream")

        ui.label("Guard against stalled Ollama inference during long generations.").classes(
            "text-xs text-gray-500 mb-1"
        )

        with ui.row().classes("items-center gap-3 flex-wrap"):
            with ui.column().classes("gap-1"):
                ui.label("Inter-Chunk (s)").classes("text-xs text-gray-500")
                page._streaming_inter_chunk_timeout_input = page._build_number_input(
                    value=page.settings.streaming_inter_chunk_timeout,
                    min_val=30,
                    max_val=600,
                    step=10,
                    tooltip_text="Max seconds between stream chunks before timeout (30-600)",
                    width="w-16",
                )

            with ui.column().classes("gap-1"):
                ui.label("Wall Clock (s)").classes("text-xs text-gray-500")
                page._streaming_wall_clock_timeout_input = page._build_number_input(
                    value=page.settings.streaming_wall_clock_timeout,
                    min_val=120,
                    max_val=3600,
                    step=60,
                    tooltip_text="Max total seconds for entire generation (120-3600)",
                    width="w-16",
                )

        ui.separator().classes("my-3")
        _subsection_header("Model Service Cache TTLs", "cached")

        ui.label(
            "How often the UI re-queries Ollama for health, models, and VRAM (in seconds)."
        ).classes("text-xs text-gray-500 mb-1")

        with ui.row().classes("items-center gap-3 flex-wrap"):
            with ui.column().classes("gap-1"):
                ui.label("Health (s)").classes("text-xs text-gray-500")
                page._model_health_cache_ttl_input = page._build_number_input(
                    value=page.settings.model_health_cache_ttl,
                    min_val=1.0,
                    max_val=300.0,
                    step=1.0,
                    tooltip_text="Health check cache lifetime (1-300s)",
                    width="w-16",
                )

            with ui.column().classes("gap-1"):
                ui.label("Installed (s)").classes("text-xs text-gray-500")
                page._model_installed_cache_ttl_input = page._build_number_input(
                    value=page.settings.model_installed_cache_ttl,
                    min_val=1.0,
                    max_val=300.0,
                    step=1.0,
                    tooltip_text="Installed model list cache lifetime (1-300s)",
                    width="w-16",
                )

            with ui.column().classes("gap-1"):
                ui.label("VRAM (s)").classes("text-xs text-gray-500")
                page._model_vram_cache_ttl_input = page._build_number_input(
                    value=page.settings.model_vram_cache_ttl,
                    min_val=1.0,
                    max_val=300.0,
                    step=1.0,
                    tooltip_text="VRAM query cache lifetime (1-300s)",
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

            with ui.column().classes("gap-1"):
                ui.label("Dim. Floor").classes("text-xs text-gray-500")
                page._dimension_minimum_input = page._build_number_input(
                    value=page.settings.world_quality_dimension_minimum,
                    min_val=0.0,
                    max_val=10.0,
                    step=0.5,
                    tooltip_text=(
                        "Per-dimension minimum floor — any dimension below this forces "
                        "refinement even if average meets threshold (0.0 disables)"
                    ),
                    width="w-16",
                )

            with ui.column().classes("gap-1"):
                ui.label("HM Min.").classes("text-xs text-gray-500")
                page._hail_mary_min_attempts_input = page._build_number_input(
                    value=page.settings.world_quality_hail_mary_min_attempts,
                    min_val=1,
                    max_val=100,
                    step=1,
                    tooltip_text=(
                        "Minimum recorded hail-mary attempts before the "
                        "win-rate gate activates (1-100)"
                    ),
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

    Delegates to _advanced_persistence module.

    Args:
        page: The SettingsPage instance.
    """
    from src.ui.pages.settings._advanced_persistence import (
        save_to_settings as _save,
    )

    _save(page)


def refresh_from_settings(page: SettingsPage) -> None:
    """Refresh advanced UI elements from current settings values.

    Delegates to _advanced_persistence module.

    Args:
        page: The SettingsPage instance.
    """
    from src.ui.pages.settings._advanced_persistence import (
        refresh_from_settings as _refresh,
    )

    _refresh(page)

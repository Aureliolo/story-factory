"""Settings page - base class with core functionality."""

import logging
from typing import TYPE_CHECKING, Any

from nicegui import ui

if TYPE_CHECKING:
    from src.services import ServiceContainer
    from src.ui.state import AppState

logger = logging.getLogger(__name__)


class SettingsPageBase:
    """Base class for SettingsPage with core functionality.

    Features:
    - Ollama connection settings
    - Model selection (per-agent or global)
    - Temperature settings
    - Interaction mode
    - Context limits
    - Generation modes (presets for model combinations)
    - Adaptive learning settings (autonomy, triggers, thresholds)
    """

    def __init__(self, state: AppState, services: ServiceContainer):
        """Initialize settings page.

        Args:
            state: Application state.
            services: Service container.
        """
        self.state = state
        self.services = services

        # Settings reference
        self.settings = services.settings

        # Register undo/redo handlers for this page
        self.state.on_undo(self._do_undo)
        self.state.on_redo(self._do_redo)

    def build(self) -> None:
        """Build the settings page UI."""
        with ui.column().classes("w-full gap-6 p-4"):
            ui.label("Settings").classes("text-2xl font-bold")

            # Top row: Connection, Workflow, Memory, Generation Mode, Adaptive Learning
            with ui.element("div").classes("flex flex-wrap gap-4 w-full items-stretch"):
                with ui.element("div").style("flex: 1 1 240px; min-width: 240px;"):
                    self._build_connection_section()

                with ui.element("div").style("flex: 1 1 280px; min-width: 280px;"):
                    self._build_interaction_section()

                with ui.element("div").style("flex: 1 1 280px; min-width: 280px;"):
                    self._build_context_section()

                with ui.element("div").style("flex: 1 1 300px; min-width: 300px;"):
                    self._build_mode_section()

                with ui.element("div").style("flex: 1 1 300px; min-width: 300px;"):
                    self._build_learning_section()

            # Bottom row: Creativity, Model Selection, Story Structure, World Generation, Data Integrity
            with ui.element("div").classes("flex flex-wrap gap-4 w-full items-stretch mt-4"):
                with ui.element("div").style("flex: 1.5 1 450px; min-width: 450px;"):
                    self._build_temperature_section()

                with ui.element("div").style("flex: 1.2 1 380px; min-width: 380px;"):
                    self._build_model_section()

                with ui.element("div").style("flex: 1 1 200px; min-width: 200px;"):
                    self._build_story_structure_section()

                with ui.element("div").style("flex: 1 1 320px; min-width: 320px;"):
                    self._build_world_gen_section()

                with ui.element("div").style("flex: 1 1 280px; min-width: 280px;"):
                    self._build_data_integrity_section()

                with ui.element("div").style("flex: 1 1 400px; min-width: 400px;"):
                    self._build_advanced_llm_section()

                with ui.element("div").style("flex: 1 1 320px; min-width: 320px;"):
                    self._build_relationship_validation_section()

            # Save button
            ui.button(
                "Save Settings",
                on_click=self._save_settings,
                icon="save",
            ).props("color=primary").classes("mt-4")

    def _section_header(self, title: str, icon: str, tooltip: str) -> None:
        """Build a section header with help icon."""
        with ui.row().classes("items-center gap-2 mb-4"):
            ui.icon(icon).classes("text-blue-500")
            ui.label(title).classes("text-lg font-semibold")
            ui.icon("help_outline", size="xs").classes(
                "text-gray-400 dark:text-gray-500 cursor-help"
            ).tooltip(tooltip)

    def _build_number_input(
        self,
        value: int | float,
        min_val: int | float,
        max_val: int | float,
        step: int | float,
        tooltip_text: str,
        width: str = "w-20",
    ) -> ui.number:
        """Build a standardized number input with common styling.

        Args:
            value: Initial value for the input.
            min_val: Minimum allowed value.
            max_val: Maximum allowed value.
            step: Step increment for the input.
            tooltip_text: Tooltip text to display on hover.
            width: Tailwind CSS width class (default: "w-20").

        Returns:
            Configured ui.number element.
        """
        return (
            ui.number(value=value, min=min_val, max=max_val, step=step)
            .props("outlined dense")
            .classes(width)
            .tooltip(tooltip_text)
        )

    def _save_settings(self) -> None:
        """
        Persist current UI-configured settings to the application's settings store and record an undo snapshot.

        Read values from the page's UI controls and apply them to the settings object.
        Validate and save the updated settings.
        Record an undo action that contains before and after snapshots for undo and redo.
        On success, display a positive notification.
        On validation error or any other failure, log the issue and display a negative notification.
        """
        from src.ui.state import ActionType, UndoAction

        try:
            # Capture old state for undo
            old_snapshot = self._capture_settings_snapshot()

            # Update settings from UI
            self.settings.ollama_url = self._ollama_url_input.value
            self.settings.default_model = self._default_model_select.value
            self.settings.use_per_agent_models = self._use_per_agent.value

            # Per-agent models
            for role, select in self._agent_model_selects.items():
                self.settings.agent_models[role] = select.value

            # Temperatures
            for role, slider in self._temp_sliders.items():
                self.settings.agent_temperatures[role] = slider.value

            # Interaction
            self.settings.interaction_mode = self._interaction_mode_select.value
            self.settings.chapters_between_checkpoints = int(self._checkpoint_input.value)
            self.settings.max_revision_iterations = int(self._revision_input.value)

            # Context
            self.settings.context_size = int(self._context_size_input.value)
            self.settings.max_tokens = int(self._max_tokens_input.value)
            self.settings.previous_chapter_context_chars = int(self._prev_chapter_chars.value)
            self.settings.chapter_analysis_chars = int(self._chapter_analysis_chars.value)
            self.settings.full_text_preview_chars = int(self._full_text_preview_chars.value)

            # Generation mode
            self.settings.use_mode_system = self._use_mode_system.value
            if hasattr(self, "_mode_select"):
                self.settings.current_mode = self._mode_select.value
            if hasattr(self, "_vram_strategy_select"):
                self.settings.vram_strategy = self._vram_strategy_select.value

            # Learning settings
            self.settings.learning_autonomy = self._autonomy_select.value

            # Collect enabled triggers
            enabled_triggers = []
            for trigger_value, checkbox in self._trigger_checkboxes.items():
                if checkbox.value:
                    enabled_triggers.append(trigger_value)
            if not enabled_triggers:
                enabled_triggers = ["off"]
            self.settings.learning_triggers = enabled_triggers

            self.settings.learning_periodic_interval = int(self._periodic_interval.value)
            self.settings.learning_min_samples = int(self._min_samples.value)
            self.settings.learning_confidence_threshold = self._confidence_slider.value

            # World generation settings
            for key, (min_input, max_input) in self._world_gen_inputs.items():
                min_val = int(min_input.value)
                max_val = int(max_input.value)
                # Ensure min <= max
                if min_val > max_val:
                    max_val = min_val
                setattr(self.settings, f"world_gen_{key}_min", min_val)
                setattr(self.settings, f"world_gen_{key}_max", max_val)

            # Quality refinement settings
            if hasattr(self, "_quality_threshold_input"):
                self.settings.world_quality_threshold = float(self._quality_threshold_input.value)
            if hasattr(self, "_quality_max_iterations_input"):
                self.settings.world_quality_max_iterations = int(
                    self._quality_max_iterations_input.value
                )
            if hasattr(self, "_quality_patience_input"):
                self.settings.world_quality_early_stopping_patience = int(
                    self._quality_patience_input.value
                )

            # Story structure settings (chapter counts)
            if hasattr(self, "_chapter_inputs"):
                for key, input_widget in self._chapter_inputs.items():
                    setattr(self.settings, f"chapters_{key}", int(input_widget.value))

            # Data integrity settings
            if hasattr(self, "_entity_version_retention_input"):
                self.settings.entity_version_retention = int(
                    self._entity_version_retention_input.value
                )
            if hasattr(self, "_backup_verify_on_restore_switch"):
                self.settings.backup_verify_on_restore = self._backup_verify_on_restore_switch.value

            # Advanced LLM settings (WP1/WP2) - use mapping to reduce repetition
            advanced_llm_settings_map = [
                ("_circuit_breaker_enabled_switch", "circuit_breaker_enabled", None),
                (
                    "_circuit_breaker_failure_threshold_input",
                    "circuit_breaker_failure_threshold",
                    int,
                ),
                (
                    "_circuit_breaker_success_threshold_input",
                    "circuit_breaker_success_threshold",
                    int,
                ),
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
                if hasattr(self, ui_attr):
                    ui_element = getattr(self, ui_attr)
                    value = ui_element.value
                    if type_conv:
                        value = type_conv(value)
                    setattr(self.settings, setting_attr, value)
                    logger.debug(f"Updated {setting_attr} to {value}")

            # Relationship validation settings
            if hasattr(self, "_relationship_validation_switch"):
                self.settings.relationship_validation_enabled = (
                    self._relationship_validation_switch.value
                )
            if hasattr(self, "_orphan_detection_switch"):
                self.settings.orphan_detection_enabled = self._orphan_detection_switch.value
            if hasattr(self, "_circular_detection_switch"):
                self.settings.circular_detection_enabled = self._circular_detection_switch.value
            if hasattr(self, "_fuzzy_threshold_input"):
                self.settings.fuzzy_match_threshold = float(self._fuzzy_threshold_input.value)
            if hasattr(self, "_max_relationships_input"):
                self.settings.max_relationships_per_entity = int(
                    self._max_relationships_input.value
                )
            if hasattr(self, "_circular_types_input"):
                # Parse comma-separated types, strip whitespace, filter empty
                types_str = self._circular_types_input.value or ""
                types_list = [t.strip() for t in types_str.split(",") if t.strip()]
                self.settings.circular_relationship_types = types_list

            # Validate and save first - only record undo if successful
            self.settings.validate()
            self.settings.save()

            # Capture new state for redo AFTER successful save
            new_snapshot = self._capture_settings_snapshot()

            # Record undo action only after successful validation and save
            self.state.record_action(
                UndoAction(
                    action_type=ActionType.UPDATE_SETTINGS,
                    data=new_snapshot,
                    inverse_data=old_snapshot,
                )
            )

            ui.notify("Settings saved!", type="positive")

        except ValueError as e:
            logger.warning(f"Invalid setting value: {e}")
            ui.notify(f"Invalid setting: {e}", type="negative")
        except Exception as e:
            logger.exception("Failed to save settings")
            ui.notify(f"Error saving: {e}", type="negative")

    def _capture_settings_snapshot(self) -> dict[str, Any]:
        """
        Create a serializable snapshot of current settings suitable for undo/redo operations.

        Returns:
            snapshot (dict[str, Any]): A dictionary snapshot of settings including:
                - Core connection and model: `ollama_url`, `default_model`, `use_per_agent_models`, `agent_models`, `agent_temperatures`
                - Interaction and workflow: `interaction_mode`, `chapters_between_checkpoints`, `max_revision_iterations`
                - Context and generation: `context_size`, `max_tokens`, `previous_chapter_context_chars`, `chapter_analysis_chars`, `full_text_preview_chars`
                - Mode and VRAM: `use_mode_system`, `current_mode`, `vram_strategy`
                - Adaptive learning: `learning_autonomy`, `learning_triggers`, `learning_periodic_interval`, `learning_min_samples`, `learning_confidence_threshold`
                - World generation counts: `world_gen_*_min` and `world_gen_*_max` for `characters`, `locations`, `factions`, `items`, `concepts`, and `relationships`
                - Quality refinement: `world_quality_threshold`, `world_quality_max_iterations`, `world_quality_early_stopping_patience`
                - Data integrity: `entity_version_retention`, `backup_verify_on_restore`
                - Advanced LLM (WP1/WP2): `circuit_breaker_enabled`, `circuit_breaker_failure_threshold`, `circuit_breaker_success_threshold`, `circuit_breaker_timeout`, `retry_temp_increase`, `retry_simplify_on_attempt`, `semantic_duplicate_enabled`, `semantic_duplicate_threshold`, `embedding_model`, `world_quality_refinement_temp_start`, `world_quality_refinement_temp_end`, `world_quality_refinement_temp_decay`, `world_quality_early_stopping_min_iterations`, `world_quality_early_stopping_variance_tolerance`
        """
        snapshot = {
            "ollama_url": self.settings.ollama_url,
            "default_model": self.settings.default_model,
            "use_per_agent_models": self.settings.use_per_agent_models,
            "agent_models": dict(self.settings.agent_models),
            "agent_temperatures": dict(self.settings.agent_temperatures),
            "interaction_mode": self.settings.interaction_mode,
            "chapters_between_checkpoints": self.settings.chapters_between_checkpoints,
            "max_revision_iterations": self.settings.max_revision_iterations,
            "context_size": self.settings.context_size,
            "max_tokens": self.settings.max_tokens,
            "previous_chapter_context_chars": self.settings.previous_chapter_context_chars,
            "chapter_analysis_chars": self.settings.chapter_analysis_chars,
            "full_text_preview_chars": self.settings.full_text_preview_chars,
            "use_mode_system": self.settings.use_mode_system,
            "current_mode": self.settings.current_mode,
            "vram_strategy": self.settings.vram_strategy,
            "learning_autonomy": self.settings.learning_autonomy,
            "learning_triggers": list(self.settings.learning_triggers),
            "learning_periodic_interval": self.settings.learning_periodic_interval,
            "learning_min_samples": self.settings.learning_min_samples,
            "learning_confidence_threshold": self.settings.learning_confidence_threshold,
            # World generation
            "world_gen_characters_min": self.settings.world_gen_characters_min,
            "world_gen_characters_max": self.settings.world_gen_characters_max,
            "world_gen_locations_min": self.settings.world_gen_locations_min,
            "world_gen_locations_max": self.settings.world_gen_locations_max,
            "world_gen_factions_min": self.settings.world_gen_factions_min,
            "world_gen_factions_max": self.settings.world_gen_factions_max,
            "world_gen_items_min": self.settings.world_gen_items_min,
            "world_gen_items_max": self.settings.world_gen_items_max,
            "world_gen_concepts_min": self.settings.world_gen_concepts_min,
            "world_gen_concepts_max": self.settings.world_gen_concepts_max,
            "world_gen_relationships_min": self.settings.world_gen_relationships_min,
            "world_gen_relationships_max": self.settings.world_gen_relationships_max,
            # Quality refinement
            "world_quality_threshold": self.settings.world_quality_threshold,
            "world_quality_max_iterations": self.settings.world_quality_max_iterations,
            "world_quality_early_stopping_patience": self.settings.world_quality_early_stopping_patience,
            # Data integrity
            "entity_version_retention": self.settings.entity_version_retention,
            "backup_verify_on_restore": self.settings.backup_verify_on_restore,
            # Relationship validation
            "relationship_validation_enabled": self.settings.relationship_validation_enabled,
            "orphan_detection_enabled": self.settings.orphan_detection_enabled,
            "circular_detection_enabled": self.settings.circular_detection_enabled,
            "fuzzy_match_threshold": self.settings.fuzzy_match_threshold,
            "max_relationships_per_entity": self.settings.max_relationships_per_entity,
            "circular_relationship_types": self.settings.circular_relationship_types.copy(),
        }

        # Advanced LLM settings (WP1/WP2) - add using key iteration
        advanced_llm_keys = [
            "circuit_breaker_enabled",
            "circuit_breaker_failure_threshold",
            "circuit_breaker_success_threshold",
            "circuit_breaker_timeout",
            "retry_temp_increase",
            "retry_simplify_on_attempt",
            "semantic_duplicate_enabled",
            "semantic_duplicate_threshold",
            "embedding_model",
            "world_quality_refinement_temp_start",
            "world_quality_refinement_temp_end",
            "world_quality_refinement_temp_decay",
            "world_quality_early_stopping_min_iterations",
            "world_quality_early_stopping_variance_tolerance",
        ]
        for key in advanced_llm_keys:
            snapshot[key] = getattr(self.settings, key)

        return snapshot

    def _restore_settings_snapshot(self, snapshot: dict[str, Any]) -> None:
        """
        Restore the SettingsPage state from a snapshot and persist the restored values.

        Parameters:
            snapshot (dict[str, Any]): Snapshot containing saved settings values.

                Required keys:
                - Core settings: `ollama_url`, `default_model`, `use_per_agent_models`,
                  `agent_models`, `agent_temperatures`
                - Workflow: `interaction_mode`, `chapters_between_checkpoints`,
                  `max_revision_iterations`
                - Context: `context_size`, `max_tokens`, `previous_chapter_context_chars`,
                  `chapter_analysis_chars`
                - Mode: `use_mode_system`, `current_mode`
                - Learning: `learning_autonomy`, `learning_triggers`,
                  `learning_periodic_interval`, `learning_min_samples`,
                  `learning_confidence_threshold`
                - Data integrity: `entity_version_retention`, `backup_verify_on_restore`

                Optional keys:
                - `full_text_preview_chars`, `vram_strategy`
                - World generation: `world_gen_*_min` / `world_gen_*_max` pairs for
                  `characters`, `locations`, `factions`, `items`, `concepts`,
                  `relationships`
                - Quality refinement: `world_quality_threshold`, `world_quality_max_iterations`,
                  `world_quality_early_stopping_patience`
                - Advanced LLM (WP1/WP2): `circuit_breaker_enabled`, `circuit_breaker_failure_threshold`,
                  `circuit_breaker_success_threshold`, `circuit_breaker_timeout`, `retry_temp_increase`,
                  `retry_simplify_on_attempt`, `semantic_duplicate_enabled`, `semantic_duplicate_threshold`,
                  `embedding_model`, `world_quality_refinement_temp_start`, `world_quality_refinement_temp_end`,
                  `world_quality_refinement_temp_decay`, `world_quality_early_stopping_min_iterations`,
                  `world_quality_early_stopping_variance_tolerance`

        Behavior:
            Applies values from `snapshot` to the persistent settings, saves the settings, updates UI controls
            to reflect the restored values, and shows an informational notification indicating the restore completed.
        """
        self.settings.ollama_url = snapshot["ollama_url"]
        self.settings.default_model = snapshot["default_model"]
        self.settings.use_per_agent_models = snapshot["use_per_agent_models"]
        self.settings.agent_models = dict(snapshot["agent_models"])
        self.settings.agent_temperatures = dict(snapshot["agent_temperatures"])
        self.settings.interaction_mode = snapshot["interaction_mode"]
        self.settings.chapters_between_checkpoints = snapshot["chapters_between_checkpoints"]
        self.settings.max_revision_iterations = snapshot["max_revision_iterations"]
        self.settings.context_size = snapshot["context_size"]
        self.settings.max_tokens = snapshot["max_tokens"]
        self.settings.previous_chapter_context_chars = snapshot["previous_chapter_context_chars"]
        self.settings.chapter_analysis_chars = snapshot["chapter_analysis_chars"]
        if "full_text_preview_chars" in snapshot:
            self.settings.full_text_preview_chars = snapshot["full_text_preview_chars"]
        self.settings.use_mode_system = snapshot["use_mode_system"]
        self.settings.current_mode = snapshot["current_mode"]
        if "vram_strategy" in snapshot:
            self.settings.vram_strategy = snapshot["vram_strategy"]
        self.settings.learning_autonomy = snapshot["learning_autonomy"]
        self.settings.learning_triggers = list(snapshot["learning_triggers"])
        self.settings.learning_periodic_interval = snapshot["learning_periodic_interval"]
        self.settings.learning_min_samples = snapshot["learning_min_samples"]
        self.settings.learning_confidence_threshold = snapshot["learning_confidence_threshold"]

        # World generation (with backward compatibility for old snapshots)
        if "world_gen_characters_min" in snapshot:
            self.settings.world_gen_characters_min = snapshot["world_gen_characters_min"]
            self.settings.world_gen_characters_max = snapshot["world_gen_characters_max"]
            self.settings.world_gen_locations_min = snapshot["world_gen_locations_min"]
            self.settings.world_gen_locations_max = snapshot["world_gen_locations_max"]
            self.settings.world_gen_factions_min = snapshot["world_gen_factions_min"]
            self.settings.world_gen_factions_max = snapshot["world_gen_factions_max"]
            self.settings.world_gen_items_min = snapshot["world_gen_items_min"]
            self.settings.world_gen_items_max = snapshot["world_gen_items_max"]
            self.settings.world_gen_concepts_min = snapshot["world_gen_concepts_min"]
            self.settings.world_gen_concepts_max = snapshot["world_gen_concepts_max"]
            self.settings.world_gen_relationships_min = snapshot["world_gen_relationships_min"]
            self.settings.world_gen_relationships_max = snapshot["world_gen_relationships_max"]

        # Quality refinement (with backward compatibility for old snapshots)
        if "world_quality_threshold" in snapshot:
            self.settings.world_quality_threshold = snapshot["world_quality_threshold"]
        if "world_quality_max_iterations" in snapshot:
            self.settings.world_quality_max_iterations = snapshot["world_quality_max_iterations"]
        if "world_quality_early_stopping_patience" in snapshot:
            self.settings.world_quality_early_stopping_patience = snapshot[
                "world_quality_early_stopping_patience"
            ]

        # Data integrity (with backward compatibility for old snapshots)
        if "entity_version_retention" in snapshot:
            self.settings.entity_version_retention = snapshot["entity_version_retention"]
        if "backup_verify_on_restore" in snapshot:
            self.settings.backup_verify_on_restore = snapshot["backup_verify_on_restore"]

        # Advanced LLM settings (with backward compatibility for old snapshots)
        advanced_llm_keys = [
            "circuit_breaker_enabled",
            "circuit_breaker_failure_threshold",
            "circuit_breaker_success_threshold",
            "circuit_breaker_timeout",
            "retry_temp_increase",
            "retry_simplify_on_attempt",
            "semantic_duplicate_enabled",
            "semantic_duplicate_threshold",
            "embedding_model",
            "world_quality_refinement_temp_start",
            "world_quality_refinement_temp_end",
            "world_quality_refinement_temp_decay",
            "world_quality_early_stopping_min_iterations",
            "world_quality_early_stopping_variance_tolerance",
        ]
        for key in advanced_llm_keys:
            if key in snapshot:
                setattr(self.settings, key, snapshot[key])

        # Relationship validation (with backward compatibility for old snapshots)
        if "relationship_validation_enabled" in snapshot:
            self.settings.relationship_validation_enabled = snapshot[
                "relationship_validation_enabled"
            ]
        if "orphan_detection_enabled" in snapshot:
            self.settings.orphan_detection_enabled = snapshot["orphan_detection_enabled"]
        if "circular_detection_enabled" in snapshot:
            self.settings.circular_detection_enabled = snapshot["circular_detection_enabled"]
        if "fuzzy_match_threshold" in snapshot:
            self.settings.fuzzy_match_threshold = snapshot["fuzzy_match_threshold"]
        if "max_relationships_per_entity" in snapshot:
            self.settings.max_relationships_per_entity = snapshot["max_relationships_per_entity"]
        if "circular_relationship_types" in snapshot:
            self.settings.circular_relationship_types = snapshot["circular_relationship_types"]

        # Save changes
        self.settings.save()

        # Update UI elements to reflect restored values
        self._refresh_ui_from_settings()

        ui.notify("Settings restored", type="info")

    def _refresh_ui_from_settings(self) -> None:
        """Refresh all UI input elements from current settings values."""
        logger.debug("Refreshing UI elements from settings")

        # Core settings
        if hasattr(self, "_ollama_url_input") and self._ollama_url_input:
            self._ollama_url_input.value = self.settings.ollama_url
        if hasattr(self, "_default_model_select") and self._default_model_select:
            self._default_model_select.value = self.settings.default_model
        if hasattr(self, "_use_per_agent") and self._use_per_agent:
            self._use_per_agent.value = self.settings.use_per_agent_models

        # Per-agent model selects
        if hasattr(self, "_agent_model_selects"):
            for role, select in self._agent_model_selects.items():
                if select and role in self.settings.agent_models:
                    select.value = self.settings.agent_models[role]

        # Temperature sliders
        if hasattr(self, "_temp_sliders"):
            for role, slider in self._temp_sliders.items():
                if slider and role in self.settings.agent_temperatures:
                    slider.value = self.settings.agent_temperatures[role]

        # Workflow settings
        if hasattr(self, "_interaction_mode_select") and self._interaction_mode_select:
            self._interaction_mode_select.value = self.settings.interaction_mode
        if hasattr(self, "_checkpoint_input") and self._checkpoint_input:
            self._checkpoint_input.value = self.settings.chapters_between_checkpoints
        if hasattr(self, "_revision_input") and self._revision_input:
            self._revision_input.value = self.settings.max_revision_iterations

        # Context settings
        if hasattr(self, "_context_size_input") and self._context_size_input:
            self._context_size_input.value = self.settings.context_size
        if hasattr(self, "_max_tokens_input") and self._max_tokens_input:
            self._max_tokens_input.value = self.settings.max_tokens
        if hasattr(self, "_full_text_preview_chars") and self._full_text_preview_chars:
            self._full_text_preview_chars.value = self.settings.full_text_preview_chars

        # Mode settings
        if hasattr(self, "_mode_select") and self._mode_select:
            self._mode_select.value = self.settings.current_mode
        if hasattr(self, "_vram_strategy_select") and self._vram_strategy_select:
            self._vram_strategy_select.value = self.settings.vram_strategy

        # Learning settings
        if hasattr(self, "_autonomy_select") and self._autonomy_select:
            self._autonomy_select.value = self.settings.learning_autonomy
        if hasattr(self, "_confidence_slider") and self._confidence_slider:
            self._confidence_slider.value = self.settings.learning_confidence_threshold
        if hasattr(self, "_periodic_interval") and self._periodic_interval:
            self._periodic_interval.value = self.settings.learning_periodic_interval
        if hasattr(self, "_min_samples") and self._min_samples:
            self._min_samples.value = self.settings.learning_min_samples

        # World generation settings
        if hasattr(self, "_world_gen_inputs"):
            for key, (min_input, max_input) in self._world_gen_inputs.items():
                min_attr = f"world_gen_{key}_min"
                max_attr = f"world_gen_{key}_max"
                if hasattr(self.settings, min_attr):
                    min_input.value = getattr(self.settings, min_attr)
                    max_input.value = getattr(self.settings, max_attr)

        # Quality refinement settings
        if hasattr(self, "_quality_threshold_input") and self._quality_threshold_input:
            self._quality_threshold_input.value = self.settings.world_quality_threshold
        if hasattr(self, "_quality_max_iterations_input") and self._quality_max_iterations_input:
            self._quality_max_iterations_input.value = self.settings.world_quality_max_iterations
        if hasattr(self, "_quality_patience_input") and self._quality_patience_input:
            self._quality_patience_input.value = self.settings.world_quality_early_stopping_patience

        # Data integrity settings
        if (
            hasattr(self, "_entity_version_retention_input")
            and self._entity_version_retention_input
        ):
            self._entity_version_retention_input.value = self.settings.entity_version_retention
        if (
            hasattr(self, "_backup_verify_on_restore_switch")
            and self._backup_verify_on_restore_switch
        ):
            self._backup_verify_on_restore_switch.value = self.settings.backup_verify_on_restore

        # Advanced LLM settings (WP1/WP2) - use mapping to reduce repetition
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
            if hasattr(self, ui_attr) and getattr(self, ui_attr):
                getattr(self, ui_attr).value = getattr(self.settings, setting_attr)

        # Relationship validation settings
        if (
            hasattr(self, "_relationship_validation_switch")
            and self._relationship_validation_switch
        ):
            self._relationship_validation_switch.value = (
                self.settings.relationship_validation_enabled
            )
        if hasattr(self, "_orphan_detection_switch") and self._orphan_detection_switch:
            self._orphan_detection_switch.value = self.settings.orphan_detection_enabled
        if hasattr(self, "_circular_detection_switch") and self._circular_detection_switch:
            self._circular_detection_switch.value = self.settings.circular_detection_enabled
        if hasattr(self, "_fuzzy_threshold_input") and self._fuzzy_threshold_input:
            self._fuzzy_threshold_input.value = self.settings.fuzzy_match_threshold
        if hasattr(self, "_max_relationships_input") and self._max_relationships_input:
            self._max_relationships_input.value = self.settings.max_relationships_per_entity
        if hasattr(self, "_circular_types_input") and self._circular_types_input:
            self._circular_types_input.value = ", ".join(self.settings.circular_relationship_types)

    def _do_undo(self) -> None:
        """Handle undo for settings changes."""
        from src.ui.state import ActionType

        action = self.state.undo()
        if not action:
            return

        try:
            if action.action_type == ActionType.UPDATE_SETTINGS:
                # Restore old settings
                self._restore_settings_snapshot(action.inverse_data)
                logger.debug("Undone settings change")
        except Exception as e:
            logger.exception("Undo failed for settings")
            ui.notify(f"Undo failed: {e}", type="negative")

    def _do_redo(self) -> None:
        """Handle redo for settings changes."""
        from src.ui.state import ActionType

        action = self.state.redo()
        if not action:
            return

        try:
            if action.action_type == ActionType.UPDATE_SETTINGS:
                # Restore new settings
                self._restore_settings_snapshot(action.data)
                logger.debug("Redone settings change")
        except Exception as e:
            logger.exception("Redo failed for settings")
            ui.notify(f"Redo failed: {e}", type="negative")

    # Mixin methods will be added by subclasses
    def _build_connection_section(self) -> None:
        """Build Ollama connection settings. Implemented by ConnectionMixin."""
        raise NotImplementedError

    def _build_model_section(self) -> None:
        """Build model selection settings. Implemented by ModelsMixin."""
        raise NotImplementedError

    def _build_temperature_section(self) -> None:
        """Build temperature settings. Implemented by ModelsMixin."""
        raise NotImplementedError

    def _build_interaction_section(self) -> None:
        """Build interaction mode settings. Implemented by InteractionMixin."""
        raise NotImplementedError

    def _build_context_section(self) -> None:
        """Build context settings. Implemented by InteractionMixin."""
        raise NotImplementedError

    def _build_mode_section(self) -> None:
        """Build generation mode settings. Implemented by ModesMixin."""
        raise NotImplementedError

    def _build_learning_section(self) -> None:
        """Build learning/tuning settings. Implemented by ModesMixin."""
        raise NotImplementedError

    def _build_world_gen_section(self) -> None:
        """Build world generation settings. Implemented by WorldGenMixin."""
        raise NotImplementedError

    def _build_story_structure_section(self) -> None:
        """Build story structure settings. Implemented by WorldGenMixin."""
        raise NotImplementedError

    def _build_data_integrity_section(self) -> None:
        """Build data integrity settings. Implemented by AdvancedMixin."""
        raise NotImplementedError

    def _build_advanced_llm_section(self) -> None:
        """Build advanced LLM settings. Implemented by AdvancedMixin."""
        raise NotImplementedError

    def _build_relationship_validation_section(self) -> None:
        """Build relationship validation settings. Implemented by AdvancedMixin."""
        raise NotImplementedError

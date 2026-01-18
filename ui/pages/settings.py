"""Settings page - application configuration."""

import logging
from typing import Any

from nicegui import ui

from memory.mode_models import (
    PRESET_MODES,
    AutonomyLevel,
    LearningTrigger,
    VramStrategy,
)
from services import ServiceContainer
from settings import AGENT_ROLES
from ui.state import AppState
from utils import extract_model_name

logger = logging.getLogger(__name__)


class SettingsPage:
    """Settings page for application configuration.

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

            # All cards in flex container - same height, different widths
            with ui.element("div").classes("flex flex-wrap gap-4 w-full items-stretch"):
                with ui.element("div").style("flex: 1 1 260px; min-width: 260px;"):
                    self._build_connection_section()

                with ui.element("div").style("flex: 1 1 300px; min-width: 300px;"):
                    self._build_interaction_section()

                with ui.element("div").style("flex: 1 1 300px; min-width: 300px;"):
                    self._build_context_section()

                with ui.element("div").style("flex: 1.2 1 380px; min-width: 380px;"):
                    self._build_model_section()

                with ui.element("div").style("flex: 2 1 550px; min-width: 550px;"):
                    self._build_temperature_section()

                with ui.element("div").style("flex: 1.5 1 400px; min-width: 400px;"):
                    self._build_mode_section()

                with ui.element("div").style("flex: 1.5 1 400px; min-width: 400px;"):
                    self._build_learning_section()

                with ui.element("div").style("flex: 1.5 1 450px; min-width: 450px;"):
                    self._build_world_gen_section()

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

    def _build_connection_section(self) -> None:
        """Build Ollama connection settings."""
        with ui.card().classes("w-full h-full"):
            self._section_header(
                "Connection",
                "link",
                "Configure how Story Factory connects to your local Ollama instance",
            )

            self._ollama_url_input = (
                ui.input(
                    label="Ollama URL",
                    value=self.settings.ollama_url,
                )
                .classes("w-full")
                .props("outlined dense")
                .tooltip("Usually http://localhost:11434 for local Ollama")
            )

            with ui.row().classes("w-full items-center gap-4 mt-3"):
                ui.button(
                    "Test",
                    on_click=self._test_connection,
                    icon="play_arrow",
                ).props("outline dense")

                # Connection status
                health = self.services.model.check_health()
                if health.is_healthy:
                    with ui.row().classes("items-center gap-1"):
                        ui.icon("check_circle", size="sm").classes("text-green-500")
                        ui.label(f"{health.available_vram} GB VRAM").classes(
                            "text-sm text-green-600 dark:text-green-400"
                        )
                else:
                    with ui.row().classes("items-center gap-1"):
                        ui.icon("error", size="sm").classes("text-red-500")
                        ui.label("Offline").classes(
                            "text-sm text-red-600 dark:text-red-400"
                        ).tooltip(health.message)

    def _build_model_section(self) -> None:
        """Build model selection settings."""
        with ui.card().classes("w-full h-full"):
            self._section_header(
                "Model Selection",
                "smart_toy",
                "Choose which AI models to use. 'Auto' picks the best available model.",
            )

            # Get installed models
            installed_models = self.services.model.list_installed()
            model_options = {"auto": "Auto-select (recommended)"} | {m: m for m in installed_models}

            # Default model (fall back to "auto" if saved model not installed)
            default_model_value = self.settings.default_model
            if default_model_value not in model_options:
                default_model_value = "auto"

            with ui.row().classes("w-full items-end gap-4"):
                self._default_model_select = (
                    ui.select(
                        label="Default Model",
                        options=model_options,
                        value=default_model_value,
                    )
                    .classes("flex-grow")
                    .props("outlined")
                    .tooltip("The model used for all agents unless overridden below")
                )

                # Per-agent toggle
                self._use_per_agent = ui.switch(
                    "Per-agent",
                    value=self.settings.use_per_agent_models,
                ).tooltip("Use different models for each agent type")

            # Per-agent model selects
            with (
                ui.element("div")
                .classes("w-full mt-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg")
                .bind_visibility_from(self._use_per_agent, "value")
            ):
                ui.label("Per-Agent Model Overrides").classes(
                    "font-medium text-sm text-gray-600 dark:text-gray-400 mb-3"
                )

                self._agent_model_selects = {}
                with ui.element("div").classes("grid grid-cols-1 md:grid-cols-2 gap-3"):
                    for role, info in AGENT_ROLES.items():
                        # Fall back to "auto" if saved model not installed
                        agent_model_value = self.settings.agent_models.get(role, "auto")
                        if agent_model_value not in model_options:
                            agent_model_value = "auto"

                        self._agent_model_selects[role] = (
                            ui.select(
                                label=info["name"],
                                options=model_options,
                                value=agent_model_value,
                            )
                            .classes("w-full")
                            .props("outlined dense")
                            .tooltip(info["description"])
                        )

    def _build_temperature_section(self) -> None:
        """Build temperature settings."""
        with ui.card().classes("w-full h-full"):
            self._section_header(
                "Creativity (Temperature)",
                "thermostat",
                "Controls randomness in AI responses. Higher = more creative/varied, Lower = more focused/consistent",
            )

            # Temperature presets legend
            with ui.row().classes("w-full justify-center gap-6 mb-4 text-xs"):
                with ui.row().classes("items-center gap-1"):
                    ui.element("div").classes("w-3 h-3 rounded bg-blue-500")
                    ui.label("Focused (0.3)").classes("text-gray-500 dark:text-gray-400")
                with ui.row().classes("items-center gap-1"):
                    ui.element("div").classes("w-3 h-3 rounded bg-green-500")
                    ui.label("Balanced (0.7)").classes("text-gray-500 dark:text-gray-400")
                with ui.row().classes("items-center gap-1"):
                    ui.element("div").classes("w-3 h-3 rounded bg-orange-500")
                    ui.label("Creative (1.2+)").classes("text-gray-500 dark:text-gray-400")

            self._temp_sliders: dict[str, Any] = {}
            self._temp_labels: dict[str, Any] = {}
            # 3-column grid for temperature sliders
            with ui.element("div").classes("grid grid-cols-2 lg:grid-cols-3 gap-3"):
                for role, info in AGENT_ROLES.items():
                    default_temp = self.settings.get_temperature_for_agent(role)
                    with (
                        ui.card()
                        .classes("p-4 bg-gray-50 dark:bg-gray-800")
                        .tooltip(info["description"])
                    ):
                        with ui.row().classes("w-full items-center justify-between mb-3"):
                            ui.label(info["name"]).classes("font-medium text-sm")
                            self._temp_labels[role] = ui.label(f"{default_temp:.1f}").classes(
                                "text-sm font-mono bg-gray-200 dark:bg-gray-700 px-2 py-0.5 rounded"
                            )

                        # Slider without floating label (value shown in corner)
                        slider = ui.slider(
                            min=0.0,
                            max=2.0,
                            step=0.1,
                            value=default_temp,
                        ).classes("w-full")
                        self._temp_sliders[role] = slider

                        # Bind the label to the slider value
                        self._temp_labels[role].bind_text_from(
                            slider, "value", backward=lambda v: f"{v:.1f}"
                        )

    def _build_interaction_section(self) -> None:
        """Build interaction mode settings."""
        with ui.card().classes("w-full h-full"):
            self._section_header(
                "Workflow",
                "tune",
                "Control how much the AI asks for your input during story generation",
            )

            with ui.column().classes("w-full gap-3"):
                self._interaction_mode_select = (
                    ui.select(
                        label="Interaction Mode",
                        options={
                            "minimal": "Minimal - Fewest interruptions",
                            "checkpoint": "Checkpoint - Review every N chapters",
                            "interactive": "Interactive - More control points",
                            "collaborative": "Collaborative - Maximum interaction",
                        },
                        value=self.settings.interaction_mode,
                    )
                    .classes("w-full")
                    .props("outlined dense")
                    .tooltip("How often the AI pauses for your feedback")
                )

                self._checkpoint_input = (
                    ui.number(
                        label="Chapters between checkpoints",
                        value=self.settings.chapters_between_checkpoints,
                        min=1,
                        max=20,
                    )
                    .classes("w-full")
                    .props("outlined dense")
                    .tooltip("How many chapters to write before pausing for review")
                )

                self._revision_input = (
                    ui.number(
                        label="Max revision iterations",
                        value=self.settings.max_revision_iterations,
                        min=0,
                        max=10,
                    )
                    .classes("w-full")
                    .props("outlined dense")
                    .tooltip("Max edit passes per chapter (0 = unlimited)")
                )

    def _build_context_section(self) -> None:
        """Build context limit settings."""
        with ui.card().classes("w-full h-full"):
            self._section_header(
                "Memory & Context",
                "memory",
                "Control how much information the AI remembers. Higher values use more VRAM but improve coherence.",
            )

            # Single column with proper spacing for readable labels
            with ui.column().classes("w-full gap-3"):
                self._context_size_input = (
                    ui.number(
                        label="Context window (tokens)",
                        value=self.settings.context_size,
                        min=1024,
                        max=128000,
                        step=1024,
                    )
                    .classes("w-full")
                    .props("outlined dense")
                    .tooltip("Total tokens the AI can 'see' at once (default: 32768)")
                )

                self._max_tokens_input = (
                    ui.number(
                        label="Max output (tokens)",
                        value=self.settings.max_tokens,
                        min=256,
                        max=32000,
                        step=256,
                    )
                    .classes("w-full")
                    .props("outlined dense")
                    .tooltip("Maximum tokens per AI response (default: 4096)")
                )

                self._prev_chapter_chars = (
                    ui.number(
                        label="Chapter memory (chars)",
                        value=self.settings.previous_chapter_context_chars,
                        min=500,
                        max=10000,
                    )
                    .classes("w-full")
                    .props("outlined dense")
                    .tooltip("Characters from previous chapter to include for continuity")
                )

                self._chapter_analysis_chars = (
                    ui.number(
                        label="Analysis context (chars)",
                        value=self.settings.chapter_analysis_chars,
                        min=1000,
                        max=20000,
                    )
                    .classes("w-full")
                    .props("outlined dense")
                    .tooltip("Characters to analyze when reviewing chapter quality")
                )

    def _build_mode_section(self) -> None:
        """Build generation mode settings."""
        with ui.card().classes("w-full h-full"):
            self._section_header(
                "Generation Mode",
                "tune",
                "Select model combinations optimized for different use cases",
            )

            # Mode options from presets
            mode_options = {
                mode_id: f"{mode.name} - {mode.description}"
                for mode_id, mode in PRESET_MODES.items()
            }

            with ui.column().classes("w-full gap-3"):
                # Use mode system toggle
                self._use_mode_system = ui.switch(
                    "Use Generation Modes",
                    value=self.settings.use_mode_system,
                ).tooltip("When enabled, uses preset model combinations per agent")

                # Mode selector
                with (
                    ui.element("div")
                    .classes("w-full")
                    .bind_visibility_from(self._use_mode_system, "value")
                ):
                    self._mode_select = (
                        ui.select(
                            label="Active Mode",
                            options=mode_options,
                            value=self.settings.current_mode,
                        )
                        .classes("w-full")
                        .props("outlined dense")
                        .tooltip("Choose a preset mode for model assignments")
                    )

                    # Show current mode details
                    current_mode = PRESET_MODES.get(self.settings.current_mode)
                    if current_mode:
                        with ui.expansion("Mode Details", icon="info").classes("w-full mt-2"):
                            with ui.column().classes("gap-1 text-sm"):
                                for role, model in current_mode.agent_models.items():
                                    temp = current_mode.agent_temperatures.get(
                                        role, self.settings.get_temperature_for_agent(role)
                                    )
                                    with ui.row().classes("items-center gap-2"):
                                        ui.label(f"{role.title()}:").classes("font-medium w-24")
                                        ui.label(extract_model_name(model)).classes(
                                            "text-gray-600 dark:text-gray-400"
                                        )
                                        ui.label(f"({temp})").classes("text-xs text-gray-500")

                    # VRAM strategy
                    vram_options = {
                        VramStrategy.SEQUENTIAL.value: "Sequential - Full unload between agents",
                        VramStrategy.PARALLEL.value: "Parallel - Keep models loaded",
                        VramStrategy.ADAPTIVE.value: "Adaptive - Smart loading (recommended)",
                    }
                    current_vram = (
                        current_mode.vram_strategy if current_mode else VramStrategy.ADAPTIVE.value
                    )
                    self._vram_strategy_select = (
                        ui.select(
                            label="VRAM Strategy",
                            options=vram_options,
                            value=current_vram,
                        )
                        .classes("w-full mt-3")
                        .props("outlined dense")
                        .tooltip("How to manage GPU memory when switching models")
                    )

    def _build_learning_section(self) -> None:
        """Build learning/tuning settings."""
        with ui.card().classes("w-full h-full"):
            self._section_header(
                "Adaptive Learning",
                "psychology",
                "Configure how the system learns from generation quality",
            )

            with ui.column().classes("w-full gap-4"):
                # Autonomy level
                autonomy_options = {
                    AutonomyLevel.MANUAL.value: "Manual - All changes require approval",
                    AutonomyLevel.CAUTIOUS.value: "Cautious - Auto-apply minor changes",
                    AutonomyLevel.BALANCED.value: "Balanced - Auto-apply high confidence",
                    AutonomyLevel.AGGRESSIVE.value: "Aggressive - Auto-apply all, notify",
                    AutonomyLevel.EXPERIMENTAL.value: "Experimental - Try variations",
                }
                self._autonomy_select = (
                    ui.select(
                        label="Autonomy Level",
                        options=autonomy_options,
                        value=self.settings.learning_autonomy,
                    )
                    .classes("w-full")
                    .props("outlined dense")
                    .tooltip("How autonomous the tuning system should be")
                )

                # Learning triggers
                ui.label("Learning Triggers").classes(
                    "text-sm font-medium text-gray-600 dark:text-gray-400"
                )
                trigger_labels = {
                    LearningTrigger.OFF.value: "Off (disabled)",
                    LearningTrigger.AFTER_PROJECT.value: "After completing a story",
                    LearningTrigger.PERIODIC.value: "Every N chapters",
                    LearningTrigger.CONTINUOUS.value: "Continuous background analysis",
                }

                self._trigger_checkboxes = {}
                with ui.column().classes("gap-1 pl-2"):
                    for trigger_value, label in trigger_labels.items():
                        if trigger_value == LearningTrigger.OFF.value:
                            continue  # Skip OFF, use others as toggles
                        is_enabled = trigger_value in self.settings.learning_triggers
                        self._trigger_checkboxes[trigger_value] = ui.checkbox(
                            label, value=is_enabled
                        ).classes("text-sm")

                # Periodic interval (show when periodic is checked)
                with ui.row().classes("w-full items-end gap-2"):
                    self._periodic_interval = (
                        ui.number(
                            label="Periodic interval (chapters)",
                            value=self.settings.learning_periodic_interval,
                            min=1,
                            max=20,
                        )
                        .classes("flex-grow")
                        .props("outlined dense")
                        .tooltip("Analyze every N chapters when periodic trigger is enabled")
                    )

                    self._min_samples = (
                        ui.number(
                            label="Min samples",
                            value=self.settings.learning_min_samples,
                            min=1,
                            max=50,
                        )
                        .classes("w-24")
                        .props("outlined dense")
                        .tooltip("Minimum samples before making recommendations")
                    )

                # Confidence threshold slider
                with ui.column().classes("w-full"):
                    with ui.row().classes("w-full items-center justify-between"):
                        ui.label("Auto-apply Confidence").classes("text-sm")
                        self._confidence_label = ui.label(
                            f"{self.settings.learning_confidence_threshold:.0%}"
                        ).classes(
                            "text-sm font-mono bg-gray-200 dark:bg-gray-700 px-2 py-0.5 rounded"
                        )

                    self._confidence_slider = ui.slider(
                        min=0.5,
                        max=1.0,
                        step=0.05,
                        value=self.settings.learning_confidence_threshold,
                    ).classes("w-full")

                    self._confidence_label.bind_text_from(
                        self._confidence_slider,
                        "value",
                        backward=lambda v: f"{v:.0%}",
                    )

    def _build_world_gen_section(self) -> None:
        """Build world generation settings section."""
        with ui.card().classes("w-full h-full"):
            self._section_header(
                "World Generation",
                "public",
                "Configure entity counts for world building. "
                "Actual counts are randomized between min and max values.",
            )

            # Store sliders for saving
            self._world_gen_sliders: dict[str, tuple[ui.slider, ui.slider]] = {}

            entity_configs = [
                ("characters", "Characters", "people", 1, 20),
                ("locations", "Locations", "place", 1, 15),
                ("factions", "Factions", "groups", 0, 10),
                ("items", "Items", "inventory", 0, 15),
                ("concepts", "Concepts", "lightbulb", 0, 10),
                ("relationships", "Relationships", "share", 1, 40),
            ]

            with ui.column().classes("w-full gap-3"):
                for key, label, icon, abs_min, abs_max in entity_configs:
                    min_attr = f"world_gen_{key}_min"
                    max_attr = f"world_gen_{key}_max"
                    current_min = getattr(self.settings, min_attr)
                    current_max = getattr(self.settings, max_attr)

                    with ui.column().classes("w-full gap-1"):
                        with ui.row().classes("w-full items-center justify-between"):
                            with ui.row().classes("items-center gap-1"):
                                ui.icon(icon, size="xs").classes("text-gray-500")
                                ui.label(label).classes("text-sm font-medium")
                            value_label = ui.label(f"{current_min} - {current_max}").classes(
                                "text-sm font-mono bg-gray-200 dark:bg-gray-700 px-2 py-0.5 rounded"
                            )

                        with ui.row().classes("w-full gap-2 items-center"):
                            ui.label("Min").classes("text-xs text-gray-500 w-8")
                            min_slider = ui.slider(
                                min=abs_min,
                                max=abs_max,
                                step=1,
                                value=current_min,
                            ).classes("flex-1")

                        with ui.row().classes("w-full gap-2 items-center"):
                            ui.label("Max").classes("text-xs text-gray-500 w-8")
                            max_slider = ui.slider(
                                min=abs_min,
                                max=abs_max,
                                step=1,
                                value=current_max,
                            ).classes("flex-1")

                        # Update label when sliders change
                        def update_label(label_ref, min_s, max_s):
                            def _update():
                                # Ensure min <= max
                                if min_s.value > max_s.value:
                                    max_s.value = min_s.value
                                label_ref.text = f"{int(min_s.value)} - {int(max_s.value)}"

                            return _update

                        min_slider.on(
                            "update:model-value", update_label(value_label, min_slider, max_slider)
                        )
                        max_slider.on(
                            "update:model-value", update_label(value_label, min_slider, max_slider)
                        )

                        self._world_gen_sliders[key] = (min_slider, max_slider)

    async def _test_connection(self) -> None:
        """Test Ollama connection."""
        # Update URL first
        self.settings.ollama_url = self._ollama_url_input.value

        health = self.services.model.check_health()
        if health.is_healthy:
            ui.notify(
                f"Connection successful! {health.available_vram} GB VRAM",
                type="positive",
            )
        else:
            ui.notify(f"Connection failed: {health.message}", type="negative")

    def _save_settings(self) -> None:
        """Save all settings."""
        from ui.state import ActionType, UndoAction

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

            # Generation mode
            self.settings.use_mode_system = self._use_mode_system.value
            if hasattr(self, "_mode_select"):
                self.settings.current_mode = self._mode_select.value

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
            for key, (min_slider, max_slider) in self._world_gen_sliders.items():
                min_val = int(min_slider.value)
                max_val = int(max_slider.value)
                # Ensure min <= max
                if min_val > max_val:
                    max_val = min_val
                setattr(self.settings, f"world_gen_{key}_min", min_val)
                setattr(self.settings, f"world_gen_{key}_max", max_val)

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
        """Capture current settings state for undo/redo.

        Returns:
            Dictionary containing all current settings values.
        """
        return {
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
            "use_mode_system": self.settings.use_mode_system,
            "current_mode": self.settings.current_mode,
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
        }

    def _restore_settings_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore settings from a snapshot.

        Args:
            snapshot: Settings snapshot to restore.
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
        self.settings.use_mode_system = snapshot["use_mode_system"]
        self.settings.current_mode = snapshot["current_mode"]
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

        # Mode settings
        if hasattr(self, "_mode_select") and self._mode_select:
            self._mode_select.value = self.settings.current_mode

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
        if hasattr(self, "_world_gen_sliders"):
            for key, (min_slider, max_slider) in self._world_gen_sliders.items():
                min_attr = f"world_gen_{key}_min"
                max_attr = f"world_gen_{key}_max"
                if hasattr(self.settings, min_attr):
                    min_slider.value = getattr(self.settings, min_attr)
                    max_slider.value = getattr(self.settings, max_attr)

    def _do_undo(self) -> None:
        """Handle undo for settings changes."""
        from ui.state import ActionType

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
        from ui.state import ActionType

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

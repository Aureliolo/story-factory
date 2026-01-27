"""Settings page - Generation modes and learning section mixins."""

from nicegui import ui

from src.memory.mode_models import (
    PRESET_MODES,
    AutonomyLevel,
    LearningTrigger,
    VramStrategy,
)
from src.ui.pages.settings._page import SettingsPageBase


class ModesMixin(SettingsPageBase):
    """Mixin providing generation mode and learning settings functionality."""

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
                            with ui.column().classes("gap-2 text-sm"):
                                # Description
                                ui.label(current_mode.description).classes(
                                    "text-gray-600 dark:text-gray-400 italic"
                                )

                    # VRAM strategy (persisted in settings)
                    vram_options = {
                        VramStrategy.SEQUENTIAL.value: "Sequential - Full unload between agents",
                        VramStrategy.PARALLEL.value: "Parallel - Keep models loaded",
                        VramStrategy.ADAPTIVE.value: "Adaptive - Smart loading (recommended)",
                    }
                    self._vram_strategy_select = (
                        ui.select(
                            label="VRAM Strategy",
                            options=vram_options,
                            value=self.settings.vram_strategy,
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

            with ui.column().classes("w-full gap-3"):
                # Autonomy level
                autonomy_options = {
                    AutonomyLevel.MANUAL.value: "Manual - Require approval",
                    AutonomyLevel.CAUTIOUS.value: "Cautious - Auto minor",
                    AutonomyLevel.BALANCED.value: "Balanced - Auto confident",
                    AutonomyLevel.AGGRESSIVE.value: "Aggressive - Auto all",
                    AutonomyLevel.EXPERIMENTAL.value: "Experimental",
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

                # Learning triggers as compact checkboxes
                trigger_labels = {
                    LearningTrigger.AFTER_PROJECT.value: "After story",
                    LearningTrigger.PERIODIC.value: "Every N chapters",
                    LearningTrigger.CONTINUOUS.value: "Continuous",
                }

                self._trigger_checkboxes = {}
                with ui.row().classes("flex-wrap gap-x-4 gap-y-1"):
                    for trigger_value, label in trigger_labels.items():
                        is_enabled = trigger_value in self.settings.learning_triggers
                        self._trigger_checkboxes[trigger_value] = ui.checkbox(
                            label, value=is_enabled
                        ).classes("text-xs")

                # Settings row
                with ui.row().classes("w-full items-end gap-2"):
                    self._periodic_interval = (
                        ui.number(
                            label="Interval",
                            value=self.settings.learning_periodic_interval,
                            min=1,
                            max=20,
                        )
                        .classes("w-20")
                        .props("outlined dense")
                        .tooltip("Chapters between analysis")
                    )

                    self._min_samples = (
                        ui.number(
                            label="Samples",
                            value=self.settings.learning_min_samples,
                            min=1,
                            max=50,
                        )
                        .classes("w-20")
                        .props("outlined dense")
                        .tooltip("Min samples for recommendations")
                    )

                    # Compact confidence with inline label
                    with ui.column().classes("flex-grow"):
                        with ui.row().classes("items-center justify-between"):
                            ui.label("Confidence").classes("text-xs text-gray-500")
                            self._confidence_label = ui.label(
                                f"{self.settings.learning_confidence_threshold:.0%}"
                            ).classes("text-xs font-mono")

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

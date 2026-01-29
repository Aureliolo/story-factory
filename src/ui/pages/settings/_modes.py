"""Settings page - generation mode and adaptive learning sections."""

import logging
from typing import TYPE_CHECKING

from nicegui import ui

from src.memory.mode_models import (
    PRESET_MODES,
    AutonomyLevel,
    LearningTrigger,
    VramStrategy,
)

if TYPE_CHECKING:
    from src.ui.pages.settings import SettingsPage

logger = logging.getLogger(__name__)


def build_mode_section(page: SettingsPage) -> None:
    """Build generation mode settings.

    Args:
        page: The SettingsPage instance.
    """
    with ui.card().classes("w-full"):
        page._section_header(
            "Generation Mode",
            "tune",
            "Select model combinations optimized for different use cases",
        )

        # Mode options from presets
        mode_options = {
            mode_id: f"{mode.name} - {mode.description}" for mode_id, mode in PRESET_MODES.items()
        }

        with ui.column().classes("w-full gap-3"):
            # Use mode system toggle
            page._use_mode_system = ui.switch(
                "Use Generation Modes",
                value=page.settings.use_mode_system,
            ).tooltip("When enabled, uses preset model combinations per agent")

            # Mode selector
            with (
                ui.element("div")
                .classes("w-full")
                .bind_visibility_from(page._use_mode_system, "value")
            ):
                page._mode_select = (
                    ui.select(
                        label="Active Mode",
                        options=mode_options,
                        value=page.settings.current_mode,
                    )
                    .classes("w-full")
                    .props("outlined dense")
                    .tooltip("Choose a preset mode for model assignments")
                )

                # Show current mode description (inline, no expansion)
                current_mode = PRESET_MODES.get(page.settings.current_mode)
                if current_mode:
                    ui.label(current_mode.description).classes(
                        "text-xs text-gray-600 dark:text-gray-400 italic mt-2"
                    )

                # VRAM strategy (persisted in settings)
                vram_options = {
                    VramStrategy.SEQUENTIAL.value: "Sequential - Full unload between agents",
                    VramStrategy.PARALLEL.value: "Parallel - Keep models loaded",
                    VramStrategy.ADAPTIVE.value: "Adaptive - Smart loading (recommended)",
                }
                page._vram_strategy_select = (
                    ui.select(
                        label="VRAM Strategy",
                        options=vram_options,
                        value=page.settings.vram_strategy,
                    )
                    .classes("w-full mt-3")
                    .props("outlined dense")
                    .tooltip("How to manage GPU memory when switching models")
                )

    logger.debug("Mode section built")


def build_learning_section(page: SettingsPage) -> None:
    """Build learning/tuning settings.

    Args:
        page: The SettingsPage instance.
    """
    with ui.card().classes("w-full"):
        page._section_header(
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
            page._autonomy_select = (
                ui.select(
                    label="Autonomy Level",
                    options=autonomy_options,
                    value=page.settings.learning_autonomy,
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

            page._trigger_checkboxes = {}
            with ui.row().classes("flex-wrap gap-x-4 gap-y-1"):
                for trigger_value, label in trigger_labels.items():
                    is_enabled = trigger_value in page.settings.learning_triggers
                    page._trigger_checkboxes[trigger_value] = ui.checkbox(
                        label, value=is_enabled
                    ).classes("text-xs")

            # Settings row
            with ui.row().classes("w-full items-end gap-2"):
                page._periodic_interval = (
                    ui.number(
                        label="Interval",
                        value=page.settings.learning_periodic_interval,
                        min=1,
                        max=20,
                    )
                    .classes("w-20")
                    .props("outlined dense")
                    .tooltip("Chapters between analysis")
                )

                page._min_samples = (
                    ui.number(
                        label="Samples",
                        value=page.settings.learning_min_samples,
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
                        page._confidence_label = ui.label(
                            f"{page.settings.learning_confidence_threshold:.0%}"
                        ).classes("text-xs font-mono")

                    page._confidence_slider = ui.slider(
                        min=0.5,
                        max=1.0,
                        step=0.05,
                        value=page.settings.learning_confidence_threshold,
                    ).classes("w-full")

                    page._confidence_label.bind_text_from(
                        page._confidence_slider,
                        "value",
                        backward=lambda v: f"{v:.0%}",
                    )

    logger.debug("Learning section built")


def save_to_settings(page: SettingsPage) -> None:
    """Extract mode and learning settings from UI and save to settings.

    Args:
        page: The SettingsPage instance.
    """
    # Generation mode
    page.settings.use_mode_system = page._use_mode_system.value
    if hasattr(page, "_mode_select"):
        page.settings.current_mode = page._mode_select.value
    if hasattr(page, "_vram_strategy_select"):
        page.settings.vram_strategy = page._vram_strategy_select.value

    # Learning settings
    page.settings.learning_autonomy = page._autonomy_select.value

    # Collect enabled triggers using list comprehension
    enabled_triggers = [
        trigger_value
        for trigger_value, checkbox in page._trigger_checkboxes.items()
        if checkbox.value
    ]
    if not enabled_triggers:
        enabled_triggers = ["off"]
    page.settings.learning_triggers = enabled_triggers

    page.settings.learning_periodic_interval = int(page._periodic_interval.value)
    page.settings.learning_min_samples = int(page._min_samples.value)
    page.settings.learning_confidence_threshold = page._confidence_slider.value

    logger.debug("Mode and learning settings saved")


def refresh_from_settings(page: SettingsPage) -> None:
    """Refresh mode and learning UI elements from current settings values.

    Args:
        page: The SettingsPage instance.
    """
    settings = page.settings

    # Mode settings
    if hasattr(page, "_use_mode_system"):
        page._use_mode_system.value = settings.use_mode_system
    if hasattr(page, "_mode_select"):
        page._mode_select.value = settings.current_mode
    if hasattr(page, "_vram_strategy_select"):
        page._vram_strategy_select.value = settings.vram_strategy

    # Learning settings
    if hasattr(page, "_autonomy_select"):
        page._autonomy_select.value = settings.learning_autonomy
    if hasattr(page, "_trigger_checkboxes"):
        for trigger_value, checkbox in page._trigger_checkboxes.items():
            checkbox.value = trigger_value in settings.learning_triggers
    if hasattr(page, "_confidence_slider"):
        page._confidence_slider.value = settings.learning_confidence_threshold
    if hasattr(page, "_periodic_interval"):
        page._periodic_interval.value = settings.learning_periodic_interval
    if hasattr(page, "_min_samples"):
        page._min_samples.value = settings.learning_min_samples

    logger.debug("Mode and learning UI refreshed from settings")

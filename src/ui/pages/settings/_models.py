"""Settings page - model selection and temperature sections."""

import logging
from typing import TYPE_CHECKING, Any

from nicegui import ui

from src.settings import AGENT_ROLES

if TYPE_CHECKING:
    from src.ui.pages.settings import SettingsPage

logger = logging.getLogger(__name__)


def build_model_section(page: SettingsPage) -> None:
    """Build model selection settings.

    Args:
        page: The SettingsPage instance.
    """
    with ui.card().classes("w-full h-full"):
        page._section_header(
            "Model Selection",
            "smart_toy",
            "Choose which AI models to use. 'Auto' picks the best available model.",
        )

        # Get installed models
        installed_models = page.services.model.list_installed()
        model_options = {"auto": "Auto-select (recommended)"} | {m: m for m in installed_models}

        # Default model (fall back to "auto" if saved model not installed)
        default_model_value = page.settings.default_model
        if default_model_value not in model_options:
            default_model_value = "auto"

        with ui.row().classes("w-full items-end gap-4"):
            page._default_model_select = (
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
            page._use_per_agent = ui.switch(
                "Per-agent",
                value=page.settings.use_per_agent_models,
            ).tooltip("Use different models for each agent type")

        # Per-agent model selects
        with (
            ui.element("div")
            .classes("w-full mt-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg")
            .bind_visibility_from(page._use_per_agent, "value")
        ):
            ui.label("Per-Agent Model Overrides").classes(
                "font-medium text-sm text-gray-600 dark:text-gray-400 mb-3"
            )

            page._agent_model_selects = {}
            with ui.element("div").classes("grid grid-cols-1 md:grid-cols-2 gap-3"):
                for role, info in AGENT_ROLES.items():
                    # Fall back to "auto" if saved model not installed
                    agent_model_value = page.settings.agent_models.get(role, "auto")
                    if agent_model_value not in model_options:
                        agent_model_value = "auto"

                    page._agent_model_selects[role] = (
                        ui.select(
                            label=info["name"],
                            options=model_options,
                            value=agent_model_value,
                        )
                        .classes("w-full")
                        .props("outlined dense")
                        .tooltip(info["description"])
                    )

    logger.debug("Model selection section built")


def build_temperature_section(page: SettingsPage) -> None:
    """Build temperature settings.

    Args:
        page: The SettingsPage instance.
    """
    with ui.card().classes("w-full h-full"):
        page._section_header(
            "Creativity (Temperature)",
            "thermostat",
            "Controls randomness in AI responses. Higher = more creative/varied, "
            "Lower = more focused/consistent",
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

        page._temp_sliders: dict[str, Any] = {}  # type: ignore[misc]
        page._temp_labels: dict[str, Any] = {}  # type: ignore[misc]
        # 3-column grid for temperature sliders
        with ui.element("div").classes("grid grid-cols-2 lg:grid-cols-3 gap-3"):
            for role, info in AGENT_ROLES.items():
                default_temp = page.settings.get_temperature_for_agent(role)
                with (
                    ui.card()
                    .classes("p-4 bg-gray-50 dark:bg-gray-800")
                    .tooltip(info["description"])
                ):
                    with ui.row().classes("w-full items-center justify-between mb-3"):
                        ui.label(info["name"]).classes("font-medium text-sm")
                        page._temp_labels[role] = ui.label(f"{default_temp:.1f}").classes(
                            "text-sm font-mono bg-gray-200 dark:bg-gray-700 px-2 py-0.5 rounded"
                        )

                    # Slider without floating label (value shown in corner)
                    slider = ui.slider(
                        min=0.0,
                        max=2.0,
                        step=0.1,
                        value=default_temp,
                    ).classes("w-full")
                    page._temp_sliders[role] = slider

                    # Bind the label to the slider value
                    page._temp_labels[role].bind_text_from(
                        slider, "value", backward=lambda v: f"{v:.1f}"
                    )

    logger.debug("Temperature section built")


def save_to_settings(page: SettingsPage) -> None:
    """Extract model settings from UI and save to settings.

    Args:
        page: The SettingsPage instance.
    """
    page.settings.default_model = page._default_model_select.value
    page.settings.use_per_agent_models = page._use_per_agent.value

    # Per-agent models
    for role, select in page._agent_model_selects.items():
        page.settings.agent_models[role] = select.value

    # Temperatures
    for role, slider in page._temp_sliders.items():
        page.settings.agent_temperatures[role] = slider.value

    logger.debug("Model settings saved")


def refresh_from_settings(page: SettingsPage) -> None:
    """Refresh model UI elements from current settings values.

    Args:
        page: The SettingsPage instance.
    """
    settings = page.settings

    if hasattr(page, "_default_model_select") and page._default_model_select:
        page._default_model_select.value = settings.default_model
    if hasattr(page, "_use_per_agent") and page._use_per_agent:
        page._use_per_agent.value = settings.use_per_agent_models

    # Per-agent model selects
    if hasattr(page, "_agent_model_selects"):
        for role, select in page._agent_model_selects.items():
            if select and role in settings.agent_models:
                select.value = settings.agent_models[role]

    # Temperature sliders
    if hasattr(page, "_temp_sliders"):
        for role, slider in page._temp_sliders.items():
            if slider and role in settings.agent_temperatures:
                slider.value = settings.agent_temperatures[role]

    logger.debug("Model UI refreshed from settings")

"""Settings page - connection section."""

import logging
from typing import TYPE_CHECKING

from nicegui import ui

if TYPE_CHECKING:
    from src.ui.pages.settings import SettingsPage

logger = logging.getLogger(__name__)


def build_connection_section(page: SettingsPage) -> None:
    """Build Ollama connection settings.

    Args:
        page: The SettingsPage instance.
    """
    with ui.card().classes("w-full h-full"):
        page._section_header(
            "Connection",
            "link",
            "Configure how Story Factory connects to your local Ollama instance",
        )

        page._ollama_url_input = (
            ui.input(
                label="Ollama URL",
                value=page.settings.ollama_url,
            )
            .classes("w-full")
            .props("outlined dense")
            .tooltip("Usually http://localhost:11434 for local Ollama")
        )

        with ui.row().classes("w-full items-center gap-4 mt-3"):
            ui.button(
                "Test",
                on_click=page._test_connection,
                icon="play_arrow",
            ).props("outline dense")

            # Connection status
            health = page.services.model.check_health()
            if health.is_healthy:
                with ui.row().classes("items-center gap-1"):
                    ui.icon("check_circle", size="sm").classes("text-green-500")
                    ui.label(f"{health.available_vram} GB VRAM").classes(
                        "text-sm text-green-600 dark:text-green-400"
                    )
            else:
                with ui.row().classes("items-center gap-1"):
                    ui.icon("error", size="sm").classes("text-red-500")
                    ui.label("Offline").classes("text-sm text-red-600 dark:text-red-400").tooltip(
                        health.message
                    )

    logger.debug("Connection section built")


async def test_connection(page: SettingsPage) -> None:
    """Test Ollama connection.

    Args:
        page: The SettingsPage instance.
    """
    # Update URL first
    page.settings.ollama_url = page._ollama_url_input.value

    health = page.services.model.check_health()
    if health.is_healthy:
        ui.notify(
            f"Connection successful! {health.available_vram} GB VRAM",
            type="positive",
        )
        logger.info("Connection test successful")
    else:
        ui.notify(f"Connection failed: {health.message}", type="negative")
        logger.warning(f"Connection test failed: {health.message}")


def save_to_settings(page: SettingsPage) -> None:
    """Extract connection settings from UI and save to settings.

    Args:
        page: The SettingsPage instance.
    """
    page.settings.ollama_url = page._ollama_url_input.value
    logger.debug("Connection settings saved")


def refresh_from_settings(page: SettingsPage) -> None:
    """Refresh connection UI elements from current settings values.

    Args:
        page: The SettingsPage instance.
    """
    if hasattr(page, "_ollama_url_input"):
        page._ollama_url_input.value = page.settings.ollama_url
    logger.debug("Connection UI refreshed from settings")

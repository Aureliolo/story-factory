"""Settings page - Connection section mixin."""

from nicegui import ui

from src.ui.pages.settings._page import SettingsPageBase


class ConnectionMixin(SettingsPageBase):
    """Mixin providing connection settings functionality."""

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

    async def _test_connection(self: SettingsPageBase) -> None:
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

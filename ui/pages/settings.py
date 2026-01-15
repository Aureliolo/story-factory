"""Settings page - application configuration."""

from nicegui import ui

from services import ServiceContainer
from settings import AGENT_ROLES
from ui.state import AppState


class SettingsPage:
    """Settings page for application configuration.

    Features:
    - Ollama connection settings
    - Model selection (per-agent or global)
    - Temperature settings
    - Interaction mode
    - Context limits
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

    def build(self) -> None:
        """Build the settings page UI."""
        with ui.column().classes("w-full gap-6 p-4 max-w-4xl mx-auto"):
            ui.label("Settings").classes("text-2xl font-bold")

            # Connection settings
            self._build_connection_section()

            # Model settings
            self._build_model_section()

            # Agent temperatures
            self._build_temperature_section()

            # Interaction settings
            self._build_interaction_section()

            # Context limits
            self._build_context_section()

            # Save button
            ui.button(
                "Save Settings",
                on_click=self._save_settings,
                icon="save",
            ).props("color=primary").classes("mt-4")

    def _build_connection_section(self) -> None:
        """Build Ollama connection settings."""
        with ui.card().classes("w-full"):
            ui.label("Ollama Connection").classes("text-lg font-semibold mb-4")

            with ui.row().classes("w-full gap-4"):
                self._ollama_url_input = ui.input(
                    label="Ollama URL",
                    value=self.settings.ollama_url,
                ).classes("flex-grow")

                ui.button(
                    "Test Connection",
                    on_click=self._test_connection,
                ).props("outline")

            # Connection status
            health = self.services.model.check_health()
            if health.is_healthy:
                ui.label(f"Connected - {health.available_vram} GB VRAM available").classes(
                    "text-sm text-green-600 mt-2"
                )
            else:
                ui.label(f"Not connected: {health.message}").classes("text-sm text-red-600 mt-2")

    def _build_model_section(self) -> None:
        """Build model selection settings."""
        with ui.card().classes("w-full"):
            ui.label("Model Settings").classes("text-lg font-semibold mb-4")

            # Get installed models
            installed_models = self.services.model.list_installed()
            model_options = {"auto": "Auto-select"} | {m: m for m in installed_models}

            # Default model
            self._default_model_select = ui.select(
                label="Default Model",
                options=model_options,
                value=self.settings.default_model,
            ).classes("w-full mb-4")

            # Per-agent toggle
            self._use_per_agent = ui.switch(
                "Use per-agent models",
                value=self.settings.use_per_agent_models,
            )

            # Per-agent model selects
            with (
                ui.column()
                .classes("w-full gap-2 mt-4")
                .bind_visibility_from(self._use_per_agent, "value")
            ):
                ui.label("Per-Agent Models").classes("font-medium")

                self._agent_model_selects = {}
                for role, info in AGENT_ROLES.items():
                    with ui.row().classes("w-full items-center gap-4"):
                        ui.label(info["name"]).classes("w-32")
                        self._agent_model_selects[role] = ui.select(
                            options=model_options,
                            value=self.settings.agent_models.get(role, "auto"),
                        ).classes("flex-grow")
                        ui.label(info["description"]).classes("text-sm text-gray-500")

    def _build_temperature_section(self) -> None:
        """Build temperature settings."""
        with ui.card().classes("w-full"):
            ui.label("Agent Temperatures").classes("text-lg font-semibold mb-4")
            ui.label("Higher values = more creative, lower = more focused").classes(
                "text-sm text-gray-500 mb-4"
            )

            self._temp_sliders = {}
            for role, info in AGENT_ROLES.items():
                with ui.row().classes("w-full items-center gap-4"):
                    ui.label(info["name"]).classes("w-32")
                    self._temp_sliders[role] = ui.slider(
                        min=0.0,
                        max=2.0,
                        step=0.1,
                        value=self.settings.agent_temperatures.get(role, 0.8),
                    ).classes("flex-grow")
                    ui.label().bind_text_from(
                        self._temp_sliders[role], "value", lambda v: f"{v:.1f}"
                    ).classes("w-12 text-right")

    def _build_interaction_section(self) -> None:
        """Build interaction mode settings."""
        with ui.card().classes("w-full"):
            ui.label("Interaction Mode").classes("text-lg font-semibold mb-4")

            self._interaction_mode_select = ui.select(
                label="Mode",
                options={
                    "minimal": "Minimal - Fewest interruptions",
                    "checkpoint": "Checkpoint - Review every N chapters",
                    "interactive": "Interactive - More control points",
                    "collaborative": "Collaborative - Maximum interaction",
                },
                value=self.settings.interaction_mode,
            ).classes("w-full")

            with ui.row().classes("w-full gap-4 mt-4"):
                self._checkpoint_input = ui.number(
                    label="Chapters between checkpoints",
                    value=self.settings.chapters_between_checkpoints,
                    min=1,
                    max=20,
                )

                self._revision_input = ui.number(
                    label="Max revision iterations",
                    value=self.settings.max_revision_iterations,
                    min=0,
                    max=10,
                )

    def _build_context_section(self) -> None:
        """Build context limit settings."""
        with ui.card().classes("w-full"):
            ui.label("Context Limits").classes("text-lg font-semibold mb-4")
            ui.label("Control how much context is sent to the AI").classes(
                "text-sm text-gray-500 mb-4"
            )

            with ui.row().classes("w-full gap-4"):
                self._context_size_input = ui.number(
                    label="Context window size",
                    value=self.settings.context_size,
                    min=1024,
                    max=128000,
                    step=1024,
                )

                self._max_tokens_input = ui.number(
                    label="Max output tokens",
                    value=self.settings.max_tokens,
                    min=256,
                    max=32000,
                    step=256,
                )

            with ui.row().classes("w-full gap-4 mt-4"):
                self._prev_chapter_chars = ui.number(
                    label="Previous chapter context (chars)",
                    value=self.settings.previous_chapter_context_chars,
                    min=500,
                    max=10000,
                )

                self._chapter_analysis_chars = ui.number(
                    label="Chapter analysis context (chars)",
                    value=self.settings.chapter_analysis_chars,
                    min=1000,
                    max=20000,
                )

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
        try:
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

            # Validate and save
            self.settings.validate()
            self.settings.save()

            ui.notify("Settings saved!", type="positive")

        except ValueError as e:
            ui.notify(f"Invalid setting: {e}", type="negative")
        except Exception as e:
            ui.notify(f"Error saving: {e}", type="negative")

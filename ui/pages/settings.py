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
        with ui.column().classes("w-full gap-6 p-4"):
            ui.label("Settings").classes("text-2xl font-bold")

            # Row 1: Small/medium cards - same height via align-items-stretch
            with ui.element("div").classes("flex flex-wrap gap-4 w-full items-stretch"):
                with ui.element("div").style("flex: 1 1 260px; min-width: 260px;"):
                    self._build_connection_section()

                with ui.element("div").style("flex: 1 1 300px; min-width: 300px;"):
                    self._build_interaction_section()

                with ui.element("div").style("flex: 1 1 300px; min-width: 300px;"):
                    self._build_context_section()

                with ui.element("div").style("flex: 1.5 1 380px; min-width: 380px;"):
                    self._build_model_section()

            # Row 2: Temperature sliders (own row since it's wide)
            self._build_temperature_section()

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
        with ui.card().classes("w-full"):
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

            self._temp_sliders: dict = {}
            self._temp_labels: dict = {}
            # 3-column grid for temperature sliders
            with ui.element("div").classes("grid grid-cols-2 lg:grid-cols-3 gap-3"):
                for role, info in AGENT_ROLES.items():
                    default_temp = self.settings.agent_temperatures.get(role, 0.8)
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

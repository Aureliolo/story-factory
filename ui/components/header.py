"""Header component with project selector and status."""

from nicegui import ui
from nicegui.elements.label import Label
from nicegui.elements.select import Select

from services import ServiceContainer
from ui.state import AppState


class Header:
    """Application header with project selector and Ollama status.

    Features:
    - Project dropdown selector
    - New project button
    - Ollama connection status
    - VRAM display
    """

    def __init__(self, state: AppState, services: ServiceContainer):
        """Initialize header.

        Args:
            state: Application state.
            services: Service container.
        """
        self.state = state
        self.services = services
        self._project_select: Select | None = None
        self._status_label: Label | None = None
        self._vram_label: Label | None = None

    def build(self) -> None:
        """Build the header UI."""
        # NiceGUI header has built-in white background that overrides Tailwind classes
        # Use inline style to force dark mode background
        bg_color = "#1f2937" if self.state.dark_mode else "#ffffff"  # gray-800 or white
        with ui.header().classes("shadow-sm items-center").style(f"background-color: {bg_color}"):
            with ui.row().classes("w-full items-center gap-4 px-4 py-2"):
                # Logo/Title
                ui.icon("auto_stories", size="lg").classes("text-blue-500")
                ui.label("Story Factory").classes("text-xl font-bold")

                ui.separator().props("vertical").classes("h-8 bg-gray-300 dark:bg-gray-600")

                # Project selector
                self._build_project_selector()

                # New project button
                ui.button(
                    "+ New Project",
                    on_click=self._create_project,
                ).props("flat color=primary")

                # Spacer
                ui.space()

                # Dark mode toggle
                self._build_theme_toggle()

                ui.separator().props("vertical").classes("h-6 bg-gray-300 dark:bg-gray-600")

                # Ollama status
                self._build_status_display()

    def _build_project_selector(self) -> None:
        """Build the project dropdown selector."""
        projects = self.services.project.list_projects()

        options = {p.id: p.name for p in projects}
        if not options:
            options = {"": "No projects yet"}

        self._project_select = ui.select(
            options=options,
            value=self.state.project_id,
            label="Active Project",
            on_change=self._on_project_change,
        ).classes("w-64")

    def _build_status_display(self) -> None:
        """Build Ollama status and VRAM display."""
        with ui.row().classes("items-center gap-4"):
            # Ollama status
            health = self.services.model.check_health()

            if health.is_healthy:
                ui.icon("check_circle", color="green").classes("text-green-500 dark:text-green-400")
                self._status_label = ui.label("Ollama Connected").classes(
                    "text-sm text-green-500 dark:text-green-400"
                )
            else:
                ui.icon("error", color="red").classes("text-red-500 dark:text-red-400")
                self._status_label = ui.label("Ollama Offline").classes(
                    "text-sm text-red-500 dark:text-red-400"
                )

            ui.separator().props("vertical").classes("h-6 bg-gray-300 dark:bg-gray-600")

            # VRAM display
            vram = self.services.model.get_vram()
            with ui.row().classes("items-center gap-1"):
                ui.icon("memory", size="sm").classes("text-gray-500 dark:text-gray-400")
                self._vram_label = ui.label(f"{vram} GB VRAM").classes(
                    "text-sm text-gray-600 dark:text-gray-400"
                )

    def _build_theme_toggle(self) -> None:
        """Build the dark mode toggle button."""

        def toggle_theme():
            """Toggle between dark and light mode."""
            self.state.dark_mode = not self.state.dark_mode
            self.services.settings.dark_mode = self.state.dark_mode
            self.services.settings.save()
            ui.notify(
                f"{'Dark' if self.state.dark_mode else 'Light'} mode enabled. Refresh to apply.",
                type="info",
            )

        icon = "dark_mode" if not self.state.dark_mode else "light_mode"
        tooltip = "Enable dark mode" if not self.state.dark_mode else "Enable light mode"

        ui.button(
            icon=icon,
            on_click=toggle_theme,
        ).props("flat round").tooltip(tooltip)

    async def _on_project_change(self, e) -> None:
        """Handle project selection change."""
        project_id = e.value
        if not project_id:
            self.state.clear_project()
            return

        try:
            project, world_db = self.services.project.load_project(project_id)
            self.state.set_project(project_id, project, world_db)
            ui.notify(f"Loaded: {project.project_name}", type="positive")
        except FileNotFoundError:
            ui.notify("Project not found", type="negative")
        except Exception as ex:
            ui.notify(f"Error loading project: {ex}", type="negative")

    async def _create_project(self) -> None:
        """Create a new project."""
        try:
            project, world_db = self.services.project.create_project()
            self.state.set_project(project.id, project, world_db)

            # Update dropdown
            self._refresh_project_list()

            ui.notify("New project created!", type="positive")
        except Exception as ex:
            ui.notify(f"Error creating project: {ex}", type="negative")

    def _refresh_project_list(self) -> None:
        """Refresh the project dropdown options."""
        if self._project_select:
            projects = self.services.project.list_projects()
            options = {p.id: p.name for p in projects}
            if not options:
                options = {"": "No projects yet"}

            self._project_select.options = options
            self._project_select.value = self.state.project_id
            self._project_select.update()

    def refresh_status(self) -> None:
        """Refresh the Ollama status display."""
        health = self.services.model.check_health()

        if self._status_label:
            if health.is_healthy:
                self._status_label.text = "Ollama Connected"
                self._status_label.classes(replace="text-sm text-green-500 dark:text-green-400")
            else:
                self._status_label.text = "Ollama Offline"
                self._status_label.classes(replace="text-sm text-red-500 dark:text-red-400")

        if self._vram_label:
            vram = self.services.model.get_vram()
            self._vram_label.text = f"{vram} GB VRAM"

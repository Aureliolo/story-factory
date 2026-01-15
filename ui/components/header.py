"""Header component with navigation, project selector and status."""

from nicegui import ui
from nicegui.elements.label import Label
from nicegui.elements.select import Select

from services import ServiceContainer
from ui.state import AppState

# Navigation items: (path, label, icon)
NAV_ITEMS = [
    ("/", "Write", "edit"),
    ("/world", "World", "public"),
    ("/projects", "Projects", "folder"),
    ("/settings", "Settings", "settings"),
    ("/models", "Models", "smart_toy"),
]


class Header:
    """Application header with navigation, project selector and status."""

    def __init__(self, state: AppState, services: ServiceContainer, current_path: str = "/"):
        """Initialize header."""
        self.state = state
        self.services = services
        self.current_path = current_path
        self._project_select: Select | None = None
        self._status_label: Label | None = None

    def build(self) -> None:
        """Build the header UI."""
        bg_color = "#1f2937" if self.state.dark_mode else "#ffffff"
        with ui.header().classes("shadow-sm items-center").style(f"background-color: {bg_color}"):
            with ui.row().classes("w-full items-center gap-2 px-4 py-2"):
                # Logo/Title
                ui.icon("auto_stories", size="lg").classes("text-blue-500")
                ui.label("Story Factory").classes("text-xl font-bold mr-2")

                # Navigation links
                self._build_navigation()

                # Spacer
                ui.space()

                # Project selector (compact)
                self._build_project_selector()

                # Dark mode toggle
                self._build_theme_toggle()

                # Ollama status (compact)
                self._build_status_display()

    def _build_navigation(self) -> None:
        """Build navigation links."""
        for path, label, icon in NAV_ITEMS:
            is_active = self.current_path == path
            if is_active:
                classes = "text-blue-400 bg-blue-500/20"
            else:
                classes = "text-gray-400 hover:text-gray-200 hover:bg-gray-700/50"

            with ui.link(target=path).classes(
                f"flex items-center gap-1 px-3 py-1.5 rounded-md transition-colors {classes}"
            ):
                ui.icon(icon, size="xs")
                ui.label(label).classes("text-sm")

    def _build_project_selector(self) -> None:
        """Build the project dropdown selector."""
        projects = self.services.project.list_projects()

        options = {p.id: p.name for p in projects}
        if not options:
            options = {"": "No projects"}

        self._project_select = (
            ui.select(
                options=options,
                value=self.state.project_id,
                on_change=self._on_project_change,
            )
            .classes("w-40")
            .props("dense outlined dark")
        )

    def _build_status_display(self) -> None:
        """Build Ollama status and VRAM display (compact)."""
        with ui.row().classes("items-center gap-2"):
            health = self.services.model.check_health()
            vram = self.services.model.get_vram()

            if health.is_healthy:
                ui.icon("check_circle", size="xs").classes("text-green-500")
                self._status_label = ui.label(f"{vram}GB").classes("text-xs text-green-500")
            else:
                ui.icon("error", size="xs").classes("text-red-500")
                self._status_label = ui.label("Offline").classes("text-xs text-red-500")

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
            self.services.settings.last_project_id = None
            self.services.settings.save()
            return

        try:
            project, world_db = self.services.project.load_project(project_id)
            self.state.set_project(project_id, project, world_db)
            self.services.settings.last_project_id = project_id
            self.services.settings.save()
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
            self.services.settings.last_project_id = project.id
            self.services.settings.save()
            ui.notify("New project created!", type="positive")
            self._refresh_project_list()
        except Exception as ex:
            ui.notify(f"Error: {ex}", type="negative")

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
        if self._status_label:
            health = self.services.model.check_health()
            vram = self.services.model.get_vram()

            if health.is_healthy:
                self._status_label.text = f"{vram}GB"
                self._status_label.classes(replace="text-xs text-green-500")
            else:
                self._status_label.text = "Offline"
                self._status_label.classes(replace="text-xs text-red-500")

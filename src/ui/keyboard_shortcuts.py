"""Keyboard shortcuts handler for Story Factory."""

import logging

from nicegui import ui

from src.services import ServiceContainer
from src.ui.state import AppState

logger = logging.getLogger(__name__)


class KeyboardShortcuts:
    """Manage keyboard shortcuts for the application.

    Common shortcuts:
    - Ctrl+N: New project
    - Ctrl+S: Save current work
    - Ctrl+/: Show shortcuts help
    - Ctrl+D: Toggle dark mode
    - Alt+1-5: Switch between tabs
    """

    def __init__(self, state: AppState, services: ServiceContainer) -> None:
        """Initialize keyboard shortcuts.

        Args:
            state: Application state.
            services: Service container.
        """
        self.state = state
        self.services = services
        self._shortcuts_dialog = None

    def register(self) -> None:
        """Register keyboard shortcuts."""
        # Note: NiceGUI uses JavaScript for keyboard handling
        js_code = """
        document.addEventListener('keydown', function(event) {
            // Ctrl+N - New project
            if (event.ctrlKey && event.key === 'n') {
                event.preventDefault();
                window.__nicegui.emit('shortcut:new-project');
            }

            // Ctrl+S - Save
            if (event.ctrlKey && event.key === 's') {
                event.preventDefault();
                window.__nicegui.emit('shortcut:save');
            }

            // Ctrl+Z - Undo (skip in input fields to preserve browser undo)
            if (event.ctrlKey && !event.shiftKey && event.key === 'z') {
                if (event.target.tagName !== 'INPUT' && event.target.tagName !== 'TEXTAREA') {
                    event.preventDefault();
                    window.__nicegui.emit('shortcut:undo');
                }
            }

            // Ctrl+Y or Ctrl+Shift+Z - Redo (skip in input fields)
            if ((event.ctrlKey && event.key === 'y') || (event.ctrlKey && event.shiftKey && event.key === 'z')) {
                if (event.target.tagName !== 'INPUT' && event.target.tagName !== 'TEXTAREA') {
                    event.preventDefault();
                    window.__nicegui.emit('shortcut:redo');
                }
            }

            // Ctrl+/ - Show shortcuts
            if (event.ctrlKey && event.key === '/') {
                event.preventDefault();
                window.__nicegui.emit('shortcut:show-help');
            }

            // Ctrl+D - Toggle dark mode
            if (event.ctrlKey && event.key === 'd') {
                event.preventDefault();
                window.__nicegui.emit('shortcut:toggle-dark-mode');
            }

            // Alt+1-5 - Tab navigation
            if (event.altKey && ['1','2','3','4','5'].includes(event.key)) {
                event.preventDefault();
                window.__nicegui.emit('shortcut:tab-' + event.key);
            }
        });
        """

        ui.add_head_html(f"<script>{js_code}</script>")

        # Register event handlers
        ui.on("shortcut:new-project", self._handle_new_project)
        ui.on("shortcut:save", self._handle_save)
        ui.on("shortcut:undo", self._handle_undo)
        ui.on("shortcut:redo", self._handle_redo)
        ui.on("shortcut:show-help", self._handle_show_help)
        ui.on("shortcut:toggle-dark-mode", self._handle_toggle_dark_mode)
        ui.on("shortcut:tab-1", lambda: self._switch_tab("write"))
        ui.on("shortcut:tab-2", lambda: self._switch_tab("world"))
        ui.on("shortcut:tab-3", lambda: self._switch_tab("projects"))
        ui.on("shortcut:tab-4", lambda: self._switch_tab("settings"))
        ui.on("shortcut:tab-5", lambda: self._switch_tab("models"))

    def _handle_new_project(self) -> None:
        """Handle new project shortcut."""
        try:
            project, world_db = self.services.project.create_project()
            self.state.set_project(project.id, project, world_db)
            self.services.settings.last_project_id = project.id
            self.services.settings.save()
            ui.notify("New project created! (Ctrl+N)", type="positive")
        except Exception as e:
            logger.exception("Failed to create project via shortcut")
            ui.notify(f"Error: {e}", type="negative")

    def _handle_save(self) -> None:
        """Handle save shortcut."""
        if self.state.project and self.state.project_id:
            try:
                self.services.project.save_project(self.state.project)
                ui.notify("Project saved! (Ctrl+S)", type="positive")
            except Exception as e:
                logger.exception(f"Failed to save project {self.state.project_id}")
                ui.notify(f"Error saving: {e}", type="negative")
        else:
            ui.notify("No active project to save", type="info")

    def _handle_undo(self) -> None:
        """Handle undo shortcut."""
        if self.state.can_undo():
            self.state.trigger_undo()
            logger.debug("Undo triggered (Ctrl+Z)")
        else:
            logger.debug("Undo not available")

    def _handle_redo(self) -> None:
        """Handle redo shortcut."""
        if self.state.can_redo():
            self.state.trigger_redo()
            logger.debug("Redo triggered (Ctrl+Y)")
        else:
            logger.debug("Redo not available")

    def _handle_show_help(self) -> None:
        """Show keyboard shortcuts help dialog."""
        with ui.dialog() as dialog, ui.card().classes("w-96"):
            ui.label("Keyboard Shortcuts").classes("text-xl font-bold mb-4")

            shortcuts = [
                ("Ctrl+N", "Create new project"),
                ("Ctrl+S", "Save current project"),
                ("Ctrl+Z", "Undo last action"),
                ("Ctrl+Y", "Redo last action"),
                ("Ctrl+D", "Toggle dark mode"),
                ("Ctrl+/", "Show this help"),
                ("Alt+1", "Go to Write Story tab"),
                ("Alt+2", "Go to World Builder tab"),
                ("Alt+3", "Go to Projects tab"),
                ("Alt+4", "Go to Settings tab"),
                ("Alt+5", "Go to Models tab"),
            ]

            for key, description in shortcuts:
                with ui.row().classes("w-full items-center gap-4 py-2"):
                    ui.badge(key).props("color=primary")
                    ui.label(description).classes("text-sm text-gray-700 dark:text-gray-300")

            ui.button("Close", on_click=dialog.close).props("color=primary").classes("mt-4 w-full")

        dialog.open()

    def _handle_toggle_dark_mode(self) -> None:
        """Handle dark mode toggle shortcut."""
        self.state.dark_mode = not self.state.dark_mode
        self.services.settings.dark_mode = self.state.dark_mode
        self.services.settings.save()
        ui.notify(
            f"{'Dark' if self.state.dark_mode else 'Light'} mode enabled. Refresh to apply. (Ctrl+D)",
            type="info",
        )

    def _switch_tab(self, tab_name: str) -> None:
        """Switch to a specific tab using route navigation."""
        routes = {
            "write": "/",
            "world": "/world",
            "projects": "/projects",
            "settings": "/settings",
            "models": "/models",
        }
        route = routes.get(tab_name, "/")
        ui.navigate.to(route)

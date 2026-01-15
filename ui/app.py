"""Main NiceGUI application for Story Factory."""

import logging
from typing import Protocol

from nicegui import app, ui

from services import ServiceContainer
from ui.components.header import Header
from ui.keyboard_shortcuts import KeyboardShortcuts
from ui.pages.models import ModelsPage
from ui.pages.projects import ProjectsPage
from ui.pages.settings import SettingsPage
from ui.pages.world import WorldPage
from ui.pages.write import WritePage
from ui.state import AppState

logger = logging.getLogger(__name__)


class Page(Protocol):
    """Protocol for page classes."""

    def build(self) -> None:
        """Build the page UI."""
        ...


class StoryFactoryApp:
    """Main Story Factory application.

    This class builds and runs the NiceGUI-based web interface,
    coordinating between the UI components, pages, and services.
    """

    def __init__(self, services: ServiceContainer):
        """Initialize the application.

        Args:
            services: Service container with all services.
        """
        self.services = services
        self.state = AppState()

        # Load dark mode preference from settings
        self.state.dark_mode = services.settings.dark_mode

        # Page instances (created on build)
        self._header: Header | None = None
        self._pages: dict[str, Page] = {}
        self._shortcuts: KeyboardShortcuts | None = None

    def build(self) -> None:
        """Build the application routes and pages."""

        @ui.page("/")
        def main_page():
            """Main application page."""
            # Add custom styles (inside page function)
            from pathlib import Path

            css_path = Path(__file__).parent / "styles.css"
            if css_path.exists():
                with open(css_path) as f:
                    ui.add_head_html(f"<style>{f.read()}</style>")

            # Register keyboard shortcuts (inside page function)
            shortcuts = KeyboardShortcuts(self.state, self.services)
            shortcuts.register()

            self._build_main_page()

        # Store app reference for cleanup
        app.on_shutdown(self._on_shutdown)

        logger.info("Story Factory app built successfully")

    def _build_main_page(self) -> None:
        """Build the main page UI."""
        from ui.theme import get_background_class

        # Apply theme-based background
        bg_class = get_background_class(self.state.dark_mode)
        ui.query("body").classes(bg_class)

        # Apply NiceGUI dark mode
        if self.state.dark_mode:
            ui.dark_mode().enable()
        else:
            ui.dark_mode().disable()

        # Header with project selector
        self._header = Header(self.state, self.services)
        self._header.build()

        # Main content area
        with ui.column().classes("w-full flex-grow"):
            # Tab navigation
            with ui.tabs().classes("w-full bg-white dark:bg-gray-800 shadow-sm") as tabs:
                ui.tab("write", label="Write Story", icon="edit")
                ui.tab("world", label="World Builder", icon="public")
                ui.tab("projects", label="Projects", icon="folder")
                ui.tab("settings", label="Settings", icon="settings")
                ui.tab("models", label="Models", icon="smart_toy")

            # Tab panels
            with ui.tab_panels(tabs, value="write").classes("w-full flex-grow"):
                with ui.tab_panel("write").classes("p-0"):
                    self._pages["write"] = WritePage(self.state, self.services)
                    self._pages["write"].build()

                with ui.tab_panel("world").classes("p-0"):
                    self._pages["world"] = WorldPage(self.state, self.services)
                    self._pages["world"].build()

                with ui.tab_panel("projects").classes("p-0"):
                    self._pages["projects"] = ProjectsPage(self.state, self.services)
                    self._pages["projects"].build()

                with ui.tab_panel("settings").classes("p-0"):
                    self._pages["settings"] = SettingsPage(self.state, self.services)
                    self._pages["settings"].build()

                with ui.tab_panel("models").classes("p-0"):
                    self._pages["models"] = ModelsPage(self.state, self.services)
                    self._pages["models"].build()

        # Register state change handlers
        self._setup_state_handlers()

    def _setup_state_handlers(self) -> None:
        """Set up handlers for state changes."""

        def on_project_change():
            """Handle project change - refresh relevant pages."""
            logger.debug(f"Project changed to: {self.state.project_id}")
            # Pages will refresh on next view

        self.state.on_project_change(on_project_change)

    def _on_shutdown(self) -> None:
        """Handle application shutdown."""
        logger.info("Story Factory shutting down")

        # Save current project if any
        if self.state.project and self.state.project_id:
            try:
                self.services.project.save_project(self.state.project)
                logger.info(f"Saved project {self.state.project_id} on shutdown")
            except Exception as e:
                logger.error(f"Failed to save project on shutdown: {e}")

    def run(
        self,
        host: str = "127.0.0.1",
        port: int = 7860,
        title: str = "Story Factory",
        reload: bool = False,
    ) -> None:
        """Run the application.

        Args:
            host: Host to bind to.
            port: Port to listen on.
            title: Browser tab title.
            reload: Enable auto-reload for development.
        """
        logger.info(f"Starting Story Factory on http://{host}:{port}")

        ui.run(
            host=host,
            port=port,
            title=title,
            reload=reload,
            favicon="📚",
            show=False,  # Don't auto-open browser
        )


def create_app(services: ServiceContainer | None = None) -> StoryFactoryApp:
    """Create and configure the Story Factory application.

    Args:
        services: Optional service container. If not provided,
                  creates one with default settings.

    Returns:
        Configured StoryFactoryApp instance.
    """
    if services is None:
        services = ServiceContainer()

    app_instance = StoryFactoryApp(services)
    app_instance.build()

    return app_instance

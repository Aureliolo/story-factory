"""Main NiceGUI application for Story Factory."""

import logging
from collections.abc import Callable
from pathlib import Path
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


# Navigation items: (path, label, icon)
NAV_ITEMS = [
    ("/", "Write Story", "edit"),
    ("/world", "World Builder", "public"),
    ("/projects", "Projects", "folder"),
    ("/settings", "Settings", "settings"),
    ("/models", "Models", "smart_toy"),
]


class StoryFactoryApp:
    """Main Story Factory application.

    Uses path-based routing for proper browser navigation support.
    Each page is a separate route with shared layout.
    """

    def __init__(self, services: ServiceContainer):
        """Initialize the application."""
        self.services = services
        self.state = AppState()
        self.state.dark_mode = services.settings.dark_mode

    def _apply_theme(self) -> None:
        """Apply theme settings to the page."""
        from ui.theme import get_background_class

        bg_class = get_background_class(self.state.dark_mode)
        ui.query("body").classes(bg_class)

        if self.state.dark_mode:
            ui.dark_mode().enable()
        else:
            ui.dark_mode().disable()

    def _add_styles(self) -> None:
        """Add custom CSS styles."""
        css_path = Path(__file__).parent / "styles.css"
        if css_path.exists():
            with open(css_path) as f:
                ui.add_head_html(f"<style>{f.read()}</style>")

    def _build_navigation(self, current_path: str) -> None:
        """Build the navigation bar."""
        with ui.row().classes(
            "w-full justify-center bg-gray-100 dark:bg-gray-800 shadow-sm py-2 gap-1"
        ):
            for path, label, icon in NAV_ITEMS:
                is_active = current_path == path
                btn_classes = "px-4 py-2 rounded-lg transition-colors flex items-center gap-2"
                if is_active:
                    btn_classes += " bg-blue-500 text-white"
                else:
                    btn_classes += (
                        " text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700"
                    )

                with ui.link(target=path).classes(btn_classes):
                    ui.icon(icon, size="sm")
                    ui.label(label)

    def _page_layout(self, current_path: str, build_content: Callable[[], None]) -> None:
        """Shared page layout with header and navigation."""
        self._add_styles()
        self._apply_theme()

        # Register keyboard shortcuts
        shortcuts = KeyboardShortcuts(self.state, self.services)
        shortcuts.register()

        # Header with project selector
        header = Header(self.state, self.services)
        header.build()

        # Navigation bar
        self._build_navigation(current_path)

        # Page content
        with ui.column().classes("w-full flex-grow p-0"):
            build_content()

    def build(self) -> None:
        """Build the application routes."""

        @ui.page("/")
        def write_page():
            def content():
                page = WritePage(self.state, self.services)
                page.build()

            self._page_layout("/", content)

        @ui.page("/world")
        def world_page():
            def content():
                page = WorldPage(self.state, self.services)
                page.build()

            self._page_layout("/world", content)

        @ui.page("/projects")
        def projects_page():
            def content():
                page = ProjectsPage(self.state, self.services)
                page.build()

            self._page_layout("/projects", content)

        @ui.page("/settings")
        def settings_page():
            def content():
                page = SettingsPage(self.state, self.services)
                page.build()

            self._page_layout("/settings", content)

        @ui.page("/models")
        def models_page():
            def content():
                page = ModelsPage(self.state, self.services)
                page.build()

            self._page_layout("/models", content)

        # Cleanup on shutdown
        app.on_shutdown(self._on_shutdown)
        logger.info("Story Factory app built with path-based routing")

    def _on_shutdown(self) -> None:
        """Handle application shutdown."""
        logger.info("Story Factory shutting down")
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
        """Run the application."""
        logger.info(f"Starting Story Factory on http://{host}:{port}")
        ui.run(
            host=host,
            port=port,
            title=title,
            reload=reload,
            favicon="📚",
            show=False,
        )


def create_app(services: ServiceContainer | None = None) -> StoryFactoryApp:
    """Create and configure the Story Factory application."""
    if services is None:
        services = ServiceContainer()

    app_instance = StoryFactoryApp(services)
    app_instance.build()
    return app_instance

"""Main NiceGUI application for Story Factory."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from nicegui import app, ui

from src.services import ServiceContainer
from src.ui.components.header import Header
from src.ui.keyboard_shortcuts import KeyboardShortcuts
from src.ui.pages.analytics import AnalyticsPage
from src.ui.pages.comparison import ComparisonPage
from src.ui.pages.models import ModelsPage
from src.ui.pages.projects import ProjectsPage
from src.ui.pages.settings import SettingsPage
from src.ui.pages.templates import TemplatesPage
from src.ui.pages.timeline import TimelinePage
from src.ui.pages.world import WorldPage
from src.ui.pages.write import WritePage
from src.ui.state import AppState

logger = logging.getLogger(__name__)


class Page(Protocol):
    """Protocol for page classes."""

    def build(self) -> None:
        """Build the page UI."""
        ...


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

        # Auto-load last project if available
        self._load_last_project()

    def _load_last_project(self) -> None:
        """Load the last opened project if it still exists."""
        last_id = self.services.settings.last_project_id
        if not last_id:
            return

        try:
            project, world_db = self.services.project.load_project(last_id)
            self.state.set_project(last_id, project, world_db)
            logger.info(f"Auto-loaded last project: {project.project_name}")
        except FileNotFoundError:
            # Project was deleted, clear the setting
            logger.info(f"Last project {last_id} no longer exists, clearing setting")
            self.services.settings.last_project_id = None
            self.services.settings.save()
        except Exception as e:
            logger.warning(f"Failed to auto-load last project {last_id}: {e}")
            # Clear invalid project reference
            self.services.settings.last_project_id = None
            self.services.settings.save()

    def _apply_theme(self) -> None:
        """Apply theme settings to the page."""
        from src.ui.theme import get_background_class

        bg_class = get_background_class()
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

    def _page_layout(self, current_path: str, build_content: Callable[[], None]) -> None:
        """Shared page layout with header (includes navigation)."""
        self._add_styles()
        self._apply_theme()

        # Register keyboard shortcuts
        shortcuts = KeyboardShortcuts(self.state, self.services)
        shortcuts.register()

        # Header with navigation and project selector
        header = Header(self.state, self.services, current_path)
        header.build()

        # Page content
        with ui.column().classes("w-full flex-grow p-0"):
            build_content()

    def build(self) -> None:
        """Build the application routes."""

        @ui.page("/")
        def write_page() -> None:
            """Render the Write page."""

            def content() -> None:
                """Build the Write page content."""
                page = WritePage(self.state, self.services)
                page.build()

            self._page_layout("/", content)

        @ui.page("/world")
        def world_page() -> None:
            """Render the World page."""

            def content() -> None:
                """Build the World page content."""
                page = WorldPage(self.state, self.services)
                page.build()

            self._page_layout("/world", content)

        @ui.page("/timeline")
        def timeline_page():
            """Render the Timeline page."""

            def content():
                """Build the Timeline page content."""
                page = TimelinePage(self.state, self.services)
                page.build()

            self._page_layout("/timeline", content)

        @ui.page("/projects")
        def projects_page() -> None:
            """Render the Projects page."""

            def content() -> None:
                """Build the Projects page content."""
                page = ProjectsPage(self.state, self.services)
                page.build()

            self._page_layout("/projects", content)

        @ui.page("/settings")
        def settings_page() -> None:
            """Render the Settings page."""

            def content() -> None:
                """Build the Settings page content."""
                page = SettingsPage(self.state, self.services)
                page.build()

            self._page_layout("/settings", content)

        @ui.page("/models")
        def models_page() -> None:
            """Render the Models page."""

            def content() -> None:
                """Build the Models page content."""
                page = ModelsPage(self.state, self.services)
                page.build()

            self._page_layout("/models", content)

        @ui.page("/analytics")
        def analytics_page() -> None:
            """Render the Analytics page."""

            def content() -> None:
                """Build the Analytics page content."""
                page = AnalyticsPage(self.state, self.services)
                page.build()

            self._page_layout("/analytics", content)

        @ui.page("/templates")
        def templates_page():
            """Render the Templates page."""

            def content():
                """Build the Templates page content."""
                page = TemplatesPage(self.state, self.services)
                page.build()

            self._page_layout("/templates", content)

        @ui.page("/compare")
        def compare_page():
            def content():
                page = ComparisonPage(self.state, self.services)
                page.build()

            self._page_layout("/compare", content)

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

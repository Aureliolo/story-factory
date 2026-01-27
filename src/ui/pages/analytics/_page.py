"""Analytics page base class with core functionality."""

import logging
from typing import TYPE_CHECKING

from nicegui import ui
from nicegui.elements.column import Column

from src.memory.mode_database import ModeDatabase
from src.settings import AGENT_ROLES

if TYPE_CHECKING:
    from src.services import ServiceContainer
    from src.ui.state import AppState

logger = logging.getLogger(__name__)


class AnalyticsPageBase:
    """Base class for AnalyticsPage with core methods.

    Features:
    - Overall performance summary
    - Per-model quality metrics
    - Per-agent role breakdown
    - World entity quality metrics
    - Score history over time
    - Recommendation history
    - CSV data export
    """

    # Type hints for instance variables set by subclass
    state: AppState
    services: ServiceContainer
    _db: ModeDatabase
    _summary_section: Column | None
    _model_section: Column | None
    _content_stats_section: Column | None
    _generation_costs_section: Column | None
    _quality_trends_section: Column | None
    _world_quality_section: Column | None
    _recommendations_section: Column | None
    _filter_agent_role: str | None
    _filter_genre: str | None

    def __init__(self, state: AppState, services: ServiceContainer):
        """
        Initialize the AnalyticsPage with the application state and available services.

        Parameters:
                state (AppState): Application state providing current project and UI context.
                services (ServiceContainer): Service container exposing dependencies used by the page.
        """
        self.state = state
        self.services = services
        self._db = ModeDatabase()

        # UI element references
        self._summary_section = None
        self._model_section = None
        self._content_stats_section = None
        self._generation_costs_section = None
        self._quality_trends_section = None
        self._world_quality_section = None
        self._recommendations_section = None

        # Filter state
        self._filter_agent_role = None
        self._filter_genre = None

    def build(self) -> None:
        """
        Constructs and renders the full analytics page layout.

        Creates the main page column, adds the header and filters, initializes per-section UI containers (summary, content statistics, generation costs, quality trends, model performance, world quality, and recommendations) and invokes each section's builder to populate its contents.
        """
        with ui.column().classes("w-full gap-6 p-4"):
            # Header with refresh and export
            self._build_header()

            # Filters
            self._build_filters()

            # Summary cards
            self._summary_section = ui.column().classes("w-full")
            self._build_summary_section()

            # Content statistics
            self._content_stats_section = ui.column().classes("w-full")
            self._build_content_statistics_section()

            # Generation costs
            self._generation_costs_section = ui.column().classes("w-full")
            self._build_generation_costs_section()

            # Quality trends over time
            self._quality_trends_section = ui.column().classes("w-full")
            self._build_quality_trends_section()

            # Model performance table
            self._model_section = ui.column().classes("w-full")
            self._build_model_section()

            # World Quality section
            self._world_quality_section = ui.column().classes("w-full")
            self._build_world_quality_section()

            # Recommendations history
            self._recommendations_section = ui.column().classes("w-full")
            self._build_recommendations_section()

    def _build_header(self) -> None:
        """Build the header with title and actions."""
        with ui.row().classes("w-full items-center flex-wrap gap-2"):
            ui.label("Analytics").classes("text-2xl font-bold")
            ui.space()

            with ui.row().classes("gap-2 flex-wrap"):
                ui.button(
                    "Refresh",
                    on_click=self._refresh_all,
                    icon="refresh",
                ).props("flat")

                ui.button(
                    "Export CSV",
                    on_click=self._export_csv,
                    icon="download",
                ).props("outline")

    def _build_filters(self) -> None:
        """Build filter controls."""
        with ui.card().classes("w-full"):
            with ui.row().classes("w-full items-end gap-4 flex-wrap"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("filter_list").classes("text-blue-500")
                    ui.label("Filters").classes("font-semibold")

                ui.space()

                # Filter dropdowns - responsive width
                with ui.row().classes("gap-2 flex-wrap"):
                    # Agent role filter
                    role_options = {"": "All Agents"} | {
                        role: info["name"] for role, info in AGENT_ROLES.items()
                    }
                    ui.select(
                        label="Agent Role",
                        options=role_options,
                        value="",
                        on_change=lambda e: self._apply_filter(agent_role=e.value or None),
                    ).classes("w-full sm:w-40").props("dense outlined")

                    # Genre filter (populated from data)
                    genres = self._db.get_unique_genres()
                    genre_options = {"": "All Genres"} | {g: g for g in genres if g}
                    ui.select(
                        label="Genre",
                        options=genre_options,
                        value="",
                        on_change=lambda e: self._apply_filter(genre=e.value or None),
                    ).classes("w-full sm:w-40").props("dense outlined")

    def _apply_filter(self, agent_role: str | None = None, genre: str | None = None) -> None:
        """Apply filters and refresh data."""
        if agent_role is not None:
            self._filter_agent_role = agent_role if agent_role else None
        if genre is not None:
            self._filter_genre = genre if genre else None

        self._build_summary_section()
        self._build_model_section()

    def _build_stat_card(self, title: str, value: str, icon: str, color_class: str) -> None:
        """Build a single stat card."""
        with ui.card().classes("p-4"):
            with ui.row().classes("items-center gap-2 mb-2"):
                ui.icon(icon).classes(color_class)
                ui.label(title).classes("text-sm text-gray-500 dark:text-gray-400")
            ui.label(value).classes("text-2xl font-bold")

    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time.

        Args:
            seconds: Time in seconds.

        Returns:
            Formatted string like '5m 32s' or '2h 15m'.
        """
        total_secs = int(seconds)
        if total_secs < 60:
            return f"{total_secs}s"
        elif total_secs < 3600:
            mins = total_secs // 60
            secs = total_secs % 60
            return f"{mins}m {secs}s"
        else:
            hours = total_secs // 3600
            mins = (total_secs % 3600) // 60
            return f"{hours}h {mins}m"

    def _format_tokens(self, tokens: int) -> str:
        """
        Format a token count using K (thousands) or M (millions) suffix when appropriate.

        Returns:
            Formatted string representation of the token count (e.g., '999', '1.2K', '3.5M').
        """
        if tokens < 1000:
            return str(tokens)
        elif tokens < 1_000_000:
            return f"{tokens / 1000:.1f}K"
        else:
            return f"{tokens / 1_000_000:.1f}M"

    def _refresh_all(self) -> None:
        """Refresh all sections."""
        self._build_summary_section()
        self._build_content_statistics_section()
        self._build_generation_costs_section()
        self._build_quality_trends_section()
        self._build_model_section()
        self._build_world_quality_section()
        self._build_recommendations_section()
        ui.notify("Analytics refreshed", type="positive")

    # Abstract methods to be implemented by mixins
    def _build_summary_section(self) -> None:
        """Build the summary cards section."""
        raise NotImplementedError

    def _build_content_statistics_section(self) -> None:
        """Build the content statistics section."""
        raise NotImplementedError

    def _build_generation_costs_section(self) -> None:
        """Build the generation costs section."""
        raise NotImplementedError

    def _build_quality_trends_section(self) -> None:
        """Build the quality trends section."""
        raise NotImplementedError

    def _build_model_section(self) -> None:
        """Build the model performance table."""
        raise NotImplementedError

    def _build_world_quality_section(self) -> None:
        """Build the world quality metrics section."""
        raise NotImplementedError

    def _build_recommendations_section(self) -> None:
        """Build the recommendations history section."""
        raise NotImplementedError

    def _export_csv(self) -> None:
        """Export score data to CSV."""
        raise NotImplementedError

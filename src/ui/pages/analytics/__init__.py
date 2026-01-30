"""Analytics page - model performance dashboard."""

import logging

from nicegui import ui
from nicegui.elements.column import Column

from src.memory.mode_database import ModeDatabase
from src.services import ServiceContainer
from src.settings import AGENT_ROLES
from src.ui.local_prefs import load_prefs_deferred, save_pref
from src.ui.pages.analytics._content import build_content_statistics_section
from src.ui.pages.analytics._costs import build_generation_costs_section, format_time, format_tokens
from src.ui.pages.analytics._export import export_csv
from src.ui.pages.analytics._model import build_model_section
from src.ui.pages.analytics._summary import build_stat_card, build_summary_section
from src.ui.pages.analytics._trends import (
    build_quality_trends_section,
    build_recommendations_section,
    build_world_quality_section,
)
from src.ui.state import AppState

logger = logging.getLogger(__name__)

_PAGE_KEY = "analytics"

__all__ = ["AnalyticsPage"]


class AnalyticsPage:
    """Analytics page for model performance tracking.

    Features:
    - Overall performance summary
    - Per-model quality metrics
    - Per-agent role breakdown
    - World entity quality metrics
    - Score history over time
    - Recommendation history
    - CSV data export
    """

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
        self._summary_section: Column | None = None
        self._model_section: Column | None = None
        self._content_stats_section: Column | None = None
        self._generation_costs_section: Column | None = None
        self._quality_trends_section: Column | None = None
        self._world_quality_section: Column | None = None
        self._recommendations_section: Column | None = None

        # Filter state
        self._filter_agent_role: str | None = None
        self._filter_genre: str | None = None

        # Widget references for deferred preference loading
        self._agent_role_select: ui.select | None = None
        self._genre_select: ui.select | None = None

    def build(self) -> None:
        """
        Constructs and renders the full analytics page layout.

        Creates the main page column, adds the header and filters, initializes per-section
        UI containers (summary, content statistics, generation costs, quality trends, model
        performance, world quality, and recommendations) and invokes each section's builder
        to populate its contents.
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

        # Restore persisted preferences from localStorage
        load_prefs_deferred(_PAGE_KEY, self._apply_prefs)

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
                    self._agent_role_select = (
                        ui.select(
                            label="Agent Role",
                            options=role_options,
                            value="",
                            on_change=lambda e: self._apply_filter(agent_role=e.value or None),
                        )
                        .classes("w-full sm:w-40")
                        .props("dense outlined")
                    )

                    # Genre filter (populated from data)
                    genres = self._db.get_unique_genres()
                    genre_options = {"": "All Genres"} | {g: g for g in genres if g}
                    self._genre_select = (
                        ui.select(
                            label="Genre",
                            options=genre_options,
                            value="",
                            on_change=lambda e: self._apply_filter(genre=e.value or None),
                        )
                        .classes("w-full sm:w-40")
                        .props("dense outlined")
                    )

    def _apply_filter(self, agent_role: str | None = None, genre: str | None = None) -> None:
        """Apply filters and refresh data."""
        if agent_role is not None:
            self._filter_agent_role = agent_role if agent_role else None
            save_pref(_PAGE_KEY, "filter_agent_role", self._filter_agent_role)
        if genre is not None:
            self._filter_genre = genre if genre else None
            save_pref(_PAGE_KEY, "filter_genre", self._filter_genre)

        self._build_summary_section()
        self._build_model_section()

    def _apply_prefs(self, prefs: dict) -> None:
        """Apply loaded preferences to analytics filter state and UI widgets.

        Args:
            prefs: Dict of field→value from localStorage.
        """
        if not prefs:
            return

        changed = False

        if "filter_agent_role" in prefs:
            val = prefs["filter_agent_role"]
            if val != self._filter_agent_role:
                self._filter_agent_role = val
                changed = True
                if self._agent_role_select:
                    self._agent_role_select.value = val or ""

        if "filter_genre" in prefs:
            val = prefs["filter_genre"]
            if val != self._filter_genre:
                self._filter_genre = val
                changed = True
                if self._genre_select:
                    self._genre_select.value = val or ""

        if changed:
            logger.info("Restored analytics preferences from localStorage")
            self._build_summary_section()
            self._build_model_section()

    # --- Delegated section builders ---

    def _build_summary_section(self) -> None:
        """Build the summary cards section."""
        build_summary_section(self)

    def _build_stat_card(self, title: str, value: str, icon: str, color_class: str) -> None:
        """Build a single stat card."""
        build_stat_card(title, value, icon, color_class)

    def _build_model_section(self) -> None:
        """Build the model performance table."""
        build_model_section(self)

    def _build_content_statistics_section(self) -> None:
        """Build the content statistics section."""
        build_content_statistics_section(self)

    def _build_generation_costs_section(self) -> None:
        """Build the generation costs section."""
        build_generation_costs_section(self)

    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time.

        Args:
            seconds: Time in seconds.

        Returns:
            Formatted string like '5m 32s' or '2h 15m'.
        """
        return format_time(seconds)

    def _format_tokens(self, tokens: int) -> str:
        """Format a token count using K/M suffix when appropriate.

        Args:
            tokens: Number of tokens.

        Returns:
            Formatted string representation of the token count.
        """
        return format_tokens(tokens)

    def _build_quality_trends_section(self) -> None:
        """Build the quality trends section."""
        build_quality_trends_section(self)

    def _build_world_quality_section(self) -> None:
        """Build the world quality metrics section."""
        build_world_quality_section(self)

    def _build_recommendations_section(self) -> None:
        """Build the recommendations history section."""
        build_recommendations_section(self)

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

    def _export_csv(self) -> None:
        """Export score data to CSV."""
        export_csv(self)

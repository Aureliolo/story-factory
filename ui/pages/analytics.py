"""Analytics page - model performance dashboard."""

import csv
import io
import logging
from datetime import datetime

from nicegui import ui
from nicegui.elements.column import Column

from memory.mode_database import ModeDatabase
from services import ServiceContainer
from settings import AGENT_ROLES
from ui.state import AppState
from utils import extract_model_name

logger = logging.getLogger(__name__)


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
        """Initialize analytics page.

        Args:
            state: Application state.
            services: Service container.
        """
        self.state = state
        self.services = services
        self._db = ModeDatabase()

        # UI element references
        self._summary_section: Column | None = None
        self._model_section: Column | None = None
        self._world_quality_section: Column | None = None
        self._recommendations_section: Column | None = None

        # Filter state
        self._filter_agent_role: str | None = None
        self._filter_genre: str | None = None

    def build(self) -> None:
        """Build the analytics page UI."""
        with ui.column().classes("w-full gap-6 p-4"):
            # Header with refresh and export
            self._build_header()

            # Filters
            self._build_filters()

            # Summary cards
            self._summary_section = ui.column().classes("w-full")
            self._build_summary_section()

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

    def _build_summary_section(self) -> None:
        """Build the summary cards section."""
        if self._summary_section is None:
            return

        self._summary_section.clear()

        try:
            # Get summary stats
            total_scores = self._db.get_score_count(
                agent_role=self._filter_agent_role, genre=self._filter_genre
            )
            avg_quality = self._db.get_average_score(
                "prose_quality",
                agent_role=self._filter_agent_role,
                genre=self._filter_genre,
            )
            avg_instruction = self._db.get_average_score(
                "instruction_following",
                agent_role=self._filter_agent_role,
                genre=self._filter_genre,
            )
            avg_consistency = self._db.get_average_score(
                "consistency_score",
                agent_role=self._filter_agent_role,
                genre=self._filter_genre,
            )
            avg_speed = self._db.get_average_score(
                "tokens_per_second",
                agent_role=self._filter_agent_role,
                genre=self._filter_genre,
            )
            quality_str = f"{avg_quality:.1f}" if avg_quality else "N/A"
            speed_str = f"{avg_speed:.1f}" if avg_speed else "N/A"
            logger.debug(
                f"Analytics summary: {total_scores} scores, "
                f"quality={quality_str}, "
                f"speed={speed_str} t/s"
            )
        except Exception as e:
            logger.error(f"Failed to load analytics summary: {e}", exc_info=True)
            with self._summary_section:
                ui.label("Failed to load analytics data. Check logs for details.").classes(
                    "text-red-500 p-4"
                )
            return

        with self._summary_section:
            with ui.element("div").classes("grid grid-cols-2 md:grid-cols-5 gap-4"):
                self._build_stat_card(
                    "Total Samples",
                    str(total_scores),
                    "analytics",
                    "text-blue-500",
                )
                self._build_stat_card(
                    "Avg Prose Quality",
                    f"{avg_quality:.1f}/10" if avg_quality else "N/A",
                    "auto_stories",
                    "text-purple-500",
                )
                self._build_stat_card(
                    "Avg Instruction",
                    f"{avg_instruction:.1f}/10" if avg_instruction else "N/A",
                    "checklist",
                    "text-green-500",
                )
                self._build_stat_card(
                    "Avg Consistency",
                    f"{avg_consistency:.1f}/10" if avg_consistency else "N/A",
                    "timeline",
                    "text-orange-500",
                )
                self._build_stat_card(
                    "Avg Speed",
                    f"{avg_speed:.1f} t/s" if avg_speed else "N/A",
                    "speed",
                    "text-cyan-500",
                )

    def _build_stat_card(self, title: str, value: str, icon: str, color_class: str) -> None:
        """Build a single stat card."""
        with ui.card().classes("p-4"):
            with ui.row().classes("items-center gap-2 mb-2"):
                ui.icon(icon).classes(color_class)
                ui.label(title).classes("text-sm text-gray-500 dark:text-gray-400")
            ui.label(value).classes("text-2xl font-bold")

    def _build_model_section(self) -> None:
        """Build the model performance table."""
        if self._model_section is None:
            return

        self._model_section.clear()

        try:
            # Get model performance summaries
            summaries = self._db.get_model_summaries(
                agent_role=self._filter_agent_role, genre=self._filter_genre
            )
            logger.debug(f"Loaded {len(summaries)} model performance summaries")
        except Exception as e:
            logger.error(f"Failed to load model performance data: {e}", exc_info=True)
            with self._model_section:
                with ui.card().classes("w-full"):
                    ui.label(
                        "Failed to load model performance data. Check logs for details."
                    ).classes("text-red-500 p-4")
            return

        with self._model_section:
            with ui.card().classes("w-full"):
                with ui.row().classes("w-full items-center mb-4"):
                    ui.icon("leaderboard").classes("text-blue-500")
                    ui.label("Model Performance").classes("text-lg font-semibold")
                    ui.space()
                    ui.label(f"{len(summaries)} models tracked").classes("text-sm text-gray-500")

                if not summaries:
                    ui.label(
                        "No performance data yet. Generate some stories to collect metrics!"
                    ).classes("text-gray-500 dark:text-gray-400 py-8 text-center")
                    return

                # Performance table
                columns = [
                    {"name": "model", "label": "Model", "field": "model", "sortable": True},
                    {"name": "role", "label": "Role", "field": "role", "sortable": True},
                    {"name": "prose", "label": "Prose", "field": "prose", "sortable": True},
                    {
                        "name": "instruction",
                        "label": "Instruction",
                        "field": "instruction",
                        "sortable": True,
                    },
                    {
                        "name": "consistency",
                        "label": "Consistency",
                        "field": "consistency",
                        "sortable": True,
                    },
                    {"name": "speed", "label": "Speed (t/s)", "field": "speed", "sortable": True},
                    {"name": "samples", "label": "Samples", "field": "samples", "sortable": True},
                ]

                rows = []
                for s in summaries:
                    model_name = extract_model_name(s.model_id)
                    rows.append(
                        {
                            "model": model_name,
                            "role": s.agent_role.title(),
                            "prose": f"{s.avg_prose_quality:.1f}" if s.avg_prose_quality else "-",
                            "instruction": f"{s.avg_instruction_following:.1f}"
                            if s.avg_instruction_following
                            else "-",
                            "consistency": f"{s.avg_consistency:.1f}" if s.avg_consistency else "-",
                            "speed": f"{s.avg_tokens_per_second:.1f}"
                            if s.avg_tokens_per_second
                            else "-",
                            "samples": s.sample_count,
                        }
                    )

                # Wrap table in scrollable container for mobile
                with ui.element("div").classes("w-full overflow-x-auto"):
                    ui.table(columns=columns, rows=rows, row_key="model").classes("w-full")

    def _build_world_quality_section(self) -> None:
        """Build the world quality metrics section."""
        if self._world_quality_section is None:
            return

        self._world_quality_section.clear()

        try:
            summary = self._db.get_world_quality_summary()
            entity_count = summary.get("total_entities", 0)
            logger.debug(f"Loaded world quality summary: {entity_count} entities")
        except Exception as e:
            logger.error(f"Failed to load world quality data: {e}", exc_info=True)
            with self._world_quality_section:
                with ui.card().classes("w-full"):
                    ui.label("Failed to load world quality data.").classes("text-red-500 p-4")
            return

        with self._world_quality_section:
            with ui.card().classes("w-full"):
                with ui.row().classes("w-full items-center mb-4"):
                    ui.icon("public").classes("text-green-500")
                    ui.label("World Quality Metrics").classes("text-lg font-semibold")
                    ui.space()
                    ui.label(f"{entity_count} entities generated").classes("text-sm text-gray-500")

                if entity_count == 0:
                    ui.label(
                        "No world entity quality data yet. Generate entities with Quality Refinement "
                        "enabled to collect metrics!"
                    ).classes("text-gray-500 dark:text-gray-400 py-8 text-center")
                    return

                # Summary stats row
                avg_quality = summary.get("avg_quality")
                avg_iterations = summary.get("avg_iterations")
                avg_time = summary.get("avg_generation_time")

                with ui.element("div").classes("grid grid-cols-2 md:grid-cols-4 gap-4 mb-4"):
                    self._build_stat_card(
                        "Avg Quality",
                        f"{avg_quality:.1f}/10" if avg_quality else "N/A",
                        "star",
                        "text-yellow-500",
                    )
                    self._build_stat_card(
                        "Avg Iterations",
                        f"{avg_iterations:.1f}" if avg_iterations else "N/A",
                        "loop",
                        "text-blue-500",
                    )
                    self._build_stat_card(
                        "Avg Gen Time",
                        f"{avg_time:.1f}s" if avg_time else "N/A",
                        "timer",
                        "text-purple-500",
                    )
                    self._build_stat_card(
                        "Total Entities",
                        str(entity_count),
                        "category",
                        "text-green-500",
                    )

                # Breakdown by entity type
                by_type = summary.get("by_entity_type", [])
                if by_type:
                    ui.label("Quality by Entity Type").classes("font-semibold mt-4 mb-2")
                    columns = [
                        {"name": "type", "label": "Entity Type", "field": "type", "sortable": True},
                        {"name": "count", "label": "Count", "field": "count", "sortable": True},
                        {
                            "name": "quality",
                            "label": "Avg Quality",
                            "field": "quality",
                            "sortable": True,
                        },
                    ]
                    rows = [
                        {
                            "type": item["entity_type"].title(),
                            "count": item["count"],
                            "quality": f"{item['avg_quality']:.1f}" if item["avg_quality"] else "-",
                        }
                        for item in by_type
                    ]
                    with ui.element("div").classes("w-full overflow-x-auto"):
                        ui.table(columns=columns, rows=rows).classes("w-full")

                # Breakdown by model
                by_model = summary.get("by_model", [])
                if by_model:
                    ui.label("Quality by Model").classes("font-semibold mt-4 mb-2")
                    columns = [
                        {"name": "model", "label": "Model", "field": "model", "sortable": True},
                        {"name": "count", "label": "Entities", "field": "count", "sortable": True},
                        {
                            "name": "quality",
                            "label": "Avg Quality",
                            "field": "quality",
                            "sortable": True,
                        },
                    ]
                    rows = [
                        {
                            "model": extract_model_name(item["model_id"]),
                            "count": item["count"],
                            "quality": f"{item['avg_quality']:.1f}" if item["avg_quality"] else "-",
                        }
                        for item in by_model
                    ]
                    with ui.element("div").classes("w-full overflow-x-auto"):
                        ui.table(columns=columns, rows=rows).classes("w-full")

    def _build_recommendations_section(self) -> None:
        """Build the recommendations history section."""
        if self._recommendations_section is None:
            return

        self._recommendations_section.clear()

        try:
            recommendations = self._db.get_recent_recommendations(limit=10)
            logger.debug(f"Loaded {len(recommendations)} recent recommendations")
        except Exception as e:
            logger.error(f"Failed to load recommendations: {e}", exc_info=True)
            with self._recommendations_section:
                with ui.card().classes("w-full"):
                    ui.label("Failed to load recommendations. Check logs for details.").classes(
                        "text-red-500 p-4"
                    )
            return

        with self._recommendations_section:
            with ui.card().classes("w-full"):
                with ui.row().classes("w-full items-center mb-4"):
                    ui.icon("lightbulb").classes("text-yellow-500")
                    ui.label("Recent Recommendations").classes("text-lg font-semibold")

                if not recommendations:
                    ui.label("No recommendations yet.").classes("text-gray-500 dark:text-gray-400")
                    return

                for rec in recommendations:
                    with ui.card().classes("w-full mb-2").props("flat bordered"):
                        with ui.row().classes("w-full items-start gap-3 flex-wrap sm:flex-nowrap"):
                            # Status icon
                            if rec.was_applied:
                                ui.icon("check_circle", color="green")
                            elif rec.user_feedback == "rejected":
                                ui.icon("cancel", color="red")
                            else:
                                ui.icon("pending", color="grey")

                            with ui.column().classes("flex-grow gap-1 min-w-0"):
                                # Type and change
                                with ui.row().classes("items-center gap-2 flex-wrap"):
                                    ui.badge(rec.recommendation_type).props("color=primary")
                                    if rec.affected_role:
                                        ui.label(f"({rec.affected_role})").classes(
                                            "text-sm text-gray-500"
                                        )

                                # Current -> Suggested
                                with ui.row().classes("items-center gap-2 text-sm flex-wrap"):
                                    ui.label(rec.current_value).classes(
                                        "font-mono truncate max-w-[120px] sm:max-w-none"
                                    )
                                    ui.icon("arrow_forward", size="xs")
                                    ui.label(rec.suggested_value).classes(
                                        "font-mono text-blue-500 truncate max-w-[120px] sm:max-w-none"
                                    )

                                # Reason
                                ui.label(rec.reason).classes(
                                    "text-sm text-gray-600 dark:text-gray-400"
                                )

                            # Confidence badge
                            confidence_color = "green" if rec.confidence >= 0.8 else "orange"
                            ui.badge(f"{rec.confidence:.0%}").props(f"color={confidence_color}")

    def _refresh_all(self) -> None:
        """Refresh all sections."""
        self._build_summary_section()
        self._build_model_section()
        self._build_world_quality_section()
        self._build_recommendations_section()
        ui.notify("Analytics refreshed", type="positive")

    def _export_csv(self) -> None:
        """Export score data to CSV."""
        try:
            scores = self._db.get_all_scores(
                agent_role=self._filter_agent_role, genre=self._filter_genre
            )
            logger.info(f"Exporting {len(scores)} scores to CSV")

            if not scores:
                logger.warning("No data to export")
                ui.notify("No data to export", type="warning")
                return

            # Build CSV in memory
            output = io.StringIO()
            writer = csv.writer(output)

            # Header
            writer.writerow(
                [
                    "timestamp",
                    "project_id",
                    "chapter_id",
                    "agent_role",
                    "model_id",
                    "mode_name",
                    "genre",
                    "prose_quality",
                    "instruction_following",
                    "consistency_score",
                    "tokens_generated",
                    "time_seconds",
                    "tokens_per_second",
                    "was_regenerated",
                    "edit_distance",
                    "user_rating",
                ]
            )

            # Data rows
            for score in scores:
                writer.writerow(
                    [
                        score.timestamp.isoformat() if score.timestamp else "",
                        score.project_id,
                        score.chapter_id or "",
                        score.agent_role,
                        score.model_id,
                        score.mode_name,
                        score.genre or "",
                        score.quality.prose_quality if score.quality.prose_quality else "",
                        score.quality.instruction_following
                        if score.quality.instruction_following
                        else "",
                        score.quality.consistency_score if score.quality.consistency_score else "",
                        score.performance.tokens_generated
                        if score.performance.tokens_generated
                        else "",
                        score.performance.time_seconds if score.performance.time_seconds else "",
                        score.performance.tokens_per_second
                        if score.performance.tokens_per_second
                        else "",
                        score.signals.was_regenerated,
                        score.signals.edit_distance if score.signals.edit_distance else "",
                        score.signals.user_rating if score.signals.user_rating else "",
                    ]
                )

            csv_content = output.getvalue()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"story_factory_analytics_{timestamp}.csv"

            # Trigger download
            ui.download(csv_content.encode(), filename)
            logger.info(f"Successfully exported {len(scores)} records to {filename}")
            ui.notify(f"Exported {len(scores)} records to {filename}", type="positive")

        except Exception as e:
            logger.error(f"Failed to export CSV: {e}", exc_info=True)
            ui.notify(f"Export failed: {str(e)}", type="negative")

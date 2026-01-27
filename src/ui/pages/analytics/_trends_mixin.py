"""Quality trends section mixin for AnalyticsPage."""

import logging

from nicegui import ui

from src.ui.pages.analytics._page import AnalyticsPageBase

logger = logging.getLogger(__name__)


class TrendsMixin(AnalyticsPageBase):
    """Mixin providing quality trends section functionality."""

    def _build_quality_trends_section(self) -> None:
        """
        Populate the quality trends UI section with recent prose quality and generation speed data.

        Fetches daily averages for "prose_quality" and "tokens_per_second" over the past 30 days for the current filter, then renders:
        - A header "Quality Trends (30 Days)".
        - If no trend data is available, a message indicating more content is needed.
        - If prose quality data is available, a "Prose Quality Over Time" table showing the most recent 10 dates with date, average quality (formatted as X.X/10), and sample count.
        - If generation speed data is available, a "Generation Speed Over Time" table showing the most recent 10 dates with date, average speed (tokens/second) and sample count.

        On data load failure, displays an error message in the section.
        """
        if self._quality_trends_section is None:
            return

        self._quality_trends_section.clear()

        try:
            # Get daily quality averages for the past 30 days
            prose_trends = self._db.get_daily_quality_averages(
                metric="prose_quality",
                days=30,
                agent_role=self._filter_agent_role,
            )
            speed_trends = self._db.get_daily_quality_averages(
                metric="tokens_per_second",
                days=30,
                agent_role=self._filter_agent_role,
            )
            logger.debug(f"Loaded {len(prose_trends)} prose quality trend points")
        except Exception as e:
            logger.error(f"Failed to load quality trends: {e}", exc_info=True)
            with self._quality_trends_section:
                with ui.card().classes("w-full"):
                    ui.label("Failed to load quality trends.").classes("text-red-500 p-4")
            return

        with self._quality_trends_section:
            with ui.card().classes("w-full"):
                with ui.row().classes("w-full items-center mb-4"):
                    ui.icon("trending_up").classes("text-green-500")
                    ui.label("Quality Trends (30 Days)").classes("text-lg font-semibold")

                if not prose_trends and not speed_trends:
                    ui.label("Not enough data yet. Generate more content to see trends!").classes(
                        "text-gray-500 dark:text-gray-400 py-8 text-center"
                    )
                    return

                # Create a simple text-based trend visualization
                if prose_trends:
                    ui.label("Prose Quality Over Time").classes("font-semibold mt-4 mb-2")
                    with ui.element("div").classes("w-full overflow-x-auto"):
                        # Build trend table
                        columns = [
                            {"name": "date", "label": "Date", "field": "date", "sortable": True},
                            {
                                "name": "avg_quality",
                                "label": "Avg Quality",
                                "field": "avg_quality",
                                "sortable": True,
                            },
                            {
                                "name": "samples",
                                "label": "Samples",
                                "field": "samples",
                                "sortable": True,
                            },
                        ]
                        rows = [
                            {
                                "date": trend["date"],
                                "avg_quality": f"{trend['avg_value']:.1f}/10",
                                "samples": trend["sample_count"],
                            }
                            for trend in prose_trends[:10]  # Show last 10 days
                        ]
                        ui.table(columns=columns, rows=rows).classes("w-full")

                if speed_trends:
                    ui.label("Generation Speed Over Time").classes("font-semibold mt-4 mb-2")
                    with ui.element("div").classes("w-full overflow-x-auto"):
                        columns = [
                            {"name": "date", "label": "Date", "field": "date", "sortable": True},
                            {
                                "name": "avg_speed",
                                "label": "Avg Speed (t/s)",
                                "field": "avg_speed",
                                "sortable": True,
                            },
                            {
                                "name": "samples",
                                "label": "Samples",
                                "field": "samples",
                                "sortable": True,
                            },
                        ]
                        rows = [
                            {
                                "date": trend["date"],
                                "avg_speed": f"{trend['avg_value']:.1f}",
                                "samples": trend["sample_count"],
                            }
                            for trend in speed_trends[:10]
                        ]
                        ui.table(columns=columns, rows=rows).classes("w-full")

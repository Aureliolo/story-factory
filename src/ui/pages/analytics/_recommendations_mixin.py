"""Recommendations section mixin for AnalyticsPage."""

import logging

from nicegui import ui

from src.ui.pages.analytics._page import AnalyticsPageBase

logger = logging.getLogger(__name__)


class RecommendationsMixin(AnalyticsPageBase):
    """Mixin providing recommendations section functionality."""

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

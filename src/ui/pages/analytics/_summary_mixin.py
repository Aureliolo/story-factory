"""Summary section mixin for AnalyticsPage."""

import logging

from nicegui import ui

from src.ui.pages.analytics._page import AnalyticsPageBase

logger = logging.getLogger(__name__)


class SummaryMixin(AnalyticsPageBase):
    """Mixin providing summary section functionality."""

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

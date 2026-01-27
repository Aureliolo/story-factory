"""Analytics page - content statistics section helpers."""

import logging

from nicegui import ui

from src.ui.pages.analytics._summary import build_stat_card

logger = logging.getLogger("src.ui.pages.analytics._content")


def build_content_statistics_section(page) -> None:
    """Build the content statistics section.

    Args:
        page: The AnalyticsPage instance.
    """
    if page._content_stats_section is None:
        return

    page._content_stats_section.clear()

    try:
        # Get content statistics
        stats = page._db.get_content_statistics(agent_role=page._filter_agent_role)
        logger.debug(f"Loaded content statistics: {stats['generation_count']} generations")
    except Exception as e:
        logger.error(f"Failed to load content statistics: {e}", exc_info=True)
        with page._content_stats_section:
            with ui.card().classes("w-full"):
                ui.label("Failed to load content statistics.").classes("text-red-500 p-4")
        return

    with page._content_stats_section:
        with ui.card().classes("w-full"):
            with ui.row().classes("w-full items-center mb-4"):
                ui.icon("insights").classes("text-indigo-500")
                ui.label("Content Statistics").classes("text-lg font-semibold")
                ui.space()
                ui.label(f"{stats['generation_count']} generations").classes(
                    "text-sm text-gray-500"
                )

            if stats["generation_count"] == 0:
                ui.label("No content generated yet. Start writing to see statistics!").classes(
                    "text-gray-500 dark:text-gray-400 py-8 text-center"
                )
                return

            # Statistics grid
            with ui.element("div").classes("grid grid-cols-2 md:grid-cols-4 gap-4"):
                build_stat_card(
                    "Total Tokens",
                    f"{stats['total_tokens']:,}",
                    "functions",
                    "text-indigo-500",
                )
                build_stat_card(
                    "Avg Tokens/Gen",
                    f"{stats['avg_tokens']:.0f}" if stats["avg_tokens"] else "N/A",
                    "bar_chart",
                    "text-blue-500",
                )
                build_stat_card(
                    "Token Range",
                    f"{stats['min_tokens']}-{stats['max_tokens']}"
                    if stats["min_tokens"] and stats["max_tokens"]
                    else "N/A",
                    "straighten",
                    "text-purple-500",
                )
                build_stat_card(
                    "Avg Gen Time",
                    f"{stats['avg_generation_time']:.1f}s"
                    if stats["avg_generation_time"]
                    else "N/A",
                    "schedule",
                    "text-orange-500",
                )

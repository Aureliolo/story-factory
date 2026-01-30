"""Analytics page - summary section helpers."""

import logging

from nicegui import ui

logger = logging.getLogger(__name__)


def build_summary_section(page) -> None:
    """Build the summary cards section.

    Args:
        page: The AnalyticsPage instance.
    """
    if page._summary_section is None:
        return

    page._summary_section.clear()

    try:
        # Get summary stats
        total_scores = page._db.get_score_count(
            agent_role=page._filter_agent_role, genre=page._filter_genre
        )
        avg_quality = page._db.get_average_score(
            "prose_quality",
            agent_role=page._filter_agent_role,
            genre=page._filter_genre,
        )
        avg_instruction = page._db.get_average_score(
            "instruction_following",
            agent_role=page._filter_agent_role,
            genre=page._filter_genre,
        )
        avg_consistency = page._db.get_average_score(
            "consistency_score",
            agent_role=page._filter_agent_role,
            genre=page._filter_genre,
        )
        avg_speed = page._db.get_average_score(
            "tokens_per_second",
            agent_role=page._filter_agent_role,
            genre=page._filter_genre,
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
        with page._summary_section:
            ui.label("Failed to load analytics data. Check logs for details.").classes(
                "text-red-500 p-4"
            )
        return

    with page._summary_section:
        with ui.element("div").classes("grid grid-cols-2 md:grid-cols-5 gap-4"):
            build_stat_card(
                "Total Samples",
                str(total_scores),
                "analytics",
                "text-blue-500",
            )
            build_stat_card(
                "Avg Prose Quality",
                f"{avg_quality:.1f}/10" if avg_quality else "N/A",
                "auto_stories",
                "text-purple-500",
            )
            build_stat_card(
                "Avg Instruction",
                f"{avg_instruction:.1f}/10" if avg_instruction else "N/A",
                "checklist",
                "text-green-500",
            )
            build_stat_card(
                "Avg Consistency",
                f"{avg_consistency:.1f}/10" if avg_consistency else "N/A",
                "timeline",
                "text-orange-500",
            )
            build_stat_card(
                "Avg Speed",
                f"{avg_speed:.1f} t/s" if avg_speed else "N/A",
                "speed",
                "text-cyan-500",
            )


def build_stat_card(title: str, value: str, icon: str, color_class: str) -> None:
    """Build a single stat card.

    Args:
        title: Card title label.
        value: Card value label.
        icon: Material icon name.
        color_class: Tailwind color class for the icon.
    """
    with ui.card().classes("p-4"):
        with ui.row().classes("items-center gap-2 mb-2"):
            ui.icon(icon).classes(color_class)
            ui.label(title).classes("text-sm text-gray-400")
        ui.label(value).classes("text-2xl font-bold")

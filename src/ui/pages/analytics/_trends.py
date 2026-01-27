"""Analytics page - quality trends, world quality, and recommendations section helpers."""

import logging

from nicegui import ui

from src.ui.pages.analytics._summary import build_stat_card
from src.utils import extract_model_name

logger = logging.getLogger("src.ui.pages.analytics._trends")


def build_quality_trends_section(page) -> None:
    """Populate the quality trends UI section with recent prose quality and speed data.

    Fetches daily averages for "prose_quality" and "tokens_per_second" over the past 30
    days for the current filter, then renders tables for each metric.

    Args:
        page: The AnalyticsPage instance.
    """
    if page._quality_trends_section is None:
        return

    page._quality_trends_section.clear()

    try:
        # Get daily quality averages for the past 30 days
        prose_trends = page._db.get_daily_quality_averages(
            metric="prose_quality",
            days=30,
            agent_role=page._filter_agent_role,
        )
        speed_trends = page._db.get_daily_quality_averages(
            metric="tokens_per_second",
            days=30,
            agent_role=page._filter_agent_role,
        )
        logger.debug(f"Loaded {len(prose_trends)} prose quality trend points")
    except Exception as e:
        logger.error(f"Failed to load quality trends: {e}", exc_info=True)
        with page._quality_trends_section:
            with ui.card().classes("w-full"):
                ui.label("Failed to load quality trends.").classes("text-red-500 p-4")
        return

    with page._quality_trends_section:
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
                _build_prose_trends_table(prose_trends)

            if speed_trends:
                _build_speed_trends_table(speed_trends)


def _build_prose_trends_table(prose_trends: list) -> None:
    """Build the prose quality over time table.

    Args:
        prose_trends: List of daily trend dicts with date, avg_value, sample_count.
    """
    ui.label("Prose Quality Over Time").classes("font-semibold mt-4 mb-2")
    with ui.element("div").classes("w-full overflow-x-auto"):
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


def _build_speed_trends_table(speed_trends: list) -> None:
    """Build the generation speed over time table.

    Args:
        speed_trends: List of daily trend dicts with date, avg_value, sample_count.
    """
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


def build_world_quality_section(page) -> None:
    """Build the world quality metrics section.

    Args:
        page: The AnalyticsPage instance.
    """
    if page._world_quality_section is None:
        return

    page._world_quality_section.clear()

    try:
        summary = page._db.get_world_quality_summary()
        entity_count = summary.get("total_entities", 0)
        logger.debug(f"Loaded world quality summary: {entity_count} entities")
    except Exception as e:
        logger.error(f"Failed to load world quality data: {e}", exc_info=True)
        with page._world_quality_section:
            with ui.card().classes("w-full"):
                ui.label("Failed to load world quality data.").classes("text-red-500 p-4")
        return

    with page._world_quality_section:
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
                build_stat_card(
                    "Avg Quality",
                    f"{avg_quality:.1f}/10" if avg_quality else "N/A",
                    "star",
                    "text-yellow-500",
                )
                build_stat_card(
                    "Avg Iterations",
                    f"{avg_iterations:.1f}" if avg_iterations else "N/A",
                    "loop",
                    "text-blue-500",
                )
                build_stat_card(
                    "Avg Gen Time",
                    f"{avg_time:.1f}s" if avg_time else "N/A",
                    "timer",
                    "text-purple-500",
                )
                build_stat_card(
                    "Total Entities",
                    str(entity_count),
                    "category",
                    "text-green-500",
                )

            # Breakdown by entity type
            by_type = summary.get("by_entity_type", [])
            if by_type:
                _build_world_quality_by_type(by_type)

            # Breakdown by model
            by_model = summary.get("by_model", [])
            if by_model:
                _build_world_quality_by_model(by_model)


def _build_world_quality_by_type(by_type: list) -> None:
    """Build the world quality breakdown by entity type table.

    Args:
        by_type: List of dicts with entity_type, count, avg_quality.
    """
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


def _build_world_quality_by_model(by_model: list) -> None:
    """Build the world quality breakdown by model table.

    Args:
        by_model: List of dicts with model_id, count, avg_quality.
    """
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


def build_recommendations_section(page) -> None:
    """Build the recommendations history section.

    Args:
        page: The AnalyticsPage instance.
    """
    if page._recommendations_section is None:
        return

    page._recommendations_section.clear()

    try:
        recommendations = page._db.get_recent_recommendations(limit=10)
        logger.debug(f"Loaded {len(recommendations)} recent recommendations")
    except Exception as e:
        logger.error(f"Failed to load recommendations: {e}", exc_info=True)
        with page._recommendations_section:
            with ui.card().classes("w-full"):
                ui.label("Failed to load recommendations. Check logs for details.").classes(
                    "text-red-500 p-4"
                )
        return

    with page._recommendations_section:
        with ui.card().classes("w-full"):
            with ui.row().classes("w-full items-center mb-4"):
                ui.icon("lightbulb").classes("text-yellow-500")
                ui.label("Recent Recommendations").classes("text-lg font-semibold")

            if not recommendations:
                ui.label("No recommendations yet.").classes("text-gray-500 dark:text-gray-400")
                return

            for rec in recommendations:
                _build_recommendation_card(rec)


def _build_recommendation_card(rec) -> None:
    """Build a single recommendation card.

    Args:
        rec: A recommendation record object.
    """
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
                        ui.label(f"({rec.affected_role})").classes("text-sm text-gray-500")

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
                ui.label(rec.reason).classes("text-sm text-gray-600 dark:text-gray-400")

            # Confidence badge
            confidence_color = "green" if rec.confidence >= 0.8 else "orange"
            ui.badge(f"{rec.confidence:.0%}").props(f"color={confidence_color}")

"""Analytics page - generation costs section helpers."""

import logging

from nicegui import ui

from src.ui.pages.analytics._summary import build_stat_card
from src.utils import extract_model_name

logger = logging.getLogger("src.ui.pages.analytics._costs")


def format_time(seconds: float) -> str:
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


def format_tokens(tokens: int) -> str:
    """Format a token count using K (thousands) or M (millions) suffix when appropriate.

    Args:
        tokens: Number of tokens.

    Returns:
        Formatted string representation of the token count (e.g., '999', '1.2K', '3.5M').
    """
    if tokens < 1000:
        return str(tokens)
    elif tokens < 1_000_000:
        return f"{tokens / 1000:.1f}K"
    else:
        return f"{tokens / 1_000_000:.1f}M"


def build_generation_costs_section(page) -> None:
    """Populate the generation costs UI section with 30-day cost summaries.

    Fetches a 30-day cost summary and optional model- and entity-type breakdowns
    for the current project when available (or across all projects), and renders
    summary grid, tables for cost-by-model and cost-by-entity-type, and an
    efficiency insight card when efficiency is below 80%.

    Args:
        page: The AnalyticsPage instance.
    """
    if page._generation_costs_section is None:
        return

    page._generation_costs_section.clear()

    # Get project ID filter if we have a current project
    project_id = page.state.project_id if page.state.has_project else None

    try:
        # Get cost summary from database
        cost_summary = page._db.get_cost_summary(project_id=project_id, days=30)
        model_breakdown = page._db.get_model_cost_breakdown(project_id=project_id, days=30)
        entity_breakdown = page._db.get_entity_type_cost_breakdown(project_id=project_id, days=30)
        logger.debug(
            f"Loaded generation costs: {cost_summary['total_runs']} runs, "
            f"{cost_summary['total_tokens']} tokens"
        )
    except Exception as e:
        logger.error(f"Failed to load generation costs: {e}", exc_info=True)
        with page._generation_costs_section:
            with ui.card().classes("w-full"):
                ui.label("Failed to load generation costs.").classes("text-red-500 p-4")
        return

    with page._generation_costs_section:
        with ui.card().classes("w-full"):
            with ui.row().classes("w-full items-center mb-4"):
                ui.icon("paid").classes("text-emerald-500")
                ui.label("Generation Costs").classes("text-lg font-semibold")
                ui.space()
                scope_label = "current project" if project_id else "all projects"
                ui.label(f"Last 30 days ({scope_label})").classes("text-sm text-gray-500")

            if cost_summary["total_runs"] == 0:
                ui.label("No generation runs recorded yet. Generate content to see costs!").classes(
                    "text-gray-500 dark:text-gray-400 py-8 text-center"
                )
                return

            # Summary stats
            total_time = cost_summary["total_time_seconds"]
            time_str = format_time(total_time) if total_time else "N/A"
            total_tokens = cost_summary["total_tokens"]
            tokens_str = format_tokens(total_tokens) if total_tokens else "N/A"
            efficiency = cost_summary.get("efficiency_ratio", 1.0)

            with ui.element("div").classes("grid grid-cols-2 md:grid-cols-4 gap-4 mb-4"):
                build_stat_card(
                    "Total Tokens",
                    tokens_str,
                    "token",
                    "text-emerald-500",
                )
                build_stat_card(
                    "Total Time",
                    time_str,
                    "schedule",
                    "text-blue-500",
                )
                build_stat_card(
                    "Generation Runs",
                    str(cost_summary["total_runs"]),
                    "play_circle",
                    "text-purple-500",
                )
                build_stat_card(
                    "Efficiency",
                    f"{efficiency:.0%}",
                    "trending_up" if efficiency >= 0.8 else "trending_down",
                    "text-green-500" if efficiency >= 0.8 else "text-orange-500",
                )

            # Model breakdown
            if model_breakdown:
                _build_model_cost_breakdown(model_breakdown)

            # Entity type breakdown
            if entity_breakdown:
                _build_entity_type_cost_breakdown(entity_breakdown)

            # Efficiency insight
            if efficiency < 0.8:
                with (
                    ui.card()
                    .classes("w-full mt-4 bg-amber-50 dark:bg-amber-900/20")
                    .props("flat bordered")
                ):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("lightbulb", color="amber")
                        ui.label(
                            f"Efficiency is {efficiency:.0%}. "
                            f"{cost_summary.get('wasted_iterations', 0)} iterations were rejected. "
                            "Consider adjusting quality thresholds or improving prompts."
                        ).classes("text-sm")


def _build_model_cost_breakdown(model_breakdown: list) -> None:
    """Build the cost-by-model table.

    Args:
        model_breakdown: List of model cost breakdown dicts.
    """
    ui.label("Cost by Model").classes("font-semibold mt-4 mb-2")
    columns = [
        {"name": "model", "label": "Model", "field": "model", "sortable": True},
        {"name": "calls", "label": "Calls", "field": "calls", "sortable": True},
        {"name": "tokens", "label": "Tokens", "field": "tokens", "sortable": True},
        {"name": "time", "label": "Time", "field": "time", "sortable": True},
        {
            "name": "speed",
            "label": "Speed (t/s)",
            "field": "speed",
            "sortable": True,
        },
    ]
    rows = []
    for item in model_breakdown:
        model_name = extract_model_name(item["model_id"])
        rows.append(
            {
                "model": model_name,
                "calls": item["call_count"],
                "tokens": format_tokens(item["total_tokens"]),
                "time": format_time(item["total_time_seconds"]),
                "speed": f"{item['avg_tokens_per_second']:.1f}"
                if item.get("avg_tokens_per_second")
                else "-",
            }
        )
    with ui.element("div").classes("w-full overflow-x-auto"):
        ui.table(columns=columns, rows=rows).classes("w-full")


def _build_entity_type_cost_breakdown(entity_breakdown: list) -> None:
    """Build the cost-by-entity-type table.

    Args:
        entity_breakdown: List of entity type cost breakdown dicts.
    """
    ui.label("Cost by Entity Type").classes("font-semibold mt-4 mb-2")
    columns = [
        {"name": "type", "label": "Entity Type", "field": "type", "sortable": True},
        {"name": "count", "label": "Count", "field": "count", "sortable": True},
        {"name": "time", "label": "Time", "field": "time", "sortable": True},
        {
            "name": "iterations",
            "label": "Avg Iterations",
            "field": "iterations",
            "sortable": True,
        },
        {
            "name": "quality",
            "label": "Avg Quality",
            "field": "quality",
            "sortable": True,
        },
    ]
    rows = []
    for item in entity_breakdown:
        rows.append(
            {
                "type": item["entity_type"].title(),
                "count": item["count"],
                "time": format_time(item["total_time_seconds"]),
                "iterations": f"{item['avg_iterations']:.1f}"
                if item.get("avg_iterations")
                else "-",
                "quality": f"{item['avg_quality']:.1f}" if item.get("avg_quality") else "-",
            }
        )
    with ui.element("div").classes("w-full overflow-x-auto"):
        ui.table(columns=columns, rows=rows).classes("w-full")

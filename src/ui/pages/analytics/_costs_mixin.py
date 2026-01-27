"""Generation costs section mixin for AnalyticsPage."""

import logging

from nicegui import ui

from src.ui.pages.analytics._page import AnalyticsPageBase
from src.utils import extract_model_name

logger = logging.getLogger(__name__)


class CostsMixin(AnalyticsPageBase):
    """Mixin providing generation costs section functionality."""

    def _build_generation_costs_section(self) -> None:
        """
        Populate the generation costs UI section with 30-day cost summaries, breakdowns, and efficiency insights.

        Fetches a 30-day cost summary and optional model- and entity-type breakdowns for the current project when available (or across all projects), and renders a summary grid (total tokens, total time, generation runs, efficiency), optional tables for cost-by-model and cost-by-entity-type, and an efficiency insight card when efficiency is below 80%. On data load failure an error message is displayed; when there are no runs a friendly empty-state message is shown.
        """
        if self._generation_costs_section is None:
            return

        self._generation_costs_section.clear()

        # Get project ID filter if we have a current project
        project_id = self.state.project_id if self.state.has_project else None

        try:
            # Get cost summary from database
            cost_summary = self._db.get_cost_summary(project_id=project_id, days=30)
            model_breakdown = self._db.get_model_cost_breakdown(project_id=project_id, days=30)
            entity_breakdown = self._db.get_entity_type_cost_breakdown(
                project_id=project_id, days=30
            )
            logger.debug(
                f"Loaded generation costs: {cost_summary['total_runs']} runs, "
                f"{cost_summary['total_tokens']} tokens"
            )
        except Exception as e:
            logger.error(f"Failed to load generation costs: {e}", exc_info=True)
            with self._generation_costs_section:
                with ui.card().classes("w-full"):
                    ui.label("Failed to load generation costs.").classes("text-red-500 p-4")
            return

        with self._generation_costs_section:
            with ui.card().classes("w-full"):
                with ui.row().classes("w-full items-center mb-4"):
                    ui.icon("paid").classes("text-emerald-500")
                    ui.label("Generation Costs").classes("text-lg font-semibold")
                    ui.space()
                    scope_label = "current project" if project_id else "all projects"
                    ui.label(f"Last 30 days ({scope_label})").classes("text-sm text-gray-500")

                if cost_summary["total_runs"] == 0:
                    ui.label(
                        "No generation runs recorded yet. Generate content to see costs!"
                    ).classes("text-gray-500 dark:text-gray-400 py-8 text-center")
                    return

                # Summary stats
                total_time = cost_summary["total_time_seconds"]
                time_str = self._format_time(total_time) if total_time else "N/A"
                total_tokens = cost_summary["total_tokens"]
                tokens_str = self._format_tokens(total_tokens) if total_tokens else "N/A"
                efficiency = cost_summary.get("efficiency_ratio", 1.0)

                with ui.element("div").classes("grid grid-cols-2 md:grid-cols-4 gap-4 mb-4"):
                    self._build_stat_card(
                        "Total Tokens",
                        tokens_str,
                        "token",
                        "text-emerald-500",
                    )
                    self._build_stat_card(
                        "Total Time",
                        time_str,
                        "schedule",
                        "text-blue-500",
                    )
                    self._build_stat_card(
                        "Generation Runs",
                        str(cost_summary["total_runs"]),
                        "play_circle",
                        "text-purple-500",
                    )
                    self._build_stat_card(
                        "Efficiency",
                        f"{efficiency:.0%}",
                        "trending_up" if efficiency >= 0.8 else "trending_down",
                        "text-green-500" if efficiency >= 0.8 else "text-orange-500",
                    )

                # Model breakdown
                if model_breakdown:
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
                                "tokens": self._format_tokens(item["total_tokens"]),
                                "time": self._format_time(item["total_time_seconds"]),
                                "speed": f"{item['avg_tokens_per_second']:.1f}"
                                if item.get("avg_tokens_per_second")
                                else "-",
                            }
                        )
                    with ui.element("div").classes("w-full overflow-x-auto"):
                        ui.table(columns=columns, rows=rows).classes("w-full")

                # Entity type breakdown
                if entity_breakdown:
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
                                "time": self._format_time(item["total_time_seconds"]),
                                "iterations": f"{item['avg_iterations']:.1f}"
                                if item.get("avg_iterations")
                                else "-",
                                "quality": f"{item['avg_quality']:.1f}"
                                if item.get("avg_quality")
                                else "-",
                            }
                        )
                    with ui.element("div").classes("w-full overflow-x-auto"):
                        ui.table(columns=columns, rows=rows).classes("w-full")

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

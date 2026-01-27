"""World quality section mixin for AnalyticsPage."""

import logging

from nicegui import ui

from src.ui.pages.analytics._page import AnalyticsPageBase
from src.utils import extract_model_name

logger = logging.getLogger(__name__)


class WorldQualityMixin(AnalyticsPageBase):
    """Mixin providing world quality section functionality."""

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

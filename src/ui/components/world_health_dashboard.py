"""World Health Dashboard component for displaying world quality metrics."""

import logging
from collections.abc import Callable
from typing import Any

from nicegui import ui

from src.memory.world_health import WorldHealthMetrics
from src.ui.theme import get_entity_color, get_entity_icon

logger = logging.getLogger(__name__)


def _get_health_color(score: float) -> str:
    """Get color based on health score.

    Args:
        score: Health score (0-100).

    Returns:
        Color string (red, orange, or green).
    """
    if score >= 80:
        return "green"
    elif score >= 50:
        return "orange"
    else:
        return "red"


def _get_severity_color(severity: str) -> str:
    """Get color based on severity level.

    Args:
        severity: Severity string (low, medium, high, critical).

    Returns:
        Color string.
    """
    return {
        "low": "blue",
        "medium": "orange",
        "high": "red",
        "critical": "deep-purple",
    }.get(severity.lower(), "grey")


class WorldHealthDashboard:
    """Dashboard component for displaying world health metrics.

    Features:
    - Overall health score with circular progress
    - Entity count summary
    - Orphan entity warnings
    - Circular relationship warnings
    - Quality distribution chart
    - Actionable recommendations
    """

    def __init__(
        self,
        metrics: WorldHealthMetrics,
        on_fix_orphan: Callable[[str], Any] | None = None,
        on_view_circular: Callable[[dict], Any] | None = None,
        on_improve_quality: Callable[[str], Any] | None = None,
        on_refresh: Callable[[], Any] | None = None,
    ):
        """Initialize world health dashboard.

        Args:
            metrics: World health metrics to display.
            on_fix_orphan: Callback when user wants to fix an orphan entity.
            on_view_circular: Callback to view a single circular relationship chain.
            on_improve_quality: Callback to improve a low-quality entity.
            on_refresh: Callback to refresh metrics after changes.
        """
        self.metrics = metrics
        self.on_fix_orphan = on_fix_orphan
        self.on_view_circular = on_view_circular
        self.on_improve_quality = on_improve_quality
        self.on_refresh = on_refresh

    def build(self) -> None:
        """Build the dashboard UI."""
        logger.debug(
            f"Rendering World Health dashboard (score={self.metrics.health_score:.1f}, "
            f"entities={self.metrics.total_entities}, has_refresh={self.on_refresh is not None})"
        )
        with ui.card().classes("w-full"):
            # Header
            with ui.row().classes("w-full items-center gap-2 mb-4"):
                ui.icon("health_and_safety").classes("text-blue-500")
                ui.label("World Health").classes("text-lg font-semibold")
                ui.space()
                if self.on_refresh:
                    ui.button(
                        icon="refresh",
                        on_click=self.on_refresh,
                    ).props("flat round size=sm").tooltip("Refresh health metrics")
                self._build_health_score()

            # Main content
            with ui.row().classes("w-full gap-4 flex-wrap"):
                # Left column: Stats
                with ui.column().classes("flex-1 min-w-[200px] gap-4"):
                    self._build_entity_counts()
                    self._build_relationship_stats()

                # Center column: Warnings
                with ui.column().classes("flex-1 min-w-[250px] gap-4"):
                    self._build_orphan_section()
                    self._build_circular_section()

                # Right column: Quality & Recommendations
                with ui.column().classes("flex-1 min-w-[200px] gap-4"):
                    self._build_quality_section()
                    self._build_recommendations()

    def _build_health_score(self) -> None:
        """Build the overall health score display."""
        score = self.metrics.health_score
        color = _get_health_color(score)

        with ui.row().classes("items-center gap-2"):
            ui.circular_progress(
                value=score / 100,
                show_value=True,
                size="md",
                color=color,
            ).tooltip(f"Overall world health: {score:.0f}/100")

            health_text = (
                "Excellent"
                if score >= 80
                else "Good"
                if score >= 60
                else "Needs Work"
                if score >= 40
                else "Critical"
            )
            ui.label(health_text).classes(f"text-sm text-{color}-600 dark:text-{color}-400")

    def _build_entity_counts(self) -> None:
        """Build entity count summary."""
        with ui.card().classes("p-3 bg-gray-50 dark:bg-gray-800"):
            ui.label("Entities").classes("text-sm font-medium mb-2")

            total = self.metrics.total_entities
            ui.label(f"{total} total").classes("text-2xl font-bold mb-2")

            # Entity type breakdown
            with ui.row().classes("flex-wrap gap-x-3 gap-y-1"):
                for entity_type, count in self.metrics.entity_counts.items():
                    if count > 0:
                        color = get_entity_color(entity_type)
                        icon = get_entity_icon(entity_type)
                        with ui.row().classes("items-center gap-1"):
                            ui.icon(icon, size="xs").style(f"color: {color};")
                            ui.label(f"{count}").classes("text-sm font-medium")

    def _build_relationship_stats(self) -> None:
        """Build relationship statistics."""
        with ui.card().classes("p-3 bg-gray-50 dark:bg-gray-800"):
            ui.label("Relationships").classes("text-sm font-medium mb-2")

            with ui.row().classes("items-baseline gap-2"):
                ui.label(f"{self.metrics.total_relationships}").classes("text-2xl font-bold")
                ui.label("total").classes("text-sm text-gray-500")

            density = self.metrics.relationship_density
            density_color = "green" if density >= 1.5 else "orange" if density >= 1.0 else "red"
            with ui.row().classes("items-center gap-1 mt-1"):
                ui.icon("density_small", size="xs").classes(f"text-{density_color}-500")
                ui.label(f"Density: {density:.2f}").classes(
                    f"text-xs text-{density_color}-600 dark:text-{density_color}-400"
                )

    def _build_orphan_section(self) -> None:
        """Build orphan entity warning section."""
        orphan_count = self.metrics.orphan_count

        if orphan_count == 0:
            with ui.row().classes("items-center gap-2 p-2 bg-green-50 dark:bg-green-900 rounded"):
                ui.icon("check_circle", size="sm").classes("text-green-500")
                ui.label("No orphan entities").classes("text-sm text-green-700 dark:text-green-300")
            return

        # Warning card
        with ui.card().classes("p-3 bg-yellow-50 dark:bg-yellow-900"):
            with ui.row().classes("items-center gap-2 mb-2"):
                ui.icon("warning", size="sm").classes("text-yellow-600")
                ui.label(f"{orphan_count} Orphan Entities").classes(
                    "text-sm font-medium text-yellow-800 dark:text-yellow-200"
                )

            # List orphans (max 5)
            with ui.column().classes("gap-1"):
                for orphan in self.metrics.orphan_entities[:5]:
                    entity_type = orphan.get("type", "unknown")
                    color = get_entity_color(entity_type)
                    icon = get_entity_icon(entity_type)

                    orphan_id = orphan.get("id")
                    orphan_name = orphan.get("name", "Unknown")
                    with ui.row().classes("items-center gap-2 w-full"):
                        ui.icon(icon, size="xs").style(f"color: {color};")
                        ui.label(orphan_name).classes("text-sm flex-grow")

                        if self.on_fix_orphan and orphan_id:
                            ui.button(
                                icon="link",
                                on_click=lambda e_id=orphan_id: self.on_fix_orphan(e_id),
                            ).props("flat round size=xs").tooltip("Add relationship")

                if orphan_count > 5:
                    ui.label(f"+{orphan_count - 5} more...").classes(
                        "text-xs text-gray-500 dark:text-gray-400"
                    )

    def _build_circular_section(self) -> None:
        """Build circular relationship warning section."""
        circular_count = self.metrics.circular_count

        if circular_count == 0:
            with ui.row().classes("items-center gap-2 p-2 bg-green-50 dark:bg-green-900 rounded"):
                ui.icon("check_circle", size="sm").classes("text-green-500")
                ui.label("No circular relationships").classes(
                    "text-sm text-green-700 dark:text-green-300"
                )
            return

        # Warning card
        with ui.card().classes("p-3 bg-orange-50 dark:bg-orange-900"):
            with ui.row().classes("items-center gap-2 mb-2"):
                ui.icon("autorenew", size="sm").classes("text-orange-600")
                ui.label(f"{circular_count} Circular Chains").classes(
                    "text-sm font-medium text-orange-800 dark:text-orange-200"
                )

            # List cycles (max 3)
            with ui.column().classes("gap-1"):
                for cycle in self.metrics.circular_relationships[:3]:
                    edges = cycle.get("edges", [])
                    length = cycle.get("length", len(edges))

                    # Build readable cycle description using entity names
                    if edges:
                        # Extract unique entity names in cycle order
                        cycle_names: list[str] = []
                        for edge in edges:
                            source_name = edge.get("source_name", edge.get("source", "?"))
                            if not cycle_names or cycle_names[-1] != source_name:
                                cycle_names.append(source_name)
                        # Add first name again to show it's a cycle
                        if cycle_names:
                            cycle_names.append(cycle_names[0])
                        cycle_display = " → ".join(cycle_names[:5])  # Limit to avoid overflow
                        if len(cycle_names) > 5:
                            cycle_display += " → ..."
                    else:
                        cycle_display = f"Cycle of {length} entities"

                    with ui.row().classes("items-center gap-2 w-full"):
                        ui.label(cycle_display).classes("text-sm flex-grow").tooltip(
                            f"Circular chain involving {length} entities"
                        )

                        if self.on_view_circular:
                            ui.button(
                                icon="visibility",
                                on_click=lambda c=cycle: self.on_view_circular(c),
                            ).props("flat round size=xs").tooltip("View cycle")

                if circular_count > 3:
                    ui.label(f"+{circular_count - 3} more...").classes(
                        "text-xs text-gray-500 dark:text-gray-400"
                    )

    def _build_quality_section(self) -> None:
        """Build quality distribution section."""
        with ui.card().classes("p-3 bg-gray-50 dark:bg-gray-800"):
            ui.label("Quality Distribution").classes("text-sm font-medium mb-2")

            # Average quality
            avg = self.metrics.average_quality
            if avg > 0:
                avg_color = "green" if avg >= 7 else "orange" if avg >= 5 else "red"
                with ui.row().classes("items-baseline gap-2 mb-2"):
                    ui.label(f"{avg:.1f}").classes(f"text-xl font-bold text-{avg_color}-600")
                    ui.label("average").classes("text-sm text-gray-500")

            # Distribution bars
            dist = self.metrics.quality_distribution
            total = sum(dist.values()) if dist else 0

            if total > 0:
                with ui.column().classes("gap-1"):
                    for bracket, count in dist.items():
                        if count > 0:
                            pct = (count / total) * 100
                            color = {
                                "0-2": "red",
                                "2-4": "orange",
                                "4-6": "yellow",
                                "6-8": "lime",
                                "8-10": "green",
                            }.get(bracket, "grey")

                            with ui.row().classes("items-center gap-2 w-full"):
                                ui.label(bracket).classes("text-xs w-8")
                                with ui.element("div").classes(
                                    "flex-grow h-3 bg-gray-200 dark:bg-gray-700 rounded"
                                ):
                                    ui.element("div").classes(
                                        f"h-full bg-{color}-500 rounded"
                                    ).style(f"width: {pct}%;")
                                ui.label(f"{count}").classes("text-xs w-6 text-right")

            # Low quality warnings
            low_quality = self.metrics.low_quality_entities
            if low_quality:
                ui.separator().classes("my-2")
                ui.label(f"{len(low_quality)} low quality").classes(
                    "text-xs text-red-600 dark:text-red-400"
                )

    def _build_recommendations(self) -> None:
        """Build recommendations section."""
        recommendations = self.metrics.recommendations

        if not recommendations:
            return

        with ui.expansion("Recommendations", icon="lightbulb").classes("w-full"):
            for rec in recommendations:
                with ui.row().classes("items-start gap-2 mb-2"):
                    ui.icon("arrow_right", size="xs").classes("text-blue-500 mt-1")
                    ui.label(rec).classes("text-sm text-gray-700 dark:text-gray-300")


def build_health_summary_compact(metrics: WorldHealthMetrics) -> None:
    """Build a compact health summary widget.

    Args:
        metrics: World health metrics to display.
    """
    score = metrics.health_score
    color = _get_health_color(score)

    with ui.row().classes("items-center gap-3 p-2 bg-gray-50 dark:bg-gray-800 rounded"):
        # Health score
        ui.circular_progress(
            value=score / 100,
            show_value=True,
            size="sm",
            color=color,
        )

        # Quick stats
        with ui.column().classes("gap-0"):
            ui.label("World Health").classes("text-xs text-gray-500")
            with ui.row().classes("items-center gap-2"):
                if metrics.orphan_count > 0:
                    ui.badge(
                        f"{metrics.orphan_count} orphans",
                        color="yellow",
                    ).props("outline").classes("text-xs")
                if metrics.circular_count > 0:
                    ui.badge(
                        f"{metrics.circular_count} cycles",
                        color="orange",
                    ).props("outline").classes("text-xs")
                if metrics.orphan_count == 0 and metrics.circular_count == 0:
                    ui.badge(
                        "Healthy",
                        color="green",
                    ).props("outline").classes("text-xs")

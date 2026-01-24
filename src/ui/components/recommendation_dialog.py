"""Recommendation dialog component for adaptive learning.

Shows pending tuning recommendations with approve/reject options.
"""

import logging
from collections.abc import Callable

from nicegui import ui

from src.memory.mode_models import TuningRecommendation

logger = logging.getLogger(__name__)


class RecommendationDialog:
    """Modal dialog showing pending tuning recommendations.

    Allows users to review, approve, or reject recommendations
    from the adaptive learning system.
    """

    def __init__(
        self,
        recommendations: list[TuningRecommendation],
        on_apply: Callable[[list[TuningRecommendation]], None] | None = None,
        on_dismiss: Callable[[], None] | None = None,
    ):
        """Initialize recommendation dialog.

        Args:
            recommendations: List of recommendations to display.
            on_apply: Callback when user applies selected recommendations.
            on_dismiss: Callback when user dismisses without applying.
        """
        self.recommendations = recommendations
        self.on_apply = on_apply
        self.on_dismiss = on_dismiss
        self._dialog: ui.dialog | None = None
        self._selected: dict[int, bool] = {}

        # Initialize all as selected
        for i, _rec in enumerate(recommendations):
            self._selected[i] = True

    def show(self) -> None:
        """Show the recommendation dialog."""
        with ui.dialog() as self._dialog:
            self._dialog.props("persistent")
            with ui.card().classes("w-full max-w-2xl"):
                self._build_header()
                self._build_recommendations()
                self._build_actions()

        self._dialog.open()

    def _build_header(self) -> None:
        """Build dialog header."""
        with ui.row().classes("w-full items-center justify-between mb-4"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("psychology", size="sm").classes("text-blue-500")
                ui.label("Tuning Recommendations").classes("text-xl font-bold")

            ui.label(f"{len(self.recommendations)} suggestions").classes("text-sm text-gray-500")

        ui.label(
            "Based on your generation history, the learning system suggests "
            "these changes to improve quality or performance."
        ).classes("text-sm text-gray-600 dark:text-gray-400 mb-4")

    def _build_recommendations(self) -> None:
        """Build recommendation cards."""
        with ui.column().classes("w-full gap-3 max-h-96 overflow-y-auto"):
            for i, rec in enumerate(self.recommendations):
                self._build_recommendation_card(i, rec)

    def _build_recommendation_card(self, index: int, rec: TuningRecommendation) -> None:
        """Build a single recommendation card.

        Args:
            index: Index of the recommendation.
            rec: The recommendation to display.
        """
        with ui.card().classes("w-full").props("flat bordered"):
            with ui.row().classes("w-full items-start gap-3"):
                # Checkbox
                checkbox = ui.checkbox(value=self._selected.get(index, True))
                checkbox.on_value_change(lambda e, idx=index: self._toggle_selection(idx))

                # Content
                with ui.column().classes("flex-grow"):
                    # Type and role badge
                    with ui.row().classes("items-center gap-2 mb-1"):
                        rec_type_value = rec.recommendation_type
                        if hasattr(rec_type_value, "value"):
                            rec_type_str = str(rec_type_value.value)
                        else:
                            rec_type_str = str(rec_type_value)

                        type_colors = {
                            "model_swap": "bg-purple-100 text-purple-800",
                            "temp_adjust": "bg-blue-100 text-blue-800",
                            "mode_change": "bg-green-100 text-green-800",
                            "vram_strategy": "bg-orange-100 text-orange-800",
                        }
                        type_color = type_colors.get(rec_type_str, "bg-gray-100 text-gray-800")

                        ui.label(rec_type_str.replace("_", " ").title()).classes(
                            f"text-xs px-2 py-0.5 rounded {type_color}"
                        )

                        if rec.affected_role:
                            ui.label(rec.affected_role.title()).classes(
                                "text-xs px-2 py-0.5 rounded bg-gray-100 text-gray-700"
                            )

                    # Change description
                    ui.label(f"{rec.current_value} → {rec.suggested_value}").classes("font-medium")

                    # Reason
                    ui.label(rec.reason).classes("text-sm text-gray-600 dark:text-gray-400")

                    # Confidence and expected improvement
                    with ui.row().classes("items-center gap-4 mt-1"):
                        confidence_pct = int(rec.confidence * 100)
                        confidence_color = (
                            "text-green-600"
                            if confidence_pct >= 80
                            else "text-yellow-600"
                            if confidence_pct >= 60
                            else "text-red-600"
                        )
                        ui.label(f"Confidence: {confidence_pct}%").classes(
                            f"text-xs {confidence_color}"
                        )

                        if rec.expected_improvement:
                            ui.label(f"Expected: {rec.expected_improvement}").classes(
                                "text-xs text-blue-600"
                            )

    def _toggle_selection(self, index: int) -> None:
        """Toggle selection state for a recommendation."""
        self._selected[index] = not self._selected.get(index, True)

    def _build_actions(self) -> None:
        """Build action buttons."""
        with ui.row().classes("w-full justify-end gap-2 mt-4"):
            ui.button("Dismiss", on_click=self._on_dismiss).props("flat")
            ui.button("Apply Selected", on_click=self._on_apply).props("color=primary")

    def _on_apply(self) -> None:
        """Handle apply button click."""
        if self._dialog:
            self._dialog.close()

        # Get selected recommendations
        selected = [
            self.recommendations[i] for i, is_selected in self._selected.items() if is_selected
        ]

        logger.info(f"Applying {len(selected)} recommendations")

        if self.on_apply and selected:
            self.on_apply(selected)

    def _on_dismiss(self) -> None:
        """Handle dismiss button click."""
        if self._dialog:
            self._dialog.close()

        logger.debug("Recommendations dismissed")

        if self.on_dismiss:
            self.on_dismiss()


def show_recommendations(
    recommendations: list[TuningRecommendation],
    on_apply: Callable[[list[TuningRecommendation]], None] | None = None,
    on_dismiss: Callable[[], None] | None = None,
) -> None:
    """Show recommendation dialog as a convenience function.

    Args:
        recommendations: List of recommendations to display.
        on_apply: Callback when user applies selected recommendations.
        on_dismiss: Callback when user dismisses without applying.
    """
    if not recommendations:
        ui.notify("No recommendations to show", type="info")
        return

    dialog = RecommendationDialog(recommendations, on_apply, on_dismiss)
    dialog.show()

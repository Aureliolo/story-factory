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
        on_dismiss: Callable[[list[TuningRecommendation]], None] | None = None,
    ):
        """
        Create a modal dialog configured to display and manage a list of tuning recommendations.

        Parameters:
            recommendations (list[TuningRecommendation]): Recommendations to present in the dialog.
            on_apply (Callable[[list[TuningRecommendation]], None] | None): Optional callback invoked with the list of recommendations the user applied.
            on_dismiss (Callable[[list[TuningRecommendation]], None] | None): Optional callback invoked with the full list when the user dismisses the dialog.

        Notes:
            All provided recommendations are initially marked as selected. The dialog instance is not created until `show()` is called; internal selection state is stored in `_selected`.
        """
        self.recommendations = recommendations
        self.on_apply = on_apply
        self.on_dismiss = on_dismiss
        self._dialog: ui.dialog | None = None
        self._selected: dict[int, bool] = {}

        # Initialize all as selected
        for i, _rec in enumerate(recommendations):
            self._selected[i] = True

        logger.debug(
            f"Initialized RecommendationDialog with {len(self.recommendations)} recommendations, "
            f"on_apply={self.on_apply is not None}, on_dismiss={self.on_dismiss is not None}"
        )

    def show(self) -> None:
        """
        Open and display the recommendation modal dialog.

        Builds the dialog contents (header, recommendations list, and action buttons), marks the dialog as persistent, and opens it for user interaction.
        """
        logger.debug(
            f"Opening recommendation dialog with {len(self.recommendations)} recommendations"
        )
        with ui.dialog() as self._dialog:
            self._dialog.props("persistent")
            with ui.card().classes("w-full max-w-2xl"):
                self._build_header()
                self._build_recommendations()
                self._build_actions()

        self._dialog.open()

    def _build_header(self) -> None:
        """
        Render the header area of the recommendations dialog, including icon, title, suggestion count, and a short descriptive subtitle.

        Builds the UI row containing the "Tuning Recommendations" title with an icon, a label showing the number of suggestions, and a brief explanatory subheader about the purpose of the recommendations.
        """
        logger.debug("Building recommendation dialog header")
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
        """
        Render the recommendations as individual cards inside a scrollable column in the dialog.
        """
        logger.debug(f"Building {len(self.recommendations)} recommendation cards")
        with ui.column().classes("w-full gap-3 max-h-96 overflow-y-auto"):
            for i, rec in enumerate(self.recommendations):
                self._build_recommendation_card(i, rec)

    def _build_recommendation_card(self, index: int, rec: TuningRecommendation) -> None:
        """
        Render a UI card for a single tuning recommendation and bind its controls to the dialog's selection state.

        Renders a card showing the recommendation's type and affected role, current → suggested value, reason, confidence percentage, and optional expected improvement. The card's checkbox is bound to the dialog's internal selection mapping and updates selection via _set_selection when changed.

        Parameters:
            index (int): Position of the recommendation in the list; used as the key for the dialog's selection state.
            rec (TuningRecommendation): Recommendation object whose fields are displayed in the card.
        """
        logger.debug(f"Building recommendation card {index}: type={rec.recommendation_type}")
        with ui.card().classes("w-full").props("flat bordered"):
            with ui.row().classes("w-full items-start gap-3"):
                # Checkbox
                checkbox = ui.checkbox(value=self._selected.get(index, True))
                checkbox.on_value_change(
                    lambda e, idx=index: self._set_selection(idx, bool(e.value))
                )

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

    def _set_selection(self, index: int, selected: bool) -> None:
        """Set selection state for a recommendation.

        Args:
            index: Index of the recommendation.
            selected: Whether the recommendation is selected.
        """
        self._selected[index] = selected
        logger.debug(f"Set recommendation {index} selection to {selected}")

    def _build_actions(self) -> None:
        """
        Render the dialog's action buttons aligned to the end: "Dismiss" and "Apply Selected".
        """
        logger.debug("Building recommendation dialog actions")
        with ui.row().classes("w-full justify-end gap-2 mt-4"):
            ui.button("Dismiss", on_click=self._on_dismiss).props("flat")
            ui.button("Apply Selected", on_click=self._on_apply).props("color=primary")

    def _on_apply(self) -> None:
        """
        Apply the currently selected recommendations and close the dialog.

        Closes the dialog if it is open, collects recommendations that are marked selected, and—if there is at least one selected recommendation and an `on_apply` callback—invokes `on_apply` with the list of selected recommendations.
        """
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
        """
        Dismiss the recommendations dialog and notify the dismiss handler.

        Closes the modal dialog if it is open and, if an `on_dismiss` callback was provided, invokes it with the full list of recommendations.
        """
        if self._dialog:
            self._dialog.close()

        logger.debug(f"Dismissing {len(self.recommendations)} recommendations")

        if self.on_dismiss:
            self.on_dismiss(self.recommendations)


def show_recommendations(
    recommendations: list[TuningRecommendation],
    on_apply: Callable[[list[TuningRecommendation]], None] | None = None,
    on_dismiss: Callable[[list[TuningRecommendation]], None] | None = None,
) -> None:
    """
    Display a modal dialog listing tuning recommendations and handle user actions.

    Parameters:
        recommendations (list[TuningRecommendation]): Recommendations to present in the dialog.
        on_apply (Callable[[list[TuningRecommendation]], None] | None): Optional callback invoked with the list of recommendations selected by the user when "Apply Selected" is chosen.
        on_dismiss (Callable[[list[TuningRecommendation]], None] | None): Optional callback invoked with the full list of recommendations when the dialog is dismissed.
    """
    if not recommendations:
        logger.debug("No recommendations to show")
        ui.notify("No recommendations to show", type="info")
        return

    logger.debug(f"Showing recommendations dialog for {len(recommendations)} items")
    dialog = RecommendationDialog(recommendations, on_apply, on_dismiss)
    dialog.show()

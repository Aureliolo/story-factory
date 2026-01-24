"""Comparison page - multi-model chapter comparison view."""

import logging
from typing import Any

from nicegui import ui
from nicegui.elements.button import Button
from nicegui.elements.select import Select

from src.services import ServiceContainer
from src.services.comparison_service import ComparisonRecord, ComparisonResult
from src.settings import RECOMMENDED_MODELS
from src.ui.state import AppState
from src.utils import extract_model_name

logger = logging.getLogger(__name__)


class ComparisonPage:
    """Comparison page for multi-model chapter generation.

    Features:
    - Select 2-4 models to compare
    - Generate same chapter with each model
    - Side-by-side comparison view
    - Highlight differences
    - Select best version
    - Track results for analytics
    """

    def __init__(self, state: AppState, services: ServiceContainer):
        """Initialize comparison page.

        Args:
            state: Application state.
            services: Service container.
        """
        self.state = state
        self.services = services

        # UI element references
        self._model_selects: list[Select] = []
        self._chapter_select: Select | None = None
        self._generate_btn: Button | None = None
        self._results_container: ui.column | None = None
        self._progress_label: ui.label | None = None

        # State
        self._selected_models: list[str] = []
        self._current_comparison: ComparisonRecord | None = None
        self._is_generating = False

    def build(self) -> None:
        """Build the comparison page UI."""
        with ui.column().classes("w-full gap-6 p-4"):
            # Header
            self._build_header()

            if not self.state.has_project:
                self._build_no_project_message()
                return

            # Model selection
            self._build_model_selection()

            # Chapter selection and generate button
            self._build_controls()

            # Progress indicator
            with ui.row().classes("w-full items-center gap-2"):
                self._progress_label = ui.label("").classes("text-sm text-gray-400")

            # Results container
            self._results_container = ui.column().classes("w-full gap-4")

            # Show comparison history
            self._build_history_section()

    def _build_header(self) -> None:
        """Build page header."""
        with ui.row().classes("w-full items-center"):
            ui.label("Model Comparison").classes("text-2xl font-bold")
            ui.space()
            ui.button(
                "Clear History",
                on_click=self._clear_history,
                icon="delete_sweep",
            ).props("flat color=red")

    def _build_no_project_message(self) -> None:
        """Show message when no project is loaded."""
        with ui.card().classes("w-full p-6"):
            ui.label("No Project Loaded").classes("text-xl font-bold mb-2")
            ui.label("Please create or load a project to compare models.").classes("text-gray-400")

    def _build_model_selection(self) -> None:
        """Build model selection controls."""
        with ui.card().classes("w-full p-4"):
            ui.label("Select Models to Compare").classes("text-lg font-bold mb-2")
            ui.label("Choose 2-4 models for side-by-side comparison").classes(
                "text-sm text-gray-400 mb-4"
            )

            # Get installed models
            installed_models = self.services.model.list_installed()
            model_options: dict[str, str] = {}
            for model in installed_models:
                if model in RECOMMENDED_MODELS:
                    model_info = RECOMMENDED_MODELS[model]
                    model_options[model] = f"{model_info['name']} ({model})"
                else:
                    model_options[model] = model

            # Default to comparison_models from settings if available
            default_models = self.services.settings.comparison_models[:4]
            if not installed_models:
                ui.label("No models installed. Install models in Settings first.").classes(
                    "text-orange-400 text-sm mb-4"
                )
                return
            while len(default_models) < 2:
                default_models.append(installed_models[0])

            with ui.grid(columns=2).classes("w-full gap-4"):
                for i in range(4):
                    with ui.column().classes("gap-2"):
                        ui.label(
                            f"Model {i + 1}{' (Required)' if i < 2 else ' (Optional)'}"
                        ).classes("text-sm font-medium")
                        select = (
                            ui.select(
                                options=model_options,
                                value=default_models[i] if i < len(default_models) else None,
                                on_change=self._on_model_selection_change,
                            )
                            .classes("w-full")
                            .props("outlined dense")
                        )
                        self._model_selects.append(select)

            # Update selected models from defaults
            self._update_selected_models()

    def _build_controls(self) -> None:
        """Build chapter selection and generate controls."""
        if not self.state.project:
            return

        with ui.card().classes("w-full p-4"):
            ui.label("Chapter Selection").classes("text-lg font-bold mb-2")

            with ui.row().classes("w-full items-center gap-4"):
                # Chapter dropdown
                chapters = self.state.project.chapters
                if chapters:
                    chapter_options = {c.number: f"Chapter {c.number}: {c.title}" for c in chapters}
                    self._chapter_select = (
                        ui.select(
                            options=chapter_options,
                            value=chapters[0].number,
                            label="Chapter to Compare",
                        )
                        .classes("flex-grow")
                        .props("outlined dense")
                    )
                else:
                    ui.label("No chapters available. Build story structure first.").classes(
                        "text-gray-400"
                    )
                    return

                # Generate button
                self._generate_btn = (
                    ui.button(
                        "Generate Comparison",
                        on_click=self._generate_comparison,
                        icon="compare_arrows",
                    )
                    .props("color=primary")
                    .bind_enabled_from(self, "_is_generating", backward=lambda x: not x)
                )

    def _build_history_section(self) -> None:
        """Build comparison history section."""
        history = self.services.comparison.get_comparison_history()
        if not history:
            return

        with ui.card().classes("w-full p-4 mt-6"):
            ui.label("Recent Comparisons").classes("text-lg font-bold mb-4")

            for record in history[:5]:  # Show last 5
                with ui.card().classes("w-full p-3 mb-2"):
                    with ui.row().classes("w-full items-center gap-2"):
                        ui.icon("history").classes("text-gray-400")
                        ui.label(
                            f"Chapter {record.chapter_number} - {record.timestamp.strftime('%Y-%m-%d %H:%M')}"
                        ).classes("font-medium")
                        ui.space()
                        if record.selected_model:
                            model_name = extract_model_name(record.selected_model)
                            ui.label(f"Winner: {model_name}").classes(
                                "text-sm text-green-500 font-medium"
                            )
                        ui.button(
                            "View",
                            on_click=lambda r=record: self._load_comparison(r),
                            icon="visibility",
                        ).props("flat dense")

    def _on_model_selection_change(self, e: Any) -> None:
        """Handle model selection change."""
        self._update_selected_models()

    def _update_selected_models(self) -> None:
        """Update the list of selected models."""
        self._selected_models = [select.value for select in self._model_selects if select.value]
        logger.debug(f"Selected models: {self._selected_models}")

    async def _generate_comparison(self) -> None:
        """
        Initiates generation of a chapter comparison for the currently selected chapter and models, processes progress events from the generator, stores the resulting ComparisonRecord, and updates the UI.
        
        Performs validation of project presence, chapter selection, and that 2–4 models are chosen; notifies the user on validation failures or generation errors. Updates internal state (_is_generating and _current_comparison) and progress UI while the comparison is being produced.
        """
        if self._is_generating:
            return

        # Validation
        if not self.state.project:
            ui.notify("No project loaded", type="negative")
            return

        if not self._chapter_select or not self._chapter_select.value:
            ui.notify("Please select a chapter", type="negative")
            return

        self._update_selected_models()
        if len(self._selected_models) < 2:
            ui.notify("Please select at least 2 models", type="negative")
            return

        if len(self._selected_models) > 4:
            ui.notify("Maximum 4 models allowed", type="negative")
            return

        chapter_num = self._chapter_select.value
        self._is_generating = True

        try:
            ui.notify(
                f"Generating chapter {chapter_num} with {len(self._selected_models)} models...",
                type="info",
            )

            # Clear previous results
            if self._results_container:
                self._results_container.clear()

            # Generate comparison
            comparison_gen = self.services.comparison.generate_chapter_comparison(
                state=self.state.project,
                chapter_num=chapter_num,
                models=self._selected_models,
            )

            # Process events and capture return value via StopIteration
            comparison_record: ComparisonRecord | None = None
            try:
                while True:
                    event_dict = next(comparison_gen)
                    if event_dict.get("completed"):
                        model_id = event_dict["model_id"]
                        model_name = extract_model_name(model_id)
                        if self._progress_label:
                            self._progress_label.text = (
                                f"✓ {model_name} complete ({event_dict['progress'] * 100:.0f}%)"
                            )
                    else:
                        # Update progress
                        event = event_dict.get("event")
                        if event and self._progress_label:
                            model_name = extract_model_name(event_dict["model_id"])
                            self._progress_label.text = (
                                f"{model_name}: {event.agent_name} - {event.message[:50]}..."
                            )
            except StopIteration as stop_exc:
                comparison_record = stop_exc.value

            # Get the comparison record
            if comparison_record is None:
                logger.error("Comparison generator completed without returning a record")
                ui.notify("Comparison failed to complete", type="negative")
                return
            self._current_comparison = comparison_record

            # Display results
            self._display_comparison_results(self._current_comparison)

            if self._progress_label:
                self._progress_label.text = "✓ Comparison complete!"

            ui.notify("Comparison complete!", type="positive")

        except Exception as e:
            logger.exception("Error generating comparison")
            ui.notify(f"Error: {e}", type="negative")
            if self._progress_label:
                self._progress_label.text = f"Error: {e}"
        finally:
            self._is_generating = False

    def _display_comparison_results(self, record: ComparisonRecord) -> None:
        """Display comparison results in side-by-side view.

        Args:
            record: Comparison record with results.
        """
        if not self._results_container:
            return

        self._results_container.clear()

        with self._results_container:
            ui.label(f"Chapter {record.chapter_number} Comparison").classes(
                "text-xl font-bold mb-4"
            )

            # Results grid
            columns = min(len(record.results), 2)  # 2 columns max for readability
            with ui.grid(columns=columns).classes("w-full gap-4"):
                for model_id, result in record.results.items():
                    self._build_result_card(record, model_id, result)

            # Selection controls
            if record.results:
                self._build_selection_controls(record)

    def _build_result_card(
        self, record: ComparisonRecord, model_id: str, result: ComparisonResult
    ) -> None:
        """Build a result card for a single model.

        Args:
            record: Comparison record.
            model_id: Model ID.
            result: Generation result.
        """
        model_name = extract_model_name(model_id)
        is_selected = record.selected_model == model_id

        card_class = "w-full p-4 border-2"
        if is_selected:
            card_class += " border-green-500"
        elif result.error:
            card_class += " border-red-500"
        else:
            card_class += " border-gray-700"

        with ui.card().classes(card_class):
            # Header
            with ui.row().classes("w-full items-center gap-2 mb-3"):
                if is_selected:
                    ui.icon("check_circle").classes("text-green-500")
                ui.label(model_name).classes("text-lg font-bold flex-grow")
                if result.error:
                    ui.icon("error").classes("text-red-500")

            # Metrics
            with ui.row().classes("w-full gap-4 mb-3 flex-wrap"):
                with ui.column().classes("gap-1"):
                    ui.label("Words").classes("text-xs text-gray-400")
                    ui.label(str(result.word_count)).classes("font-medium")

                with ui.column().classes("gap-1"):
                    ui.label("Time").classes("text-xs text-gray-400")
                    ui.label(f"{result.generation_time:.1f}s").classes("font-medium")

                if result.word_count > 0 and result.generation_time > 0:
                    wpm = (result.word_count / result.generation_time) * 60
                    with ui.column().classes("gap-1"):
                        ui.label("Speed").classes("text-xs text-gray-400")
                        ui.label(f"{wpm:.0f} w/m").classes("font-medium")

            # Content or error
            if result.error:
                ui.label(f"Error: {result.error}").classes("text-red-400 text-sm")
            else:
                # Content preview with expansion
                with ui.expansion("Preview", icon="visibility").classes("w-full"):
                    ui.markdown(
                        result.content[:2000] + "..."
                        if len(result.content) > 2000
                        else result.content
                    ).classes("text-sm max-h-96 overflow-y-auto")

            # Select button
            if not is_selected and not result.error:
                ui.button(
                    "Select This Version",
                    on_click=lambda m=model_id: self._select_winner(record.id, m),
                    icon="thumb_up",
                ).props("flat color=primary").classes("w-full mt-2")

    def _build_selection_controls(self, record: ComparisonRecord) -> None:
        """Build selection controls for comparison.

        Args:
            record: Comparison record.
        """
        with ui.card().classes("w-full p-4 mt-4"):
            ui.label("Selection").classes("text-lg font-bold mb-2")

            if record.selected_model:
                model_name = extract_model_name(record.selected_model)
                ui.label(f"Selected: {model_name}").classes("text-green-500 font-medium mb-2")

                if record.user_notes:
                    ui.label(f"Notes: {record.user_notes}").classes("text-sm text-gray-400")
            else:
                ui.label("Click 'Select This Version' on your preferred result above").classes(
                    "text-sm text-gray-400"
                )

    def _select_winner(self, comparison_id: str, model_id: str) -> None:
        """Record user's model selection.

        Args:
            comparison_id: Comparison ID.
            model_id: Selected model ID.
        """
        try:
            # Prompt for notes
            with ui.dialog() as dialog, ui.card():
                ui.label("Why did you choose this model?").classes("text-lg font-bold mb-2")
                notes_input = ui.textarea(
                    label="Optional notes",
                    placeholder="e.g., Better dialogue, more engaging prose...",
                ).classes("w-full")

                with ui.row().classes("w-full gap-2 justify-end mt-4"):
                    ui.button("Cancel", on_click=dialog.close).props("flat")
                    ui.button(
                        "Confirm Selection",
                        on_click=lambda: self._confirm_selection(
                            dialog, comparison_id, model_id, notes_input.value or ""
                        ),
                    ).props("color=primary")

            dialog.open()

        except Exception as e:
            logger.exception("Error selecting winner")
            ui.notify(f"Error: {e}", type="negative")

    def _confirm_selection(
        self, dialog: ui.dialog, comparison_id: str, model_id: str, notes: str
    ) -> None:
        """Confirm the selection and close dialog.

        Args:
            dialog: Dialog to close.
            comparison_id: Comparison ID.
            model_id: Selected model ID.
            notes: User notes.
        """
        try:
            self.services.comparison.select_winner(comparison_id, model_id, notes)
            model_name = extract_model_name(model_id)
            ui.notify(f"Selected {model_name}!", type="positive")

            # Refresh display
            comparison = self.services.comparison.get_comparison(comparison_id)
            if comparison:
                self._display_comparison_results(comparison)

            dialog.close()

        except Exception as e:
            logger.exception("Error confirming selection")
            ui.notify(f"Error: {e}", type="negative")

    def _load_comparison(self, record: ComparisonRecord) -> None:
        """Load and display a previous comparison.

        Args:
            record: Comparison record to display.
        """
        self._current_comparison = record
        self._display_comparison_results(record)

        # Scroll to results
        if self._results_container:
            ui.run_javascript(
                f"document.getElementById('c{self._results_container.id}').scrollIntoView({{behavior: 'smooth'}})"
            )

    def _clear_history(self) -> None:
        """Clear comparison history."""
        with ui.dialog() as dialog, ui.card():
            ui.label("Clear all comparison history?").classes("text-lg mb-4")
            with ui.row().classes("gap-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button(
                    "Clear",
                    on_click=lambda: self._confirm_clear_history(dialog),
                ).props("color=red")

        dialog.open()

    def _confirm_clear_history(self, dialog: ui.dialog) -> None:
        """Confirm clearing history.

        Args:
            dialog: Dialog to close.
        """
        self.services.comparison.clear_history()
        ui.notify("History cleared", type="positive")
        dialog.close()

        # Rebuild history section
        self.build()
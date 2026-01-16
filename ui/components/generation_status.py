"""Generation status component for displaying and controlling ongoing generation."""

import logging

from nicegui import ui

from ui.state import AppState

logger = logging.getLogger(__name__)


class GenerationStatus:
    """Component for displaying generation status and controls.

    Shows:
    - Current generation progress
    - Cancel and pause/resume buttons
    - Progress indicator

    Usage:
        status = GenerationStatus(state)
        status.build()
        # Update progress during generation
        status.update_progress("Writing chapter 3...")
    """

    def __init__(self, state: AppState):
        """Initialize generation status component.

        Args:
            state: Application state for managing generation flags.
        """
        self.state = state
        self._container: ui.card | None = None
        self._progress_label: ui.label | None = None
        self._cancel_btn: ui.button | None = None
        self._pause_btn: ui.button | None = None
        self._progress_bar: ui.linear_progress | None = None

    def update_progress(self, message: str) -> None:
        """Update the progress message.

        Args:
            message: Progress message to display.
        """
        if self._progress_label:
            self._progress_label.text = message
            logger.debug(f"Generation progress: {message}")

    def _on_cancel_click(self) -> None:
        """Handle cancel button click."""
        self.state.request_cancel_generation()
        if self._cancel_btn:
            self._cancel_btn.set_enabled(False)
        logger.info("User requested generation cancellation")

    def _on_pause_click(self) -> None:
        """Handle pause/resume button click."""
        if self.state.generation_is_paused:
            self.state.resume_generation()
            if self._pause_btn:
                self._pause_btn.props("icon=pause")
                self._pause_btn.update()
        else:
            self.state.request_pause_generation()
            if self._pause_btn:
                self._pause_btn.props("icon=play_arrow")
                self._pause_btn.update()

    def show(self) -> None:
        """Show the generation status component."""
        if self._container:
            self._container.set_visibility(True)

    def hide(self) -> None:
        """Hide the generation status component."""
        if self._container:
            self._container.set_visibility(False)

    def build(self) -> None:
        """Build the generation status UI."""
        self._container = ui.card().classes("w-full p-4")
        self._container.set_visibility(False)  # Hidden by default

        with self._container:
            with ui.row().classes("w-full items-center gap-2"):
                ui.icon("auto_stories").classes("text-2xl")
                self._progress_label = ui.label("Generating...").classes("flex-grow")

                # Pause/Resume button
                self._pause_btn = (
                    ui.button(icon="pause", on_click=self._on_pause_click)
                    .props("flat round")
                    .tooltip("Pause generation")
                )

                # Cancel button
                self._cancel_btn = (
                    ui.button(icon="stop", on_click=self._on_cancel_click)
                    .props("flat round color=negative")
                    .tooltip("Cancel generation")
                )

            # Progress bar
            self._progress_bar = ui.linear_progress().props("indeterminate color=primary")

    def set_progress(self, value: float) -> None:
        """Set progress bar value (0.0 to 1.0).

        Args:
            value: Progress value between 0 and 1.
        """
        if self._progress_bar:
            self._progress_bar.props(f"value={value}")
            if value > 0:
                self._progress_bar.props(remove="indeterminate")
            else:
                self._progress_bar.props("indeterminate")

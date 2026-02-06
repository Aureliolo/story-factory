"""Generation status component for displaying and controlling ongoing generation."""

import logging

from nicegui import ui

from src.ui.state import AppState

logger = logging.getLogger(__name__)


class GenerationStatus:
    """Component for displaying generation status and controls.

    Shows:
    - Current generation phase (Interview → Architect → Writer → Editor → Continuity)
    - Progress bar with current percentage
    - Current chapter being processed
    - Estimated time remaining (ETA)
    - Cancel and pause/resume buttons

    Usage:
        status = GenerationStatus(state)
        status.build()
        # Update progress during generation from WorkflowEvent
        status.update_from_event(workflow_event)
    """

    def __init__(self, state: AppState):
        """Initialize generation status component.

        Args:
            state: Application state for managing generation flags.
        """
        self.state = state
        self._container: ui.card | None = None
        self._progress_label: ui.label | None = None
        self._phase_label: ui.label | None = None
        self._eta_label: ui.label | None = None
        self._chapter_label: ui.label | None = None
        self._cancel_btn: ui.button | None = None
        self._pause_btn: ui.button | None = None
        self._progress_bar: ui.linear_progress | None = None
        self._phase_icons: ui.row | None = None

    def update_progress(self, message: str) -> None:
        """Update the progress message.

        Args:
            message: Progress message to display.
        """
        if self._progress_label:
            self._progress_label.text = message
            logger.debug(f"Generation progress: {message}")

    def update_from_event(self, event) -> None:
        """Update the status display from a WorkflowEvent.

        Args:
            event: WorkflowEvent with progress information
        """
        # Update message
        if self._progress_label:
            self._progress_label.text = event.message

        # Update phase indicator
        if self._phase_label and event.phase:
            phase_name = event.phase.replace("_", " ").title()
            self._phase_label.text = f"Phase: {phase_name}"
            self._update_phase_icons(event.phase)

        # Update progress bar (set_progress handles validation and clamping)
        if self._progress_bar and event.progress is not None:
            self.set_progress(event.progress)

        # Update chapter indicator
        if self._chapter_label and event.chapter_number:
            self._chapter_label.text = f"Chapter {event.chapter_number}"
            self._chapter_label.set_visibility(True)
        elif self._chapter_label:
            self._chapter_label.set_visibility(False)

        # Update ETA
        if self._eta_label and event.eta_seconds is not None:
            eta_str = self._format_eta(event.eta_seconds)
            self._eta_label.text = f"ETA: {eta_str}"
            self._eta_label.set_visibility(True)
        elif self._eta_label:
            self._eta_label.set_visibility(False)

    def _format_eta(self, seconds: float) -> str:
        """Format ETA seconds into human-readable string.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted string like "2m 30s" or "1h 5m"
        """
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"

    def _update_phase_icons(self, current_phase: str) -> None:
        """Update phase indicator icons to show progress.

        Args:
            current_phase: Current phase name
        """
        if not self._phase_icons:
            return

        phases = ["interview", "architect", "writer", "editor", "continuity"]
        phase_display = {
            "interview": ("chat", "Interview"),
            "architect": ("architecture", "Architect"),
            "writer": ("edit_note", "Writer"),
            "editor": ("rate_review", "Editor"),
            "continuity": ("fact_check", "Continuity"),
        }

        # Clear and rebuild phase indicators
        self._phase_icons.clear()
        with self._phase_icons:
            current_idx = phases.index(current_phase) if current_phase in phases else -1

            for idx, phase in enumerate(phases):
                icon_name, label = phase_display[phase]

                # Determine color based on status
                if idx < current_idx:
                    # Completed phase
                    color = "green"
                    icon = "check_circle"
                elif idx == current_idx:
                    # Current phase
                    color = "blue"
                    icon = icon_name
                else:
                    # Future phase
                    color = "grey"
                    icon = "radio_button_unchecked"

                with ui.column().classes("items-center gap-0"):
                    ui.icon(icon, size="sm", color=color)
                    ui.label(label).classes(f"text-xs text-{color}-600")

                # Add arrow between phases (except after last)
                if idx < len(phases) - 1:
                    ui.icon("arrow_forward", size="xs").classes("text-grey-400 mt-2")

    def _on_cancel_click(self) -> None:
        """Handle cancel button click."""
        self.state.request_cancel_generation()
        if self._cancel_btn:
            self._cancel_btn.set_enabled(False)
        logger.info("User requested generation cancellation")

    def _on_pause_click(self) -> None:
        """Handle pause/resume button click.

        Button shows the action it will perform:
        - When paused: shows play_arrow (will resume)
        - When running: shows pause (will pause)
        """
        if self.state.generation_is_paused:
            # Currently paused, so resume
            self.state.resume_generation()
            logger.info("User resumed generation")
            if self._pause_btn:
                # Now running, so show pause icon
                self._pause_btn.props("icon=pause")
                self._pause_btn.update()
        else:
            # Currently running, so pause
            self.state.request_pause_generation()
            logger.info("User requested generation pause")
            if self._pause_btn:
                # Now paused, so show play icon
                self._pause_btn.props("icon=play_arrow")
                self._pause_btn.update()

    def show(self) -> None:
        """Show the generation status component."""
        if self._container:
            self._container.set_visibility(True)
            logger.debug("Generation status component shown")

    def hide(self) -> None:
        """Hide the generation status component."""
        if self._container:
            self._container.set_visibility(False)
            logger.debug("Generation status component hidden")

    def build(self) -> None:
        """Build the generation status UI."""
        self._container = ui.card().classes("w-full p-4")
        self._container.set_visibility(False)  # Hidden by default

        with self._container:
            # Phase indicator icons
            ui.label("Generation Progress").classes("text-sm font-medium mb-2")
            self._phase_icons = ui.row().classes("w-full items-center justify-between mb-3")

            # Progress message and details
            with ui.row().classes("w-full items-center gap-2 mb-2"):
                ui.icon("auto_stories").classes("text-2xl")

                with ui.column().classes("flex-grow gap-1"):
                    self._progress_label = ui.label("Generating...").classes("font-medium")

                    with ui.row().classes("gap-2 items-center"):
                        self._phase_label = ui.label("Phase: Starting...").classes(
                            "text-sm text-gray-400"
                        )
                        self._chapter_label = ui.label("Chapter 1").classes("text-sm text-gray-400")
                        self._chapter_label.set_visibility(
                            False
                        )  # Hidden until we have chapter info
                        self._eta_label = ui.label("ETA: --").classes("text-sm text-gray-400")
                        self._eta_label.set_visibility(False)  # Hidden until we have ETA

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

            # Progress bar (determinate, not indeterminate)
            self._progress_bar = ui.linear_progress(value=0.0).props("color=primary")

    def set_progress(self, value: float) -> None:
        """Set progress bar value (0.0 to 1.0).

        Values outside the valid range are clamped with a warning logged.

        Args:
            value: Progress value between 0 and 1.
        """
        if self._progress_bar:
            # Validate and clamp value between 0 and 1
            if value < 0.0 or value > 1.0:
                logger.warning("Progress value %s clamped to [0.0, 1.0]", value)
                value = max(0.0, min(1.0, value))
            self._progress_bar.value = value
            self._progress_bar.update()

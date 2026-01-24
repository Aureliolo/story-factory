"""Timeline page - story event visualization and tracking."""

import logging

from nicegui import ui

from src.services import ServiceContainer
from src.ui.components.timeline import TimelineComponent
from src.ui.state import AppState

logger = logging.getLogger(__name__)


class TimelinePage:
    """Timeline page for visualizing and tracking story events.

    Features:
    - Visual timeline of story progression
    - Character appearance tracking
    - Chapter boundaries
    - Location changes
    - Event reordering (drag and drop)
    - Zoom and pan controls
    """

    def __init__(self, state: AppState, services: ServiceContainer):
        """Initialize timeline page.

        Args:
            state: Application state.
            services: Service container.
        """
        self.state = state
        self.services = services

        # UI references
        self._timeline: TimelineComponent | None = None

    def build(self) -> None:
        """Build the timeline page UI."""
        if not self.state.has_project:
            self._build_no_project_message()
            return

        # Check if interview is complete
        if not self.state.interview_complete:
            self._build_interview_required_message()
            return

        # Main timeline section
        with ui.column().classes("w-full h-full gap-4 p-4"):
            self._build_info_section()
            self._build_timeline_section()
            self._build_events_list()

    def _build_no_project_message(self) -> None:
        """Build message when no project is selected."""
        with ui.column().classes("w-full min-h-96 items-center justify-center gap-4 py-16"):
            ui.icon("timeline", size="xl").classes("text-gray-400 dark:text-gray-500")
            ui.label("No Project Selected").classes("text-xl text-gray-500 dark:text-gray-400")
            ui.label("Select a project from the header to view its timeline.").classes(
                "text-gray-400 dark:text-gray-500"
            )

    def _build_interview_required_message(self) -> None:
        """Build message when interview is not complete."""
        with ui.column().classes("w-full min-h-96 items-center justify-center gap-6 py-16"):
            ui.icon("chat", size="xl").classes("text-blue-400")
            ui.label("Complete the Interview First").classes(
                "text-xl font-semibold text-gray-700 dark:text-gray-200"
            )
            ui.label(
                "The Timeline requires story context from the interview. "
                "Complete the interview to see your story's timeline."
            ).classes("text-gray-500 dark:text-gray-400 text-center max-w-md")

            ui.button(
                "Go to Interview",
                on_click=lambda: ui.navigate.to("/"),
                icon="arrow_forward",
            ).props("color=primary size=lg")

    def _build_info_section(self) -> None:
        """Build the info section with timeline statistics."""
        if not self.state.project:
            return

        project = self.state.project

        # Count timeline elements
        event_count = len(project.timeline)
        chapter_count = len(project.chapters)
        character_count = len(project.characters)

        with ui.card().classes("w-full p-4"):
            with ui.row().classes("w-full items-center gap-6"):
                # Events count
                with ui.column().classes("items-center"):
                    ui.icon("event", size="lg").classes("text-blue-500")
                    ui.label(str(event_count)).classes("text-2xl font-bold")
                    ui.label("Events").classes("text-sm text-gray-500 dark:text-gray-400")

                ui.separator().props("vertical")

                # Chapters count
                with ui.column().classes("items-center"):
                    ui.icon("menu_book", size="lg").classes("text-green-500")
                    ui.label(str(chapter_count)).classes("text-2xl font-bold")
                    ui.label("Chapters").classes("text-sm text-gray-500 dark:text-gray-400")

                ui.separator().props("vertical")

                # Characters count
                with ui.column().classes("items-center"):
                    ui.icon("people", size="lg").classes("text-orange-500")
                    ui.label(str(character_count)).classes("text-2xl font-bold")
                    ui.label("Characters").classes("text-sm text-gray-500 dark:text-gray-400")

                ui.space()

                # Add event button
                ui.button(
                    "+ Add Event",
                    on_click=self._show_add_event_dialog,
                    icon="add",
                ).props("color=primary")

    def _build_timeline_section(self) -> None:
        """Build the main timeline visualization section."""
        with ui.card().classes("w-full p-4"):
            self._timeline = TimelineComponent(
                story_state=self.state.project,
                settings=self.services.settings,
                height=500,
                editable=True,
            )
            self._timeline.build()

    def _build_events_list(self) -> None:
        """Build the events list section."""
        if not self.state.project:
            return

        with ui.expansion("Story Events", icon="list", value=True).classes("w-full"):
            if not self.state.project.timeline:
                ui.label("No events recorded yet. Add events using the button above.").classes(
                    "text-gray-500 dark:text-gray-400 text-sm p-4"
                )
            else:
                with ui.column().classes("w-full gap-2 p-2"):
                    for idx, event in enumerate(self.state.project.timeline):
                        with ui.card().classes(
                            "w-full p-3 hover:bg-gray-50 dark:hover:bg-gray-700"
                        ):
                            with ui.row().classes("w-full items-center gap-2"):
                                ui.icon("event", size="sm").classes("text-blue-500")
                                ui.label(f"Event {idx + 1}").classes(
                                    "text-sm font-medium text-gray-500 dark:text-gray-400"
                                )
                                ui.label(event).classes("flex-grow")

                                # Delete button
                                ui.button(
                                    icon="delete",
                                    on_click=lambda i=idx: self._delete_event(i),
                                ).props("flat dense round color=negative")

    def _show_add_event_dialog(self) -> None:
        """Show dialog to add a new timeline event."""
        with ui.dialog() as dialog, ui.card().classes("p-4 min-w-[500px]"):
            ui.label("Add Timeline Event").classes("text-lg font-semibold")

            event_input = (
                ui.textarea(
                    label="Event Description",
                    placeholder="Describe what happens in this event...",
                )
                .classes("w-full")
                .props("rows=3")
            )

            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=dialog.close).props("flat")

                def add_event():
                    """Add new event to timeline."""
                    if event_input.value and self.state.project:
                        logger.debug("Adding timeline event: %s", event_input.value[:50])
                        self.state.project.timeline.append(event_input.value)
                        self.services.project.save_project(self.state.project)

                        # Refresh timeline
                        if self._timeline:
                            self._timeline.refresh()

                        # Refresh the page to update event list
                        ui.navigate.reload()

                        dialog.close()
                        ui.notify("Event added to timeline", type="positive")
                    else:
                        ui.notify("Event description is required", type="warning")

                ui.button("Add", on_click=add_event).props("color=primary")

        dialog.open()

    def _delete_event(self, index: int) -> None:
        """Delete an event from the timeline.

        Args:
            index: Index of the event to delete.
        """
        if not self.state.project:
            return

        try:
            event = self.state.project.timeline[index]

            # Show confirmation dialog
            with ui.dialog() as dialog, ui.card().classes("p-4 min-w-[400px]"):
                ui.label("Delete Event?").classes("text-lg font-bold text-red-600")
                ui.label(f'Are you sure you want to delete this event?\n\n"{event}"').classes(
                    "text-gray-600 dark:text-gray-400 whitespace-pre-line mt-2"
                )

                with ui.row().classes("w-full justify-end gap-2 mt-4"):
                    ui.button("Cancel", on_click=dialog.close).props("flat")

                    def do_delete():
                        """Execute timeline event deletion."""
                        if self.state.project:
                            logger.debug("Deleting timeline event at index %d", index)
                            del self.state.project.timeline[index]
                            self.services.project.save_project(self.state.project)

                            # Refresh timeline
                            if self._timeline:
                                self._timeline.refresh()

                            # Refresh the page
                            ui.navigate.reload()

                            dialog.close()
                            ui.notify("Event deleted", type="positive")

                    ui.button("Delete", on_click=do_delete, icon="delete").props("color=negative")

            dialog.open()

        except (IndexError, AttributeError):
            logger.exception("Error deleting event at index %d", index)
            ui.notify("Error deleting event", type="negative")

"""Write Story page - Fundamentals and Live Writing tabs.

This package splits the WritePage into focused modules:
- _interview: Interview section UI and event handlers
- _structure: Story structure, generation settings, reviews, world overview
- _writing: Live writing tab, chapter navigation, display, controls, version history, suggestions
- _generation: Chapter writing, regeneration, and learning recommendations
- _export: Story export in various formats
"""

import asyncio
import logging
from typing import Any, Literal

from nicegui import Client, context, ui
from nicegui.elements.button import Button
from nicegui.elements.html import Html
from nicegui.elements.label import Label
from nicegui.elements.markdown import Markdown
from nicegui.elements.select import Select
from nicegui.elements.textarea import Textarea

from src.services import ServiceContainer
from src.ui.components.generation_status import GenerationStatus
from src.ui.state import AppState

from . import _interview, _structure, _writing

logger = logging.getLogger(__name__)


class WritePage:
    """Write Story page with Fundamentals and Live Writing tabs.

    Fundamentals tab:
    - Interview chat
    - World overview
    - Story structure
    - Reviews

    Live Writing tab:
    - Chapter navigator
    - Writing display with streaming
    - Feedback controls
    """

    def __init__(self, state: AppState, services: ServiceContainer):
        """Create a WritePage instance and initialize its UI references.

        Initializes internal references for UI components to None and sets up
        a background task tracker to hold asyncio.Task objects (prevents silent
        task garbage collection).

        Parameters:
            state: Application state used by the page to read and mutate current
                   project and UI state.
            services: Service container providing access to story, world, project,
                      export, settings, and other application services.
        """
        self.state = state
        self.services = services

        # UI references
        self._chat: Any = None  # ChatComponent
        self._structure_display: Markdown | None = None
        self._entity_summary: Html | None = None
        self._chapter_select: Select | None = None
        self._writing_display: Markdown | None = None
        self._word_count_label: Label | None = None
        self._regenerate_feedback_input: Textarea | None = None
        self._version_history_container: ui.column | None = None
        self._client: Client | None = None  # For background task safety
        self._finalize_btn: Button | None = None
        self._build_structure_btn: Button | None = None
        self._scene_list_container: ui.column | None = None  # Container for scene list
        self._generation_status: GenerationStatus | None = None
        self._world_overview_container: ui.column | None = None  # Container for world overview
        self._feedback_mode_select: Select | None = None
        self._background_tasks: set[asyncio.Task[Any]] = set()  # Prevent task GC

    def _notify(
        self,
        message: str,
        type: Literal["positive", "negative", "warning", "info", "ongoing"] = "info",
    ) -> None:
        """Display a UI notification and fall back to logging if not possible.

        Parameters:
            message: The message text to display in the notification.
            type: Notification style; defaults to "info".
        """
        if self._client:
            with self._client:
                ui.notify(message, type=type)
        else:
            try:
                ui.notify(message, type=type)
            except RuntimeError:
                logger.warning(f"Could not show notification: {message}")

    def build(self) -> None:
        """Build the write page UI."""
        # Capture client for background task safety
        try:
            self._client = context.client
        except RuntimeError:
            logger.warning("Could not capture client context during build")

        if not self.state.has_project:
            self._build_no_project_message()
            return

        # Sub-tabs
        with ui.tabs().classes("w-full") as tabs:
            ui.tab("fundamentals", label="Fundamentals", icon="foundation")
            ui.tab("writing", label="Live Writing", icon="edit_note")

        with ui.tab_panels(tabs, value="fundamentals").classes("w-full flex-grow"):
            with ui.tab_panel("fundamentals"):
                self._build_fundamentals()

            with ui.tab_panel("writing"):
                _writing.build_live_writing(self)

    def _build_no_project_message(self) -> None:
        """Build message when no project is selected."""
        from src.ui.components.common import empty_state

        empty_state(
            icon="folder_off",
            title="No Project Selected",
            description="Select a project from the header dropdown or create a new one.",
        )

    def _build_fundamentals(self) -> None:
        """Build the Fundamentals tab content."""
        with ui.column().classes("w-full gap-4 p-4"):
            # Interview Section
            with ui.expansion(
                "Interview",
                icon="chat",
                value=not self.state.interview_complete,
            ).classes("w-full"):
                _interview.build_interview_section(self)

            # World Overview Section
            with ui.expansion("World Overview", icon="public").classes("w-full"):
                self._world_overview_container = ui.column().classes("w-full")
                with self._world_overview_container:
                    _structure.build_world_overview_content(self)

            # Story Structure Section
            with ui.expansion("Story Structure", icon="list_alt").classes("w-full"):
                _structure.build_structure_section(self)

            # Generation Settings Section
            with ui.expansion("Generation Settings", icon="tune", value=False).classes("w-full"):
                _structure.build_generation_settings_section(self)

            # Reviews Section
            with ui.expansion("Reviews & Notes", icon="rate_review").classes("w-full"):
                _structure.build_reviews_section(self)


__all__ = ["WritePage"]

"""World Builder page - base class with init and build methods."""

import logging
import threading
from typing import TYPE_CHECKING, Any

from nicegui import ui
from nicegui.elements.button import Button
from nicegui.elements.column import Column
from nicegui.elements.html import Html
from nicegui.elements.input import Input
from nicegui.elements.select import Select
from nicegui.elements.textarea import Textarea

if TYPE_CHECKING:
    from src.services import ServiceContainer
    from src.ui.state import AppState

logger = logging.getLogger(__name__)

# Default value for relationship strength when creating via drag-and-drop
DEFAULT_RELATIONSHIP_STRENGTH = 0.5


class WorldPageBase:
    """World Builder page base class for managing entities and relationships.

    Features:
    - Interactive graph visualization
    - Entity browser with filtering
    - Entity editor
    - Relationship management
    - Graph analysis tools
    """

    def __init__(self, state: AppState, services: ServiceContainer):
        """Initialize world page.

        Args:
            state: Application state.
            services: Service container.
        """
        self.state = state
        self.services = services

        # Register undo/redo handlers for this page
        self.state.on_undo(self._do_undo)
        self.state.on_redo(self._do_redo)

        # UI references
        self._graph: Any = None  # GraphComponent | None
        self._entity_list: Column | None = None
        self._editor_container: Column | None = None
        self._search_input: Input | None = None
        self._sort_direction_btn: Button | None = None
        self._entity_name_input: Input | None = None
        self._entity_type_select: Select | None = None
        self._entity_desc_input: Textarea | None = None
        # Type-specific attribute form fields
        self._attr_role_select: Select | None = None
        self._attr_traits_input: Input | None = None
        self._attr_goals_input: Input | None = None
        self._attr_arc_input: Textarea | None = None
        self._attr_significance_input: Textarea | None = None
        self._attr_leader_input: Input | None = None
        self._attr_values_input: Input | None = None
        self._attr_properties_input: Input | None = None
        self._attr_manifestations_input: Textarea | None = None
        self._entity_attrs: dict[str, Any] = {}
        self._rel_source_select: Select | None = None
        self._rel_type_select: Select | None = None
        self._rel_target_select: Select | None = None
        self._analysis_result: Html | None = None
        self._undo_btn: Button | None = None
        self._redo_btn: Button | None = None
        # Generation progress dialog state
        self._generation_cancel_event: threading.Event | None = None
        self._generation_dialog: ui.dialog | None = None

    def build(self) -> None:
        """Build the world page UI."""
        if not self.state.has_project:
            self._build_no_project_message()
            return

        # Check if interview is complete
        if not self.state.interview_complete:
            self._build_interview_required_message()
            return

        # World generation toolbar
        self._build_generation_toolbar()

        # Responsive layout: stack on mobile, 3-column on desktop
        # All panels should have the same height
        with (
            ui.row()
            .classes("w-full gap-4 p-4 flex-wrap lg:flex-nowrap")
            .style("min-height: calc(100vh - 250px)")
        ):
            # Left panel - Entity browser (full width on mobile, 20% on desktop)
            with ui.column().classes("w-full lg:w-1/5 gap-4 min-w-[250px] h-full"):
                self._build_entity_browser()

            # Center panel - Graph visualization (full width on mobile, 60% on desktop)
            with ui.column().classes("w-full lg:w-3/5 gap-4 min-w-[300px] h-full"):
                self._build_graph_section()

            # Right panel - Entity editor (full width on mobile, 20% on desktop)
            self._editor_container = ui.column().classes(
                "w-full lg:w-1/5 gap-4 min-w-[250px] h-full"
            )
            with self._editor_container:
                self._build_entity_editor()

        # Bottom sections
        with ui.column().classes("w-full gap-4 p-4"):
            self._build_health_section()
            self._build_relationships_section()
            self._build_analysis_section()

    def _build_no_project_message(self) -> None:
        """Build message when no project is selected."""
        with ui.column().classes("w-full min-h-96 items-center justify-center gap-4 py-16"):
            ui.icon("public_off", size="xl").classes("text-gray-400 dark:text-gray-500")
            ui.label("No Project Selected").classes("text-xl text-gray-500 dark:text-gray-400")
            ui.label("Select a project from the header to explore its world.").classes(
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
                "The World Builder requires story context from the interview. "
                "Complete the interview to populate your story's world."
            ).classes("text-gray-500 dark:text-gray-400 text-center max-w-md")

            ui.button(
                "Go to Interview",
                on_click=lambda: ui.navigate.to("/"),
                icon="arrow_forward",
            ).props("color=primary size=lg")

    def _get_all_entity_names(self) -> list[str]:
        """Get all entity names from the world database.

        Returns:
            List of all entity names across all types.
        """
        if not self.state.world_db:
            return []
        return [e.name for e in self.state.world_db.list_entities()]

    def _get_entity_names_by_type(self, entity_type: str) -> list[str]:
        """Get entity names filtered by type.

        Args:
            entity_type: Type to filter by.

        Returns:
            List of entity names of the specified type.
        """
        if not self.state.world_db:
            return []
        return [e.name for e in self.state.world_db.list_entities() if e.type == entity_type]

    def _get_entity_options(self) -> dict[str, str]:
        """Get entity options for select dropdowns."""
        if not self.state.world_db:
            return {}

        entities = self.services.world.list_entities(self.state.world_db)
        return {e.id: e.name for e in entities}

    # Placeholder methods to be implemented by mixins
    def _do_undo(self) -> None:
        """Execute undo operation - implemented by UndoMixin."""
        raise NotImplementedError

    def _do_redo(self) -> None:
        """Execute redo operation - implemented by UndoMixin."""
        raise NotImplementedError

    def _build_generation_toolbar(self) -> None:
        """Build the world generation toolbar - implemented by GenerationMixin."""
        raise NotImplementedError

    def _build_entity_browser(self) -> None:
        """Build the entity browser panel - implemented by BrowserMixin."""
        raise NotImplementedError

    def _build_graph_section(self) -> None:
        """Build the graph visualization section - implemented by GraphMixin."""
        raise NotImplementedError

    def _build_entity_editor(self) -> None:
        """Build the entity editor panel - implemented by EditorMixin."""
        raise NotImplementedError

    def _build_health_section(self) -> None:
        """Build the world health dashboard section - implemented by AnalysisMixin."""
        raise NotImplementedError

    def _build_relationships_section(self) -> None:
        """Build the relationships management section - implemented by EditorMixin."""
        raise NotImplementedError

    def _build_analysis_section(self) -> None:
        """Build the analysis tools section - implemented by AnalysisMixin."""
        raise NotImplementedError

    def _refresh_entity_list(self) -> None:
        """Refresh the entity list display - implemented by BrowserMixin."""
        raise NotImplementedError

    def _refresh_entity_editor(self) -> None:
        """Refresh the entity editor panel - implemented by EditorMixin."""
        raise NotImplementedError

    def _update_undo_redo_buttons(self) -> None:
        """Update undo/redo button states - implemented by UndoMixin."""
        raise NotImplementedError

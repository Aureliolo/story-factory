"""World Builder page - entity and relationship management.

This package splits the WorldPage class across several helper modules using
plain-function extraction (no mixins, no multiple inheritance).  Every helper
function receives ``page`` (the WorldPage instance) as its first argument.

Module layout:
    __init__.py        -- WorldPage class, build(), shared state, delegation
    _generation.py     -- toolbar, rebuild/clear world, mini descriptions
    _gen_dialogs.py    -- generation dialogs & preview UI
    _gen_operations.py -- generate_more dispatcher + character/location generators
    _gen_entity_types.py -- faction/item/concept/relationship generators
    _browser.py        -- entity browser panel
    _editor.py         -- entity editor display/form functions
    _editor_ops.py     -- entity regeneration, mutation, deletion operations
    _graph.py          -- graph visualization & relationship CRUD
    _analysis.py       -- health dashboard, analysis tools, conflict map
    _undo.py           -- undo / redo operations
    _import.py         -- import wizard
"""

import logging
import threading
from typing import Any

from nicegui import ui
from nicegui.elements.button import Button
from nicegui.elements.column import Column
from nicegui.elements.html import Html
from nicegui.elements.input import Input
from nicegui.elements.select import Select
from nicegui.elements.textarea import Textarea

from src.services import ServiceContainer
from src.ui.components.graph import GraphComponent
from src.ui.pages.world._analysis import (
    build_analysis_section,
    build_health_section,
    handle_fix_orphan,
    handle_improve_quality,
    handle_view_circular,
)
from src.ui.pages.world._browser import (
    build_entity_browser,
    refresh_entity_list,
)
from src.ui.pages.world._editor import (
    build_entity_editor,
    refresh_entity_editor,
)
from src.ui.pages.world._gen_dialogs import (
    show_entity_preview_dialog,
    show_generate_dialog,
    show_quality_settings_dialog,
)
from src.ui.pages.world._gen_operations import (
    generate_more,
    generate_relationships_for_entities,
)
from src.ui.pages.world._generation import (
    build_generation_toolbar,
)
from src.ui.pages.world._graph import (
    build_graph_section,
    build_relationships_section,
    get_entity_options,
    on_create_relationship,
    on_edge_context_menu,
    on_edge_select,
    on_node_select,
)
from src.ui.pages.world._import import show_import_wizard
from src.ui.pages.world._undo import (
    do_redo,
    do_undo,
    update_undo_redo_buttons,
)
from src.ui.state import AppState

logger = logging.getLogger(__name__)


class WorldPage:
    """World Builder page for managing entities and relationships.

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
        self._graph: GraphComponent | None = None
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

    # ========== Build ==========

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
        with (
            ui.row()
            .classes("w-full gap-4 p-4 flex-wrap lg:flex-nowrap")
            .style("min-height: calc(100vh - 250px)")
        ):
            # Left panel - Entity browser
            with ui.column().classes("w-full lg:w-1/5 gap-4 min-w-[250px] h-full"):
                self._build_entity_browser()

            # Center panel - Graph visualization
            with ui.column().classes("w-full lg:w-3/5 gap-4 min-w-[300px] h-full"):
                self._build_graph_section()

            # Right panel - Entity editor
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

    # ========== Inline (small) helpers ==========

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

    def _get_entity_options(self) -> dict[str, str]:
        """Get entity options for select dropdowns."""
        return get_entity_options(self)

    # ========== Delegations to _generation.py ==========

    def _build_generation_toolbar(self) -> None:
        """Build the world generation toolbar."""
        build_generation_toolbar(self)

    # ========== Delegations to _gen_dialogs.py ==========

    def _show_quality_settings_dialog(self) -> None:
        """Show quality settings dialog."""
        show_quality_settings_dialog(self)

    def _show_generate_dialog(self, entity_type: str) -> None:
        """Show entity generation dialog."""
        show_generate_dialog(self, entity_type)

    def _show_entity_preview_dialog(
        self,
        entity_type: str,
        entities: list[tuple[Any, Any]],
        on_confirm: Any,
    ) -> None:
        """Show entity preview dialog."""
        show_entity_preview_dialog(self, entity_type, entities, on_confirm)

    # ========== Delegations to _gen_operations.py ==========

    async def _generate_more(
        self,
        entity_type: str,
        count: int | None = None,
        custom_instructions: str | None = None,
    ) -> None:
        """Generate more entities."""
        await generate_more(self, entity_type, count, custom_instructions)

    async def _generate_relationships_for_entities(
        self, entity_names: list[str], count: int
    ) -> None:
        """Generate relationships for specific entities."""
        await generate_relationships_for_entities(self, entity_names, count)

    # ========== Delegations to _browser.py ==========

    def _build_entity_browser(self) -> None:
        """Build the entity browser panel."""
        build_entity_browser(self)

    def _refresh_entity_list(self) -> None:
        """Refresh the entity list display."""
        refresh_entity_list(self)

    # ========== Delegations to _editor.py ==========

    def _build_entity_editor(self) -> None:
        """Build the entity editor panel."""
        build_entity_editor(self)

    def _refresh_entity_editor(self) -> None:
        """Refresh the entity editor panel."""
        refresh_entity_editor(self)

    # ========== Delegations to _graph.py ==========

    def _build_graph_section(self) -> None:
        """Build the graph visualization section."""
        build_graph_section(self)

    def _build_relationships_section(self) -> None:
        """Build the relationships management section."""
        build_relationships_section(self)

    def _on_node_select(self, entity_id: str) -> None:
        """Handle graph node selection."""
        on_node_select(self, entity_id)

    def _on_edge_select(self, relationship_id: str) -> None:
        """Handle graph edge selection."""
        on_edge_select(self, relationship_id)

    def _on_create_relationship(self, source_id: str, target_id: str) -> None:
        """Handle drag-to-connect relationship creation."""
        on_create_relationship(self, source_id, target_id)

    def _on_edge_context_menu(self, edge_id: str) -> None:
        """Handle edge right-click context menu."""
        on_edge_context_menu(self, edge_id)

    # ========== Delegations to _analysis.py ==========

    def _build_health_section(self) -> None:
        """Build the world health dashboard section."""
        build_health_section(self)

    def _build_analysis_section(self) -> None:
        """Build the analysis tools section."""
        build_analysis_section(self)

    async def _handle_fix_orphan(self, entity_id: str) -> None:
        """Handle fix orphan entity request."""
        await handle_fix_orphan(self, entity_id)

    async def _handle_view_circular(self, cycle: dict) -> None:
        """Handle view circular relationship chain request."""
        await handle_view_circular(self, cycle)

    async def _handle_improve_quality(self, entity_id: str) -> None:
        """Handle improve entity quality request."""
        await handle_improve_quality(self, entity_id)

    # ========== Delegations to _undo.py ==========

    def _update_undo_redo_buttons(self) -> None:
        """Update undo/redo button states."""
        update_undo_redo_buttons(self)

    def _do_undo(self) -> None:
        """Execute undo operation."""
        do_undo(self)

    def _do_redo(self) -> None:
        """Execute redo operation."""
        do_redo(self)

    # ========== Delegations to _import.py ==========

    def _show_import_wizard(self) -> None:
        """Show import wizard."""
        show_import_wizard(self)

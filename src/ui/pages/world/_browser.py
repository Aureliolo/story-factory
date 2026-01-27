"""World Builder page - browser mixin with entity browser methods."""

import logging
from typing import Any

from nicegui import ui

from src.memory.entities import Entity
from src.ui.components.entity_card import entity_list_item
from src.ui.pages.world._page import WorldPageBase

logger = logging.getLogger(__name__)


class BrowserMixin(WorldPageBase):
    """Mixin providing entity browser methods for WorldPage."""

    def _build_entity_browser(self) -> None:
        """Build the entity browser panel."""
        with ui.card().classes("w-full h-full"):
            ui.label("Entity Browser").classes("text-lg font-semibold")

            # Search with Ctrl+F hint
            self._search_input = (
                ui.input(
                    placeholder="Search entities... (Ctrl+F)",
                    value=self.state.entity_search_query,
                    on_change=self._on_search,
                )
                .classes("w-full")
                .props("outlined dense clearable")
            )

            # Search scope checkboxes
            with ui.row().classes("w-full gap-4 text-xs items-center"):
                ui.checkbox(
                    "Names",
                    value=self.state.entity_search_names,
                    on_change=lambda e: self._update_search_scope("names", e.value),
                ).props("dense")
                ui.checkbox(
                    "Descriptions",
                    value=self.state.entity_search_descriptions,
                    on_change=lambda e: self._update_search_scope("descriptions", e.value),
                ).props("dense")

            # Filter and sort row
            with ui.row().classes("w-full gap-2 items-center mt-1"):
                # Quality filter
                ui.select(
                    label="Quality",
                    options={"all": "All", "high": "8+", "medium": "6-8", "low": "<6"},
                    value=self.state.entity_quality_filter,
                    on_change=self._on_quality_filter_change,
                ).classes("w-20").props("dense outlined")

                # Sort dropdown
                ui.select(
                    label="Sort",
                    options={
                        "name": "Name",
                        "type": "Type",
                        "quality": "Quality",
                        "relationships": "Relationships",
                    },
                    value=self.state.entity_sort_by,
                    on_change=self._on_sort_change,
                ).classes("w-28").props("dense outlined")

                # Sort direction toggle
                self._sort_direction_btn = (
                    ui.button(
                        icon="arrow_upward"
                        if not self.state.entity_sort_descending
                        else "arrow_downward",
                        on_click=self._toggle_sort_direction,
                    )
                    .props("flat dense")
                    .tooltip("Toggle sort direction")
                )

            # Entity list with flexible height to match editor
            self._entity_list = (
                ui.column()
                .classes(
                    "w-full gap-1 overflow-auto flex-grow p-2 bg-gray-50 dark:bg-gray-800 rounded-lg"
                )
                .style("max-height: calc(100vh - 520px); min-height: 200px")
            )
            self._refresh_entity_list()

            # Add button
            ui.button(
                "+ Add Entity",
                on_click=self._show_add_dialog,
                icon="add",
            ).props("color=primary").classes("w-full mt-2")

            # Undo/Redo buttons
            with ui.row().classes("w-full gap-2 mt-2"):
                self._undo_btn = (
                    ui.button(
                        icon="undo",
                        on_click=self._do_undo,
                    )
                    .props("flat dense")
                    .tooltip("Undo (Ctrl+Z)")
                )
                self._redo_btn = (
                    ui.button(
                        icon="redo",
                        on_click=self._do_redo,
                    )
                    .props("flat dense")
                    .tooltip("Redo (Ctrl+Y)")
                )
                self._update_undo_redo_buttons()

            # Register Ctrl+F keyboard shortcut
            self._register_keyboard_shortcuts()

    def _refresh_entity_list(self) -> None:
        """Refresh the entity list display."""
        if not self._entity_list or not self.state.world_db:
            return

        self._entity_list.clear()

        entities = self.services.world.list_entities(self.state.world_db)

        # Filter by type
        if self.state.entity_filter_types:
            entities = [e for e in entities if e.type in self.state.entity_filter_types]

        # Filter by search (with scope) - concise list comprehension
        if self.state.entity_search_query:
            query = self.state.entity_search_query.lower()
            entities = [
                e
                for e in entities
                if (self.state.entity_search_names and query in e.name.lower())
                or (self.state.entity_search_descriptions and query in e.description.lower())
            ]

        # Filter by quality
        if self.state.entity_quality_filter != "all":
            entities = self._filter_by_quality(entities)

        # Sort entities
        entities = self._sort_entities(entities)

        # Clear selection if selected entity is filtered out
        if self.state.selected_entity_id:
            visible_ids = {e.id for e in entities}
            if self.state.selected_entity_id not in visible_ids:
                self.state.select_entity(None)
                self._refresh_entity_editor()

        with self._entity_list:
            if not entities:
                # Check if there are entities at all or just filtered out
                all_entities = self.services.world.list_entities(self.state.world_db)
                if not all_entities:
                    # No entities exist at all - show guidance
                    with ui.column().classes("items-center gap-2 py-4"):
                        ui.icon("group_add", size="md").classes("text-gray-400 dark:text-gray-500")
                        ui.label("No entities yet").classes(
                            "text-gray-500 dark:text-gray-400 font-medium"
                        )
                        ui.label("Add characters, locations, and more").classes(
                            "text-xs text-gray-400 dark:text-gray-500 text-center"
                        )
                        ui.label("using the button below.").classes(
                            "text-xs text-gray-400 dark:text-gray-500 text-center"
                        )
                else:
                    # Entities exist but are filtered out
                    ui.label("No matching entities").classes(
                        "text-gray-500 dark:text-gray-400 text-sm"
                    )
                    ui.label("Try adjusting filters or search").classes(
                        "text-xs text-gray-400 dark:text-gray-500"
                    )
            else:
                for entity in entities:
                    entity_list_item(
                        entity=entity,
                        on_select=self._select_entity,
                        selected=entity.id == self.state.selected_entity_id,
                    )

    def _filter_by_quality(self, entities: list[Entity]) -> list[Entity]:
        """Filter entities by quality score.

        Args:
            entities: List of entities to filter.

        Returns:
            Filtered list based on quality filter setting.
        """
        logger.debug(
            "Filtering %d entities by quality: %s", len(entities), self.state.entity_quality_filter
        )
        result = []
        for entity in entities:
            scores = entity.attributes.get("quality_scores") if entity.attributes else None
            avg = scores.get("average", 0) if scores else 0

            if self.state.entity_quality_filter == "high" and avg >= 8:
                result.append(entity)
            elif self.state.entity_quality_filter == "medium" and 6 <= avg < 8:
                result.append(entity)
            elif self.state.entity_quality_filter == "low" and avg < 6:
                result.append(entity)
        logger.debug("Quality filter returned %d entities", len(result))
        return result

    def _sort_entities(self, entities: list[Entity]) -> list[Entity]:
        """Sort entities by current sort setting.

        Args:
            entities: List of entities to sort.

        Returns:
            Sorted list based on sort setting.
        """
        sort_key = self.state.entity_sort_by
        descending = self.state.entity_sort_descending
        logger.debug("Sorting %d entities by %s (desc=%s)", len(entities), sort_key, descending)

        if sort_key == "name":
            return sorted(entities, key=lambda e: e.name.lower(), reverse=descending)
        elif sort_key == "type":
            return sorted(entities, key=lambda e: e.type, reverse=descending)
        elif sort_key == "quality":
            return sorted(
                entities,
                key=lambda e: (e.attributes.get("quality_scores") or {}).get("average", 0),
                reverse=descending,
            )
        elif sort_key == "relationships":

            def count_rels(entity: Entity) -> int:
                """Count relationships for sorting."""
                if self.state.world_db:
                    count = len(self.state.world_db.get_relationships(entity.id))
                    logger.debug("Entity %s has %d relationships", entity.id, count)
                    return count
                return 0

            return sorted(entities, key=count_rels, reverse=descending)
        else:
            return sorted(entities, key=lambda e: e.name.lower(), reverse=descending)

    def _toggle_type_filter(self, entity_type: str, enabled: bool) -> None:
        """Toggle entity type filter."""
        if enabled and entity_type not in self.state.entity_filter_types:
            self.state.entity_filter_types.append(entity_type)
        elif not enabled and entity_type in self.state.entity_filter_types:
            self.state.entity_filter_types.remove(entity_type)

        self._refresh_entity_list()
        if self._graph:
            self._graph.set_filter(self.state.entity_filter_types)

    def _on_search(self, e: Any) -> None:
        """Handle search input change."""
        self.state.entity_search_query = e.value
        self._refresh_entity_list()

        # Highlight matching nodes in the graph
        if self._graph:
            self._graph.highlight_search(e.value)

    def _update_search_scope(self, scope: str, enabled: bool) -> None:
        """Update search scope settings.

        Args:
            scope: 'names' or 'descriptions'.
            enabled: Whether this scope is enabled.
        """
        if scope == "names":
            self.state.entity_search_names = enabled
        elif scope == "descriptions":
            self.state.entity_search_descriptions = enabled
        logger.debug(f"Search scope updated: {scope}={enabled}")
        self._refresh_entity_list()

    def _on_quality_filter_change(self, e: Any) -> None:
        """Handle quality filter dropdown change."""
        self.state.entity_quality_filter = e.value
        logger.debug(f"Quality filter changed to: {e.value}")
        self._refresh_entity_list()

    def _on_sort_change(self, e: Any) -> None:
        """Handle sort dropdown change."""
        self.state.entity_sort_by = e.value
        logger.debug(f"Sort changed to: {e.value}")
        self._refresh_entity_list()

    def _toggle_sort_direction(self) -> None:
        """Toggle sort direction between ascending and descending."""
        self.state.entity_sort_descending = not self.state.entity_sort_descending
        logger.debug(
            f"Sort direction: {'descending' if self.state.entity_sort_descending else 'ascending'}"
        )

        # Update button icon
        if self._sort_direction_btn:
            icon = "arrow_downward" if self.state.entity_sort_descending else "arrow_upward"
            self._sort_direction_btn.props(f"icon={icon}")

        self._refresh_entity_list()

    def _register_keyboard_shortcuts(self) -> None:
        """Register keyboard shortcuts for the world page."""
        ui.keyboard(on_key=self._handle_keyboard)

    async def _handle_keyboard(self, e: Any) -> None:
        """Handle keyboard events.

        Args:
            e: Keyboard event with key and modifiers.
        """
        # Ctrl+F or Cmd+F focuses the search input (cross-platform)
        ctrl_pressed = getattr(e.modifiers, "ctrl", False)
        meta_pressed = getattr(e.modifiers, "meta", False)
        if (ctrl_pressed or meta_pressed) and e.key.lower() == "f":
            if self._search_input:
                await self._search_input.run_method("focus")
                await self._search_input.run_method("select")
            return

        # Esc clears the search
        if e.key == "Escape" and self._search_input:
            self._search_input.value = ""
            self.state.entity_search_query = ""
            self._refresh_entity_list()
            if self._graph:
                self._graph.highlight_search("")

    # Methods to be implemented by other mixins
    def _select_entity(self, entity: Entity) -> None:
        """Select an entity for editing - implemented by EditorMixin."""
        raise NotImplementedError

    def _show_add_dialog(self) -> None:
        """Show dialog to add new entity - implemented by EditorMixin."""
        raise NotImplementedError

    def _do_undo(self) -> None:
        """Execute undo operation - implemented by UndoMixin."""
        raise NotImplementedError

    def _do_redo(self) -> None:
        """Execute redo operation - implemented by UndoMixin."""
        raise NotImplementedError

    def _update_undo_redo_buttons(self) -> None:
        """Update undo/redo button states - implemented by UndoMixin."""
        raise NotImplementedError

    def _refresh_entity_editor(self) -> None:
        """Refresh the entity editor panel - implemented by EditorMixin."""
        raise NotImplementedError

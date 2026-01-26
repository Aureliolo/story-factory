"""World Timeline page - entity and event visualization on timeline.

This page displays entities with lifecycle information and world events
on a horizontal timeline visualization.
"""

import logging

from nicegui import ui

from src.services import ServiceContainer
from src.ui.components.world_timeline import WorldTimelineComponent
from src.ui.state import AppState
from src.ui.theme import ENTITY_COLORS

logger = logging.getLogger(__name__)


class WorldTimelinePage:
    """World Timeline page for visualizing entity lifespans and events.

    Features:
    - Entity lifespans as ranges (birth to death)
    - Events as point markers
    - Entity type filtering
    - Zoom and pan controls
    - Group by entity type
    """

    def __init__(self, state: AppState, services: ServiceContainer):
        """Initialize world timeline page.

        Args:
            state: Application state.
            services: Service container.
        """
        self.state = state
        self.services = services

        # UI references
        self._timeline: WorldTimelineComponent | None = None

    def build(self) -> None:
        """Build the world timeline page UI."""
        if not self.state.has_project:
            self._build_no_project_message()
            return

        if not self.state.world_db:
            self._build_no_world_message()
            return

        # Main content
        with ui.column().classes("w-full h-full gap-4 p-4"):
            self._build_info_section()
            self._build_timeline_section()
            self._build_help_section()

    def _build_no_project_message(self) -> None:
        """Build message when no project is selected."""
        with ui.column().classes("w-full min-h-96 items-center justify-center gap-4 py-16"):
            ui.icon("timeline", size="xl").classes("text-gray-400 dark:text-gray-500")
            ui.label("No Project Selected").classes("text-xl text-gray-500 dark:text-gray-400")
            ui.label("Select a project from the header to view its world timeline.").classes(
                "text-gray-400 dark:text-gray-500"
            )

    def _build_no_world_message(self) -> None:
        """Build message when no world data exists."""
        with ui.column().classes("w-full min-h-96 items-center justify-center gap-6 py-16"):
            ui.icon("public", size="xl").classes("text-gray-400")
            ui.label("No World Data").classes(
                "text-xl font-semibold text-gray-700 dark:text-gray-200"
            )
            ui.label("Build the world first to see entities and events on the timeline.").classes(
                "text-gray-500 dark:text-gray-400 text-center max-w-md"
            )

            ui.button(
                "Go to World Builder",
                on_click=lambda: ui.navigate.to("/world"),
                icon="arrow_forward",
            ).props("color=primary size=lg")

    def _build_info_section(self) -> None:
        """Build the info section with timeline statistics."""
        if not self.state.world_db:
            return

        world_db = self.state.world_db

        # Count entities by type
        character_count = world_db.count_entities("character")
        faction_count = world_db.count_entities("faction")
        location_count = world_db.count_entities("location")
        event_count = len(world_db.list_events())

        with ui.card().classes("w-full p-4"):
            with ui.row().classes("w-full items-center gap-6"):
                # Page title
                with ui.row().classes("items-center gap-2"):
                    ui.icon("timeline", size="lg").classes("text-blue-500")
                    ui.label("World Timeline").classes("text-xl font-semibold")

                ui.separator().props("vertical")

                # Entity counts
                with ui.row().classes("items-center gap-1"):
                    ui.html(
                        f'<div style="width: 12px; height: 12px; background: {ENTITY_COLORS["character"]}; border-radius: 50%;"></div>',
                        sanitize=False,
                    )
                    ui.label(f"{character_count} Characters").classes("text-sm")

                with ui.row().classes("items-center gap-1"):
                    ui.html(
                        f'<div style="width: 12px; height: 12px; background: {ENTITY_COLORS["faction"]}; border-radius: 50%;"></div>',
                        sanitize=False,
                    )
                    ui.label(f"{faction_count} Factions").classes("text-sm")

                with ui.row().classes("items-center gap-1"):
                    ui.html(
                        f'<div style="width: 12px; height: 12px; background: {ENTITY_COLORS["location"]}; border-radius: 50%;"></div>',
                        sanitize=False,
                    )
                    ui.label(f"{location_count} Locations").classes("text-sm")

                with ui.row().classes("items-center gap-1"):
                    ui.html(
                        '<div style="width: 12px; height: 12px; background: #FF5722; border-radius: 50%;"></div>',
                        sanitize=False,
                    )
                    ui.label(f"{event_count} Events").classes("text-sm")

                ui.space()

                # Link back to world page
                ui.button(
                    "World Builder",
                    on_click=lambda: ui.navigate.to("/world"),
                    icon="public",
                ).props("flat")

    def _build_timeline_section(self) -> None:
        """Build the main timeline visualization section."""
        with ui.card().classes("w-full p-4"):
            self._timeline = WorldTimelineComponent(
                world_db=self.state.world_db,
                services=self.services,
                on_item_select=self._handle_item_select,
                height=500,
            )
            self._timeline.build()

    def _build_help_section(self) -> None:
        """Build the help section with usage tips."""
        with ui.expansion("Timeline Tips", icon="help_outline").classes("w-full"):
            with ui.column().classes("gap-2 p-2"):
                with ui.row().classes("items-start gap-2"):
                    ui.icon("info", size="xs").classes("text-blue-500 mt-1")
                    ui.label(
                        "Entity lifespans are shown as horizontal bars when birth/death dates are available."
                    ).classes("text-sm text-gray-600 dark:text-gray-400")

                with ui.row().classes("items-start gap-2"):
                    ui.icon("info", size="xs").classes("text-blue-500 mt-1")
                    ui.label("Events appear as point markers on the timeline.").classes(
                        "text-sm text-gray-600 dark:text-gray-400"
                    )

                with ui.row().classes("items-start gap-2"):
                    ui.icon("info", size="xs").classes("text-blue-500 mt-1")
                    ui.label(
                        "Use the filter checkboxes to show/hide different entity types."
                    ).classes("text-sm text-gray-600 dark:text-gray-400")

                with ui.row().classes("items-start gap-2"):
                    ui.icon("info", size="xs").classes("text-blue-500 mt-1")
                    ui.label("Scroll to zoom, drag to pan. Click 'Fit' to show all items.").classes(
                        "text-sm text-gray-600 dark:text-gray-400"
                    )

                with ui.row().classes("items-start gap-2"):
                    ui.icon("lightbulb", size="xs").classes("text-amber-500 mt-1")
                    ui.label(
                        "To add lifecycle data to entities, edit them in the World page "
                        "and include birth/death dates in attributes."
                    ).classes("text-sm text-gray-600 dark:text-gray-400")

    def _handle_item_select(self, item_id: str) -> None:
        """Handle timeline item selection.

        Args:
            item_id: The selected item ID (e.g., 'entity-uuid' or 'event-uuid').
        """
        logger.debug(f"Timeline item selected: {item_id}")

        # Parse the item ID to determine type
        if item_id.startswith("entity-"):
            entity_id = item_id[7:]  # Remove 'entity-' prefix
            ui.notify(f"Selected entity: {entity_id[:8]}...", type="info")
            # Could navigate to entity detail or show info dialog
        elif item_id.startswith("event-"):
            event_id = item_id[6:]  # Remove 'event-' prefix
            ui.notify(f"Selected event: {event_id[:8]}...", type="info")

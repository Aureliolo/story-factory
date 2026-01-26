"""World timeline visualization component using vis-timeline.

Displays entity lifespans and events on a horizontal timeline.
"""

import json
import logging
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from nicegui import ui
from nicegui.elements.html import Html

from src.memory.timeline_types import TimelineItem
from src.ui.theme import ENTITY_COLORS

if TYPE_CHECKING:
    from src.memory.world_database import WorldDatabase
    from src.services import ServiceContainer

logger = logging.getLogger(__name__)


def _ensure_vis_timeline_loaded() -> None:
    """
    Inject the vis-timeline JavaScript and stylesheet into the current page.

    Adds the vis-timeline script and CSS to the document body so the timeline component can be initialized. Calling this function multiple times is safe because NiceGUI deduplicates injected resources.
    """
    # vis-timeline version tracked for Dependabot
    ui.add_body_html(
        '<script src="https://unpkg.com/vis-timeline@7.7.3/standalone/umd/vis-timeline-graph2d.min.js"></script>'
    )
    ui.add_body_html(
        '<link href="https://unpkg.com/vis-timeline@7.7.3/styles/vis-timeline-graph2d.min.css" rel="stylesheet" type="text/css" />'
    )


class WorldTimelineComponent:
    """Interactive timeline visualization for world entities and events.

    Features:
    - Entity lifespans as ranges
    - Events as point markers
    - Entity type filtering
    - Zoom and pan controls
    - Entity selection
    """

    def __init__(
        self,
        world_db: WorldDatabase | None = None,
        services: ServiceContainer | None = None,
        on_item_select: Callable[[str], Any] | None = None,
        height: int = 400,
    ):
        """
        Initialize the timeline component with optional world data, services, selection callback, and visual height.

        Parameters:
            world_db (WorldDatabase | None): WorldDatabase to visualize; if None the component will render a placeholder until set.
            services (ServiceContainer | None): Service container providing the timeline service used to fetch timeline items.
            on_item_select (Callable[[str], Any] | None): Callback invoked with the selected item's ID when a timeline item is selected.
            height (int): Height of the timeline container in pixels.
        """
        self.world_db = world_db
        self.services = services
        self.on_item_select = on_item_select
        self.height = height

        self._container: Html | None = None
        self._filter_types: list[str] = [
            "character",
            "faction",
            "location",
            "item",
            "concept",
            "event",
        ]
        self._include_events = True
        self._callback_id = f"timeline_select_{uuid.uuid4().hex[:8]}"

    def build(self) -> None:
        """
        Build the timeline UI and initialize its interactive components.

        Ensures the vis-timeline assets are loaded, registers the item selection callback if provided, creates the control row (filters, zoom controls, refresh) and the timeline container, and performs the initial render of timeline content.
        """
        logger.debug("Building WorldTimelineComponent")
        # Ensure vis-timeline library is loaded
        _ensure_vis_timeline_loaded()

        # Register event handler for item selection
        if self.on_item_select:

            def handle_item_select(e: Any) -> None:
                """
                Handle a UI selection event by invoking the configured on_item_select callback with the selected item ID.

                Parameters:
                    e (Any): Event object whose `args` mapping may contain the key `"item_id"`. If `"item_id"` is present and `self.on_item_select` is set, the callback is called with that ID. Otherwise no action is taken.
                """
                item_id = e.args.get("item_id") if e.args else None
                if item_id and self.on_item_select:
                    self.on_item_select(item_id)

            ui.on(self._callback_id, handle_item_select)

        with ui.column().classes("w-full gap-4"):
            # Controls
            with ui.row().classes("w-full items-center gap-4 flex-wrap"):
                ui.label("Filter:").classes("text-sm font-medium")

                # Entity type filters
                ui.checkbox(
                    "Characters",
                    value="character" in self._filter_types,
                    on_change=lambda e: self._toggle_filter("character", e.value),
                )
                ui.checkbox(
                    "Factions",
                    value="faction" in self._filter_types,
                    on_change=lambda e: self._toggle_filter("faction", e.value),
                )
                ui.checkbox(
                    "Locations",
                    value="location" in self._filter_types,
                    on_change=lambda e: self._toggle_filter("location", e.value),
                )
                ui.checkbox(
                    "Items",
                    value="item" in self._filter_types,
                    on_change=lambda e: self._toggle_filter("item", e.value),
                )
                ui.checkbox(
                    "Events",
                    value=self._include_events,
                    on_change=lambda e: self._toggle_events(e.value),
                )

                ui.space()

                # Zoom controls
                with ui.row().classes("gap-2"):
                    ui.button(
                        icon="zoom_out",
                        on_click=lambda: ui.run_javascript(
                            "if(window.worldTimeline) window.worldTimeline.zoomOut(0.5)"
                        ),
                    ).props("flat round size=sm").tooltip("Zoom out")
                    ui.button(
                        icon="zoom_in",
                        on_click=lambda: ui.run_javascript(
                            "if(window.worldTimeline) window.worldTimeline.zoomIn(0.5)"
                        ),
                    ).props("flat round size=sm").tooltip("Zoom in")
                    ui.button(
                        icon="fit_screen",
                        on_click=lambda: ui.run_javascript(
                            "if(window.worldTimeline) window.worldTimeline.fit()"
                        ),
                    ).props("flat round size=sm").tooltip("Fit all items")

                # Refresh
                ui.button(
                    icon="refresh",
                    on_click=self.refresh,
                ).props("flat round size=sm").tooltip("Refresh timeline")

            # Timeline container
            self._container = (
                ui.html("", sanitize=False)
                .classes("w-full border dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800")
                .style(f"height: {self.height}px;")
            )

            # Initial render
            self._render_timeline()

    def set_world_db(self, world_db: WorldDatabase | None) -> None:
        """
        Set the world database used by the component and re-render the timeline.

        Parameters:
            world_db (WorldDatabase | None): WorldDatabase to visualize, or None to clear the current database.
        """
        self.world_db = world_db
        self._render_timeline()

    def refresh(self) -> None:
        """Refresh the timeline visualization."""
        self._render_timeline()

    def _render_timeline(self) -> None:
        """Render the timeline HTML and JavaScript."""
        if not self._container:
            return

        if not self.world_db or not self.services:
            self._container.content = """
                <style>
                    .timeline-empty { color: #6b7280; }
                    .dark .timeline-empty { color: #9ca3af; }
                </style>
                <div class="timeline-empty" style="
                    height: 100%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                ">
                    <p>No world data to display. Create or load a project first.</p>
                </div>
            """
            return

        # Get timeline data from service
        # Filter entity types (exclude 'event' since it's handled separately)
        entity_types = [t for t in self._filter_types if t != "event"]
        # Pass entity_types directly - None means no filter, [] means include none
        logger.debug(
            f"Fetching timeline items: entity_types={entity_types}, "
            f"include_events={self._include_events}"
        )
        items = self.services.timeline.get_timeline_items(
            self.world_db,
            entity_types=entity_types,
            include_events=self._include_events,
        )

        if not items:
            self._container.content = """
                <style>
                    .timeline-empty { color: #6b7280; }
                    .dark .timeline-empty { color: #9ca3af; }
                </style>
                <div class="timeline-empty" style="
                    height: 100%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                ">
                    <p>No timeline data available. Add entities with lifecycle information.</p>
                </div>
            """
            return

        # Convert items to vis-timeline format
        timeline_data = self._items_to_visjs(items)
        groups = self._get_groups()

        container_id = f"world-timeline-{uuid.uuid4().hex[:8]}"

        # HTML container
        html = f"""
        <style>
            #{container_id} {{
                border: none;
                background: transparent;
            }}
            #{container_id} .vis-item {{
                border-radius: 4px;
                border-width: 2px;
            }}
            #{container_id} .vis-item.vis-point {{
                border-radius: 50%;
            }}
            #{container_id} .vis-item.vis-selected {{
                border-color: #FFD700 !important;
                box-shadow: 0 0 8px rgba(255, 215, 0, 0.5);
            }}
            .dark #{container_id} .vis-time-axis .vis-text {{
                color: #e5e7eb;
            }}
            .dark #{container_id} .vis-labelset .vis-label {{
                color: #e5e7eb;
            }}
        </style>
        <div id="{container_id}" style="height: {self.height - 20}px;"></div>
        """

        self._container.content = html

        # JavaScript initialization
        js = f"""
        (function() {{
            var attempts = 0;
            var maxAttempts = 50;

            function initTimeline() {{
                attempts++;
                if (typeof vis === 'undefined' || typeof vis.Timeline === 'undefined') {{
                    if (attempts >= maxAttempts) {{
                        console.error('vis-timeline failed to load');
                        return;
                    }}
                    setTimeout(initTimeline, 100);
                    return;
                }}

                var container = document.getElementById('{container_id}');
                if (!container) return;

                var items = new vis.DataSet({json.dumps(timeline_data)});
                var groups = new vis.DataSet({json.dumps(groups)});

                var isDarkMode = document.body.classList.contains('dark') ||
                                 document.documentElement.classList.contains('dark');

                var options = {{
                    height: '{self.height - 20}px',
                    showCurrentTime: false,
                    zoomMin: 1000 * 60 * 60 * 24 * 365,  // 1 year
                    zoomMax: 1000 * 60 * 60 * 24 * 365 * 1000,  // 1000 years
                    orientation: 'top',
                    stack: true,
                    stackSubgroups: true,
                    margin: {{ item: 10 }},
                    selectable: true,
                    multiselect: false,
                    tooltip: {{
                        followMouse: true,
                        overflowMethod: 'cap'
                    }}
                }};

                var timeline = new vis.Timeline(container, items, groups, options);

                // Handle selection
                timeline.on('select', function(properties) {{
                    if (properties.items && properties.items.length > 0) {{
                        var itemId = properties.items[0];
                        if ('{self._callback_id}') {{
                            emitEvent('{self._callback_id}', {{item_id: itemId}});
                        }}
                    }}
                }});

                // Store reference for external control
                window.worldTimeline = timeline;

                // Fit to show all items
                timeline.fit();
            }}

            initTimeline();
        }})();
        """

        ui.run_javascript(js)
        logger.debug(f"Timeline rendered with {len(items)} items")

    def _items_to_visjs(self, items: list[TimelineItem]) -> list[dict]:
        """
        Convert TimelineItem objects into vis-timeline item dictionaries suitable for rendering.

        Parameters:
            items (list[TimelineItem]): Timeline items to convert; items without a valid start are skipped.

        Returns:
            list[dict]: vis-timeline item dictionaries containing keys such as 'id', 'content', 'group', 'title', 'style', 'start', optional 'end' (for ranges), and 'type' ('point' or 'range').
        """
        visjs_items = []
        base_year = 1000  # Base year for relative ordering

        for item in items:
            vis_item: dict[str, Any] = {
                "id": item.id,
                "content": item.label,
                "group": item.group or item.item_type,
                "title": item.description or item.label,
                "style": f"background-color: {item.color}; border-color: {item.color};",
            }

            # Determine start date
            if item.start.year is not None:
                vis_item["start"] = f"{item.start.year:04d}-06-01"
            elif item.start.relative_order is not None:
                # Map relative order to years
                pseudo_year = base_year + item.start.relative_order
                vis_item["start"] = f"{pseudo_year:04d}-06-01"
            else:
                continue  # Skip items without valid start

            # Determine end date (for ranges)
            if item.end:
                if item.end.year is not None:
                    vis_item["end"] = f"{item.end.year:04d}-12-31"
                    vis_item["type"] = "range"
                elif item.end.relative_order is not None:
                    pseudo_year = base_year + item.end.relative_order
                    vis_item["end"] = f"{pseudo_year:04d}-12-31"
                    vis_item["type"] = "range"
                else:
                    vis_item["type"] = "point"
            else:
                vis_item["type"] = "point"

            visjs_items.append(vis_item)

        return visjs_items

    def _get_groups(self) -> list[dict]:
        """Get group definitions for vis-timeline.

        Returns:
            List of group dictionaries.
        """
        group_defs = [
            {
                "id": "character",
                "content": "Characters",
                "style": f"color: {ENTITY_COLORS['character']}",
            },
            {"id": "faction", "content": "Factions", "style": f"color: {ENTITY_COLORS['faction']}"},
            {
                "id": "location",
                "content": "Locations",
                "style": f"color: {ENTITY_COLORS['location']}",
            },
            {"id": "item", "content": "Items", "style": f"color: {ENTITY_COLORS['item']}"},
            {"id": "concept", "content": "Concepts", "style": f"color: {ENTITY_COLORS['concept']}"},
            {"id": "event", "content": "Events", "style": "color: #FF5722"},
        ]

        # Only include groups that are in the filter
        return [
            g
            for g in group_defs
            if g["id"] in self._filter_types or (g["id"] == "event" and self._include_events)
        ]

    def _toggle_filter(self, entity_type: str, enabled: bool) -> None:
        """
        Update the active entity-type filter and re-render the timeline.

        Parameters:
            entity_type (str): The entity type to enable or disable (e.g., "character", "faction").
            enabled (bool): True to include the entity type in the filters, False to exclude it.
        """
        logger.debug(f"Toggle filter {entity_type}: enabled={enabled}")
        if enabled and entity_type not in self._filter_types:
            self._filter_types.append(entity_type)
        elif not enabled and entity_type in self._filter_types:
            self._filter_types.remove(entity_type)
        self._render_timeline()

    def _toggle_events(self, enabled: bool) -> None:
        """
        Toggle inclusion of event items in the active timeline filters.

        Parameters:
            enabled: True to include events in the timeline, False to exclude them.
        """
        logger.debug(f"Toggle events: enabled={enabled}")
        self._include_events = enabled
        if enabled and "event" not in self._filter_types:
            self._filter_types.append("event")
        elif not enabled and "event" in self._filter_types:
            self._filter_types.remove("event")
        self._render_timeline()

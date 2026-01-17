"""Timeline visualization component using vis-timeline."""

import logging
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from nicegui import ui
from nicegui.elements.html import Html

from memory.story_state import StoryState

if TYPE_CHECKING:
    from settings import Settings

logger = logging.getLogger(__name__)


def _ensure_vis_timeline_loaded() -> None:
    """Add vis-timeline library scripts to current page.

    This must be called for each page/client that needs the timeline,
    since ui.add_body_html only adds to the current client's page.
    NiceGUI handles deduplication if called multiple times in same page.
    """
    # vis-timeline version tracked in /package.json for Dependabot
    ui.add_body_html(
        '<link href="https://unpkg.com/vis-timeline@7.7.3/styles/vis-timeline-graph2d.min.css" rel="stylesheet" type="text/css" />'
    )
    ui.add_body_html(
        '<script src="https://unpkg.com/vis-timeline@7.7.3/standalone/umd/vis-timeline-graph2d.min.js"></script>'
    )


def _extract_timeline_data(story_state: StoryState | None) -> dict[str, Any]:
    """Extract timeline data from story state.

    Args:
        story_state: Story state to extract data from.

    Returns:
        Dictionary with 'items' and 'groups' for vis-timeline.
    """
    if not story_state:
        return {"items": [], "groups": []}

    items = []
    groups = []
    item_id = 0

    # Add groups for different tracks
    groups.append({"id": "events", "content": "Story Events"})
    groups.append({"id": "chapters", "content": "Chapters"})
    groups.append({"id": "characters", "content": "Characters"})
    groups.append({"id": "locations", "content": "Locations"})

    # Add timeline events from story_state.timeline
    for idx, event in enumerate(story_state.timeline):
        items.append(
            {
                "id": item_id,
                "group": "events",
                "content": event,
                "start": idx,
                "type": "box",
                "className": "timeline-event",
            }
        )
        item_id += 1

    # Add chapter boundaries
    for chapter in story_state.chapters:
        items.append(
            {
                "id": item_id,
                "group": "chapters",
                "content": f"Ch{chapter.number}: {chapter.title}",
                "start": chapter.number - 1,
                "type": "range",
                "end": chapter.number,
                "className": "timeline-chapter",
            }
        )
        item_id += 1

    # Add character first appearances
    for char in story_state.characters:
        # Find first arc progress entry or use 0
        first_appearance = 0
        if char.arc_progress:
            first_appearance = min(char.arc_progress.keys())

        items.append(
            {
                "id": item_id,
                "group": "characters",
                "content": char.name,
                "start": first_appearance,
                "type": "point",
                "className": "timeline-character",
                "title": f"{char.name} - {char.role}",
            }
        )
        item_id += 1

    # Add location introductions from world database if available
    # For now, we'll use a simple approach - locations appear at the beginning
    if hasattr(story_state, "world_description") and story_state.world_description:
        items.append(
            {
                "id": item_id,
                "group": "locations",
                "content": "Primary Setting",
                "start": 0,
                "type": "background",
                "className": "timeline-location",
            }
        )
        item_id += 1

    return {"items": items, "groups": groups}


def _render_timeline_html(
    story_state: StoryState | None,
    settings: "Settings",
    height: int = 400,
    container_id: str = "timeline-container",
    editable: bool = True,
) -> dict[str, str]:
    """Render timeline HTML and JavaScript.

    Args:
        story_state: Story state to visualize.
        settings: Application settings.
        height: Timeline container height in pixels.
        container_id: Unique ID for the timeline container.
        editable: Whether timeline items can be edited/moved.

    Returns:
        Dictionary with 'html' and 'js' keys containing the rendered content.
    """
    data = _extract_timeline_data(story_state)

    # Build HTML container
    html = f'<div id="{container_id}" style="height: {height}px; border: 1px solid #ccc; border-radius: 4px;"></div>'

    # Build JavaScript initialization
    items_json = (
        str(data["items"]).replace("'", '"').replace("True", "true").replace("False", "false")
    )
    groups_json = (
        str(data["groups"]).replace("'", '"').replace("True", "true").replace("False", "false")
    )

    # Timeline options
    options = {
        "editable": editable,
        "stack": True,
        "horizontalScroll": True,
        "zoomKey": "ctrlKey",
        "orientation": "top",
        "showCurrentTime": False,
        "margin": {"item": 10, "axis": 5},
    }
    options_json = str(options).replace("'", '"').replace("True", "true").replace("False", "false")

    js = f"""
    (function() {{
        function initTimeline() {{
            if (typeof vis === 'undefined' || !vis.Timeline) {{
                console.error('vis-timeline not loaded yet, retrying...');
                setTimeout(initTimeline, 100);
                return;
            }}

            const container = document.getElementById('{container_id}');
            if (!container) {{
                console.error('Timeline container not found: {container_id}');
                return;
            }}

            try {{
                const items = new vis.DataSet({items_json});
                const groups = new vis.DataSet({groups_json});
                const options = {options_json};

                const timeline = new vis.Timeline(container, items, groups, options);

                // Store globally for interaction
                window.timeline_{container_id.replace("-", "_")} = timeline;
                window.timelineItems_{container_id.replace("-", "_")} = items;
                window.timelineGroups_{container_id.replace("-", "_")} = groups;

                console.log('Timeline initialized successfully');
            }} catch (error) {{
                console.error('Error initializing timeline:', error);
            }}
        }}

        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initTimeline);
        }} else {{
            initTimeline();
        }}
    }})();
    """

    return {"html": html, "js": js}


class TimelineComponent:
    """Interactive timeline visualization for story events.

    Features:
    - vis-timeline powered visualization
    - Multiple tracks (events, chapters, characters, locations)
    - Editable items (drag to reorder)
    - Zoom and pan
    - Tracks story progression chronologically
    """

    def __init__(
        self,
        story_state: StoryState | None = None,
        settings: "Settings | None" = None,
        on_item_move: Callable[[str, Any], None] | None = None,
        height: int = 400,
        editable: bool = True,
    ):
        """Initialize timeline component.

        Args:
            story_state: StoryState to visualize.
            settings: Application settings for configuration values.
            on_item_move: Callback when an item is moved (receives item_id, new_time).
            height: Timeline container height in pixels.
            editable: Whether items can be edited/moved.
        """
        from settings import Settings

        self.story_state = story_state
        self.settings = settings or Settings.load()
        self.on_item_move = on_item_move
        self.height = height
        self.editable = editable

        self._container: Html | None = None
        self._container_id = f"timeline-container-{uuid.uuid4().hex[:8]}"

    def build(self) -> None:
        """Build the timeline UI."""
        # Ensure vis-timeline library is loaded
        _ensure_vis_timeline_loaded()

        with ui.column().classes("w-full"):
            # Controls
            with ui.row().classes("w-full items-center gap-4 mb-2"):
                ui.label("Story Timeline").classes("text-lg font-semibold")
                ui.space()

                # Zoom controls
                ui.button(
                    icon="zoom_in",
                    on_click=self._zoom_in,
                ).props("flat round dense").tooltip("Zoom in (Ctrl+Scroll)")

                ui.button(
                    icon="zoom_out",
                    on_click=self._zoom_out,
                ).props("flat round dense").tooltip("Zoom out (Ctrl+Scroll)")

                ui.button(
                    icon="fit_screen",
                    on_click=self._fit_timeline,
                ).props("flat round dense").tooltip("Fit all events")

                # Refresh button
                ui.button(
                    icon="refresh",
                    on_click=self.refresh,
                ).props("flat round dense").tooltip("Refresh timeline")

            # Timeline container
            self._container = ui.html("", sanitize=False).classes(
                "w-full border dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800"
            )

            # Initial render
            self._render_timeline()

    def set_story_state(self, story_state: StoryState | None) -> None:
        """Set the story state and refresh.

        Args:
            story_state: StoryState to visualize.
        """
        self.story_state = story_state
        self._render_timeline()

    def refresh(self) -> None:
        """Refresh the timeline visualization."""
        self._render_timeline()

    def _render_timeline(self) -> None:
        """Render the timeline HTML."""
        if not self._container:
            return

        if not self.story_state:
            self._container.content = """
                <style>
                    .timeline-empty { color: #6b7280; }
                    .dark .timeline-empty { color: #9ca3af; }
                </style>
                <div class="timeline-empty" style="
                    height: 400px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                ">
                    <p>No story data to display. Create or load a project first.</p>
                </div>
            """
            return

        # Generate vis-timeline HTML and JavaScript separately
        result = _render_timeline_html(
            story_state=self.story_state,
            settings=self.settings,
            height=self.height,
            container_id=self._container_id,
            editable=self.editable,
        )

        # Set HTML content
        self._container.content = result["html"]

        # Run JavaScript initialization
        ui.run_javascript(result["js"])

        # Add custom styles for timeline items
        ui.run_javascript(
            """
            const style = document.createElement('style');
            style.textContent = `
                .timeline-event {
                    background-color: #3b82f6;
                    border-color: #2563eb;
                    color: white;
                }
                .timeline-chapter {
                    background-color: #10b981;
                    border-color: #059669;
                    color: white;
                }
                .timeline-character {
                    background-color: #f59e0b;
                    border-color: #d97706;
                    color: white;
                }
                .timeline-location {
                    background-color: #8b5cf6;
                    border-color: #7c3aed;
                    color: white;
                }
                .vis-item {
                    font-size: 12px;
                    border-radius: 4px;
                }
                .vis-item.vis-selected {
                    border-width: 2px;
                }
            `;
            if (!document.querySelector('style[data-timeline-styles]')) {
                style.setAttribute('data-timeline-styles', 'true');
                document.head.appendChild(style);
            }
            """
        )

    def _zoom_in(self) -> None:
        """Zoom in on the timeline."""
        var_name = f"timeline_{self._container_id.replace('-', '_')}"
        ui.run_javascript(
            f"""
            if (window.{var_name}) {{
                window.{var_name}.zoomIn(0.2);
            }}
            """
        )

    def _zoom_out(self) -> None:
        """Zoom out on the timeline."""
        var_name = f"timeline_{self._container_id.replace('-', '_')}"
        ui.run_javascript(
            f"""
            if (window.{var_name}) {{
                window.{var_name}.zoomOut(0.2);
            }}
            """
        )

    def _fit_timeline(self) -> None:
        """Fit all events in the timeline view."""
        var_name = f"timeline_{self._container_id.replace('-', '_')}"
        ui.run_javascript(
            f"""
            if (window.{var_name}) {{
                window.{var_name}.fit();
            }}
            """
        )

"""Graph visualization component using vis.js."""

from collections.abc import Callable
from typing import Any

from nicegui import ui
from nicegui.elements.html import Html

from memory.world_database import WorldDatabase
from ui.graph_renderer import render_graph_html

# Flag to track if vis-network script has been added
_vis_network_loaded = False


def _ensure_vis_network_loaded() -> None:
    """Ensure vis-network library is loaded (once per app)."""
    global _vis_network_loaded
    if not _vis_network_loaded:
        # Add vis-network script to body (allows script tags)
        ui.add_body_html(
            "<!-- vis-network version tracked in /package.json for Dependabot -->"
            '<script src="https://unpkg.com/vis-network@10.0.2/standalone/umd/vis-network.min.js"></script>'
        )
        _vis_network_loaded = True


class GraphComponent:
    """Interactive graph visualization for world entities.

    Features:
    - vis.js powered graph
    - Entity type filtering
    - Multiple layout options
    - Node selection
    - Zoom and pan
    """

    def __init__(
        self,
        world_db: WorldDatabase | None = None,
        on_node_select: Callable[[str], Any] | None = None,
        height: int = 500,
    ):
        """Initialize graph component.

        Args:
            world_db: WorldDatabase to visualize.
            on_node_select: Callback when a node is selected.
            height: Graph container height in pixels.
        """
        self.world_db = world_db
        self.on_node_select = on_node_select
        self.height = height

        self._container: Html | None = None
        self._filter_types: list[str] = ["character", "location"]
        self._layout = "force-directed"
        self._selected_entity_id: str | None = None

    def build(self) -> None:
        """Build the graph UI."""
        # Ensure vis-network library is loaded
        _ensure_vis_network_loaded()

        with ui.column().classes("w-full"):
            # Controls
            with ui.row().classes("w-full items-center gap-4 mb-2"):
                # Type filter
                ui.label("Show:").classes("text-sm font-medium")
                ui.checkbox(
                    "Characters",
                    value="character" in self._filter_types,
                    on_change=lambda e: self._toggle_filter("character", e.value),
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
                    "Factions",
                    value="faction" in self._filter_types,
                    on_change=lambda e: self._toggle_filter("faction", e.value),
                )

                ui.space()

                # Layout selector
                ui.select(
                    options=["force-directed", "hierarchical", "circular"],
                    value=self._layout,
                    label="Layout",
                    on_change=self._on_layout_change,
                ).classes("w-40")

                # Refresh button
                ui.button(
                    icon="refresh",
                    on_click=self.refresh,
                ).props("flat round")

            # Graph container (no script tags allowed here)
            self._container = (
                ui.html("", sanitize=False)
                .classes("w-full border dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800")
                .style(f"height: {self.height}px;")
            )

            # Initial render
            self._render_graph()

    def set_world_db(self, world_db: WorldDatabase | None) -> None:
        """Set the world database and refresh.

        Args:
            world_db: WorldDatabase to visualize.
        """
        self.world_db = world_db
        self._render_graph()

    def set_selected(self, entity_id: str | None) -> None:
        """Set the selected entity.

        Args:
            entity_id: Entity ID to highlight.
        """
        self._selected_entity_id = entity_id
        self._render_graph()

    def set_filter(self, types: list[str]) -> None:
        """Set entity type filter.

        Args:
            types: List of entity types to show.
        """
        self._filter_types = types
        self._render_graph()

    def refresh(self) -> None:
        """Refresh the graph visualization."""
        self._render_graph()

    def _render_graph(self) -> None:
        """Render the graph HTML."""
        if not self._container:
            return

        if not self.world_db:
            self._container.content = """
                <style>
                    .graph-empty { color: #6b7280; }
                    .dark .graph-empty { color: #9ca3af; }
                </style>
                <div class="graph-empty" style="
                    height: 100%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                ">
                    <p>No world data to display. Create or load a project first.</p>
                </div>
            """
            return

        # Generate vis.js HTML and JavaScript separately
        html, js = render_graph_html(
            world_db=self.world_db,
            filter_types=self._filter_types if self._filter_types else None,
            layout=self._layout,
            height=self.height - 20,  # Account for border
            selected_entity_id=self._selected_entity_id,
            container_id="graph-container-main",
        )

        # Set HTML content (no script tags)
        self._container.content = html

        # Run JavaScript initialization
        ui.run_javascript(js)

        # Add JavaScript handler for node selection
        if self.on_node_select:
            ui.run_javascript(
                """
                if (window.graphNetwork) {
                    window.graphNetwork.off('click');
                    window.graphNetwork.on('click', function(params) {
                        if (params.nodes.length > 0) {
                            const nodeId = params.nodes[0];
                            // Send to Python via NiceGUI
                            window.parent.postMessage({
                                type: 'graph_node_select',
                                nodeId: nodeId
                            }, '*');
                        }
                    });
                }
            """
            )

    def _toggle_filter(self, entity_type: str, enabled: bool) -> None:
        """Toggle a type filter.

        Args:
            entity_type: Entity type to toggle.
            enabled: Whether to enable the filter.
        """
        if enabled and entity_type not in self._filter_types:
            self._filter_types.append(entity_type)
        elif not enabled and entity_type in self._filter_types:
            self._filter_types.remove(entity_type)

        self._render_graph()

    def _on_layout_change(self, e) -> None:
        """Handle layout change.

        Args:
            e: Change event with new value.
        """
        self._layout = e.value
        self._render_graph()


def mini_graph(
    world_db: WorldDatabase | None,
    height: int = 200,
    filter_types: list[str] | None = None,
) -> None:
    """Create a small, non-interactive graph preview.

    Args:
        world_db: WorldDatabase to visualize.
        height: Container height.
        filter_types: Entity types to show.
    """
    if not world_db:
        ui.label("No world data").classes("text-gray-500 dark:text-gray-400 text-sm")
        return

    # Ensure vis-network library is loaded
    _ensure_vis_network_loaded()

    # Use unique container ID for mini graphs
    import uuid

    container_id = f"mini-graph-{uuid.uuid4().hex[:8]}"

    html, js = render_graph_html(
        world_db=world_db,
        filter_types=filter_types or ["character", "location"],
        layout="force-directed",
        height=height,
        container_id=container_id,
    )

    # Add HTML container (no script tags)
    ui.html(html, sanitize=False).classes(
        "w-full border dark:border-gray-700 rounded bg-white dark:bg-gray-800"
    ).style(f"height: {height}px;")

    # Run JavaScript initialization
    ui.run_javascript(js)

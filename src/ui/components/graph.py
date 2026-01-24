"""Graph visualization component using vis.js."""

import uuid
from collections.abc import Callable
from typing import Any

from nicegui import ui
from nicegui.elements.html import Html

from src.memory.world_database import WorldDatabase
from src.settings import Settings
from src.ui.graph_renderer import render_graph_html


def _ensure_vis_network_loaded() -> None:
    """Add vis-network library script to current page.

    This must be called for each page/client that needs the graph,
    since ui.add_body_html only adds to the current client's page.
    NiceGUI handles deduplication if called multiple times in same page.
    """
    # vis-network version tracked in /package.json for Dependabot
    ui.add_body_html(
        '<script src="https://unpkg.com/vis-network@10.0.2/standalone/umd/vis-network.min.js"></script>'
    )


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
        settings: Settings | None = None,
        on_node_select: Callable[[str], Any] | None = None,
        on_edge_select: Callable[[str], Any] | None = None,
        on_create_relationship: Callable[[str, str], Any] | None = None,
        on_edge_context_menu: Callable[[str], Any] | None = None,
        height: int = 500,
    ):
        """Initialize graph component.

        Args:
            world_db: WorldDatabase to visualize.
            settings: Application settings for configuration values.
            on_node_select: Callback when a node is selected.
            on_edge_select: Callback when an edge is selected (passes relationship ID).
            on_create_relationship: Callback when creating relationship via drag (source_id, target_id).
            on_edge_context_menu: Callback for edge right-click (edge_id).
            height: Graph container height in pixels.
        """
        from src.settings import Settings

        self.world_db = world_db
        self.settings = settings or Settings.load()
        self.on_node_select = on_node_select
        self.on_edge_select = on_edge_select
        self.on_create_relationship = on_create_relationship
        self.on_edge_context_menu = on_edge_context_menu
        self.height = height

        self._container: Html | None = None
        # Use filter types from settings if available, otherwise default to all
        self._filter_types: list[str] = (
            list(self.settings.graph_filter_types)
            if hasattr(self.settings, "graph_filter_types")
            else ["character", "location", "item", "faction", "concept"]
        )
        self._layout = "force-directed"
        self._selected_entity_id: str | None = None
        self._callback_id = f"graph_node_select_{uuid.uuid4().hex[:8]}"
        self._edge_callback_id = f"graph_edge_select_{uuid.uuid4().hex[:8]}"
        self._create_rel_callback_id = f"graph_create_relationship_{uuid.uuid4().hex[:8]}"
        self._edge_context_callback_id = f"graph_edge_context_menu_{uuid.uuid4().hex[:8]}"

    def build(self) -> None:
        """Build the graph UI."""
        # Ensure vis-network library is loaded
        _ensure_vis_network_loaded()

        # Register event handler for node selection
        if self.on_node_select:

            def handle_node_select(e: Any) -> None:
                """Handle node selection event.

                Args:
                    e: Event containing node_id.
                """
                node_id = e.args.get("node_id") if e.args else None
                if node_id and self.on_node_select:
                    self.on_node_select(node_id)

            ui.on(self._callback_id, handle_node_select)

        # Register event handler for edge selection
        if self.on_edge_select:

            def handle_edge_select(e: Any) -> None:
                """Handle edge selection event.

                Args:
                    e: Event containing edge_id.
                """
                edge_id = e.args.get("edge_id") if e.args else None
                if edge_id and self.on_edge_select:
                    self.on_edge_select(edge_id)

            ui.on(self._edge_callback_id, handle_edge_select)

        # Register event handler for creating relationships via drag
        if self.on_create_relationship:

            def handle_create_relationship(e) -> None:
                """Handle relationship creation event.

                Args:
                    e: Event containing source_id and target_id.
                """
                source_id = e.args.get("source_id") if e.args else None
                target_id = e.args.get("target_id") if e.args else None
                if source_id and target_id and self.on_create_relationship:
                    self.on_create_relationship(source_id, target_id)

            ui.on(self._create_rel_callback_id, handle_create_relationship)

        # Register event handler for edge context menu
        if self.on_edge_context_menu:

            def handle_edge_context_menu(e) -> None:
                """Handle right-click context menu on graph edge."""
                edge_id = e.args.get("edge_id") if e.args else None
                if edge_id and self.on_edge_context_menu:
                    self.on_edge_context_menu(edge_id)

            ui.on(self._edge_context_callback_id, handle_edge_context_menu)

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
                ui.checkbox(
                    "Concepts",
                    value="concept" in self._filter_types,
                    on_change=lambda e: self._toggle_filter("concept", e.value),
                )

                ui.space()

                # Help text for drag-to-connect
                ui.label("Shift+Drag to connect").classes(
                    "text-xs text-gray-500 dark:text-gray-400"
                ).tooltip("Hold Shift and drag from one node to another to create a relationship")

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
        result = render_graph_html(
            world_db=self.world_db,
            settings=self.settings,
            filter_types=self._filter_types if self._filter_types else None,
            layout=self._layout,
            height=self.height - 20,  # Account for border
            selected_entity_id=self._selected_entity_id,
            container_id="graph-container-main",
            create_rel_callback_id=self._create_rel_callback_id
            if self.on_create_relationship
            else "",
            edge_context_callback_id=self._edge_context_callback_id
            if self.on_edge_context_menu
            else "",
        )

        # Set HTML content (no script tags)
        self._container.content = result.html

        # Run JavaScript initialization
        ui.run_javascript(result.js)

        # Add JavaScript handler for node and edge selection
        if self.on_node_select or self.on_edge_select:
            node_callback = self._callback_id if self.on_node_select else ""
            edge_callback = self._edge_callback_id if self.on_edge_select else ""
            ui.run_javascript(
                f"""
                if (window.graphNetwork) {{
                    window.graphNetwork.off('click');
                    window.graphNetwork.on('click', function(params) {{
                        if (params.nodes.length > 0) {{
                            const nodeId = params.nodes[0];
                            if ('{node_callback}') {{
                                emitEvent('{node_callback}', {{node_id: nodeId}});
                            }}
                        }} else if (params.edges.length > 0) {{
                            const edgeId = params.edges[0];
                            if ('{edge_callback}') {{
                                emitEvent('{edge_callback}', {{edge_id: edgeId}});
                            }}
                        }}
                    }});
                }}
            """
            )

    def highlight_search(self, query: str) -> None:
        """Highlight nodes matching a search query.

        Args:
            query: Search query to match against node labels.
        """
        if not query:
            # Clear highlighting
            ui.run_javascript(
                """
                if (window.graphNetwork && window.graphNodes) {
                    // Reset all nodes to original styling
                    var updates = [];
                    window.graphNodes.forEach(function(node) {
                        if (node._originalColor) {
                            updates.push({
                                id: node.id,
                                color: node._originalColor,
                                borderWidth: 1,
                                font: { color: node._originalFontColor || '#333' }
                            });
                        }
                    });
                    if (updates.length > 0) {
                        window.graphNodes.update(updates);
                    }
                }
                """
            )
            return

        # Highlight matching nodes and dim non-matching ones
        escaped_query = query.replace("\\", "\\\\").replace("'", "\\'").lower()
        ui.run_javascript(
            f"""
            if (window.graphNetwork && window.graphNodes) {{
                var query = '{escaped_query}';
                var updates = [];

                window.graphNodes.forEach(function(node) {{
                    var label = (node.label || '').toLowerCase();
                    var matches = label.includes(query);

                    // Store original colors on first highlight
                    if (!node._originalColor) {{
                        node._originalColor = node.color;
                        node._originalFontColor = node.font ? node.font.color : '#333';
                    }}

                    if (matches) {{
                        // Highlight matching nodes
                        updates.push({{
                            id: node.id,
                            borderWidth: 3,
                            color: {{
                                background: node._originalColor.background || node._originalColor,
                                border: '#ef4444',
                                highlight: {{ background: '#fef2f2', border: '#ef4444' }}
                            }},
                            font: {{ color: '#ef4444', bold: true }}
                        }});
                    }} else {{
                        // Dim non-matching nodes
                        updates.push({{
                            id: node.id,
                            borderWidth: 1,
                            color: {{
                                background: '#d1d5db',
                                border: '#9ca3af'
                            }},
                            font: {{ color: '#9ca3af' }}
                        }});
                    }}
                }});

                if (updates.length > 0) {{
                    window.graphNodes.update(updates);
                }}
            }}
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

        # Persist to settings
        if hasattr(self.settings, "graph_filter_types"):
            self.settings.graph_filter_types = list(self._filter_types)
            self.settings.save()

        self._render_graph()

    def _on_layout_change(self, e: Any) -> None:
        """Handle layout change.

        Args:
            e: Change event with new value.
        """
        self._layout = e.value
        self._render_graph()


def mini_graph(
    world_db: WorldDatabase | None,
    settings: Settings | None = None,
    height: int = 200,
    filter_types: list[str] | None = None,
) -> None:
    """Create a small, non-interactive graph preview.

    Args:
        world_db: WorldDatabase to visualize.
        settings: Application settings for configuration values.
        height: Container height.
        filter_types: Entity types to show.
    """
    from src.settings import Settings

    if not world_db:
        ui.label("No world data").classes("text-gray-500 dark:text-gray-400 text-sm")
        return

    settings = settings or Settings.load()

    # Ensure vis-network library is loaded
    _ensure_vis_network_loaded()

    # Use unique container ID for mini graphs
    container_id = f"mini-graph-{uuid.uuid4().hex[:8]}"

    result = render_graph_html(
        world_db=world_db,
        settings=settings,
        filter_types=filter_types or ["character", "location"],
        layout="force-directed",
        height=height,
        container_id=container_id,
    )

    # Add HTML container (no script tags)
    ui.html(result.html, sanitize=False).classes(
        "w-full border dark:border-gray-700 rounded bg-white dark:bg-gray-800"
    ).style(f"height: {height}px;")

    # Run JavaScript initialization
    ui.run_javascript(result.js)

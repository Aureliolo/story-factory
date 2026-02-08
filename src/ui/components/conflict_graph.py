"""Conflict graph visualization component.

Displays entity relationships colored by conflict category (alliance, rivalry, tension).
"""

import json
import logging
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from nicegui import ui
from nicegui.element import Element

from src.memory.conflict_types import (
    CONFLICT_COLORS,
    ConflictCategory,
    ConflictGraphData,
)
from src.ui.local_prefs import load_prefs_deferred, save_pref

if TYPE_CHECKING:
    from src.memory.world_database import WorldDatabase
    from src.services import ServiceContainer

logger = logging.getLogger(__name__)

_PAGE_KEY = "conflict_graph"


def _ensure_vis_network_loaded() -> None:
    """Add vis-network library script to current page."""
    # vis-network version hardcoded here (not tracked by Dependabot); update if needed
    ui.add_body_html(
        '<script src="https://unpkg.com/vis-network@10.0.2/standalone/umd/vis-network.min.js"></script>'
    )


class ConflictGraphComponent:
    """Interactive conflict graph visualization.

    Features:
    - Edges colored by conflict category
    - Edge width reflects relationship strength
    - Dashed lines for tension relationships
    - Category filtering
    - Metrics panel
    """

    def __init__(
        self,
        world_db: WorldDatabase | None = None,
        services: ServiceContainer | None = None,
        on_node_select: Callable[[str], Any] | None = None,
        height: int = 500,
    ):
        """
        Initialize the ConflictGraphComponent and configure its initial state and callbacks.

        Parameters:
            world_db (WorldDatabase | None): Optional world database whose conflict data will be visualized; can be set later with set_world_db.
            services (ServiceContainer | None): Optional services container used to fetch conflict graph data (conflict_analysis).
            on_node_select (Callable[[str], Any] | None): Optional callback invoked with a node's id when a node is selected in the graph.
            height (int): Height of the graph container in pixels (default 500).

        Notes:
            - By default all conflict categories are enabled for filtering and both "character" and "faction" entity types are included.
            - Internal attributes initialized include UI container references, a unique selection callback id, and a placeholder for the latest ConflictGraphData.
        """
        self.world_db = world_db
        self.services = services
        self.on_node_select = on_node_select
        self.height = height

        self._container: Element | None = None
        self._no_data_label: Element | None = None
        self._metrics_container: ui.column | None = None
        self._filter_categories: list[ConflictCategory] = list(ConflictCategory)
        self._entity_types: list[str] = ["character", "faction"]
        self._callback_id = f"conflict_select_{uuid.uuid4().hex[:8]}"
        self._graph_data: ConflictGraphData | None = None

    def build(self) -> None:
        """
        Construct the conflict graph user interface, load the vis-network library, and wire interactions.

        This creates the category and entity filter controls, the graph display container, and the metrics panel; registers a node-selection handler if an on_node_select callback was provided; stores references to the graph and metrics containers on the instance; and performs the initial graph render.
        """
        logger.debug("Building ConflictGraphComponent")
        _ensure_vis_network_loaded()

        # Register event handler for node selection
        if self.on_node_select:

            def handle_node_select(e: Any) -> None:
                """
                Process a node-selection event and invoke the registered node selection callback if a node id is provided.

                Parameters:
                    e (Any): Event object expected to have an `args` mapping that may contain a `"node_id"` key; when present and an `on_node_select` callback is set, the callback is called with that node id.
                """
                node_id = e.args.get("node_id") if e.args else None
                if node_id and self.on_node_select:
                    self.on_node_select(node_id)

            ui.on(self._callback_id, handle_node_select)

        # Widget references for deferred preference loading
        self._category_checkboxes: dict[str, ui.checkbox] = {}
        self._entity_type_checkboxes: dict[str, ui.checkbox] = {}

        with ui.row().classes("w-full gap-4"):
            # Main graph area
            with ui.column().classes("flex-grow gap-4"):
                # Controls
                with ui.row().classes("w-full items-center gap-4 flex-wrap"):
                    ui.label("Categories:").classes("text-sm font-medium")

                    # Category filters with color indicators
                    for category in ConflictCategory:
                        color = CONFLICT_COLORS[category.value]
                        with ui.row().classes("items-center gap-1"):
                            ui.html(
                                f'<div style="width: 12px; height: 12px; background: {color}; border-radius: 2px;"></div>',
                                sanitize=False,
                            )
                            self._category_checkboxes[category.value] = ui.checkbox(
                                category.value.title(),
                                value=category in self._filter_categories,
                                on_change=lambda e, c=category: self._toggle_category(c, e.value),
                            )

                    ui.space()

                    ui.label("Entities:").classes("text-sm font-medium")
                    self._entity_type_checkboxes["character"] = ui.checkbox(
                        "Characters",
                        value="character" in self._entity_types,
                        on_change=lambda e: self._toggle_entity_type("character", e.value),
                    )
                    self._entity_type_checkboxes["faction"] = ui.checkbox(
                        "Factions",
                        value="faction" in self._entity_types,
                        on_change=lambda e: self._toggle_entity_type("faction", e.value),
                    )

                    ui.space()

                    ui.button(
                        icon="refresh",
                        on_click=self.refresh,
                    ).props("flat round size=sm").tooltip("Refresh graph")

                # Graph container — plain div instead of ui.html() because NiceGUI's
                # Html Vue component calls renderContent() on every re-render,
                # which resets innerHTML and destroys the vis.js canvas.
                self._no_data_label = ui.label(
                    "No world data to display. Create or load a project first."
                ).classes("text-gray-400 text-center w-full py-16")
                self._container = (
                    ui.element("div")
                    .classes("w-full border border-gray-700 rounded-lg")
                    .style(f"height: {self.height}px; background: #1f2937;")
                )

            # Metrics panel
            with ui.card().classes("w-64 p-4"):
                ui.label("Conflict Metrics").classes("text-lg font-semibold mb-4")
                self._metrics_container = ui.column().classes("w-full gap-2")

        # Initial render
        self._render_graph()

        # Restore persisted preferences from localStorage
        load_prefs_deferred(_PAGE_KEY, self._apply_prefs)

    def set_world_db(self, world_db: WorldDatabase | None) -> None:
        """
        Set the world database used by the component and re-render the graph.

        Parameters:
            world_db (WorldDatabase | None): The world database to visualize, or `None` to clear the current data.
        """
        self.world_db = world_db
        self._render_graph()

    def refresh(self) -> None:
        """Refresh the graph visualization."""
        self._render_graph()

    def _render_graph(self) -> None:
        """
        Render the current conflict graph into the component's container.

        Fetches conflict graph data from the configured services and world database, converts it to the vis-network format, injects the HTML/JavaScript needed to initialize the vis-network visualization, and updates the metrics panel. If world data or services are missing, or if no nodes match the active filters, replaces the graph area with a user-facing message and updates metrics accordingly. When the network is created, node click events are emitted to the component's selection callback via its internal event ID.
        """
        if not self._container:
            return

        if not self.world_db or not self.services:
            if self._no_data_label:
                self._no_data_label.set_visibility(True)
            self._container.set_visibility(False)
            self._update_metrics(None)
            return

        # Get conflict graph data from service
        # Pass filters directly - None means no filter (include all), [] means include none
        # Empty list is a valid filter that includes nothing
        logger.debug(
            f"Fetching conflict graph data: categories={len(self._filter_categories)}, "
            f"entity_types={self._entity_types}"
        )
        self._graph_data = self.services.conflict_analysis.get_conflict_graph_data(
            self.world_db,
            categories=self._filter_categories,
            entity_types=self._entity_types if self._entity_types else None,
        )

        # Show graph container, hide "no data" label
        if self._no_data_label:
            self._no_data_label.set_visibility(False)
        self._container.set_visibility(True)

        if not self._graph_data.nodes:
            # Clear any previous vis.js canvas and show empty message via JS
            container_id = f"c{self._container.id}"
            ui.run_javascript(
                f"var el = document.getElementById('{container_id}');"
                f"if (el) el.innerHTML = '<div style=\"height:100%;display:flex;"
                f'align-items:center;justify-content:center;color:#9ca3af;">'
                f"<p>No entities match the current filters.</p></div>';"
            )
            self._update_metrics(self._graph_data.metrics)
            return

        # Convert to vis.js format
        nodes = [
            {
                "id": n.id,
                "label": n.label,
                "color": n.color,
                "size": n.size,
                "title": n.title,
            }
            for n in self._graph_data.nodes
        ]

        edges = [
            {
                "from": e.from_id,
                "to": e.to_id,
                "color": {"color": e.color, "highlight": e.color},
                "width": e.width,
                "title": e.title,
                "dashes": e.dashes,
                "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}},
            }
            for e in self._graph_data.edges
        ]

        # Use the plain div's auto-generated NiceGUI ID as the vis.js container
        container_id = f"c{self._container.id}"

        # JavaScript initialization
        js = f"""
        (function() {{
            var attempts = 0;
            var maxAttempts = 50;

            function initConflictGraph() {{
                attempts++;
                if (typeof vis === 'undefined') {{
                    if (attempts >= maxAttempts) {{
                        console.error('vis-network failed to load');
                        return;
                    }}
                    setTimeout(initConflictGraph, 100);
                    return;
                }}

                // Destroy previous network to prevent stale canvas on re-render
                if (window.conflictNetwork) {{
                    window.conflictNetwork.destroy();
                    window.conflictNetwork = null;
                }}

                var container = document.getElementById('{container_id}');
                if (!container) return;

                var isDarkMode = document.body.classList.contains('dark') ||
                                 document.documentElement.classList.contains('dark');
                var fontColor = isDarkMode ? '#e5e7eb' : '#374151';
                var bgColor = isDarkMode ? '#1f2937' : '#ffffff';

                container.style.backgroundColor = bgColor;

                var nodes = new vis.DataSet({json.dumps(nodes)});
                var edges = new vis.DataSet({json.dumps(edges)});

                var data = {{ nodes: nodes, edges: edges }};

                var options = {{
                    nodes: {{
                        font: {{ size: 14, color: fontColor }},
                        borderWidth: 2,
                        shape: 'dot'
                    }},
                    edges: {{
                        smooth: {{ type: 'continuous' }},
                        font: {{ size: 10, color: fontColor }}
                    }},
                    interaction: {{
                        hover: true,
                        tooltipDelay: 200,
                        dragNodes: true,
                        dragView: true,
                        zoomView: true
                    }},
                    physics: {{
                        enabled: true,
                        solver: 'forceAtlas2Based',
                        forceAtlas2Based: {{
                            gravitationalConstant: -50,
                            centralGravity: 0.01,
                            springLength: 150,
                            springConstant: 0.08,
                            damping: 0.4
                        }},
                        stabilization: {{
                            iterations: 100
                        }}
                    }}
                }};

                var network = new vis.Network(container, data, options);

                // Handle node selection
                network.on('click', function(params) {{
                    if (params.nodes.length > 0) {{
                        var nodeId = params.nodes[0];
                        if ('{self._callback_id}') {{
                            emitEvent('{self._callback_id}', {{node_id: nodeId}});
                        }}
                    }}
                }});

                // Store reference
                window.conflictNetwork = network;
            }}

            initConflictGraph();
        }})();
        """

        ui.run_javascript(js)
        logger.debug(
            f"Conflict graph rendered: {len(self._graph_data.nodes)} nodes, "
            f"{len(self._graph_data.edges)} edges"
        )

        # Update metrics panel
        self._update_metrics(self._graph_data.metrics)

    def _update_metrics(self, metrics: Any) -> None:
        """
        Update the metrics panel to reflect provided conflict metrics.

        Renders relationship counts by category, an overall conflict density indicator, up to three highest-tension pairs, and up to three faction cluster summaries. If no metrics are provided or the metrics container is unavailable, the panel is cleared and a "No data available" message is shown.

        Parameters:
            metrics (ConflictMetrics | None): Metrics object containing the fields
                `alliance_count`, `rivalry_count`, `tension_count`, `neutral_count`,
                `conflict_density` (0.0-1.0), `highest_tension_pairs` (iterable of
                objects with `entity_a_name` and `entity_b_name`), and
                `faction_clusters` (iterable of clusters with `entity_ids`). Pass
                None to clear the panel and display the empty-state message.
        """
        if not self._metrics_container:
            return

        self._metrics_container.clear()

        if not metrics:
            with self._metrics_container:
                ui.label("No data available").classes("text-gray-400 text-sm")
            return

        with self._metrics_container:
            # Relationship counts by category
            ui.label("Relationships").classes("font-medium text-sm")
            with ui.row().classes("w-full justify-between"):
                with ui.row().classes("items-center gap-1"):
                    ui.html(
                        f'<div style="width: 8px; height: 8px; background: {CONFLICT_COLORS["alliance"]}; border-radius: 50%;"></div>',
                        sanitize=False,
                    )
                    ui.label("Alliance").classes("text-xs")
                ui.label(str(metrics.alliance_count)).classes("text-sm font-medium")

            with ui.row().classes("w-full justify-between"):
                with ui.row().classes("items-center gap-1"):
                    ui.html(
                        f'<div style="width: 8px; height: 8px; background: {CONFLICT_COLORS["rivalry"]}; border-radius: 50%;"></div>',
                        sanitize=False,
                    )
                    ui.label("Rivalry").classes("text-xs")
                ui.label(str(metrics.rivalry_count)).classes("text-sm font-medium")

            with ui.row().classes("w-full justify-between"):
                with ui.row().classes("items-center gap-1"):
                    ui.html(
                        f'<div style="width: 8px; height: 8px; background: {CONFLICT_COLORS["tension"]}; border-radius: 50%;"></div>',
                        sanitize=False,
                    )
                    ui.label("Tension").classes("text-xs")
                ui.label(str(metrics.tension_count)).classes("text-sm font-medium")

            with ui.row().classes("w-full justify-between"):
                with ui.row().classes("items-center gap-1"):
                    ui.html(
                        f'<div style="width: 8px; height: 8px; background: {CONFLICT_COLORS["neutral"]}; border-radius: 50%;"></div>',
                        sanitize=False,
                    )
                    ui.label("Neutral").classes("text-xs")
                ui.label(str(metrics.neutral_count)).classes("text-sm font-medium")

            ui.separator().classes("my-2")

            # Conflict density
            ui.label("Conflict Density").classes("font-medium text-sm")
            density_pct = int(metrics.conflict_density * 100)
            with ui.row().classes("w-full items-center gap-2"):
                ui.linear_progress(value=metrics.conflict_density).classes("flex-grow")
                ui.label(f"{density_pct}%").classes("text-sm")

            ui.separator().classes("my-2")

            # Highest tension pairs
            if metrics.highest_tension_pairs:
                ui.label("Top Tensions").classes("font-medium text-sm")
                for pair in metrics.highest_tension_pairs[:3]:
                    with ui.row().classes("w-full items-center gap-1 text-xs"):
                        ui.label(pair.entity_a_name).classes("truncate flex-grow")
                        ui.icon("flash_on", size="xs", color="red")
                        ui.label(pair.entity_b_name).classes("truncate flex-grow")

            # Faction clusters
            if metrics.faction_clusters:
                ui.separator().classes("my-2")
                ui.label("Factions").classes("font-medium text-sm")
                for cluster in metrics.faction_clusters[:3]:
                    ui.label(f"{len(cluster.entity_ids)} members").classes("text-xs text-gray-500")

    def _toggle_category(self, category: ConflictCategory, enabled: bool) -> None:
        """
        Enable or disable a conflict category in the active filters and refresh the visualization.

        Parameters:
            category (ConflictCategory): The conflict category to enable or disable.
            enabled (bool): If True, add the category to active filters; if False, remove it.
        """
        logger.debug(f"Toggle category {category.value}: enabled={enabled}")
        if enabled and category not in self._filter_categories:
            self._filter_categories.append(category)
        elif not enabled and category in self._filter_categories:
            self._filter_categories.remove(category)
        save_pref(_PAGE_KEY, "filter_categories", [c.value for c in self._filter_categories])
        self._render_graph()

    def _toggle_entity_type(self, entity_type: str, enabled: bool) -> None:
        """
        Enable or disable filtering for the specified entity type and update the rendered graph.

        Parameters:
            entity_type (str): The entity type to toggle (e.g., "character" or "faction").
            enabled (bool): True to include this entity type in the filters, False to exclude it.

        """
        logger.debug(f"Toggle entity type {entity_type}: enabled={enabled}")
        if enabled and entity_type not in self._entity_types:
            self._entity_types.append(entity_type)
        elif not enabled and entity_type in self._entity_types:
            self._entity_types.remove(entity_type)
        save_pref(_PAGE_KEY, "entity_types", self._entity_types)
        self._render_graph()

    def _apply_prefs(self, prefs: dict) -> None:
        """Apply loaded preferences to conflict graph state and UI widgets.

        Args:
            prefs: Dict of field→value from localStorage.
        """
        if not prefs:
            return

        changed = False

        if "filter_categories" in prefs and isinstance(prefs["filter_categories"], list):
            valid_values = {c.value for c in ConflictCategory}
            loaded_cats = [
                ConflictCategory(v) for v in prefs["filter_categories"] if v in valid_values
            ]
            if loaded_cats != self._filter_categories:
                self._filter_categories = loaded_cats
                changed = True
                for cat_value, cb in self._category_checkboxes.items():
                    cb.value = any(c.value == cat_value for c in self._filter_categories)

        if "entity_types" in prefs and isinstance(prefs["entity_types"], list):
            valid_types = {"character", "faction"}
            loaded_types: list[str] = [t for t in prefs["entity_types"] if t in valid_types]
            if loaded_types != self._entity_types:
                self._entity_types = loaded_types
                changed = True
                for etype, cb in self._entity_type_checkboxes.items():
                    cb.value = etype in self._entity_types

        if changed:
            logger.info("Restored conflict graph preferences from localStorage")
            self._render_graph()

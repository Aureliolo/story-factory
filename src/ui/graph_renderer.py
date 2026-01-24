"""Graph rendering utilities for vis.js visualization."""

import html
import json
import logging
from dataclasses import dataclass
from typing import Any

from src.memory.world_database import WorldDatabase
from src.settings import Settings
from src.ui.theme import ENTITY_COLORS, RELATION_COLORS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphRenderResult:
    """Result of rendering a graph for vis.js.

    Attributes:
        html: HTML string for the graph container (no script tags).
        js: JavaScript string for graph initialization.
    """

    html: str
    js: str


# Entity type shapes
ENTITY_SHAPES = {
    "character": "dot",
    "location": "square",
    "item": "diamond",
    "faction": "triangle",
    "concept": "star",
}


def render_graph_html(
    world_db: WorldDatabase,
    settings: Settings,
    filter_types: list[str] | None = None,
    layout: str = "force-directed",
    height: int = 500,
    selected_entity_id: str | None = None,
    container_id: str = "graph-container",
    create_rel_callback_id: str = "",
    edge_context_callback_id: str = "",
) -> GraphRenderResult:
    """Generate vis.js graph HTML and JavaScript separately.

    Args:
        world_db: WorldDatabase instance
        settings: Application settings for configuration values
        filter_types: List of entity types to show (None = all)
        layout: Layout algorithm ("force-directed", "hierarchical", "circular", "grid")
        height: Graph container height in pixels
        selected_entity_id: ID of selected entity to highlight
        container_id: Unique ID for the graph container
        create_rel_callback_id: Callback ID for creating relationships via drag
        edge_context_callback_id: Callback ID for edge context menu

    Returns:
        GraphRenderResult with HTML and JavaScript strings.
    """
    max_words = settings.mini_description_words_max
    logger.debug(f"Rendering graph with max_words={max_words}, filter_types={filter_types}")
    graph = world_db.get_graph()

    # Filter nodes by type
    if filter_types:
        filter_types_lower = [t.lower() for t in filter_types]
    else:
        filter_types_lower = None

    nodes = []
    for node_id, data in graph.nodes(data=True):
        node_type = data.get("type", "concept")
        if filter_types_lower and node_type not in filter_types_lower:
            continue

        node = {
            "id": node_id,
            "label": data.get("name", "Unknown"),
            "group": node_type,
            "color": ENTITY_COLORS.get(node_type, "#607D8B"),
            "shape": ENTITY_SHAPES.get(node_type, "dot"),
            "title": _build_tooltip(data, max_words),
        }

        # Highlight selected node
        if node_id == selected_entity_id:
            node["borderWidth"] = 3
            node["borderWidthSelected"] = 4
            node["color"] = {
                "background": ENTITY_COLORS.get(node_type, "#607D8B"),
                "border": "#FFD700",  # Gold border for selected
            }

        nodes.append(node)

    # Apply positions for circular/grid layouts
    if layout == "circular" and nodes:
        positions = _calculate_circular_positions(len(nodes))
        for i, node in enumerate(nodes):
            node["x"] = positions[i][0]
            node["y"] = positions[i][1]
    elif layout == "grid" and nodes:
        positions = _calculate_grid_positions(len(nodes))
        for i, node in enumerate(nodes):
            node["x"] = positions[i][0]
            node["y"] = positions[i][1]

    # Get node IDs that made it through the filter
    visible_node_ids = {n["id"] for n in nodes}

    edges = []
    for u, v, data in graph.edges(data=True):
        # Only include edges between visible nodes
        if u not in visible_node_ids or v not in visible_node_ids:
            continue

        rel_type = data.get("relation_type", "related")
        description = data.get("description", "")
        # Use plain text tooltip for reliable display
        if description:
            # Truncate to configured word limit
            words = description.split()
            if len(words) > max_words:
                short_desc = " ".join(words[:max_words]) + "..."
            else:
                short_desc = description
            tooltip = f"{rel_type}: {short_desc}"
        else:
            tooltip = rel_type
        edge = {
            "from": u,
            "to": v,
            "color": RELATION_COLORS.get(rel_type, "#6b7280"),
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}},
            "title": tooltip,
            "width": 2,
        }
        edges.append(edge)

    # Build layout options
    layout_options = _get_layout_options(layout)

    # HTML container only (no scripts)
    html = f"""
    <style>
        #{container_id} {{
            border: 1px solid #e5e7eb;
            border-radius: 4px;
            background: #ffffff;
        }}
        .dark #{container_id} {{
            border-color: #374151;
            background: #1f2937;
        }}
    </style>
    <div id="{container_id}" style="height: {height}px;"></div>
    """

    # JavaScript initialization (to be run via ui.run_javascript)
    js = f"""
    (function() {{
        // Wait for vis-network to be available with timeout
        var attempts = 0;
        var maxAttempts = 50; // 5 seconds max (50 * 100ms)

        function initGraph() {{
            attempts++;
            if (typeof vis === 'undefined') {{
                if (attempts >= maxAttempts) {{
                    console.error('vis-network failed to load after 5 seconds');
                    var container = document.getElementById('{container_id}');
                    if (container) {{
                        container.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #ef4444;">' +
                            '<p>Graph library failed to load. Please check your network connection and refresh.</p></div>';
                    }}
                    return;
                }}
                setTimeout(initGraph, 100);
                return;
            }}

            // Robust dark mode detection for NiceGUI
            var isDarkMode = (function() {{
                // Check body class (NiceGUI's primary method)
                if (document.body.classList.contains('dark')) return true;
                // Check html element
                if (document.documentElement.classList.contains('dark')) return true;
                // Check for NiceGUI's dark theme attribute
                if (document.body.getAttribute('data-theme') === 'dark') return true;
                // Check computed background color (fallback)
                var bgColor = getComputedStyle(document.body).backgroundColor;
                if (bgColor) {{
                    var rgb = bgColor.match(/\\d+/g);
                    if (rgb && rgb.length >= 3) {{
                        var luminance = (parseInt(rgb[0]) * 0.299 + parseInt(rgb[1]) * 0.587 + parseInt(rgb[2]) * 0.114);
                        if (luminance < 128) return true;
                    }}
                }}
                // Check system preference as last resort
                return window.matchMedia('(prefers-color-scheme: dark)').matches;
            }})();
            var fontColor = isDarkMode ? '#e5e7eb' : '#374151';
            var bgColor = isDarkMode ? '#1f2937' : '#ffffff';

            var nodes = new vis.DataSet({json.dumps(nodes)});
            var edges = new vis.DataSet({json.dumps(edges)});

            var container = document.getElementById('{container_id}');
            if (!container) return;

            // Set background color based on dark mode
            container.style.backgroundColor = bgColor;

            var data = {{ nodes: nodes, edges: edges }};

            var options = {{
                nodes: {{
                    font: {{ size: 14, color: fontColor }},
                    scaling: {{ min: 10, max: 30 }},
                    borderWidth: 2
                }},
                edges: {{
                    smooth: false,
                    width: 2,
                    color: {{ color: '#6b7280', highlight: '#3b82f6' }},
                    arrows: {{ to: {{ enabled: true, scaleFactor: 0.5 }} }}
                }},
                interaction: {{
                    hover: true,
                    tooltipDelay: 200,
                    navigationButtons: false,
                    keyboard: true,
                    dragNodes: true,
                    dragView: true,
                    zoomView: true,
                    multiselect: false,
                    selectionBox: false
                }},
                {layout_options}
            }};

            var network = new vis.Network(container, data, options);

            // Handle node click - update hidden input for NiceGUI
            network.on('click', function(params) {{
                if (params.nodes.length > 0) {{
                    var selectedId = params.nodes[0];
                    // Try to find and update the entity selection input
                    var inputs = document.querySelectorAll('input[data-testid="textbox"]');
                    inputs.forEach(function(input) {{
                        if (input.closest('[id*="selected_entity"]')) {{
                            input.value = selectedId;
                            input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        }}
                    }});
                    // Also store in window for other handlers
                    window.selectedEntityId = selectedId;
                }}
            }});

            // Handle double-click to focus on node
            network.on('doubleClick', function(params) {{
                if (params.nodes.length > 0) {{
                    network.focus(params.nodes[0], {{
                        scale: 1.5,
                        animation: {{ duration: 500, easingFunction: 'easeInOutQuad' }}
                    }});
                }}
            }});

            // Smooth dragging: pause physics during drag, resume after
            network.on('dragStart', function(params) {{
                if (params.nodes.length > 0) {{
                    // Pause physics while dragging for precise control
                    network.setOptions({{ physics: {{ enabled: false }} }});
                }}
            }});

            network.on('dragEnd', function(params) {{
                if (params.nodes.length > 0) {{
                    // Re-enable physics with gentle settling
                    network.setOptions({{
                        physics: {{
                            enabled: true,
                            stabilization: {{ enabled: false }},
                            barnesHut: {{
                                damping: 0.9,
                                springConstant: 0.02
                            }}
                        }}
                    }});
                    // After brief settling, restore normal physics
                    setTimeout(function() {{
                        network.setOptions({{
                            physics: {{
                                barnesHut: {{
                                    damping: 0.5,
                                    springConstant: 0.04
                                }}
                            }}
                        }});
                    }}, 500);
                }}
            }});

            // Store network reference for external control
            window.graphNetwork = network;
            window.graphNodes = nodes;
            window.graphEdges = edges;

            // Drag-to-connect relationship creation
            var dragStartNode = null;
            var tempEdge = null;

            // Enable manipulation for adding edges via drag
            network.on('dragStart', function(params) {{
                if (params.nodes.length > 0 && params.event && params.event.shiftKey) {{
                    dragStartNode = params.nodes[0];
                    // Show visual feedback
                    var nodeData = nodes.get(dragStartNode);
                    if (nodeData) {{
                        // Store original color so it can be restored on dragEnd
                        if (typeof nodeData._originalColor === 'undefined') {{
                            nodeData._originalColor = nodeData.color;
                        }}
                        nodes.update({{
                            id: dragStartNode,
                            borderWidth: 4,
                            color: {{
                                border: '#3b82f6',
                                background: nodeData.color && nodeData.color.background ? nodeData.color.background : nodeData.color
                            }},
                            _originalColor: nodeData._originalColor
                        }});
                    }}
                }}
            }});

            network.on('dragging', function(params) {{
                if (dragStartNode && params.event && params.event.shiftKey) {{
                    // Visual feedback for drag-to-connect mode
                    var pointer = network.getPointer(params.event);
                    var nearestNode = network.getNodeAt(pointer);

                    // Remove temporary edge if exists
                    if (tempEdge) {{
                        try {{ edges.remove(tempEdge); }} catch(e) {{}}
                        tempEdge = null;
                    }}

                    // If hovering over another node, show preview edge
                    if (nearestNode && nearestNode !== dragStartNode) {{
                        tempEdge = 'temp_edge_' + Date.now();
                        edges.add({{
                            id: tempEdge,
                            from: dragStartNode,
                            to: nearestNode,
                            dashes: [5, 5],
                            color: {{ color: '#3b82f6', opacity: 0.5 }},
                            arrows: {{ to: {{ enabled: true, scaleFactor: 0.5 }} }},
                            width: 2
                        }});
                    }}
                }}
            }});

            network.on('dragEnd', function(params) {{
                if (dragStartNode && params.event && params.event.shiftKey) {{
                    var pointer = network.getPointer(params.event);
                    var targetNode = network.getNodeAt(pointer);

                    // Remove temporary edge
                    if (tempEdge) {{
                        try {{ edges.remove(tempEdge); }} catch(e) {{}}
                        tempEdge = null;
                    }}

                    // Reset source node styling
                    var nodeData = nodes.get(dragStartNode);
                    if (nodeData && nodeData._originalColor) {{
                        nodes.update({{
                            id: dragStartNode,
                            borderWidth: 2,
                            color: nodeData._originalColor
                        }});
                    }}

                    // Create relationship if dropped on valid target
                    if (targetNode && targetNode !== dragStartNode) {{
                        // Emit event to create relationship
                        if ('{create_rel_callback_id}') {{
                            emitEvent('{create_rel_callback_id}', {{
                                source_id: dragStartNode,
                                target_id: targetNode
                            }});
                        }}
                    }}

                    dragStartNode = null;
                }}
            }});

            // Right-click on edge for context menu
            network.on('oncontext', function(params) {{
                params.event.preventDefault();

                if (params.edges.length > 0) {{
                    var edgeId = params.edges[0];
                    // Emit event for edge context menu
                    if ('{edge_context_callback_id}') {{
                        emitEvent('{edge_context_callback_id}', {{
                            edge_id: edgeId,
                            x: params.event.pageX,
                            y: params.event.pageY
                        }});
                    }}
                }}
            }});
        }}

        initGraph();
    }})();
    """

    return GraphRenderResult(html=html, js=js)


def _build_tooltip(data: dict[str, Any], max_words: int) -> str:
    """Build plain text tooltip for a node.

    Uses mini_description if available, otherwise truncates description.
    Returns plain text (no HTML) for reliable vis.js display.

    Args:
        data: Node data dictionary.
        max_words: Maximum words for description truncation.

    Returns:
        Plain text tooltip string.
    """
    name = data.get("name", "Unknown")
    entity_type = data.get("type", "unknown").title()

    # Check for mini description (generated summary for hover)
    attributes = data.get("attributes", {})
    mini_desc = attributes.get("mini_description", "")

    if mini_desc:
        # Use the short mini description
        return f"{name} ({entity_type})\n{mini_desc}"

    description = data.get("description", "")
    if description:
        # Truncate to configured word limit
        words = description.split()
        if len(words) > max_words:
            short_desc = " ".join(words[:max_words]) + "..."
        else:
            short_desc = description
        return f"{name} ({entity_type})\n{short_desc}"

    return f"{name} ({entity_type})"


def _get_layout_options(layout: str) -> str:
    """Get vis.js layout options string."""
    if layout == "hierarchical":
        return """
            layout: {
                hierarchical: {
                    direction: 'UD',
                    sortMethod: 'hubsize',
                    levelSeparation: 100,
                    nodeSpacing: 150
                }
            }
        """
    elif layout == "circular":
        # Circular layout - disable physics to keep positions fixed
        return """
            layout: {
                improvedLayout: false,
                hierarchical: false
            },
            physics: {
                enabled: false
            }
        """
    elif layout == "grid":
        # Grid layout - disable physics to keep positions fixed
        return """
            layout: {
                improvedLayout: false,
                hierarchical: false
            },
            physics: {
                enabled: false
            }
        """
    else:  # force-directed (default)
        return """
            layout: {
                improvedLayout: true,
                hierarchical: false
            },
            physics: {
                enabled: true,
                solver: 'barnesHut',
                barnesHut: {
                    gravitationalConstant: -2000,
                    centralGravity: 0.3,
                    springLength: 150,
                    springConstant: 0.04,
                    damping: 0.3,
                    avoidOverlap: 0.5
                },
                stabilization: {
                    iterations: 100,
                    fit: true
                },
                minVelocity: 0.75
            }
        """


def _calculate_circular_positions(node_count: int, radius: int = 300) -> list[tuple[int, int]]:
    """Calculate positions for circular layout.

    Args:
        node_count: Number of nodes.
        radius: Circle radius in pixels.

    Returns:
        List of (x, y) positions.
    """
    import math

    positions = []
    for i in range(node_count):
        angle = (2 * math.pi * i) / node_count - math.pi / 2  # Start from top
        x = int(radius * math.cos(angle))
        y = int(radius * math.sin(angle))
        positions.append((x, y))
    return positions


def _calculate_grid_positions(node_count: int, spacing: int = 150) -> list[tuple[int, int]]:
    """Calculate positions for grid layout.

    Args:
        node_count: Number of nodes.
        spacing: Space between nodes in pixels.

    Returns:
        List of (x, y) positions.
    """
    import math

    cols = max(1, math.ceil(math.sqrt(node_count)))
    positions = []
    for i in range(node_count):
        row = i // cols
        col = i % cols
        x = col * spacing - (cols * spacing) // 2
        y = row * spacing - ((node_count // cols) * spacing) // 2
        positions.append((x, y))
    return positions


def render_entity_summary_html(world_db: WorldDatabase) -> str:
    """Render a summary of entities for the Fundamentals tab.

    Args:
        world_db: WorldDatabase instance

    Returns:
        HTML string with entity summary cards
    """
    counts = {
        "character": world_db.count_entities("character"),
        "location": world_db.count_entities("location"),
        "item": world_db.count_entities("item"),
        "faction": world_db.count_entities("faction"),
        "concept": world_db.count_entities("concept"),
    }

    total = sum(counts.values())
    rel_count = len(world_db.list_relationships())

    # Use CSS class-based dark mode for proper theming
    return f"""
    <style>
        .entity-summary-label {{ color: #6b7280; }}
        .dark .entity-summary-label {{ color: #9ca3af; }}
        .entity-summary-total {{ color: #9ca3af; }}
        .dark .entity-summary-total {{ color: #6b7280; }}
    </style>
    <div style="display: flex; gap: 10px; flex-wrap: wrap;">
        <div style="background: {ENTITY_COLORS["character"]}22; border-left: 4px solid {ENTITY_COLORS["character"]}; padding: 10px; border-radius: 4px; min-width: 100px;">
            <div style="font-size: 24px; font-weight: bold; color: {ENTITY_COLORS["character"]};">{counts["character"]}</div>
            <div class="entity-summary-label" style="font-size: 12px;">Characters</div>
        </div>
        <div style="background: {ENTITY_COLORS["location"]}22; border-left: 4px solid {ENTITY_COLORS["location"]}; padding: 10px; border-radius: 4px; min-width: 100px;">
            <div style="font-size: 24px; font-weight: bold; color: {ENTITY_COLORS["location"]};">{counts["location"]}</div>
            <div class="entity-summary-label" style="font-size: 12px;">Locations</div>
        </div>
        <div style="background: {ENTITY_COLORS["item"]}22; border-left: 4px solid {ENTITY_COLORS["item"]}; padding: 10px; border-radius: 4px; min-width: 100px;">
            <div style="font-size: 24px; font-weight: bold; color: {ENTITY_COLORS["item"]};">{counts["item"]}</div>
            <div class="entity-summary-label" style="font-size: 12px;">Items</div>
        </div>
        <div style="background: {ENTITY_COLORS["faction"]}22; border-left: 4px solid {ENTITY_COLORS["faction"]}; padding: 10px; border-radius: 4px; min-width: 100px;">
            <div style="font-size: 24px; font-weight: bold; color: {ENTITY_COLORS["faction"]};">{counts["faction"]}</div>
            <div class="entity-summary-label" style="font-size: 12px;">Factions</div>
        </div>
        <div style="background: #60606022; border-left: 4px solid #6b7280; padding: 10px; border-radius: 4px; min-width: 100px;">
            <div style="font-size: 24px; font-weight: bold; color: #6b7280;">{rel_count}</div>
            <div class="entity-summary-label" style="font-size: 12px;">Relationships</div>
        </div>
    </div>
    <div class="entity-summary-total" style="margin-top: 10px; font-size: 12px;">
        Total: {total} entities
    </div>
    """


def render_path_result(world_db: WorldDatabase, path: list[str]) -> str:
    """Render a path between entities as HTML.

    Args:
        world_db: WorldDatabase instance
        path: List of entity IDs forming the path

    Returns:
        HTML string showing the path
    """
    if not path:
        return """
        <style>
            .path-no-result { color: #6b7280; }
            .dark .path-no-result { color: #9ca3af; }
        </style>
        <p class='path-no-result'>No path found between these entities.</p>
        """

    parts = []
    for i, entity_id in enumerate(path):
        entity = world_db.get_entity(entity_id)
        if entity:
            color = ENTITY_COLORS.get(entity.type, "#607D8B")
            escaped_name = html.escape(entity.name)
            parts.append(
                f"<span style='background: {color}22; color: {color}; padding: 4px 8px; border-radius: 4px; font-weight: bold;'>{escaped_name}</span>"
            )

            # Add arrow if not last
            if i < len(path) - 1:
                # Get relationship between this and next
                next_id = path[i + 1]
                rel = world_db.get_relationship_between(entity_id, next_id)
                if rel:
                    escaped_rel = html.escape(rel.relation_type)
                    parts.append(f"<span class='path-arrow'>--{escaped_rel}--></span>")
                else:
                    parts.append("<span class='path-arrow'>---></span>")

    return f"""
    <style>
        .path-arrow {{ color: #6b7280; margin: 0 8px; }}
        .dark .path-arrow {{ color: #9ca3af; }}
    </style>
    <div style='display: flex; align-items: center; flex-wrap: wrap; gap: 4px;'>{"".join(parts)}</div>
    """


def render_centrality_result(world_db: WorldDatabase, limit: int = 10) -> str:
    """Render most connected entities.

    Args:
        world_db: WorldDatabase instance
        limit: Number of entities to show

    Returns:
        HTML string with centrality ranking
    """
    most_connected = world_db.get_most_connected(limit)

    if not most_connected:
        return """
        <style>
            .centrality-no-result { color: #6b7280; }
            .dark .centrality-no-result { color: #9ca3af; }
        </style>
        <p class='centrality-no-result'>No entities found.</p>
        """

    rows = []
    for i, (entity, degree) in enumerate(most_connected, 1):
        color = ENTITY_COLORS.get(entity.type, "#607D8B")
        escaped_type = html.escape(entity.type)
        escaped_name = html.escape(entity.name)
        rows.append(
            f"""
            <tr class='centrality-row'>
                <td style='padding: 8px; text-align: center;'>{i}</td>
                <td style='padding: 8px;'>
                    <span style='background: {color}22; color: {color}; padding: 2px 6px; border-radius: 3px;'>{escaped_type}</span>
                </td>
                <td style='padding: 8px; font-weight: bold;'>{escaped_name}</td>
                <td style='padding: 8px; text-align: center;'>{degree}</td>
            </tr>
        """
        )

    return f"""
    <style>
        .centrality-header {{ background: #f3f4f6; }}
        .dark .centrality-header {{ background: #374151; }}
        .centrality-row {{ color: #374151; }}
        .dark .centrality-row {{ color: #e5e7eb; }}
    </style>
    <table style='width: 100%; border-collapse: collapse;'>
        <thead>
            <tr class='centrality-header'>
                <th style='padding: 8px; text-align: center;'>#</th>
                <th style='padding: 8px;'>Type</th>
                <th style='padding: 8px;'>Name</th>
                <th style='padding: 8px; text-align: center;'>Connections</th>
            </tr>
        </thead>
        <tbody>
            {"".join(rows)}
        </tbody>
    </table>
    """


def render_communities_result(world_db: WorldDatabase) -> str:
    """Render community/cluster analysis.

    Args:
        world_db: WorldDatabase instance

    Returns:
        HTML string with community information
    """
    communities = world_db.get_communities()

    if not communities:
        return """
        <style>
            .communities-no-result { color: #6b7280; }
            .dark .communities-no-result { color: #9ca3af; }
        </style>
        <p class='communities-no-result'>No communities found.</p>
        """

    parts = []
    for i, community in enumerate(communities, 1):
        members = []
        for entity_id in community[:10]:  # Limit to 10 per community
            entity = world_db.get_entity(entity_id)
            if entity:
                color = ENTITY_COLORS.get(entity.type, "#607D8B")
                escaped_name = html.escape(entity.name)
                members.append(
                    f"<span style='background: {color}22; color: {color}; padding: 2px 6px; border-radius: 3px; margin: 2px;'>{escaped_name}</span>"
                )

        overflow = ""
        if len(community) > 10:
            overflow = f"<span class='community-overflow'>+{len(community) - 10} more</span>"

        parts.append(
            f"""
            <div class='community-card'>
                <div class='community-title'>Community {i} ({len(community)} members)</div>
                <div style='display: flex; flex-wrap: wrap; gap: 4px;'>
                    {"".join(members)}{overflow}
                </div>
            </div>
        """
        )

    return f"""
    <style>
        .community-card {{
            margin-bottom: 12px;
            padding: 10px;
            background: #f3f4f6;
            border-radius: 4px;
        }}
        .dark .community-card {{
            background: #374151;
        }}
        .community-title {{
            font-weight: bold;
            margin-bottom: 8px;
            color: #374151;
        }}
        .dark .community-title {{
            color: #e5e7eb;
        }}
        .community-overflow {{
            color: #6b7280;
            margin-left: 8px;
        }}
        .dark .community-overflow {{
            color: #9ca3af;
        }}
    </style>
    <div>{"".join(parts)}</div>
    """

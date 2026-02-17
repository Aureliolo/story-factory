"""Core graph rendering for vis.js visualization."""

import json
import logging
from dataclasses import dataclass
from typing import Any

from src.memory.world_database import WorldDatabase
from src.settings import Settings
from src.ui.graph_renderer._layout import (
    calculate_circular_positions,
    calculate_grid_positions,
    get_layout_options,
)
from src.ui.theme import RELATION_COLORS, get_role_graph_style
from src.utils.constants import ENTITY_COLORS

logger = logging.getLogger(__name__)

# Entity type shapes (fallback for non-icon mode)
ENTITY_SHAPES = {
    "character": "dot",
    "location": "square",
    "item": "diamond",
    "faction": "triangle",
    "concept": "star",
}

# Material Icons unicode codepoints for graph visualization
# Same icons as the entity browser (Material Design via NiceGUI):
# character: person, location: place, item: inventory_2, faction: groups, concept: lightbulb
ENTITY_ICON_CODES = {
    "character": "\ue7fd",  # person
    "location": "\ue55f",  # place
    "item": "\ue1a1",  # inventory_2
    "faction": "\uf233",  # groups
    "concept": "\ue0f0",  # lightbulb
}


@dataclass(frozen=True)
class GraphRenderResult:
    """Result of rendering a graph for vis.js.

    Attributes:
        html: HTML string for the graph container (no script tags).
        js: JavaScript string for graph initialization.
    """

    html: str
    js: str


def build_tooltip(data: dict[str, Any], max_words: int) -> str:
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

        base_color = ENTITY_COLORS.get(node_type, "#607D8B")
        node = {
            "id": node_id,
            "label": data.get("name", "Unknown"),
            "group": node_type,
            "shape": "icon",
            "icon": {
                "face": "'Material Icons'",
                "code": ENTITY_ICON_CODES.get(node_type, "\ue8fd"),  # help_outline as fallback
                "size": 40,
                "color": base_color,
                "weight": "normal",
            },
            "title": build_tooltip(data, max_words),
        }

        # Apply role-based glow using centralized helper
        attributes = data.get("attributes", {})
        role_style = get_role_graph_style(attributes, base_color)
        if role_style:
            node["glowColor"] = role_style["glow_color"]
            node["glowSize"] = role_style["glow_size"]
            node["icon"]["color"] = role_style["icon_color"]

        # Highlight selected node (overrides role-based glow)
        if node_id == selected_entity_id:
            node["glowColor"] = "#FFD700"
            node["glowSize"] = 40
            node["icon"]["color"] = base_color

        nodes.append(node)

    # Apply positions for circular/grid layouts
    if layout == "circular" and nodes:
        positions = calculate_circular_positions(len(nodes))
        for i, node in enumerate(nodes):
            node["x"] = positions[i][0]
            node["y"] = positions[i][1]
    elif layout == "grid" and nodes:
        positions = calculate_grid_positions(len(nodes))
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
            "color": {"color": RELATION_COLORS.get(rel_type, "#6b7280"), "inherit": False},
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}},
            "title": tooltip,
            "width": 2,
        }
        edges.append(edge)

    # Build layout options
    layout_options = get_layout_options(layout)

    # HTML container only (no scripts)
    graph_html = f"""
    <style>
        #{container_id} {{
            border: 1px solid #374151;
            border-radius: 4px;
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

            // Ensure Material Icons font is loaded before canvas rendering.
            // vis.js draws icons on <canvas>, which requires the font to be ready;
            // without this, icons render as empty boxes on first load.
            document.fonts.load('40px "Material Icons"').then(function() {{
            _buildGraph();
            }});
        }}

        function _buildGraph() {{
            // Destroy previous network to prevent stale canvas on re-render
            if (window.graphNetwork) {{
                window.graphNetwork.destroy();
                window.graphNetwork = null;
                window.graphNodes = null;
                window.graphEdges = null;
            }}

            var fontColor = '#e5e7eb';
            var bgColor = '#1f2937';

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
                    color: {{ color: '#6b7280', highlight: '#3b82f6', inherit: false }},
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
                    selectable: true
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

            // Precompute glow node list — only recompute when data changes
            var _glowNodes = [];
            function _updateGlowNodes() {{
                _glowNodes = [];
                nodes.forEach(function(node) {{
                    if (node.glowColor) {{
                        _glowNodes.push({{
                            id: node.id,
                            glowColor: node.glowColor,
                            glowSize: node.glowSize || 35
                        }});
                    }}
                }});
            }}
            _updateGlowNodes();
            nodes.on('*', _updateGlowNodes);

            // Draw radial gradient glow behind role/selected nodes
            // Only iterates precomputed glow nodes, not all nodes
            network.on('beforeDrawing', function(ctx) {{
                if (_glowNodes.length === 0) return;
                var glowIds = _glowNodes.map(function(n) {{ return n.id; }});
                var positions = network.getPositions(glowIds);
                for (var i = 0; i < _glowNodes.length; i++) {{
                    var gn = _glowNodes[i];
                    var pos = positions[gn.id];
                    if (!pos) continue;
                    var radius = gn.glowSize;
                    var gradient = ctx.createRadialGradient(
                        pos.x, pos.y, 0,
                        pos.x, pos.y, radius
                    );
                    gradient.addColorStop(0, gn.glowColor + 'BB');
                    gradient.addColorStop(0.4, gn.glowColor + '77');
                    gradient.addColorStop(1, gn.glowColor + '00');
                    ctx.beginPath();
                    ctx.arc(pos.x, pos.y, radius, 0, 2 * Math.PI);
                    ctx.fillStyle = gradient;
                    ctx.fill();
                }}
            }});

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

    return GraphRenderResult(html=graph_html, js=js)

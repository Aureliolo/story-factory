"""Graph rendering utilities for vis.js visualization."""

import json
import logging
from typing import Any

from memory.world_database import WorldDatabase

logger = logging.getLogger(__name__)

# Entity type colors for visualization
ENTITY_COLORS = {
    "character": "#4CAF50",  # Green
    "location": "#2196F3",  # Blue
    "item": "#FF9800",  # Orange
    "faction": "#9C27B0",  # Purple
    "concept": "#607D8B",  # Grey
}

# Entity type shapes
ENTITY_SHAPES = {
    "character": "dot",
    "location": "square",
    "item": "diamond",
    "faction": "triangle",
    "concept": "star",
}

# Relationship type colors
RELATION_COLORS = {
    "knows": "#90A4AE",
    "loves": "#E91E63",
    "hates": "#F44336",
    "located_in": "#2196F3",
    "owns": "#FF9800",
    "member_of": "#9C27B0",
    "enemy_of": "#F44336",
    "ally_of": "#4CAF50",
    "parent_of": "#795548",
    "child_of": "#795548",
}


def render_graph_html(
    world_db: WorldDatabase,
    filter_types: list[str] | None = None,
    layout: str = "force-directed",
    height: int = 500,
    selected_entity_id: str | None = None,
) -> str:
    """Generate vis.js graph HTML.

    Args:
        world_db: WorldDatabase instance
        filter_types: List of entity types to show (None = all)
        layout: Layout algorithm ("force-directed", "hierarchical", "circular", "grid")
        height: Graph container height in pixels
        selected_entity_id: ID of selected entity to highlight

    Returns:
        HTML string with embedded vis.js graph
    """
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
            "title": _build_tooltip(data),
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

    # Get node IDs that made it through the filter
    visible_node_ids = {n["id"] for n in nodes}

    edges = []
    for u, v, data in graph.edges(data=True):
        # Only include edges between visible nodes
        if u not in visible_node_ids or v not in visible_node_ids:
            continue

        rel_type = data.get("relation_type", "related")
        edge = {
            "from": u,
            "to": v,
            "label": rel_type,
            "color": RELATION_COLORS.get(rel_type, "#90A4AE"),
            "arrows": "to",
            "title": data.get("description", ""),
        }
        edges.append(edge)

    # Build layout options
    layout_options = _get_layout_options(layout)

    # Generate unique container ID
    container_id = "graph-container"

    return f"""
    <div id="{container_id}" style="height: {height}px; border: 1px solid #ddd; border-radius: 4px;"></div>
    <!-- vis-network version tracked in /package.json for Dependabot -->
    <script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
    <script>
    (function() {{
        var nodes = new vis.DataSet({json.dumps(nodes)});
        var edges = new vis.DataSet({json.dumps(edges)});

        var container = document.getElementById('{container_id}');
        var data = {{ nodes: nodes, edges: edges }};

        var options = {{
            physics: {{
                enabled: true,
                stabilization: {{ iterations: 100 }},
                barnesHut: {{
                    gravitationalConstant: -2000,
                    centralGravity: 0.3,
                    springLength: 150,
                    springConstant: 0.04
                }}
            }},
            nodes: {{
                font: {{ size: 14, color: '#333' }},
                scaling: {{ min: 10, max: 30 }}
            }},
            edges: {{
                font: {{ size: 10, align: 'middle' }},
                smooth: {{ type: 'continuous' }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 200,
                navigationButtons: true,
                keyboard: true
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

        // Store network reference for external control
        window.graphNetwork = network;
    }})();
    </script>
    """


def _build_tooltip(data: dict[str, Any]) -> str:
    """Build HTML tooltip for a node."""
    name = data.get("name", "Unknown")
    entity_type = data.get("type", "unknown").title()
    description = data.get("description", "")

    tooltip = f"<b>{name}</b><br><i>{entity_type}</i>"
    if description:
        # Truncate long descriptions
        if len(description) > 200:
            description = description[:200] + "..."
        tooltip += f"<br><br>{description}"

    return tooltip


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
        return """
            layout: {
                improvedLayout: true,
                hierarchical: false
            }
        """
    elif layout == "grid":
        return """
            layout: {
                improvedLayout: true,
                hierarchical: false
            }
        """
    else:  # force-directed (default)
        return """
            layout: {
                improvedLayout: true,
                hierarchical: false
            }
        """


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

    return f"""
    <div style="display: flex; gap: 10px; flex-wrap: wrap;">
        <div style="background: {ENTITY_COLORS["character"]}22; border-left: 4px solid {ENTITY_COLORS["character"]}; padding: 10px; border-radius: 4px; min-width: 100px;">
            <div style="font-size: 24px; font-weight: bold; color: {ENTITY_COLORS["character"]};">{counts["character"]}</div>
            <div style="font-size: 12px; color: #666;">Characters</div>
        </div>
        <div style="background: {ENTITY_COLORS["location"]}22; border-left: 4px solid {ENTITY_COLORS["location"]}; padding: 10px; border-radius: 4px; min-width: 100px;">
            <div style="font-size: 24px; font-weight: bold; color: {ENTITY_COLORS["location"]};">{counts["location"]}</div>
            <div style="font-size: 12px; color: #666;">Locations</div>
        </div>
        <div style="background: {ENTITY_COLORS["item"]}22; border-left: 4px solid {ENTITY_COLORS["item"]}; padding: 10px; border-radius: 4px; min-width: 100px;">
            <div style="font-size: 24px; font-weight: bold; color: {ENTITY_COLORS["item"]};">{counts["item"]}</div>
            <div style="font-size: 12px; color: #666;">Items</div>
        </div>
        <div style="background: {ENTITY_COLORS["faction"]}22; border-left: 4px solid {ENTITY_COLORS["faction"]}; padding: 10px; border-radius: 4px; min-width: 100px;">
            <div style="font-size: 24px; font-weight: bold; color: {ENTITY_COLORS["faction"]};">{counts["faction"]}</div>
            <div style="font-size: 12px; color: #666;">Factions</div>
        </div>
        <div style="background: #60606022; border-left: 4px solid #606060; padding: 10px; border-radius: 4px; min-width: 100px;">
            <div style="font-size: 24px; font-weight: bold; color: #606060;">{rel_count}</div>
            <div style="font-size: 12px; color: #666;">Relationships</div>
        </div>
    </div>
    <div style="margin-top: 10px; font-size: 12px; color: #888;">
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
        return "<p style='color: #666;'>No path found between these entities.</p>"

    parts = []
    for i, entity_id in enumerate(path):
        entity = world_db.get_entity(entity_id)
        if entity:
            color = ENTITY_COLORS.get(entity.type, "#607D8B")
            parts.append(
                f"<span style='background: {color}22; color: {color}; padding: 4px 8px; border-radius: 4px; font-weight: bold;'>{entity.name}</span>"
            )

            # Add arrow if not last
            if i < len(path) - 1:
                # Get relationship between this and next
                next_id = path[i + 1]
                rel = world_db.get_relationship_between(entity_id, next_id)
                if rel:
                    parts.append(
                        f"<span style='color: #666; margin: 0 8px;'>--{rel.relation_type}--></span>"
                    )
                else:
                    parts.append("<span style='color: #666; margin: 0 8px;'>---></span>")

    return f"<div style='display: flex; align-items: center; flex-wrap: wrap; gap: 4px;'>{''.join(parts)}</div>"


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
        return "<p style='color: #666;'>No entities found.</p>"

    rows = []
    for i, (entity, degree) in enumerate(most_connected, 1):
        color = ENTITY_COLORS.get(entity.type, "#607D8B")
        rows.append(
            f"""
            <tr>
                <td style='padding: 8px; text-align: center;'>{i}</td>
                <td style='padding: 8px;'>
                    <span style='background: {color}22; color: {color}; padding: 2px 6px; border-radius: 3px;'>{entity.type}</span>
                </td>
                <td style='padding: 8px; font-weight: bold;'>{entity.name}</td>
                <td style='padding: 8px; text-align: center;'>{degree}</td>
            </tr>
        """
        )

    return f"""
    <table style='width: 100%; border-collapse: collapse;'>
        <thead>
            <tr style='background: #f5f5f5;'>
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
        return "<p style='color: #666;'>No communities found.</p>"

    parts = []
    for i, community in enumerate(communities, 1):
        members = []
        for entity_id in community[:10]:  # Limit to 10 per community
            entity = world_db.get_entity(entity_id)
            if entity:
                color = ENTITY_COLORS.get(entity.type, "#607D8B")
                members.append(
                    f"<span style='background: {color}22; color: {color}; padding: 2px 6px; border-radius: 3px; margin: 2px;'>{entity.name}</span>"
                )

        overflow = ""
        if len(community) > 10:
            overflow = (
                f"<span style='color: #666; margin-left: 8px;'>+{len(community) - 10} more</span>"
            )

        parts.append(
            f"""
            <div style='margin-bottom: 12px; padding: 10px; background: #f9f9f9; border-radius: 4px;'>
                <div style='font-weight: bold; margin-bottom: 8px;'>Community {i} ({len(community)} members)</div>
                <div style='display: flex; flex-wrap: wrap; gap: 4px;'>
                    {"".join(members)}{overflow}
                </div>
            </div>
        """
        )

    return f"<div>{''.join(parts)}</div>"

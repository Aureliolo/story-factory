"""Result rendering functions for graph analysis output."""

import html

from src.memory.world_database import WorldDatabase
from src.ui.theme import ENTITY_COLORS


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

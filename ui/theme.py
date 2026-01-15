"""Theme configuration for Story Factory UI.

Centralized colors, styles, and visual constants.
"""

# ========== Entity Type Colors ==========
# Used for graph visualization and entity cards
ENTITY_COLORS = {
    "character": "#4CAF50",  # Green
    "location": "#2196F3",  # Blue
    "item": "#FF9800",  # Orange
    "faction": "#9C27B0",  # Purple
    "concept": "#607D8B",  # Grey
}

# ========== Entity Type Icons ==========
# Material Design icon names
ENTITY_ICONS = {
    "character": "person",
    "location": "place",
    "item": "inventory_2",
    "faction": "groups",
    "concept": "lightbulb",
}

# ========== Relationship Colors ==========
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
    "related_to": "#607D8B",
}

# ========== Status Colors ==========
STATUS_COLORS = {
    "interview": "#2196F3",  # Blue - gathering info
    "outlining": "#FF9800",  # Orange - planning
    "writing": "#4CAF50",  # Green - active work
    "editing": "#9C27B0",  # Purple - refining
    "complete": "#607D8B",  # Grey - done
    "error": "#F44336",  # Red - problem
}

# ========== Primary Colors ==========
COLORS = {
    "primary": "#2196F3",
    "primary_dark": "#1976D2",
    "primary_light": "#BBDEFB",
    "secondary": "#607D8B",
    "secondary_dark": "#455A64",
    "secondary_light": "#CFD8DC",
    "success": "#4CAF50",
    "warning": "#FF9800",
    "error": "#F44336",
    "info": "#00BCD4",
    "background": "#FAFAFA",
    "surface": "#FFFFFF",
    "text_primary": "#212121",
    "text_secondary": "#757575",
    "divider": "#BDBDBD",
}

# ========== Tailwind-style Classes ==========
# Reusable class strings for consistent styling
STYLES = {
    # Cards
    "card": "rounded-lg shadow-md bg-white",
    "card_header": "bg-gray-100 p-4 rounded-t-lg border-b",
    "card_body": "p-4",
    # Buttons
    "btn_primary": "bg-blue-500 hover:bg-blue-600 text-white",
    "btn_secondary": "bg-gray-500 hover:bg-gray-600 text-white",
    "btn_success": "bg-green-500 hover:bg-green-600 text-white",
    "btn_danger": "bg-red-500 hover:bg-red-600 text-white",
    "btn_outline": "border border-gray-300 hover:bg-gray-100",
    # Text
    "text_heading": "text-xl font-bold text-gray-800",
    "text_subheading": "text-lg font-semibold text-gray-700",
    "text_body": "text-base text-gray-600",
    "text_muted": "text-sm text-gray-500",
    "text_error": "text-red-500",
    # Layout
    "container": "max-w-7xl mx-auto px-4",
    "section": "py-6",
    "row": "flex flex-row gap-4",
    "column": "flex flex-col gap-4",
    # Form elements
    "input": "border rounded px-3 py-2 focus:ring-2 focus:ring-blue-500",
    "label": "text-sm font-medium text-gray-700",
    # Status badges
    "badge": "px-2 py-1 rounded-full text-xs font-medium",
    "badge_success": "bg-green-100 text-green-800",
    "badge_warning": "bg-yellow-100 text-yellow-800",
    "badge_error": "bg-red-100 text-red-800",
    "badge_info": "bg-blue-100 text-blue-800",
}

# ========== Graph Visualization Settings ==========
GRAPH_SETTINGS = {
    "node_size_min": 10,
    "node_size_max": 30,
    "edge_width": 1,
    "edge_width_strong": 3,
    "font_size": 14,
    "layout_spring_length": 150,
    "physics_gravity": -2000,
}

# ========== Quality Indicators ==========
# Used for model quality display
QUALITY_COLORS = {
    1: "#F44336",  # Red
    2: "#F44336",
    3: "#FF5722",
    4: "#FF9800",
    5: "#FFC107",  # Yellow
    6: "#CDDC39",
    7: "#8BC34A",
    8: "#4CAF50",  # Green
    9: "#2E7D32",
    10: "#1B5E20",  # Dark green
}


def get_entity_color(entity_type: str) -> str:
    """Get color for an entity type.

    Args:
        entity_type: Type of entity.

    Returns:
        Hex color code.
    """
    return ENTITY_COLORS.get(entity_type.lower(), ENTITY_COLORS["concept"])


def get_entity_icon(entity_type: str) -> str:
    """Get icon for an entity type.

    Args:
        entity_type: Type of entity.

    Returns:
        Material icon name.
    """
    return ENTITY_ICONS.get(entity_type.lower(), "help_outline")


def get_status_color(status: str) -> str:
    """Get color for a project status.

    Args:
        status: Project status string.

    Returns:
        Hex color code.
    """
    return STATUS_COLORS.get(status.lower(), COLORS["secondary"])


def get_quality_color(quality: int | float) -> str:
    """Get color for a quality rating.

    Args:
        quality: Quality rating (1-10).

    Returns:
        Hex color code.
    """
    quality_int = max(1, min(10, int(quality)))
    return QUALITY_COLORS.get(quality_int, COLORS["secondary"])


def entity_badge_style(entity_type: str) -> str:
    """Get inline style for an entity badge.

    Args:
        entity_type: Type of entity.

    Returns:
        CSS inline style string.
    """
    color = get_entity_color(entity_type)
    return f"background-color: {color}22; color: {color}; padding: 2px 8px; border-radius: 4px;"


def status_badge_style(status: str) -> str:
    """Get inline style for a status badge.

    Args:
        status: Project status.

    Returns:
        CSS inline style string.
    """
    color = get_status_color(status)
    return f"background-color: {color}22; color: {color}; padding: 2px 8px; border-radius: 4px;"

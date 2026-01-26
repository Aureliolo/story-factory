"""Theme configuration for Story Factory UI.

Centralized colors, styles, and visual constants.
"""

from src.utils.constants import ENTITY_COLORS, get_entity_color

# Re-export for backwards compatibility
__all__ = ["ENTITY_COLORS", "get_entity_color"]

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

# ========== Conflict Category Colors ==========
# Used for conflict mapping visualization
CONFLICT_COLORS = {
    "alliance": "#4CAF50",  # Green - positive relationships
    "rivalry": "#F44336",  # Red - active opposition
    "tension": "#FFC107",  # Yellow/Amber - potential conflict
    "neutral": "#2196F3",  # Blue - informational
}

# ========== Primary Colors (Light Mode) ==========
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


def get_entity_icon(entity_type: str) -> str:
    """
    Return the Material Design icon name for the given entity type.
    
    Parameters:
        entity_type (str): Entity type name (case-insensitive).
    
    Returns:
        str: Material Design icon name for the entity type, or "help_outline" if no mapping exists.
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


def get_background_class() -> str:
    """Get background class for the current theme.

    Returns:
        Background class string with Tailwind dark: variant.
    """
    return "bg-gray-50 dark:bg-gray-900"


def get_surface_class() -> str:
    """Get surface (card/panel) class for the current theme.

    Returns:
        Surface class string with Tailwind dark: variant.
    """
    return "bg-white dark:bg-gray-800"


def get_text_class(variant: str = "primary") -> str:
    """Get text color class for the current theme.

    Uses Tailwind dark: variants for automatic dark mode support.

    Args:
        variant: Text variant (primary, secondary, muted).

    Returns:
        Text class string with dark: variants.
    """
    return {
        "primary": "text-gray-800 dark:text-gray-100",
        "secondary": "text-gray-600 dark:text-gray-300",
        "muted": "text-gray-500 dark:text-gray-400",
    }.get(variant, "text-gray-800 dark:text-gray-100")
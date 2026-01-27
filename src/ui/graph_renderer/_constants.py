"""Constants for graph rendering."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GraphRenderResult:
    """Result of rendering a graph for vis.js.

    Attributes:
        html: HTML string for the graph container (no script tags).
        js: JavaScript string for graph initialization.
    """

    html: str
    js: str


# Entity type shapes (fallback for non-icon mode)
ENTITY_SHAPES = {
    "character": "dot",
    "location": "square",
    "item": "diamond",
    "faction": "triangle",
    "concept": "star",
}

# Font Awesome 5 icon codes for consistent visualization
# These match the Material Design icons used in the entity list:
# character: person -> fa-user, location: place -> fa-map-marker-alt,
# item: inventory_2 -> fa-box, faction: groups -> fa-users, concept: lightbulb -> fa-lightbulb
ENTITY_ICON_CODES = {
    "character": "\uf007",  # fa-user
    "location": "\uf3c5",  # fa-map-marker-alt
    "item": "\uf466",  # fa-box
    "faction": "\uf0c0",  # fa-users
    "concept": "\uf0eb",  # fa-lightbulb
}

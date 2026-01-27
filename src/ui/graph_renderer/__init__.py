"""Graph rendering utilities for vis.js visualization.

This package provides functions for rendering entity graphs using vis.js,
including layout calculations, result rendering, and analysis visualizations.
"""

from src.ui.graph_renderer._constants import (
    ENTITY_ICON_CODES,
    ENTITY_SHAPES,
    GraphRenderResult,
)
from src.ui.graph_renderer._layout import (
    calculate_circular_positions,
    calculate_grid_positions,
    get_layout_options,
)
from src.ui.graph_renderer._renderer import render_graph_html
from src.ui.graph_renderer._results import (
    render_centrality_result,
    render_communities_result,
    render_entity_summary_html,
    render_path_result,
)

# Re-export theme constants for backward compatibility
from src.ui.theme import ENTITY_COLORS, RELATION_COLORS

# Preserve internal names for backward compatibility
_get_layout_options = get_layout_options
_calculate_circular_positions = calculate_circular_positions
_calculate_grid_positions = calculate_grid_positions

__all__ = [
    # Theme constants (re-exported for backward compatibility)
    "ENTITY_COLORS",
    # Constants
    "ENTITY_ICON_CODES",
    "ENTITY_SHAPES",
    # Theme constants (re-exported for backward compatibility)
    "RELATION_COLORS",
    "GraphRenderResult",
    # Layout functions (internal names for backward compatibility)
    "_calculate_circular_positions",
    "_calculate_grid_positions",
    "_get_layout_options",
    # Layout functions
    "calculate_circular_positions",
    "calculate_grid_positions",
    "get_layout_options",
    # Result rendering
    "render_centrality_result",
    "render_communities_result",
    "render_entity_summary_html",
    # Core rendering
    "render_graph_html",
    "render_path_result",
]

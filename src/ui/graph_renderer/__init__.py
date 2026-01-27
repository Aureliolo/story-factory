"""Graph rendering utilities for vis.js visualization.

This package provides functions for rendering world graphs, entity summaries,
path results, centrality rankings, and community analyses as HTML/JS for vis.js.
"""

from src.ui.graph_renderer._renderer import (
    ENTITY_ICON_CODES,
    ENTITY_SHAPES,
    GraphRenderResult,
    render_graph_html,
)
from src.ui.graph_renderer._results import (
    render_centrality_result,
    render_communities_result,
    render_entity_summary_html,
    render_path_result,
)
from src.ui.theme import ENTITY_COLORS

__all__ = [
    "ENTITY_COLORS",
    "ENTITY_ICON_CODES",
    "ENTITY_SHAPES",
    "GraphRenderResult",
    "render_centrality_result",
    "render_communities_result",
    "render_entity_summary_html",
    "render_graph_html",
    "render_path_result",
]

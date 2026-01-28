"""Layout calculation utilities for graph rendering."""

import logging
import math

logger = logging.getLogger(__name__)


def get_layout_options(layout: str) -> str:
    """Get vis.js layout options string.

    Args:
        layout: Layout algorithm name ("force-directed", "hierarchical", "circular", "grid").

    Returns:
        JavaScript options string for vis.js layout configuration.
    """
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


def calculate_circular_positions(node_count: int, radius: int = 300) -> list[tuple[int, int]]:
    """Calculate positions for circular layout.

    Args:
        node_count: Number of nodes.
        radius: Circle radius in pixels.

    Returns:
        List of (x, y) positions.
    """
    positions = []
    for i in range(node_count):
        angle = (2 * math.pi * i) / node_count - math.pi / 2  # Start from top
        x = int(radius * math.cos(angle))
        y = int(radius * math.sin(angle))
        positions.append((x, y))
    return positions


def calculate_grid_positions(node_count: int, spacing: int = 150) -> list[tuple[int, int]]:
    """Compute (x, y) pixel coordinates for nodes arranged in a centered grid.

    Args:
        node_count: Number of nodes to place.
        spacing: Distance in pixels between adjacent grid cells.

    Returns:
        List of (x, y) integer coordinates for each node index (0..node_count-1)
        laid out row-wise; coordinates are centered around (0, 0).
    """
    logger.debug("Calculating grid positions: node_count=%s, spacing=%s", node_count, spacing)
    cols = max(1, math.ceil(math.sqrt(node_count)))
    positions = []
    for i in range(node_count):
        row = i // cols
        col = i % cols
        x = col * spacing - (cols * spacing) // 2
        y = row * spacing - ((node_count // cols) * spacing) // 2
        positions.append((x, y))
    return positions

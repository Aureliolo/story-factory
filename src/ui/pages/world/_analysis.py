"""Analysis, health dashboard, and conflict map functions for the World page."""

import logging

from nicegui import ui

from src.ui.components.world_health_dashboard import WorldHealthDashboard
from src.ui.graph_renderer import (
    render_centrality_result,
    render_communities_result,
    render_path_result,
)

logger = logging.getLogger(__name__)


def build_health_section(page) -> None:
    """Build the world health dashboard section.

    Args:
        page: WorldPage instance.
    """
    logger.debug("Building health dashboard section")
    if not page.state.world_db:
        return

    # Build dashboard in expansion with refreshable container
    with ui.expansion("World Health", icon="health_and_safety", value=False).classes("w-full"):
        page._health_container = ui.column().classes("w-full")
        # Initial build without notification toast
        refresh_health_dashboard(page, notify=False)


def refresh_health_dashboard(page, notify: bool = True) -> None:
    """Refresh the health dashboard content.

    Args:
        page: WorldPage instance.
        notify: Whether to show a notification toast. Default True for user-initiated
            refreshes, False for initial page build.
    """
    logger.debug(f"Refreshing health dashboard (notify={notify})")
    if not page.state.world_db or not hasattr(page, "_health_container"):
        return

    page._health_container.clear()
    with page._health_container:
        metrics = page.services.world.get_world_health_metrics(page.state.world_db)
        logger.debug(
            f"Health metrics retrieved: score={metrics.health_score:.1f}, "
            f"entities={metrics.total_entities}, orphans={metrics.orphan_count}"
        )
        dashboard = WorldHealthDashboard(
            metrics=metrics,
            on_fix_orphan=page._handle_fix_orphan,
            on_view_circular=page._handle_view_circular,
            on_improve_quality=page._handle_improve_quality,
            # User-initiated refresh from dashboard button should show toast
            on_refresh=lambda: refresh_health_dashboard(page, notify=True),
        )
        dashboard.build()
    if notify:
        ui.notify("Health metrics refreshed", type="positive")


async def handle_fix_orphan(page, entity_id: str) -> None:
    """Handle fix orphan entity request - select the entity for editing.

    Args:
        page: WorldPage instance.
        entity_id: ID of the orphan entity to fix.
    """
    logger.debug(f"_handle_fix_orphan called for entity_id={entity_id}")
    if not page.state.world_db:
        return

    # Use service call instead of direct state access
    entity = page.services.world.get_entity(page.state.world_db, entity_id)
    if entity:
        logger.info(f"Selecting orphan entity '{entity.name}' (id={entity_id}) for editing")
        page.state.select_entity(entity.id)
        # Refresh UI to show selection
        page._refresh_entity_list()
        page._refresh_entity_editor()
        if page._graph:
            page._graph.refresh()
        ui.notify(
            f"Selected '{entity.name}' - add relationships in the editor, then click Refresh",
            type="info",
        )
    else:
        logger.warning(f"Could not find entity with id={entity_id} for orphan fix")


async def handle_view_circular(page, cycle: dict) -> None:
    """Handle view circular relationship chain request.

    Args:
        page: WorldPage instance.
        cycle: Dictionary containing circular chain edge data.
    """
    logger.debug(f"_handle_view_circular called with cycle keys={list(cycle.keys())}")
    edges = cycle.get("edges", [])
    if not edges:
        return

    # Build description of the cycle with source and target names for clarity
    hop_descriptions = []
    for edge in edges:
        source = edge.get("source_name", edge.get("source", "?"))
        target = edge.get("target_name", edge.get("target", "?"))
        rel_type = edge.get("type", "?")
        hop_descriptions.append(f"{source} -[{rel_type}]-> {target}")
    cycle_desc = " ; ".join(hop_descriptions)

    logger.info(f"Displaying circular chain: {cycle_desc}")
    ui.notify(f"Circular chain: {cycle_desc}", type="warning", timeout=10000)


async def handle_improve_quality(page, entity_id: str) -> None:
    """Handle improve entity quality request.

    Args:
        page: WorldPage instance.
        entity_id: ID of the entity to improve.
    """
    logger.debug(f"_handle_improve_quality called for entity_id={entity_id}")
    if not page.state.world_db:
        return

    # Use service call instead of direct state access
    entity = page.services.world.get_entity(page.state.world_db, entity_id)
    if entity:
        logger.info(
            f"Selecting low-quality entity '{entity.name}' (id={entity_id}) for improvement"
        )
        page.state.select_entity(entity.id)
        # Refresh UI to show selection
        page._refresh_entity_list()
        page._refresh_entity_editor()
        if page._graph:
            page._graph.refresh()
        ui.notify(
            f"Selected '{entity.name}' - use 'Refine Entity' to improve quality, then click Refresh",
            type="info",
        )
    else:
        logger.warning(f"Could not find entity with id={entity_id} for quality improvement")


def build_analysis_section(page) -> None:
    """Construct the Analysis Tools UI section with tabs for graph analyses.

    Creates tabs for finding paths, showing most-connected nodes (centrality),
    detecting communities, and rendering a conflict map; wires tab controls to their
    respective handlers and initializes the _analysis_result HTML container for
    rendering analysis output.

    Args:
        page: WorldPage instance.
    """
    with ui.expansion("Analysis Tools", icon="analytics", value=False).classes("w-full"):
        with ui.tabs().classes("w-full") as tabs:
            ui.tab("path", label="Find Path")
            ui.tab("centrality", label="Most Connected")
            ui.tab("communities", label="Communities")
            ui.tab("conflicts", label="Conflict Map")

        with ui.tab_panels(tabs, value="path").classes("w-full"):
            # Path finder
            with ui.tab_panel("path"):
                with ui.row().classes("items-end gap-4"):
                    entities = page._get_entity_options()
                    path_source = ui.select(label="From", options=entities).classes("w-48")
                    path_target = ui.select(label="To", options=entities).classes("w-48")
                    ui.button(
                        "Find Path",
                        on_click=lambda: find_path(page, path_source.value, path_target.value),
                    )

            # Centrality analysis
            with ui.tab_panel("centrality"):
                ui.button(
                    "Show Most Connected",
                    on_click=lambda: show_centrality(page),
                )

            # Community detection
            with ui.tab_panel("communities"):
                ui.button(
                    "Detect Communities",
                    on_click=lambda: show_communities(page),
                )

            # Conflict mapping
            with ui.tab_panel("conflicts"):
                build_conflict_map_tab(page)

        # Analysis result display
        page._analysis_result = ui.html(sanitize=False).classes("w-full mt-4")


def find_path(page, source_id: str, target_id: str) -> None:
    """Find path between two entities.

    Args:
        page: WorldPage instance.
        source_id: Source entity ID.
        target_id: Target entity ID.
    """
    if not page.state.world_db or not source_id or not target_id:
        return

    path = page.services.world.find_path(page.state.world_db, source_id, target_id)

    if page._analysis_result:
        page._analysis_result.content = render_path_result(page.state.world_db, path or [])


def show_centrality(page) -> None:
    """Show most connected entities.

    Args:
        page: WorldPage instance.
    """
    if not page.state.world_db:
        return

    if page._analysis_result:
        page._analysis_result.content = render_centrality_result(page.state.world_db)


def show_communities(page) -> None:
    """Render and display community-detection analysis in the analysis result pane.

    Args:
        page: WorldPage instance.
    """
    if not page.state.world_db:
        return

    if page._analysis_result:
        page._analysis_result.content = render_communities_result(page.state.world_db)


def build_conflict_map_tab(page) -> None:
    """Construct the Conflict Map tab UI.

    Shows relationships colored by conflict category and a link to the World Timeline.

    Args:
        page: WorldPage instance.
    """
    logger.debug("Building conflict map tab")
    from src.ui.components.conflict_graph import ConflictGraphComponent

    with ui.column().classes("w-full gap-4"):
        ui.label(
            "Visualize relationships colored by conflict category: "
            "alliances (green), rivalries (red), tensions (yellow), and neutral (blue)."
        ).classes("text-sm text-gray-600 dark:text-gray-400")

        # Conflict graph component
        conflict_graph = ConflictGraphComponent(
            world_db=page.state.world_db,
            services=page.services,
            on_node_select=page._on_node_select,
            height=400,
        )
        conflict_graph.build()

        # Link to full world timeline
        with ui.row().classes("items-center gap-4 mt-2"):
            ui.button(
                "View World Timeline",
                on_click=lambda: ui.navigate.to("/world-timeline"),
                icon="timeline",
            ).props("flat")

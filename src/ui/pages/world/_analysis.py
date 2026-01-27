"""World Builder page - analysis mixin with health and analysis methods."""

import logging

from nicegui import ui

from src.ui.components.world_health_dashboard import WorldHealthDashboard
from src.ui.pages.world._page import WorldPageBase

logger = logging.getLogger(__name__)


class AnalysisMixin(WorldPageBase):
    """Mixin providing health and analysis methods for WorldPage."""

    def _build_health_section(self) -> None:
        """Build the world health dashboard section."""
        logger.debug("Building health dashboard section")
        if not self.state.world_db:
            return

        # Build dashboard in expansion with refreshable container
        with ui.expansion("World Health", icon="health_and_safety", value=False).classes("w-full"):
            self._health_container = ui.column().classes("w-full")
            # Initial build without notification toast
            self._refresh_health_dashboard(notify=False)

    def _refresh_health_dashboard(self, notify: bool = True) -> None:
        """Refresh the health dashboard content.

        Args:
            notify: Whether to show a notification toast. Default True for user-initiated
                refreshes, False for initial page build.
        """
        logger.debug(f"Refreshing health dashboard (notify={notify})")
        if not self.state.world_db or not hasattr(self, "_health_container"):
            return

        self._health_container.clear()
        with self._health_container:
            metrics = self.services.world.get_world_health_metrics(self.state.world_db)
            logger.debug(
                f"Health metrics retrieved: score={metrics.health_score:.1f}, "
                f"entities={metrics.total_entities}, orphans={metrics.orphan_count}"
            )
            dashboard = WorldHealthDashboard(
                metrics=metrics,
                on_fix_orphan=self._handle_fix_orphan,
                on_view_circular=self._handle_view_circular,
                on_improve_quality=self._handle_improve_quality,
                # User-initiated refresh from dashboard button should show toast
                on_refresh=lambda: self._refresh_health_dashboard(notify=True),
            )
            dashboard.build()
        if notify:
            ui.notify("Health metrics refreshed", type="positive")

    async def _handle_fix_orphan(self, entity_id: str) -> None:
        """Handle fix orphan entity request - select the entity for editing."""
        logger.debug(f"_handle_fix_orphan called for entity_id={entity_id}")
        if not self.state.world_db:
            return

        # Use service call instead of direct state access
        entity = self.services.world.get_entity(self.state.world_db, entity_id)
        if entity:
            logger.info(f"Selecting orphan entity '{entity.name}' (id={entity_id}) for editing")
            self.state.select_entity(entity.id)
            # Refresh UI to show selection
            self._refresh_entity_list()
            self._refresh_entity_editor()
            if self._graph:
                self._graph.refresh()
            ui.notify(
                f"Selected '{entity.name}' - add relationships in the editor, then click Refresh",
                type="info",
            )
        else:
            logger.warning(f"Could not find entity with id={entity_id} for orphan fix")

    async def _handle_view_circular(self, cycle: dict) -> None:
        """Handle view circular relationship chain request."""
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

    async def _handle_improve_quality(self, entity_id: str) -> None:
        """Handle improve entity quality request."""
        logger.debug(f"_handle_improve_quality called for entity_id={entity_id}")
        if not self.state.world_db:
            return

        # Use service call instead of direct state access
        entity = self.services.world.get_entity(self.state.world_db, entity_id)
        if entity:
            logger.info(
                f"Selecting low-quality entity '{entity.name}' (id={entity_id}) for improvement"
            )
            self.state.select_entity(entity.id)
            # Refresh UI to show selection
            self._refresh_entity_list()
            self._refresh_entity_editor()
            if self._graph:
                self._graph.refresh()
            ui.notify(
                f"Selected '{entity.name}' - use 'Refine Entity' to improve quality, then click Refresh",
                type="info",
            )
        else:
            logger.warning(f"Could not find entity with id={entity_id} for quality improvement")

    def _build_analysis_section(self) -> None:
        """
        Constructs the Analysis Tools UI section with tabs for graph analyses.

        Creates tabs for finding paths, showing most-connected nodes (centrality), detecting communities, and rendering a conflict map; wires tab controls to their respective handlers and initializes the _analysis_result HTML container for rendering analysis output.
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
                        entities = self._get_entity_options()
                        path_source = ui.select(label="From", options=entities).classes("w-48")
                        path_target = ui.select(label="To", options=entities).classes("w-48")
                        ui.button(
                            "Find Path",
                            on_click=lambda: self._find_path(path_source.value, path_target.value),
                        )

                # Centrality analysis
                with ui.tab_panel("centrality"):
                    ui.button(
                        "Show Most Connected",
                        on_click=self._show_centrality,
                    )

                # Community detection
                with ui.tab_panel("communities"):
                    ui.button(
                        "Detect Communities",
                        on_click=self._show_communities,
                    )

                # Conflict mapping
                with ui.tab_panel("conflicts"):
                    self._build_conflict_map_tab()

            # Analysis result display
            self._analysis_result = ui.html(sanitize=False).classes("w-full mt-4")

    # Methods to be implemented by other mixins
    def _refresh_entity_list(self) -> None:
        """Refresh the entity list display - implemented by BrowserMixin."""
        raise NotImplementedError

    def _refresh_entity_editor(self) -> None:
        """Refresh the entity editor panel - implemented by EditorMixin."""
        raise NotImplementedError

    def _get_entity_options(self) -> dict[str, str]:
        """Get entity options for select dropdowns - implemented by base class."""
        raise NotImplementedError

    def _find_path(self, source_id: str, target_id: str) -> None:
        """Find path between two entities - implemented by GraphMixin."""
        raise NotImplementedError

    def _show_centrality(self) -> None:
        """Show most connected entities - implemented by GraphMixin."""
        raise NotImplementedError

    def _show_communities(self) -> None:
        """Show communities - implemented by GraphMixin."""
        raise NotImplementedError

    def _build_conflict_map_tab(self) -> None:
        """Build conflict map tab - implemented by GraphMixin."""
        raise NotImplementedError

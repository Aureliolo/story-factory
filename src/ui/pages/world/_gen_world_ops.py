"""World Builder page - world-level generation operations (rebuild, clear, relationships)."""

import logging
import threading
from collections.abc import Callable
from typing import Any

from nicegui import ui

from src.services.world_quality import EntityGenerationProgress
from src.ui.components.build_dialog import show_build_structure_dialog
from src.ui.pages.world._page import WorldPageBase

logger = logging.getLogger(__name__)


class GenWorldOpsMixin(WorldPageBase):
    """Mixin providing world-level generation operations for WorldPage.

    Includes: relationship generation for entities, rebuild, clear, and mini descriptions.
    """

    async def _generate_relationships_for_entities(
        self, entity_names: list[str], count_per_entity: int
    ) -> None:
        """Generate relationships for specific entities.

        Args:
            entity_names: Names of entities to generate relationships for.
            count_per_entity: Number of relationships to generate per entity.
        """
        if not self.state.project or not self.state.world_db:
            ui.notify("No project loaded", type="negative")
            return

        logger.info(
            f"Generating {count_per_entity} relationships for each of {len(entity_names)} entities"
        )

        # Check if quality refinement is enabled
        use_quality = (
            self.state.quality_refinement_enabled and self.services.settings.world_quality_enabled
        )

        # Get all entity names for relationship generation
        all_entity_names = self._get_all_entity_names()

        # Get existing relationships to avoid duplicates (use source_id and target_id)
        existing_rels = [
            (r.source_id, r.target_id) for r in self.state.world_db.list_relationships()
        ]

        total_count = len(entity_names) * count_per_entity

        # Create cancellation infrastructure for quality generation
        self._generation_cancel_event = threading.Event()

        def should_cancel() -> bool:
            """Check if generation should be cancelled."""
            return (
                self._generation_cancel_event is not None and self._generation_cancel_event.is_set()
            )

        if use_quality:
            # Create progress dialog with cancel button
            self._generation_dialog = ui.dialog().props("persistent")
            progress_label: ui.label | None = None
            progress_bar: ui.linear_progress | None = None
            eta_label: ui.label | None = None
            cancel_btn: ui.button | None = None

            with self._generation_dialog, ui.card().classes("w-96 p-4"):
                ui.label("Generating Relationships").classes("text-lg font-bold")
                progress_label = ui.label(f"Starting generation of {total_count} relationships...")
                progress_bar = ui.linear_progress(value=0).classes("w-full my-2")
                eta_label = ui.label("Calculating...").classes("text-sm text-gray-500")

                def do_cancel() -> None:
                    """Handle cancel button click."""
                    logger.info("User requested cancellation of relationship generation")
                    if self._generation_cancel_event:
                        self._generation_cancel_event.set()
                    if cancel_btn:
                        cancel_btn.disable()
                    if progress_label:
                        progress_label.text = "Cancelling after current relationship..."

                cancel_btn = ui.button("Cancel", on_click=do_cancel).props("flat color=negative")

            self._generation_dialog.open()

            def update_progress(progress: EntityGenerationProgress) -> None:
                """Update dialog with generation progress."""
                if progress_label:
                    if progress.entity_name:
                        progress_label.text = f"Generated: {progress.entity_name}"
                    else:
                        progress_label.text = (
                            f"Generating relationship {progress.current}/{progress.total}..."
                        )

                if progress_bar:
                    progress_bar.value = progress.progress_fraction

                if eta_label:
                    if progress.estimated_remaining_seconds is not None:
                        total_secs = int(progress.estimated_remaining_seconds)
                        if total_secs >= 3600:
                            hours, remainder = divmod(total_secs, 3600)
                            mins, secs = divmod(remainder, 60)
                            eta_label.text = f"~{hours}:{mins:02d}:{secs:02d} remaining"
                        else:
                            mins, secs = divmod(total_secs, 60)
                            eta_label.text = f"~{mins}:{secs:02d} remaining"
                    elif progress.current > 1:
                        eta_label.text = "Calculating..."

            notification = None  # No notification when using dialog
        else:
            notification = ui.notification(
                message=f"Generating relationships for {len(entity_names)} entities...",
                spinner=True,
                timeout=None,
            )
            update_progress = None  # type: ignore[assignment]

        try:
            from nicegui import run

            if use_quality:
                # Generate relationships with quality refinement
                results = await run.io_bound(
                    self.services.world_quality.generate_relationships_with_quality,
                    self.state.project,
                    all_entity_names,
                    existing_rels,
                    total_count,
                    should_cancel,
                    update_progress,
                )

                # Check for partial failure and notify user
                self._notify_partial_failure(
                    len(results), total_count, "relationships", should_cancel
                )

                if len(results) == 0:
                    if self._generation_dialog:
                        self._generation_dialog.close()
                    elif notification:
                        notification.dismiss()
                    ui.notify("Failed to generate any relationships", type="negative")
                    return

                if self._generation_dialog:
                    self._generation_dialog.close()
                elif notification:
                    notification.dismiss()

                # Show preview dialog
                def add_selected_relationships(selected: list[tuple[Any, Any]]) -> None:
                    """Add selected relationships to the world database from the preview."""
                    if not selected:
                        ui.notify("No relationships selected", type="info")
                        return
                    if not self.state.world_db:
                        ui.notify("No world database", type="negative")
                        return

                    # Get all entities to look up IDs from names
                    entities = self.state.world_db.list_entities()
                    added_count = 0

                    for rel_data, _scores in selected:
                        source_name = rel_data.get("source", "")
                        target_name = rel_data.get("target", "")
                        rel_type = rel_data.get("relation_type", "related_to")
                        desc = rel_data.get("description", "")

                        # Look up entity IDs from names
                        source_entity = next((e for e in entities if e.name == source_name), None)
                        target_entity = next((e for e in entities if e.name == target_name), None)

                        if source_entity and target_entity:
                            self.services.world.add_relationship(
                                self.state.world_db,
                                source_id=source_entity.id,
                                target_id=target_entity.id,
                                relation_type=rel_type,
                                description=desc,
                            )
                            added_count += 1
                        else:
                            logger.warning(
                                f"Could not find entities for relationship: "
                                f"{source_name} -> {target_name}"
                            )

                    self.state.world_db.invalidate_graph_cache()
                    self._refresh_entity_list()
                    if self._graph:
                        self._graph.refresh()
                    ui.notify(
                        f"Added {added_count} relationships",
                        type="positive",
                    )

                self._show_entity_preview_dialog(
                    "relationship", results, add_selected_relationships
                )
            else:
                # Non-quality generation - simpler approach
                if notification:
                    notification.dismiss()
                ui.notify(
                    "Relationship generation requires quality refinement to be enabled",
                    type="warning",
                )

        except Exception as e:
            if self._generation_dialog:
                self._generation_dialog.close()
            elif notification:
                notification.dismiss()
            logger.exception(f"Error generating relationships: {e}")
            ui.notify(f"Error: {e}", type="negative")

    def _confirm_regenerate(self) -> None:
        """Show confirmation dialog before regenerating world.

        Shows a simple confirmation asking the user to confirm deletion of existing data,
        then opens the shared build dialog in rebuild mode.
        """
        logger.info("Rebuild World button clicked - showing confirmation dialog")

        if not self.state.project or not self.state.world_db:
            ui.notify("No project available", type="negative")
            return

        # Count existing data for the warning
        entity_count = self.state.world_db.count_entities()
        rel_count = len(self.state.world_db.list_relationships())
        chapter_count = len(self.state.project.chapters)
        char_count = len(self.state.project.characters)

        with ui.dialog() as dialog, ui.card().classes("p-4 min-w-[400px]"):
            ui.label("Rebuild World?").classes("text-lg font-bold")
            ui.label(
                f"This will permanently delete all existing world data:\n"
                f"• {entity_count} entities\n"
                f"• {rel_count} relationships\n"
                f"• {chapter_count} chapter outlines\n"
                f"• {char_count} characters\n\n"
                "Then generate everything fresh from your story brief."
            ).classes("text-gray-600 dark:text-gray-400 whitespace-pre-line")

            async def confirm_and_build() -> None:
                """Close the confirmation dialog and trigger a full rebuild of the story structure."""
                dialog.close()
                await self._build_structure(rebuild=True)

            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button(
                    "Continue to Rebuild",
                    on_click=confirm_and_build,
                    icon="refresh",
                ).props("color=negative")

        dialog.open()

    async def _build_structure(self, rebuild: bool = False) -> None:
        """Build story structure using the shared dialog.

        Args:
            rebuild: If True, clears all existing data before building (uses full_rebuild).
                     If False, builds without clearing (uses full).
        """

        async def on_complete() -> None:
            """Generate mini descriptions and reload page after build."""
            await self._generate_mini_descriptions()
            ui.navigate.reload()

        await show_build_structure_dialog(
            state=self.state,
            services=self.services,
            rebuild=rebuild,
            on_complete=on_complete,
        )

    def _confirm_clear_world(self) -> None:
        """Show confirmation dialog before clearing world data."""
        logger.info("Clear World button clicked - showing confirmation dialog")

        if not self.state.world_db or not self.state.project:
            ui.notify("No world data to clear", type="info")
            return

        # Count existing data
        entity_count = self.state.world_db.count_entities()
        rel_count = len(self.state.world_db.list_relationships())
        chapter_count = len(self.state.project.chapters)
        char_count = len(self.state.project.characters)

        if entity_count == 0 and rel_count == 0 and chapter_count == 0 and char_count == 0:
            ui.notify("World is already empty", type="info")
            return

        with ui.dialog() as dialog, ui.card().classes("p-4 min-w-[400px]"):
            ui.label("Clear World?").classes("text-lg font-bold")
            ui.label(
                f"This will permanently delete all world and story structure data:\n"
                f"• {entity_count} entities\n"
                f"• {rel_count} relationships\n"
                f"• {chapter_count} chapter outlines\n"
                f"• {char_count} characters\n\n"
                f"Your interview and story brief will be kept.\n"
                f"This action cannot be undone."
            ).classes("text-gray-600 dark:text-gray-400 whitespace-pre-line")

            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button(
                    "Clear All",
                    on_click=lambda: self._do_clear_world(dialog),
                    icon="delete_sweep",
                ).props("color=warning")

        dialog.open()

    def _do_clear_world(self, dialog: ui.dialog) -> None:
        """Execute world clear - removes all entities, relationships, and story structure."""
        logger.info("User confirmed clear - removing all world data and story structure")
        dialog.close()

        if not self.state.world_db or not self.state.project:
            logger.warning("Clear failed: no world_db or project available")
            ui.notify("No world database available", type="negative")
            return

        try:
            # Delete all relationships first (they reference entities)
            relationships = self.state.world_db.list_relationships()
            for rel in relationships:
                self.state.world_db.delete_relationship(rel.id)
            logger.info(f"Deleted {len(relationships)} relationships")

            # Delete all entities
            entities = self.state.world_db.list_entities()
            for entity in entities:
                self.state.world_db.delete_entity(entity.id)
            logger.info(f"Deleted {len(entities)} entities")

            # Clear story structure (but keep brief and interview)
            chapter_count = len(self.state.project.chapters)
            char_count = len(self.state.project.characters)
            self.state.project.chapters = []
            self.state.project.characters = []
            self.state.project.world_description = ""
            self.state.project.plot_points = []
            logger.info(
                f"Cleared {chapter_count} chapters and {char_count} characters from project"
            )

            # Save the project with cleared structure
            self.services.project.save_project(self.state.project)
            logger.info("Project saved with cleared structure")

            # Refresh UI
            self._refresh_entity_list()
            if self._graph:
                self._graph.refresh()

            ui.notify(
                f"World cleared: removed {len(entities)} entities, {len(relationships)} relationships, "
                f"{chapter_count} chapters, {char_count} characters",
                type="positive",
            )

            # Reload page to update toolbar (Clear button visibility, Build Structure button)
            ui.navigate.reload()

        except Exception as e:
            logger.exception(f"Error clearing world: {e}")
            ui.notify(f"Error: {e}", type="negative")

    async def _generate_mini_descriptions(self) -> None:
        """Generate mini descriptions for entity tooltips."""
        from nicegui import run

        if not self.state.world_db:
            return

        all_entities = self.state.world_db.list_entities()
        entity_data = [
            {"name": e.name, "type": e.type, "description": e.description} for e in all_entities
        ]
        mini_descs = await run.io_bound(
            self.services.world_quality.generate_mini_descriptions_batch,
            entity_data,
        )
        logger.info(f"Generated {len(mini_descs)} mini descriptions")

        # Update entities with mini descriptions
        for entity in all_entities:
            if entity.name in mini_descs:
                # Merge mini_description into existing attributes
                attrs = dict(entity.attributes) if entity.attributes else {}
                attrs["mini_description"] = mini_descs[entity.name]
                self.state.world_db.update_entity(
                    entity_id=entity.id,
                    attributes=attrs,
                )
        logger.info("Updated entities with mini descriptions")

        # Invalidate graph cache to ensure fresh tooltips
        self.state.world_db.invalidate_graph_cache()

    # Methods to be implemented by other mixins
    def _refresh_entity_list(self) -> None:
        """Refresh the entity list display - implemented by BrowserMixin."""
        raise NotImplementedError

    def _get_all_entity_names(self) -> list[str]:
        """Get all entity names - implemented by base class."""
        raise NotImplementedError

    def _notify_partial_failure(
        self,
        results_count: int,
        requested_count: int,
        entity_type: str,
        should_cancel: Callable[[], bool],
    ) -> None:
        """Notify user of partial failure - implemented by GenerationMixin."""
        raise NotImplementedError

    def _show_entity_preview_dialog(
        self,
        entity_type: str,
        entities: list[tuple[Any, Any]],
        on_confirm: Any,
    ) -> None:
        """Show entity preview dialog - implemented by GenDialogsMixin."""
        raise NotImplementedError

    @staticmethod
    def _add_relationships_from_dicts(
        world_service: Any,
        world_db: Any,
        entities: list[Any],
        relationships: list[dict[str, Any]],
    ) -> int:
        """Add relationships from a list of dicts to the world database.

        Args:
            world_service: WorldService instance.
            world_db: WorldDatabase instance.
            entities: List of Entity objects for name-to-ID resolution.
            relationships: List of relationship dicts with source, target, relation_type, description.

        Returns:
            Number of relationships successfully added.
        """
        added = 0
        for rel in relationships:
            if isinstance(rel, dict) and "source" in rel and "target" in rel:
                source_entity = next((e for e in entities if e.name == rel["source"]), None)
                target_entity = next((e for e in entities if e.name == rel["target"]), None)
                if source_entity and target_entity:
                    world_service.add_relationship(
                        world_db,
                        source_entity.id,
                        target_entity.id,
                        rel.get("relation_type", "knows"),
                        rel.get("description", ""),
                    )
                    added += 1
                else:
                    logger.warning(
                        f"Skipping relationship: source={rel['source']} or "
                        f"target={rel['target']} not found"
                    )
            else:
                logger.warning(f"Skipping invalid relationship: {rel}")
        return added

    def _add_quality_relationships(
        self,
        selected: list[tuple[Any, Any]],
        entities: list[Any],
    ) -> None:
        """Add selected quality-scored relationships to the world database.

        Called as a callback from the entity preview dialog after relationship
        generation with quality refinement.

        Args:
            selected: List of (rel_data_dict, quality_scores) tuples.
            entities: List of Entity objects for name-to-ID resolution.
        """
        if not selected:
            ui.notify("No relationships selected", type="info")
            return
        if not self.state.world_db or not self.state.project:
            ui.notify("No project loaded", type="negative")
            return
        added = 0
        for rel_data, scores in selected:
            if isinstance(rel_data, dict) and "source" in rel_data and "target" in rel_data:
                source_entity = next((e for e in entities if e.name == rel_data["source"]), None)
                target_entity = next((e for e in entities if e.name == rel_data["target"]), None)
                if source_entity and target_entity:
                    rel_id = self.services.world.add_relationship(
                        self.state.world_db,
                        source_entity.id,
                        target_entity.id,
                        rel_data.get("relation_type", "knows"),
                        rel_data.get("description", ""),
                    )
                    self.state.world_db.update_relationship(
                        relationship_id=rel_id,
                        attributes={"quality_scores": scores.to_dict()},
                    )
                    added += 1
        self._finalize_generation("relationships")
        avg_quality = sum(s.average for _, s in selected) / len(selected) if selected else 0
        ui.notify(
            f"Added {added} relationships (avg quality: {avg_quality:.1f})",
            type="positive",
        )

    def _finalize_generation(self, entity_type: str) -> None:
        """Finalize generation by refreshing UI and saving project.

        Args:
            entity_type: Type of entity that was generated.
        """
        if self.state.world_db:
            self.state.world_db.invalidate_graph_cache()
        logger.info("Refreshing UI after generation...")
        self._refresh_entity_list()
        if self._graph:
            self._graph.refresh()
        if self.state.project:
            logger.info(f"Saving project {self.state.project.id}...")
            self.services.project.save_project(self.state.project)
            logger.info("Project saved successfully")
        logger.info(f"Generation of {entity_type} completed successfully")

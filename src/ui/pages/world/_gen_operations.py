"""World Builder page - entity generation operations (the core _generate_more method)."""

import logging
import threading
from typing import Any

from nicegui import ui

from src.services.world_quality import EntityGenerationProgress
from src.ui.pages.world._gen_world_ops import GenWorldOpsMixin
from src.ui.pages.world._page import WorldPageBase
from src.utils.exceptions import WorldGenerationError

logger = logging.getLogger(__name__)


class GenOperationsMixin(GenWorldOpsMixin, WorldPageBase):
    """Mixin providing the core entity generation method for WorldPage."""

    async def _generate_more(
        self, entity_type: str, count: int | None = None, custom_instructions: str | None = None
    ) -> None:
        """Generate more entities of a specific type.

        Args:
            entity_type: Type of entities to generate (characters, locations, factions, items, concepts, relationships)
            count: Number of entities to generate (defaults to random from settings).
            custom_instructions: Optional custom instructions to refine generation.
        """
        logger.info(
            f"Generate more: entity_type={entity_type}, count={count}, "
            f"custom_instructions={custom_instructions[:50] if custom_instructions else None}"
        )

        if not self.state.project or not self.state.world_db:
            logger.warning("Generate more failed: no project or world_db")
            ui.notify("No project loaded", type="negative")
            return

        logger.info(f"Starting generation of {entity_type} for project {self.state.project.id}")

        # Check if quality refinement is enabled
        use_quality = (
            self.state.quality_refinement_enabled and self.services.settings.world_quality_enabled
        )
        logger.info(f"Quality refinement enabled: {use_quality}")

        # Use provided count or get random from settings
        if count is None:
            count = self._get_random_count(entity_type)
        logger.info(f"Will generate {count} {entity_type}")

        # Get ALL existing entity names to avoid duplicates
        all_existing_names = self._get_all_entity_names()
        logger.info(f"Found {len(all_existing_names)} existing entities to avoid duplicates")

        # Create cancellation infrastructure for quality generation
        self._generation_cancel_event = threading.Event()

        def should_cancel() -> bool:
            """Check if generation should be cancelled."""
            return (
                self._generation_cancel_event is not None and self._generation_cancel_event.is_set()
            )

        # Create progress dialog for quality generation, or simple notification for non-quality
        quality_msg = " with quality refinement" if use_quality else ""

        if use_quality:
            # Create progress dialog with cancel button
            self._generation_dialog = ui.dialog().props("persistent")
            progress_label: ui.label | None = None
            progress_bar: ui.linear_progress | None = None
            eta_label: ui.label | None = None
            cancel_btn: ui.button | None = None

            with self._generation_dialog, ui.card().classes("w-96 p-4"):
                ui.label(f"Generating {entity_type.title()}").classes("text-lg font-bold")
                progress_label = ui.label(f"Starting generation of {count} {entity_type}...")
                progress_bar = ui.linear_progress(value=0).classes("w-full my-2")
                eta_label = ui.label("Calculating...").classes("text-sm text-gray-500")

                def do_cancel() -> None:
                    """Handle cancel button click."""
                    logger.info(f"User requested cancellation of {entity_type} generation")
                    if self._generation_cancel_event:
                        self._generation_cancel_event.set()
                    if cancel_btn:
                        cancel_btn.disable()
                    if progress_label:
                        progress_label.text = "Cancelling after current entity..."

                cancel_btn = ui.button("Cancel", on_click=do_cancel).props("flat color=negative")

            self._generation_dialog.open()

            def update_progress(progress: EntityGenerationProgress) -> None:
                """Update dialog with generation progress."""
                if progress_label:
                    if progress.entity_name:
                        progress_label.text = f"Generated: {progress.entity_name}"
                    else:
                        progress_label.text = (
                            f"Generating {progress.entity_type} "
                            f"{progress.current}/{progress.total}..."
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
            # Use simple notification for non-quality generation
            notification = ui.notification(
                message=f"Generating {count} {entity_type}{quality_msg}...",
                spinner=True,
                timeout=None,
            )
            update_progress = None  # type: ignore[assignment]

        try:
            from nicegui import run

            if entity_type == "characters":
                if use_quality:
                    # Generate characters with quality refinement
                    logger.info(
                        f"Calling world quality service to generate characters "
                        f"(custom_instructions: {custom_instructions is not None})..."
                    )
                    results = await run.io_bound(
                        self.services.world_quality.generate_characters_with_quality,
                        self.state.project,
                        all_existing_names,
                        count,
                        custom_instructions,
                        should_cancel,
                        update_progress,
                    )
                    logger.info(f"Generated {len(results)} characters with quality refinement")

                    # Check for partial failure and notify user
                    self._notify_partial_failure(len(results), count, "characters", should_cancel)
                    if len(results) == 0:
                        if self._generation_dialog:
                            self._generation_dialog.close()
                        ui.notify("Failed to generate any characters", type="negative")
                        return

                    # Generate mini descriptions for hover tooltips
                    if progress_label:
                        progress_label.text = "Generating hover summaries..."
                    entity_data = [
                        {"name": c.name, "type": "character", "description": c.description}
                        for c, _ in results
                    ]
                    mini_descs = await run.io_bound(
                        self.services.world_quality.generate_mini_descriptions_batch,
                        entity_data,
                    )
                    if self._generation_dialog:
                        self._generation_dialog.close()

                    # Define callback to add selected characters
                    def add_selected_characters(selected: list[tuple[Any, Any]]) -> None:
                        """Add selected characters to the world database and project."""
                        if not selected:
                            ui.notify("No characters selected", type="info")
                            return
                        if not self.state.world_db or not self.state.project:
                            ui.notify("No project loaded", type="negative")
                            return
                        added_names = []
                        for char, scores in selected:
                            attrs = {
                                "role": char.role,
                                "traits": char.personality_traits,
                                "goals": char.goals,
                                "arc": char.arc_notes,
                                "quality_scores": scores.to_dict(),
                            }
                            if char.name in mini_descs:
                                attrs["mini_description"] = mini_descs[char.name]
                            self.services.world.add_entity(
                                self.state.world_db,
                                name=char.name,
                                entity_type="character",
                                description=char.description,
                                attributes=attrs,
                            )
                            # Also add to story state
                            self.state.project.characters.append(char)
                            added_names.append(char.name)
                        # Refresh UI and save
                        self.state.world_db.invalidate_graph_cache()
                        self._refresh_entity_list()
                        if self._graph:
                            self._graph.refresh()
                        self.services.project.save_project(self.state.project)
                        avg_quality = (
                            sum(s.average for _, s in selected) / len(selected) if selected else 0
                        )
                        ui.notify(
                            f"Added {len(selected)} characters (avg quality: {avg_quality:.1f})",
                            type="positive",
                        )
                        # Prompt for relationship generation
                        self._prompt_for_relationships_after_add(added_names)

                    # Show preview dialog
                    self._show_entity_preview_dialog("character", results, add_selected_characters)
                    return  # Early return - callback handles the rest
                else:
                    # Generate characters via service (original method)
                    logger.info("Calling story service to generate characters...")
                    new_chars = await run.io_bound(
                        self.services.story.generate_more_characters, self.state.project, count
                    )
                    logger.info(f"Generated {len(new_chars)} characters from LLM")
                    # Add to world database
                    for char in new_chars:
                        self.services.world.add_entity(
                            self.state.world_db,
                            name=char.name,
                            entity_type="character",
                            description=char.description,
                            attributes={
                                "role": char.role,
                                "traits": char.personality_traits,
                                "goals": char.goals,
                                "arc": char.arc_notes,
                            },
                        )
                    logger.info(f"Added {len(new_chars)} characters to world database")
                    if notification:
                        notification.dismiss()
                    ui.notify(f"Added {len(new_chars)} new characters!", type="positive")

            elif entity_type == "locations":
                if use_quality:
                    # Generate locations with quality refinement
                    logger.info("Calling world quality service to generate locations...")
                    loc_results = await run.io_bound(
                        self.services.world_quality.generate_locations_with_quality,
                        self.state.project,
                        all_existing_names,
                        count,
                        should_cancel,
                        update_progress,
                    )
                    logger.info(f"Generated {len(loc_results)} locations with quality refinement")

                    # Check for partial failure and notify user
                    self._notify_partial_failure(
                        len(loc_results), count, "locations", should_cancel
                    )
                    if len(loc_results) == 0:
                        if self._generation_dialog:
                            self._generation_dialog.close()
                        ui.notify("Failed to generate any locations", type="negative")
                        return

                    # Generate mini descriptions for hover tooltips
                    if progress_label:
                        progress_label.text = "Generating hover summaries..."
                    entity_data = [
                        {
                            "name": loc.get("name", ""),
                            "type": "location",
                            "description": loc.get("description", ""),
                        }
                        for loc, _ in loc_results
                        if isinstance(loc, dict) and loc.get("name")
                    ]
                    mini_descs = await run.io_bound(
                        self.services.world_quality.generate_mini_descriptions_batch,
                        entity_data,
                    )
                    if self._generation_dialog:
                        self._generation_dialog.close()

                    # Define callback to add selected locations
                    def add_selected_locations(selected: list[tuple[Any, Any]]) -> None:
                        """Add selected locations to the world database."""
                        if not selected:
                            ui.notify("No locations selected", type="info")
                            return
                        if not self.state.world_db or not self.state.project:
                            ui.notify("No project loaded", type="negative")
                            return
                        added_names = []
                        for loc, scores in selected:
                            if isinstance(loc, dict) and "name" in loc:
                                attrs = {
                                    "significance": loc.get("significance", ""),
                                    "quality_scores": scores.to_dict(),
                                }
                                if loc["name"] in mini_descs:
                                    attrs["mini_description"] = mini_descs[loc["name"]]
                                self.services.world.add_entity(
                                    self.state.world_db,
                                    name=loc["name"],
                                    entity_type="location",
                                    description=loc.get("description", ""),
                                    attributes=attrs,
                                )
                                added_names.append(loc["name"])
                        # Refresh UI and save
                        self.state.world_db.invalidate_graph_cache()
                        self._refresh_entity_list()
                        if self._graph:
                            self._graph.refresh()
                        self.services.project.save_project(self.state.project)
                        avg_quality = (
                            sum(s.average for _, s in selected) / len(selected) if selected else 0
                        )
                        ui.notify(
                            f"Added {len(selected)} locations (avg quality: {avg_quality:.1f})",
                            type="positive",
                        )
                        # Prompt for relationship generation
                        self._prompt_for_relationships_after_add(added_names)

                    # Show preview dialog
                    self._show_entity_preview_dialog(
                        "location", loc_results, add_selected_locations
                    )
                    return  # Early return - callback handles the rest
                else:
                    # Generate locations via service (original method)
                    logger.info("Calling story service to generate locations...")
                    locations = await run.io_bound(
                        self.services.story.generate_locations, self.state.project, count
                    )
                    logger.info(f"Generated {len(locations)} locations from LLM")
                    # Add to world database
                    added_count = 0
                    for loc in locations:
                        if isinstance(loc, dict) and "name" in loc:
                            self.services.world.add_entity(
                                self.state.world_db,
                                name=loc["name"],
                                entity_type="location",
                                description=loc.get("description", ""),
                                attributes={"significance": loc.get("significance", "")},
                            )
                            added_count += 1
                        else:
                            logger.warning(f"Skipping invalid location: {loc}")
                    logger.info(f"Added {added_count} locations to world database")
                    if notification:
                        notification.dismiss()
                    ui.notify(f"Added {added_count} new locations!", type="positive")

            elif entity_type == "factions":
                if use_quality:
                    # Get existing locations for spatial grounding
                    existing_entities = self.state.world_db.list_entities()
                    existing_locations = [e.name for e in existing_entities if e.type == "location"]
                    logger.info(
                        f"Found {len(existing_locations)} existing locations for faction grounding"
                    )

                    # Generate factions with quality refinement
                    logger.info("Calling world quality service to generate factions...")
                    faction_results = await run.io_bound(
                        self.services.world_quality.generate_factions_with_quality,
                        self.state.project,
                        all_existing_names,
                        count,
                        existing_locations,
                        should_cancel,
                        update_progress,
                    )
                    logger.info(
                        f"Generated {len(faction_results)} factions with quality refinement"
                    )

                    # Check for partial failure and notify user
                    self._notify_partial_failure(
                        len(faction_results), count, "factions", should_cancel
                    )
                    if len(faction_results) == 0:
                        if self._generation_dialog:
                            self._generation_dialog.close()
                        ui.notify("Failed to generate any factions", type="negative")
                        return

                    # Generate mini descriptions for hover tooltips
                    if progress_label:
                        progress_label.text = "Generating hover summaries..."
                    entity_data = [
                        {
                            "name": faction.get("name", ""),
                            "type": "faction",
                            "description": faction.get("description", ""),
                        }
                        for faction, _ in faction_results
                        if isinstance(faction, dict) and faction.get("name")
                    ]
                    mini_descs = await run.io_bound(
                        self.services.world_quality.generate_mini_descriptions_batch,
                        entity_data,
                    )
                    if self._generation_dialog:
                        self._generation_dialog.close()

                    # Define callback to add selected factions
                    def add_selected_factions(selected: list[tuple[Any, Any]]) -> None:
                        """Add selected factions to the world database with location relationships."""
                        if not selected:
                            ui.notify("No factions selected", type="info")
                            return
                        if not self.state.world_db or not self.state.project:
                            ui.notify("No project loaded", type="negative")
                            return
                        added_names = []
                        for faction, scores in selected:
                            if isinstance(faction, dict) and "name" in faction:
                                attrs = {
                                    "leader": faction.get("leader", ""),
                                    "goals": faction.get("goals", []),
                                    "values": faction.get("values", []),
                                    "base_location": faction.get("base_location", ""),
                                    "quality_scores": scores.to_dict(),
                                }
                                if faction["name"] in mini_descs:
                                    attrs["mini_description"] = mini_descs[faction["name"]]
                                faction_entity_id = self.services.world.add_entity(
                                    self.state.world_db,
                                    name=faction["name"],
                                    entity_type="faction",
                                    description=faction.get("description", ""),
                                    attributes=attrs,
                                )
                                added_names.append(faction["name"])
                                # Create relationship to base location if it exists
                                base_loc = faction.get("base_location", "")
                                if base_loc:
                                    location_entity = next(
                                        (
                                            e
                                            for e in existing_entities
                                            if e.name == base_loc and e.type == "location"
                                        ),
                                        None,
                                    )
                                    if location_entity:
                                        self.services.world.add_relationship(
                                            self.state.world_db,
                                            faction_entity_id,
                                            location_entity.id,
                                            "based_in",
                                            f"{faction['name']} is headquartered in {base_loc}",
                                        )
                                        logger.info(
                                            f"Created relationship: {faction['name']} -> based_in -> {base_loc}"
                                        )
                        # Refresh UI and save
                        self.state.world_db.invalidate_graph_cache()
                        self._refresh_entity_list()
                        if self._graph:
                            self._graph.refresh()
                        self.services.project.save_project(self.state.project)
                        avg_quality = (
                            sum(s.average for _, s in selected) / len(selected) if selected else 0
                        )
                        ui.notify(
                            f"Added {len(selected)} factions (avg quality: {avg_quality:.1f})",
                            type="positive",
                        )
                        # Prompt for relationship generation
                        self._prompt_for_relationships_after_add(added_names)

                    # Show preview dialog
                    self._show_entity_preview_dialog(
                        "faction", faction_results, add_selected_factions
                    )
                    return  # Early return - callback handles the rest
                else:
                    if notification:
                        notification.dismiss()
                    ui.notify("Enable Quality Refinement to generate factions", type="warning")
                    return

            elif entity_type == "items":
                if use_quality:
                    # Generate items with quality refinement
                    logger.info("Calling world quality service to generate items...")
                    item_results = await run.io_bound(
                        self.services.world_quality.generate_items_with_quality,
                        self.state.project,
                        all_existing_names,
                        count,
                        should_cancel,
                        update_progress,
                    )
                    logger.info(f"Generated {len(item_results)} items with quality refinement")

                    # Check for partial failure and notify user
                    self._notify_partial_failure(len(item_results), count, "items", should_cancel)
                    if len(item_results) == 0:
                        if self._generation_dialog:
                            self._generation_dialog.close()
                        ui.notify("Failed to generate any items", type="negative")
                        return

                    # Generate mini descriptions for hover tooltips
                    if progress_label:
                        progress_label.text = "Generating hover summaries..."
                    entity_data = [
                        {
                            "name": item.get("name", ""),
                            "type": "item",
                            "description": item.get("description", ""),
                        }
                        for item, _ in item_results
                        if isinstance(item, dict) and item.get("name")
                    ]
                    mini_descs = await run.io_bound(
                        self.services.world_quality.generate_mini_descriptions_batch,
                        entity_data,
                    )
                    if self._generation_dialog:
                        self._generation_dialog.close()

                    # Define callback to add selected items
                    def add_selected_items(selected: list[tuple[Any, Any]]) -> None:
                        """Add selected items to the world database."""
                        if not selected:
                            ui.notify("No items selected", type="info")
                            return
                        if not self.state.world_db or not self.state.project:
                            ui.notify("No project loaded", type="negative")
                            return
                        added_names = []
                        for item, scores in selected:
                            if isinstance(item, dict) and "name" in item:
                                attrs = {
                                    "significance": item.get("significance", ""),
                                    "properties": item.get("properties", []),
                                    "quality_scores": scores.to_dict(),
                                }
                                if item["name"] in mini_descs:
                                    attrs["mini_description"] = mini_descs[item["name"]]
                                self.services.world.add_entity(
                                    self.state.world_db,
                                    name=item["name"],
                                    entity_type="item",
                                    description=item.get("description", ""),
                                    attributes=attrs,
                                )
                                added_names.append(item["name"])
                        # Refresh UI and save
                        self.state.world_db.invalidate_graph_cache()
                        self._refresh_entity_list()
                        if self._graph:
                            self._graph.refresh()
                        self.services.project.save_project(self.state.project)
                        avg_quality = (
                            sum(s.average for _, s in selected) / len(selected) if selected else 0
                        )
                        ui.notify(
                            f"Added {len(selected)} items (avg quality: {avg_quality:.1f})",
                            type="positive",
                        )
                        # Prompt for relationship generation
                        self._prompt_for_relationships_after_add(added_names)

                    # Show preview dialog
                    self._show_entity_preview_dialog("item", item_results, add_selected_items)
                    return  # Early return - callback handles the rest
                else:
                    if notification:
                        notification.dismiss()
                    ui.notify("Enable Quality Refinement to generate items", type="warning")
                    return

            elif entity_type == "concepts":
                if use_quality:
                    # Generate concepts with quality refinement
                    logger.info("Calling world quality service to generate concepts...")
                    concept_results = await run.io_bound(
                        self.services.world_quality.generate_concepts_with_quality,
                        self.state.project,
                        all_existing_names,
                        count,
                        should_cancel,
                        update_progress,
                    )
                    logger.info(
                        f"Generated {len(concept_results)} concepts with quality refinement"
                    )

                    # Check for partial failure and notify user
                    self._notify_partial_failure(
                        len(concept_results), count, "concepts", should_cancel
                    )
                    if len(concept_results) == 0:
                        if self._generation_dialog:
                            self._generation_dialog.close()
                        ui.notify("Failed to generate any concepts", type="negative")
                        return

                    # Generate mini descriptions for hover tooltips
                    if progress_label:
                        progress_label.text = "Generating hover summaries..."
                    entity_data = [
                        {
                            "name": concept.get("name", ""),
                            "type": "concept",
                            "description": concept.get("description", ""),
                        }
                        for concept, _ in concept_results
                        if isinstance(concept, dict) and concept.get("name")
                    ]
                    mini_descs = await run.io_bound(
                        self.services.world_quality.generate_mini_descriptions_batch,
                        entity_data,
                    )
                    if self._generation_dialog:
                        self._generation_dialog.close()

                    # Define callback to add selected concepts
                    def add_selected_concepts(selected: list[tuple[Any, Any]]) -> None:
                        """Add selected concepts to the world database."""
                        if not selected:
                            ui.notify("No concepts selected", type="info")
                            return
                        if not self.state.world_db or not self.state.project:
                            ui.notify("No project loaded", type="negative")
                            return
                        added_names = []
                        for concept, scores in selected:
                            if isinstance(concept, dict) and "name" in concept:
                                attrs = {
                                    "manifestations": concept.get("manifestations", ""),
                                    "quality_scores": scores.to_dict(),
                                }
                                if concept["name"] in mini_descs:
                                    attrs["mini_description"] = mini_descs[concept["name"]]
                                self.services.world.add_entity(
                                    self.state.world_db,
                                    name=concept["name"],
                                    entity_type="concept",
                                    description=concept.get("description", ""),
                                    attributes=attrs,
                                )
                                added_names.append(concept["name"])
                        # Refresh UI and save
                        self.state.world_db.invalidate_graph_cache()
                        self._refresh_entity_list()
                        if self._graph:
                            self._graph.refresh()
                        self.services.project.save_project(self.state.project)
                        avg_quality = (
                            sum(s.average for _, s in selected) / len(selected) if selected else 0
                        )
                        ui.notify(
                            f"Added {len(selected)} concepts (avg quality: {avg_quality:.1f})",
                            type="positive",
                        )
                        # Prompt for relationship generation
                        self._prompt_for_relationships_after_add(added_names)

                    # Show preview dialog
                    self._show_entity_preview_dialog(
                        "concept", concept_results, add_selected_concepts
                    )
                    return  # Early return - callback handles the rest
                else:
                    if notification:
                        notification.dismiss()
                    ui.notify("Enable Quality Refinement to generate concepts", type="warning")
                    return

            elif entity_type == "relationships":
                # Get existing entities and relationships
                entities = self.state.world_db.list_entities()
                entity_names = [e.name for e in entities]
                logger.info(f"Found {len(entities)} existing entities: {entity_names}")

                # Get existing relationships - look up entity names from IDs
                existing_rels = []
                for rel in self.state.world_db.list_relationships():
                    source = self.services.world.get_entity(self.state.world_db, rel.source_id)
                    target = self.services.world.get_entity(self.state.world_db, rel.target_id)
                    if source and target:
                        existing_rels.append((source.name, target.name))
                logger.info(f"Found {len(existing_rels)} existing relationships")

                if len(entity_names) < 2:
                    logger.warning("Cannot generate relationships: need at least 2 entities")
                    if self._generation_dialog:
                        self._generation_dialog.close()
                    elif notification:
                        notification.dismiss()
                    ui.notify("Need at least 2 entities to create relationships", type="warning")
                    return

                if use_quality:
                    # Generate relationships with quality refinement
                    logger.info("Calling world quality service to generate relationships...")
                    rel_results = await run.io_bound(
                        self.services.world_quality.generate_relationships_with_quality,
                        self.state.project,
                        entity_names,
                        existing_rels,
                        count,
                        should_cancel,
                        update_progress,
                    )
                    logger.info(
                        f"Generated {len(rel_results)} relationships with quality refinement"
                    )

                    # Check for partial failure and notify user
                    self._notify_partial_failure(
                        len(rel_results), count, "relationships", should_cancel
                    )
                    if len(rel_results) == 0:
                        if self._generation_dialog:
                            self._generation_dialog.close()
                        ui.notify("Failed to generate any relationships", type="negative")
                        return
                    if self._generation_dialog:
                        self._generation_dialog.close()

                    # Show preview dialog with callback to add selected relationships
                    self._show_entity_preview_dialog(
                        "relationship",
                        rel_results,
                        lambda selected: self._add_quality_relationships(selected, entities),
                    )
                    return  # Early return - callback handles the rest
                else:
                    # Generate relationships via service (original method)
                    logger.info("Calling story service to generate relationships...")
                    relationships = await run.io_bound(
                        self.services.story.generate_relationships,
                        self.state.project,
                        entity_names,
                        existing_rels,
                        count,
                    )
                    logger.info(f"Generated {len(relationships)} relationships from LLM")

                    # Add to world database
                    added = self._add_relationships_from_dicts(
                        self.services.world,
                        self.state.world_db,
                        entities,
                        relationships,
                    )
                    logger.info(f"Added {added} relationships to world database")
                    if notification:
                        notification.dismiss()
                    ui.notify(f"Added {added} new relationships!", type="positive")

            # Finalize: refresh UI, save project
            self._finalize_generation(entity_type)

        except WorldGenerationError as e:
            if self._generation_dialog:
                self._generation_dialog.close()
            elif notification:
                notification.dismiss()
            logger.error(f"World generation failed for {entity_type}: {e}")
            ui.notify(f"Generation failed: {e}", type="negative", close_button=True, timeout=10)
        except Exception as e:
            if self._generation_dialog:
                self._generation_dialog.close()
            elif notification:
                notification.dismiss()
            logger.exception(f"Unexpected error generating {entity_type}: {e}")
            ui.notify(f"Error: {e}", type="negative")

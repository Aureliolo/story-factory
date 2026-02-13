"""Entity-type-specific generation functions for the World page.

Contains generators for factions, items, concepts, relationships,
and the generate_relationships_for_entities helper.
"""

import logging
import threading
from typing import Any

from nicegui import ui

from src.ui.pages.world._gen_dialogs import (
    create_progress_dialog,
    get_all_entity_names,
    get_entity_names_by_type,
    make_update_progress,
    notify_partial_failure,
    prompt_for_relationships_after_add,
    show_entity_preview_dialog,
)

logger = logging.getLogger(__name__)


async def _generate_factions(
    page,
    count,
    use_quality,
    all_existing_names,
    should_cancel,
    update_progress,
    progress_label,
    notification,
) -> None:
    """Generate factions (quality only).

    Args:
        page: WorldPage instance.
        count: Number to generate.
        use_quality: Whether quality refinement is enabled.
        all_existing_names: Existing entity names to avoid duplicates.
        should_cancel: Cancel check callable.
        update_progress: Progress update callback.
        progress_label: Progress label widget.
        notification: Notification widget.
    """
    from nicegui import run

    if use_quality:
        existing_entities = page.state.world_db.list_entities()
        existing_locations = [e.name for e in existing_entities if e.type == "location"]
        faction_names = get_entity_names_by_type(page, "faction")
        logger.info(f"Found {len(existing_locations)} existing locations for faction grounding")

        logger.info("Calling world quality service to generate factions...")
        faction_results = await run.io_bound(
            page.services.world_quality.generate_factions_with_quality,
            page.state.project,
            faction_names,
            count,
            existing_locations,
            should_cancel,
            update_progress,
        )
        logger.info(f"Generated {len(faction_results)} factions with quality refinement")

        notify_partial_failure(len(faction_results), count, "factions", should_cancel)
        if len(faction_results) == 0:
            if page._generation_dialog:
                page._generation_dialog.close()
            ui.notify("Failed to generate any factions", type="negative")
            return

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
            page.services.world_quality.generate_mini_descriptions_batch,
            entity_data,
        )
        if page._generation_dialog:
            page._generation_dialog.close()

        def add_selected_factions(selected: list[tuple[Any, Any]]) -> None:
            """Add selected factions to the world database with location relationships."""
            if not selected:
                ui.notify("No factions selected", type="info")
                return
            if not page.state.world_db or not page.state.project:
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
                    faction_entity_id = page.services.world.add_entity(
                        page.state.world_db,
                        name=faction["name"],
                        entity_type="faction",
                        description=faction.get("description", ""),
                        attributes=attrs,
                    )
                    added_names.append(faction["name"])
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
                            page.services.world.add_relationship(
                                page.state.world_db,
                                faction_entity_id,
                                location_entity.id,
                                "based_in",
                                f"{faction['name']} is headquartered in {base_loc}",
                            )
                            logger.info(
                                f"Created relationship: {faction['name']} -> based_in -> {base_loc}"
                            )
            page.state.world_db.invalidate_graph_cache()
            page._refresh_entity_list()
            if page._graph:
                page._graph.refresh()
            page.services.project.save_project(page.state.project)
            avg_quality = sum(s.average for _, s in selected) / len(selected) if selected else 0
            ui.notify(
                f"Added {len(selected)} factions (avg quality: {avg_quality:.1f})",
                type="positive",
            )
            prompt_for_relationships_after_add(page, added_names)

        show_entity_preview_dialog(page, "faction", faction_results, add_selected_factions)
        return
    else:
        if notification:
            notification.dismiss()
        ui.notify("Enable Quality Refinement to generate factions", type="warning")
        return


async def _generate_items(
    page,
    count,
    use_quality,
    all_existing_names,
    should_cancel,
    update_progress,
    progress_label,
    notification,
) -> None:
    """Generate items (quality only).

    Args:
        page: WorldPage instance.
        count: Number to generate.
        use_quality: Whether quality refinement is enabled.
        all_existing_names: Existing entity names to avoid duplicates.
        should_cancel: Cancel check callable.
        update_progress: Progress update callback.
        progress_label: Progress label widget.
        notification: Notification widget.
    """
    from nicegui import run

    if use_quality:
        item_names = get_entity_names_by_type(page, "item")
        logger.info("Calling world quality service to generate items...")
        item_results = await run.io_bound(
            page.services.world_quality.generate_items_with_quality,
            page.state.project,
            item_names,
            count,
            should_cancel,
            update_progress,
        )
        logger.info(f"Generated {len(item_results)} items with quality refinement")

        notify_partial_failure(len(item_results), count, "items", should_cancel)
        if len(item_results) == 0:
            if page._generation_dialog:
                page._generation_dialog.close()
            ui.notify("Failed to generate any items", type="negative")
            return

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
            page.services.world_quality.generate_mini_descriptions_batch,
            entity_data,
        )
        if page._generation_dialog:
            page._generation_dialog.close()

        def add_selected_items(selected: list[tuple[Any, Any]]) -> None:
            """Add selected items to the world database."""
            if not selected:
                ui.notify("No items selected", type="info")
                return
            if not page.state.world_db or not page.state.project:
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
                    page.services.world.add_entity(
                        page.state.world_db,
                        name=item["name"],
                        entity_type="item",
                        description=item.get("description", ""),
                        attributes=attrs,
                    )
                    added_names.append(item["name"])
            page.state.world_db.invalidate_graph_cache()
            page._refresh_entity_list()
            if page._graph:
                page._graph.refresh()
            page.services.project.save_project(page.state.project)
            avg_quality = sum(s.average for _, s in selected) / len(selected) if selected else 0
            ui.notify(
                f"Added {len(selected)} items (avg quality: {avg_quality:.1f})",
                type="positive",
            )
            prompt_for_relationships_after_add(page, added_names)

        show_entity_preview_dialog(page, "item", item_results, add_selected_items)
        return
    else:
        if notification:
            notification.dismiss()
        ui.notify("Enable Quality Refinement to generate items", type="warning")
        return


async def _generate_concepts(
    page,
    count,
    use_quality,
    all_existing_names,
    should_cancel,
    update_progress,
    progress_label,
    notification,
) -> None:
    """Generate concepts (quality only).

    Args:
        page: WorldPage instance.
        count: Number to generate.
        use_quality: Whether quality refinement is enabled.
        all_existing_names: Existing entity names to avoid duplicates.
        should_cancel: Cancel check callable.
        update_progress: Progress update callback.
        progress_label: Progress label widget.
        notification: Notification widget.
    """
    from nicegui import run

    if use_quality:
        concept_names = get_entity_names_by_type(page, "concept")
        logger.info("Calling world quality service to generate concepts...")
        concept_results = await run.io_bound(
            page.services.world_quality.generate_concepts_with_quality,
            page.state.project,
            concept_names,
            count,
            should_cancel,
            update_progress,
        )
        logger.info(f"Generated {len(concept_results)} concepts with quality refinement")

        notify_partial_failure(len(concept_results), count, "concepts", should_cancel)
        if len(concept_results) == 0:
            if page._generation_dialog:
                page._generation_dialog.close()
            ui.notify("Failed to generate any concepts", type="negative")
            return

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
            page.services.world_quality.generate_mini_descriptions_batch,
            entity_data,
        )
        if page._generation_dialog:
            page._generation_dialog.close()

        def add_selected_concepts(selected: list[tuple[Any, Any]]) -> None:
            """Add selected concepts to the world database."""
            if not selected:
                ui.notify("No concepts selected", type="info")
                return
            if not page.state.world_db or not page.state.project:
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
                    page.services.world.add_entity(
                        page.state.world_db,
                        name=concept["name"],
                        entity_type="concept",
                        description=concept.get("description", ""),
                        attributes=attrs,
                    )
                    added_names.append(concept["name"])
            page.state.world_db.invalidate_graph_cache()
            page._refresh_entity_list()
            if page._graph:
                page._graph.refresh()
            page.services.project.save_project(page.state.project)
            avg_quality = sum(s.average for _, s in selected) / len(selected) if selected else 0
            ui.notify(
                f"Added {len(selected)} concepts (avg quality: {avg_quality:.1f})",
                type="positive",
            )
            prompt_for_relationships_after_add(page, added_names)

        show_entity_preview_dialog(page, "concept", concept_results, add_selected_concepts)
        return
    else:
        if notification:
            notification.dismiss()
        ui.notify("Enable Quality Refinement to generate concepts", type="warning")
        return


async def _generate_relationships(
    page,
    count,
    use_quality,
    all_existing_names,
    should_cancel,
    update_progress,
    notification,
) -> None:
    """Generate relationships (quality or standard).

    Args:
        page: WorldPage instance.
        count: Number to generate.
        use_quality: Whether quality refinement is enabled.
        all_existing_names: Existing entity names to avoid duplicates.
        should_cancel: Cancel check callable.
        update_progress: Progress update callback.
        notification: Notification widget.
    """
    from nicegui import run

    # Get existing entities and relationships
    entities = page.state.world_db.list_entities()
    entity_names = [e.name for e in entities]
    logger.info(f"Found {len(entities)} existing entities: {entity_names}")

    # Get existing relationships as 3-tuples (source_name, target_name, relation_type)
    existing_rels: list[tuple[str, str, str]] = []
    for rel in page.state.world_db.list_relationships():
        source = page.services.world.get_entity(page.state.world_db, rel.source_id)
        target = page.services.world.get_entity(page.state.world_db, rel.target_id)
        if source and target:
            existing_rels.append((source.name, target.name, rel.relation_type))
    logger.info(f"Found {len(existing_rels)} existing relationships")

    if len(entity_names) < 2:
        logger.warning("Cannot generate relationships: need at least 2 entities")
        if page._generation_dialog:
            page._generation_dialog.close()
        elif notification:
            notification.dismiss()
        ui.notify("Need at least 2 entities to create relationships", type="warning")
        return

    if use_quality:
        logger.info("Calling world quality service to generate relationships...")
        rel_results = await run.io_bound(
            page.services.world_quality.generate_relationships_with_quality,
            page.state.project,
            entity_names,
            existing_rels,
            count,
            should_cancel,
            update_progress,
        )
        logger.info(f"Generated {len(rel_results)} relationships with quality refinement")

        notify_partial_failure(len(rel_results), count, "relationships", should_cancel)
        if len(rel_results) == 0:
            if page._generation_dialog:
                page._generation_dialog.close()
            ui.notify("Failed to generate any relationships", type="negative")
            return
        if page._generation_dialog:
            page._generation_dialog.close()

        def add_selected_relationships(selected: list[tuple[Any, Any]]) -> None:
            """Add selected relationships to the world database."""
            if not selected:
                ui.notify("No relationships selected", type="info")
                return
            if not page.state.world_db or not page.state.project:
                ui.notify("No project loaded", type="negative")
                return
            added = 0
            for rel_data, scores in selected:
                if isinstance(rel_data, dict) and "source" in rel_data and "target" in rel_data:
                    source_entity = next(
                        (e for e in entities if e.name == rel_data["source"]), None
                    )
                    target_entity = next(
                        (e for e in entities if e.name == rel_data["target"]), None
                    )
                    if source_entity and target_entity:
                        rel_id = page.services.world.add_relationship(
                            page.state.world_db,
                            source_entity.id,
                            target_entity.id,
                            rel_data.get("relation_type", "knows"),
                            rel_data.get("description", ""),
                        )
                        page.state.world_db.update_relationship(
                            relationship_id=rel_id,
                            attributes={"quality_scores": scores.to_dict()},
                        )
                        added += 1
            page.state.world_db.invalidate_graph_cache()
            page._refresh_entity_list()
            if page._graph:
                page._graph.refresh()
            page.services.project.save_project(page.state.project)
            avg_quality = sum(s.average for _, s in selected) / len(selected) if selected else 0
            ui.notify(
                f"Added {added} relationships (avg quality: {avg_quality:.1f})",
                type="positive",
            )

        show_entity_preview_dialog(page, "relationship", rel_results, add_selected_relationships)
        return
    else:
        logger.info("Calling story service to generate relationships...")
        relationships = await run.io_bound(
            page.services.story.generate_relationships,
            page.state.project,
            entity_names,
            existing_rels,
            count,
        )
        logger.info(f"Generated {len(relationships)} relationships from LLM")

        added = 0
        for rel in relationships:
            if isinstance(rel, dict) and "source" in rel and "target" in rel:
                source_entity = next((e for e in entities if e.name == rel["source"]), None)
                target_entity = next((e for e in entities if e.name == rel["target"]), None)
                if source_entity and target_entity:
                    page.services.world.add_relationship(
                        page.state.world_db,
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
        logger.info(f"Added {added} relationships to world database")
        if notification:
            notification.dismiss()
        ui.notify(f"Added {added} new relationships!", type="positive")


async def generate_relationships_for_entities(
    page, entity_names: list[str], count_per_entity: int
) -> None:
    """Generate relationships for specific entities.

    Args:
        page: WorldPage instance.
        entity_names: Names of entities to generate relationships for.
        count_per_entity: Number of relationships to generate per entity.
    """
    if not page.state.project or not page.state.world_db:
        ui.notify("No project loaded", type="negative")
        return

    logger.info(
        f"Generating {count_per_entity} relationships for each of {len(entity_names)} entities"
    )

    use_quality = (
        page.state.quality_refinement_enabled and page.services.settings.world_quality_enabled
    )

    all_entity_names = get_all_entity_names(page)
    existing_rels: list[tuple[str, str, str]] = [
        (r.source_id, r.target_id, r.relation_type)
        for r in page.state.world_db.list_relationships()
    ]
    total_count = len(entity_names) * count_per_entity

    page._generation_cancel_event = threading.Event()

    def should_cancel() -> bool:
        """Check if generation should be cancelled."""
        return page._generation_cancel_event is not None and page._generation_cancel_event.is_set()

    if use_quality:
        progress_label, progress_bar, eta_label = create_progress_dialog(
            page, "relationships", total_count
        )
        update_progress = make_update_progress(progress_label, progress_bar, eta_label)
        notification = None
    else:
        notification = ui.notification(
            message=f"Generating relationships for {len(entity_names)} entities...",
            spinner=True,
            timeout=None,
        )
        update_progress = None

    page.state.begin_background_task("generate_relationships_for_entities")
    try:
        from nicegui import run

        if use_quality:
            results = await run.io_bound(
                page.services.world_quality.generate_relationships_with_quality,
                page.state.project,
                all_entity_names,
                existing_rels,
                total_count,
                should_cancel,
                update_progress,
            )

            notify_partial_failure(len(results), total_count, "relationships", should_cancel)

            if len(results) == 0:
                if page._generation_dialog:
                    page._generation_dialog.close()
                elif notification:
                    notification.dismiss()
                ui.notify("Failed to generate any relationships", type="negative")
                return

            if page._generation_dialog:
                page._generation_dialog.close()
            elif notification:
                notification.dismiss()

            def add_selected_relationships(selected: list[tuple[Any, Any]]) -> None:
                """Add selected relationships to the world database from the preview."""
                if not selected:
                    ui.notify("No relationships selected", type="info")
                    return
                if not page.state.world_db:
                    ui.notify("No world database", type="negative")
                    return

                entities = page.state.world_db.list_entities()
                added_count = 0

                for rel_data, _scores in selected:
                    source_name = rel_data.get("source", "")
                    target_name = rel_data.get("target", "")
                    rel_type = rel_data.get("relation_type", "related_to")
                    desc = rel_data.get("description", "")

                    source_entity = next((e for e in entities if e.name == source_name), None)
                    target_entity = next((e for e in entities if e.name == target_name), None)

                    if source_entity and target_entity:
                        page.services.world.add_relationship(
                            page.state.world_db,
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

                page.state.world_db.invalidate_graph_cache()
                page._refresh_entity_list()
                if page._graph:
                    page._graph.refresh()
                ui.notify(
                    f"Added {added_count} relationships",
                    type="positive",
                )

            show_entity_preview_dialog(page, "relationship", results, add_selected_relationships)
        else:
            if notification:
                notification.dismiss()
            ui.notify(
                "Relationship generation requires quality refinement to be enabled",
                type="warning",
            )

    except Exception as e:
        if page._generation_dialog:
            page._generation_dialog.close()
        elif notification:
            notification.dismiss()
        logger.exception(f"Error generating relationships: {e}")
        ui.notify(f"Error: {e}", type="negative")
    finally:
        page.state.end_background_task("generate_relationships_for_entities")

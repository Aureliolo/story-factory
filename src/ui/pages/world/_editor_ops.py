"""Entity editor operations for the World page.

Contains regeneration, mutation, and deletion functions for entities:
show_regenerate_dialog, execute_regenerate, refine_entity, regenerate_full,
regenerate_with_guidance, confirm_delete_entity, delete_entity.
"""

import logging
from typing import TYPE_CHECKING

from nicegui import ui

from src.memory.entities import Entity
from src.ui.pages.world._editor import refresh_entity_editor
from src.ui.state import ActionType, UndoAction

if TYPE_CHECKING:
    from . import WorldPage

logger = logging.getLogger(__name__)


async def show_regenerate_dialog(page: WorldPage) -> None:
    """Show dialog for regenerating the selected entity.

    Args:
        page: WorldPage instance.
    """
    if not page.state.selected_entity_id or not page.state.world_db:
        ui.notify("No entity selected", type="warning")
        return

    entity = page.services.world.get_entity(page.state.world_db, page.state.selected_entity_id)
    if not entity:
        ui.notify("Entity not found", type="negative")
        return

    # Get relationship count
    relationships = page.state.world_db.get_relationships(entity.id)
    rel_count = len(relationships)

    with ui.dialog() as dialog, ui.card().classes("w-96 p-4"):
        ui.label(f"Regenerate: {entity.name}").classes("text-lg font-semibold")

        # Mode selection
        mode_ref = {"value": "refine"}
        with ui.column().classes("w-full gap-2 mt-4"):
            ui.label("Regeneration Mode").classes("text-sm font-medium")
            mode_radio = ui.radio(
                options={
                    "refine": "Refine existing (improve weak areas)",
                    "full": "Full regenerate (create new version)",
                    "guided": "Regenerate with guidance",
                },
                value="refine",
                on_change=lambda e: mode_ref.update({"value": e.value}),
            ).classes("w-full")

        # Guidance input (visible only for guided mode)
        guidance_container = ui.column().classes("w-full mt-2")
        guidance_input_ref: dict = {"input": None}

        def update_guidance_visibility() -> None:
            """Update guidance input visibility based on mode."""
            guidance_container.clear()
            if mode_ref["value"] == "guided":
                with guidance_container:
                    guidance_input_ref["input"] = (
                        ui.textarea(
                            label="Guidance",
                            placeholder="Describe how you want this entity to change...",
                        )
                        .classes("w-full")
                        .props("rows=3")
                    )

        mode_radio.on("update:model-value", lambda _: update_guidance_visibility())
        update_guidance_visibility()

        # Relationship warning
        if rel_count > 0:
            with ui.row().classes("w-full items-center gap-2 p-2 bg-amber-900 rounded mt-4"):
                ui.icon("warning", color="amber")
                ui.label(f"{rel_count} relationship(s) will be preserved.").classes("text-sm")

        # Quality info if available
        quality_scores = entity.attributes.get("quality_scores") if entity.attributes else None
        if quality_scores and isinstance(quality_scores, dict):
            avg = quality_scores.get("average", 0)
            with ui.row().classes("w-full items-center gap-2 mt-2"):
                ui.label(f"Current quality: {avg:.1f}/10").classes("text-sm text-gray-400")

        # Actions
        with ui.row().classes("w-full justify-end gap-2 mt-4"):
            ui.button("Cancel", on_click=dialog.close).props("flat")

            async def do_regenerate() -> None:
                """Execute regeneration."""
                guidance = None
                if guidance_input_ref.get("input"):
                    guidance = guidance_input_ref["input"].value
                dialog.close()
                await execute_regenerate(page, entity, mode_ref["value"], guidance)

            ui.button("Regenerate", on_click=do_regenerate).props("color=primary")

    dialog.open()


async def execute_regenerate(
    page: WorldPage, entity: Entity, mode: str, guidance: str | None
) -> None:
    """Execute entity regeneration.

    Args:
        page: WorldPage instance.
        entity: Entity to regenerate.
        mode: Regeneration mode (refine, full, guided).
        guidance: Optional guidance text for guided mode.
    """
    logger.info(f"Regenerating entity {entity.id} ({entity.name}) mode={mode}")

    if not page.state.world_db or not page.state.project:
        ui.notify("No project loaded", type="negative")
        return

    # Show progress indicator
    progress_dialog = None
    try:
        with ui.dialog() as progress_dialog, ui.card().classes("w-64 items-center p-4"):
            ui.label("Regenerating...").classes("text-center")
            ui.spinner(size="lg")
        progress_dialog.open()

        # Perform regeneration based on mode
        result = None
        if mode == "refine":
            result = await refine_entity(page, entity)
        elif mode == "guided":
            if guidance and guidance.strip():
                result = await regenerate_with_guidance(page, entity, guidance)
            else:
                logger.warning(
                    "Guided regeneration requested for entity %s without guidance; aborting.",
                    entity.id,
                )
                if progress_dialog:
                    progress_dialog.close()
                ui.notify("Guidance text is required for guided regeneration.", type="warning")
                return
        else:
            result = await regenerate_full(page, entity)

        if result:
            # Update entity in database (relationships preserved since ID unchanged)
            new_name = result.get("name", entity.name)
            new_description = result.get("description", entity.description)
            new_attributes = {**(entity.attributes or {}), **result.get("attributes", {})}

            page.services.world.update_entity(
                page.state.world_db,
                entity.id,
                name=new_name,
                description=new_description,
                attributes=new_attributes,
            )

            # Invalidate graph cache for fresh tooltips
            page.state.world_db.invalidate_graph_cache()

            # Refresh UI
            page._refresh_entity_list()
            refresh_entity_editor(page)
            if page._graph:
                page._graph.refresh()

            ui.notify(f"Regenerated {new_name}", type="positive")
        else:
            ui.notify("Regeneration failed - no result returned", type="negative")

    except Exception as e:
        logger.exception(f"Regeneration failed: {e}")
        ui.notify(f"Error: {e}", type="negative")
    finally:
        if progress_dialog:
            progress_dialog.close()


async def refine_entity(page: WorldPage, entity: Entity) -> dict | None:
    """Refine entity using quality service.

    Args:
        page: WorldPage instance.
        entity: Entity to refine.

    Returns:
        Dictionary with refined entity data, or None on failure.
    """
    if not page.state.project:
        return None

    try:
        # Use WorldQualityService to refine the entity
        refined = await page.services.world_quality.refine_entity(
            entity=entity,
            story_brief=page.state.project.brief,
        )
        return refined
    except Exception as e:
        logger.exception(f"Failed to refine entity: {e}")
        return None


async def regenerate_full(page: WorldPage, entity: Entity) -> dict | None:
    """Fully regenerate entity.

    Args:
        page: WorldPage instance.
        entity: Entity to regenerate.

    Returns:
        Dictionary with new entity data, or None on failure.
    """
    if not page.state.project:
        return None

    try:
        # Use WorldQualityService to regenerate based on entity type
        regenerated = await page.services.world_quality.regenerate_entity(
            entity=entity,
            story_brief=page.state.project.brief,
        )
        return regenerated
    except Exception as e:
        logger.exception(f"Failed to regenerate entity: {e}")
        return None


async def regenerate_with_guidance(page: WorldPage, entity: Entity, guidance: str) -> dict | None:
    """Regenerate with user guidance.

    Args:
        page: WorldPage instance.
        entity: Entity to regenerate.
        guidance: User-provided guidance text.

    Returns:
        Dictionary with regenerated entity data, or None on failure.
    """
    if not page.state.project:
        return None

    try:
        # Use WorldQualityService with custom instructions
        regenerated = await page.services.world_quality.regenerate_entity(
            entity=entity,
            story_brief=page.state.project.brief,
            custom_instructions=guidance,
        )
        return regenerated
    except Exception as e:
        logger.exception(f"Failed to regenerate entity with guidance: {e}")
        return None


def confirm_delete_entity(page: WorldPage) -> None:
    """Show confirmation dialog before deleting entity.

    Args:
        page: WorldPage instance.
    """
    if not page.state.selected_entity_id or not page.state.world_db:
        return

    try:
        # Get entity name for better UX
        entity = page.services.world.get_entity(
            page.state.world_db,
            page.state.selected_entity_id,
        )
        entity_name = entity.name if entity else "this entity"

        # Count attached relationships
        all_rels = page.state.world_db.list_relationships()
        attached_rels = [
            r
            for r in all_rels
            if r.source_id == page.state.selected_entity_id
            or r.target_id == page.state.selected_entity_id
        ]
        rel_count = len(attached_rels)

        # Build message with relationship info
        message = f'Are you sure you want to delete "{entity_name}"?'
        if rel_count > 0:
            rel_word = "relationship" if rel_count == 1 else "relationships"
            message += f"\n\nThis will also delete {rel_count} attached {rel_word}."
        message += "\n\nThis action cannot be undone."

        # Custom dialog with more info
        with ui.dialog() as dialog, ui.card().classes("p-4 min-w-[400px]"):
            ui.label("Delete Entity?").classes("text-lg font-bold text-red-600")
            ui.label(message).classes("text-gray-400 whitespace-pre-line mt-2")

            if rel_count > 0:
                with ui.expansion("Show affected relationships", icon="link").classes(
                    "w-full mt-2"
                ):
                    for rel in attached_rels[:10]:  # Show max 10
                        source = page.services.world.get_entity(page.state.world_db, rel.source_id)
                        target = page.services.world.get_entity(page.state.world_db, rel.target_id)
                        src_name = source.name if source else "?"
                        tgt_name = target.name if target else "?"
                        ui.label(f"{src_name} → {tgt_name} ({rel.relation_type})").classes(
                            "text-sm text-gray-500"
                        )
                    if rel_count > 10:
                        ui.label(f"... and {rel_count - 10} more").classes(
                            "text-sm text-gray-400 italic"
                        )

            def _do_delete() -> None:
                """Close the dialog and delete the selected entity."""
                dialog.close()
                delete_entity(page)

            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button(
                    "Delete",
                    on_click=_do_delete,
                    icon="delete",
                ).props("color=negative")

        dialog.open()

    except Exception as e:
        logger.exception("Error showing delete confirmation")
        ui.notify(f"Error: {e}", type="negative")


def delete_entity(page: WorldPage) -> None:
    """Delete the selected entity.

    Args:
        page: WorldPage instance.
    """
    if not page.state.selected_entity_id or not page.state.world_db:
        return

    try:
        # Get entity data for inverse (restore) operation
        entity = page.services.world.get_entity(page.state.world_db, page.state.selected_entity_id)
        if not entity:
            ui.notify("Entity not found", type="negative")
            return

        entity_id = page.state.selected_entity_id

        page.services.world.delete_entity(
            page.state.world_db,
            entity_id,
        )

        # Record action for undo
        page.state.record_action(
            UndoAction(
                action_type=ActionType.DELETE_ENTITY,
                data={"entity_id": entity_id},
                inverse_data={
                    "type": entity.type,
                    "name": entity.name,
                    "description": entity.description,
                    "attributes": entity.attributes,
                },
            )
        )

        page.state.select_entity(None)
        page._refresh_entity_list()
        refresh_entity_editor(page)
        if page._graph:
            page._graph.refresh()
        page._update_undo_redo_buttons()
        ui.notify("Entity deleted", type="positive")
    except Exception as e:
        logger.exception(f"Failed to delete entity {page.state.selected_entity_id}")
        ui.notify(f"Error: {e}", type="negative")

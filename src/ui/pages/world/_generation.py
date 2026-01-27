"""World generation toolbar and rebuild/clear functions for the World page."""

import logging

from nicegui import ui

from src.ui.components.build_dialog import show_build_structure_dialog

logger = logging.getLogger("src.ui.pages.world._generation")


def build_generation_toolbar(page) -> None:
    """Build the world generation toolbar with readiness score and action buttons.

    Args:
        page: WorldPage instance.
    """
    if not page.state.world_db:
        return

    # Count entities
    char_count = page.state.world_db.count_entities("character")
    loc_count = page.state.world_db.count_entities("location")
    rel_count = len(page.state.world_db.list_relationships())

    # Calculate simple readiness score
    # Based on: characters (weight 3), locations (weight 2), relationships (weight 1)
    target_chars = 5  # Minimum recommended
    target_locs = 3
    target_rels = 8

    char_score = min(100, (char_count / target_chars) * 100)
    loc_score = min(100, (loc_count / target_locs) * 100)
    rel_score = min(100, (rel_count / target_rels) * 100)

    readiness = int(char_score * 0.4 + loc_score * 0.3 + rel_score * 0.3)
    readiness_color = "green" if readiness >= 80 else "orange" if readiness >= 50 else "red"
    readiness_text = (
        "Ready to write!"
        if readiness >= 80
        else "Needs more content"
        if readiness >= 50
        else "World is sparse"
    )

    with ui.row().classes("w-full items-center gap-4 px-4 pt-4 pb-2"):
        # Readiness indicator
        with ui.card().classes("p-3"):
            with ui.row().classes("items-center gap-3"):
                ui.circular_progress(
                    value=readiness / 100,
                    show_value=True,
                    size="lg",
                    color=readiness_color,
                )
                with ui.column().classes("gap-0"):
                    ui.label("World Readiness").classes("text-sm font-medium")
                    ui.label(readiness_text).classes(
                        f"text-xs text-{readiness_color}-600 dark:text-{readiness_color}-400"
                    )

        # Quality refinement toggle
        with ui.row().classes("items-center gap-2"):
            ui.switch(
                "Quality Refinement",
                value=page.state.quality_refinement_enabled,
                on_change=lambda e: setattr(page.state, "quality_refinement_enabled", e.value),
            ).tooltip(
                "When enabled, entities are iteratively refined until they meet quality standards"
            )
            ui.button(
                icon="settings",
                on_click=page._show_quality_settings_dialog,
            ).props("flat dense").tooltip("Quality settings")

        ui.space()

        # Generation buttons - first row
        with ui.row().classes("gap-2 flex-wrap"):
            ui.button(
                "Characters",
                on_click=lambda: page._show_generate_dialog("characters"),
                icon="person_add",
            ).props("outline dense").classes("text-green-600").tooltip("Add more characters")

            ui.button(
                "Locations",
                on_click=lambda: page._show_generate_dialog("locations"),
                icon="add_location",
            ).props("outline dense").classes("text-blue-600").tooltip("Add more locations")

            ui.button(
                "Factions",
                on_click=lambda: page._show_generate_dialog("factions"),
                icon="groups",
            ).props("outline dense").classes("text-amber-600").tooltip("Add factions/organizations")

            ui.button(
                "Items",
                on_click=lambda: page._show_generate_dialog("items"),
                icon="category",
            ).props("outline dense").classes("text-cyan-600").tooltip("Add significant items")

            ui.button(
                "Concepts",
                on_click=lambda: page._show_generate_dialog("concepts"),
                icon="lightbulb",
            ).props("outline dense").classes("text-pink-600").tooltip("Add thematic concepts")

            ui.button(
                "Relationships",
                on_click=lambda: page._generate_more("relationships"),
                icon="link",
            ).props("outline dense").classes("text-purple-600").tooltip("Add relationships")

        ui.separator().props("vertical")

        # Import from text button
        ui.button(
            "Import from Text",
            on_click=page._show_import_wizard,
            icon="upload_file",
        ).props("outline color=primary").tooltip("Extract entities from existing story text")

        # Check if chapters have written content - block destructive actions if so
        has_written_content = (
            page.state.project
            and page.state.project.chapters
            and any(c.content for c in page.state.project.chapters)
        )

        # Regenerate button (dangerous action) - only show if no written content
        if not has_written_content:
            ui.button(
                "Rebuild World",
                on_click=lambda: confirm_regenerate(page),
                icon="refresh",
            ).props("outline color=negative").tooltip(
                "Rebuild all entities and relationships (only available before writing)"
            )

            # Clear World button - only show if no story content written yet
            ui.button(
                "Clear World",
                on_click=lambda: confirm_clear_world(page),
                icon="delete_sweep",
            ).props("outline color=warning").tooltip(
                "Remove all entities and relationships (only available before writing)"
            )

    # Build Story Structure button - centered, only show if no chapters yet
    has_chapters = page.state.project and page.state.project.chapters
    if not has_chapters:
        with ui.row().classes("w-full justify-center mt-4"):
            ui.button(
                "Build Story Structure",
                on_click=lambda: build_structure(page),
                icon="auto_fix_high",
            ).props("color=primary size=lg").tooltip(
                "Generate characters, locations, plot points, and chapter outlines"
            )


def confirm_regenerate(page) -> None:
    """Show confirmation dialog before regenerating world.

    Args:
        page: WorldPage instance.
    """
    logger.info("Rebuild World button clicked - showing confirmation dialog")

    if not page.state.project or not page.state.world_db:
        ui.notify("No project available", type="negative")
        return

    entity_count = page.state.world_db.count_entities()
    rel_count = len(page.state.world_db.list_relationships())
    chapter_count = len(page.state.project.chapters)
    char_count = len(page.state.project.characters)

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
            """Close the confirmation dialog and trigger a full rebuild."""
            dialog.close()
            await build_structure(page, rebuild=True)

        with ui.row().classes("w-full justify-end gap-2 mt-4"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button(
                "Continue to Rebuild",
                on_click=confirm_and_build,
                icon="refresh",
            ).props("color=negative")

    dialog.open()


async def build_structure(page, rebuild: bool = False) -> None:
    """Build story structure using the shared dialog.

    Args:
        page: WorldPage instance.
        rebuild: If True, clears all existing data before building.
    """

    async def on_complete() -> None:
        """Generate mini descriptions and reload page after build."""
        await generate_mini_descriptions(page)
        ui.navigate.reload()

    await show_build_structure_dialog(
        state=page.state,
        services=page.services,
        rebuild=rebuild,
        on_complete=on_complete,
    )


def confirm_clear_world(page) -> None:
    """Show confirmation dialog before clearing world data.

    Args:
        page: WorldPage instance.
    """
    logger.info("Clear World button clicked - showing confirmation dialog")

    if not page.state.world_db or not page.state.project:
        ui.notify("No world data to clear", type="info")
        return

    entity_count = page.state.world_db.count_entities()
    rel_count = len(page.state.world_db.list_relationships())
    chapter_count = len(page.state.project.chapters)
    char_count = len(page.state.project.characters)

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
                on_click=lambda: do_clear_world(page, dialog),
                icon="delete_sweep",
            ).props("color=warning")

    dialog.open()


def do_clear_world(page, dialog: ui.dialog) -> None:
    """Execute world clear - removes all entities, relationships, and story structure.

    Args:
        page: WorldPage instance.
        dialog: The dialog to close.
    """
    logger.info("User confirmed clear - removing all world data and story structure")
    dialog.close()

    if not page.state.world_db or not page.state.project:
        logger.warning("Clear failed: no world_db or project available")
        ui.notify("No world database available", type="negative")
        return

    try:
        # Delete all relationships first (they reference entities)
        relationships = page.state.world_db.list_relationships()
        for rel in relationships:
            page.state.world_db.delete_relationship(rel.id)
        logger.info(f"Deleted {len(relationships)} relationships")

        # Delete all entities
        entities = page.state.world_db.list_entities()
        for entity in entities:
            page.state.world_db.delete_entity(entity.id)
        logger.info(f"Deleted {len(entities)} entities")

        # Clear story structure (but keep brief and interview)
        chapter_count = len(page.state.project.chapters)
        char_count = len(page.state.project.characters)
        page.state.project.chapters = []
        page.state.project.characters = []
        page.state.project.world_description = ""
        page.state.project.plot_points = []
        logger.info(f"Cleared {chapter_count} chapters and {char_count} characters from project")

        # Save the project with cleared structure
        page.services.project.save_project(page.state.project)
        logger.info("Project saved with cleared structure")

        # Refresh UI
        page._refresh_entity_list()
        if page._graph:
            page._graph.refresh()

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


async def generate_mini_descriptions(page) -> None:
    """Generate mini descriptions for entity tooltips.

    Args:
        page: WorldPage instance.
    """
    from nicegui import run

    if not page.state.world_db:
        return

    all_entities = page.state.world_db.list_entities()
    entity_data = [
        {"name": e.name, "type": e.type, "description": e.description} for e in all_entities
    ]
    mini_descs = await run.io_bound(
        page.services.world_quality.generate_mini_descriptions_batch,
        entity_data,
    )
    logger.info(f"Generated {len(mini_descs)} mini descriptions")

    # Update entities with mini descriptions
    for entity in all_entities:
        if entity.name in mini_descs:
            attrs = dict(entity.attributes) if entity.attributes else {}
            attrs["mini_description"] = mini_descs[entity.name]
            page.state.world_db.update_entity(
                entity_id=entity.id,
                attributes=attrs,
            )
    logger.info("Updated entities with mini descriptions")

    # Invalidate graph cache to ensure fresh tooltips
    page.state.world_db.invalidate_graph_cache()

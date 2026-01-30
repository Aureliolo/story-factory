"""Structure, generation settings, reviews, and world overview sections."""

import logging
from typing import TYPE_CHECKING, Any

from nicegui import ui

from src.ui.graph_renderer import render_entity_summary_html

if TYPE_CHECKING:
    from . import WritePage

logger = logging.getLogger(__name__)


def build_world_overview_content(page: WritePage) -> None:
    """Build the world overview section content.

    Args:
        page: The WritePage instance.
    """
    if not page.state.world_db:
        ui.label("No world data yet. Complete the interview first.").classes("text-gray-400")
        return

    # Entity summary cards
    page._entity_summary = ui.html(sanitize=False)
    page._entity_summary.content = render_entity_summary_html(page.state.world_db)

    # Entity list (simple view instead of graph)
    characters = page.state.world_db.list_entities("character")
    if characters:
        ui.label("Characters").classes("text-sm font-medium mt-4")
        with ui.row().classes("flex-wrap gap-2"):
            for char in characters[:8]:  # Limit to 8 for preview
                ui.chip(char.name, icon="person", color="green").props("outline")
            if len(characters) > 8:
                ui.chip(f"+{len(characters) - 8} more", color="grey").props("outline")

    locations = page.state.world_db.list_entities("location")
    if locations:
        ui.label("Locations").classes("text-sm font-medium mt-2")
        with ui.row().classes("flex-wrap gap-2"):
            for loc in locations[:6]:  # Limit to 6 for preview
                ui.chip(loc.name, icon="place", color="blue").props("outline")
            if len(locations) > 6:
                ui.chip(f"+{len(locations) - 6} more", color="grey").props("outline")

    ui.button(
        "Open World Builder",
        on_click=lambda: ui.navigate.to("/world"),
        icon="public",
    ).props("flat").classes("mt-4")


def refresh_world_overview(page: WritePage) -> None:
    """Refresh the world overview section after data changes.

    Args:
        page: The WritePage instance.
    """
    if not page._world_overview_container:
        return
    page._world_overview_container.clear()
    with page._world_overview_container:
        build_world_overview_content(page)


def build_structure_section(page: WritePage) -> None:
    """Build the story structure section.

    Args:
        page: The WritePage instance.
    """
    if not page.state.project or not page.state.project.brief:
        ui.label("Complete the interview to see story structure.").classes("text-gray-400")
        return

    project = page.state.project
    brief = project.brief
    assert brief is not None  # Already checked above

    # Brief summary
    with ui.card().classes("w-full bg-gray-800"):
        ui.label("Story Brief").classes("text-lg font-semibold mb-2")

        with ui.row().classes("gap-4 flex-wrap"):
            with ui.column().classes("gap-1"):
                ui.label("Genre").classes("text-xs text-gray-400")
                ui.label(brief.genre).classes("font-medium")

            with ui.column().classes("gap-1"):
                ui.label("Tone").classes("text-xs text-gray-400")
                ui.label(brief.tone).classes("font-medium")

            with ui.column().classes("gap-1"):
                ui.label("Setting").classes("text-xs text-gray-400")
                ui.label(f"{brief.setting_place}, {brief.setting_time}")

            with ui.column().classes("gap-1"):
                ui.label("Length").classes("text-xs text-gray-400")
                ui.label(brief.target_length.replace("_", " ").title())

        ui.separator().classes("my-2")
        ui.label("Premise").classes("text-xs text-gray-400")
        ui.label(brief.premise).classes("text-sm")

    # Chapter outline
    if project.chapters:
        ui.label("Chapter Outline").classes("text-lg font-semibold mt-4 mb-2")

        for chapter in project.chapters:
            status_color = (
                "green"
                if chapter.status == "final"
                else "orange"
                if chapter.status in ["drafting", "edited"]
                else "gray"
            )

            with ui.row().classes("w-full items-start gap-2 p-2 hover:bg-gray-700 rounded"):
                ui.icon(
                    "check_circle" if chapter.status == "final" else "radio_button_unchecked",
                    color=status_color,
                )
                with ui.column().classes("flex-grow gap-1"):
                    ui.label(f"Chapter {chapter.number}: {chapter.title}").classes("font-medium")
                    ui.label(
                        chapter.outline[:150] + "..."
                        if len(chapter.outline) > 150
                        else chapter.outline
                    ).classes("text-sm text-gray-400")
                if chapter.word_count:
                    ui.badge(f"{chapter.word_count} words").props("color=grey-7")


def build_generation_settings_section(page: WritePage) -> None:
    """Build the project-specific generation settings section.

    Args:
        page: The WritePage instance.
    """
    if not page.state.project:
        return

    project = page.state.project
    settings = page.services.story.settings

    ui.label("Override default generation settings for this project.").classes(
        "text-sm text-gray-400 mb-2"
    )

    with ui.row().classes("w-full gap-4 flex-wrap"):
        # Chapter count setting
        with ui.column().classes("gap-1"):
            ui.label("Target Chapters").classes("text-xs text-gray-400")
            ui.label("Leave empty to use length-based default").classes("text-xs text-gray-500")

            def update_chapters(e: Any) -> None:
                """Update target chapter count setting from UI input."""
                value = e.value if e.value else None
                project.target_chapters = int(value) if value else None
                page.services.project.save_project(project)
                logger.info(f"Updated target chapters: {project.target_chapters}")

            ui.number(
                value=project.target_chapters,
                min=1,
                max=100,
                step=1,
                on_change=update_chapters,
            ).props("clearable").classes("w-24")

            # Show what default would be used
            if project.brief:
                length_map = {
                    "short_story": settings.chapters_short_story,
                    "novella": settings.chapters_novella,
                    "novel": settings.chapters_novel,
                }
                default_chapters = length_map.get(project.brief.target_length)
                if default_chapters:
                    ui.label(f"Default: {default_chapters}").classes("text-xs text-gray-500")

        # Character count settings
        with ui.column().classes("gap-1"):
            ui.label("Character Count Range").classes("text-xs text-gray-400")
            ui.label("Leave empty to use global defaults").classes("text-xs text-gray-500")

            with ui.row().classes("gap-2 items-center"):

                def update_min_chars(e: Any) -> None:
                    """Update minimum character count setting from UI input."""
                    value = e.value if e.value else None
                    project.target_characters_min = int(value) if value else None
                    page.services.project.save_project(project)
                    logger.info(f"Updated min characters: {project.target_characters_min}")

                ui.number(
                    value=project.target_characters_min,
                    min=1,
                    max=50,
                    step=1,
                    on_change=update_min_chars,
                ).props("clearable label=Min").classes("w-20")

                ui.label("-").classes("text-gray-400")

                def update_max_chars(e: Any) -> None:
                    """Update maximum character count setting from UI input."""
                    value = e.value if e.value else None
                    project.target_characters_max = int(value) if value else None
                    page.services.project.save_project(project)
                    logger.info(f"Updated max characters: {project.target_characters_max}")

                ui.number(
                    value=project.target_characters_max,
                    min=1,
                    max=50,
                    step=1,
                    on_change=update_max_chars,
                ).props("clearable label=Max").classes("w-20")

            ui.label(
                f"Default: {settings.world_gen_characters_min}-{settings.world_gen_characters_max}"
            ).classes("text-xs text-gray-500")


def build_reviews_section(page: WritePage) -> None:
    """Build the reviews and notes section.

    Args:
        page: The WritePage instance.
    """
    if not page.state.project:
        return

    reviews = page.state.project.reviews

    if not reviews:
        ui.label("No reviews or notes yet.").classes("text-gray-400")

    for review in reviews:
        with ui.card().classes("w-full bg-gray-800"):
            with ui.row().classes("items-center gap-2"):
                ui.badge(review.get("type", "note")).props("color=grey-7")
                if review.get("chapter"):
                    ui.label(f"Chapter {review['chapter']}").classes("text-sm text-gray-400")
            ui.label(review.get("content", "")).classes("mt-2")

    # Add note form
    with ui.row().classes("w-full gap-2 mt-4"):
        note_input = ui.input(placeholder="Add a note...").props("filled").classes("flex-grow")
        ui.button(
            "Add Note",
            on_click=lambda: add_note(page, note_input.value),
        ).props("flat")


def add_note(page: WritePage, content: str) -> None:
    """Add a review note.

    Args:
        page: The WritePage instance.
        content: The note text to add.
    """
    if not content or not page.state.project:
        return

    page.services.story.add_review(
        page.state.project,
        review_type="user_note",
        content=content,
        chapter_num=page.state.current_chapter,
    )
    page.services.project.save_project(page.state.project)
    page._notify("Note added", type="positive")

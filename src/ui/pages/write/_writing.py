"""Live writing tab - chapter navigator, display, controls, scenes, version history, suggestions."""

import html as html_lib
import logging
from typing import TYPE_CHECKING, Any

from nicegui import run, ui
from nicegui.elements.html import Html

from src.memory.mode_models import PRESET_MODES
from src.ui.components.generation_status import GenerationStatus

if TYPE_CHECKING:
    from . import WritePage

logger = logging.getLogger(__name__)


def build_live_writing(page: WritePage) -> None:
    """Build the Live Writing tab content.

    Args:
        page: The WritePage instance.
    """
    from src.ui.components.common import empty_state

    if not page.state.can_write:
        empty_state(
            icon="edit_off",
            title="Story structure not ready",
            description="Complete the interview and build the story structure first.",
        )
        return

    with ui.row().classes("w-full h-full gap-4 p-4"):
        # Left panel - Chapter navigator
        with ui.column().classes("w-1/5 gap-4"):
            build_chapter_navigator(page)

        # Center panel - Writing display
        with ui.column().classes("w-3/5 gap-4"):
            build_writing_display(page)

        # Right panel - Controls
        with ui.column().classes("w-1/5 gap-4"):
            build_writing_controls(page)


def build_chapter_navigator(page: WritePage) -> None:
    """Build the chapter navigation panel.

    Args:
        page: The WritePage instance.
    """
    ui.label("Chapters").classes("text-lg font-semibold")

    if not page.state.project:
        return

    chapters = page.state.project.chapters
    options = {ch.number: f"Ch {ch.number}: {ch.title[:20]}..." for ch in chapters}

    page._chapter_select = ui.select(
        options=options,
        value=page.state.current_chapter or (chapters[0].number if chapters else None),
        label="Select Chapter",
        on_change=lambda e: on_chapter_select(page, e),
    ).classes("w-full")

    # Chapter list
    for chapter in chapters:
        is_current = chapter.number == page.state.current_chapter
        status_icon = (
            "check_circle"
            if chapter.status == "final"
            else ("edit" if chapter.status in ["drafting", "edited"] else "radio_button_unchecked")
        )

        with (
            ui.row()
            .classes(
                f"w-full items-center gap-2 p-2 rounded cursor-pointer "
                f"{'bg-blue-50 dark:bg-blue-900' if is_current else 'hover:bg-gray-50 dark:hover:bg-gray-700'}"
            )
            .on("click", lambda n=chapter.number: select_chapter(page, n))
        ):
            ui.icon(status_icon, size="sm")
            ui.label(f"{chapter.number}. {chapter.title[:15]}...").classes("truncate")


def build_writing_display(page: WritePage) -> None:
    """Build the main writing display area.

    Args:
        page: The WritePage instance.
    """
    # Header
    with ui.row().classes("w-full items-center"):
        ui.label("Story Content").classes("text-lg font-semibold")
        ui.space()
        page._word_count_label = ui.label("0 words").classes(
            "text-sm text-gray-500 dark:text-gray-400"
        )

    # Scene editor section (collapsible)
    with ui.expansion("Scene Editor", icon="view_list", value=False).classes("w-full mb-4"):
        page._scene_list_container = ui.column().classes("w-full")
        with page._scene_list_container:
            build_scene_editor(page)

    # Generation status component
    page._generation_status = GenerationStatus(page.state)
    page._generation_status.build()

    # Writing area - use state-based dark mode detection for reliable styling
    bg_color = "#1f2937" if page.state.dark_mode else "#ffffff"
    border_color = "#374151" if page.state.dark_mode else "#e5e7eb"
    text_class = "prose-invert" if page.state.dark_mode else ""
    with (
        ui.element("div")
        .classes("w-full flex-grow p-4 rounded-lg border overflow-auto")
        .style(f"background-color: {bg_color}; border-color: {border_color}")
    ):
        page._writing_display = ui.markdown().classes(f"prose {text_class} max-w-none")

    # Load current chapter content
    refresh_writing_display(page)


def build_writing_controls(page: WritePage) -> None:
    """Build the writing controls panel.

    Args:
        page: The WritePage instance.
    """
    from . import _export as export_mod
    from . import _generation as gen_mod

    ui.label("Controls").classes("text-lg font-semibold")

    # Mode indicator
    if page.services.settings.use_mode_system:
        mode_id = page.services.settings.current_mode
        mode = PRESET_MODES.get(mode_id)
        if mode:
            with ui.row().classes("w-full items-center gap-2 mb-2"):
                ui.icon("tune", size="xs").classes("text-blue-500")
                ui.label(mode.name).classes("text-sm font-medium")
                ui.space()
                with (
                    ui.link(target="/settings")
                    .classes("text-xs text-gray-500 hover:text-blue-500")
                    .tooltip("Change mode in Settings")
                ):
                    ui.icon("settings", size="xs")

    # Write button
    ui.button(
        "Write Chapter",
        on_click=lambda: gen_mod.write_current_chapter(page),
        icon="edit",
    ).props("color=primary").classes("w-full")

    ui.button(
        "Write All",
        on_click=lambda: gen_mod.write_all_chapters(page),
        icon="auto_fix_high",
    ).props("outline").classes("w-full")

    ui.separator()

    # Writing Suggestions
    ui.button(
        "Need Inspiration?",
        on_click=lambda: show_suggestions(page),
        icon="lightbulb",
    ).props("outline color=amber-7").classes("w-full")

    ui.separator()

    # Feedback mode
    ui.label("Feedback Mode").classes("text-sm font-medium")
    ui.select(
        options=["per-chapter", "mid-chapter", "on-demand"],
        value=page.state.feedback_mode,
        on_change=lambda e: setattr(page.state, "feedback_mode", e.value),
    ).classes("w-full")

    # Regenerate with Feedback
    with ui.expansion("Regenerate with Feedback", icon="autorenew", value=False).classes("w-full"):
        ui.label("Provide specific feedback to improve this chapter").classes(
            "text-sm text-gray-600 dark:text-gray-400 mb-2"
        )
        page._regenerate_feedback_input = (
            ui.textarea(
                placeholder="e.g., 'Add more dialogue between the characters' or "
                "'Make the action sequence more suspenseful'"
            )
            .props("filled")
            .classes("w-full min-h-[100px]")
        )
        ui.button(
            "Regenerate Chapter",
            on_click=lambda: gen_mod.regenerate_with_feedback(page),
            icon="autorenew",
        ).props("color=primary").classes("w-full mt-2")

    # Version History
    with ui.expansion("Version History", icon="history", value=False).classes("w-full"):
        page._version_history_container = ui.column().classes("w-full gap-2")
        with page._version_history_container:
            build_version_history(page)

    ui.separator()

    # Export
    ui.label("Export").classes("text-sm font-medium")
    with ui.row().classes("w-full gap-2"):
        ui.button("MD", on_click=lambda: export_mod.export(page, "markdown")).props("flat size=sm")
        ui.button("TXT", on_click=lambda: export_mod.export(page, "text")).props("flat size=sm")
        ui.button("EPUB", on_click=lambda: export_mod.export(page, "epub")).props("flat size=sm")
        ui.button("PDF", on_click=lambda: export_mod.export(page, "pdf")).props("flat size=sm")


def build_scene_editor(page: WritePage) -> None:
    """Build the scene editor component.

    Args:
        page: The WritePage instance.
    """
    from src.ui.components.scene_editor import SceneListComponent

    if not page.state.project or page.state.current_chapter is None:
        ui.label("Select a chapter to manage scenes.").classes("text-gray-500 dark:text-gray-400")
        return

    # Get current chapter
    chapter = next(
        (c for c in page.state.project.chapters if c.number == page.state.current_chapter),
        None,
    )

    if not chapter:
        ui.label("Chapter not found.").classes("text-gray-500 dark:text-gray-400")
        return

    # Build scene list component
    def on_scene_updated():
        """Handle scene updates."""
        if page.state.project:
            # Update chapter word count from scenes
            chapter.update_chapter_word_count()
            # Save project
            page.services.project.save_project(page.state.project)
            # Refresh displays
            refresh_writing_display(page)
            logger.debug(f"Scenes updated for chapter {chapter.number}")

    scene_list = SceneListComponent(
        chapter=chapter,
        on_scene_updated=on_scene_updated,
    )
    scene_list.build()


def refresh_scene_editor(page: WritePage) -> None:
    """Refresh the scene editor component.

    Args:
        page: The WritePage instance.
    """
    if page._scene_list_container:
        page._scene_list_container.clear()
        with page._scene_list_container:
            build_scene_editor(page)


def on_chapter_select(page: WritePage, e: Any) -> None:
    """Handle chapter selection change.

    Args:
        page: The WritePage instance.
        e: The change event from the select widget.
    """
    page.state.select_chapter(e.value)
    refresh_writing_display(page)
    refresh_scene_editor(page)
    refresh_version_history(page)


def select_chapter(page: WritePage, chapter_num: int) -> None:
    """Select a chapter by number.

    Args:
        page: The WritePage instance.
        chapter_num: Chapter number to select.
    """
    page.state.select_chapter(chapter_num)
    if page._chapter_select:
        page._chapter_select.value = chapter_num
    refresh_writing_display(page)
    refresh_scene_editor(page)
    refresh_version_history(page)


def refresh_writing_display(page: WritePage) -> None:
    """Refresh the writing display with current chapter.

    Args:
        page: The WritePage instance.
    """
    if not page.state.project or not page._writing_display:
        return

    chapter = next(
        (c for c in page.state.project.chapters if c.number == page.state.current_chapter), None
    )

    if chapter and chapter.content:
        page._writing_display.content = chapter.content
        if page._word_count_label:
            page._word_count_label.text = f"{chapter.word_count} words"
    else:
        page._writing_display.content = "*No content yet. Click 'Write Chapter' to generate.*"
        if page._word_count_label:
            page._word_count_label.text = "0 words"


def build_version_history(page: WritePage) -> None:
    """Build the version history display.

    Args:
        page: The WritePage instance.
    """
    if not page.state.project or page.state.current_chapter is None:
        ui.label("Select a chapter to see its version history").classes(
            "text-gray-500 dark:text-gray-400 text-sm"
        )
        return

    chapter = next(
        (c for c in page.state.project.chapters if c.number == page.state.current_chapter),
        None,
    )

    if not chapter or not chapter.versions:
        ui.label("No previous versions yet").classes("text-gray-500 dark:text-gray-400 text-sm")
        return

    # Sort versions by version number (newest first)
    sorted_versions = sorted(chapter.versions, key=lambda v: v.version_number, reverse=True)

    for version in sorted_versions:
        with ui.card().classes("w-full p-2 bg-gray-50 dark:bg-gray-800"):
            with ui.row().classes("w-full items-center justify-between"):
                with ui.column().classes("gap-1 flex-grow"):
                    with ui.row().classes("items-center gap-2"):
                        ui.badge(f"v{version.version_number}").props("color=blue-7")
                        if version.is_current:
                            ui.badge("Current").props("color=green-7")
                        ui.label(version.created_at.strftime("%b %d, %I:%M %p")).classes(
                            "text-xs text-gray-500 dark:text-gray-400"
                        )

                    if version.feedback:
                        ui.label(f'Feedback: "{version.feedback}"').classes(
                            "text-xs italic text-gray-600 dark:text-gray-300 mt-1"
                        )

                    ui.label(f"{version.word_count} words").classes(
                        "text-xs text-gray-500 dark:text-gray-400"
                    )

                # Action buttons
                with ui.row().classes("gap-1"):
                    if not version.is_current:
                        ui.button(
                            icon="restore",
                            on_click=lambda v=version: rollback_to_version(page, v.id),
                        ).props("flat round size=sm").tooltip("Restore this version")

                    ui.button(
                        icon="visibility",
                        on_click=lambda v=version: view_version(page, v.id),
                    ).props("flat round size=sm").tooltip("View this version")


def refresh_version_history(page: WritePage) -> None:
    """Refresh the version history display.

    Args:
        page: The WritePage instance.
    """
    logger.debug("Refreshing version history display")
    if page._version_history_container:
        page._version_history_container.clear()
        with page._version_history_container:
            build_version_history(page)


def rollback_to_version(page: WritePage, version_id: str) -> None:
    """Rollback to a previous version.

    Args:
        page: The WritePage instance.
        version_id: The ID of the version to restore.
    """
    logger.debug(
        "Rolling back to version %s for chapter %s",
        version_id,
        page.state.current_chapter,
    )
    if not page.state.project or page.state.current_chapter is None:
        return

    chapter = next(
        (c for c in page.state.project.chapters if c.number == page.state.current_chapter),
        None,
    )

    if not chapter:
        return

    # Rollback
    success = chapter.rollback_to_version(version_id)
    if success:
        logger.info(
            "Successfully rolled back chapter %s to version %s",
            chapter.number,
            version_id,
        )
        # Update word count
        chapter.update_chapter_word_count()
        page.services.project.save_project(page.state.project)
        refresh_writing_display(page)
        refresh_version_history(page)
        page._notify("Version restored successfully", type="positive")
    else:
        logger.warning(
            "Failed to rollback chapter %s to version %s",
            page.state.current_chapter,
            version_id,
        )
        page._notify("Failed to restore version", type="negative")


def view_version(page: WritePage, version_id: str) -> None:
    """View a specific version in a dialog.

    Args:
        page: The WritePage instance.
        version_id: The ID of the version to view.
    """
    logger.debug(
        "Viewing version %s for chapter %s",
        version_id,
        page.state.current_chapter,
    )
    if not page.state.project or page.state.current_chapter is None:
        return

    chapter = next(
        (c for c in page.state.project.chapters if c.number == page.state.current_chapter),
        None,
    )

    if not chapter:
        return

    version = chapter.get_version_by_id(version_id)
    if not version:
        return

    # Create dialog to show version
    dialog = ui.dialog().props("maximized")

    with dialog, ui.card().classes("w-full h-full flex flex-col bg-white dark:bg-gray-800"):
        # Header
        with ui.row().classes(
            "w-full items-center justify-between p-4 border-b dark:border-gray-700"
        ):
            with ui.column().classes("gap-1"):
                with ui.row().classes("items-center gap-2"):
                    ui.label(f"Chapter {chapter.number}: {chapter.title}").classes(
                        "text-2xl font-bold"
                    )
                    ui.badge(f"Version {version.version_number}").props("color=blue-7")
                    if version.is_current:
                        ui.badge("Current").props("color=green-7")

                with ui.row().classes("items-center gap-4 text-sm text-gray-500"):
                    ui.label(version.created_at.strftime("%B %d, %Y at %I:%M %p"))
                    ui.label(f"{version.word_count} words")

                if version.feedback:
                    ui.label(f'Feedback used: "{version.feedback}"').classes(
                        "text-sm italic text-gray-600 dark:text-gray-400 mt-1"
                    )

            ui.button(icon="close", on_click=dialog.close).props("flat round")

        # Content
        with ui.column().classes("flex-grow overflow-auto p-6"):
            ui.markdown(version.content).classes("prose dark:prose-invert max-w-none")

        # Footer with actions
        with ui.row().classes("w-full justify-end gap-2 p-4 border-t dark:border-gray-700"):
            if not version.is_current:

                def restore_and_close() -> None:
                    """Restore the viewed version and close the dialog."""
                    rollback_to_version(page, version_id)
                    dialog.close()

                ui.button(
                    "Restore This Version",
                    on_click=restore_and_close,
                    icon="restore",
                ).props("color=primary")
            ui.button("Close", on_click=dialog.close).props("flat")

    dialog.open()


async def show_suggestions(page: WritePage) -> None:
    """Show AI-powered writing suggestions dialog.

    Args:
        page: The WritePage instance.
    """
    logger.debug("Opening writing suggestions dialog")
    if not page.state.project:
        page._notify("No project selected", type="warning")
        return

    # Create dialog
    dialog = ui.dialog().props("maximized")
    suggestions_html: Html | None = None
    loading_spinner: ui.spinner | None = None

    async def load_suggestions():
        """Load suggestions from the service."""
        nonlocal suggestions_html, loading_spinner

        if not page.state.project:
            dialog.close()
            return

        try:
            logger.debug("Loading suggestions...")
            # Show loading state
            if loading_spinner:
                loading_spinner.set_visibility(True)
            if suggestions_html:
                suggestions_html.set_visibility(False)

            # Generate suggestions
            suggestions = await run.io_bound(
                page.services.suggestion.generate_suggestions, page.state.project
            )

            logger.debug(f"Received suggestions with {len(suggestions)} categories")

            # Hide loading
            if loading_spinner:
                loading_spinner.set_visibility(False)

            # Build HTML for suggestions
            html_content = build_suggestions_html(page, suggestions)

            # Update display
            if suggestions_html:
                suggestions_html.content = html_content
                suggestions_html.set_visibility(True)
                logger.info("Suggestions loaded and displayed successfully")

        except Exception as e:
            logger.exception("Failed to generate suggestions")
            if loading_spinner:
                loading_spinner.set_visibility(False)
            page._notify(f"Error generating suggestions: {e}", type="negative")

    with dialog, ui.card().classes("w-full h-full flex flex-col bg-white dark:bg-gray-800"):
        # Header
        with ui.row().classes(
            "w-full items-center justify-between p-4 border-b dark:border-gray-700"
        ):
            with ui.row().classes("items-center gap-2"):
                ui.icon("lightbulb", size="md").classes("text-amber-500")
                ui.label("Writing Suggestions").classes("text-2xl font-bold")
            ui.button(icon="close", on_click=dialog.close).props("flat round")

        # Content area
        with ui.column().classes("flex-grow overflow-auto p-4"):
            loading_spinner = ui.spinner(size="lg").classes("mx-auto mt-8")
            suggestions_html = ui.html(sanitize=False)
            suggestions_html.set_visibility(False)

        # Footer
        with ui.row().classes("w-full justify-end gap-2 p-4 border-t dark:border-gray-700"):
            ui.button("Refresh", on_click=load_suggestions, icon="refresh").props("flat")

            def close_dialog():
                """Close the suggestions dialog."""
                logger.debug("Closing suggestions dialog")
                dialog.close()

            ui.button("Close", on_click=close_dialog).props("color=primary")

    # Open dialog and load suggestions
    logger.info("Opening suggestions dialog")
    dialog.open()
    await load_suggestions()


def build_suggestions_html(page: WritePage, suggestions: dict[str, list[str]]) -> str:
    """Build HTML content for suggestions display.

    Args:
        page: The WritePage instance.
        suggestions: Dictionary of categorized suggestions.

    Returns:
        HTML string for rendering.
    """
    # Category metadata - all values are hardcoded and safe
    category_info = {
        "plot": {"icon": "auto_stories", "title": "Plot Prompts", "color": "#8B5CF6"},
        "character": {"icon": "person", "title": "Character Prompts", "color": "#10B981"},
        "scene": {"icon": "theater_comedy", "title": "Scene Prompts", "color": "#F59E0B"},
        "transition": {
            "icon": "arrow_forward",
            "title": "Transition Prompts",
            "color": "#3B82F6",
        },
    }

    html_parts = []

    for category, items in suggestions.items():
        if not items:
            continue

        info = category_info.get(
            category, {"icon": "star", "title": category.title(), "color": "#6B7280"}
        )

        # Escape all dynamic content for security
        safe_icon = html_lib.escape(info["icon"])
        safe_title = html_lib.escape(info["title"])
        safe_color = html_lib.escape(info["color"])

        html_parts.append(
            f"""
        <div style="margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                <span class="material-icons" style="color: {safe_color}; font-size: 1.5rem;">{safe_icon}</span>
                <h3 style="margin: 0; font-size: 1.25rem; font-weight: 600; color: {safe_color};">{safe_title}</h3>
            </div>
            <div style="display: flex; flex-direction: column; gap: 0.75rem;">
        """
        )

        for suggestion in items:
            # Escape suggestion content to prevent XSS
            safe_suggestion = html_lib.escape(suggestion)
            html_parts.append(
                f"""
                <div style="padding: 1rem; background-color: rgba(0, 0, 0, 0.05); border-radius: 0.5rem; border-left: 3px solid {safe_color};">
                    <p style="margin: 0; line-height: 1.6;">{safe_suggestion}</p>
                </div>
            """
            )

        html_parts.append("</div></div>")

    if not html_parts:
        return (
            "<p style='text-align: center; color: #6B7280; padding: 2rem;'>"
            "No suggestions available.</p>"
        )

    return "".join(html_parts)

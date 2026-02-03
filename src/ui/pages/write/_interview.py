"""Interview section - chat, finalize, build structure."""

import asyncio
import logging
from typing import TYPE_CHECKING

from nicegui import run, ui

from src.ui.components.build_dialog import show_build_structure_dialog
from src.ui.theme import get_status_color

if TYPE_CHECKING:
    from . import WritePage

logger = logging.getLogger(__name__)


def build_interview_section(page: WritePage) -> None:
    """Build the Interview section of the UI.

    Constructs the status badge, chat component, action buttons, and loads
    existing interview history. When no history is present, starts a background
    task to begin the interview.

    Args:
        page: The WritePage instance.
    """
    from src.ui.components.chat import ChatComponent

    # Status indicator
    status = page.state.project.status if page.state.project else "unknown"
    color = get_status_color(status)

    with ui.row().classes("w-full items-center gap-2 mb-2"):
        ui.badge(status.title()).style(f"background-color: {color}; color: white;")

        if page.state.interview_complete:
            ui.label("Interview complete - structure ready").classes("text-sm text-gray-400")
            ui.button(
                "Continue Interview",
                on_click=lambda: enable_continue_interview(page),
            ).props("flat size=sm")

    # Chat component
    page._chat = ChatComponent(
        on_send=lambda msg: _handle_interview_message(page, msg),
        placeholder="Describe your story idea...",
        disabled=page.state.interview_complete,
    )
    page._chat.build()

    # Load existing messages
    if page.state.interview_history:
        page._chat.set_messages(page.state.interview_history)
    elif not page.state.interview_complete:
        # Start interview if fresh
        task = asyncio.create_task(_start_interview(page))
        page._background_tasks.add(task)
        task.add_done_callback(page._background_tasks.discard)

    # Finalize button - right-aligned
    with ui.row().classes("w-full justify-end gap-2 mt-2"):
        page._finalize_btn = ui.button(
            "Finalize Interview",
            on_click=lambda: _finalize_interview(page),
        ).props("flat")
        page._finalize_btn.set_visibility(not page.state.interview_complete)

    # Build Structure button - centered, only when no entities exist yet
    with ui.row().classes("w-full justify-center mt-4"):
        page._build_structure_btn = ui.button(
            "Build Story Structure",
            on_click=lambda: _build_structure(page),
            icon="auto_fix_high",
        ).props("color=primary size=lg")
        has_entities = page.state.world_db and page.state.world_db.count_entities() > 0
        show_build = bool(page.state.interview_complete and page.state.project and not has_entities)
        page._build_structure_btn.set_visibility(show_build)


async def _start_interview(page: WritePage) -> None:
    """Start the interview process.

    Args:
        page: The WritePage instance.
    """
    if not page.state.project or not page._chat:
        return

    page.state.begin_background_task("start_interview")
    try:
        page._chat.show_typing(True)
        # Run LLM call in thread pool to avoid blocking event loop
        questions = await run.io_bound(page.services.story.start_interview, page.state.project)
        page._chat.show_typing(False)
        page._chat.add_message("assistant", questions)
        page.state.add_interview_message("assistant", questions)
    except Exception as e:
        logger.exception("Failed to start interview")
        page._chat.show_typing(False)
        page._notify(f"Error starting interview: {e}", type="negative")
    finally:
        page.state.end_background_task("start_interview")


async def _handle_interview_message(page: WritePage, message: str) -> None:
    """Handle user message in interview.

    Args:
        page: The WritePage instance.
        message: The user's message text.
    """
    if not page.state.project or not page._chat:
        return

    page.state.begin_background_task("handle_interview_message")
    try:
        page._chat.show_typing(True)
        page.state.add_interview_message("user", message)

        # Run LLM call in thread pool to avoid blocking event loop
        response, is_complete = await run.io_bound(
            page.services.story.process_interview, page.state.project, message
        )

        page._chat.show_typing(False)
        page._chat.add_message("assistant", response)
        page.state.add_interview_message("assistant", response)

        if is_complete:
            page.state.interview_complete = True
            page._chat.set_disabled(True)
            update_interview_buttons(page)
            page._notify(
                "Interview complete! You can now build the story structure.", type="positive"
            )

        # Save progress
        page.services.project.save_project(page.state.project)

    except Exception as e:
        logger.exception("Failed to process interview message")
        page._chat.show_typing(False)
        page._notify(f"Error: {e}", type="negative")
    finally:
        page.state.end_background_task("handle_interview_message")


def update_interview_buttons(page: WritePage) -> None:
    """Update visibility of interview action buttons.

    Args:
        page: The WritePage instance.
    """
    if hasattr(page, "_finalize_btn") and page._finalize_btn:
        page._finalize_btn.set_visibility(not page.state.interview_complete)
    if hasattr(page, "_build_structure_btn") and page._build_structure_btn:
        has_entities = page.state.world_db and page.state.world_db.count_entities() > 0
        show_build = bool(page.state.interview_complete and page.state.project and not has_entities)
        page._build_structure_btn.set_visibility(show_build)


async def _finalize_interview(page: WritePage) -> None:
    """Force finalize the interview with confirmation.

    Args:
        page: The WritePage instance.
    """
    if not page.state.project:
        return

    # Create dialog reference for the async handler
    dialog = ui.dialog()

    async def do_finalize() -> None:
        """Handle the actual finalization."""
        dialog.close()
        if not page.state.project:
            return
        page.state.begin_background_task("finalize_interview")
        try:
            page._notify("Finalizing interview...", type="info")
            brief = await run.io_bound(page.services.story.finalize_interview, page.state.project)
            page.state.interview_complete = True
            if page._chat:
                page._chat.set_disabled(True)
                brief_summary = (
                    f"**Story Brief Generated:**\n\n"
                    f"- **Premise:** {brief.premise}\n"
                    f"- **Genre:** {brief.genre}\n"
                    f"- **Tone:** {brief.tone}\n"
                    f"- **Setting:** {brief.setting_place}, {brief.setting_time}"
                )
                page._chat.add_message("assistant", brief_summary)
            update_interview_buttons(page)
            page.services.project.save_project(page.state.project)
            page._notify("Interview finalized!", type="positive")
        except Exception as e:
            logger.exception("Failed to finalize interview")
            page._notify(f"Error: {e}", type="negative")
        finally:
            page.state.end_background_task("finalize_interview")

    with dialog, ui.card().classes("w-96").style("background-color: #1f2937"):
        ui.label("Finalize Interview?").classes("text-xl font-bold mb-2")
        ui.label(
            "This will generate a story brief based on the conversation so far. "
            "Are you sure you want to finalize?"
        ).classes("text-gray-400 mb-4")

        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button("Finalize", on_click=do_finalize).props("color=primary")

    dialog.open()


def enable_continue_interview(page: WritePage) -> None:
    """Enable continuing the interview.

    Args:
        page: The WritePage instance.
    """
    if page._chat:
        page._chat.set_disabled(False)
        page._chat.add_message(
            "assistant",
            "I see you want to make some changes. Please note that significant "
            "changes at this stage may have unintended consequences on the story "
            "structure. What would you like to adjust?",
        )


async def _build_structure(page: WritePage) -> None:
    """Build the story structure using the shared dialog.

    Args:
        page: The WritePage instance.
    """
    from . import _structure as structure_mod

    async def on_complete() -> None:
        """Update UI after build completes."""
        update_interview_buttons(page)
        structure_mod.refresh_world_overview(page)

    await show_build_structure_dialog(
        state=page.state,
        services=page.services,
        rebuild=False,
        on_complete=on_complete,
    )

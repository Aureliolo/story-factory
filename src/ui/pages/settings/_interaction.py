"""Settings page - interaction and context sections."""

import logging
from typing import TYPE_CHECKING

from nicegui import ui

if TYPE_CHECKING:
    from src.ui.pages.settings import SettingsPage

logger = logging.getLogger("src.ui.pages.settings._interaction")


def build_interaction_section(page: SettingsPage) -> None:
    """Build interaction mode settings.

    Args:
        page: The SettingsPage instance.
    """
    with ui.card().classes("w-full h-full"):
        page._section_header(
            "Workflow",
            "tune",
            "Control how much the AI asks for your input during story generation",
        )

        with ui.column().classes("w-full gap-3"):
            page._interaction_mode_select = (
                ui.select(
                    label="Interaction Mode",
                    options={
                        "minimal": "Minimal - Fewest interruptions",
                        "checkpoint": "Checkpoint - Review every N chapters",
                        "interactive": "Interactive - More control points",
                        "collaborative": "Collaborative - Maximum interaction",
                    },
                    value=page.settings.interaction_mode,
                )
                .classes("w-full")
                .props("outlined dense")
                .tooltip("How often the AI pauses for your feedback")
            )

            page._checkpoint_input = (
                ui.number(
                    label="Chapters between checkpoints",
                    value=page.settings.chapters_between_checkpoints,
                    min=1,
                    max=20,
                )
                .classes("w-full")
                .props("outlined dense")
                .tooltip("How many chapters to write before pausing for review")
            )

            page._revision_input = (
                ui.number(
                    label="Max revision iterations",
                    value=page.settings.max_revision_iterations,
                    min=0,
                    max=10,
                )
                .classes("w-full")
                .props("outlined dense")
                .tooltip("Max edit passes per chapter (0 = unlimited)")
            )

    logger.debug("Interaction section built")


def build_context_section(page: SettingsPage) -> None:
    """Create the "Memory & Context" settings card with inputs for context-related limits.

    Constructs a UI card containing number inputs for context window (tokens), max output tokens,
    previous chapter memory (chars), analysis context (chars), and editor preview (chars).
    Stores the input component references on page as:
    ``_context_size_input``, ``_max_tokens_input``, ``_prev_chapter_chars``,
    ``_chapter_analysis_chars``, and ``_full_text_preview_chars``.

    Args:
        page: The SettingsPage instance.
    """
    with ui.card().classes("w-full h-full"):
        page._section_header(
            "Memory & Context",
            "memory",
            "Control how much information the AI remembers. "
            "Higher values use more VRAM but improve coherence.",
        )

        # Single column with proper spacing for readable labels
        with ui.column().classes("w-full gap-3"):
            page._context_size_input = (
                ui.number(
                    label="Context window (tokens)",
                    value=page.settings.context_size,
                    min=1024,
                    max=128000,
                    step=1024,
                )
                .classes("w-full")
                .props("outlined dense")
                .tooltip("Total tokens the AI can 'see' at once (default: 32768)")
            )

            page._max_tokens_input = (
                ui.number(
                    label="Max output (tokens)",
                    value=page.settings.max_tokens,
                    min=256,
                    max=32000,
                    step=256,
                )
                .classes("w-full")
                .props("outlined dense")
                .tooltip("Maximum tokens per AI response (default: 4096)")
            )

            page._prev_chapter_chars = (
                ui.number(
                    label="Chapter memory (chars)",
                    value=page.settings.previous_chapter_context_chars,
                    min=500,
                    max=10000,
                )
                .classes("w-full")
                .props("outlined dense")
                .tooltip("Characters from previous chapter to include for continuity")
            )

            page._chapter_analysis_chars = (
                ui.number(
                    label="Analysis context (chars)",
                    value=page.settings.chapter_analysis_chars,
                    min=1000,
                    max=20000,
                )
                .classes("w-full")
                .props("outlined dense")
                .tooltip("Characters to analyze when reviewing chapter quality")
            )

            page._full_text_preview_chars = (
                ui.number(
                    label="Editor preview (chars)",
                    value=page.settings.full_text_preview_chars,
                    min=500,
                    max=10000,
                )
                .classes("w-full")
                .props("outlined dense")
                .tooltip("Characters sent to editor for suggestions")
            )

    logger.debug("Context section built")

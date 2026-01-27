"""Write Story page - Live writing display mixin."""

import logging
from typing import Any

from nicegui import ui

from src.ui.pages.write._page import WritePageBase

logger = logging.getLogger(__name__)


class WritingMixin(WritePageBase):
    """Mixin providing live writing display methods for WritePage.

    This mixin handles:
    - Building the Live Writing tab layout
    - Chapter navigation panel
    - Writing display area with scenes
    - Writing controls panel
    """

    def _build_live_writing(self) -> None:
        """Build the Live Writing tab content."""
        from src.ui.components.common import empty_state

        if not self.state.can_write:
            empty_state(
                icon="edit_off",
                title="Story structure not ready",
                description="Complete the interview and build the story structure first.",
            )
            return

        with ui.row().classes("w-full h-full gap-4 p-4"):
            # Left panel - Chapter navigator
            with ui.column().classes("w-1/5 gap-4"):
                self._build_chapter_navigator()

            # Center panel - Writing display
            with ui.column().classes("w-3/5 gap-4"):
                self._build_writing_display()

            # Right panel - Controls
            with ui.column().classes("w-1/5 gap-4"):
                self._build_writing_controls()

    def _build_chapter_navigator(self) -> None:
        """Build the chapter navigation panel."""
        ui.label("Chapters").classes("text-lg font-semibold")

        if not self.state.project:
            return

        chapters = self.state.project.chapters
        options = {ch.number: f"Ch {ch.number}: {ch.title[:20]}..." for ch in chapters}

        self._chapter_select = ui.select(
            options=options,
            value=self.state.current_chapter or (chapters[0].number if chapters else None),
            label="Select Chapter",
            on_change=self._on_chapter_select,
        ).classes("w-full")

        # Chapter list
        for chapter in chapters:
            is_current = chapter.number == self.state.current_chapter
            status_icon = (
                "check_circle"
                if chapter.status == "final"
                else (
                    "edit" if chapter.status in ["drafting", "edited"] else "radio_button_unchecked"
                )
            )

            with (
                ui.row()
                .classes(
                    f"w-full items-center gap-2 p-2 rounded cursor-pointer "
                    f"{'bg-blue-50 dark:bg-blue-900' if is_current else 'hover:bg-gray-50 dark:hover:bg-gray-700'}"
                )
                .on("click", lambda n=chapter.number: self._select_chapter(n))
            ):
                ui.icon(status_icon, size="sm")
                ui.label(f"{chapter.number}. {chapter.title[:15]}...").classes("truncate")

    def _build_writing_display(self) -> None:
        """Build the main writing display area."""
        from src.ui.components.generation_status import GenerationStatus

        # Header
        with ui.row().classes("w-full items-center"):
            ui.label("Story Content").classes("text-lg font-semibold")
            ui.space()
            self._word_count_label = ui.label("0 words").classes(
                "text-sm text-gray-500 dark:text-gray-400"
            )

        # Scene editor section (collapsible)
        with ui.expansion("Scene Editor", icon="view_list", value=False).classes("w-full mb-4"):
            self._scene_list_container = ui.column().classes("w-full")
            with self._scene_list_container:
                self._build_scene_editor()

        # Generation status component
        self._generation_status = GenerationStatus(self.state)
        self._generation_status.build()

        # Writing area - use state-based dark mode detection for reliable styling
        bg_color = "#1f2937" if self.state.dark_mode else "#ffffff"
        border_color = "#374151" if self.state.dark_mode else "#e5e7eb"
        text_class = "prose-invert" if self.state.dark_mode else ""
        with (
            ui.element("div")
            .classes("w-full flex-grow p-4 rounded-lg border overflow-auto")
            .style(f"background-color: {bg_color}; border-color: {border_color}")
        ):
            self._writing_display = ui.markdown().classes(f"prose {text_class} max-w-none")

        # Load current chapter content
        self._refresh_writing_display()

    def _build_writing_controls(self) -> None:
        """Build the writing controls panel."""
        from src.memory.mode_models import PRESET_MODES

        ui.label("Controls").classes("text-lg font-semibold")

        # Mode indicator
        if self.services.settings.use_mode_system:
            mode_id = self.services.settings.current_mode
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
            on_click=self._write_current_chapter,
            icon="edit",
        ).props("color=primary").classes("w-full")

        ui.button(
            "Write All",
            on_click=self._write_all_chapters,
            icon="auto_fix_high",
        ).props("outline").classes("w-full")

        ui.separator()

        # Writing Suggestions
        ui.button(
            "Need Inspiration?",
            on_click=self._show_suggestions,
            icon="lightbulb",
        ).props("outline color=amber-7").classes("w-full")

        ui.separator()

        # Feedback mode
        ui.label("Feedback Mode").classes("text-sm font-medium")
        ui.select(
            options=["per-chapter", "mid-chapter", "on-demand"],
            value=self.state.feedback_mode,
            on_change=lambda e: setattr(self.state, "feedback_mode", e.value),
        ).classes("w-full")

        # Regenerate with Feedback
        with ui.expansion("Regenerate with Feedback", icon="autorenew", value=False).classes(
            "w-full"
        ):
            ui.label("Provide specific feedback to improve this chapter").classes(
                "text-sm text-gray-600 dark:text-gray-400 mb-2"
            )
            self._regenerate_feedback_input = (
                ui.textarea(
                    placeholder="e.g., 'Add more dialogue between the characters' or 'Make the action sequence more suspenseful'"
                )
                .props("filled")
                .classes("w-full min-h-[100px]")
            )
            ui.button(
                "Regenerate Chapter",
                on_click=self._regenerate_with_feedback,
                icon="autorenew",
            ).props("color=primary").classes("w-full mt-2")

        # Version History
        with ui.expansion("Version History", icon="history", value=False).classes("w-full"):
            self._version_history_container = ui.column().classes("w-full gap-2")
            with self._version_history_container:
                self._build_version_history()

        ui.separator()

        # Export
        ui.label("Export").classes("text-sm font-medium")
        with ui.row().classes("w-full gap-2"):
            ui.button("MD", on_click=lambda: self._export("markdown")).props("flat size=sm")
            ui.button("TXT", on_click=lambda: self._export("text")).props("flat size=sm")
            ui.button("EPUB", on_click=lambda: self._export("epub")).props("flat size=sm")
            ui.button("PDF", on_click=lambda: self._export("pdf")).props("flat size=sm")

    def _build_scene_editor(self) -> None:
        """Build the scene editor component."""
        from src.ui.components.scene_editor import SceneListComponent

        if not self.state.project or self.state.current_chapter is None:
            ui.label("Select a chapter to manage scenes.").classes(
                "text-gray-500 dark:text-gray-400"
            )
            return

        # Get current chapter
        chapter = next(
            (c for c in self.state.project.chapters if c.number == self.state.current_chapter),
            None,
        )

        if not chapter:
            ui.label("Chapter not found.").classes("text-gray-500 dark:text-gray-400")
            return

        # Build scene list component
        def on_scene_updated():
            """Handle scene updates."""
            if self.state.project:
                # Update chapter word count from scenes
                chapter.update_chapter_word_count()
                # Save project
                self.services.project.save_project(self.state.project)
                # Refresh displays
                self._refresh_writing_display()
                logger.debug(f"Scenes updated for chapter {chapter.number}")

        scene_list = SceneListComponent(
            chapter=chapter,
            on_scene_updated=on_scene_updated,
        )
        scene_list.build()

    def _refresh_scene_editor(self) -> None:
        """Refresh the scene editor component."""
        if self._scene_list_container:
            self._scene_list_container.clear()
            with self._scene_list_container:
                self._build_scene_editor()

    def _on_chapter_select(self, e: Any) -> None:
        """Handle chapter selection change."""
        self.state.select_chapter(e.value)
        self._refresh_writing_display()
        self._refresh_scene_editor()
        self._refresh_version_history()

    def _select_chapter(self, chapter_num: int) -> None:
        """Select a chapter by number."""
        self.state.select_chapter(chapter_num)
        if self._chapter_select:
            self._chapter_select.value = chapter_num
        self._refresh_writing_display()
        self._refresh_scene_editor()
        self._refresh_version_history()

    def _refresh_writing_display(self) -> None:
        """Refresh the writing display with current chapter."""
        if not self.state.project or not self._writing_display:
            return

        chapter = next(
            (c for c in self.state.project.chapters if c.number == self.state.current_chapter), None
        )

        if chapter and chapter.content:
            self._writing_display.content = chapter.content
            if self._word_count_label:
                self._word_count_label.text = f"{chapter.word_count} words"
        else:
            self._writing_display.content = "*No content yet. Click 'Write Chapter' to generate.*"
            if self._word_count_label:
                self._word_count_label.text = "0 words"

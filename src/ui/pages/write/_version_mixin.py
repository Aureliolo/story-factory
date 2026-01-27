"""Write Story page - Version history mixin."""

import logging

from nicegui import ui

from src.ui.pages.write._page import WritePageBase

logger = logging.getLogger(__name__)


class VersionMixin(WritePageBase):
    """Mixin providing version history methods for WritePage.

    This mixin handles:
    - Building version history display
    - Refreshing version history
    - Rolling back to previous versions
    - Viewing specific versions
    """

    def _build_version_history(self) -> None:
        """Build the version history display."""
        if not self.state.project or self.state.current_chapter is None:
            ui.label("Select a chapter to see its version history").classes(
                "text-gray-500 dark:text-gray-400 text-sm"
            )
            return

        chapter = next(
            (c for c in self.state.project.chapters if c.number == self.state.current_chapter),
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
                                on_click=lambda v=version: self._rollback_to_version(v.id),
                            ).props("flat round size=sm").tooltip("Restore this version")

                        ui.button(
                            icon="visibility",
                            on_click=lambda v=version: self._view_version(v.id),
                        ).props("flat round size=sm").tooltip("View this version")

    def _refresh_version_history(self) -> None:
        """Refresh the version history display."""
        logger.debug("Refreshing version history display")
        if self._version_history_container:
            self._version_history_container.clear()
            with self._version_history_container:
                self._build_version_history()

    def _rollback_to_version(self, version_id: str) -> None:
        """Rollback to a previous version."""
        logger.debug(
            "Rolling back to version %s for chapter %s",
            version_id,
            self.state.current_chapter,
        )
        if not self.state.project or self.state.current_chapter is None:
            return

        chapter = next(
            (c for c in self.state.project.chapters if c.number == self.state.current_chapter),
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
            self.services.project.save_project(self.state.project)
            self._refresh_writing_display()
            self._refresh_version_history()
            self._notify("Version restored successfully", type="positive")
        else:
            logger.warning(
                "Failed to rollback chapter %s to version %s",
                self.state.current_chapter,
                version_id,
            )
            self._notify("Failed to restore version", type="negative")

    def _view_version(self, version_id: str) -> None:
        """View a specific version in a dialog."""
        logger.debug(
            "Viewing version %s for chapter %s",
            version_id,
            self.state.current_chapter,
        )
        if not self.state.project or self.state.current_chapter is None:
            return

        chapter = next(
            (c for c in self.state.project.chapters if c.number == self.state.current_chapter),
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
                        self._rollback_to_version(version_id)
                        dialog.close()

                    ui.button(
                        "Restore This Version",
                        on_click=restore_and_close,
                        icon="restore",
                    ).props("color=primary")
                ui.button("Close", on_click=dialog.close).props("flat")

        dialog.open()

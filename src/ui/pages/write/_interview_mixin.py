"""Write Story page - Interview phase mixin."""

import asyncio
import logging

from nicegui import run, ui

from src.ui.pages.write._page import WritePageBase

logger = logging.getLogger(__name__)


class InterviewMixin(WritePageBase):
    """Mixin providing interview phase methods for WritePage.

    This mixin handles:
    - Building the interview section UI
    - Starting and conducting interviews
    - Finalizing interviews and generating story briefs
    - Continuing interviews after finalization
    """

    def _build_interview_section(self) -> None:
        """
        Builds the Interview section of the UI, including the status badge, chat component, and action buttons.

        This constructs and configures the chat UI used for conducting or continuing the interview, loads any existing interview history into the chat, and when no history is present starts a background task to begin the interview. It also creates the Finalize Interview and Build Story Structure buttons and updates their visibility based on interview and project state. Side effects: mutates UI component attributes on the instance (e.g., `_chat`, `_finalize_btn`, `_build_structure_btn`) and registers background tasks in `self._background_tasks`.
        """
        from src.ui.components.chat import ChatComponent
        from src.ui.theme import get_status_color

        # Status indicator
        status = self.state.project.status if self.state.project else "unknown"
        color = get_status_color(status)

        with ui.row().classes("w-full items-center gap-2 mb-2"):
            ui.badge(status.title()).style(f"background-color: {color}; color: white;")

            if self.state.interview_complete:
                ui.label("Interview complete - structure ready").classes(
                    "text-sm text-gray-500 dark:text-gray-400"
                )
                ui.button(
                    "Continue Interview",
                    on_click=self._enable_continue_interview,
                ).props("flat size=sm")

        # Chat component
        self._chat = ChatComponent(
            on_send=self._handle_interview_message,
            placeholder="Describe your story idea...",
            disabled=self.state.interview_complete,
        )
        self._chat.build()

        # Load existing messages
        if self.state.interview_history:
            self._chat.set_messages(self.state.interview_history)
        elif not self.state.interview_complete:
            # Start interview if fresh
            task = asyncio.create_task(self._start_interview())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        # Finalize button - right-aligned
        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            self._finalize_btn = ui.button(
                "Finalize Interview",
                on_click=self._finalize_interview,
            ).props("flat")
            self._finalize_btn.set_visibility(not self.state.interview_complete)

        # Build Structure button - centered
        with ui.row().classes("w-full justify-center mt-4"):
            self._build_structure_btn = ui.button(
                "Build Story Structure",
                on_click=self._build_structure,
                icon="auto_fix_high",
            ).props("color=primary size=lg")
            show_build = bool(
                self.state.interview_complete
                and self.state.project
                and not self.state.project.chapters
            )
            self._build_structure_btn.set_visibility(show_build)

    async def _start_interview(self) -> None:
        """Start the interview process."""
        if not self.state.project or not self._chat:
            return

        try:
            self._chat.show_typing(True)
            # Run LLM call in thread pool to avoid blocking event loop
            questions = await run.io_bound(self.services.story.start_interview, self.state.project)
            self._chat.show_typing(False)
            self._chat.add_message("assistant", questions)
            self.state.add_interview_message("assistant", questions)
        except Exception as e:
            logger.exception("Failed to start interview")
            self._chat.show_typing(False)
            self._notify(f"Error starting interview: {e}", type="negative")

    async def _handle_interview_message(self, message: str) -> None:
        """Handle user message in interview."""
        if not self.state.project or not self._chat:
            return

        try:
            self._chat.show_typing(True)
            self.state.add_interview_message("user", message)

            # Run LLM call in thread pool to avoid blocking event loop
            response, is_complete = await run.io_bound(
                self.services.story.process_interview, self.state.project, message
            )

            self._chat.show_typing(False)
            self._chat.add_message("assistant", response)
            self.state.add_interview_message("assistant", response)

            if is_complete:
                self.state.interview_complete = True
                self._chat.set_disabled(True)
                self._update_interview_buttons()
                self._notify(
                    "Interview complete! You can now build the story structure.", type="positive"
                )

            # Save progress
            self.services.project.save_project(self.state.project)

        except Exception as e:
            logger.exception("Failed to process interview message")
            self._chat.show_typing(False)
            self._notify(f"Error: {e}", type="negative")

    def _update_interview_buttons(self) -> None:
        """Update visibility of interview action buttons."""
        if hasattr(self, "_finalize_btn") and self._finalize_btn:
            self._finalize_btn.set_visibility(not self.state.interview_complete)
        if hasattr(self, "_build_structure_btn") and self._build_structure_btn:
            show_build = bool(
                self.state.interview_complete
                and self.state.project
                and not self.state.project.chapters
            )
            self._build_structure_btn.set_visibility(show_build)

    async def _finalize_interview(self) -> None:
        """Force finalize the interview with confirmation."""
        if not self.state.project:
            return

        # Create dialog reference for the async handler
        dialog = ui.dialog()

        async def do_finalize() -> None:
            """Handle the actual finalization."""
            dialog.close()
            if not self.state.project:
                return
            try:
                self._notify("Finalizing interview...", type="info")
                brief = await run.io_bound(
                    self.services.story.finalize_interview, self.state.project
                )
                self.state.interview_complete = True
                if self._chat:
                    self._chat.set_disabled(True)
                    brief_summary = (
                        f"**Story Brief Generated:**\n\n"
                        f"- **Premise:** {brief.premise}\n"
                        f"- **Genre:** {brief.genre}\n"
                        f"- **Tone:** {brief.tone}\n"
                        f"- **Setting:** {brief.setting_place}, {brief.setting_time}"
                    )
                    self._chat.add_message("assistant", brief_summary)
                self._update_interview_buttons()
                self.services.project.save_project(self.state.project)
                self._notify("Interview finalized!", type="positive")
            except Exception as e:
                logger.exception("Failed to finalize interview")
                self._notify(f"Error: {e}", type="negative")

        dialog_bg = "#1f2937" if self.state.dark_mode else "#ffffff"
        with dialog, ui.card().classes("w-96").style(f"background-color: {dialog_bg}"):
            ui.label("Finalize Interview?").classes("text-xl font-bold mb-2")
            ui.label(
                "This will generate a story brief based on the conversation so far. "
                "Are you sure you want to finalize?"
            ).classes("text-gray-600 dark:text-gray-400 mb-4")

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button("Finalize", on_click=do_finalize).props("color=primary")

        dialog.open()

    def _enable_continue_interview(self) -> None:
        """Enable continuing the interview."""
        if self._chat:
            self._chat.set_disabled(False)
            self._chat.add_message(
                "assistant",
                "I see you want to make some changes. Please note that significant "
                "changes at this stage may have unintended consequences on the story "
                "structure. What would you like to adjust?",
            )

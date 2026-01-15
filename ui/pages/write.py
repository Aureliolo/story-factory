"""Write Story page - Fundamentals and Live Writing tabs."""

import asyncio

from nicegui import ui
from nicegui.elements.html import Html
from nicegui.elements.label import Label
from nicegui.elements.markdown import Markdown
from nicegui.elements.select import Select
from nicegui.elements.textarea import Textarea

from services import ServiceContainer
from ui.components.chat import ChatComponent
from ui.components.graph import mini_graph
from ui.graph_renderer import render_entity_summary_html
from ui.state import AppState
from ui.theme import get_status_color


class WritePage:
    """Write Story page with Fundamentals and Live Writing tabs.

    Fundamentals tab:
    - Interview chat
    - World overview
    - Story structure
    - Reviews

    Live Writing tab:
    - Chapter navigator
    - Writing display with streaming
    - Feedback controls
    """

    def __init__(self, state: AppState, services: ServiceContainer):
        """Initialize write page.

        Args:
            state: Application state.
            services: Service container.
        """
        self.state = state
        self.services = services

        # UI references
        self._chat: ChatComponent | None = None
        self._structure_display: Markdown | None = None
        self._entity_summary: Html | None = None
        self._chapter_select: Select | None = None
        self._writing_display: Markdown | None = None
        self._word_count_label: Label | None = None
        self._feedback_input: Textarea | None = None

    def build(self) -> None:
        """Build the write page UI."""
        if not self.state.has_project:
            self._build_no_project_message()
            return

        # Sub-tabs
        with ui.tabs().classes("w-full") as tabs:
            ui.tab("fundamentals", label="Fundamentals", icon="foundation")
            ui.tab("writing", label="Live Writing", icon="edit_note")

        with ui.tab_panels(tabs, value="fundamentals").classes("w-full flex-grow"):
            with ui.tab_panel("fundamentals"):
                self._build_fundamentals()

            with ui.tab_panel("writing"):
                self._build_live_writing()

    def _build_no_project_message(self) -> None:
        """Build message when no project is selected."""
        with ui.column().classes("w-full h-full items-center justify-center gap-4"):
            ui.icon("folder_off", size="xl").classes("text-gray-400")
            ui.label("No Project Selected").classes("text-xl text-gray-500")
            ui.label("Select a project from the header dropdown or create a new one.").classes(
                "text-gray-400"
            )

    def _build_fundamentals(self) -> None:
        """Build the Fundamentals tab content."""
        with ui.column().classes("w-full gap-4 p-4"):
            # Interview Section
            with ui.expansion(
                "Interview",
                icon="chat",
                value=not self.state.interview_complete,
            ).classes("w-full"):
                self._build_interview_section()

            # World Overview Section
            with ui.expansion("World Overview", icon="public").classes("w-full"):
                self._build_world_overview()

            # Story Structure Section
            with ui.expansion("Story Structure", icon="list_alt").classes("w-full"):
                self._build_structure_section()

            # Reviews Section
            with ui.expansion("Reviews & Notes", icon="rate_review").classes("w-full"):
                self._build_reviews_section()

    def _build_interview_section(self) -> None:
        """Build the interview chat section."""
        # Status indicator
        status = self.state.project.status if self.state.project else "unknown"
        color = get_status_color(status)

        with ui.row().classes("w-full items-center gap-2 mb-2"):
            ui.badge(status.title()).style(f"background-color: {color}22; color: {color};")

            if self.state.interview_complete:
                ui.label("Interview complete - structure ready").classes("text-sm text-gray-500")
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
            asyncio.create_task(self._start_interview())

        # Action buttons
        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            if not self.state.interview_complete:
                ui.button(
                    "Finalize Interview",
                    on_click=self._finalize_interview,
                ).props("flat")

            if self.state.interview_complete and self.state.project:
                if not self.state.project.chapters:
                    ui.button(
                        "Build Story Structure",
                        on_click=self._build_structure,
                    ).props("color=primary")

    def _build_world_overview(self) -> None:
        """Build the world overview section."""
        if not self.state.world_db:
            ui.label("No world data yet. Complete the interview first.").classes("text-gray-500")
            return

        # Entity summary cards
        self._entity_summary = ui.html(sanitize=False)
        self._entity_summary.content = render_entity_summary_html(self.state.world_db)

        # Mini graph preview
        ui.label("Relationship Graph").classes("text-sm font-medium mt-4")
        mini_graph(self.state.world_db, height=200)

        ui.button(
            "Open World Builder",
            on_click=lambda: setattr(self.state, "active_tab", "world"),
        ).props("flat").classes("mt-2")

    def _build_structure_section(self) -> None:
        """Build the story structure section."""
        if not self.state.project or not self.state.project.brief:
            ui.label("Complete the interview to see story structure.").classes("text-gray-500")
            return

        project = self.state.project
        brief = project.brief
        assert brief is not None  # Already checked above

        # Brief summary
        with ui.card().classes("w-full bg-gray-50"):
            ui.label("Story Brief").classes("text-lg font-semibold mb-2")

            with ui.row().classes("gap-4 flex-wrap"):
                with ui.column().classes("gap-1"):
                    ui.label("Genre").classes("text-xs text-gray-500")
                    ui.label(brief.genre).classes("font-medium")

                with ui.column().classes("gap-1"):
                    ui.label("Tone").classes("text-xs text-gray-500")
                    ui.label(brief.tone).classes("font-medium")

                with ui.column().classes("gap-1"):
                    ui.label("Setting").classes("text-xs text-gray-500")
                    ui.label(f"{brief.setting_place}, {brief.setting_time}")

                with ui.column().classes("gap-1"):
                    ui.label("Length").classes("text-xs text-gray-500")
                    ui.label(brief.target_length.replace("_", " ").title())

            ui.separator().classes("my-2")
            ui.label("Premise").classes("text-xs text-gray-500")
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

                with ui.row().classes("w-full items-start gap-2 p-2 hover:bg-gray-50 rounded"):
                    ui.icon(
                        "check_circle" if chapter.status == "final" else "radio_button_unchecked",
                        color=status_color,
                    )
                    with ui.column().classes("flex-grow gap-1"):
                        ui.label(f"Chapter {chapter.number}: {chapter.title}").classes(
                            "font-medium"
                        )
                        ui.label(
                            chapter.outline[:150] + "..."
                            if len(chapter.outline) > 150
                            else chapter.outline
                        ).classes("text-sm text-gray-600")
                    if chapter.word_count:
                        ui.badge(f"{chapter.word_count} words").props("outline")

    def _build_reviews_section(self) -> None:
        """Build the reviews and notes section."""
        if not self.state.project:
            return

        reviews = self.state.project.reviews

        if not reviews:
            ui.label("No reviews or notes yet.").classes("text-gray-500")

        for review in reviews:
            with ui.card().classes("w-full"):
                with ui.row().classes("items-center gap-2"):
                    ui.badge(review.get("type", "note"))
                    if review.get("chapter"):
                        ui.label(f"Chapter {review['chapter']}").classes("text-sm text-gray-500")
                ui.label(review.get("content", "")).classes("mt-2")

        # Add note form
        with ui.row().classes("w-full gap-2 mt-4"):
            note_input = ui.input(placeholder="Add a note...").classes("flex-grow")
            ui.button(
                "Add Note",
                on_click=lambda: self._add_note(note_input.value),
            ).props("flat")

    def _build_live_writing(self) -> None:
        """Build the Live Writing tab content."""
        if not self.state.can_write:
            with ui.column().classes("w-full h-full items-center justify-center gap-4"):
                ui.icon("edit_off", size="xl").classes("text-gray-400")
                ui.label("Story structure not ready").classes("text-xl text-gray-500")
                ui.label("Complete the interview and build the story structure first.").classes(
                    "text-gray-400"
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
                    f"{'bg-blue-50' if is_current else 'hover:bg-gray-50'}"
                )
                .on("click", lambda n=chapter.number: self._select_chapter(n))
            ):
                ui.icon(status_icon, size="sm")
                ui.label(f"{chapter.number}. {chapter.title[:15]}...").classes("truncate")

    def _build_writing_display(self) -> None:
        """Build the main writing display area."""
        # Header
        with ui.row().classes("w-full items-center"):
            ui.label("Story Content").classes("text-lg font-semibold")
            ui.space()
            self._word_count_label = ui.label("0 words").classes("text-sm text-gray-500")

        # Writing area
        self._writing_display = ui.markdown().classes(
            "w-full flex-grow p-4 bg-white rounded-lg border prose max-w-none overflow-auto"
        )

        # Load current chapter content
        self._refresh_writing_display()

    def _build_writing_controls(self) -> None:
        """Build the writing controls panel."""
        ui.label("Controls").classes("text-lg font-semibold")

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

        # Feedback mode
        ui.label("Feedback Mode").classes("text-sm font-medium")
        ui.select(
            options=["per-chapter", "mid-chapter", "on-demand"],
            value=self.state.feedback_mode,
            on_change=lambda e: setattr(self.state, "feedback_mode", e.value),
        ).classes("w-full")

        # Instant feedback
        with ui.expansion("Instant Feedback", icon="feedback").classes("w-full"):
            self._feedback_input = ui.textarea(
                placeholder="Enter feedback for the current chapter..."
            ).classes("w-full")
            ui.button(
                "Apply Feedback",
                on_click=self._apply_feedback,
            ).props("flat").classes("w-full")

        ui.separator()

        # Export
        ui.label("Export").classes("text-sm font-medium")
        with ui.row().classes("w-full gap-2"):
            ui.button("MD", on_click=lambda: self._export("markdown")).props("flat size=sm")
            ui.button("TXT", on_click=lambda: self._export("text")).props("flat size=sm")
            ui.button("EPUB", on_click=lambda: self._export("epub")).props("flat size=sm")
            ui.button("PDF", on_click=lambda: self._export("pdf")).props("flat size=sm")

    # ========== Event Handlers ==========

    async def _start_interview(self) -> None:
        """Start the interview process."""
        if not self.state.project or not self._chat:
            return

        try:
            self._chat.show_typing(True)
            questions = self.services.story.start_interview(self.state.project)
            self._chat.show_typing(False)
            self._chat.add_message("assistant", questions)
            self.state.add_interview_message("assistant", questions)
        except Exception as e:
            self._chat.show_typing(False)
            ui.notify(f"Error starting interview: {e}", type="negative")

    async def _handle_interview_message(self, message: str) -> None:
        """Handle user message in interview."""
        if not self.state.project or not self._chat:
            return

        try:
            self._chat.show_typing(True)
            self.state.add_interview_message("user", message)

            response, is_complete = self.services.story.process_interview(
                self.state.project, message
            )

            self._chat.show_typing(False)
            self._chat.add_message("assistant", response)
            self.state.add_interview_message("assistant", response)

            if is_complete:
                self.state.interview_complete = True
                self._chat.set_disabled(True)
                ui.notify(
                    "Interview complete! You can now build the story structure.", type="positive"
                )

            # Save progress
            self.services.project.save_project(self.state.project)

        except Exception as e:
            self._chat.show_typing(False)
            ui.notify(f"Error: {e}", type="negative")

    async def _finalize_interview(self) -> None:
        """Force finalize the interview."""
        if not self.state.project:
            return

        try:
            self.services.story.finalize_interview(self.state.project)
            self.state.interview_complete = True
            if self._chat:
                self._chat.set_disabled(True)
            self.services.project.save_project(self.state.project)
            ui.notify("Interview finalized!", type="positive")
        except Exception as e:
            ui.notify(f"Error: {e}", type="negative")

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

    async def _build_structure(self) -> None:
        """Build the story structure."""
        if not self.state.project or not self.state.world_db:
            return

        try:
            ui.notify("Building story structure...", type="info")
            self.services.story.build_structure(self.state.project, self.state.world_db)
            self.services.project.save_project(self.state.project)
            ui.notify("Story structure built!", type="positive")
            # Refresh the page
            ui.navigate.reload()
        except Exception as e:
            ui.notify(f"Error building structure: {e}", type="negative")

    def _on_chapter_select(self, e) -> None:
        """Handle chapter selection change."""
        self.state.select_chapter(e.value)
        self._refresh_writing_display()

    def _select_chapter(self, chapter_num: int) -> None:
        """Select a chapter by number."""
        self.state.select_chapter(chapter_num)
        if self._chapter_select:
            self._chapter_select.value = chapter_num
        self._refresh_writing_display()

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

    async def _write_current_chapter(self) -> None:
        """Write the current chapter."""
        if not self.state.project or not self.state.current_chapter:
            ui.notify("Select a chapter first", type="warning")
            return

        try:
            self.state.is_writing = True
            ui.notify(f"Writing chapter {self.state.current_chapter}...", type="info")

            for event in self.services.story.write_chapter(
                self.state.project, self.state.current_chapter
            ):
                self.state.writing_progress = event.message

            self.state.is_writing = False
            self._refresh_writing_display()
            self.services.project.save_project(self.state.project)
            ui.notify("Chapter complete!", type="positive")

        except Exception as e:
            self.state.is_writing = False
            ui.notify(f"Error: {e}", type="negative")

    async def _write_all_chapters(self) -> None:
        """Write all chapters."""
        if not self.state.project:
            return

        try:
            self.state.is_writing = True

            for event in self.services.story.write_all_chapters(self.state.project):
                self.state.writing_progress = event.message
                ui.notify(event.message, type="info")

            self.state.is_writing = False
            self._refresh_writing_display()
            self.services.project.save_project(self.state.project)
            ui.notify("All chapters complete!", type="positive")

        except Exception as e:
            self.state.is_writing = False
            ui.notify(f"Error: {e}", type="negative")

    def _apply_feedback(self) -> None:
        """Apply instant feedback to current chapter."""
        if not self._feedback_input or not self._feedback_input.value:
            return
        if not self.state.project or self.state.current_chapter is None:
            ui.notify("No chapter selected", type="warning")
            return

        feedback = self._feedback_input.value

        # Add as a review note for tracking and future revision
        self.services.story.add_review(
            self.state.project,
            review_type="user_feedback",
            content=feedback,
            chapter_num=self.state.current_chapter,
        )

        self.services.project.save_project(self.state.project)
        ui.notify(f"Feedback saved for chapter {self.state.current_chapter}", type="positive")
        self._feedback_input.value = ""

    def _add_note(self, content: str) -> None:
        """Add a review note."""
        if not content or not self.state.project:
            return

        self.services.story.add_review(
            self.state.project,
            review_type="user_note",
            content=content,
            chapter_num=self.state.current_chapter,
        )
        self.services.project.save_project(self.state.project)
        ui.notify("Note added", type="positive")

    async def _export(self, fmt: str) -> None:
        """Export the story."""
        if not self.state.project:
            return

        try:
            if fmt == "markdown":
                text_content = self.services.export.to_markdown(self.state.project)
                ui.download(text_content.encode(), f"{self.state.project.project_name}.md")
            elif fmt == "text":
                text_content = self.services.export.to_text(self.state.project)
                ui.download(text_content.encode(), f"{self.state.project.project_name}.txt")
            elif fmt == "epub":
                bytes_content = self.services.export.to_epub(self.state.project)
                ui.download(bytes_content, f"{self.state.project.project_name}.epub")
            elif fmt == "pdf":
                bytes_content = self.services.export.to_pdf(self.state.project)
                ui.download(bytes_content, f"{self.state.project.project_name}.pdf")

            ui.notify(f"Exported as {fmt.upper()}", type="positive")
        except Exception as e:
            ui.notify(f"Export failed: {e}", type="negative")

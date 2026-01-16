"""Write Story page - Fundamentals and Live Writing tabs."""

import asyncio
import html
import logging
from typing import Any, Literal

from nicegui import Client, context, run, ui
from nicegui.elements.button import Button
from nicegui.elements.html import Html
from nicegui.elements.label import Label
from nicegui.elements.markdown import Markdown
from nicegui.elements.select import Select
from nicegui.elements.textarea import Textarea

from memory.mode_models import PRESET_MODES
from services import ServiceContainer
from services.story_service import GenerationCancelled
from ui.components.chat import ChatComponent
from ui.components.generation_status import GenerationStatus
from ui.graph_renderer import render_entity_summary_html
from ui.state import AppState
from ui.theme import get_status_color
from workflows.orchestrator import WorkflowEvent

logger = logging.getLogger(__name__)


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
        self._client: Client | None = None  # For background task safety
        self._finalize_btn: Button | None = None
        self._build_structure_btn: Button | None = None
        self._scene_list_container: ui.column | None = None  # Container for scene list
        self._generation_status: GenerationStatus | None = None

    def _notify(
        self,
        message: str,
        type: Literal["positive", "negative", "warning", "info", "ongoing"] = "info",
    ) -> None:
        """Show notification safely from background tasks."""
        if self._client:
            with self._client:
                ui.notify(message, type=type)
        else:
            try:
                ui.notify(message, type=type)
            except RuntimeError:
                logger.warning(f"Could not show notification: {message}")

    def build(self) -> None:
        """Build the write page UI."""
        # Capture client for background task safety
        try:
            self._client = context.client
        except RuntimeError:
            logger.warning("Could not capture client context during build")

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
        from ui.components.common import empty_state

        empty_state(
            icon="folder_off",
            title="No Project Selected",
            description="Select a project from the header dropdown or create a new one.",
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
            asyncio.create_task(self._start_interview())

        # Action buttons - stored as instance vars for dynamic visibility
        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            self._finalize_btn = ui.button(
                "Finalize Interview",
                on_click=self._finalize_interview,
            ).props("flat")
            self._finalize_btn.set_visibility(not self.state.interview_complete)

            self._build_structure_btn = ui.button(
                "Build Story Structure",
                on_click=self._build_structure,
            ).props("color=primary")
            show_build = bool(
                self.state.interview_complete
                and self.state.project
                and not self.state.project.chapters
            )
            self._build_structure_btn.set_visibility(show_build)

    def _build_world_overview(self) -> None:
        """Build the world overview section."""
        if not self.state.world_db:
            ui.label("No world data yet. Complete the interview first.").classes(
                "text-gray-500 dark:text-gray-400"
            )
            return

        # Entity summary cards
        self._entity_summary = ui.html(sanitize=False)
        self._entity_summary.content = render_entity_summary_html(self.state.world_db)

        # Entity list (simple view instead of graph)
        characters = self.state.world_db.list_entities("character")
        if characters:
            ui.label("Characters").classes("text-sm font-medium mt-4")
            with ui.row().classes("flex-wrap gap-2"):
                for char in characters[:8]:  # Limit to 8 for preview
                    ui.chip(char.name, icon="person", color="green").props("outline")
                if len(characters) > 8:
                    ui.chip(f"+{len(characters) - 8} more", color="grey").props("outline")

        locations = self.state.world_db.list_entities("location")
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

    def _build_structure_section(self) -> None:
        """Build the story structure section."""
        if not self.state.project or not self.state.project.brief:
            ui.label("Complete the interview to see story structure.").classes(
                "text-gray-500 dark:text-gray-400"
            )
            return

        project = self.state.project
        brief = project.brief
        assert brief is not None  # Already checked above

        # Brief summary
        with ui.card().classes("w-full bg-gray-50 dark:bg-gray-800"):
            ui.label("Story Brief").classes("text-lg font-semibold mb-2")

            with ui.row().classes("gap-4 flex-wrap"):
                with ui.column().classes("gap-1"):
                    ui.label("Genre").classes("text-xs text-gray-500 dark:text-gray-400")
                    ui.label(brief.genre).classes("font-medium")

                with ui.column().classes("gap-1"):
                    ui.label("Tone").classes("text-xs text-gray-500 dark:text-gray-400")
                    ui.label(brief.tone).classes("font-medium")

                with ui.column().classes("gap-1"):
                    ui.label("Setting").classes("text-xs text-gray-500 dark:text-gray-400")
                    ui.label(f"{brief.setting_place}, {brief.setting_time}")

                with ui.column().classes("gap-1"):
                    ui.label("Length").classes("text-xs text-gray-500 dark:text-gray-400")
                    ui.label(brief.target_length.replace("_", " ").title())

            ui.separator().classes("my-2")
            ui.label("Premise").classes("text-xs text-gray-500 dark:text-gray-400")
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

                with ui.row().classes(
                    "w-full items-start gap-2 p-2 hover:bg-gray-50 dark:hover:bg-gray-700 rounded"
                ):
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
                        ).classes("text-sm text-gray-600 dark:text-gray-400")
                    if chapter.word_count:
                        ui.badge(f"{chapter.word_count} words").props("color=grey-7")

    def _build_reviews_section(self) -> None:
        """Build the reviews and notes section."""
        if not self.state.project:
            return

        reviews = self.state.project.reviews

        if not reviews:
            ui.label("No reviews or notes yet.").classes("text-gray-500 dark:text-gray-400")

        for review in reviews:
            with ui.card().classes("w-full"):
                with ui.row().classes("items-center gap-2"):
                    ui.badge(review.get("type", "note")).props("color=grey-7")
                    if review.get("chapter"):
                        ui.label(f"Chapter {review['chapter']}").classes(
                            "text-sm text-gray-500 dark:text-gray-400"
                        )
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
        from ui.components.common import empty_state

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

        # Writing area - use dark: prefix for automatic dark mode support
        self._writing_display = ui.markdown().classes(
            "w-full flex-grow p-4 bg-white dark:bg-gray-800 rounded-lg border dark:border-gray-700 prose dark:prose-invert max-w-none overflow-auto"
        )

        # Load current chapter content
        self._refresh_writing_display()

    def _build_writing_controls(self) -> None:
        """Build the writing controls panel."""
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

    def _build_scene_editor(self) -> None:
        """Build the scene editor component."""
        from ui.components.scene_editor import SceneListComponent

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

    # ========== Event Handlers ==========

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

        with dialog, ui.card().classes("w-96"):
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

    async def _build_structure(self) -> None:
        """Build the story structure with progress dialog."""
        if not self.state.project or not self.state.world_db:
            return

        brief = self.state.project.brief
        if not brief:
            self._notify("No story brief found. Complete the interview first.", type="warning")
            return

        # Create progress dialog
        dialog = ui.dialog().props("persistent")
        progress_label: Label
        progress_bar: ui.linear_progress

        async def do_build() -> None:
            """Execute the build with progress updates."""
            if not self.state.project or not self.state.world_db:
                dialog.close()
                return

            try:
                # Update progress
                progress_label.text = "Building world and characters..."
                progress_bar.value = 0.25
                await asyncio.sleep(0.1)  # Let UI update

                # Run the actual build
                await run.io_bound(
                    self.services.story.build_structure,
                    self.state.project,
                    self.state.world_db,
                )

                progress_label.text = "Saving project..."
                progress_bar.value = 0.9

                self.services.project.save_project(self.state.project)

                progress_bar.value = 1.0
                await asyncio.sleep(0.3)  # Show completion briefly

                dialog.close()

                # Update UI without full page reload
                self._update_interview_buttons()

                # Show completion with summary
                chapters = len(self.state.project.chapters)
                chars = len(self.state.project.characters)
                self._notify(
                    f"Story structure built! {chapters} chapters, {chars} characters. "
                    "Expand 'Story Structure' section to view.",
                    type="positive",
                )

            except Exception as e:
                logger.exception("Failed to build story structure")
                dialog.close()
                self._notify(f"Error: {e}", type="negative")

        with dialog, ui.card().classes("w-[500px]"):
            ui.label("Building Story Structure").classes("text-xl font-bold mb-4")

            # Show what we're building
            with ui.card().classes("w-full bg-gray-50 dark:bg-gray-800 mb-4"):
                ui.label("Story Overview:").classes("font-medium mb-2")
                ui.label(f"Genre: {brief.genre} • Tone: {brief.tone}").classes(
                    "text-sm text-gray-600 dark:text-gray-400"
                )
                ui.label(f"Length: {brief.target_length.replace('_', ' ').title()}").classes(
                    "text-sm text-gray-600 dark:text-gray-400"
                )
                ui.label(f"Premise: {brief.premise[:150]}...").classes(
                    "text-sm text-gray-600 dark:text-gray-400 mt-2"
                ) if len(brief.premise) > 150 else ui.label(f"Premise: {brief.premise}").classes(
                    "text-sm text-gray-600 dark:text-gray-400 mt-2"
                )

            ui.label("The AI will now:").classes("font-medium mb-2")
            with ui.column().classes("gap-1 mb-4"):
                ui.label("• Create detailed world description").classes("text-sm")
                ui.label("• Design main characters with backstories").classes("text-sm")
                ui.label("• Outline chapter structure and plot points").classes("text-sm")
                ui.label("• Establish story rules and timeline").classes("text-sm")

            progress_label = ui.label("Ready to build...").classes(
                "text-sm text-gray-500 dark:text-gray-400 mb-2"
            )
            progress_bar = ui.linear_progress(value=0, show_value=False).classes("w-full mb-4")

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button("Build Structure", on_click=do_build).props("color=primary")

        dialog.open()

    def _on_chapter_select(self, e: Any) -> None:
        """Handle chapter selection change."""
        self.state.select_chapter(e.value)
        self._refresh_writing_display()
        self._refresh_scene_editor()

    def _select_chapter(self, chapter_num: int) -> None:
        """Select a chapter by number."""
        self.state.select_chapter(chapter_num)
        if self._chapter_select:
            self._chapter_select.value = chapter_num
        self._refresh_writing_display()
        self._refresh_scene_editor()

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
        """Write the current chapter with background processing and live updates."""
        if not self.state.project or not self.state.current_chapter:
            self._notify("Select a chapter first", type="warning")
            return

        # Prevent multiple concurrent generations
        if self.state.is_writing:
            self._notify("Generation already in progress", type="warning")
            return

        # Capture for closure type narrowing
        project = self.state.project
        chapter_num = self.state.current_chapter

        try:
            # Reset generation flags
            self.state.reset_generation_flags()
            self.state.is_writing = True

            # Show generation status
            if self._generation_status:
                self._generation_status.show()
                self._generation_status.update_progress(f"Starting chapter {chapter_num}...")

            self._notify(f"Writing chapter {chapter_num}...", type="info")

            # Define cancellation check
            def should_cancel() -> bool:
                return self.state.generation_cancel_requested

            # Run generation in background with progressive updates
            async def background_generation():
                """Run generation in a background thread and yield events."""
                events = []
                try:
                    # Run blocking generator in thread pool
                    def write_chapter_blocking():
                        return list(
                            self.services.story.write_chapter(
                                project, chapter_num, cancel_check=should_cancel
                            )
                        )

                    events = await run.io_bound(write_chapter_blocking)
                    return events, None
                except GenerationCancelled as e:
                    logger.info(f"Chapter {chapter_num} generation cancelled")
                    return [], e
                except Exception as e:
                    logger.exception(f"Failed to write chapter {chapter_num}")
                    return [], e

            # Process events with UI updates
            events, error = await background_generation()

            # Update UI based on results
            if error:
                if isinstance(error, GenerationCancelled):
                    self._notify("Generation cancelled by user", type="warning")
                else:
                    self._notify(f"Error: {error}", type="negative")
            else:
                # Process events for progress display
                for event in events:
                    self.state.writing_progress = event.message
                    if self._generation_status:
                        with self._client:
                            self._generation_status.update_progress(event.message)

                self._refresh_writing_display()
                self.services.project.save_project(self.state.project)
                self._notify("Chapter complete!", type="positive")

        finally:
            self.state.is_writing = False
            self.state.reset_generation_flags()
            if self._generation_status:
                with self._client:
                    self._generation_status.hide()

    async def _write_all_chapters(self) -> None:
        """Write all chapters with background processing and live updates."""
        if not self.state.project:
            return

        # Prevent multiple concurrent generations
        if self.state.is_writing:
            self._notify("Generation already in progress", type="warning")
            return

        # Capture for closure type narrowing
        project = self.state.project

        try:
            # Reset generation flags
            self.state.reset_generation_flags()
            self.state.is_writing = True

            # Show generation status
            if self._generation_status:
                self._generation_status.show()
                self._generation_status.update_progress("Starting story generation...")

            self._notify("Writing all chapters...", type="info")

            # Define cancellation check
            def should_cancel() -> bool:
                return self.state.generation_cancel_requested

            # Run generation in background with progressive updates
            async def background_generation():
                """Run generation in a background thread and yield events."""
                events = []
                try:
                    # Run blocking generator in thread pool
                    def write_all_blocking():
                        return list(
                            self.services.story.write_all_chapters(
                                project, cancel_check=should_cancel
                            )
                        )

                    events = await run.io_bound(write_all_blocking)
                    return events, None
                except GenerationCancelled as e:
                    logger.info("Write all chapters cancelled")
                    return [], e
                except Exception as e:
                    logger.exception("Failed to write all chapters")
                    return [], e

            # Process events with UI updates
            events, error = await background_generation()

            # Update UI based on results
            if error:
                if isinstance(error, GenerationCancelled):
                    self._notify("Generation cancelled by user", type="warning")
                else:
                    self._notify(f"Error: {error}", type="negative")
            else:
                # Process events for progress display
                for event in events:
                    self.state.writing_progress = event.message
                    if self._generation_status:
                        with self._client:
                            self._generation_status.update_progress(event.message)

                self._refresh_writing_display()
                self.services.project.save_project(self.state.project)
                self._notify("All chapters complete!", type="positive")

        finally:
            self.state.is_writing = False
            self.state.reset_generation_flags()
            if self._generation_status:
                with self._client:
                    self._generation_status.hide()

    def _apply_feedback(self) -> None:
        """Apply instant feedback to current chapter."""
        if not self._feedback_input or not self._feedback_input.value:
            return
        if not self.state.project or self.state.current_chapter is None:
            self._notify("No chapter selected", type="warning")
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
        self._notify(f"Feedback saved for chapter {self.state.current_chapter}", type="positive")
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
        self._notify("Note added", type="positive")

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

            self._notify(f"Exported as {fmt.upper()}", type="positive")
        except PermissionError:
            logger.exception(f"Permission denied exporting as {fmt}")
            self._notify(
                "Export failed: Permission denied. Check file permissions.", type="negative"
            )
        except OSError as e:
            logger.exception(f"OS error exporting as {fmt}")
            if "No space" in str(e) or "disk full" in str(e).lower():
                self._notify("Export failed: Insufficient disk space.", type="negative")
            else:
                self._notify(f"Export failed: {e}", type="negative")
        except ValueError as e:
            logger.exception(f"Value error exporting as {fmt}")
            self._notify(f"Export failed: {e}. Check chapter content.", type="negative")
        except Exception as e:
            logger.exception(f"Failed to export as {fmt}")
            self._notify(f"Export failed: {e}", type="negative")

    async def _show_suggestions(self) -> None:
        """Show AI-powered writing suggestions dialog."""
        logger.debug("Opening writing suggestions dialog")
        if not self.state.project:
            self._notify("No project selected", type="warning")
            return

        # Create dialog
        dialog = ui.dialog().props("maximized")
        suggestions_html: Html | None = None
        loading_spinner: ui.spinner | None = None

        async def load_suggestions():
            """Load suggestions from the service."""
            nonlocal suggestions_html, loading_spinner

            if not self.state.project:
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
                    self.services.suggestion.generate_suggestions, self.state.project
                )

                logger.debug(f"Received suggestions with {len(suggestions)} categories")

                # Hide loading
                if loading_spinner:
                    loading_spinner.set_visibility(False)

                # Build HTML for suggestions
                html_content = self._build_suggestions_html(suggestions)

                # Update display
                if suggestions_html:
                    suggestions_html.content = html_content
                    suggestions_html.set_visibility(True)
                    logger.info("Suggestions loaded and displayed successfully")

            except Exception as e:
                logger.exception("Failed to generate suggestions")
                if loading_spinner:
                    loading_spinner.set_visibility(False)
                self._notify(f"Error generating suggestions: {e}", type="negative")

        with dialog, ui.card().classes("w-full h-full flex flex-col"):
            # Header
            with ui.row().classes("w-full items-center justify-between p-4 border-b"):
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
            with ui.row().classes("w-full justify-end gap-2 p-4 border-t"):
                ui.button("Refresh", on_click=load_suggestions, icon="refresh").props("flat")

                def close_dialog():
                    logger.debug("Closing suggestions dialog")
                    dialog.close()

                ui.button("Close", on_click=close_dialog).props("color=primary")

        # Open dialog and load suggestions
        logger.info("Opening suggestions dialog")
        dialog.open()
        await load_suggestions()

    def _build_suggestions_html(self, suggestions: dict[str, list[str]]) -> str:
        """Build HTML content for suggestions display.

        Args:
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
            safe_icon = html.escape(info["icon"])
            safe_title = html.escape(info["title"])
            safe_color = html.escape(info["color"])

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
                safe_suggestion = html.escape(suggestion)
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

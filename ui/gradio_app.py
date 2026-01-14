"""Gradio-based UI for Story Factory."""

import gradio as gr
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflows.orchestrator import StoryOrchestrator
from config import InteractionMode, STORY_LENGTHS


class StoryFactoryUI:
    """Gradio UI for the Story Factory."""

    def __init__(self):
        self.orchestrator: StoryOrchestrator = None
        self.interview_complete = False
        self.outline_approved = False

    def start_new_story(self, interaction_mode: str):
        """Initialize a new story session."""
        self.orchestrator = StoryOrchestrator(interaction_mode=interaction_mode)
        self.orchestrator.create_new_story()
        self.interview_complete = False
        self.outline_approved = False

        initial_questions = self.orchestrator.start_interview()
        return (
            initial_questions,
            "",  # Clear story output
            "Interview started. Answer the questions to shape your story.",
            gr.update(interactive=True),  # Enable chat input
        )

    def chat_response(self, message: str, history: list):
        """Handle chat messages during interview or writing."""
        if not self.orchestrator:
            return history + [[message, "Please start a new story first."]], ""

        if not self.interview_complete:
            # Interview phase
            response, is_complete = self.orchestrator.process_interview_response(message)
            self.interview_complete = is_complete

            if is_complete:
                response += "\n\n---\n**Interview complete!** Click 'Build Story Structure' to continue."

            return history + [[message, response]], ""

        return history + [[message, "Use the action buttons to continue."]], ""

    def build_structure(self, progress_output):
        """Build the story structure."""
        if not self.orchestrator or not self.interview_complete:
            return "Please complete the interview first.", ""

        yield "Building story structure...\n\nCreating world...", ""

        self.orchestrator.build_story_structure()

        outline = self.orchestrator.get_outline_summary()
        yield outline, "Structure complete! Review the outline above, then click 'Approve & Write' to begin writing."

    def write_story(self):
        """Write the story."""
        if not self.orchestrator:
            yield "Please start a new story first.", "", ""
            return

        state = self.orchestrator.story_state
        brief = state.brief

        # Determine if short story or multi-chapter
        if brief.target_length == "short_story":
            yield "Writing short story...", "", "Writer is working..."

            for event in self.orchestrator.write_short_story():
                yield "", "", f"{event.agent_name}: {event.message}"

            story = self.orchestrator.get_full_story()
            stats = self.orchestrator.get_statistics()
            yield story, f"Complete! {stats['total_words']} words", "Story complete!"

        else:
            # Multi-chapter
            total_chapters = len(state.chapters)
            full_story = ""

            for i, chapter in enumerate(state.chapters):
                yield full_story, f"Writing chapter {chapter.number}/{total_chapters}...", f"Working on: {chapter.title}"

                for event in self.orchestrator.write_chapter(chapter.number):
                    yield full_story, f"Chapter {chapter.number}: {event.message}", f"{event.agent_name}: {event.message}"

                full_story = self.orchestrator.get_full_story()
                yield full_story, f"Chapter {chapter.number} complete", f"Finished: {chapter.title}"

            stats = self.orchestrator.get_statistics()
            yield full_story, f"Complete! {stats['total_words']} words across {stats['total_chapters']} chapters", "All chapters complete!"

    def export_markdown(self):
        """Export story as markdown."""
        if not self.orchestrator or not self.orchestrator.story_state.chapters:
            return "No story to export."

        return self.orchestrator.export_to_markdown()

    def build_ui(self):
        """Build and return the Gradio interface."""
        with gr.Blocks(title="Story Factory", theme=gr.themes.Soft()) as app:
            gr.Markdown(
                """
                # Story Factory
                ### AI-Powered Story Production Team

                Your personal writing team: **Interviewer** → **Architect** → **Writer** → **Editor** → **Continuity Checker**
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Settings")
                    interaction_mode = gr.Radio(
                        choices=[
                            ("Minimal - Just start & end", InteractionMode.MINIMAL),
                            ("Checkpoint - Review every few chapters", InteractionMode.CHECKPOINT),
                            ("Interactive - Review each chapter", InteractionMode.INTERACTIVE),
                        ],
                        value=InteractionMode.CHECKPOINT,
                        label="Interaction Mode",
                    )
                    start_btn = gr.Button("Start New Story", variant="primary")

                    gr.Markdown("### Actions")
                    build_btn = gr.Button("Build Story Structure", interactive=False)
                    write_btn = gr.Button("Approve & Write Story", variant="primary", interactive=False)
                    export_btn = gr.Button("Export as Markdown")

                    status_box = gr.Textbox(
                        label="Status",
                        value="Click 'Start New Story' to begin",
                        interactive=False,
                        lines=2,
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### Interview")
                    chatbot = gr.Chatbot(
                        label="Chat with the Interviewer",
                        height=300,
                    )
                    chat_input = gr.Textbox(
                        label="Your response",
                        placeholder="Type your answer here...",
                        interactive=False,
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Story Outline")
                    outline_display = gr.Textbox(
                        label="Outline",
                        lines=15,
                        interactive=False,
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Story Output")
                    story_display = gr.Textbox(
                        label="Your Story",
                        lines=20,
                        interactive=False,
                    )
                    progress_display = gr.Textbox(
                        label="Progress",
                        lines=2,
                        interactive=False,
                    )

            # Event handlers
            start_btn.click(
                self.start_new_story,
                inputs=[interaction_mode],
                outputs=[chatbot, story_display, status_box, chat_input],
            ).then(
                lambda: (gr.update(interactive=True), gr.update(interactive=False)),
                outputs=[build_btn, write_btn],
            )

            chat_input.submit(
                self.chat_response,
                inputs=[chat_input, chatbot],
                outputs=[chatbot, chat_input],
            ).then(
                lambda: "",
                outputs=[chat_input],
            )

            build_btn.click(
                self.build_structure,
                inputs=[progress_display],
                outputs=[outline_display, status_box],
            ).then(
                lambda: gr.update(interactive=True),
                outputs=[write_btn],
            )

            write_btn.click(
                self.write_story,
                outputs=[story_display, progress_display, status_box],
            )

            export_btn.click(
                self.export_markdown,
                outputs=[story_display],
            )

        return app


def main():
    """Run the Gradio app."""
    ui = StoryFactoryUI()
    app = ui.build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()

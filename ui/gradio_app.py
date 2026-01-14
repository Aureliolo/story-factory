"""Gradio-based UI for Story Factory with web-based configuration."""

import gradio as gr
import sys
import os
import subprocess
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflows.orchestrator import StoryOrchestrator
from settings import (
    Settings, AVAILABLE_MODELS, AGENT_ROLES,
    get_installed_models, get_available_vram, get_model_info
)


class StoryFactoryUI:
    """Gradio UI for the Story Factory."""

    def __init__(self):
        self.settings = Settings.load()
        self.orchestrator: StoryOrchestrator = None
        self.interview_complete = False
        self.detected_vram = get_available_vram()

    # ============ Settings Tab Functions ============

    def get_installed_models_list(self):
        """Get formatted list of installed models."""
        installed = get_installed_models()
        if not installed:
            return "No models installed. Pull models using the button below."

        lines = []
        for model in installed:
            info = get_model_info(model)
            status = "Ready"
            lines.append(f"- **{info.get('name', model)}** ({model}) - {status}")
        return "\n".join(lines)

    def get_available_models_list(self):
        """Get formatted list of available models to download."""
        installed = set(get_installed_models())
        lines = []
        for model_id, info in AVAILABLE_MODELS.items():
            status = "Installed" if model_id in installed else f"~{info['size_gb']}GB download"
            vram_ok = "Yes" if info['vram_required'] <= self.detected_vram else "No"
            lines.append(
                f"| {info['name']} | {info['release']} | {info['quality']}/10 | "
                f"{info['vram_required']}GB | {status} |"
            )
        header = "| Model | Release | Quality | VRAM | Status |\n|-------|---------|---------|------|--------|\n"
        return header + "\n".join(lines)

    def pull_model(self, model_id: str, progress=gr.Progress()):
        """Pull a model from Ollama."""
        if not model_id:
            return "Please select a model to pull."

        progress(0, desc=f"Pulling {model_id}...")

        try:
            process = subprocess.Popen(
                ["ollama", "pull", model_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            output_lines = []
            for line in process.stdout:
                output_lines.append(line.strip())
                if "pulling" in line.lower():
                    progress(0.5, desc=line.strip()[:50])

            process.wait()

            if process.returncode == 0:
                return f"Successfully pulled {model_id}"
            else:
                return f"Error pulling model:\n" + "\n".join(output_lines[-10:])
        except Exception as e:
            return f"Error: {str(e)}"

    def save_settings(
        self,
        default_model,
        context_size,
        interaction_mode,
        chapters_between,
        use_per_agent,
        interviewer_model,
        architect_model,
        writer_model,
        editor_model,
        continuity_model,
    ):
        """Save settings from the UI."""
        self.settings.default_model = default_model
        self.settings.context_size = int(context_size)
        self.settings.interaction_mode = interaction_mode
        self.settings.chapters_between_checkpoints = int(chapters_between)
        self.settings.use_per_agent_models = use_per_agent

        if use_per_agent:
            self.settings.agent_models = {
                "interviewer": interviewer_model,
                "architect": architect_model,
                "writer": writer_model,
                "editor": editor_model,
                "continuity": continuity_model,
            }

        self.settings.save()
        return "Settings saved successfully!"

    # ============ Story Generation Functions ============

    def start_new_story(self):
        """Initialize a new story session."""
        self.settings = Settings.load()  # Reload settings
        self.orchestrator = StoryOrchestrator(settings=self.settings)
        self.orchestrator.create_new_story()
        self.interview_complete = False

        initial_questions = self.orchestrator.start_interview()

        # Show which models are being used
        models_info = f"""**Models in use:**
- Interviewer: {self.orchestrator.interviewer.model}
- Architect: {self.orchestrator.architect.model}
- Writer: {self.orchestrator.writer.model}
- Editor: {self.orchestrator.editor.model}
- Continuity: {self.orchestrator.continuity.model}
"""
        return (
            [[None, initial_questions]],
            "",
            models_info,
            gr.update(interactive=True),
        )

    def chat_response(self, message: str, history: list):
        """Handle chat messages during interview."""
        if not self.orchestrator:
            return history + [[message, "Please start a new story first."]], ""

        if not self.interview_complete:
            response, is_complete = self.orchestrator.process_interview_response(message)
            self.interview_complete = is_complete

            if is_complete:
                response += "\n\n---\n**Interview complete!** Click 'Build Story Structure' to continue."

            return history + [[message, response]], ""

        return history + [[message, "Use the action buttons to continue."]], ""

    def build_structure(self):
        """Build the story structure."""
        if not self.orchestrator or not self.interview_complete:
            yield "Please complete the interview first.", ""
            return

        yield "Building story structure...\n\nCreating world and characters...", "Architect is working..."

        self.orchestrator.build_story_structure()

        outline = self.orchestrator.get_outline_summary()
        yield outline, "Structure complete! Review and click 'Write Story' to begin."

    def write_story(self):
        """Write the story."""
        if not self.orchestrator:
            yield "Please start a new story first.", "", ""
            return

        state = self.orchestrator.story_state
        brief = state.brief

        if brief.target_length == "short_story":
            yield "Writing short story...", "", "Writer is working..."

            for event in self.orchestrator.write_short_story():
                yield "", "", f"{event.agent_name}: {event.message}"

            story = self.orchestrator.get_full_story()
            stats = self.orchestrator.get_statistics()
            yield story, f"Complete! {stats['total_words']} words", "Story complete!"

        else:
            total_chapters = len(state.chapters)
            full_story = ""

            for chapter in state.chapters:
                yield full_story, f"Writing chapter {chapter.number}/{total_chapters}...", f"Working on: {chapter.title}"

                for event in self.orchestrator.write_chapter(chapter.number):
                    yield full_story, f"Chapter {chapter.number}: {event.message}", f"{event.agent_name}: {event.message}"

                full_story = self.orchestrator.get_full_story()
                yield full_story, f"Chapter {chapter.number} complete", f"Finished: {chapter.title}"

            stats = self.orchestrator.get_statistics()
            yield full_story, f"Complete! {stats['total_words']} words across {stats['total_chapters']} chapters", "All chapters complete!"

    # ============ Comparison Functions ============

    def run_comparison(self, prompt: str, selected_models: list):
        """Run the same prompt through multiple models for comparison."""
        if not prompt:
            yield "Please enter a story prompt.", ""
            return

        if not selected_models or len(selected_models) < 2:
            yield "Please select at least 2 models to compare.", ""
            return

        results = {}
        output_parts = []

        for i, model in enumerate(selected_models):
            model_info = get_model_info(model)
            yield f"Running model {i+1}/{len(selected_models)}: {model_info.get('name', model)}...", ""

            try:
                # Create a fresh orchestrator with this model
                test_orchestrator = StoryOrchestrator(
                    settings=self.settings,
                    model_override=model
                )

                # Generate a short sample
                start_time = datetime.now()
                response = test_orchestrator.writer.generate(
                    f"Write a single compelling paragraph (100-150 words) for this story concept:\n\n{prompt}\n\nWrite only the story paragraph, nothing else."
                )
                elapsed = (datetime.now() - start_time).total_seconds()

                results[model] = {
                    "output": response,
                    "time": elapsed,
                    "model_info": model_info,
                }

                output_parts.append(f"""
## {model_info.get('name', model)}
**Model:** `{model}`
**Time:** {elapsed:.1f}s
**Quality Rating:** {model_info.get('quality', '?')}/10

### Output:
{response}

---
""")

            except Exception as e:
                output_parts.append(f"""
## {model_info.get('name', model)}
**Error:** {str(e)}

---
""")

        final_output = "# Model Comparison Results\n\n" + "\n".join(output_parts)
        yield final_output, "Comparison complete!"

    # ============ Build UI ============

    def build_ui(self):
        """Build and return the Gradio interface."""
        installed_models = get_installed_models()
        model_choices = list(AVAILABLE_MODELS.keys())

        with gr.Blocks(title="Story Factory") as app:
            gr.Markdown(
                f"""
                # Story Factory
                ### AI-Powered Story Production Team
                **Detected VRAM:** {self.detected_vram}GB | **Installed Models:** {len(installed_models)}
                """
            )

            with gr.Tabs():
                # ============ WRITE TAB ============
                with gr.Tab("Write Story"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Controls")
                            start_btn = gr.Button("Start New Story", variant="primary")
                            build_btn = gr.Button("Build Story Structure")
                            write_btn = gr.Button("Write Story", variant="primary")

                            status_box = gr.Textbox(
                                label="Status",
                                value="Click 'Start New Story' to begin",
                                interactive=False,
                                lines=8,
                            )

                        with gr.Column(scale=2):
                            gr.Markdown("### Interview")
                            chatbot = gr.Chatbot(label="Chat with the Interviewer", height=300)
                            chat_input = gr.Textbox(
                                label="Your response",
                                placeholder="Type your answer here...",
                                interactive=False,
                            )

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Story Outline")
                            outline_display = gr.Textbox(label="Outline", lines=10, interactive=False)

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Story Output")
                            story_display = gr.Textbox(label="Your Story", lines=15, interactive=False)
                            progress_display = gr.Textbox(label="Progress", lines=2, interactive=False)

                # ============ COMPARE TAB ============
                with gr.Tab("Compare Models"):
                    gr.Markdown("""
                    ### Model Comparison
                    Test the same story prompt across different models to compare output quality.
                    """)

                    compare_prompt = gr.Textbox(
                        label="Story Prompt",
                        placeholder="Enter a story concept to test (e.g., 'A detective discovers their partner is the killer')",
                        lines=3,
                    )

                    compare_models = gr.CheckboxGroup(
                        choices=[(get_model_info(m).get('name', m), m) for m in installed_models] if installed_models else [],
                        label="Select Models to Compare (choose 2+)",
                    )

                    compare_btn = gr.Button("Run Comparison", variant="primary")

                    compare_output = gr.Markdown(label="Comparison Results")
                    compare_status = gr.Textbox(label="Status", interactive=False)

                # ============ SETTINGS TAB ============
                with gr.Tab("Settings"):
                    gr.Markdown("### Model Settings")

                    with gr.Row():
                        with gr.Column():
                            default_model = gr.Dropdown(
                                choices=model_choices,
                                value=self.settings.default_model,
                                label="Default Model",
                            )

                            context_size = gr.Slider(
                                minimum=2048,
                                maximum=65536,
                                step=1024,
                                value=self.settings.context_size,
                                label="Context Size (tokens)",
                            )

                        with gr.Column():
                            interaction_mode = gr.Radio(
                                choices=["minimal", "checkpoint", "interactive", "collaborative"],
                                value=self.settings.interaction_mode,
                                label="Interaction Mode",
                            )

                            chapters_between = gr.Slider(
                                minimum=1,
                                maximum=10,
                                step=1,
                                value=self.settings.chapters_between_checkpoints,
                                label="Chapters Between Checkpoints",
                            )

                    gr.Markdown("### Per-Agent Model Settings")

                    use_per_agent = gr.Checkbox(
                        value=self.settings.use_per_agent_models,
                        label="Use different models for different agents",
                    )

                    with gr.Row():
                        with gr.Column():
                            interviewer_model = gr.Dropdown(
                                choices=["auto"] + model_choices,
                                value=self.settings.agent_models.get("interviewer", "auto"),
                                label="Interviewer Model",
                            )
                            architect_model = gr.Dropdown(
                                choices=["auto"] + model_choices,
                                value=self.settings.agent_models.get("architect", "auto"),
                                label="Architect Model",
                            )
                            writer_model = gr.Dropdown(
                                choices=["auto"] + model_choices,
                                value=self.settings.agent_models.get("writer", "auto"),
                                label="Writer Model (highest quality)",
                            )

                        with gr.Column():
                            editor_model = gr.Dropdown(
                                choices=["auto"] + model_choices,
                                value=self.settings.agent_models.get("editor", "auto"),
                                label="Editor Model",
                            )
                            continuity_model = gr.Dropdown(
                                choices=["auto"] + model_choices,
                                value=self.settings.agent_models.get("continuity", "auto"),
                                label="Continuity Checker Model",
                            )

                    save_btn = gr.Button("Save Settings", variant="primary")
                    settings_status = gr.Textbox(label="Status", interactive=False)

                # ============ MODELS TAB ============
                with gr.Tab("Manage Models"):
                    gr.Markdown("### Installed Models")
                    installed_display = gr.Markdown(self.get_installed_models_list())

                    refresh_btn = gr.Button("Refresh List")

                    gr.Markdown("### Available Models (2024-2025)")
                    available_display = gr.Markdown(self.get_available_models_list())

                    gr.Markdown("### Pull New Model")
                    with gr.Row():
                        model_to_pull = gr.Dropdown(
                            choices=model_choices,
                            label="Select Model",
                        )
                        pull_btn = gr.Button("Pull Model")

                    pull_status = gr.Textbox(label="Status", interactive=False)

            # ============ Event Handlers ============

            # Write tab
            start_btn.click(
                self.start_new_story,
                outputs=[chatbot, story_display, status_box, chat_input],
            )

            chat_input.submit(
                self.chat_response,
                inputs=[chat_input, chatbot],
                outputs=[chatbot, chat_input],
            ).then(lambda: "", outputs=[chat_input])

            build_btn.click(
                self.build_structure,
                outputs=[outline_display, status_box],
            )

            write_btn.click(
                self.write_story,
                outputs=[story_display, progress_display, status_box],
            )

            # Compare tab
            compare_btn.click(
                self.run_comparison,
                inputs=[compare_prompt, compare_models],
                outputs=[compare_output, compare_status],
            )

            # Settings tab
            save_btn.click(
                self.save_settings,
                inputs=[
                    default_model, context_size, interaction_mode, chapters_between,
                    use_per_agent, interviewer_model, architect_model, writer_model,
                    editor_model, continuity_model,
                ],
                outputs=[settings_status],
            )

            # Models tab
            refresh_btn.click(
                self.get_installed_models_list,
                outputs=[installed_display],
            )

            pull_btn.click(
                self.pull_model,
                inputs=[model_to_pull],
                outputs=[pull_status],
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

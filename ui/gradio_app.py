"""Gradio-based UI for Story Factory with web-based configuration."""

import gradio as gr
import logging
import sys
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflows.orchestrator import StoryOrchestrator
from agents.base import BaseAgent, LLMError
from settings import (
    Settings, AVAILABLE_MODELS, AGENT_ROLES,
    get_installed_models, get_available_vram, get_model_info
)

logger = logging.getLogger(__name__)


class StoryFactoryUI:
    """Gradio UI for the Story Factory."""

    def __init__(self):
        self.settings = Settings.load()
        self.orchestrator: StoryOrchestrator = None
        self.interview_complete = False
        self.detected_vram = get_available_vram()
        self.ollama_status = self._check_ollama()
        self.model_warnings = self._validate_models()

    def _check_ollama(self) -> tuple[bool, str]:
        """Check Ollama connectivity at startup."""
        return BaseAgent.check_ollama_health(self.settings.ollama_url)

    def _validate_models(self) -> list[str]:
        """Validate that configured models are installed."""
        warnings = []
        installed = set(get_installed_models())

        if not installed:
            warnings.append("No models installed. Please pull a model first.")
            return warnings

        # Check default model
        if self.settings.default_model not in installed:
            warnings.append(f"Default model '{self.settings.default_model}' not installed.")

        # Check agent models
        if self.settings.use_per_agent_models:
            for role, model in self.settings.agent_models.items():
                if model != "auto" and model not in installed:
                    warnings.append(f"{role.title()} model '{model}' not installed.")

        return warnings

    # ============ Settings Tab Functions ============

    def _normalize_model_name(self, model: str) -> str:
        """Normalize model name by removing :latest tag for comparison."""
        if model.endswith(":latest"):
            return model[:-7]  # Remove ':latest'
        return model

    def _is_model_installed(self, model_id: str, installed_set: set) -> bool:
        """Check if a model is installed, handling :latest tag variations."""
        # Direct match
        if model_id in installed_set:
            return True
        # Check with :latest suffix
        if f"{model_id}:latest" in installed_set:
            return True
        # Check without tag if model_id has a tag
        base_name = self._normalize_model_name(model_id)
        if base_name in installed_set or f"{base_name}:latest" in installed_set:
            return True
        return False

    def get_installed_models_list(self):
        """Get formatted list of installed models."""
        installed = get_installed_models()
        if not installed:
            return "No models installed. Pull models using the button below."

        lines = []
        for model in installed:
            # Try to get info with normalized name
            normalized = self._normalize_model_name(model)
            info = get_model_info(normalized)
            if info.get('name') == normalized:
                # Fallback didn't find it, try original
                info = get_model_info(model)
            display_name = info.get('name', model)
            lines.append(f"- **{display_name}** (`{model}`) - Ready")
        return "\n".join(lines)

    def get_available_models_list(self):
        """Get formatted list of available models to download."""
        installed = set(get_installed_models())
        lines = []
        for model_id, info in AVAILABLE_MODELS.items():
            is_installed = self._is_model_installed(model_id, installed)
            status = "Installed" if is_installed else f"~{info['size_gb']}GB download"
            lines.append(
                f"| {info['name']} | {info['release']} | {info['quality']}/10 | "
                f"{info['vram_required']}GB | {status} |"
            )
        header = "| Model | Release | Quality | VRAM | Status |\n|-------|---------|---------|------|--------|\n"
        return header + "\n".join(lines)

    def refresh_models_tab(self):
        """Refresh all model-related displays. Returns (installed_md, available_md, dropdown_choices)."""
        logger.debug("Refreshing models tab data")
        installed_md = self.get_installed_models_list()
        available_md = self.get_available_models_list()
        # Return updated dropdown choices
        installed = get_installed_models()
        dropdown_choices = list(AVAILABLE_MODELS.keys())
        return installed_md, available_md, gr.update(choices=dropdown_choices)

    def pull_model(self, model_id: str, progress=gr.Progress(track_tqdm=True)):
        """Pull a model from Ollama."""
        if not model_id:
            logger.warning("Pull attempted with no model selected")
            return "Please select a model to pull."

        logger.info(f"Starting pull of model: {model_id}")

        try:
            # Update progress if available
            if progress is not None:
                try:
                    progress(0, desc=f"Starting pull of {model_id}...")
                except Exception:
                    pass  # Progress callback might fail, continue anyway

            # Use shell=True on Windows for proper command resolution
            import platform
            use_shell = platform.system() == "Windows"

            process = subprocess.Popen(
                ["ollama", "pull", model_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=use_shell,
                encoding='utf-8',
                errors='replace',  # Handle non-UTF8 chars from progress bars
            )

            output_lines = []
            for line in process.stdout:
                line_stripped = line.strip()
                if line_stripped:
                    output_lines.append(line_stripped)
                    # Update progress if available
                    if progress is not None and "pulling" in line.lower():
                        try:
                            progress(0.5, desc=line_stripped[:50])
                        except Exception:
                            pass

            process.wait()

            if process.returncode == 0:
                logger.info(f"Successfully pulled model: {model_id}")
                # Refresh installed models list
                self._refresh_models()
                return f"Successfully pulled {model_id}"
            else:
                error_output = "\n".join(output_lines[-10:]) if output_lines else "No output captured"
                logger.error(f"Failed to pull {model_id} (exit {process.returncode}): {error_output}")
                return f"Error pulling model (exit code {process.returncode}):\n{error_output}"
        except FileNotFoundError:
            logger.error("ollama command not found in PATH")
            return "Error: 'ollama' command not found. Is Ollama installed and in PATH?"
        except Exception as e:
            logger.exception(f"Exception pulling model {model_id}")
            return f"Error: {type(e).__name__}: {str(e)}"

    def _refresh_models(self):
        """Refresh the list of installed models."""
        try:
            from settings import get_installed_models
            self.installed_models = get_installed_models()
        except Exception:
            pass

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
        try:
            self.settings = Settings.load()  # Reload settings

            # Reset existing orchestrator state if any
            if self.orchestrator:
                self.orchestrator.reset_state()

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
                [{"role": "assistant", "content": initial_questions}],
                "",
                models_info,
                gr.update(interactive=True),
            )
        except Exception as e:
            logger.exception("start_new_story failed")
            return (
                [{"role": "assistant", "content": f"Error starting story: {e}"}],
                "",
                f"Error: {e}",
                gr.update(interactive=False),
            )

    def export_story(self, format_type: str = "markdown"):
        """Export the current story to a downloadable file."""
        logger.info(f"Export requested: format={format_type}")

        if not self.orchestrator or not self.orchestrator.story_state:
            logger.warning("Export failed: No story to export")
            return None, "No story to export. Please write a story first."

        if not self.orchestrator.story_state.brief:
            logger.warning("Export failed: Story has no brief/content")
            return None, "No story content to export. Please write a story first."

        try:
            output_dir = Path(tempfile.gettempdir())
            story_id = self.orchestrator.story_state.id[:8]

            if format_type == "text":
                content = self.orchestrator.export_to_text()
                filepath = output_dir / f"story_{story_id}.txt"
            else:
                content = self.orchestrator.export_to_markdown()
                filepath = output_dir / f"story_{story_id}.md"

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info(f"Export successful: {filepath}")
            return str(filepath), f"Story exported as {format_type}!"
        except Exception as e:
            logger.exception(f"Export failed")
            return None, f"Export failed: {e}"

    def save_story(self):
        """Save the current story to disk."""
        logger.info("Save story requested")

        if not self.orchestrator or not self.orchestrator.story_state:
            logger.warning("Save failed: No story to save")
            return "No story to save. Please create a story first."

        try:
            filepath = self.orchestrator.save_story()
            logger.info(f"Story saved: {filepath}")
            return f"Story saved to: {filepath}"
        except Exception as e:
            logger.exception("Save failed")
            return f"Save failed: {e}"

    def get_saved_stories(self):
        """Get list of saved stories for dropdown."""
        stories = StoryOrchestrator.list_saved_stories()
        choices = []
        for s in stories:
            label = f"{s.get('premise', 'Untitled')[:40]}... ({s.get('status', '?')})"
            choices.append((label, s.get('path', '')))
        return choices

    def load_saved_story(self, filepath: str):
        """Load a saved story."""
        logger.info(f"Load story requested: {filepath}")

        if not filepath:
            logger.warning("Load failed: No filepath provided")
            return "Please select a story to load.", "", []

        try:
            if not self.orchestrator:
                self.settings = Settings.load()
                self.orchestrator = StoryOrchestrator(settings=self.settings)

            self.orchestrator.load_story(filepath)
            self.interview_complete = True  # Skip interview for loaded stories

            story_content = self.orchestrator.get_full_story()
            outline = self.orchestrator.get_outline_summary()

            logger.info(f"Story loaded successfully: {self.orchestrator.story_state.id}")
            return f"Story loaded! Status: {self.orchestrator.story_state.status}", outline, story_content
        except Exception as e:
            logger.exception("Load story failed")
            return f"Load failed: {e}", "", ""

    def chat_response(self, message: str, history: list):
        """Handle chat messages during interview."""
        try:
            if not self.orchestrator:
                return history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": "Please start a new story first."}
                ], ""

            if not self.interview_complete:
                response, is_complete = self.orchestrator.process_interview_response(message)
                self.interview_complete = is_complete

                if is_complete:
                    response += "\n\n---\n**Interview complete!** Click 'Build Story Structure' to continue."

                return history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": response}
                ], ""

            return history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "Use the action buttons to continue."}
            ], ""
        except Exception as e:
            logger.exception("chat_response failed")
            return history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": f"Error: {e}"}
            ], ""

    def build_structure(self):
        """Build the story structure."""
        logger.info("Build structure requested")

        if not self.orchestrator:
            logger.warning("Build structure failed: No orchestrator")
            yield "Please start a new story first.", "Error: No story started"
            return

        if not self.interview_complete:
            logger.warning("Build structure failed: Interview not complete")
            yield "Please complete the interview first.", "Error: Interview incomplete"
            return

        try:
            yield "Building story structure...\n\nCreating world and characters...", "Architect is working..."

            self.orchestrator.build_story_structure()

            outline = self.orchestrator.get_outline_summary()
            logger.info("Structure built successfully")
            yield outline, "Structure complete! Review and click 'Write Story' to begin."
        except Exception as e:
            logger.exception("Build structure failed")
            yield f"Error building structure: {e}", f"Error: {e}"

    def write_story(self):
        """Write the story."""
        logger.info("Write story requested")

        if not self.orchestrator:
            logger.warning("Write story failed: No orchestrator")
            yield "Please start a new story first.", "", "Error: No story started"
            return

        state = self.orchestrator.story_state
        if not state or not state.brief:
            logger.warning("Write story failed: No brief/structure")
            yield "Please complete the interview and build structure first.", "", "Error: Story structure not built"
            return

        brief = state.brief

        try:
            if brief.target_length == "short_story":
                logger.info("Writing short story")
                yield "Writing short story...", "", "Writer is working..."

                for event in self.orchestrator.write_short_story():
                    yield "", "", f"{event.agent_name}: {event.message}"

                story = self.orchestrator.get_full_story()
                stats = self.orchestrator.get_statistics()
                stats_msg = f"Complete! {stats['total_words']} words | ~{stats['reading_time_minutes']} min read"
                logger.info(f"Short story complete: {stats['total_words']} words")
                yield story, stats_msg, "Story complete!"

            else:
                logger.info(f"Writing multi-chapter story: {len(state.chapters)} chapters")
                total_chapters = len(state.chapters)
                full_story = ""

                for chapter in state.chapters:
                    yield full_story, f"Writing chapter {chapter.number}/{total_chapters}...", f"Working on: {chapter.title}"

                    for event in self.orchestrator.write_chapter(chapter.number):
                        yield full_story, f"Chapter {chapter.number}: {event.message}", f"{event.agent_name}: {event.message}"

                    full_story = self.orchestrator.get_full_story()
                    logger.info(f"Chapter {chapter.number} complete")
                    yield full_story, f"Chapter {chapter.number} complete", f"Finished: {chapter.title}"

                stats = self.orchestrator.get_statistics()
                stats_msg = (
                    f"Complete! {stats['total_words']} words across {stats['total_chapters']} chapters | "
                    f"~{stats['reading_time_minutes']} min read | "
                    f"Plot: {stats['plot_points_completed']}/{stats['plot_points_total']} points"
                )
                logger.info(f"Multi-chapter story complete: {stats['total_words']} words, {stats['total_chapters']} chapters")
                yield full_story, stats_msg, "All chapters complete!"

        except Exception as e:
            logger.exception("Write story failed")
            yield f"Error writing story: {e}", "", f"Error: {e}"

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
                word_count = len(response.split())
                words_per_sec = word_count / elapsed if elapsed > 0 else 0

                results[model] = {
                    "output": response,
                    "time": elapsed,
                    "word_count": word_count,
                    "words_per_sec": words_per_sec,
                    "model_info": model_info,
                }

                output_parts.append(f"""
## {model_info.get('name', model)}
**Model:** `{model}`
**Time:** {elapsed:.1f}s | **Words:** {word_count} | **Speed:** {words_per_sec:.1f} words/sec
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
                results[model] = {"error": str(e)}

        # Add summary table at the end
        summary_lines = ["\n## Summary\n", "| Model | Time | Words | Speed | Quality |", "|-------|------|-------|-------|---------|"]
        for model, data in results.items():
            if "error" in data:
                summary_lines.append(f"| {model} | ERROR | - | - | - |")
            else:
                summary_lines.append(
                    f"| {data['model_info'].get('name', model)[:20]} | "
                    f"{data['time']:.1f}s | {data['word_count']} | "
                    f"{data['words_per_sec']:.1f} w/s | "
                    f"{data['model_info'].get('quality', '?')}/10 |"
                )

        final_output = "# Model Comparison Results\n\n" + "\n".join(output_parts) + "\n".join(summary_lines)
        yield final_output, "Comparison complete!"

    # ============ Build UI ============

    def build_ui(self):
        """Build and return the Gradio interface."""
        installed_models = get_installed_models()
        model_choices = list(AVAILABLE_MODELS.keys())

        with gr.Blocks(title="Story Factory") as app:
            # Build status line
            ollama_ok, ollama_msg = self.ollama_status
            ollama_status = "Connected" if ollama_ok else f"ERROR: {ollama_msg}"

            warning_text = ""
            if self.model_warnings:
                warning_text = "\n**Warnings:** " + " | ".join(self.model_warnings)
            if not ollama_ok:
                warning_text += f"\n**Ollama Error:** {ollama_msg}"

            gr.Markdown(
                f"""
                # Story Factory
                ### AI-Powered Story Production Team
                **Ollama:** {ollama_status} | **VRAM:** {self.detected_vram}GB | **Models:** {len(installed_models)}{warning_text}
                """
            )

            with gr.Tabs() as tabs:
                # ============ WRITE TAB ============
                with gr.Tab("Write Story", id="write"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Controls")
                            start_btn = gr.Button("Start New Story", variant="primary")
                            build_btn = gr.Button("Build Story Structure")
                            write_btn = gr.Button("Write Story", variant="primary")

                            gr.Markdown("---")
                            gr.Markdown("### Save/Export")
                            save_story_btn = gr.Button("Save Story")
                            export_format = gr.Radio(
                                choices=["markdown", "text"],
                                value="markdown",
                                label="Export Format",
                                scale=0,
                            )
                            export_btn = gr.Button("Export Story")
                            export_file = gr.File(label="Download", visible=False)

                            gr.Markdown("---")
                            gr.Markdown("### Load Saved Story")
                            saved_stories_dropdown = gr.Dropdown(
                                choices=self.get_saved_stories(),
                                label="Select Story",
                                interactive=True,
                            )
                            refresh_saved_btn = gr.Button("Refresh List", size="sm")
                            load_btn = gr.Button("Load Story")

                            status_box = gr.Textbox(
                                label="Status",
                                value="Click 'Start New Story' to begin",
                                interactive=False,
                                lines=6,
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
                with gr.Tab("Compare Models", id="compare"):
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
                with gr.Tab("Settings", id="settings"):
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
                                choices=["minimal", "checkpoint"],
                                value=self.settings.interaction_mode if self.settings.interaction_mode in ["minimal", "checkpoint"] else "checkpoint",
                                label="Interaction Mode",
                                info="minimal: auto-generate | checkpoint: pause for review",
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
                with gr.Tab("Manage Models", id="models") as models_tab:
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

            # Save/Load/Export handlers
            save_story_btn.click(
                self.save_story,
                outputs=[status_box],
            )

            export_btn.click(
                self.export_story,
                inputs=[export_format],
                outputs=[export_file, status_box],
            ).then(
                lambda f: gr.update(visible=f is not None, value=f),
                inputs=[export_file],
                outputs=[export_file],
            )

            refresh_saved_btn.click(
                lambda: gr.update(choices=self.get_saved_stories()),
                outputs=[saved_stories_dropdown],
            )

            load_btn.click(
                self.load_saved_story,
                inputs=[saved_stories_dropdown],
                outputs=[status_box, outline_display, story_display],
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

            # Models tab - refresh all displays
            refresh_btn.click(
                self.refresh_models_tab,
                outputs=[installed_display, available_display, model_to_pull],
            )

            # Auto-refresh when switching to Models tab
            models_tab.select(
                self.refresh_models_tab,
                outputs=[installed_display, available_display, model_to_pull],
            )

            pull_btn.click(
                self.pull_model,
                inputs=[model_to_pull],
                outputs=[pull_status],
            ).then(
                # Refresh displays after pull completes
                self.refresh_models_tab,
                outputs=[installed_display, available_display, model_to_pull],
            )

            # JavaScript for tab persistence via URL hash
            app.load(
                None,
                None,
                None,
                js="""
                () => {
                    // Tab persistence via URL hash
                    const tabMap = {'write': 0, 'compare': 1, 'settings': 2, 'models': 3};
                    const reverseMap = ['write', 'compare', 'settings', 'models'];

                    // Restore tab from URL hash on load
                    const hash = window.location.hash.slice(1);
                    if (hash && tabMap[hash] !== undefined) {
                        setTimeout(() => {
                            const tabs = document.querySelectorAll('button[role="tab"]');
                            if (tabs[tabMap[hash]]) {
                                tabs[tabMap[hash]].click();
                            }
                        }, 100);
                    }

                    // Save tab to URL hash on click
                    setTimeout(() => {
                        const tabs = document.querySelectorAll('button[role="tab"]');
                        tabs.forEach((tab, index) => {
                            tab.addEventListener('click', () => {
                                if (reverseMap[index]) {
                                    history.replaceState(null, '', '#' + reverseMap[index]);
                                }
                            });
                        });
                    }, 200);
                }
                """
            )

        return app


def main():
    """Run the Gradio app."""
    ui = StoryFactoryUI()
    app = ui.build_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()

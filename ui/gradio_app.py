"""Gradio-based UI for Story Factory with web-based configuration."""

import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import gradio as gr

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base import BaseAgent
from settings import (
    AVAILABLE_MODELS,
    Settings,
    get_available_vram,
    get_installed_models,
    get_model_info,
)
from workflows.orchestrator import StoryOrchestrator

logger = logging.getLogger(__name__)


class StoryFactoryUI:
    """Gradio UI for the Story Factory."""

    # Expected directory for story files (for path validation)
    _STORIES_DIR: Path = Path(__file__).parent.parent / "output" / "stories"

    def __init__(self):
        logger.info("StoryFactoryUI initializing...")
        self.settings = Settings.load()
        logger.info(f"Settings loaded: default_model={self.settings.default_model}")
        self.orchestrator: StoryOrchestrator | None = None
        self.interview_complete = False
        self.detected_vram = get_available_vram()
        self.ollama_status = self._check_ollama()
        self.model_warnings = self._validate_models()

    def _validate_story_filepath(self, filepath: str) -> Path | None:
        """Validate that a filepath is within the expected stories directory.

        Returns the resolved Path if valid, None if invalid.
        """
        if not filepath:
            return None

        try:
            path = Path(filepath).resolve()
            stories_dir = self._STORIES_DIR.resolve()

            # Ensure the file is within the stories directory using secure path comparison
            if not path.is_relative_to(stories_dir):
                logger.warning(f"Path traversal attempt blocked: {filepath}")
                return None

            return path
        except (ValueError, OSError) as e:
            logger.warning(f"Invalid filepath: {filepath} - {e}")
            return None

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
            if info.get("name") == normalized:
                # Fallback didn't find it, try original
                info = get_model_info(model)
            display_name = info.get("name", model)
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
                encoding="utf-8",
                errors="replace",  # Handle non-UTF8 chars from progress bars
            )

            output_lines: list[str] = []
            if process.stdout:
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
                error_output = (
                    "\n".join(output_lines[-10:]) if output_lines else "No output captured"
                )
                logger.error(
                    f"Failed to pull {model_id} (exit {process.returncode}): {error_output}"
                )
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
        except Exception as e:
            logger.warning(f"Failed to refresh models list: {e}")

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
        logger.info(
            f"Saving settings: default_model={default_model}, use_per_agent={use_per_agent}, "
            f"agent_models=[{interviewer_model}, {architect_model}, {writer_model}, {editor_model}, {continuity_model}]"
        )
        try:
            self.settings.default_model = default_model
            self.settings.context_size = int(context_size)
            self.settings.interaction_mode = interaction_mode
            self.settings.chapters_between_checkpoints = int(chapters_between)
            self.settings.use_per_agent_models = use_per_agent

            # Always save agent_models regardless of use_per_agent
            self.settings.agent_models = {
                "interviewer": interviewer_model,
                "architect": architect_model,
                "writer": writer_model,
                "editor": editor_model,
                "continuity": continuity_model,
            }

            self.settings.save()
            logger.info(f"Settings saved to {self.settings.__class__.__name__}")
            return "Settings saved successfully!"
        except Exception as e:
            logger.exception("Failed to save settings")
            return f"Error saving settings: {e}"

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
                gr.update(interactive=True),
            )
        except Exception as e:
            logger.exception("start_new_story failed")
            return (
                [{"role": "assistant", "content": f"Error starting story: {e}"}],
                "",
                f"Error: {e}",
                gr.update(interactive=False),
                gr.update(interactive=False),
            )

    def export_story(self, format_type: str = "markdown"):
        """Export the current story to a downloadable file."""
        logger.info(f"Export requested: format={format_type}")

        if not self.orchestrator or not self.orchestrator.story_state:
            logger.warning("Export failed: No story to export")
            return None, "No story to export. Please write a story first."

        # Allow export even without brief (for partial exports)
        if not self.orchestrator.story_state.chapters:
            logger.warning("Export failed: Story has no chapters")
            return None, "No story content to export. Please write a story first."

        try:
            output_dir = Path(tempfile.gettempdir())
            story_id = self.orchestrator.story_state.id[:8]

            filepath: Path
            if format_type == "text":
                text_content = self.orchestrator.export_to_text()
                filepath = output_dir / f"story_{story_id}.txt"
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(text_content)
            elif format_type == "epub":
                binary_content = self.orchestrator.export_to_epub()
                filepath = output_dir / f"story_{story_id}.epub"
                with open(filepath, "wb") as f:
                    f.write(binary_content)
            elif format_type == "pdf":
                binary_content = self.orchestrator.export_to_pdf()
                filepath = output_dir / f"story_{story_id}.pdf"
                with open(filepath, "wb") as f:
                    f.write(binary_content)
            else:  # markdown (default)
                text_content = self.orchestrator.export_to_markdown()
                filepath = output_dir / f"story_{story_id}.md"
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(text_content)

            logger.info(f"Export successful: {filepath}")
            return str(filepath), f"Story exported as {format_type}!"
        except Exception as e:
            logger.exception("Export failed")
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
        choices: list[tuple[str, str]] = []
        for s in stories:
            premise = (s.get("premise") or "Untitled")[:40]
            status = s.get("status") or "?"
            path = s.get("path") or ""
            label = f"{premise}... ({status})"
            choices.append((label, path))
        return choices

    # ============ Project Management Functions ============

    def get_projects_choices(self):
        """Get project choices for dropdown (label, path)."""
        stories = StoryOrchestrator.list_saved_stories()
        choices = []
        for s in stories:
            path = s.get("path") or ""
            try:
                with open(path, encoding="utf-8") as f:
                    import json

                    story_data = json.load(f)
                    # Get premise from brief (nested structure)
                    brief = story_data.get("brief") or {}
                    premise = brief.get("premise", "")[:40] if brief else ""
                    project_name = story_data.get("project_name", "") or premise or "Untitled"
                    last_saved = story_data.get("last_saved", story_data.get("created_at", ""))
                    words = sum(ch.get("word_count", 0) for ch in story_data.get("chapters", []))
                    status = s.get("status") or "?"
            except Exception:
                project_name = (s.get("premise") or "Untitled")[:40]
                last_saved = s.get("created_at") or ""
                words = 0
                status = s.get("status") or "?"

            # Create display label with key info
            date_str = last_saved[:10] if last_saved else "?"
            label = f"{project_name} | {status} | {words} words | {date_str}"
            choices.append((label, s.get("path", "")))
        return choices

    def get_project_details(self, filepath: str):
        """Get detailed info for a selected project."""
        logger.debug(f"get_project_details called with: {filepath!r}")

        if not filepath:
            logger.debug("No filepath provided for project details")
            return "Select a project to view details."

        # Validate filepath is within expected directory
        valid_path = self._validate_story_filepath(filepath)
        if not valid_path:
            return "Invalid project path."

        try:
            import json

            logger.debug(f"Loading project details from: {valid_path}")
            with open(valid_path, encoding="utf-8") as f:
                data = json.load(f)

            # Get premise from brief (nested structure)
            brief = data.get("brief") or {}
            premise = brief.get("premise", "No premise") if brief else "No premise"
            name = data.get("project_name", "") or premise[:50] or "Untitled"

            status = data.get("status", "unknown")
            created = data.get("created_at", "?")[:16]
            last_saved = data.get("last_saved", "Never")
            if isinstance(last_saved, str) and len(last_saved) > 16:
                last_saved = last_saved[:16]
            chapters = data.get("chapters", [])
            words = sum(ch.get("word_count", 0) for ch in chapters)

            # Get additional info from brief
            genre = brief.get("genre", "Unknown") if brief else "Unknown"
            tone = brief.get("tone", "Unknown") if brief else "Unknown"
            language = brief.get("language", "English") if brief else "English"

            return f"""**{name}**

**Status:** {status}
**Genre:** {genre} | **Tone:** {tone} | **Language:** {language}
**Created:** {created}
**Last Saved:** {last_saved}
**Chapters:** {len(chapters)}
**Words:** {words:,}

**Premise:**
{premise[:300]}{"..." if len(premise) > 300 else ""}
"""
        except Exception as e:
            return f"Error loading project: {e}"

    def delete_project(self, filepath: str, confirm: bool):
        """Delete a project with confirmation."""
        if not confirm:
            return gr.update(choices=self.get_projects_choices()), "Check 'Confirm delete' first."

        if not filepath:
            return gr.update(choices=self.get_projects_choices()), "No project selected."

        # Validate filepath is within expected directory
        valid_path = self._validate_story_filepath(filepath)
        if not valid_path:
            return gr.update(choices=self.get_projects_choices()), "Invalid project path."

        try:
            valid_path.unlink()
            logger.info(f"Deleted project: {valid_path}")
            return gr.update(choices=self.get_projects_choices(), value=None), "Project deleted."
        except Exception as e:
            logger.exception("Delete project failed")
            return gr.update(choices=self.get_projects_choices()), f"Delete failed: {e}"

    def rename_project(self, filepath: str, new_name: str):
        """Rename a project."""
        if not new_name or not new_name.strip():
            return (
                gr.update(choices=self.get_projects_choices()),
                self.get_project_details(filepath),
                "Enter a name first.",
            )

        if not filepath:
            return (
                gr.update(choices=self.get_projects_choices()),
                "",
                "No project selected.",
            )

        # Validate filepath is within expected directory
        valid_path = self._validate_story_filepath(filepath)
        if not valid_path:
            return (
                gr.update(choices=self.get_projects_choices()),
                "",
                "Invalid project path.",
            )

        try:
            import json

            with open(valid_path, encoding="utf-8") as f:
                story_data = json.load(f)
            story_data["project_name"] = new_name.strip()
            with open(valid_path, "w", encoding="utf-8") as f:
                json.dump(story_data, f, indent=2, default=str)
            logger.info(f"Renamed project to: {new_name}")
            return (
                gr.update(choices=self.get_projects_choices()),
                self.get_project_details(filepath),
                f"Renamed to '{new_name}'.",
            )
        except Exception as e:
            logger.exception("Rename project failed")
            return (
                gr.update(choices=self.get_projects_choices()),
                self.get_project_details(filepath),
                f"Rename failed: {e}",
            )

    def generate_title_ideas(self, filepath: str):
        """Generate AI title suggestions for a project."""
        logger.info(f"Generate title ideas requested for: {filepath}")

        if not filepath:
            logger.warning("Generate titles: No filepath provided")
            return gr.update(choices=["Select a project first"]), "No project selected."

        # Validate filepath is within expected directory
        valid_path = self._validate_story_filepath(filepath)
        if not valid_path:
            return gr.update(choices=["Invalid path"]), "Invalid project path."

        try:
            logger.info(f"Loading story for title generation: {valid_path}")
            temp_orchestrator = StoryOrchestrator(settings=self.settings)
            temp_orchestrator.load_story(str(valid_path))
            titles = temp_orchestrator.generate_title_suggestions()

            if titles and len(titles) > 0:
                logger.info(f"Generated {len(titles)} title suggestions: {titles}")
                return (
                    gr.update(choices=titles, value=titles[0]),
                    f"Generated {len(titles)} title ideas!",
                )
            else:
                logger.warning("Title generation returned empty list")
                return (
                    gr.update(choices=["No suggestions available"]),
                    "Could not generate titles - check logs for details.",
                )
        except Exception as e:
            logger.exception(f"Generate titles failed: {e}")
            return gr.update(choices=["Error generating titles"]), f"Error: {e}"

    def apply_suggested_title(self, filepath: str, selected_title: str):
        """Apply a suggested title to the project."""
        if not selected_title or selected_title in [
            "Select a project first",
            "No project selected",
            "Error generating titles",
            "No suggestions available",
        ]:
            return (
                gr.update(choices=self.get_projects_choices()),
                self.get_project_details(filepath),
                "Select a valid title first.",
            )
        return self.rename_project(filepath, selected_title)

    def get_current_project_info(self):
        """Get info about the currently loaded project."""
        if not self.orchestrator or not self.orchestrator.story_state:
            return "No project loaded."

        state = self.orchestrator.story_state
        info_parts = [
            f"**Project:** {state.project_name or 'Untitled'}",
            f"**Status:** {state.status}",
            f"**Created:** {state.created_at.strftime('%Y-%m-%d %H:%M') if state.created_at else 'Unknown'}",
        ]
        if state.last_saved:
            info_parts.append(f"**Last Saved:** {state.last_saved.strftime('%Y-%m-%d %H:%M')}")
        if state.chapters:
            total_words = sum(ch.word_count for ch in state.chapters)
            info_parts.append(f"**Words:** {total_words}")
            info_parts.append(f"**Chapters:** {len(state.chapters)}")

        return "\n".join(info_parts)

    def load_saved_story(self, filepath: str):
        """Load a saved story."""
        logger.info(f"Load story requested: {filepath}")

        if not filepath:
            logger.warning("Load failed: No filepath provided")
            return "Please select a story to load.", "", []

        # Validate filepath is within expected directory
        valid_path = self._validate_story_filepath(filepath)
        if not valid_path:
            return "Invalid story path.", "", ""

        try:
            if not self.orchestrator:
                self.settings = Settings.load()
                self.orchestrator = StoryOrchestrator(settings=self.settings)

            state = self.orchestrator.load_story(str(valid_path))
            self.interview_complete = True  # Skip interview for loaded stories

            story_content = self.orchestrator.get_full_story()
            outline = self.orchestrator.get_outline_summary()

            logger.info(f"Story loaded successfully: {state.id}")
            return (
                f"Story loaded! Status: {state.status}",
                outline,
                story_content,
            )
        except Exception as e:
            logger.exception("Load story failed")
            return f"Load failed: {e}", "", ""

    def chat_response(self, message: str, history: list):
        """Handle chat messages during interview."""
        try:
            if not self.orchestrator:
                return (
                    history
                    + [
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": "Please start a new story first."},
                    ],
                    "",
                    gr.update(interactive=False),
                )

            if not self.interview_complete:
                response, is_complete = self.orchestrator.process_interview_response(message)
                self.interview_complete = is_complete

                # Autosave after each interview response
                self.orchestrator.autosave()

                if is_complete:
                    response += "\n\n---\n**Interview complete!** Click 'Build Story Structure' to continue."
                    # Enable Build button when interview is complete
                    return (
                        history
                        + [
                            {"role": "user", "content": message},
                            {"role": "assistant", "content": response},
                        ],
                        "",
                        gr.update(interactive=True, variant="primary"),
                    )

                return (
                    history
                    + [
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": response},
                    ],
                    "",
                    gr.update(),
                )

            return (
                history
                + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": "Use the action buttons to continue."},
                ],
                "",
                gr.update(),
            )
        except Exception as e:
            logger.exception("chat_response failed")
            return (
                history
                + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": f"Error: {e}"},
                ],
                "",
                gr.update(),
            )

    def build_structure(self):
        """Build the story structure."""
        logger.info("Build structure requested")

        if not self.orchestrator:
            logger.warning("Build structure failed: No orchestrator")
            yield "Please start a new story first.", "Error: No story started", gr.update()
            return

        if not self.interview_complete:
            logger.warning("Build structure failed: Interview not complete")
            yield "Please complete the interview first.", "Error: Interview incomplete", gr.update()
            return

        try:
            yield (
                "Building story structure...\n\nCreating world and characters...",
                "Architect is working...",
                gr.update(),
            )

            self.orchestrator.build_story_structure()

            # Autosave after structure is built
            self.orchestrator.autosave()

            outline = self.orchestrator.get_outline_summary()
            logger.info("Structure built successfully")
            # Enable Write button when structure is complete
            yield (
                outline,
                "Structure complete! Review and click 'Write Story' to begin.",
                gr.update(interactive=True, variant="primary"),
            )
        except Exception as e:
            logger.exception("Build structure failed")
            yield f"Error building structure: {e}", f"Error: {e}", gr.update()

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
            yield (
                "Please complete the interview and build structure first.",
                "",
                "Error: Story structure not built",
            )
            return

        brief = state.brief

        # Validate chapters exist for multi-chapter stories
        if brief.target_length != "short_story" and len(state.chapters) == 0:
            logger.warning("Write story failed: No chapters defined")
            yield "Please build story structure first.", "", "Error: No chapters defined"
            return

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
                    yield (
                        full_story,
                        f"Writing chapter {chapter.number}/{total_chapters}...",
                        f"Working on: {chapter.title}",
                    )

                    for event in self.orchestrator.write_chapter(chapter.number):
                        yield (
                            full_story,
                            f"Chapter {chapter.number}: {event.message}",
                            f"{event.agent_name}: {event.message}",
                        )

                    full_story = self.orchestrator.get_full_story()
                    logger.info(f"Chapter {chapter.number} complete")
                    yield (
                        full_story,
                        f"Chapter {chapter.number} complete",
                        f"Finished: {chapter.title}",
                    )

                stats = self.orchestrator.get_statistics()
                stats_msg = (
                    f"Complete! {stats['total_words']} words across {stats['total_chapters']} chapters | "
                    f"~{stats['reading_time_minutes']} min read | "
                    f"Plot: {stats['plot_points_completed']}/{stats['plot_points_total']} points"
                )
                logger.info(
                    f"Multi-chapter story complete: {stats['total_words']} words, {stats['total_chapters']} chapters"
                )
                yield full_story, stats_msg, "All chapters complete!"

        except Exception as e:
            logger.exception("Write story failed")
            yield f"Error writing story: {e}", "", f"Error: {e}"

    # ============ Comparison Functions ============

    def run_comparison(self, prompt: str, selected_models: list):
        """Run the same prompt through multiple models for comparison."""
        logger.info(
            f"Model comparison requested: {len(selected_models) if selected_models else 0} models"
        )

        if not prompt:
            logger.warning("Comparison failed: No prompt provided")
            yield "Please enter a story prompt.", ""
            return

        if not selected_models or len(selected_models) < 2:
            logger.warning("Comparison failed: Not enough models selected")
            yield "Please select at least 2 models to compare.", ""
            return

        from typing import Any

        results: dict[str, dict[str, Any]] = {}
        output_parts: list[str] = []

        for i, model in enumerate(selected_models):
            model_info = get_model_info(model)
            yield (
                f"Running model {i+1}/{len(selected_models)}: {model_info.get('name', model)}...",
                "",
            )

            try:
                # Create a fresh orchestrator with this model
                test_orchestrator = StoryOrchestrator(settings=self.settings, model_override=model)

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

                output_parts.append(
                    f"""
## {model_info.get('name', model)}
**Model:** `{model}`
**Time:** {elapsed:.1f}s | **Words:** {word_count} | **Speed:** {words_per_sec:.1f} words/sec
**Quality Rating:** {model_info.get('quality', '?')}/10

### Output:
{response}

---
"""
                )

            except Exception as e:
                output_parts.append(
                    f"""
## {model_info.get('name', model)}
**Error:** {str(e)}

---
"""
                )
                results[model] = {"error": str(e)}

        # Add summary table at the end
        summary_lines = [
            "\n## Summary\n",
            "| Model | Time | Words | Speed | Quality |",
            "|-------|------|-------|-------|---------|",
        ]
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

        final_output = (
            "# Model Comparison Results\n\n" + "\n".join(output_parts) + "\n".join(summary_lines)
        )
        logger.info(f"Model comparison complete: {len(results)} models tested")
        yield final_output, "Comparison complete!"

    # ============ Build UI ============

    def build_ui(self):
        """Build and return the Gradio interface."""
        installed_models = get_installed_models()
        # Combine AVAILABLE_MODELS with installed models for complete dropdown
        model_choices = list(AVAILABLE_MODELS.keys())
        for model in installed_models:
            normalized = self._normalize_model_name(model)
            if normalized not in model_choices and model not in model_choices:
                model_choices.append(model)

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

            with gr.Tabs():
                # ============ WRITE TAB ============
                with gr.Tab("Write Story", id="write"):
                    with gr.Row():
                        with gr.Column(scale=1, min_width=280):
                            gr.Markdown("### 1. Start")
                            start_btn = gr.Button("Start New Story", variant="primary", size="lg")

                            gr.Markdown("### 2. Build (after interview)")
                            build_btn = gr.Button(
                                "Build Story Structure", size="lg", interactive=False
                            )

                            gr.Markdown("### 3. Write (after build)")
                            write_btn = gr.Button("Write Story", size="lg", interactive=False)

                            gr.Markdown("---")
                            gr.Markdown("### Export Story")
                            export_format = gr.Radio(
                                choices=["markdown", "text", "epub", "pdf"],
                                value="markdown",
                                label="Format",
                            )
                            export_btn = gr.Button("Export & Download", size="sm")
                            export_file = gr.File(label="Download", visible=False)
                            gr.Markdown("*Stories auto-save after each step*", elem_classes="hint")

                            gr.Markdown("---")
                            gr.Markdown("### Load Previous Story")
                            saved_stories_dropdown = gr.Dropdown(
                                choices=self.get_saved_stories(),
                                label="Saved Stories",
                                interactive=True,
                            )
                            with gr.Row():
                                refresh_saved_btn = gr.Button("Refresh", size="sm")
                                load_btn = gr.Button("Load", size="sm")

                            status_box = gr.Textbox(
                                label="Status",
                                value="Click 'Start New Story' to begin",
                                interactive=False,
                                lines=4,
                            )

                        with gr.Column(scale=3):
                            gr.Markdown("### Interview with the Story Architect")
                            chatbot = gr.Chatbot(
                                label="Conversation",
                                height=450,
                                autoscroll=True,
                                placeholder="Click 'Start New Story' to begin the interview...",
                                layout="bubble",
                            )
                            with gr.Row():
                                chat_input = gr.Textbox(
                                    label="Your response",
                                    placeholder="Type your answer and press Enter, or click Send...",
                                    interactive=False,
                                    lines=2,
                                    max_lines=4,
                                    autofocus=True,
                                    scale=4,
                                )
                                send_btn = gr.Button(
                                    "Send", variant="primary", scale=1, interactive=False
                                )

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Story Outline")
                            outline_display = gr.Textbox(
                                label="Outline",
                                lines=12,
                                max_lines=20,
                                interactive=False,
                            )

                        with gr.Column(scale=2):
                            gr.Markdown("### Story Output")
                            story_display = gr.Textbox(
                                label="Your Story",
                                lines=18,
                                max_lines=40,
                                interactive=False,
                                autoscroll=True,
                            )
                            progress_display = gr.Textbox(
                                label="Progress",
                                lines=1,
                                interactive=False,
                                container=False,
                            )

                # ============ PROJECTS TAB ============
                with gr.Tab("Projects", id="projects"):
                    gr.Markdown("### Manage Your Story Projects")

                    with gr.Row():
                        # Left column: Project selection
                        with gr.Column(scale=2):
                            project_dropdown = gr.Dropdown(
                                choices=self.get_projects_choices(),
                                label="Select Project",
                                interactive=True,
                            )
                            with gr.Row():
                                refresh_projects_btn = gr.Button("Refresh", size="sm")
                                load_project_btn = gr.Button(
                                    "Load Selected", variant="primary", size="sm"
                                )

                            gr.Markdown("---")

                            # Rename section
                            gr.Markdown("**Rename Project**")
                            with gr.Row():
                                custom_title_input = gr.Textbox(
                                    label="New Title",
                                    placeholder="Enter a new title...",
                                    scale=3,
                                    container=False,
                                )
                                apply_title_btn = gr.Button("Rename", size="sm", scale=1)

                            # AI titles section
                            gr.Markdown("**AI Title Suggestions**")
                            with gr.Row():
                                generate_titles_btn = gr.Button("Generate Ideas", size="sm")
                            suggested_titles_radio = gr.Radio(
                                choices=[],
                                label="",
                                container=False,
                            )
                            apply_suggestion_btn = gr.Button("Apply Selected Title", size="sm")

                            gr.Markdown("---")

                            # Delete section
                            gr.Markdown("**Delete Project**")
                            with gr.Row():
                                delete_confirm_check = gr.Checkbox(
                                    label="Confirm delete", value=False, scale=2
                                )
                                delete_project_btn = gr.Button(
                                    "Delete", variant="stop", size="sm", scale=1
                                )

                            projects_status = gr.Textbox(label="Status", interactive=False, lines=1)

                        # Right column: Project details
                        with gr.Column(scale=1):
                            gr.Markdown("**Project Details**")
                            project_details = gr.Markdown(value="Select a project to view details.")

                # ============ COMPARE TAB ============
                with gr.Tab("Compare Models", id="compare"):
                    gr.Markdown(
                        """
                    ### Model Comparison
                    Test the same story prompt across different models to compare output quality.
                    """
                    )

                    compare_prompt = gr.Textbox(
                        label="Story Prompt",
                        placeholder="Enter a story concept to test (e.g., 'A detective discovers their partner is the killer')",
                        lines=3,
                    )

                    compare_models = gr.CheckboxGroup(
                        choices=(
                            [(get_model_info(m).get("name", m), m) for m in installed_models]
                            if installed_models
                            else []
                        ),
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
                                value=(
                                    self.settings.interaction_mode
                                    if self.settings.interaction_mode in ["minimal", "checkpoint"]
                                    else "checkpoint"
                                ),
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
                outputs=[chatbot, story_display, status_box, chat_input, send_btn],
            )

            # Chat submission - both Enter key and Send button
            chat_input.submit(
                self.chat_response,
                inputs=[chat_input, chatbot],
                outputs=[chatbot, chat_input, build_btn],
            ).then(lambda: "", outputs=[chat_input])

            send_btn.click(
                self.chat_response,
                inputs=[chat_input, chatbot],
                outputs=[chatbot, chat_input, build_btn],
            ).then(lambda: "", outputs=[chat_input])

            build_btn.click(
                self.build_structure,
                outputs=[outline_display, status_box, write_btn],
            )

            write_btn.click(
                self.write_story,
                outputs=[story_display, progress_display, status_box],
            )

            # Export/Load handlers (autosave removes need for manual save)
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

            # Projects tab handlers
            refresh_projects_btn.click(
                lambda: gr.update(choices=self.get_projects_choices()),
                outputs=[project_dropdown],
            )

            # Show details when project selected
            project_dropdown.change(
                self.get_project_details,
                inputs=[project_dropdown],
                outputs=[project_details],
            )

            load_project_btn.click(
                self.load_saved_story,
                inputs=[project_dropdown],
                outputs=[projects_status, outline_display, story_display],
            )

            apply_title_btn.click(
                self.rename_project,
                inputs=[project_dropdown, custom_title_input],
                outputs=[project_dropdown, project_details, projects_status],
            )

            generate_titles_btn.click(
                self.generate_title_ideas,
                inputs=[project_dropdown],
                outputs=[suggested_titles_radio, projects_status],
            )

            apply_suggestion_btn.click(
                self.apply_suggested_title,
                inputs=[project_dropdown, suggested_titles_radio],
                outputs=[project_dropdown, project_details, projects_status],
            )

            delete_project_btn.click(
                self.delete_project,
                inputs=[project_dropdown, delete_confirm_check],
                outputs=[project_dropdown, projects_status],
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
                    const tabMap = {'write': 0, 'projects': 1, 'compare': 2, 'settings': 3, 'models': 4};
                    const reverseMap = ['write', 'projects', 'compare', 'settings', 'models'];

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

                    // Ctrl+Enter to send chat message
                    setTimeout(() => {
                        document.addEventListener('keydown', (e) => {
                            if (e.ctrlKey && e.key === 'Enter') {
                                // Find the Send button and click it
                                const sendBtn = document.querySelector('button.primary');
                                if (sendBtn && sendBtn.textContent.trim() === 'Send') {
                                    e.preventDefault();
                                    sendBtn.click();
                                }
                            }
                        });
                    }, 300);
                }
                """,
            )

        return app


def main():
    """Run the Gradio app."""
    logger.info("Starting Story Factory web UI...")
    ui = StoryFactoryUI()
    logger.info("Building UI...")
    app = ui.build_ui()
    logger.info("Launching Gradio server on http://127.0.0.1:7860")
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()

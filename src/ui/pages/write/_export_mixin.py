"""Write Story page - Export and suggestions mixin."""

import html
import logging

from nicegui import run, ui
from nicegui.elements.html import Html

from src.ui.pages.write._page import WritePageBase

logger = logging.getLogger(__name__)


class ExportMixin(WritePageBase):
    """Mixin providing export and suggestion methods for WritePage.

    This mixin handles:
    - Exporting stories in various formats (markdown, text, epub, pdf)
    - Showing AI-powered writing suggestions
    """

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

        with dialog, ui.card().classes("w-full h-full flex flex-col bg-white dark:bg-gray-800"):
            # Header
            with ui.row().classes(
                "w-full items-center justify-between p-4 border-b dark:border-gray-700"
            ):
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
            with ui.row().classes("w-full justify-end gap-2 p-4 border-t dark:border-gray-700"):
                ui.button("Refresh", on_click=load_suggestions, icon="refresh").props("flat")

                def close_dialog():
                    """Close the suggestions dialog."""
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

"""Templates page - manage story templates and structure presets."""

import logging

from nicegui import ui
from nicegui.elements.column import Column

from src.memory.templates import StoryTemplate, StructurePreset
from src.services import ServiceContainer
from src.ui.state import AppState
from src.ui.theme import COLORS
from src.utils.exceptions import BackgroundTaskActiveError

logger = logging.getLogger(__name__)


class TemplatesPage:
    """Templates management page.

    Features:
    - Browse built-in templates
    - Create custom templates from projects
    - Import/export templates
    - View structure presets
    """

    def __init__(self, state: AppState, services: ServiceContainer):
        """Initialize templates page.

        Args:
            state: Application state.
            services: Service container.
        """
        self.state = state
        self.services = services
        self._templates_container: Column | None = None
        self._selected_tab = "templates"

    def build(self) -> None:
        """Build the templates page UI."""
        with ui.column().classes("w-full gap-4 p-4"):
            # Header
            with ui.row().classes("w-full items-center"):
                ui.label("Templates & Presets").classes("text-2xl font-bold")
                ui.space()
                with ui.row().classes("gap-2"):
                    ui.button(
                        "Import Template",
                        on_click=self._show_import_dialog,
                        icon="upload_file",
                    ).props("flat")
                    ui.button(
                        "Create from Project",
                        on_click=self._show_create_dialog,
                        icon="add",
                    ).props("color=primary")

            # Tabs for templates vs presets
            with ui.tabs().classes("w-full") as tabs:
                ui.tab("templates", label="Story Templates", icon="book")
                ui.tab("presets", label="Structure Presets", icon="architecture")

            with ui.tab_panels(tabs, value="templates").classes("w-full"):
                with ui.tab_panel("templates"):
                    self._templates_container = ui.column().classes("w-full gap-4")
                    self._refresh_templates()

                with ui.tab_panel("presets"):
                    self._build_presets_view()

    def _refresh_templates(self) -> None:
        """Refresh the templates list."""
        from src.ui.components.common import empty_state

        if not self._templates_container:
            return

        self._templates_container.clear()
        templates = self.services.template.list_templates()

        if not templates:
            with self._templates_container:
                empty_state(
                    icon="book",
                    title="No templates yet",
                    description="Import a template or create one from an existing project.",
                )
            return

        # Group by built-in vs custom
        builtin = [t for t in templates if t.is_builtin]
        custom = [t for t in templates if not t.is_builtin]

        with self._templates_container:
            if builtin:
                ui.label("Built-in Templates").classes("text-lg font-semibold mt-2")
                for template in builtin:
                    self._build_template_card(template)

            if custom:
                ui.label("Custom Templates").classes("text-lg font-semibold mt-4")
                for template in custom:
                    self._build_template_card(template)

    def _build_template_card(self, template: StoryTemplate) -> None:
        """Build a template card.

        Args:
            template: StoryTemplate object.
        """
        with ui.card().classes("w-full"):
            with ui.row().classes("w-full items-start gap-4"):
                # Template icon
                with ui.column().classes("items-center justify-start"):
                    ui.icon("book", size="2rem").style(f"color: {COLORS['primary']}")

                # Template info
                with ui.column().classes("flex-grow gap-2"):
                    with ui.row().classes("items-center gap-2"):
                        ui.label(template.name).classes("text-lg font-semibold")
                        ui.badge(template.genre).props("color=primary")
                        if template.is_builtin:
                            ui.badge("Built-in").props("color=secondary")

                    ui.label(template.description).classes("text-sm text-gray-400")

                    # Details
                    with ui.row().classes("gap-4 text-sm text-gray-300"):
                        if template.subgenres:
                            ui.label(f"Subgenres: {', '.join(template.subgenres)}")
                        ui.label(f"Length: {template.target_length.replace('_', ' ').title()}")
                        ui.label(f"{len(template.characters)} characters")
                        ui.label(f"{len(template.plot_points)} plot points")

                    # Tags
                    if template.tags:
                        with ui.row().classes("gap-2 mt-2"):
                            for tag in template.tags:
                                ui.badge(tag).props("outline")

                # Actions
                with ui.column().classes("gap-2"):
                    ui.button(
                        "View Details",
                        on_click=lambda t=template: self._show_template_details(t),
                        icon="visibility",
                    ).props("flat")

                    ui.button(
                        "Use Template",
                        on_click=lambda t=template: self._use_template(t.id),
                        icon="add_circle",
                    ).props("flat color=primary")

                    if not template.is_builtin:
                        ui.button(
                            "Export",
                            on_click=lambda t=template: self._export_template(t.id),
                            icon="download",
                        ).props("flat")

                        ui.button(
                            "Delete",
                            on_click=lambda t=template: self._delete_template(t.id),
                            icon="delete",
                        ).props("flat color=negative")

    def _build_presets_view(self) -> None:
        """Build the structure presets view."""
        presets = self.services.template.list_structure_presets()

        with ui.column().classes("w-full gap-4"):
            ui.label(
                "Structure presets provide proven story frameworks to guide your narrative."
            ).classes("text-gray-400")

            for preset in presets:
                with ui.card().classes("w-full"):
                    with ui.row().classes("w-full items-start gap-4"):
                        # Preset icon
                        with ui.column().classes("items-center justify-start"):
                            ui.icon("architecture", size="2rem").style(
                                f"color: {COLORS['secondary']}"
                            )

                        # Preset info
                        with ui.column().classes("flex-grow gap-2"):
                            ui.label(preset.name).classes("text-lg font-semibold")
                            ui.label(preset.description).classes("text-sm text-gray-400")

                            # Acts
                            with ui.row().classes("gap-2 mt-2"):
                                ui.label("Structure:").classes("font-medium")
                                for act in preset.acts:
                                    ui.badge(act).props("color=secondary")

                            # Stats
                            with ui.row().classes("gap-4 text-sm text-gray-300 mt-2"):
                                ui.label(f"{len(preset.plot_points)} plot points")
                                ui.label(f"{len(preset.beats)} story beats")

                        # Actions
                        with ui.column().classes("gap-2"):
                            ui.button(
                                "View Details",
                                on_click=lambda p=preset: self._show_preset_details(p),
                                icon="visibility",
                            ).props("flat")

    def _show_template_details(self, template: StoryTemplate) -> None:
        """Show detailed template information."""
        with ui.dialog() as dialog, ui.card().classes("w-[800px] max-w-full"):
            ui.label(template.name).classes("text-xl font-bold")
            ui.separator()

            with ui.scroll_area().classes("h-[600px]"):
                with ui.column().classes("gap-4 p-4"):
                    # Basic info
                    with ui.row().classes("gap-2"):
                        ui.badge(template.genre).props("color=primary")
                        if template.is_builtin:
                            ui.badge("Built-in").props("color=secondary")

                    ui.label(template.description).classes("text-gray-400")

                    # Settings
                    ui.label("Story Settings").classes("text-lg font-semibold mt-4")
                    with ui.column().classes("gap-2"):
                        ui.label(f"Tone: {template.tone}")
                        if template.themes:
                            ui.label(f"Themes: {', '.join(template.themes)}")
                        if template.setting_time:
                            ui.label(f"Time Period: {template.setting_time}")
                        if template.setting_place:
                            ui.label(f"Setting: {template.setting_place}")
                        ui.label(
                            f"Target Length: {template.target_length.replace('_', ' ').title()}"
                        )

                    # World building
                    if template.world_description:
                        ui.label("World Description").classes("text-lg font-semibold mt-4")
                        ui.label(template.world_description).classes("text-gray-400")

                    if template.world_rules:
                        ui.label("World Rules").classes("text-lg font-semibold mt-4")
                        for rule in template.world_rules:
                            ui.label(f"• {rule}").classes("text-gray-400")

                    # Characters
                    if template.characters:
                        ui.label("Character Archetypes").classes("text-lg font-semibold mt-4")
                        for char in template.characters:
                            with ui.card().classes("w-full bg-gray-800"):
                                ui.label(f"{char.name} ({char.role})").classes("font-medium")
                                ui.label(char.description).classes("text-sm text-gray-400")
                                if char.personality_traits:
                                    with ui.row().classes("gap-2 mt-2"):
                                        for trait in char.personality_traits:
                                            color = {
                                                "core": "blue",
                                                "flaw": "red",
                                                "quirk": "purple",
                                            }.get(trait.category, "blue")
                                            ui.badge(trait.trait, color=color).props("outline")

                    # Plot points
                    if template.plot_points:
                        ui.label("Plot Points").classes("text-lg font-semibold mt-4")
                        for i, point in enumerate(template.plot_points, 1):
                            with ui.row().classes("gap-2"):
                                ui.label(f"{i}.").classes("font-medium")
                                ui.label(point.description).classes("text-gray-400")

            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Close", on_click=dialog.close).props("flat")

                async def use_and_close():
                    """Close dialog and apply template."""
                    dialog.close()
                    await self._use_template(template.id)

                ui.button(
                    "Use This Template",
                    on_click=use_and_close,
                    icon="add_circle",
                ).props("color=primary")

        dialog.open()

    def _show_preset_details(self, preset: StructurePreset) -> None:
        """Show detailed preset information."""
        with ui.dialog() as dialog, ui.card().classes("w-[700px] max-w-full"):
            ui.label(preset.name).classes("text-xl font-bold")
            ui.separator()

            with ui.scroll_area().classes("h-[500px]"):
                with ui.column().classes("gap-4 p-4"):
                    ui.label(preset.description).classes("text-gray-400")

                    # Acts
                    ui.label("Structure").classes("text-lg font-semibold mt-4")
                    for i, act in enumerate(preset.acts, 1):
                        ui.label(f"{i}. {act}").classes("font-medium")

                    # Plot points
                    ui.label("Plot Points").classes("text-lg font-semibold mt-4")
                    for point in preset.plot_points:
                        with ui.row().classes("gap-2 items-start"):
                            if point.percentage is not None:
                                ui.label(f"{point.percentage}%").classes(
                                    "text-sm text-gray-500 w-12"
                                )
                            ui.label(point.description).classes("text-gray-400")

                    # Beats
                    if preset.beats:
                        ui.label("Story Beats").classes("text-lg font-semibold mt-4")
                        for i, beat in enumerate(preset.beats, 1):
                            ui.label(f"{i}. {beat}").classes("text-gray-400")

            ui.button("Close", on_click=dialog.close).props("flat").classes("mt-4")

        dialog.open()

    async def _use_template(self, template_id: str) -> None:
        """Create a new project using a template."""
        try:
            # Get template name for default project name
            template = self.services.template.get_template(template_id)
            if not template:
                ui.notify("Template not found", type="negative")
                return

            # Create project with template
            project_name = f"New {template.name}"
            state, world_db = self.services.project.create_project(
                name=project_name, template_id=template_id
            )

            # Set as active project
            self.state.set_project(state.id, state, world_db)
            self.services.settings.last_project_id = state.id
            self.services.settings.save()

            ui.notify(f"Created project from template: {template.name}", type="positive")
            ui.navigate.to("/")  # Navigate to write page (root)
        except BackgroundTaskActiveError:
            ui.notify("Cannot create project while tasks are running", type="warning")
        except Exception as e:
            logger.exception("Failed to create project from template")
            ui.notify(f"Error: {e}", type="negative")

    async def _show_create_dialog(self) -> None:
        """Show dialog to create template from current project."""
        if not self.state.project:
            ui.notify("No active project to create template from", type="warning")
            return

        with ui.dialog() as dialog, ui.card():
            ui.label("Create Template from Project").classes("text-lg font-bold")
            ui.separator()

            with ui.column().classes("gap-4 w-96"):
                name_input = ui.input("Template Name", placeholder="My Custom Template").classes(
                    "w-full"
                )
                desc_input = ui.textarea(
                    "Description", placeholder="Describe this template..."
                ).classes("w-full")

                with ui.row().classes("w-full justify-end gap-2"):
                    ui.button("Cancel", on_click=dialog.close).props("flat")
                    ui.button(
                        "Create",
                        on_click=lambda: self._create_template(
                            name_input.value,
                            desc_input.value,
                            dialog,
                        ),
                    ).props("color=primary")

        dialog.open()

    def _create_template(self, name: str, description: str, dialog) -> None:
        """Create a template from the current project."""
        if not name or not description:
            ui.notify("Please provide name and description", type="warning")
            return

        if not self.state.project:
            ui.notify("No active project", type="warning")
            return

        try:
            template = self.services.template.create_template_from_project(
                self.state.project, name, description
            )
            ui.notify(f"Template created: {template.name}", type="positive")
            dialog.close()
            self._refresh_templates()
        except Exception as e:
            logger.exception("Failed to create template")
            ui.notify(f"Error: {e}", type="negative")

    async def _show_import_dialog(self) -> None:
        """Show dialog to import a template."""
        with ui.dialog() as dialog, ui.card():
            ui.label("Import Template").classes("text-lg font-bold")
            ui.separator()

            with ui.column().classes("gap-4 w-96"):
                ui.label("Select a template JSON file to import:").classes("text-gray-400")

                # Note: In production, this would use a proper file upload component
                ui.label("File upload UI would go here").classes("text-sm text-gray-500")

                with ui.row().classes("w-full justify-end gap-2"):
                    ui.button("Cancel", on_click=dialog.close).props("flat")

        dialog.open()

    def _export_template(self, template_id: str) -> None:
        """Export a template to file."""
        try:
            # In production, this would trigger a file download
            template = self.services.template.get_template(template_id)
            if not template:
                ui.notify("Template not found", type="negative")
                return

            # For now, just show success message
            ui.notify(
                f"Export functionality for '{template.name}' would trigger download here",
                type="info",
            )
        except Exception as e:
            logger.exception("Failed to export template")
            ui.notify(f"Error: {e}", type="negative")

    async def _delete_template(self, template_id: str) -> None:
        """Delete a custom template."""
        from src.ui.components.common import confirmation_dialog

        def do_delete():
            """Execute template deletion."""
            try:
                self.services.template.delete_template(template_id)
                ui.notify("Template deleted", type="positive")
                self._refresh_templates()
            except Exception as e:
                logger.exception("Failed to delete template")
                ui.notify(f"Error: {e}", type="negative")

        confirmation_dialog(
            title="Delete Template?",
            message="Are you sure you want to delete this template? This cannot be undone.",
            on_confirm=do_delete,
            confirm_text="Delete",
            cancel_text="Cancel",
        )

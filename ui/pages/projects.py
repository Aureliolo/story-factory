"""Projects page - project management."""

import logging
from datetime import datetime

from nicegui import ui
from nicegui.elements.column import Column

from services import ServiceContainer
from ui.state import AppState
from ui.theme import get_status_color

logger = logging.getLogger(__name__)


class ProjectsPage:
    """Projects management page.

    Features:
    - Project list with status
    - Create new project
    - Delete projects
    - Duplicate projects
    - Project details
    """

    def __init__(self, state: AppState, services: ServiceContainer):
        """Initialize projects page.

        Args:
            state: Application state.
            services: Service container.
        """
        self.state = state
        self.services = services
        self._project_list: Column | None = None

    def build(self) -> None:
        """Build the projects page UI."""
        with ui.column().classes("w-full gap-4 p-4"):
            # Header
            with ui.row().classes("w-full items-center"):
                ui.label("Projects").classes("text-2xl font-bold")
                ui.space()
                ui.button(
                    "+ New Project",
                    on_click=self._create_project,
                    icon="add",
                ).props("color=primary")

            # Project list
            self._project_list = ui.column().classes("w-full gap-4")
            self._refresh_project_list()

    def _refresh_project_list(self) -> None:
        """Refresh the project list display."""
        from ui.components.common import empty_state

        if not self._project_list:
            return

        self._project_list.clear()
        projects = self.services.project.list_projects()

        if not projects:
            with self._project_list:
                empty_state(
                    icon="folder_open",
                    title="No projects yet",
                    description="Create a new project to get started.",
                    action_text="Create Project",
                    on_action=self._create_project,
                )
            return

        with self._project_list:
            for project in projects:
                self._build_project_card(project)

    def _build_project_card(self, project) -> None:
        """Build a project card.

        Args:
            project: ProjectSummary object.
        """
        is_current = project.id == self.state.project_id
        status_color = get_status_color(project.status)

        card_classes = "w-full"
        if is_current:
            card_classes += " ring-2 ring-blue-500"

        with ui.card().classes(card_classes):
            with ui.row().classes("w-full items-start gap-4"):
                # Project info
                with ui.column().classes("flex-grow gap-2"):
                    with ui.row().classes("items-center gap-2"):
                        ui.label(project.name).classes("text-lg font-semibold")
                        ui.badge(project.status.title()).style(
                            f"background-color: {status_color}; color: white;"
                        )
                        if is_current:
                            ui.badge("Active").props("color=primary").classes("text-white")

                    if project.premise:
                        ui.label(
                            project.premise[:150] + "..."
                            if len(project.premise) > 150
                            else project.premise
                        ).classes("text-sm text-gray-600 dark:text-gray-400")

                    # Stats
                    with ui.row().classes("gap-4 text-sm text-gray-600 dark:text-gray-300"):
                        ui.label(f"{project.chapter_count} chapters")
                        ui.label(f"{project.word_count:,} words")
                        ui.label(f"Updated: {self._format_date(project.updated_at)}")

                # Actions
                with ui.column().classes("gap-2"):
                    if not is_current:
                        ui.button(
                            "Open",
                            on_click=lambda p=project: self._open_project(p.id),
                            icon="folder_open",
                        ).props("flat")

                    ui.button(
                        "Duplicate",
                        on_click=lambda p=project: self._duplicate_project(p.id),
                        icon="content_copy",
                    ).props("flat")

                    ui.button(
                        "Delete",
                        on_click=lambda p=project: self._confirm_delete(p),
                        icon="delete",
                    ).props("flat color=negative")

    def _format_date(self, dt: datetime) -> str:
        """Format datetime for display."""
        now = datetime.now()
        diff = now - dt

        if diff.days == 0:
            if diff.seconds < 3600:
                mins = diff.seconds // 60
                return f"{mins}m ago"
            hours = diff.seconds // 3600
            return f"{hours}h ago"
        elif diff.days == 1:
            return "Yesterday"
        elif diff.days < 7:
            return f"{diff.days} days ago"
        else:
            return dt.strftime("%b %d, %Y")

    async def _create_project(self) -> None:
        """Create a new project with optional template selection."""
        # Show dialog for template selection
        with ui.dialog() as dialog, ui.card().classes("w-[600px] max-w-full"):
            ui.label("Create New Project").classes("text-xl font-bold")
            ui.separator()

            with ui.column().classes("gap-4 p-4"):
                # Project name input
                name_input = ui.input(
                    "Project Name (optional)",
                    placeholder="Leave blank for auto-generated name",
                ).classes("w-full")

                # Template selection
                ui.label("Start from Template (optional)").classes("font-semibold mt-2")
                templates = self.services.template.list_templates()

                template_options = {"": "Blank Project (No Template)"}
                template_options.update({t.id: f"{t.name} ({t.genre})" for t in templates})

                template_select = ui.select(
                    label="Template",
                    options=template_options,
                    value="",
                ).classes("w-full")

                # Show template description when selected
                template_desc = ui.label("").classes(
                    "text-sm text-gray-600 dark:text-gray-400 mt-2"
                )

                def on_template_change():
                    selected_id = template_select.value
                    if selected_id:
                        template = self.services.template.get_template(selected_id)
                        if template:
                            template_desc.text = template.description
                    else:
                        template_desc.text = ""

                template_select.on_value_change(on_template_change)

                # Action buttons
                with ui.row().classes("w-full justify-end gap-2 mt-4"):
                    ui.button("Cancel", on_click=dialog.close).props("flat")
                    ui.button(
                        "Create",
                        on_click=lambda: self._do_create_project(
                            name_input.value,
                            template_select.value if template_select.value else None,
                            dialog,
                        ),
                    ).props("color=primary")

        dialog.open()

    async def _do_create_project(self, name: str, template_id: str | None, dialog) -> None:
        """Actually create the project."""
        try:
            project, world_db = self.services.project.create_project(
                name=name, template_id=template_id
            )
            self.state.set_project(project.id, project, world_db)
            self.services.settings.last_project_id = project.id
            self.services.settings.save()

            template_msg = ""
            if template_id:
                template = self.services.template.get_template(template_id)
                if template:
                    template_msg = f" from template: {template.name}"

            ui.notify(f"Project created{template_msg}!", type="positive")
            dialog.close()
            # Reload to update header dropdown
            ui.navigate.reload()
        except Exception as e:
            logger.exception("Failed to create project")
            ui.notify(f"Error: {e}", type="negative")

    async def _open_project(self, project_id: str) -> None:
        """Open a project."""
        try:
            project, world_db = self.services.project.load_project(project_id)
            self.state.set_project(project_id, project, world_db)
            self.services.settings.last_project_id = project_id
            self.services.settings.save()
            ui.notify(f"Opened: {project.project_name}", type="positive")
            self._refresh_project_list()
        except Exception as e:
            logger.exception(f"Failed to open project {project_id}")
            ui.notify(f"Error: {e}", type="negative")

    async def _duplicate_project(self, project_id: str) -> None:
        """Duplicate a project."""
        try:
            project, world_db = self.services.project.duplicate_project(project_id)
            ui.notify(f"Duplicated as: {project.project_name}", type="positive")
            self._refresh_project_list()
        except Exception as e:
            logger.exception(f"Failed to duplicate project {project_id}")
            ui.notify(f"Error: {e}", type="negative")

    async def _confirm_delete(self, project) -> None:
        """Show delete confirmation dialog."""
        from ui.components.common import confirmation_dialog

        def delete():
            self._delete_project(project.id)

        confirmation_dialog(
            title="Delete Project?",
            message=f'Are you sure you want to delete "{project.name}"? This cannot be undone.',
            on_confirm=delete,
            confirm_text="Delete",
            cancel_text="Cancel",
        )

    def _delete_project(self, project_id: str) -> None:
        """Delete a project."""
        try:
            # Clear if this is the current project
            if self.state.project_id == project_id:
                self.state.clear_project()
                self.services.settings.last_project_id = None
                self.services.settings.save()

            self.services.project.delete_project(project_id)
            self._refresh_project_list()
            ui.notify("Project deleted", type="positive")
        except Exception as e:
            logger.exception(f"Failed to delete project {project_id}")
            ui.notify(f"Error: {e}", type="negative")

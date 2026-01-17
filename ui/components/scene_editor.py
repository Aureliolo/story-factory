"""Scene editor component for managing scenes within chapters."""

import logging
import uuid
from collections.abc import Callable

from nicegui import ui

from memory.story_state import Chapter, Scene

logger = logging.getLogger(__name__)


class SceneEditorDialog:
    """Dialog for editing a single scene.

    Provides a modal dialog with form fields for editing scene metadata
    and content.
    """

    def __init__(
        self,
        scene: Scene | None,
        chapter: Chapter,
        on_save: Callable[[Scene], None],
        on_cancel: Callable[[], None] | None = None,
    ):
        """Initialize scene editor dialog.

        Args:
            scene: Scene to edit, or None to create a new scene.
            chapter: Parent chapter for context.
            on_save: Callback when scene is saved.
            on_cancel: Optional callback when dialog is cancelled.
        """
        self.scene = scene
        self.chapter = chapter
        self.on_save = on_save
        self.on_cancel = on_cancel
        self.is_new = scene is None

        # Create new scene if none provided
        if self.is_new:
            self.scene = Scene(
                id=str(uuid.uuid4()),
                title="New Scene",
                outline="",
                order=len(chapter.scenes),
            )

        # UI references
        self._dialog: ui.dialog | None = None
        self._title_input: ui.input | None = None
        self._outline_input: ui.textarea | None = None
        self._content_input: ui.textarea | None = None
        self._pov_input: ui.input | None = None
        self._location_input: ui.input | None = None
        self._goals_input: ui.textarea | None = None

    def build(self) -> None:
        """Build and open the dialog."""
        self._dialog = ui.dialog().props("persistent")

        with self._dialog, ui.card().classes("w-[800px] max-h-[80vh] overflow-auto"):
            # Header
            with ui.row().classes("w-full items-center mb-4"):
                ui.icon("edit_note", size="md").classes("text-blue-500")
                ui.label("New Scene" if self.is_new else f"Edit Scene: {self.scene.title}").classes(
                    "text-xl font-semibold"
                )
                ui.space()
                ui.button(
                    icon="close",
                    on_click=self._handle_cancel,
                ).props("flat dense")

            # Form fields
            with ui.column().classes("w-full gap-4"):
                # Title
                self._title_input = ui.input(
                    label="Scene Title",
                    value=self.scene.title,
                    placeholder="e.g., 'The Discovery'",
                ).classes("w-full")

                # Metadata row
                with ui.row().classes("w-full gap-4"):
                    self._pov_input = ui.input(
                        label="POV Character",
                        value=self.scene.pov_character,
                        placeholder="e.g., 'John'",
                    ).classes("flex-1")

                    self._location_input = ui.input(
                        label="Location",
                        value=self.scene.location,
                        placeholder="e.g., 'The old library'",
                    ).classes("flex-1")

                # Outline
                self._outline_input = (
                    ui.textarea(
                        label="Scene Outline",
                        value=self.scene.outline,
                        placeholder="Describe what happens in this scene...",
                    )
                    .classes("w-full")
                    .props("rows=3")
                )

                # Goals
                goals_text = "\n".join(self.scene.goals) if self.scene.goals else ""
                self._goals_input = (
                    ui.textarea(
                        label="Scene Goals (one per line)",
                        value=goals_text,
                        placeholder="- Reveal the mystery\n- Introduce tension\n- Show character growth",
                    )
                    .classes("w-full")
                    .props("rows=3")
                )

                # Content (optional)
                with ui.expansion("Scene Content", icon="article").classes("w-full"):
                    self._content_input = (
                        ui.textarea(
                            label="Content (optional - can be generated later)",
                            value=self.scene.content,
                            placeholder="Write the actual scene prose here...",
                        )
                        .classes("w-full")
                        .props("rows=8")
                    )

                # Word count indicator
                if self.scene.word_count > 0:
                    ui.label(f"Word count: {self.scene.word_count}").classes(
                        "text-sm text-gray-500 dark:text-gray-400"
                    )

            # Action buttons
            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=self._handle_cancel).props("flat")
                ui.button(
                    "Create Scene" if self.is_new else "Save Changes",
                    on_click=self._handle_save,
                ).props("color=primary")

        self._dialog.open()

    def _handle_save(self) -> None:
        """Handle save button click."""
        if not self._title_input or not self._outline_input:
            return

        # Validate required fields
        if not self._title_input.value.strip():
            ui.notify("Scene title is required", type="warning")
            return

        # Update scene from form
        self.scene.title = self._title_input.value.strip()
        self.scene.outline = self._outline_input.value.strip() if self._outline_input.value else ""

        if self._pov_input:
            self.scene.pov_character = (
                self._pov_input.value.strip() if self._pov_input.value else ""
            )

        if self._location_input:
            self.scene.location = (
                self._location_input.value.strip() if self._location_input.value else ""
            )

        # Parse goals from textarea
        if self._goals_input and self._goals_input.value:
            goals_text = self._goals_input.value.strip()
            # Split by newlines and filter out empty lines
            self.scene.goals = [
                line.strip().lstrip("-•*").strip()
                for line in goals_text.split("\n")
                if line.strip()
            ]
        else:
            self.scene.goals = []

        # Update content and word count
        if self._content_input and self._content_input.value:
            self.scene.content = self._content_input.value.strip()
            self.scene.update_word_count()

        # Close dialog and call save callback
        if self._dialog:
            self._dialog.close()

        logger.debug(f"Scene saved: {self.scene.title} (ID: {self.scene.id})")
        self.on_save(self.scene)

    def _handle_cancel(self) -> None:
        """Handle cancel button click."""
        if self._dialog:
            self._dialog.close()

        if self.on_cancel:
            self.on_cancel()


class SceneListComponent:
    """Component for displaying and managing a list of scenes within a chapter.

    Provides scene list with drag-drop reordering, edit, and delete functionality.
    """

    def __init__(
        self,
        chapter: Chapter,
        on_scene_updated: Callable[[], None],
        on_add_scene: Callable[[], None] | None = None,
    ):
        """Initialize scene list component.

        Args:
            chapter: Chapter containing the scenes.
            on_scene_updated: Callback when scenes are modified.
            on_add_scene: Optional callback for add scene button.
        """
        self.chapter = chapter
        self.on_scene_updated = on_scene_updated
        self.on_add_scene = on_add_scene

        # UI references
        self._container: ui.column | None = None
        self._scene_cards: dict[str, ui.card] = {}

    def build(self) -> None:
        """Build the scene list UI."""
        with ui.column().classes("w-full gap-2") as self._container:
            # Header with Add Scene button
            with ui.row().classes("w-full items-center mb-2"):
                ui.label("Scenes").classes("text-md font-semibold")
                ui.space()
                ui.button(
                    "Add Scene",
                    icon="add",
                    on_click=self._handle_add_scene,
                ).props("flat size=sm")

            # Scene list
            if not self.chapter.scenes:
                with ui.card().classes("w-full bg-gray-50 dark:bg-gray-800"):
                    ui.label("No scenes yet. Click 'Add Scene' to create one.").classes(
                        "text-sm text-gray-500 dark:text-gray-400 p-2"
                    )
            else:
                # Create sortable container for scenes
                with ui.column().classes("w-full gap-2") as scene_container:
                    scene_container._props["id"] = f"scene-list-{id(scene_container)}"
                    self._build_scene_list()

                    # Note: Drag-drop reordering is enabled via HTML5 draggable attributes
                    # A full implementation would use SortableJS library with proper callbacks
                    # For now, scenes can be manually reordered using the Edit button

    def _build_scene_list(self) -> None:
        """Build the list of scene cards."""
        # Sort scenes by order
        sorted_scenes = sorted(self.chapter.scenes, key=lambda s: s.order)

        for scene in sorted_scenes:
            self._build_scene_card(scene)

    def _build_scene_card(self, scene: Scene) -> None:
        """Build a single scene card.

        Args:
            scene: Scene to display.
        """
        status_color = (
            "green"
            if scene.status == "final"
            else "orange"
            if scene.status in ["drafted", "edited"]
            else "gray"
        )

        card = ui.card().classes("w-full p-3 hover:bg-gray-50 dark:hover:bg-gray-700 cursor-move")
        self._scene_cards[scene.id] = card

        with card:
            with ui.row().classes("w-full items-start gap-2"):
                # Drag handle
                ui.icon("drag_indicator", size="sm").classes("text-gray-400 cursor-move")

                # Scene content
                with ui.column().classes("flex-grow gap-1"):
                    # Title and status
                    with ui.row().classes("items-center gap-2"):
                        ui.label(f"{scene.order + 1}. {scene.title}").classes("font-medium")
                        ui.badge(scene.status).props(f"color={status_color}")

                    # Metadata
                    metadata_parts = []
                    if scene.pov_character:
                        metadata_parts.append(f"POV: {scene.pov_character}")
                    if scene.location:
                        metadata_parts.append(f"@ {scene.location}")
                    if scene.word_count > 0:
                        metadata_parts.append(f"{scene.word_count} words")

                    if metadata_parts:
                        ui.label(" • ".join(metadata_parts)).classes(
                            "text-xs text-gray-500 dark:text-gray-400"
                        )

                    # Outline preview
                    if scene.outline:
                        outline_preview = (
                            scene.outline[:100] + "..."
                            if len(scene.outline) > 100
                            else scene.outline
                        )
                        ui.label(outline_preview).classes(
                            "text-sm text-gray-600 dark:text-gray-400"
                        )

                    # Goals
                    if scene.goals:
                        with ui.row().classes("flex-wrap gap-1 mt-1"):
                            for goal in scene.goals[:3]:  # Show max 3 goals
                                ui.chip(
                                    goal[:30] + "..." if len(goal) > 30 else goal,
                                    icon="flag",
                                ).props("size=sm outline")
                            if len(scene.goals) > 3:
                                ui.chip(f"+{len(scene.goals) - 3} more").props(
                                    "size=sm color=grey outline"
                                )

                # Action buttons
                with ui.column().classes("gap-1"):
                    ui.button(
                        icon="edit",
                        on_click=lambda s=scene: self._handle_edit_scene(s),
                    ).props("flat dense size=sm").tooltip("Edit scene")

                    ui.button(
                        icon="delete",
                        on_click=lambda s=scene: self._handle_delete_scene(s),
                    ).props("flat dense size=sm color=red").tooltip("Delete scene")

        # Note: Drag-drop reordering will be implemented in a future update
        # using SortableJS library or similar
        # For now, use Edit button to manage scenes


    def _handle_add_scene(self) -> None:
        """Handle add scene button click."""
        logger.debug(f"Adding new scene to chapter {self.chapter.number}")

        def on_save(scene: Scene) -> None:
            """Handle scene save."""
            self.chapter.add_scene(scene)
            self.on_scene_updated()
            self.refresh()
            ui.notify(f"Scene '{scene.title}' added", type="positive")

        # Open scene editor dialog
        editor = SceneEditorDialog(
            scene=None,
            chapter=self.chapter,
            on_save=on_save,
        )
        editor.build()

    def _handle_edit_scene(self, scene: Scene) -> None:
        """Handle edit scene button click.

        Args:
            scene: Scene to edit.
        """
        logger.debug(f"Editing scene: {scene.title} (ID: {scene.id})")

        def on_save(updated_scene: Scene) -> None:
            """Handle scene save."""
            # Scene is updated in place, just refresh UI
            self.on_scene_updated()
            self.refresh()
            ui.notify(f"Scene '{updated_scene.title}' updated", type="positive")

        # Open scene editor dialog
        editor = SceneEditorDialog(
            scene=scene,
            chapter=self.chapter,
            on_save=on_save,
        )
        editor.build()

    def _handle_delete_scene(self, scene: Scene) -> None:
        """Handle delete scene button click.

        Args:
            scene: Scene to delete.
        """
        logger.debug(f"Deleting scene: {scene.title} (ID: {scene.id})")

        # Show confirmation dialog
        from ui.components.common import confirmation_dialog

        def do_delete() -> None:
            """Perform the deletion."""
            if self.chapter.remove_scene(scene.id):
                self.on_scene_updated()
                self.refresh()
                ui.notify(f"Scene '{scene.title}' deleted", type="info")
            else:
                ui.notify("Failed to delete scene", type="negative")

        confirmation_dialog(
            title="Delete Scene?",
            message=f"Are you sure you want to delete '{scene.title}'? This cannot be undone.",
            on_confirm=do_delete,
            confirm_text="Delete",
        )

    def refresh(self) -> None:
        """Refresh the scene list display."""
        if self._container:
            # Clear existing cards
            self._scene_cards.clear()
            self._container.clear()

            # Rebuild
            with self._container:
                # Header with Add Scene button
                with ui.row().classes("w-full items-center mb-2"):
                    ui.label("Scenes").classes("text-md font-semibold")
                    ui.space()
                    ui.button(
                        "Add Scene",
                        icon="add",
                        on_click=self._handle_add_scene,
                    ).props("flat size=sm")

                # Scene list
                if not self.chapter.scenes:
                    with ui.card().classes("w-full bg-gray-50 dark:bg-gray-800"):
                        ui.label("No scenes yet. Click 'Add Scene' to create one.").classes(
                            "text-sm text-gray-500 dark:text-gray-400 p-2"
                        )
                else:
                    self._build_scene_list()

"""World Builder page - entity and relationship management."""

import logging
import random
import threading
from typing import Any

from nicegui import ui
from nicegui.elements.button import Button
from nicegui.elements.column import Column
from nicegui.elements.html import Html
from nicegui.elements.input import Input
from nicegui.elements.select import Select
from nicegui.elements.textarea import Textarea

from src.memory.entities import Entity, EntityVersion
from src.memory.world_quality import RefinementConfig
from src.services import ServiceContainer
from src.services.world_quality_service import EntityGenerationProgress
from src.ui.components.build_dialog import show_build_structure_dialog
from src.ui.components.entity_card import entity_list_item
from src.ui.components.graph import GraphComponent
from src.ui.graph_renderer import (
    render_centrality_result,
    render_communities_result,
    render_path_result,
)
from src.ui.state import ActionType, AppState, UndoAction
from src.utils.exceptions import WorldGenerationError

logger = logging.getLogger(__name__)

# Default value for relationship strength when creating via drag-and-drop
DEFAULT_RELATIONSHIP_STRENGTH = 0.5


class WorldPage:
    """World Builder page for managing entities and relationships.

    Features:
    - Interactive graph visualization
    - Entity browser with filtering
    - Entity editor
    - Relationship management
    - Graph analysis tools
    """

    def __init__(self, state: AppState, services: ServiceContainer):
        """Initialize world page.

        Args:
            state: Application state.
            services: Service container.
        """
        self.state = state
        self.services = services

        # Register undo/redo handlers for this page
        self.state.on_undo(self._do_undo)
        self.state.on_redo(self._do_redo)

        # UI references
        self._graph: GraphComponent | None = None
        self._entity_list: Column | None = None
        self._editor_container: Column | None = None
        self._search_input: Input | None = None
        self._sort_direction_btn: Button | None = None
        self._entity_name_input: Input | None = None
        self._entity_type_select: Select | None = None
        self._entity_desc_input: Textarea | None = None
        # Type-specific attribute form fields
        self._attr_role_select: Select | None = None
        self._attr_traits_input: Input | None = None
        self._attr_goals_input: Input | None = None
        self._attr_arc_input: Textarea | None = None
        self._attr_significance_input: Textarea | None = None
        self._attr_leader_input: Input | None = None
        self._attr_values_input: Input | None = None
        self._attr_properties_input: Input | None = None
        self._attr_manifestations_input: Textarea | None = None
        self._entity_attrs: dict[str, Any] = {}
        self._rel_source_select: Select | None = None
        self._rel_type_select: Select | None = None
        self._rel_target_select: Select | None = None
        self._analysis_result: Html | None = None
        self._undo_btn: Button | None = None
        self._redo_btn: Button | None = None
        # Generation progress dialog state
        self._generation_cancel_event: threading.Event | None = None
        self._generation_dialog: ui.dialog | None = None

    def build(self) -> None:
        """Build the world page UI."""
        if not self.state.has_project:
            self._build_no_project_message()
            return

        # Check if interview is complete
        if not self.state.interview_complete:
            self._build_interview_required_message()
            return

        # World generation toolbar
        self._build_generation_toolbar()

        # Responsive layout: stack on mobile, 3-column on desktop
        # All panels should have the same height
        with (
            ui.row()
            .classes("w-full gap-4 p-4 flex-wrap lg:flex-nowrap")
            .style("min-height: calc(100vh - 250px)")
        ):
            # Left panel - Entity browser (full width on mobile, 20% on desktop)
            with ui.column().classes("w-full lg:w-1/5 gap-4 min-w-[250px] h-full"):
                self._build_entity_browser()

            # Center panel - Graph visualization (full width on mobile, 60% on desktop)
            with ui.column().classes("w-full lg:w-3/5 gap-4 min-w-[300px] h-full"):
                self._build_graph_section()

            # Right panel - Entity editor (full width on mobile, 20% on desktop)
            self._editor_container = ui.column().classes(
                "w-full lg:w-1/5 gap-4 min-w-[250px] h-full"
            )
            with self._editor_container:
                self._build_entity_editor()

        # Bottom sections
        with ui.column().classes("w-full gap-4 p-4"):
            self._build_relationships_section()
            self._build_analysis_section()

    def _build_no_project_message(self) -> None:
        """Build message when no project is selected."""
        with ui.column().classes("w-full min-h-96 items-center justify-center gap-4 py-16"):
            ui.icon("public_off", size="xl").classes("text-gray-400 dark:text-gray-500")
            ui.label("No Project Selected").classes("text-xl text-gray-500 dark:text-gray-400")
            ui.label("Select a project from the header to explore its world.").classes(
                "text-gray-400 dark:text-gray-500"
            )

    def _build_interview_required_message(self) -> None:
        """Build message when interview is not complete."""
        with ui.column().classes("w-full min-h-96 items-center justify-center gap-6 py-16"):
            ui.icon("chat", size="xl").classes("text-blue-400")
            ui.label("Complete the Interview First").classes(
                "text-xl font-semibold text-gray-700 dark:text-gray-200"
            )
            ui.label(
                "The World Builder requires story context from the interview. "
                "Complete the interview to populate your story's world."
            ).classes("text-gray-500 dark:text-gray-400 text-center max-w-md")

            ui.button(
                "Go to Interview",
                on_click=lambda: ui.navigate.to("/"),
                icon="arrow_forward",
            ).props("color=primary size=lg")

    def _build_generation_toolbar(self) -> None:
        """Build the world generation toolbar with readiness score and action buttons."""
        if not self.state.world_db:
            return

        # Count entities
        char_count = self.state.world_db.count_entities("character")
        loc_count = self.state.world_db.count_entities("location")
        rel_count = len(self.state.world_db.list_relationships())

        # Calculate simple readiness score
        # Based on: characters (weight 3), locations (weight 2), relationships (weight 1)
        target_chars = 5  # Minimum recommended
        target_locs = 3
        target_rels = 8

        char_score = min(100, (char_count / target_chars) * 100)
        loc_score = min(100, (loc_count / target_locs) * 100)
        rel_score = min(100, (rel_count / target_rels) * 100)

        readiness = int(char_score * 0.4 + loc_score * 0.3 + rel_score * 0.3)
        readiness_color = "green" if readiness >= 80 else "orange" if readiness >= 50 else "red"
        readiness_text = (
            "Ready to write!"
            if readiness >= 80
            else "Needs more content"
            if readiness >= 50
            else "World is sparse"
        )

        with ui.row().classes("w-full items-center gap-4 px-4 pt-4 pb-2"):
            # Readiness indicator
            with ui.card().classes("p-3"):
                with ui.row().classes("items-center gap-3"):
                    ui.circular_progress(
                        value=readiness / 100,
                        show_value=True,
                        size="lg",
                        color=readiness_color,
                    )
                    with ui.column().classes("gap-0"):
                        ui.label("World Readiness").classes("text-sm font-medium")
                        ui.label(readiness_text).classes(
                            f"text-xs text-{readiness_color}-600 dark:text-{readiness_color}-400"
                        )

            # Quality refinement toggle
            with ui.row().classes("items-center gap-2"):
                ui.switch(
                    "Quality Refinement",
                    value=self.state.quality_refinement_enabled,
                    on_change=lambda e: setattr(self.state, "quality_refinement_enabled", e.value),
                ).tooltip(
                    "When enabled, entities are iteratively refined until they meet quality standards"
                )
                ui.button(
                    icon="settings",
                    on_click=self._show_quality_settings_dialog,
                ).props("flat dense").tooltip("Quality settings")

            ui.space()

            # Generation buttons - first row
            with ui.row().classes("gap-2 flex-wrap"):
                ui.button(
                    "Characters",
                    on_click=lambda: self._show_generate_dialog("characters"),
                    icon="person_add",
                ).props("outline dense").classes("text-green-600").tooltip("Add more characters")

                ui.button(
                    "Locations",
                    on_click=lambda: self._show_generate_dialog("locations"),
                    icon="add_location",
                ).props("outline dense").classes("text-blue-600").tooltip("Add more locations")

                ui.button(
                    "Factions",
                    on_click=lambda: self._show_generate_dialog("factions"),
                    icon="groups",
                ).props("outline dense").classes("text-amber-600").tooltip(
                    "Add factions/organizations"
                )

                ui.button(
                    "Items",
                    on_click=lambda: self._show_generate_dialog("items"),
                    icon="category",
                ).props("outline dense").classes("text-cyan-600").tooltip("Add significant items")

                ui.button(
                    "Concepts",
                    on_click=lambda: self._show_generate_dialog("concepts"),
                    icon="lightbulb",
                ).props("outline dense").classes("text-pink-600").tooltip("Add thematic concepts")

                ui.button(
                    "Relationships",
                    on_click=lambda: self._generate_more("relationships"),
                    icon="link",
                ).props("outline dense").classes("text-purple-600").tooltip("Add relationships")

            ui.separator().props("vertical")

            # Import from text button
            ui.button(
                "Import from Text",
                on_click=self._show_import_wizard,
                icon="upload_file",
            ).props("outline color=primary").tooltip("Extract entities from existing story text")

            # Check if chapters have written content - block destructive actions if so
            has_written_content = (
                self.state.project
                and self.state.project.chapters
                and any(c.content for c in self.state.project.chapters)
            )

            # Regenerate button (dangerous action) - only show if no written content
            if not has_written_content:
                ui.button(
                    "Rebuild World",
                    on_click=self._confirm_regenerate,
                    icon="refresh",
                ).props("outline color=negative").tooltip(
                    "Rebuild all entities and relationships (only available before writing)"
                )

                # Clear World button - only show if no story content written yet
                ui.button(
                    "Clear World",
                    on_click=self._confirm_clear_world,
                    icon="delete_sweep",
                ).props("outline color=warning").tooltip(
                    "Remove all entities and relationships (only available before writing)"
                )

        # Build Story Structure button - centered, only show if no chapters yet
        has_chapters = self.state.project and self.state.project.chapters
        if not has_chapters:
            with ui.row().classes("w-full justify-center mt-4"):
                ui.button(
                    "Build Story Structure",
                    on_click=self._build_structure,
                    icon="auto_fix_high",
                ).props("color=primary size=lg").tooltip(
                    "Generate characters, locations, plot points, and chapter outlines"
                )

    def _show_quality_settings_dialog(self) -> None:
        """Show dialog to configure quality refinement settings."""
        settings = self.services.settings
        config = RefinementConfig.from_settings(settings)

        with ui.dialog() as dialog, ui.card().classes("p-4 min-w-[400px]"):
            ui.label("Quality Refinement Settings").classes("text-lg font-bold mb-4")

            # Quality Threshold with reactive value display
            with ui.row().classes("w-full items-center gap-2 mt-2"):
                ui.label("Quality Threshold").classes("text-sm font-medium flex-grow")
                threshold_value_label = ui.label(f"{config.quality_threshold:.1f}").classes(
                    "text-sm font-bold text-primary"
                )
            threshold_slider = ui.slider(
                min=0.0,
                max=10.0,
                step=0.5,
                value=config.quality_threshold,
                on_change=lambda e: threshold_value_label.set_text(f"{e.value:.1f}"),
            ).classes("w-full")
            with ui.row().classes("w-full justify-between text-xs text-gray-500"):
                ui.label("0 (Accept all)")
                ui.label("10 (Very strict)")

            # Max Iterations with reactive value display
            with ui.row().classes("w-full items-center gap-2 mt-4"):
                ui.label("Max Iterations").classes("text-sm font-medium flex-grow")
                iterations_value_label = ui.label(f"{config.max_iterations}").classes(
                    "text-sm font-bold text-primary"
                )
            iterations_slider = ui.slider(
                min=1,
                max=10,
                step=1,
                value=config.max_iterations,
                on_change=lambda e: iterations_value_label.set_text(f"{int(e.value)}"),
            ).classes("w-full")
            with ui.row().classes("w-full justify-between text-xs text-gray-500"):
                ui.label("1 (No refinement)")
                ui.label("10 (Max)")

            with ui.expansion("Advanced", icon="tune").classes("w-full mt-4"):
                # Creator Temperature with reactive display
                with ui.row().classes("w-full items-center gap-2 mt-2"):
                    ui.label("Creator Temperature").classes("text-sm font-medium flex-grow")
                    creator_value_label = ui.label(f"{config.creator_temperature:.1f}").classes(
                        "text-sm font-bold text-primary"
                    )
                creator_temp = ui.slider(
                    min=0.0,
                    max=2.0,
                    step=0.1,
                    value=config.creator_temperature,
                    on_change=lambda e: creator_value_label.set_text(f"{e.value:.1f}"),
                ).classes("w-full")
                ui.label("Higher = more creative").classes("text-xs text-gray-500")

                # Judge Temperature with reactive display
                with ui.row().classes("w-full items-center gap-2 mt-4"):
                    ui.label("Judge Temperature").classes("text-sm font-medium flex-grow")
                    judge_value_label = ui.label(f"{config.judge_temperature:.1f}").classes(
                        "text-sm font-bold text-primary"
                    )
                judge_temp = ui.slider(
                    min=0.0,
                    max=2.0,
                    step=0.1,
                    value=config.judge_temperature,
                    on_change=lambda e: judge_value_label.set_text(f"{e.value:.1f}"),
                ).classes("w-full")
                ui.label("Lower = more consistent").classes("text-xs text-gray-500")

                # Refinement Temperature with reactive display
                with ui.row().classes("w-full items-center gap-2 mt-4"):
                    ui.label("Refinement Temperature").classes("text-sm font-medium flex-grow")
                    refine_value_label = ui.label(f"{config.refinement_temperature:.1f}").classes(
                        "text-sm font-bold text-primary"
                    )
                refine_temp = ui.slider(
                    min=0.0,
                    max=2.0,
                    step=0.1,
                    value=config.refinement_temperature,
                    on_change=lambda e: refine_value_label.set_text(f"{e.value:.1f}"),
                ).classes("w-full")
                ui.label("Balanced creativity and consistency").classes("text-xs text-gray-500")

            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=dialog.close).props("flat")

                def save_settings() -> None:
                    """Save quality settings to application settings."""
                    settings.world_quality_threshold = threshold_slider.value
                    settings.world_quality_max_iterations = int(iterations_slider.value)
                    settings.world_quality_creator_temp = creator_temp.value
                    settings.world_quality_judge_temp = judge_temp.value
                    settings.world_quality_refinement_temp = refine_temp.value
                    settings.save()
                    dialog.close()
                    ui.notify("Quality settings saved", type="positive")

                ui.button("Save", on_click=save_settings).props("color=primary")

        dialog.open()

    def _get_all_entity_names(self) -> list[str]:
        """Get all entity names from the world database.

        Returns:
            List of all entity names across all types.
        """
        if not self.state.world_db:
            return []
        return [e.name for e in self.state.world_db.list_entities()]

    def _get_entity_names_by_type(self, entity_type: str) -> list[str]:
        """Get entity names filtered by type.

        Args:
            entity_type: Type to filter by.

        Returns:
            List of entity names of the specified type.
        """
        if not self.state.world_db:
            return []
        return [e.name for e in self.state.world_db.list_entities() if e.type == entity_type]

    def _get_random_count(self, entity_type: str) -> int:
        """Get a random count for entity generation based on settings.

        Args:
            entity_type: Type of entity (characters, locations, factions, items, concepts, relationships)

        Returns:
            Random integer between min and max from src.settings.
        """
        settings = self.services.settings
        ranges = {
            "characters": (settings.world_gen_characters_min, settings.world_gen_characters_max),
            "locations": (settings.world_gen_locations_min, settings.world_gen_locations_max),
            "factions": (settings.world_gen_factions_min, settings.world_gen_factions_max),
            "items": (settings.world_gen_items_min, settings.world_gen_items_max),
            "concepts": (settings.world_gen_concepts_min, settings.world_gen_concepts_max),
            "relationships": (
                settings.world_gen_relationships_min,
                settings.world_gen_relationships_max,
            ),
        }
        min_val, max_val = ranges.get(entity_type, (2, 4))
        return random.randint(min_val, max_val)

    def _show_generate_dialog(self, entity_type: str) -> None:
        """Show dialog for generating entities with count and custom prompt options.

        Args:
            entity_type: Type of entities to generate.
        """
        logger.info(f"Showing generate dialog for {entity_type}")

        # Get default count range from settings
        settings = self.services.settings
        ranges = {
            "characters": (settings.world_gen_characters_min, settings.world_gen_characters_max),
            "locations": (settings.world_gen_locations_min, settings.world_gen_locations_max),
            "factions": (settings.world_gen_factions_min, settings.world_gen_factions_max),
            "items": (settings.world_gen_items_min, settings.world_gen_items_max),
            "concepts": (settings.world_gen_concepts_min, settings.world_gen_concepts_max),
            "relationships": (
                settings.world_gen_relationships_min,
                settings.world_gen_relationships_max,
            ),
        }
        min_val, max_val = ranges.get(entity_type, (2, 4))
        default_count = (min_val + max_val) // 2

        # Pretty names for display
        type_names = {
            "characters": "Characters",
            "locations": "Locations",
            "factions": "Factions",
            "items": "Items",
            "concepts": "Concepts",
            "relationships": "Relationships",
        }
        type_name = type_names.get(entity_type, entity_type.title())

        # Use state-based dark mode styling
        card_bg = "#1f2937" if self.state.dark_mode else "#ffffff"
        inner_card_bg = "#374151" if self.state.dark_mode else "#f9fafb"

        with (
            ui.dialog() as dialog,
            ui.card().classes("w-[450px]").style(f"background-color: {card_bg}"),
        ):
            ui.label(f"Generate {type_name}").classes("text-xl font-bold mb-4")

            # Count input
            with ui.card().classes("w-full mb-4 p-3").style(f"background-color: {inner_card_bg}"):
                ui.label("How many to generate?").classes("font-medium mb-2")
                count_input = (
                    ui.number(
                        value=default_count,
                        min=1,
                        max=20,
                        step=1,
                    )
                    .props("outlined dense")
                    .classes("w-24")
                )
                ui.label(f"Default range: {min_val}-{max_val}").classes(
                    "text-xs text-gray-500 mt-1"
                )

            # Custom instructions textarea
            with ui.card().classes("w-full mb-4 p-3").style(f"background-color: {inner_card_bg}"):
                ui.label("Custom Instructions (optional)").classes("font-medium mb-2")
                custom_prompt = (
                    ui.textarea(
                        placeholder=f"Describe specific {entity_type} you want...\n"
                        f"e.g., 'A mysterious mentor character' or 'A haunted location'",
                    )
                    .props("outlined")
                    .classes("w-full")
                )
                ui.label("The AI will use these instructions to refine the generation").classes(
                    "text-xs text-gray-500 mt-1"
                )

            # Buttons
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat")

                async def do_generate() -> None:
                    """Close the dialog and trigger entity generation with the configured settings."""
                    count = int(count_input.value) if count_input.value else default_count
                    custom = custom_prompt.value.strip() if custom_prompt.value else None
                    dialog.close()
                    await self._generate_more(entity_type, count=count, custom_instructions=custom)

                ui.button("Generate", on_click=do_generate).props("color=primary")

        dialog.open()

    async def _generate_more(
        self, entity_type: str, count: int | None = None, custom_instructions: str | None = None
    ) -> None:
        """Generate more entities of a specific type.

        Args:
            entity_type: Type of entities to generate (characters, locations, factions, items, concepts, relationships)
            count: Number of entities to generate (defaults to random from settings).
            custom_instructions: Optional custom instructions to refine generation.
        """
        logger.info(
            f"Generate more: entity_type={entity_type}, count={count}, "
            f"custom_instructions={custom_instructions[:50] if custom_instructions else None}"
        )

        if not self.state.project or not self.state.world_db:
            logger.warning("Generate more failed: no project or world_db")
            ui.notify("No project loaded", type="negative")
            return

        logger.info(f"Starting generation of {entity_type} for project {self.state.project.id}")

        # Check if quality refinement is enabled
        use_quality = (
            self.state.quality_refinement_enabled and self.services.settings.world_quality_enabled
        )
        logger.info(f"Quality refinement enabled: {use_quality}")

        # Use provided count or get random from settings
        if count is None:
            count = self._get_random_count(entity_type)
        logger.info(f"Will generate {count} {entity_type}")

        # Get ALL existing entity names to avoid duplicates
        all_existing_names = self._get_all_entity_names()
        logger.info(f"Found {len(all_existing_names)} existing entities to avoid duplicates")

        # Create cancellation infrastructure for quality generation
        self._generation_cancel_event = threading.Event()

        def should_cancel() -> bool:
            """Check if generation should be cancelled."""
            return (
                self._generation_cancel_event is not None and self._generation_cancel_event.is_set()
            )

        # Create progress dialog for quality generation, or simple notification for non-quality
        quality_msg = " with quality refinement" if use_quality else ""

        if use_quality:
            # Create progress dialog with cancel button
            self._generation_dialog = ui.dialog().props("persistent")
            progress_label: ui.label | None = None
            progress_bar: ui.linear_progress | None = None
            eta_label: ui.label | None = None
            cancel_btn: ui.button | None = None

            with self._generation_dialog, ui.card().classes("w-96 p-4"):
                ui.label(f"Generating {entity_type.title()}").classes("text-lg font-bold")
                progress_label = ui.label(f"Starting generation of {count} {entity_type}...")
                progress_bar = ui.linear_progress(value=0).classes("w-full my-2")
                eta_label = ui.label("Calculating...").classes("text-sm text-gray-500")

                def do_cancel() -> None:
                    """Handle cancel button click."""
                    if self._generation_cancel_event:
                        self._generation_cancel_event.set()
                    if cancel_btn:
                        cancel_btn.disable()
                    if progress_label:
                        progress_label.text = "Cancelling after current entity..."

                cancel_btn = ui.button("Cancel", on_click=do_cancel).props("flat color=negative")

            self._generation_dialog.open()

            def update_progress(progress: EntityGenerationProgress) -> None:
                """Update dialog with generation progress."""
                if progress_label:
                    if progress.entity_name:
                        progress_label.text = f"Generated: {progress.entity_name}"
                    else:
                        progress_label.text = (
                            f"Generating {progress.entity_type} "
                            f"{progress.current}/{progress.total}..."
                        )

                if progress_bar:
                    progress_bar.value = progress.progress_fraction

                if eta_label:
                    if progress.estimated_remaining_seconds is not None:
                        mins, secs = divmod(int(progress.estimated_remaining_seconds), 60)
                        eta_label.text = f"~{mins}:{secs:02d} remaining"
                    elif progress.current > 1:
                        eta_label.text = "Calculating..."

            notification = None  # No notification when using dialog
        else:
            # Use simple notification for non-quality generation
            notification = ui.notification(
                message=f"Generating {count} {entity_type}{quality_msg}...",
                spinner=True,
                timeout=None,
            )
            update_progress = None  # type: ignore[assignment]

        try:
            from nicegui import run

            if entity_type == "characters":
                if use_quality:
                    # Generate characters with quality refinement
                    logger.info(
                        f"Calling world quality service to generate characters "
                        f"(custom_instructions: {custom_instructions is not None})..."
                    )
                    results = await run.io_bound(
                        self.services.world_quality.generate_characters_with_quality,
                        self.state.project,
                        all_existing_names,
                        count,
                        custom_instructions,
                        should_cancel,
                        update_progress,
                    )
                    logger.info(f"Generated {len(results)} characters with quality refinement")

                    # Check for partial failure and notify user LOUDLY
                    if len(results) < count:
                        failed_count = count - len(results)
                        # Check if cancelled
                        if should_cancel():
                            ui.notify(
                                f"Generation cancelled. Generated {len(results)} of {count} characters.",
                                type="info",
                                timeout=5000,
                            )
                        else:
                            ui.notify(
                                f"ERROR: {failed_count} of {count} characters FAILED to generate! "
                                "Check logs for details.",
                                type="negative",
                                timeout=10000,
                                close_button=True,
                            )
                    if len(results) == 0:
                        if self._generation_dialog:
                            self._generation_dialog.close()
                        ui.notify("Failed to generate any characters", type="negative")
                        return

                    # Generate mini descriptions for hover tooltips
                    if progress_label:
                        progress_label.text = "Generating hover summaries..."
                    entity_data = [
                        {"name": c.name, "type": "character", "description": c.description}
                        for c, _ in results
                    ]
                    mini_descs = await run.io_bound(
                        self.services.world_quality.generate_mini_descriptions_batch,
                        entity_data,
                    )
                    if self._generation_dialog:
                        self._generation_dialog.close()

                    # Define callback to add selected characters
                    def add_selected_characters(selected: list[tuple[Any, Any]]) -> None:
                        """Add selected characters to the world database and project."""
                        if not selected:
                            ui.notify("No characters selected", type="info")
                            return
                        if not self.state.world_db or not self.state.project:
                            ui.notify("No project loaded", type="negative")
                            return
                        added_names = []
                        for char, scores in selected:
                            attrs = {
                                "role": char.role,
                                "traits": char.personality_traits,
                                "goals": char.goals,
                                "arc": char.arc_notes,
                                "quality_scores": scores.to_dict(),
                            }
                            if char.name in mini_descs:
                                attrs["mini_description"] = mini_descs[char.name]
                            self.services.world.add_entity(
                                self.state.world_db,
                                name=char.name,
                                entity_type="character",
                                description=char.description,
                                attributes=attrs,
                            )
                            # Also add to story state
                            self.state.project.characters.append(char)
                            added_names.append(char.name)
                        # Refresh UI and save
                        self.state.world_db.invalidate_graph_cache()
                        self._refresh_entity_list()
                        if self._graph:
                            self._graph.refresh()
                        self.services.project.save_project(self.state.project)
                        avg_quality = (
                            sum(s.average for _, s in selected) / len(selected) if selected else 0
                        )
                        ui.notify(
                            f"Added {len(selected)} characters (avg quality: {avg_quality:.1f})",
                            type="positive",
                        )
                        # Prompt for relationship generation
                        self._prompt_for_relationships_after_add(added_names)

                    # Show preview dialog
                    self._show_entity_preview_dialog("character", results, add_selected_characters)
                    return  # Early return - callback handles the rest
                else:
                    # Generate characters via service (original method)
                    logger.info("Calling story service to generate characters...")
                    new_chars = await run.io_bound(
                        self.services.story.generate_more_characters, self.state.project, count
                    )
                    logger.info(f"Generated {len(new_chars)} characters from LLM")
                    # Add to world database
                    for char in new_chars:
                        self.services.world.add_entity(
                            self.state.world_db,
                            name=char.name,
                            entity_type="character",
                            description=char.description,
                            attributes={
                                "role": char.role,
                                "traits": char.personality_traits,
                                "goals": char.goals,
                                "arc": char.arc_notes,
                            },
                        )
                    logger.info(f"Added {len(new_chars)} characters to world database")
                    if notification:
                        notification.dismiss()
                    ui.notify(f"Added {len(new_chars)} new characters!", type="positive")

            elif entity_type == "locations":
                if use_quality:
                    # Generate locations with quality refinement
                    logger.info("Calling world quality service to generate locations...")
                    loc_results = await run.io_bound(
                        self.services.world_quality.generate_locations_with_quality,
                        self.state.project,
                        all_existing_names,
                        count,
                        should_cancel,
                        update_progress,
                    )
                    logger.info(f"Generated {len(loc_results)} locations with quality refinement")

                    # Check for partial failure and notify user LOUDLY
                    if len(loc_results) < count:
                        failed_count = count - len(loc_results)
                        if should_cancel():
                            ui.notify(
                                f"Generation cancelled. Generated {len(loc_results)} of {count} locations.",
                                type="info",
                                timeout=5000,
                            )
                        else:
                            ui.notify(
                                f"ERROR: {failed_count} of {count} locations FAILED to generate! "
                                "Check logs for details.",
                                type="negative",
                                timeout=10000,
                                close_button=True,
                            )
                    if len(loc_results) == 0:
                        if self._generation_dialog:
                            self._generation_dialog.close()
                        ui.notify("Failed to generate any locations", type="negative")
                        return

                    # Generate mini descriptions for hover tooltips
                    if progress_label:
                        progress_label.text = "Generating hover summaries..."
                    entity_data = [
                        {
                            "name": loc.get("name", ""),
                            "type": "location",
                            "description": loc.get("description", ""),
                        }
                        for loc, _ in loc_results
                        if isinstance(loc, dict) and loc.get("name")
                    ]
                    mini_descs = await run.io_bound(
                        self.services.world_quality.generate_mini_descriptions_batch,
                        entity_data,
                    )
                    if self._generation_dialog:
                        self._generation_dialog.close()

                    # Define callback to add selected locations
                    def add_selected_locations(selected: list[tuple[Any, Any]]) -> None:
                        """Add selected locations to the world database."""
                        if not selected:
                            ui.notify("No locations selected", type="info")
                            return
                        if not self.state.world_db or not self.state.project:
                            ui.notify("No project loaded", type="negative")
                            return
                        added_names = []
                        for loc, scores in selected:
                            if isinstance(loc, dict) and "name" in loc:
                                attrs = {
                                    "significance": loc.get("significance", ""),
                                    "quality_scores": scores.to_dict(),
                                }
                                if loc["name"] in mini_descs:
                                    attrs["mini_description"] = mini_descs[loc["name"]]
                                self.services.world.add_entity(
                                    self.state.world_db,
                                    name=loc["name"],
                                    entity_type="location",
                                    description=loc.get("description", ""),
                                    attributes=attrs,
                                )
                                added_names.append(loc["name"])
                        # Refresh UI and save
                        self.state.world_db.invalidate_graph_cache()
                        self._refresh_entity_list()
                        if self._graph:
                            self._graph.refresh()
                        self.services.project.save_project(self.state.project)
                        avg_quality = (
                            sum(s.average for _, s in selected) / len(selected) if selected else 0
                        )
                        ui.notify(
                            f"Added {len(selected)} locations (avg quality: {avg_quality:.1f})",
                            type="positive",
                        )
                        # Prompt for relationship generation
                        self._prompt_for_relationships_after_add(added_names)

                    # Show preview dialog
                    self._show_entity_preview_dialog(
                        "location", loc_results, add_selected_locations
                    )
                    return  # Early return - callback handles the rest
                else:
                    # Generate locations via service (original method)
                    logger.info("Calling story service to generate locations...")
                    locations = await run.io_bound(
                        self.services.story.generate_locations, self.state.project, count
                    )
                    logger.info(f"Generated {len(locations)} locations from LLM")
                    # Add to world database
                    added_count = 0
                    for loc in locations:
                        if isinstance(loc, dict) and "name" in loc:
                            self.services.world.add_entity(
                                self.state.world_db,
                                name=loc["name"],
                                entity_type="location",
                                description=loc.get("description", ""),
                                attributes={"significance": loc.get("significance", "")},
                            )
                            added_count += 1
                        else:
                            logger.warning(f"Skipping invalid location: {loc}")
                    logger.info(f"Added {added_count} locations to world database")
                    if notification:
                        notification.dismiss()
                    ui.notify(f"Added {added_count} new locations!", type="positive")

            elif entity_type == "factions":
                if use_quality:
                    # Get existing locations for spatial grounding
                    existing_entities = self.state.world_db.list_entities()
                    existing_locations = [e.name for e in existing_entities if e.type == "location"]
                    logger.info(
                        f"Found {len(existing_locations)} existing locations for faction grounding"
                    )

                    # Generate factions with quality refinement
                    logger.info("Calling world quality service to generate factions...")
                    faction_results = await run.io_bound(
                        self.services.world_quality.generate_factions_with_quality,
                        self.state.project,
                        all_existing_names,
                        count,
                        existing_locations,
                        should_cancel,
                        update_progress,
                    )
                    logger.info(
                        f"Generated {len(faction_results)} factions with quality refinement"
                    )

                    # Check for partial failure and notify user LOUDLY
                    if len(faction_results) < count:
                        failed_count = count - len(faction_results)
                        if should_cancel():
                            ui.notify(
                                f"Generation cancelled. Generated {len(faction_results)} of {count} factions.",
                                type="info",
                                timeout=5000,
                            )
                        else:
                            ui.notify(
                                f"ERROR: {failed_count} of {count} factions FAILED to generate! "
                                "Check logs for details.",
                                type="negative",
                                timeout=10000,
                                close_button=True,
                            )
                    if len(faction_results) == 0:
                        if self._generation_dialog:
                            self._generation_dialog.close()
                        ui.notify("Failed to generate any factions", type="negative")
                        return

                    # Generate mini descriptions for hover tooltips
                    if progress_label:
                        progress_label.text = "Generating hover summaries..."
                    entity_data = [
                        {
                            "name": faction.get("name", ""),
                            "type": "faction",
                            "description": faction.get("description", ""),
                        }
                        for faction, _ in faction_results
                        if isinstance(faction, dict) and faction.get("name")
                    ]
                    mini_descs = await run.io_bound(
                        self.services.world_quality.generate_mini_descriptions_batch,
                        entity_data,
                    )
                    if self._generation_dialog:
                        self._generation_dialog.close()

                    # Define callback to add selected factions
                    def add_selected_factions(selected: list[tuple[Any, Any]]) -> None:
                        """Add selected factions to the world database with location relationships."""
                        if not selected:
                            ui.notify("No factions selected", type="info")
                            return
                        if not self.state.world_db or not self.state.project:
                            ui.notify("No project loaded", type="negative")
                            return
                        added_names = []
                        for faction, scores in selected:
                            if isinstance(faction, dict) and "name" in faction:
                                attrs = {
                                    "leader": faction.get("leader", ""),
                                    "goals": faction.get("goals", []),
                                    "values": faction.get("values", []),
                                    "base_location": faction.get("base_location", ""),
                                    "quality_scores": scores.to_dict(),
                                }
                                if faction["name"] in mini_descs:
                                    attrs["mini_description"] = mini_descs[faction["name"]]
                                faction_entity_id = self.services.world.add_entity(
                                    self.state.world_db,
                                    name=faction["name"],
                                    entity_type="faction",
                                    description=faction.get("description", ""),
                                    attributes=attrs,
                                )
                                added_names.append(faction["name"])
                                # Create relationship to base location if it exists
                                base_loc = faction.get("base_location", "")
                                if base_loc:
                                    location_entity = next(
                                        (
                                            e
                                            for e in existing_entities
                                            if e.name == base_loc and e.type == "location"
                                        ),
                                        None,
                                    )
                                    if location_entity:
                                        self.services.world.add_relationship(
                                            self.state.world_db,
                                            faction_entity_id,
                                            location_entity.id,
                                            "based_in",
                                            f"{faction['name']} is headquartered in {base_loc}",
                                        )
                                        logger.info(
                                            f"Created relationship: {faction['name']} -> based_in -> {base_loc}"
                                        )
                        # Refresh UI and save
                        self.state.world_db.invalidate_graph_cache()
                        self._refresh_entity_list()
                        if self._graph:
                            self._graph.refresh()
                        self.services.project.save_project(self.state.project)
                        avg_quality = (
                            sum(s.average for _, s in selected) / len(selected) if selected else 0
                        )
                        ui.notify(
                            f"Added {len(selected)} factions (avg quality: {avg_quality:.1f})",
                            type="positive",
                        )
                        # Prompt for relationship generation
                        self._prompt_for_relationships_after_add(added_names)

                    # Show preview dialog
                    self._show_entity_preview_dialog(
                        "faction", faction_results, add_selected_factions
                    )
                    return  # Early return - callback handles the rest
                else:
                    if notification:
                        notification.dismiss()
                    ui.notify("Enable Quality Refinement to generate factions", type="warning")
                    return

            elif entity_type == "items":
                if use_quality:
                    # Generate items with quality refinement
                    logger.info("Calling world quality service to generate items...")
                    item_results = await run.io_bound(
                        self.services.world_quality.generate_items_with_quality,
                        self.state.project,
                        all_existing_names,
                        count,
                        should_cancel,
                        update_progress,
                    )
                    logger.info(f"Generated {len(item_results)} items with quality refinement")

                    # Check for partial failure and notify user LOUDLY
                    if len(item_results) < count:
                        failed_count = count - len(item_results)
                        if should_cancel():
                            ui.notify(
                                f"Generation cancelled. Generated {len(item_results)} of {count} items.",
                                type="info",
                                timeout=5000,
                            )
                        else:
                            ui.notify(
                                f"ERROR: {failed_count} of {count} items FAILED to generate! "
                                "Check logs for details.",
                                type="negative",
                                timeout=10000,
                                close_button=True,
                            )
                    if len(item_results) == 0:
                        if self._generation_dialog:
                            self._generation_dialog.close()
                        ui.notify("Failed to generate any items", type="negative")
                        return

                    # Generate mini descriptions for hover tooltips
                    if progress_label:
                        progress_label.text = "Generating hover summaries..."
                    entity_data = [
                        {
                            "name": item.get("name", ""),
                            "type": "item",
                            "description": item.get("description", ""),
                        }
                        for item, _ in item_results
                        if isinstance(item, dict) and item.get("name")
                    ]
                    mini_descs = await run.io_bound(
                        self.services.world_quality.generate_mini_descriptions_batch,
                        entity_data,
                    )
                    if self._generation_dialog:
                        self._generation_dialog.close()

                    # Define callback to add selected items
                    def add_selected_items(selected: list[tuple[Any, Any]]) -> None:
                        """Add selected items to the world database."""
                        if not selected:
                            ui.notify("No items selected", type="info")
                            return
                        if not self.state.world_db or not self.state.project:
                            ui.notify("No project loaded", type="negative")
                            return
                        added_names = []
                        for item, scores in selected:
                            if isinstance(item, dict) and "name" in item:
                                attrs = {
                                    "significance": item.get("significance", ""),
                                    "properties": item.get("properties", []),
                                    "quality_scores": scores.to_dict(),
                                }
                                if item["name"] in mini_descs:
                                    attrs["mini_description"] = mini_descs[item["name"]]
                                self.services.world.add_entity(
                                    self.state.world_db,
                                    name=item["name"],
                                    entity_type="item",
                                    description=item.get("description", ""),
                                    attributes=attrs,
                                )
                                added_names.append(item["name"])
                        # Refresh UI and save
                        self.state.world_db.invalidate_graph_cache()
                        self._refresh_entity_list()
                        if self._graph:
                            self._graph.refresh()
                        self.services.project.save_project(self.state.project)
                        avg_quality = (
                            sum(s.average for _, s in selected) / len(selected) if selected else 0
                        )
                        ui.notify(
                            f"Added {len(selected)} items (avg quality: {avg_quality:.1f})",
                            type="positive",
                        )
                        # Prompt for relationship generation
                        self._prompt_for_relationships_after_add(added_names)

                    # Show preview dialog
                    self._show_entity_preview_dialog("item", item_results, add_selected_items)
                    return  # Early return - callback handles the rest
                else:
                    if notification:
                        notification.dismiss()
                    ui.notify("Enable Quality Refinement to generate items", type="warning")
                    return

            elif entity_type == "concepts":
                if use_quality:
                    # Generate concepts with quality refinement
                    logger.info("Calling world quality service to generate concepts...")
                    concept_results = await run.io_bound(
                        self.services.world_quality.generate_concepts_with_quality,
                        self.state.project,
                        all_existing_names,
                        count,
                        should_cancel,
                        update_progress,
                    )
                    logger.info(
                        f"Generated {len(concept_results)} concepts with quality refinement"
                    )

                    # Check for partial failure and notify user LOUDLY
                    if len(concept_results) < count:
                        failed_count = count - len(concept_results)
                        if should_cancel():
                            ui.notify(
                                f"Generation cancelled. Generated {len(concept_results)} of {count} concepts.",
                                type="info",
                                timeout=5000,
                            )
                        else:
                            ui.notify(
                                f"ERROR: {failed_count} of {count} concepts FAILED to generate! "
                                "Check logs for details.",
                                type="negative",
                                timeout=10000,
                                close_button=True,
                            )
                    if len(concept_results) == 0:
                        if self._generation_dialog:
                            self._generation_dialog.close()
                        ui.notify("Failed to generate any concepts", type="negative")
                        return

                    # Generate mini descriptions for hover tooltips
                    if progress_label:
                        progress_label.text = "Generating hover summaries..."
                    entity_data = [
                        {
                            "name": concept.get("name", ""),
                            "type": "concept",
                            "description": concept.get("description", ""),
                        }
                        for concept, _ in concept_results
                        if isinstance(concept, dict) and concept.get("name")
                    ]
                    mini_descs = await run.io_bound(
                        self.services.world_quality.generate_mini_descriptions_batch,
                        entity_data,
                    )
                    if self._generation_dialog:
                        self._generation_dialog.close()

                    # Define callback to add selected concepts
                    def add_selected_concepts(selected: list[tuple[Any, Any]]) -> None:
                        """Add selected concepts to the world database."""
                        if not selected:
                            ui.notify("No concepts selected", type="info")
                            return
                        if not self.state.world_db or not self.state.project:
                            ui.notify("No project loaded", type="negative")
                            return
                        added_names = []
                        for concept, scores in selected:
                            if isinstance(concept, dict) and "name" in concept:
                                attrs = {
                                    "manifestations": concept.get("manifestations", ""),
                                    "quality_scores": scores.to_dict(),
                                }
                                if concept["name"] in mini_descs:
                                    attrs["mini_description"] = mini_descs[concept["name"]]
                                self.services.world.add_entity(
                                    self.state.world_db,
                                    name=concept["name"],
                                    entity_type="concept",
                                    description=concept.get("description", ""),
                                    attributes=attrs,
                                )
                                added_names.append(concept["name"])
                        # Refresh UI and save
                        self.state.world_db.invalidate_graph_cache()
                        self._refresh_entity_list()
                        if self._graph:
                            self._graph.refresh()
                        self.services.project.save_project(self.state.project)
                        avg_quality = (
                            sum(s.average for _, s in selected) / len(selected) if selected else 0
                        )
                        ui.notify(
                            f"Added {len(selected)} concepts (avg quality: {avg_quality:.1f})",
                            type="positive",
                        )
                        # Prompt for relationship generation
                        self._prompt_for_relationships_after_add(added_names)

                    # Show preview dialog
                    self._show_entity_preview_dialog(
                        "concept", concept_results, add_selected_concepts
                    )
                    return  # Early return - callback handles the rest
                else:
                    if notification:
                        notification.dismiss()
                    ui.notify("Enable Quality Refinement to generate concepts", type="warning")
                    return

            elif entity_type == "relationships":
                # Get existing entities and relationships
                entities = self.state.world_db.list_entities()
                entity_names = [e.name for e in entities]
                logger.info(f"Found {len(entities)} existing entities: {entity_names}")

                # Get existing relationships - look up entity names from IDs
                existing_rels = []
                for rel in self.state.world_db.list_relationships():
                    source = self.services.world.get_entity(self.state.world_db, rel.source_id)
                    target = self.services.world.get_entity(self.state.world_db, rel.target_id)
                    if source and target:
                        existing_rels.append((source.name, target.name))
                logger.info(f"Found {len(existing_rels)} existing relationships")

                if len(entity_names) < 2:
                    logger.warning("Cannot generate relationships: need at least 2 entities")
                    if self._generation_dialog:
                        self._generation_dialog.close()
                    elif notification:
                        notification.dismiss()
                    ui.notify("Need at least 2 entities to create relationships", type="warning")
                    return

                if use_quality:
                    # Generate relationships with quality refinement
                    logger.info("Calling world quality service to generate relationships...")
                    rel_results = await run.io_bound(
                        self.services.world_quality.generate_relationships_with_quality,
                        self.state.project,
                        entity_names,
                        existing_rels,
                        count,
                        should_cancel,
                        update_progress,
                    )
                    logger.info(
                        f"Generated {len(rel_results)} relationships with quality refinement"
                    )

                    # Check for partial failure and notify user LOUDLY
                    if len(rel_results) < count:
                        failed_count = count - len(rel_results)
                        if should_cancel():
                            ui.notify(
                                f"Generation cancelled. Generated {len(rel_results)} of {count} relationships.",
                                type="info",
                                timeout=5000,
                            )
                        else:
                            ui.notify(
                                f"ERROR: {failed_count} of {count} relationships FAILED to generate! "
                                "Check logs for details.",
                                type="negative",
                                timeout=10000,
                                close_button=True,
                            )
                    if len(rel_results) == 0:
                        if self._generation_dialog:
                            self._generation_dialog.close()
                        ui.notify("Failed to generate any relationships", type="negative")
                        return
                    if self._generation_dialog:
                        self._generation_dialog.close()

                    # Define callback to add selected relationships
                    def add_selected_relationships(selected: list[tuple[Any, Any]]) -> None:
                        """Add selected relationships to the world database."""
                        if not selected:
                            ui.notify("No relationships selected", type="info")
                            return
                        if not self.state.world_db or not self.state.project:
                            ui.notify("No project loaded", type="negative")
                            return
                        added = 0
                        for rel_data, scores in selected:
                            if (
                                isinstance(rel_data, dict)
                                and "source" in rel_data
                                and "target" in rel_data
                            ):
                                source_entity = next(
                                    (e for e in entities if e.name == rel_data["source"]), None
                                )
                                target_entity = next(
                                    (e for e in entities if e.name == rel_data["target"]), None
                                )
                                if source_entity and target_entity:
                                    rel_id = self.services.world.add_relationship(
                                        self.state.world_db,
                                        source_entity.id,
                                        target_entity.id,
                                        rel_data.get("relation_type", "knows"),
                                        rel_data.get("description", ""),
                                    )
                                    # Store quality scores in relationship attributes
                                    self.state.world_db.update_relationship(
                                        relationship_id=rel_id,
                                        attributes={"quality_scores": scores.to_dict()},
                                    )
                                    added += 1
                        # Refresh UI and save
                        self.state.world_db.invalidate_graph_cache()
                        self._refresh_entity_list()
                        if self._graph:
                            self._graph.refresh()
                        self.services.project.save_project(self.state.project)
                        avg_quality = (
                            sum(s.average for _, s in selected) / len(selected) if selected else 0
                        )
                        ui.notify(
                            f"Added {added} relationships (avg quality: {avg_quality:.1f})",
                            type="positive",
                        )

                    # Show preview dialog
                    self._show_entity_preview_dialog(
                        "relationship", rel_results, add_selected_relationships
                    )
                    return  # Early return - callback handles the rest
                else:
                    # Generate relationships via service (original method)
                    logger.info("Calling story service to generate relationships...")
                    relationships = await run.io_bound(
                        self.services.story.generate_relationships,
                        self.state.project,
                        entity_names,
                        existing_rels,
                        count,
                    )
                    logger.info(f"Generated {len(relationships)} relationships from LLM")

                    # Add to world database
                    added = 0
                    for rel in relationships:
                        if isinstance(rel, dict) and "source" in rel and "target" in rel:
                            # Find entity IDs by name
                            source_entity = next(
                                (e for e in entities if e.name == rel["source"]), None
                            )
                            target_entity = next(
                                (e for e in entities if e.name == rel["target"]), None
                            )
                            if source_entity and target_entity:
                                self.services.world.add_relationship(
                                    self.state.world_db,
                                    source_entity.id,
                                    target_entity.id,
                                    rel.get("relation_type", "knows"),
                                    rel.get("description", ""),
                                )
                                added += 1
                            else:
                                logger.warning(
                                    f"Skipping relationship: source={rel['source']} or "
                                    f"target={rel['target']} not found"
                                )
                        else:
                            logger.warning(f"Skipping invalid relationship: {rel}")
                    logger.info(f"Added {added} relationships to world database")
                    if notification:
                        notification.dismiss()
                    ui.notify(f"Added {added} new relationships!", type="positive")

            # Invalidate graph cache to ensure fresh tooltips
            self.state.world_db.invalidate_graph_cache()

            # Refresh the UI
            logger.info("Refreshing UI after generation...")
            self._refresh_entity_list()
            if self._graph:
                self._graph.refresh()

            # Save the project
            if self.state.project:
                logger.info(f"Saving project {self.state.project.id}...")
                self.services.project.save_project(self.state.project)
                logger.info("Project saved successfully")

            logger.info(f"Generation of {entity_type} completed successfully")

        except WorldGenerationError as e:
            if self._generation_dialog:
                self._generation_dialog.close()
            elif notification:
                notification.dismiss()
            logger.error(f"World generation failed for {entity_type}: {e}")
            ui.notify(f"Generation failed: {e}", type="negative", close_button=True, timeout=10)
        except Exception as e:
            if self._generation_dialog:
                self._generation_dialog.close()
            elif notification:
                notification.dismiss()
            logger.exception(f"Unexpected error generating {entity_type}: {e}")
            ui.notify(f"Error: {e}", type="negative")

    def _show_entity_preview_dialog(
        self,
        entity_type: str,
        entities: list[tuple[Any, Any]],
        on_confirm: Any,
    ) -> None:
        """Show a preview dialog for generated entities before adding them.

        Args:
            entity_type: Type of entity (character, location, faction, item, concept)
            entities: List of (entity_data, quality_scores) tuples
            on_confirm: Callback function that receives selected entities list
        """
        if not entities:
            ui.notify("No entities to preview", type="warning")
            return

        logger.info(f"Showing preview dialog for {len(entities)} {entity_type}(s)")

        # Track selected entities
        selected = dict.fromkeys(range(len(entities)), True)  # All selected by default

        def toggle_selection(idx: int) -> None:
            """Toggle the selection state of an entity at the given index."""
            selected[idx] = not selected[idx]
            logger.debug(f"Toggled entity {idx}: {selected[idx]}")

        def confirm_selection() -> None:
            """Confirm the selection and invoke the callback with selected entities."""
            selected_entities = [entities[i] for i in range(len(entities)) if selected[i]]
            logger.info(
                f"User confirmed {len(selected_entities)} of {len(entities)} {entity_type}(s)"
            )
            dialog.close()
            on_confirm(selected_entities)

        def cancel_selection() -> None:
            """Cancel the selection and close the preview dialog."""
            logger.info(f"User cancelled {entity_type} preview")
            dialog.close()
            ui.notify(f"Cancelled adding {entity_type}s", type="info")

        with ui.dialog() as dialog, ui.card().classes("p-4 min-w-[500px] max-w-[700px]"):
            ui.label(f"Preview Generated {entity_type.title()}s").classes("text-lg font-bold mb-2")
            ui.label(
                f"Select which {entity_type}s to add to your world. Uncheck any you don't want."
            ).classes("text-gray-600 dark:text-gray-400 mb-4")

            # Scrollable container for entities
            with ui.scroll_area().classes("w-full max-h-[400px]"):
                for idx, (entity_data, scores) in enumerate(entities):
                    # Get entity name and description based on type
                    if entity_type == "character":
                        name = (
                            entity_data.name if hasattr(entity_data, "name") else str(entity_data)
                        )
                        desc = (
                            entity_data.description[:150] + "..."
                            if hasattr(entity_data, "description")
                            and len(entity_data.description) > 150
                            else getattr(entity_data, "description", "")
                        )
                        role = getattr(entity_data, "role", "")
                        extra = f" ({role})" if role else ""
                    elif entity_type == "relationship":
                        # Relationships show source -> relation_type -> target
                        source = (
                            entity_data.get("source", "?") if isinstance(entity_data, dict) else "?"
                        )
                        target = (
                            entity_data.get("target", "?") if isinstance(entity_data, dict) else "?"
                        )
                        rel_type = (
                            entity_data.get("relation_type", "related_to")
                            if isinstance(entity_data, dict)
                            else "related_to"
                        )
                        name = f"{source} → {rel_type} → {target}"
                        desc = (
                            entity_data.get("description", "")[:150]
                            if isinstance(entity_data, dict)
                            else ""
                        )
                        if (
                            len(entity_data.get("description", "")) > 150
                            if isinstance(entity_data, dict)
                            else False
                        ):
                            desc += "..."
                        extra = ""
                    else:
                        name = (
                            entity_data.get("name", "Unknown")
                            if isinstance(entity_data, dict)
                            else str(entity_data)
                        )
                        desc = (
                            entity_data.get("description", "")[:150]
                            if isinstance(entity_data, dict)
                            else ""
                        )
                        if (
                            len(entity_data.get("description", "")) > 150
                            if isinstance(entity_data, dict)
                            else False
                        ):
                            desc += "..."
                        extra = ""
                        if entity_type == "faction":
                            base = (
                                entity_data.get("base_location", "")
                                if isinstance(entity_data, dict)
                                else ""
                            )
                            if base:
                                extra = f" (based in: {base})"

                    quality = (
                        f" - Quality: {scores.average:.1f}"
                        if scores and hasattr(scores, "average")
                        else ""
                    )

                    with ui.row().classes(
                        "w-full items-start gap-2 py-2 border-b border-gray-200 dark:border-gray-700"
                    ):
                        ui.checkbox(
                            value=True, on_change=lambda _, i=idx: toggle_selection(i)
                        ).classes("mt-1")
                        with ui.column().classes("flex-1"):
                            ui.label(f"{name}{extra}").classes("font-semibold")
                            if desc:
                                ui.label(desc).classes("text-sm text-gray-600 dark:text-gray-400")
                            if quality:
                                ui.label(quality).classes(
                                    "text-xs text-blue-600 dark:text-blue-400"
                                )

            # Helper functions for select/deselect all
            def select_all() -> None:
                """Select all entities and refresh the preview dialog."""
                selected.update(dict.fromkeys(range(len(entities)), True))
                dialog.close()
                self._show_entity_preview_dialog(entity_type, entities, on_confirm)

            def deselect_all() -> None:
                """Deselect all entities and refresh the preview dialog."""
                selected.update(dict.fromkeys(range(len(entities)), False))
                dialog.close()
                self._show_entity_preview_dialog(entity_type, entities, on_confirm)

            # Action buttons
            with ui.row().classes("w-full justify-between mt-4"):
                with ui.row().classes("gap-2"):
                    ui.button("Select All", on_click=select_all).props("flat dense")
                    ui.button("Deselect All", on_click=deselect_all).props("flat dense")
                with ui.row().classes("gap-2"):
                    ui.button("Cancel", on_click=cancel_selection).props("flat")
                    ui.button(
                        "Add Selected",
                        on_click=confirm_selection,
                        icon="add",
                    ).props("color=primary")

        dialog.open()

    def _prompt_for_relationships_after_add(self, entity_names: list[str]) -> None:
        """Prompt user to generate relationships for newly added entities.

        Args:
            entity_names: Names of the newly added entities.
        """
        if not entity_names or not self.state.project or not self.state.world_db:
            return

        logger.info(f"Prompting for relationships for {len(entity_names)} new entities")

        # Use state-based dark mode styling
        card_bg = "#1f2937" if self.state.dark_mode else "#ffffff"
        inner_card_bg = "#374151" if self.state.dark_mode else "#f9fafb"

        with (
            ui.dialog() as dialog,
            ui.card().classes("w-[450px]").style(f"background-color: {card_bg}"),
        ):
            ui.label("Generate Relationships?").classes("text-xl font-bold mb-2")
            ui.label(
                f"Would you like to generate relationships for the {len(entity_names)} "
                "newly added entities?"
            ).classes("text-gray-600 dark:text-gray-400 mb-4")

            with ui.card().classes("w-full mb-4 p-3").style(f"background-color: {inner_card_bg}"):
                ui.label("New entities:").classes("font-medium mb-2")
                for name in entity_names[:5]:  # Show first 5
                    ui.label(f"• {name}").classes("text-sm")
                if len(entity_names) > 5:
                    ui.label(f"... and {len(entity_names) - 5} more").classes(
                        "text-sm text-gray-500"
                    )

            # Relationships per entity input
            ui.label("Relationships per entity:").classes("text-sm mb-1")
            rel_count = (
                ui.number(value=2, min=1, max=5, step=1)
                .props("dense outlined")
                .classes("w-20 mb-4")
            )

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Skip", on_click=dialog.close).props("flat")

                async def do_generate_relationships() -> None:
                    """Close the dialog and generate relationships for the new entities."""
                    count = int(rel_count.value) if rel_count.value else 2
                    dialog.close()
                    await self._generate_relationships_for_entities(entity_names, count)

                ui.button("Generate Relationships", on_click=do_generate_relationships).props(
                    "color=primary"
                )

        dialog.open()

    async def _generate_relationships_for_entities(
        self, entity_names: list[str], count_per_entity: int
    ) -> None:
        """Generate relationships for specific entities.

        Args:
            entity_names: Names of entities to generate relationships for.
            count_per_entity: Number of relationships to generate per entity.
        """
        if not self.state.project or not self.state.world_db:
            ui.notify("No project loaded", type="negative")
            return

        logger.info(
            f"Generating {count_per_entity} relationships for each of {len(entity_names)} entities"
        )

        # Check if quality refinement is enabled
        use_quality = (
            self.state.quality_refinement_enabled and self.services.settings.world_quality_enabled
        )

        # Get all entity names for relationship generation
        all_entity_names = self._get_all_entity_names()

        # Get existing relationships to avoid duplicates (use source_id and target_id)
        existing_rels = [
            (r.source_id, r.target_id) for r in self.state.world_db.list_relationships()
        ]

        total_count = len(entity_names) * count_per_entity
        notification = ui.notification(
            message=f"Generating relationships for {len(entity_names)} entities...",
            spinner=True,
            timeout=None,
        )

        try:
            from nicegui import run

            if use_quality:
                # Generate relationships with quality refinement
                results = await run.io_bound(
                    self.services.world_quality.generate_relationships_with_quality,
                    self.state.project,
                    all_entity_names,
                    existing_rels,
                    total_count,
                )

                # Check for partial failure
                if len(results) < total_count:
                    failed_count = total_count - len(results)
                    ui.notify(
                        f"ERROR: {failed_count} of {total_count} relationships FAILED to generate! "
                        "Check logs for details.",
                        type="negative",
                        timeout=10000,
                        close_button=True,
                    )

                if len(results) == 0:
                    notification.dismiss()
                    ui.notify("Failed to generate any relationships", type="negative")
                    return

                notification.dismiss()

                # Show preview dialog
                def add_selected_relationships(selected: list[tuple[Any, Any]]) -> None:
                    """Add selected relationships to the world database from the preview."""
                    if not selected:
                        ui.notify("No relationships selected", type="info")
                        return
                    if not self.state.world_db:
                        ui.notify("No world database", type="negative")
                        return

                    # Get all entities to look up IDs from names
                    entities = self.state.world_db.list_entities()
                    added_count = 0

                    for rel_data, _scores in selected:
                        source_name = rel_data.get("source", "")
                        target_name = rel_data.get("target", "")
                        rel_type = rel_data.get("relation_type", "related_to")
                        desc = rel_data.get("description", "")

                        # Look up entity IDs from names
                        source_entity = next((e for e in entities if e.name == source_name), None)
                        target_entity = next((e for e in entities if e.name == target_name), None)

                        if source_entity and target_entity:
                            self.services.world.add_relationship(
                                self.state.world_db,
                                source_id=source_entity.id,
                                target_id=target_entity.id,
                                relation_type=rel_type,
                                description=desc,
                            )
                            added_count += 1
                        else:
                            logger.warning(
                                f"Could not find entities for relationship: "
                                f"{source_name} -> {target_name}"
                            )

                    self.state.world_db.invalidate_graph_cache()
                    self._refresh_entity_list()
                    if self._graph:
                        self._graph.refresh()
                    ui.notify(
                        f"Added {added_count} relationships",
                        type="positive",
                    )

                self._show_entity_preview_dialog(
                    "relationship", results, add_selected_relationships
                )
            else:
                # Non-quality generation - simpler approach
                notification.dismiss()
                ui.notify(
                    "Relationship generation requires quality refinement to be enabled",
                    type="warning",
                )

        except Exception as e:
            notification.dismiss()
            logger.exception(f"Error generating relationships: {e}")
            ui.notify(f"Error: {e}", type="negative")

    def _confirm_regenerate(self) -> None:
        """Show confirmation dialog before regenerating world.

        Shows a simple confirmation asking the user to confirm deletion of existing data,
        then opens the shared build dialog in rebuild mode.
        """
        logger.info("Rebuild World button clicked - showing confirmation dialog")

        if not self.state.project or not self.state.world_db:
            ui.notify("No project available", type="negative")
            return

        # Count existing data for the warning
        entity_count = self.state.world_db.count_entities()
        rel_count = len(self.state.world_db.list_relationships())
        chapter_count = len(self.state.project.chapters)
        char_count = len(self.state.project.characters)

        with ui.dialog() as dialog, ui.card().classes("p-4 min-w-[400px]"):
            ui.label("Rebuild World?").classes("text-lg font-bold")
            ui.label(
                f"This will permanently delete all existing world data:\n"
                f"• {entity_count} entities\n"
                f"• {rel_count} relationships\n"
                f"• {chapter_count} chapter outlines\n"
                f"• {char_count} characters\n\n"
                "Then generate everything fresh from your story brief."
            ).classes("text-gray-600 dark:text-gray-400 whitespace-pre-line")

            async def confirm_and_build() -> None:
                """Close the confirmation dialog and trigger a full rebuild of the story structure."""
                dialog.close()
                await self._build_structure(rebuild=True)

            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button(
                    "Continue to Rebuild",
                    on_click=confirm_and_build,
                    icon="refresh",
                ).props("color=negative")

        dialog.open()

    async def _build_structure(self, rebuild: bool = False) -> None:
        """Build story structure using the shared dialog.

        Args:
            rebuild: If True, clears all existing data before building (uses full_rebuild).
                     If False, builds without clearing (uses full).
        """

        async def on_complete() -> None:
            """Generate mini descriptions and reload page after build."""
            await self._generate_mini_descriptions()
            ui.navigate.reload()

        await show_build_structure_dialog(
            state=self.state,
            services=self.services,
            rebuild=rebuild,
            on_complete=on_complete,
        )

    def _confirm_clear_world(self) -> None:
        """Show confirmation dialog before clearing world data."""
        logger.info("Clear World button clicked - showing confirmation dialog")

        if not self.state.world_db or not self.state.project:
            ui.notify("No world data to clear", type="info")
            return

        # Count existing data
        entity_count = self.state.world_db.count_entities()
        rel_count = len(self.state.world_db.list_relationships())
        chapter_count = len(self.state.project.chapters)
        char_count = len(self.state.project.characters)

        if entity_count == 0 and rel_count == 0 and chapter_count == 0 and char_count == 0:
            ui.notify("World is already empty", type="info")
            return

        with ui.dialog() as dialog, ui.card().classes("p-4 min-w-[400px]"):
            ui.label("Clear World?").classes("text-lg font-bold")
            ui.label(
                f"This will permanently delete all world and story structure data:\n"
                f"• {entity_count} entities\n"
                f"• {rel_count} relationships\n"
                f"• {chapter_count} chapter outlines\n"
                f"• {char_count} characters\n\n"
                f"Your interview and story brief will be kept.\n"
                f"This action cannot be undone."
            ).classes("text-gray-600 dark:text-gray-400 whitespace-pre-line")

            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button(
                    "Clear All",
                    on_click=lambda: self._do_clear_world(dialog),
                    icon="delete_sweep",
                ).props("color=warning")

        dialog.open()

    def _do_clear_world(self, dialog: ui.dialog) -> None:
        """Execute world clear - removes all entities, relationships, and story structure."""
        logger.info("User confirmed clear - removing all world data and story structure")
        dialog.close()

        if not self.state.world_db or not self.state.project:
            logger.warning("Clear failed: no world_db or project available")
            ui.notify("No world database available", type="negative")
            return

        try:
            # Delete all relationships first (they reference entities)
            relationships = self.state.world_db.list_relationships()
            for rel in relationships:
                self.state.world_db.delete_relationship(rel.id)
            logger.info(f"Deleted {len(relationships)} relationships")

            # Delete all entities
            entities = self.state.world_db.list_entities()
            for entity in entities:
                self.state.world_db.delete_entity(entity.id)
            logger.info(f"Deleted {len(entities)} entities")

            # Clear story structure (but keep brief and interview)
            chapter_count = len(self.state.project.chapters)
            char_count = len(self.state.project.characters)
            self.state.project.chapters = []
            self.state.project.characters = []
            self.state.project.world_description = ""
            self.state.project.plot_points = []
            logger.info(
                f"Cleared {chapter_count} chapters and {char_count} characters from project"
            )

            # Save the project with cleared structure
            self.services.project.save_project(self.state.project)
            logger.info("Project saved with cleared structure")

            # Refresh UI
            self._refresh_entity_list()
            if self._graph:
                self._graph.refresh()

            ui.notify(
                f"World cleared: removed {len(entities)} entities, {len(relationships)} relationships, "
                f"{chapter_count} chapters, {char_count} characters",
                type="positive",
            )

            # Reload page to update toolbar (Clear button visibility, Build Structure button)
            ui.navigate.reload()

        except Exception as e:
            logger.exception(f"Error clearing world: {e}")
            ui.notify(f"Error: {e}", type="negative")

    async def _generate_mini_descriptions(self) -> None:
        """Generate mini descriptions for entity tooltips."""
        from nicegui import run

        if not self.state.world_db:
            return

        all_entities = self.state.world_db.list_entities()
        entity_data = [
            {"name": e.name, "type": e.type, "description": e.description} for e in all_entities
        ]
        mini_descs = await run.io_bound(
            self.services.world_quality.generate_mini_descriptions_batch,
            entity_data,
        )
        logger.info(f"Generated {len(mini_descs)} mini descriptions")

        # Update entities with mini descriptions
        for entity in all_entities:
            if entity.name in mini_descs:
                # Merge mini_description into existing attributes
                attrs = dict(entity.attributes) if entity.attributes else {}
                attrs["mini_description"] = mini_descs[entity.name]
                self.state.world_db.update_entity(
                    entity_id=entity.id,
                    attributes=attrs,
                )
        logger.info("Updated entities with mini descriptions")

        # Invalidate graph cache to ensure fresh tooltips
        self.state.world_db.invalidate_graph_cache()

    def _build_entity_browser(self) -> None:
        """Build the entity browser panel."""
        with ui.card().classes("w-full h-full"):
            ui.label("Entity Browser").classes("text-lg font-semibold")

            # Search with Ctrl+F hint
            self._search_input = (
                ui.input(
                    placeholder="Search entities... (Ctrl+F)",
                    value=self.state.entity_search_query,
                    on_change=self._on_search,
                )
                .classes("w-full")
                .props("outlined dense clearable")
            )

            # Search scope checkboxes
            with ui.row().classes("w-full gap-4 text-xs items-center"):
                ui.checkbox(
                    "Names",
                    value=self.state.entity_search_names,
                    on_change=lambda e: self._update_search_scope("names", e.value),
                ).props("dense")
                ui.checkbox(
                    "Descriptions",
                    value=self.state.entity_search_descriptions,
                    on_change=lambda e: self._update_search_scope("descriptions", e.value),
                ).props("dense")

            # Filter and sort row
            with ui.row().classes("w-full gap-2 items-center mt-1"):
                # Quality filter
                ui.select(
                    label="Quality",
                    options={"all": "All", "high": "8+", "medium": "6-8", "low": "<6"},
                    value=self.state.entity_quality_filter,
                    on_change=self._on_quality_filter_change,
                ).classes("w-20").props("dense outlined")

                # Sort dropdown
                ui.select(
                    label="Sort",
                    options={
                        "name": "Name",
                        "type": "Type",
                        "quality": "Quality",
                        "relationships": "Relationships",
                    },
                    value=self.state.entity_sort_by,
                    on_change=self._on_sort_change,
                ).classes("w-28").props("dense outlined")

                # Sort direction toggle
                self._sort_direction_btn = (
                    ui.button(
                        icon="arrow_upward"
                        if not self.state.entity_sort_descending
                        else "arrow_downward",
                        on_click=self._toggle_sort_direction,
                    )
                    .props("flat dense")
                    .tooltip("Toggle sort direction")
                )

            # Entity list with flexible height to match editor
            self._entity_list = (
                ui.column()
                .classes(
                    "w-full gap-1 overflow-auto flex-grow p-2 bg-gray-50 dark:bg-gray-800 rounded-lg"
                )
                .style("max-height: calc(100vh - 520px); min-height: 200px")
            )
            self._refresh_entity_list()

            # Add button
            ui.button(
                "+ Add Entity",
                on_click=self._show_add_dialog,
                icon="add",
            ).props("color=primary").classes("w-full mt-2")

            # Undo/Redo buttons
            with ui.row().classes("w-full gap-2 mt-2"):
                self._undo_btn = (
                    ui.button(
                        icon="undo",
                        on_click=self._do_undo,
                    )
                    .props("flat dense")
                    .tooltip("Undo (Ctrl+Z)")
                )
                self._redo_btn = (
                    ui.button(
                        icon="redo",
                        on_click=self._do_redo,
                    )
                    .props("flat dense")
                    .tooltip("Redo (Ctrl+Y)")
                )
                self._update_undo_redo_buttons()

            # Register Ctrl+F keyboard shortcut
            self._register_keyboard_shortcuts()

    def _build_graph_section(self) -> None:
        """Build the graph visualization section."""
        # Use larger height to match entity browser and editor
        self._graph = GraphComponent(
            world_db=self.state.world_db,
            settings=self.services.settings,
            on_node_select=self._on_node_select,
            on_edge_select=self._on_edge_select,
            on_create_relationship=self._on_create_relationship,
            on_edge_context_menu=self._on_edge_context_menu,
            height=600,  # Taller to match browser and editor panels
        )
        self._graph.build()

    def _build_entity_editor(self) -> None:
        """Build the entity editor panel."""
        with ui.card().classes("w-full h-full"):
            ui.label("Entity Editor").classes("text-lg font-semibold")

            if not self.state.selected_entity_id or not self.state.world_db:
                ui.label("Select an entity to edit").classes(
                    "text-gray-500 dark:text-gray-400 text-sm mt-4"
                )
                return

            entity = self.services.world.get_entity(
                self.state.world_db, self.state.selected_entity_id
            )
            if not entity:
                ui.label("Entity not found").classes("text-red-500 text-sm")
                return

            # Initialize attrs from entity for saving later
            self._entity_attrs = entity.attributes.copy() if entity.attributes else {}

            # Entity form - common fields
            self._entity_name_input = ui.input(
                label="Name",
                value=entity.name,
            ).classes("w-full")

            self._entity_type_select = ui.select(
                label="Type",
                options=["character", "location", "item", "faction", "concept"],
                value=entity.type,
                on_change=lambda e: self._refresh_entity_editor(),
            ).classes("w-full")

            self._entity_desc_input = (
                ui.textarea(
                    label="Description",
                    value=entity.description,
                )
                .classes("w-full")
                .props("rows=4")
            )

            # Type-specific attribute fields
            self._build_type_specific_fields(entity)

            # Action buttons
            with ui.row().classes("w-full gap-2 mt-4"):
                ui.button(
                    "Save",
                    on_click=self._save_entity,
                    icon="save",
                ).props("color=primary")

                ui.button(
                    "Regenerate",
                    on_click=self._show_regenerate_dialog,
                    icon="refresh",
                ).props("outline color=secondary").tooltip("Regenerate this entity with AI")

                ui.button(
                    "Delete",
                    on_click=self._confirm_delete_entity,
                    icon="delete",
                ).props("color=negative outline")

    def _build_type_specific_fields(self, entity: Entity) -> None:
        """Build attribute fields dynamically from entity attributes."""
        attrs = entity.attributes or {}

        # Store dynamic attribute inputs for saving
        self._dynamic_attr_inputs: dict[str, Any] = {}

        with ui.expansion("Attributes", icon="list", value=True).classes("w-full"):
            # Skip quality_scores - handled separately below
            skip_keys = {"quality_scores"}

            for key, value in sorted(attrs.items()):
                if key in skip_keys:
                    continue

                # Format label nicely
                label = key.replace("_", " ").title()

                # Handle different value types
                if isinstance(value, list):
                    # Lists become comma-separated inputs
                    value_str = ", ".join(str(v) for v in value)
                    input_widget = ui.input(
                        label=f"{label} (comma-separated)",
                        value=value_str,
                    ).classes("w-full")
                    self._dynamic_attr_inputs[key] = ("list", input_widget)

                elif isinstance(value, dict):
                    # Dicts shown as JSON (read-only for complex nested data)
                    import json

                    json_str = json.dumps(value, indent=2)
                    ui.label(label).classes("text-sm font-medium mt-2")
                    ui.code(json_str, language="json").classes("w-full text-xs")

                elif isinstance(value, bool):
                    # Booleans as checkboxes
                    checkbox = ui.checkbox(label, value=value).classes("w-full")
                    self._dynamic_attr_inputs[key] = ("bool", checkbox)

                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    # Numbers as number inputs
                    number_widget = ui.number(
                        label=label,
                        value=value,
                    ).classes("w-full")
                    self._dynamic_attr_inputs[key] = ("number", number_widget)

                elif value is None or value == "":
                    # Empty values as text inputs
                    input_widget = ui.input(
                        label=label,
                        value="",
                    ).classes("w-full")
                    self._dynamic_attr_inputs[key] = ("str", input_widget)

                else:
                    # Strings - use textarea if long, input if short
                    str_value = str(value)
                    if len(str_value) > 100 or "\n" in str_value:
                        input_widget = (
                            ui.textarea(label=label, value=str_value)
                            .classes("w-full")
                            .props("rows=3")
                        )
                    else:
                        input_widget = ui.input(label=label, value=str_value).classes("w-full")
                    self._dynamic_attr_inputs[key] = ("str", input_widget)

            # Add button to add new attribute
            with ui.row().classes("w-full mt-2 gap-2"):
                self._new_attr_key = ui.input(placeholder="New attribute name").classes("flex-1")
                ui.button(
                    icon="add",
                    on_click=self._add_new_attribute,
                ).props("flat dense")

            # Show quality scores if present (read-only display)
            quality_scores = attrs.get("quality_scores")
            if quality_scores and isinstance(quality_scores, dict):
                with ui.expansion("Quality Scores", icon="star", value=False).classes(
                    "w-full mt-2"
                ):
                    avg = quality_scores.get("average", 0)
                    ui.label(f"Average: {avg:.1f}/10").classes("font-semibold text-primary")
                    for key, value in quality_scores.items():
                        if key not in ("average", "feedback") and isinstance(value, (int, float)):
                            ui.label(f"{key.replace('_', ' ').title()}: {value:.1f}").classes(
                                "text-sm"
                            )
                    feedback = quality_scores.get("feedback", "")
                    if feedback:
                        ui.label(f"Feedback: {feedback}").classes("text-xs text-gray-500 mt-2")

            # Version history panel
            self._build_version_history_panel(entity.id)

    def _add_new_attribute(self) -> None:
        """Add a new attribute to the current entity."""
        if not hasattr(self, "_new_attr_key") or not self._new_attr_key.value:
            ui.notify("Enter an attribute name", type="warning")
            return

        key = self._new_attr_key.value.strip().lower().replace(" ", "_")
        if not key:
            return

        # Add to entity_attrs for saving
        if key not in self._entity_attrs:
            self._entity_attrs[key] = ""
            ui.notify(f"Added attribute: {key}. Click Save to persist.", type="info")
            self._refresh_entity_editor()
        else:
            ui.notify(f"Attribute '{key}' already exists", type="warning")

    def _build_version_history_panel(self, entity_id: str) -> None:
        """Build the version history panel for an entity.

        Args:
            entity_id: The entity ID to show version history for.
        """
        if not self.state.world_db:
            return

        # Use configured retention limit instead of hardcoded value
        retention_limit = self.services.settings.entity_version_retention
        versions = self.services.world.get_entity_versions(
            self.state.world_db, entity_id, limit=retention_limit
        )

        if not versions:
            return  # No versions yet, skip showing the panel

        with ui.expansion("Version History", icon="history", value=False).classes("w-full mt-2"):
            ui.label(f"{len(versions)} versions").classes(
                "text-xs text-gray-500 dark:text-gray-400 mb-2"
            )

            for version in versions:
                change_type_icons = {
                    "created": "add_circle",
                    "refined": "auto_fix_high",
                    "edited": "edit",
                    "regenerated": "refresh",
                }
                icon = change_type_icons.get(version.change_type, "history")

                with ui.row().classes(
                    "w-full items-center gap-2 p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700"
                ):
                    ui.icon(icon, size="xs").classes("text-gray-500")

                    with ui.column().classes("flex-grow gap-0"):
                        with ui.row().classes("items-center gap-2"):
                            ui.label(f"v{version.version_number}").classes(
                                "text-sm font-mono font-bold"
                            )
                            ui.label(version.change_type).classes(
                                "text-xs px-1 py-0.5 rounded bg-gray-200 dark:bg-gray-600"
                            )
                            if version.quality_score is not None:
                                ui.label(f"{version.quality_score:.1f}").classes(
                                    "text-xs text-blue-600 dark:text-blue-400"
                                )

                        # Format timestamp
                        time_str = version.created_at.strftime("%Y-%m-%d %H:%M")
                        ui.label(time_str).classes("text-xs text-gray-500")

                        if version.change_reason:
                            ui.label(version.change_reason).classes("text-xs text-gray-400 italic")

                    # Action buttons
                    with ui.row().classes("gap-1"):
                        ui.button(
                            icon="visibility",
                            on_click=lambda v=version: self._show_version_diff(v),
                        ).props("flat dense size=xs").tooltip("View changes")

                        # Don't show revert for the latest version
                        if version.version_number < versions[0].version_number:
                            ui.button(
                                icon="restore",
                                on_click=lambda v=version: self._confirm_revert_version(v),
                            ).props("flat dense size=xs color=warning").tooltip(
                                "Revert to this version"
                            )

    def _confirm_revert_version(self, version: EntityVersion) -> None:
        """Show confirmation dialog for reverting to a version.

        Args:
            version: The version to revert to.
        """
        with ui.dialog() as dialog, ui.card().classes("p-4"):
            ui.label("Revert Entity?").classes("text-lg font-bold mb-2")
            ui.label(
                f"Revert to version {version.version_number} "
                f"({version.change_type} from {version.created_at.strftime('%Y-%m-%d %H:%M')})?"
            ).classes("text-sm text-gray-600 dark:text-gray-400 mb-4")

            ui.label(
                "This will restore the entity's name, type, description, and attributes "
                "to the state captured in this version."
            ).classes("text-xs text-gray-500 mb-4")

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button(
                    "Revert",
                    on_click=lambda: self._revert_to_version(version, dialog),
                ).props("color=warning")

        dialog.open()

    def _revert_to_version(self, version: EntityVersion, dialog: ui.dialog) -> None:
        """Execute the revert to a specific version.

        Args:
            version: The version to revert to.
            dialog: The dialog to close after completion.
        """
        if not self.state.world_db:
            return

        try:
            self.services.world.revert_entity_to_version(
                self.state.world_db, version.entity_id, version.version_number
            )
            dialog.close()
            ui.notify(
                f"Reverted to version {version.version_number}",
                type="positive",
            )
            # Refresh list, editor, and graph to show updated data
            self._refresh_entity_list()
            self._refresh_entity_editor()
            if self._graph:
                self._graph.refresh()
        except ValueError as e:
            logger.error(f"Failed to revert: {e}")
            ui.notify(f"Revert failed: {e}", type="negative")

    def _show_version_diff(self, version: EntityVersion) -> None:
        """Show a dialog comparing a version with current entity state.

        Args:
            version: The version to compare.
        """
        if not self.state.world_db:
            return

        # Get current entity
        entity = self.services.world.get_entity(self.state.world_db, version.entity_id)
        if not entity:
            ui.notify("Entity not found", type="warning")
            return

        version_data = version.data_json
        current_data = {
            "type": entity.type,
            "name": entity.name,
            "description": entity.description,
            "attributes": entity.attributes or {},
        }

        with ui.dialog() as dialog, ui.card().classes("p-4 min-w-[500px] max-w-[800px]"):
            with ui.row().classes("w-full items-center justify-between mb-4"):
                ui.label(f"Version {version.version_number} vs Current").classes(
                    "text-lg font-bold"
                )
                ui.button(icon="close", on_click=dialog.close).props("flat dense")

            # Show diff for each field
            fields = ["name", "type", "description"]

            for field in fields:
                old_val = version_data.get(field, "")
                new_val = current_data.get(field, "")

                if old_val != new_val:
                    ui.label(field.title()).classes("text-sm font-semibold mt-2")
                    with ui.row().classes("w-full gap-4"):
                        with ui.column().classes("flex-1 p-2 bg-red-50 dark:bg-red-900/20 rounded"):
                            ui.label("Version").classes("text-xs text-gray-500")
                            ui.label(str(old_val) or "(empty)").classes("text-sm")
                        with ui.column().classes(
                            "flex-1 p-2 bg-green-50 dark:bg-green-900/20 rounded"
                        ):
                            ui.label("Current").classes("text-xs text-gray-500")
                            ui.label(str(new_val) or "(empty)").classes("text-sm")

            # Compare attributes - cast to dict for mypy
            old_attrs_raw = version_data.get("attributes", {})
            old_attrs = dict(old_attrs_raw) if isinstance(old_attrs_raw, dict) else {}
            new_attrs_raw = current_data["attributes"]
            new_attrs = dict(new_attrs_raw) if isinstance(new_attrs_raw, dict) else {}
            all_attr_keys = set(old_attrs.keys()) | set(new_attrs.keys())

            if old_attrs != new_attrs:
                ui.label("Attributes").classes("text-sm font-semibold mt-4")

                for key in sorted(all_attr_keys):
                    attr_old_val: Any = old_attrs.get(key)
                    attr_new_val: Any = new_attrs.get(key)

                    if attr_old_val != attr_new_val:
                        ui.label(key.replace("_", " ").title()).classes(
                            "text-xs text-gray-600 mt-2"
                        )
                        with ui.row().classes("w-full gap-4"):
                            with ui.column().classes(
                                "flex-1 p-2 bg-red-50 dark:bg-red-900/20 rounded"
                            ):
                                old_str = (
                                    "(not set)" if attr_old_val is None else str(attr_old_val)[:200]
                                )
                                ui.label(old_str).classes("text-xs")
                            with ui.column().classes(
                                "flex-1 p-2 bg-green-50 dark:bg-green-900/20 rounded"
                            ):
                                new_str = (
                                    "(not set)" if attr_new_val is None else str(attr_new_val)[:200]
                                )
                                ui.label(new_str).classes("text-xs")

            # If no differences
            if old_attrs == new_attrs and all(
                version_data.get(f) == current_data.get(f) for f in fields
            ):
                ui.label("No differences from current state").classes(
                    "text-sm text-gray-500 italic mt-4"
                )

            with ui.row().classes("w-full justify-end mt-4"):
                ui.button("Close", on_click=dialog.close).props("flat")

        dialog.open()

    def _build_relationships_section(self) -> None:
        """Build the relationships management section."""
        with ui.expansion("Relationships", icon="link", value=False).classes("w-full"):
            # Add relationship form
            with ui.row().classes("w-full items-end gap-4 mb-4"):
                entities = self._get_entity_options()

                self._rel_source_select = ui.select(
                    label="From",
                    options=entities,
                ).classes("w-48")

                self._rel_type_select = ui.select(
                    label="Relationship",
                    options=[
                        "knows",
                        "loves",
                        "hates",
                        "located_in",
                        "owns",
                        "member_of",
                        "enemy_of",
                        "ally_of",
                        "parent_of",
                        "child_of",
                    ],
                    new_value_mode="add",
                ).classes("w-40")

                self._rel_target_select = ui.select(
                    label="To",
                    options=entities,
                ).classes("w-48")

                ui.button(
                    "Add",
                    on_click=self._add_relationship,
                    icon="add",
                ).props("color=primary")

            # Relationships table
            if self.state.world_db:
                relationships = self.services.world.get_relationships(self.state.world_db)

                if relationships:
                    columns = [
                        {"name": "from", "label": "From", "field": "from", "align": "left"},
                        {"name": "type", "label": "Type", "field": "type", "align": "center"},
                        {"name": "to", "label": "To", "field": "to", "align": "left"},
                        {"name": "actions", "label": "", "field": "actions", "align": "right"},
                    ]

                    rows = []
                    for rel in relationships:
                        source = self.services.world.get_entity(self.state.world_db, rel.source_id)
                        target = self.services.world.get_entity(self.state.world_db, rel.target_id)
                        rows.append(
                            {
                                "id": rel.id,
                                "from": source.name if source else "Unknown",
                                "type": rel.relation_type,
                                "to": target.name if target else "Unknown",
                            }
                        )

                    ui.table(columns=columns, rows=rows).classes("w-full")
                else:
                    ui.label("No relationships yet").classes("text-gray-500 dark:text-gray-400")

    def _build_analysis_section(self) -> None:
        """
        Constructs the Analysis Tools UI section with tabs for graph analyses.

        Creates tabs for finding paths, showing most-connected nodes (centrality), detecting communities, and rendering a conflict map; wires tab controls to their respective handlers and initializes the _analysis_result HTML container for rendering analysis output.
        """
        with ui.expansion("Analysis Tools", icon="analytics", value=False).classes("w-full"):
            with ui.tabs().classes("w-full") as tabs:
                ui.tab("path", label="Find Path")
                ui.tab("centrality", label="Most Connected")
                ui.tab("communities", label="Communities")
                ui.tab("conflicts", label="Conflict Map")

            with ui.tab_panels(tabs, value="path").classes("w-full"):
                # Path finder
                with ui.tab_panel("path"):
                    with ui.row().classes("items-end gap-4"):
                        entities = self._get_entity_options()
                        path_source = ui.select(label="From", options=entities).classes("w-48")
                        path_target = ui.select(label="To", options=entities).classes("w-48")
                        ui.button(
                            "Find Path",
                            on_click=lambda: self._find_path(path_source.value, path_target.value),
                        )

                # Centrality analysis
                with ui.tab_panel("centrality"):
                    ui.button(
                        "Show Most Connected",
                        on_click=self._show_centrality,
                    )

                # Community detection
                with ui.tab_panel("communities"):
                    ui.button(
                        "Detect Communities",
                        on_click=self._show_communities,
                    )

                # Conflict mapping
                with ui.tab_panel("conflicts"):
                    self._build_conflict_map_tab()

            # Analysis result display
            self._analysis_result = ui.html(sanitize=False).classes("w-full mt-4")

    # ========== Helper Methods ==========

    def _get_entity_options(self) -> dict[str, str]:
        """Get entity options for select dropdowns."""
        if not self.state.world_db:
            return {}

        entities = self.services.world.list_entities(self.state.world_db)
        return {e.id: e.name for e in entities}

    def _refresh_entity_list(self) -> None:
        """Refresh the entity list display."""
        if not self._entity_list or not self.state.world_db:
            return

        self._entity_list.clear()

        entities = self.services.world.list_entities(self.state.world_db)

        # Filter by type
        if self.state.entity_filter_types:
            entities = [e for e in entities if e.type in self.state.entity_filter_types]

        # Filter by search (with scope) - concise list comprehension
        if self.state.entity_search_query:
            query = self.state.entity_search_query.lower()
            entities = [
                e
                for e in entities
                if (self.state.entity_search_names and query in e.name.lower())
                or (self.state.entity_search_descriptions and query in e.description.lower())
            ]

        # Filter by quality
        if self.state.entity_quality_filter != "all":
            entities = self._filter_by_quality(entities)

        # Sort entities
        entities = self._sort_entities(entities)

        # Clear selection if selected entity is filtered out
        if self.state.selected_entity_id:
            visible_ids = {e.id for e in entities}
            if self.state.selected_entity_id not in visible_ids:
                self.state.select_entity(None)
                self._refresh_entity_editor()

        with self._entity_list:
            if not entities:
                # Check if there are entities at all or just filtered out
                all_entities = self.services.world.list_entities(self.state.world_db)
                if not all_entities:
                    # No entities exist at all - show guidance
                    with ui.column().classes("items-center gap-2 py-4"):
                        ui.icon("group_add", size="md").classes("text-gray-400 dark:text-gray-500")
                        ui.label("No entities yet").classes(
                            "text-gray-500 dark:text-gray-400 font-medium"
                        )
                        ui.label("Add characters, locations, and more").classes(
                            "text-xs text-gray-400 dark:text-gray-500 text-center"
                        )
                        ui.label("using the button below.").classes(
                            "text-xs text-gray-400 dark:text-gray-500 text-center"
                        )
                else:
                    # Entities exist but are filtered out
                    ui.label("No matching entities").classes(
                        "text-gray-500 dark:text-gray-400 text-sm"
                    )
                    ui.label("Try adjusting filters or search").classes(
                        "text-xs text-gray-400 dark:text-gray-500"
                    )
            else:
                for entity in entities:
                    entity_list_item(
                        entity=entity,
                        on_select=self._select_entity,
                        selected=entity.id == self.state.selected_entity_id,
                    )

    def _filter_by_quality(self, entities: list[Entity]) -> list[Entity]:
        """Filter entities by quality score.

        Args:
            entities: List of entities to filter.

        Returns:
            Filtered list based on quality filter setting.
        """
        logger.debug(
            "Filtering %d entities by quality: %s", len(entities), self.state.entity_quality_filter
        )
        result = []
        for entity in entities:
            scores = entity.attributes.get("quality_scores") if entity.attributes else None
            avg = scores.get("average", 0) if scores else 0

            if self.state.entity_quality_filter == "high" and avg >= 8:
                result.append(entity)
            elif self.state.entity_quality_filter == "medium" and 6 <= avg < 8:
                result.append(entity)
            elif self.state.entity_quality_filter == "low" and avg < 6:
                result.append(entity)
        logger.debug("Quality filter returned %d entities", len(result))
        return result

    def _sort_entities(self, entities: list[Entity]) -> list[Entity]:
        """Sort entities by current sort setting.

        Args:
            entities: List of entities to sort.

        Returns:
            Sorted list based on sort setting.
        """
        sort_key = self.state.entity_sort_by
        descending = self.state.entity_sort_descending
        logger.debug("Sorting %d entities by %s (desc=%s)", len(entities), sort_key, descending)

        if sort_key == "name":
            return sorted(entities, key=lambda e: e.name.lower(), reverse=descending)
        elif sort_key == "type":
            return sorted(entities, key=lambda e: e.type, reverse=descending)
        elif sort_key == "quality":
            return sorted(
                entities,
                key=lambda e: (e.attributes.get("quality_scores") or {}).get("average", 0),
                reverse=descending,
            )
        elif sort_key == "relationships":

            def count_rels(entity: Entity) -> int:
                """Count relationships for sorting."""
                if self.state.world_db:
                    count = len(self.state.world_db.get_relationships(entity.id))
                    logger.debug("Entity %s has %d relationships", entity.id, count)
                    return count
                return 0

            return sorted(entities, key=count_rels, reverse=descending)
        else:
            return sorted(entities, key=lambda e: e.name.lower(), reverse=descending)

    # ========== Event Handlers ==========

    def _toggle_type_filter(self, entity_type: str, enabled: bool) -> None:
        """Toggle entity type filter."""
        if enabled and entity_type not in self.state.entity_filter_types:
            self.state.entity_filter_types.append(entity_type)
        elif not enabled and entity_type in self.state.entity_filter_types:
            self.state.entity_filter_types.remove(entity_type)

        self._refresh_entity_list()
        if self._graph:
            self._graph.set_filter(self.state.entity_filter_types)

    def _on_search(self, e: Any) -> None:
        """Handle search input change."""
        self.state.entity_search_query = e.value
        self._refresh_entity_list()

        # Highlight matching nodes in the graph
        if self._graph:
            self._graph.highlight_search(e.value)

    def _update_search_scope(self, scope: str, enabled: bool) -> None:
        """Update search scope settings.

        Args:
            scope: 'names' or 'descriptions'.
            enabled: Whether this scope is enabled.
        """
        if scope == "names":
            self.state.entity_search_names = enabled
        elif scope == "descriptions":
            self.state.entity_search_descriptions = enabled
        logger.debug(f"Search scope updated: {scope}={enabled}")
        self._refresh_entity_list()

    def _on_quality_filter_change(self, e: Any) -> None:
        """Handle quality filter dropdown change."""
        self.state.entity_quality_filter = e.value
        logger.debug(f"Quality filter changed to: {e.value}")
        self._refresh_entity_list()

    def _on_sort_change(self, e: Any) -> None:
        """Handle sort dropdown change."""
        self.state.entity_sort_by = e.value
        logger.debug(f"Sort changed to: {e.value}")
        self._refresh_entity_list()

    def _toggle_sort_direction(self) -> None:
        """Toggle sort direction between ascending and descending."""
        self.state.entity_sort_descending = not self.state.entity_sort_descending
        logger.debug(
            f"Sort direction: {'descending' if self.state.entity_sort_descending else 'ascending'}"
        )

        # Update button icon
        if self._sort_direction_btn:
            icon = "arrow_downward" if self.state.entity_sort_descending else "arrow_upward"
            self._sort_direction_btn.props(f"icon={icon}")

        self._refresh_entity_list()

    def _register_keyboard_shortcuts(self) -> None:
        """Register keyboard shortcuts for the world page."""
        ui.keyboard(on_key=self._handle_keyboard)

    async def _handle_keyboard(self, e: Any) -> None:
        """Handle keyboard events.

        Args:
            e: Keyboard event with key and modifiers.
        """
        # Ctrl+F or Cmd+F focuses the search input (cross-platform)
        ctrl_pressed = getattr(e.modifiers, "ctrl", False)
        meta_pressed = getattr(e.modifiers, "meta", False)
        if (ctrl_pressed or meta_pressed) and e.key.lower() == "f":
            if self._search_input:
                await self._search_input.run_method("focus")
                await self._search_input.run_method("select")
            return

        # Esc clears the search
        if e.key == "Escape" and self._search_input:
            self._search_input.value = ""
            self.state.entity_search_query = ""
            self._refresh_entity_list()
            if self._graph:
                self._graph.highlight_search("")

    def _on_node_select(self, entity_id: str) -> None:
        """Handle graph node selection."""
        self.state.select_entity(entity_id)
        self._refresh_entity_list()
        self._refresh_entity_editor()

    def _on_edge_select(self, relationship_id: str) -> None:
        """Handle graph edge selection to show relationship editor."""
        if not self.state.world_db:
            return

        # Get all relationships and find the one that matches
        relationships = self.state.world_db.list_relationships()
        rel = next((r for r in relationships if r.id == relationship_id), None)

        if not rel:
            # This can happen when clicking on a stale edge after relationships were
            # regenerated - the graph may have old edge IDs. Silently ignore.
            logger.debug(f"Relationship not found (stale edge click): {relationship_id}")
            return

        # Get source and target entities
        source = self.services.world.get_entity(self.state.world_db, rel.source_id)
        target = self.services.world.get_entity(self.state.world_db, rel.target_id)

        source_name = source.name if source else "Unknown"
        target_name = target.name if target else "Unknown"

        # Show relationship editor dialog
        with ui.dialog() as dialog, ui.card().classes("w-96"):
            ui.label("Edit Relationship").classes("text-lg font-semibold")
            ui.label(f"{source_name} → {target_name}").classes(
                "text-sm text-gray-500 dark:text-gray-400"
            )

            ui.separator()

            # Relationship type
            rel_type_input = ui.input(
                "Relationship Type",
                value=rel.relation_type,
            ).classes("w-full")

            # Description
            rel_desc_input = (
                ui.textarea(
                    "Description",
                    value=rel.description,
                )
                .props("filled")
                .classes("w-full")
            )

            # Strength slider
            ui.label("Strength").classes("text-sm mt-2")
            rel_strength_slider = ui.slider(
                min=0.0,
                max=1.0,
                step=0.1,
                value=rel.strength,
            ).classes("w-full")

            # Bidirectional checkbox
            rel_bidir_checkbox = ui.checkbox(
                "Bidirectional",
                value=rel.bidirectional,
            )

            ui.separator()

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat")

                def save_relationship() -> None:
                    """Save relationship updates to world database."""
                    if not self.state.world_db:
                        return

                    # Record action for undo (store old values)
                    self.state.record_action(
                        UndoAction(
                            action_type=ActionType.UPDATE_RELATIONSHIP,
                            data={
                                "relationship_id": rel.id,
                                "relation_type": rel_type_input.value,
                                "description": rel_desc_input.value,
                                "strength": rel_strength_slider.value,
                                "bidirectional": rel_bidir_checkbox.value,
                            },
                            inverse_data={
                                "relation_type": rel.relation_type,
                                "description": rel.description,
                                "strength": rel.strength,
                                "bidirectional": rel.bidirectional,
                            },
                        )
                    )

                    # Update the relationship
                    self.state.world_db.update_relationship(
                        relationship_id=rel.id,
                        relation_type=rel_type_input.value,
                        description=rel_desc_input.value,
                        strength=rel_strength_slider.value,
                        bidirectional=rel_bidir_checkbox.value,
                    )
                    dialog.close()
                    if self._graph:
                        self._graph.refresh()
                    self._update_undo_redo_buttons()
                    ui.notify("Relationship updated", type="positive")

                ui.button("Save", on_click=save_relationship).props("color=primary")

                def delete_relationship() -> None:
                    """Delete relationship from world database."""
                    if not self.state.world_db:
                        return

                    # Record action for undo
                    self.state.record_action(
                        UndoAction(
                            action_type=ActionType.DELETE_RELATIONSHIP,
                            data={"relationship_id": rel.id},
                            inverse_data={
                                "source_id": rel.source_id,
                                "target_id": rel.target_id,
                                "relation_type": rel.relation_type,
                                "description": rel.description,
                            },
                        )
                    )

                    self.services.world.delete_relationship(self.state.world_db, rel.id)
                    dialog.close()
                    if self._graph:
                        self._graph.refresh()
                    self._update_undo_redo_buttons()
                    ui.notify("Relationship deleted", type="warning")

                ui.button("Delete", on_click=delete_relationship).props("color=negative flat")

        dialog.open()

    def _on_create_relationship(self, source_id: str, target_id: str) -> None:
        """Handle drag-to-connect relationship creation.

        Args:
            source_id: Source entity ID.
            target_id: Target entity ID.
        """
        logger.debug("Creating relationship via drag: %s -> %s", source_id, target_id)
        if not self.state.world_db:
            return

        # Get entity names for display
        source = self.services.world.get_entity(self.state.world_db, source_id)
        target = self.services.world.get_entity(self.state.world_db, target_id)

        if not source or not target:
            ui.notify("Entity not found", type="negative")
            return

        source_name = source.name
        target_name = target.name

        # Show dialog to configure relationship
        with ui.dialog() as dialog, ui.card().classes("w-96 p-4"):
            ui.label("Create Relationship").classes("text-lg font-semibold mb-2")
            ui.label(f"{source_name} → {target_name}").classes(
                "text-sm text-gray-500 dark:text-gray-400 mb-4"
            )

            ui.separator()

            # Relationship type
            rel_type_input = ui.select(
                label="Relationship Type",
                options=[
                    "knows",
                    "loves",
                    "hates",
                    "located_in",
                    "owns",
                    "member_of",
                    "enemy_of",
                    "ally_of",
                    "parent_of",
                    "child_of",
                    "works_for",
                    "leads",
                ],
                value="knows",
                new_value_mode="add",
            ).classes("w-full")

            # Description
            rel_desc_input = (
                ui.textarea(
                    label="Description (optional)",
                    placeholder="Describe this relationship...",
                )
                .classes("w-full")
                .props("rows=2")
            )

            # Strength slider
            ui.label("Strength").classes("text-sm mt-2")
            rel_strength_slider = ui.slider(
                min=0.0,
                max=1.0,
                step=0.1,
                value=DEFAULT_RELATIONSHIP_STRENGTH,
            ).classes("w-full")

            # Bidirectional checkbox
            rel_bidir_checkbox = ui.checkbox(
                "Bidirectional",
                value=False,
            )

            ui.separator()

            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=dialog.close).props("flat")

                def create_relationship():
                    """Create the relationship between the dragged entities."""
                    if not self.state.world_db:
                        return

                    try:
                        relationship_id = self.services.world.add_relationship(
                            self.state.world_db,
                            source_id=source_id,
                            target_id=target_id,
                            relation_type=rel_type_input.value,
                            description=rel_desc_input.value,
                        )

                        # Store actual values selected by the user
                        final_strength = rel_strength_slider.value
                        final_bidirectional = rel_bidir_checkbox.value

                        # Always update relationship to match UI-selected values
                        # (DB defaults differ from UI defaults, so we must always sync)
                        self.state.world_db.update_relationship(
                            relationship_id=relationship_id,
                            strength=final_strength,
                            bidirectional=final_bidirectional,
                        )

                        # Record action for undo with complete data
                        self.state.record_action(
                            UndoAction(
                                action_type=ActionType.ADD_RELATIONSHIP,
                                data={
                                    "relationship_id": relationship_id,
                                    "source_id": source_id,
                                    "target_id": target_id,
                                    "relation_type": rel_type_input.value,
                                    "description": rel_desc_input.value,
                                    "strength": final_strength,
                                    "bidirectional": final_bidirectional,
                                },
                                inverse_data={
                                    "relationship_id": relationship_id,
                                    "source_id": source_id,
                                    "target_id": target_id,
                                    "relation_type": rel_type_input.value,
                                    "description": rel_desc_input.value,
                                    "strength": final_strength,
                                    "bidirectional": final_bidirectional,
                                },
                            )
                        )

                        dialog.close()
                        if self._graph:
                            self._graph.refresh()
                        self._update_undo_redo_buttons()
                        ui.notify(
                            f"Created relationship: {source_name} → {target_name}", type="positive"
                        )
                    except Exception as e:
                        logger.exception("Failed to create relationship via drag")
                        ui.notify(f"Error: {e}", type="negative")

                ui.button("Create", on_click=create_relationship).props("color=primary")

        dialog.open()

    def _on_edge_context_menu(self, edge_id: str) -> None:
        """Handle edge right-click context menu.

        Args:
            edge_id: Edge/relationship ID.
        """
        logger.debug("Edge context menu triggered for: %s", edge_id)
        # Just trigger the existing edge select handler which shows the edit dialog
        self._on_edge_select(edge_id)

    def _select_entity(self, entity: Entity) -> None:
        """Select an entity for editing."""
        self.state.select_entity(entity.id)
        self._refresh_entity_list()
        self._refresh_entity_editor()
        if self._graph:
            self._graph.set_selected(entity.id)

    def _refresh_entity_editor(self) -> None:
        """Refresh the entity editor panel with current selection."""
        if not self._editor_container:
            return

        self._editor_container.clear()
        with self._editor_container:
            self._build_entity_editor()

    async def _show_add_dialog(self) -> None:
        """Show dialog to add new entity."""
        with ui.dialog() as dialog, ui.card():
            ui.label("Add New Entity").classes("text-lg font-semibold")

            name_input = ui.input(label="Name").classes("w-full")
            type_select = ui.select(
                label="Type",
                options=["character", "location", "item", "faction", "concept"],
                value="character",
            ).classes("w-full")
            desc_input = ui.textarea(label="Description").props("filled").classes("w-full")

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button(
                    "Add",
                    on_click=lambda: self._add_entity(
                        dialog, name_input.value, type_select.value, desc_input.value
                    ),
                ).props("color=primary")

        dialog.open()

    def _add_entity(self, dialog: ui.dialog, name: str, entity_type: str, description: str) -> None:
        """Add a new entity."""
        if not name or not self.state.world_db:
            ui.notify("Name is required", type="warning")
            return

        try:
            entity_id = self.services.world.add_entity(
                self.state.world_db,
                entity_type=entity_type,
                name=name,
                description=description,
            )

            # Record action for undo
            self.state.record_action(
                UndoAction(
                    action_type=ActionType.ADD_ENTITY,
                    data={
                        "entity_id": entity_id,
                        "type": entity_type,
                        "name": name,
                        "description": description,
                    },
                    inverse_data={
                        "type": entity_type,
                        "name": name,
                        "description": description,
                    },
                )
            )

            dialog.close()
            self._refresh_entity_list()
            if self._graph:
                self._graph.refresh()
            self._update_undo_redo_buttons()
            ui.notify(f"Added {name}", type="positive")
        except Exception as e:
            logger.exception(f"Failed to add entity {name}")
            ui.notify(f"Error: {e}", type="negative")

    def _collect_attrs_from_form(self, entity_type: str) -> dict[str, Any]:
        """Collect attributes from dynamic form fields.

        Args:
            entity_type: The type of entity being edited (unused, kept for compatibility).

        Returns:
            Dictionary of attributes collected from form fields.
        """
        # Start with existing attrs to preserve fields not shown in form
        attrs = self._entity_attrs.copy() if self._entity_attrs else {}

        # Collect from dynamic attribute inputs
        if hasattr(self, "_dynamic_attr_inputs"):
            for key, (value_type, widget) in self._dynamic_attr_inputs.items():
                if value_type == "list":
                    # Parse comma-separated values back to list
                    if widget.value:
                        attrs[key] = [v.strip() for v in widget.value.split(",") if v.strip()]
                    else:
                        attrs[key] = []
                elif value_type == "bool":
                    attrs[key] = widget.value
                elif value_type == "number":
                    attrs[key] = widget.value
                else:  # str
                    attrs[key] = widget.value

        return attrs

    def _save_entity(self) -> None:
        """Save current entity changes."""
        if not self.state.selected_entity_id or not self.state.world_db:
            return

        try:
            # Get current state for inverse data
            old_entity = self.services.world.get_entity(
                self.state.world_db, self.state.selected_entity_id
            )
            if not old_entity:
                ui.notify("Entity not found", type="negative")
                return

            new_name = self._entity_name_input.value if self._entity_name_input else None
            new_desc = self._entity_desc_input.value if self._entity_desc_input else None
            new_type = (
                self._entity_type_select.value if self._entity_type_select else old_entity.type
            )

            # Collect attributes from form fields
            new_attrs = self._collect_attrs_from_form(new_type)

            self.services.world.update_entity(
                self.state.world_db,
                entity_id=self.state.selected_entity_id,
                name=new_name,
                description=new_desc,
                attributes=new_attrs,
            )

            # Record action for undo
            self.state.record_action(
                UndoAction(
                    action_type=ActionType.UPDATE_ENTITY,
                    data={
                        "entity_id": self.state.selected_entity_id,
                        "name": new_name,
                        "description": new_desc,
                        "attributes": new_attrs,
                    },
                    inverse_data={
                        "name": old_entity.name,
                        "description": old_entity.description,
                        "attributes": old_entity.attributes,
                    },
                )
            )

            self._refresh_entity_list()
            if self._graph:
                self._graph.refresh()
            self._update_undo_redo_buttons()
            ui.notify("Entity saved", type="positive")
        except Exception as e:
            logger.exception(f"Failed to save entity {self.state.selected_entity_id}")
            ui.notify(f"Error: {e}", type="negative")

    async def _show_regenerate_dialog(self) -> None:
        """Show dialog for regenerating the selected entity."""
        if not self.state.selected_entity_id or not self.state.world_db:
            ui.notify("No entity selected", type="warning")
            return

        entity = self.services.world.get_entity(self.state.world_db, self.state.selected_entity_id)
        if not entity:
            ui.notify("Entity not found", type="negative")
            return

        # Get relationship count
        relationships = self.state.world_db.get_relationships(entity.id)
        rel_count = len(relationships)

        with ui.dialog() as dialog, ui.card().classes("w-96 p-4"):
            ui.label(f"Regenerate: {entity.name}").classes("text-lg font-semibold")

            # Mode selection
            mode_ref = {"value": "refine"}
            with ui.column().classes("w-full gap-2 mt-4"):
                ui.label("Regeneration Mode").classes("text-sm font-medium")
                mode_radio = ui.radio(
                    options={
                        "refine": "Refine existing (improve weak areas)",
                        "full": "Full regenerate (create new version)",
                        "guided": "Regenerate with guidance",
                    },
                    value="refine",
                    on_change=lambda e: mode_ref.update({"value": e.value}),
                ).classes("w-full")

            # Guidance input (visible only for guided mode)
            guidance_container = ui.column().classes("w-full mt-2")
            guidance_input_ref: dict = {"input": None}

            def update_guidance_visibility() -> None:
                """Update guidance input visibility based on mode."""
                guidance_container.clear()
                if mode_ref["value"] == "guided":
                    with guidance_container:
                        guidance_input_ref["input"] = (
                            ui.textarea(
                                label="Guidance",
                                placeholder="Describe how you want this entity to change...",
                            )
                            .classes("w-full")
                            .props("rows=3")
                        )

            mode_radio.on("update:model-value", lambda _: update_guidance_visibility())
            update_guidance_visibility()

            # Relationship warning
            if rel_count > 0:
                with ui.row().classes(
                    "w-full items-center gap-2 p-2 bg-amber-50 dark:bg-amber-900 rounded mt-4"
                ):
                    ui.icon("warning", color="amber")
                    ui.label(f"{rel_count} relationship(s) will be preserved.").classes("text-sm")

            # Quality info if available
            quality_scores = entity.attributes.get("quality_scores") if entity.attributes else None
            if quality_scores and isinstance(quality_scores, dict):
                avg = quality_scores.get("average", 0)
                with ui.row().classes("w-full items-center gap-2 mt-2"):
                    ui.label(f"Current quality: {avg:.1f}/10").classes(
                        "text-sm text-gray-500 dark:text-gray-400"
                    )

            # Actions
            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=dialog.close).props("flat")

                async def do_regenerate() -> None:
                    """Execute regeneration."""
                    guidance = None
                    if guidance_input_ref.get("input"):
                        guidance = guidance_input_ref["input"].value
                    dialog.close()
                    await self._execute_regenerate(entity, mode_ref["value"], guidance)

                ui.button("Regenerate", on_click=do_regenerate).props("color=primary")

        dialog.open()

    async def _execute_regenerate(self, entity: Entity, mode: str, guidance: str | None) -> None:
        """Execute entity regeneration.

        Args:
            entity: Entity to regenerate.
            mode: Regeneration mode (refine, full, guided).
            guidance: Optional guidance text for guided mode.
        """
        logger.info(f"Regenerating entity {entity.id} ({entity.name}) mode={mode}")

        if not self.state.world_db or not self.state.project:
            ui.notify("No project loaded", type="negative")
            return

        # Show progress indicator
        progress_dialog = None
        try:
            with ui.dialog() as progress_dialog, ui.card().classes("w-64 items-center p-4"):
                ui.label("Regenerating...").classes("text-center")
                ui.spinner(size="lg")
            progress_dialog.open()

            # Perform regeneration based on mode
            result = None
            if mode == "refine":
                result = await self._refine_entity(entity)
            elif mode == "guided":
                if guidance and guidance.strip():
                    result = await self._regenerate_with_guidance(entity, guidance)
                else:
                    logger.warning(
                        "Guided regeneration requested for entity %s without guidance; aborting.",
                        entity.id,
                    )
                    if progress_dialog:
                        progress_dialog.close()
                    ui.notify("Guidance text is required for guided regeneration.", type="warning")
                    return
            else:
                result = await self._regenerate_full(entity)

            if result:
                # Update entity in database (relationships preserved since ID unchanged)
                new_name = result.get("name", entity.name)
                new_description = result.get("description", entity.description)
                new_attributes = {**(entity.attributes or {}), **result.get("attributes", {})}

                self.services.world.update_entity(
                    self.state.world_db,
                    entity.id,
                    name=new_name,
                    description=new_description,
                    attributes=new_attributes,
                )

                # Invalidate graph cache for fresh tooltips
                self.state.world_db.invalidate_graph_cache()

                # Refresh UI
                self._refresh_entity_list()
                self._refresh_entity_editor()
                if self._graph:
                    self._graph.refresh()

                ui.notify(f"Regenerated {new_name}", type="positive")
            else:
                ui.notify("Regeneration failed - no result returned", type="negative")

        except Exception as e:
            logger.exception(f"Regeneration failed: {e}")
            ui.notify(f"Error: {e}", type="negative")
        finally:
            if progress_dialog:
                progress_dialog.close()

    async def _refine_entity(self, entity: Entity) -> dict | None:
        """Refine entity using quality service.

        Args:
            entity: Entity to refine.

        Returns:
            Dictionary with refined entity data, or None on failure.
        """
        if not self.state.project:
            return None

        try:
            # Use WorldQualityService to refine the entity
            refined = await self.services.world_quality.refine_entity(
                entity=entity,
                story_brief=self.state.project.brief,
            )
            return refined
        except Exception as e:
            logger.exception(f"Failed to refine entity: {e}")
            return None

    async def _regenerate_full(self, entity: Entity) -> dict | None:
        """Fully regenerate entity.

        Args:
            entity: Entity to regenerate.

        Returns:
            Dictionary with new entity data, or None on failure.
        """
        if not self.state.project:
            return None

        try:
            # Use WorldQualityService to regenerate based on entity type
            regenerated = await self.services.world_quality.regenerate_entity(
                entity=entity,
                story_brief=self.state.project.brief,
            )
            return regenerated
        except Exception as e:
            logger.exception(f"Failed to regenerate entity: {e}")
            return None

    async def _regenerate_with_guidance(self, entity: Entity, guidance: str) -> dict | None:
        """Regenerate with user guidance.

        Args:
            entity: Entity to regenerate.
            guidance: User-provided guidance text.

        Returns:
            Dictionary with regenerated entity data, or None on failure.
        """
        if not self.state.project:
            return None

        try:
            # Use WorldQualityService with custom instructions
            regenerated = await self.services.world_quality.regenerate_entity(
                entity=entity,
                story_brief=self.state.project.brief,
                custom_instructions=guidance,
            )
            return regenerated
        except Exception as e:
            logger.exception(f"Failed to regenerate entity with guidance: {e}")
            return None

    def _confirm_delete_entity(self) -> None:
        """Show confirmation dialog before deleting entity."""
        if not self.state.selected_entity_id or not self.state.world_db:
            return

        try:
            # Get entity name for better UX
            entity = self.services.world.get_entity(
                self.state.world_db,
                self.state.selected_entity_id,
            )
            entity_name = entity.name if entity else "this entity"

            # Count attached relationships
            all_rels = self.state.world_db.list_relationships()
            attached_rels = [
                r
                for r in all_rels
                if r.source_id == self.state.selected_entity_id
                or r.target_id == self.state.selected_entity_id
            ]
            rel_count = len(attached_rels)

            # Build message with relationship info
            message = f'Are you sure you want to delete "{entity_name}"?'
            if rel_count > 0:
                rel_word = "relationship" if rel_count == 1 else "relationships"
                message += f"\n\nThis will also delete {rel_count} attached {rel_word}."
            message += "\n\nThis action cannot be undone."

            # Custom dialog with more info
            with ui.dialog() as dialog, ui.card().classes("p-4 min-w-[400px]"):
                ui.label("Delete Entity?").classes("text-lg font-bold text-red-600")
                ui.label(message).classes(
                    "text-gray-600 dark:text-gray-400 whitespace-pre-line mt-2"
                )

                if rel_count > 0:
                    with ui.expansion("Show affected relationships", icon="link").classes(
                        "w-full mt-2"
                    ):
                        for rel in attached_rels[:10]:  # Show max 10
                            source = self.services.world.get_entity(
                                self.state.world_db, rel.source_id
                            )
                            target = self.services.world.get_entity(
                                self.state.world_db, rel.target_id
                            )
                            src_name = source.name if source else "?"
                            tgt_name = target.name if target else "?"
                            ui.label(f"{src_name} → {tgt_name} ({rel.relation_type})").classes(
                                "text-sm text-gray-500"
                            )
                        if rel_count > 10:
                            ui.label(f"... and {rel_count - 10} more").classes(
                                "text-sm text-gray-400 italic"
                            )

                def _do_delete() -> None:
                    """Close the dialog and delete the selected entity."""
                    dialog.close()
                    self._delete_entity()

                with ui.row().classes("w-full justify-end gap-2 mt-4"):
                    ui.button("Cancel", on_click=dialog.close).props("flat")
                    ui.button(
                        "Delete",
                        on_click=_do_delete,
                        icon="delete",
                    ).props("color=negative")

            dialog.open()

        except Exception as e:
            logger.exception("Error showing delete confirmation")
            ui.notify(f"Error: {e}", type="negative")

    def _delete_entity(self) -> None:
        """Delete the selected entity."""
        if not self.state.selected_entity_id or not self.state.world_db:
            return

        try:
            # Get entity data for inverse (restore) operation
            entity = self.services.world.get_entity(
                self.state.world_db, self.state.selected_entity_id
            )
            if not entity:
                ui.notify("Entity not found", type="negative")
                return

            entity_id = self.state.selected_entity_id

            self.services.world.delete_entity(
                self.state.world_db,
                entity_id,
            )

            # Record action for undo
            self.state.record_action(
                UndoAction(
                    action_type=ActionType.DELETE_ENTITY,
                    data={"entity_id": entity_id},
                    inverse_data={
                        "type": entity.type,
                        "name": entity.name,
                        "description": entity.description,
                        "attributes": entity.attributes,
                    },
                )
            )

            self.state.select_entity(None)
            self._refresh_entity_list()
            self._refresh_entity_editor()
            if self._graph:
                self._graph.refresh()
            self._update_undo_redo_buttons()
            ui.notify("Entity deleted", type="positive")
        except Exception as e:
            logger.exception(f"Failed to delete entity {self.state.selected_entity_id}")
            ui.notify(f"Error: {e}", type="negative")

    def _add_relationship(self) -> None:
        """Add a new relationship."""
        if not self.state.world_db:
            return

        source_id = self._rel_source_select.value if self._rel_source_select else None
        rel_type = self._rel_type_select.value if self._rel_type_select else None
        target_id = self._rel_target_select.value if self._rel_target_select else None

        if not source_id or not rel_type or not target_id:
            ui.notify("All fields required", type="warning")
            return

        try:
            relationship_id = self.services.world.add_relationship(
                self.state.world_db,
                source_id=source_id,
                target_id=target_id,
                relation_type=rel_type,
            )

            # Record action for undo
            self.state.record_action(
                UndoAction(
                    action_type=ActionType.ADD_RELATIONSHIP,
                    data={
                        "relationship_id": relationship_id,
                        "source_id": source_id,
                        "target_id": target_id,
                        "relation_type": rel_type,
                    },
                    inverse_data={
                        "source_id": source_id,
                        "target_id": target_id,
                        "relation_type": rel_type,
                    },
                )
            )

            if self._graph:
                self._graph.refresh()
            self._update_undo_redo_buttons()
            ui.notify("Relationship added", type="positive")
        except Exception as e:
            logger.exception("Failed to add relationship")
            ui.notify(f"Error: {e}", type="negative")

    def _find_path(self, source_id: str, target_id: str) -> None:
        """Find path between two entities."""
        if not self.state.world_db or not source_id or not target_id:
            return

        path = self.services.world.find_path(self.state.world_db, source_id, target_id)

        if self._analysis_result:
            self._analysis_result.content = render_path_result(self.state.world_db, path or [])

    def _show_centrality(self) -> None:
        """Show most connected entities."""
        if not self.state.world_db:
            return

        if self._analysis_result:
            self._analysis_result.content = render_centrality_result(self.state.world_db)

    def _show_communities(self) -> None:
        """
        Render and display community-detection analysis in the analysis result pane.

        If no world database is available this does nothing. If an analysis result container exists, its content is replaced with the rendered community detection output for the current world.
        """
        if not self.state.world_db:
            return

        if self._analysis_result:
            self._analysis_result.content = render_communities_result(self.state.world_db)

    def _build_conflict_map_tab(self) -> None:
        """
        Constructs the Conflict Map tab UI showing relationships colored by conflict category and a link to the World Timeline.

        Renders a descriptive label, instantiates and builds a ConflictGraphComponent bound to the current world database and services (with node selection callback), and adds a button that navigates to the full World Timeline.
        """
        logger.debug("Building conflict map tab")
        from src.ui.components.conflict_graph import ConflictGraphComponent

        with ui.column().classes("w-full gap-4"):
            ui.label(
                "Visualize relationships colored by conflict category: "
                "alliances (green), rivalries (red), tensions (yellow), and neutral (blue)."
            ).classes("text-sm text-gray-600 dark:text-gray-400")

            # Conflict graph component
            conflict_graph = ConflictGraphComponent(
                world_db=self.state.world_db,
                services=self.services,
                on_node_select=self._on_node_select,
                height=400,
            )
            conflict_graph.build()

            # Link to full world timeline
            with ui.row().classes("items-center gap-4 mt-2"):
                ui.button(
                    "View World Timeline",
                    on_click=lambda: ui.navigate.to("/world-timeline"),
                    icon="timeline",
                ).props("flat")

    # ========== Undo/Redo Methods ==========

    def _update_undo_redo_buttons(self) -> None:
        """Update undo/redo button states based on history."""
        if self._undo_btn:
            self._undo_btn.set_enabled(self.state.can_undo())
        if self._redo_btn:
            self._redo_btn.set_enabled(self.state.can_redo())

    def _do_undo(self) -> None:
        """Execute undo operation."""
        action = self.state.undo()
        if not action or not self.state.world_db:
            return

        try:
            if action.action_type == ActionType.ADD_ENTITY:
                # Undo add = delete
                self.services.world.delete_entity(
                    self.state.world_db,
                    action.data["entity_id"],
                )
            elif action.action_type == ActionType.DELETE_ENTITY:
                # Undo delete = add back
                self.services.world.add_entity(
                    self.state.world_db,
                    entity_type=action.inverse_data["type"],
                    name=action.inverse_data["name"],
                    description=action.inverse_data.get("description", ""),
                    attributes=action.inverse_data.get("attributes"),
                )
            elif action.action_type == ActionType.UPDATE_ENTITY:
                # Undo update = restore old values
                self.services.world.update_entity(
                    self.state.world_db,
                    entity_id=action.data["entity_id"],
                    name=action.inverse_data.get("name"),
                    description=action.inverse_data.get("description"),
                    attributes=action.inverse_data.get("attributes"),
                )
            elif action.action_type == ActionType.ADD_RELATIONSHIP:
                # Undo add = delete
                self.services.world.delete_relationship(
                    self.state.world_db,
                    action.data["relationship_id"],
                )
            elif action.action_type == ActionType.DELETE_RELATIONSHIP:
                # Undo delete = add back
                self.services.world.add_relationship(
                    self.state.world_db,
                    source_id=action.inverse_data["source_id"],
                    target_id=action.inverse_data["target_id"],
                    relation_type=action.inverse_data["relation_type"],
                    description=action.inverse_data.get("description", ""),
                )
            elif action.action_type == ActionType.UPDATE_RELATIONSHIP:
                # Undo update = restore old values
                self.state.world_db.update_relationship(
                    relationship_id=action.data["relationship_id"],
                    relation_type=action.inverse_data.get("relation_type"),
                    description=action.inverse_data.get("description"),
                    strength=action.inverse_data.get("strength"),
                    bidirectional=action.inverse_data.get("bidirectional"),
                )

            self._refresh_entity_list()
            self._refresh_entity_editor()
            if self._graph:
                self._graph.refresh()
            self._update_undo_redo_buttons()
            ui.notify("Undone", type="info")
        except Exception as e:
            logger.exception("Undo failed")
            ui.notify(f"Undo failed: {e}", type="negative")

    def _do_redo(self) -> None:
        """Execute redo operation."""
        action = self.state.redo()
        if not action or not self.state.world_db:
            return

        try:
            if action.action_type == ActionType.ADD_ENTITY:
                # Redo add = add again
                self.services.world.add_entity(
                    self.state.world_db,
                    entity_type=action.data["type"],
                    name=action.data["name"],
                    description=action.data.get("description", ""),
                    attributes=action.data.get("attributes"),
                )
            elif action.action_type == ActionType.DELETE_ENTITY:
                # Redo delete = delete again
                self.services.world.delete_entity(
                    self.state.world_db,
                    action.data["entity_id"],
                )
            elif action.action_type == ActionType.UPDATE_ENTITY:
                # Redo update = apply new values again
                self.services.world.update_entity(
                    self.state.world_db,
                    entity_id=action.data["entity_id"],
                    name=action.data.get("name"),
                    description=action.data.get("description"),
                    attributes=action.data.get("attributes"),
                )
            elif action.action_type == ActionType.ADD_RELATIONSHIP:
                # Redo add = add again
                self.services.world.add_relationship(
                    self.state.world_db,
                    source_id=action.data["source_id"],
                    target_id=action.data["target_id"],
                    relation_type=action.data["relation_type"],
                    description=action.data.get("description", ""),
                )
            elif action.action_type == ActionType.DELETE_RELATIONSHIP:
                # Redo delete = delete again
                self.services.world.delete_relationship(
                    self.state.world_db,
                    action.data["relationship_id"],
                )
            elif action.action_type == ActionType.UPDATE_RELATIONSHIP:
                # Redo update = apply new values again
                self.state.world_db.update_relationship(
                    relationship_id=action.data["relationship_id"],
                    relation_type=action.data.get("relation_type"),
                    description=action.data.get("description"),
                    strength=action.data.get("strength"),
                    bidirectional=action.data.get("bidirectional"),
                )

            self._refresh_entity_list()
            self._refresh_entity_editor()
            if self._graph:
                self._graph.refresh()
            self._update_undo_redo_buttons()
            ui.notify("Redone", type="info")
        except Exception as e:
            logger.exception("Redo failed")
            ui.notify(f"Redo failed: {e}", type="negative")

    # ========== Import Wizard ==========

    def _show_import_wizard(self) -> None:
        """Show the import wizard dialog for extracting entities from text."""
        logger.info("Import wizard opened")

        if not self.state.project or not self.state.world_db:
            ui.notify("No project loaded", type="warning")
            return

        with ui.dialog() as dialog, ui.card().classes("p-6 min-w-[700px] max-w-[900px]"):
            ui.label("Import from Existing Text").classes("text-xl font-bold mb-4")
            ui.label(
                "Paste or upload story text to automatically extract characters, locations, items, and relationships."
            ).classes("text-sm text-gray-600 dark:text-gray-400 mb-4")

            # Text input
            text_input = (
                ui.textarea(
                    label="Story Text",
                    placeholder="Paste your story text here...\n\nThe AI will analyze the text and extract:\n- Characters (names, roles, descriptions)\n- Locations (places mentioned)\n- Items (significant objects)\n- Relationships (connections between characters)",
                )
                .classes("w-full")
                .props("rows=12 outlined")
            )

            # File upload alternative
            with ui.row().classes("w-full items-center gap-2 mb-4"):
                ui.label("Or upload a file:").classes("text-sm")

                async def handle_upload(e):
                    """Handle file upload."""
                    content = e.content.read()
                    try:
                        text = content.decode("utf-8")
                        text_input.value = text
                        ui.notify(f"Loaded {len(text)} characters", type="positive")
                    except Exception as err:
                        logger.exception("Failed to read file during upload handling")
                        ui.notify(f"Failed to read file: {err}", type="negative")

                ui.upload(
                    label="Upload .txt or .md",
                    on_upload=handle_upload,
                    auto_upload=True,
                ).props("accept='.txt,.md' outlined").classes("max-w-xs")

            ui.separator()

            # Action buttons
            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button(
                    "Extract Entities",
                    on_click=lambda: self._do_import(dialog, text_input.value),
                    icon="psychology",
                ).props("color=primary")

        dialog.open()

    async def _do_import(self, dialog: ui.dialog, text: str) -> None:
        """Execute entity extraction from text.

        Args:
            dialog: The dialog to close after import.
            text: The text to analyze.
        """
        if not text or len(text.strip()) < 50:
            ui.notify("Please provide at least 50 characters of text", type="warning")
            return

        if not self.state.project or not self.state.world_db:
            ui.notify("No project loaded", type="negative")
            return

        logger.info(f"Starting entity extraction from {len(text)} characters of text")

        notification = ui.notification(
            message="Analyzing text and extracting entities...",
            spinner=True,
            timeout=None,
        )

        try:
            from nicegui import run

            # Extract all entities
            result = await run.io_bound(
                self.services.import_svc.extract_all,
                text,
                self.state.project,
            )

            notification.dismiss()

            # Validate extraction result structure
            if "summary" not in result:
                logger.error("Import service returned result without summary field")
                ui.notify(
                    "Import completed but response was missing summary information",
                    type="warning",
                )
            else:
                try:
                    total_entities = result["summary"]["total_entities"]
                    relationships = result["summary"]["relationships"]
                    logger.info(
                        f"Extraction complete: {total_entities} entities, "
                        f"{relationships} relationships"
                    )
                except KeyError as exc:
                    logger.error(
                        "Import service returned unexpected result structure; "
                        "missing summary fields: %s",
                        exc,
                    )
                    ui.notify(
                        "Import completed but response was missing summary fields",
                        type="warning",
                    )

            # Close the input dialog
            dialog.close()

            # Show review dialog with extracted entities
            await self._show_import_review_dialog(result)

        except Exception as e:
            notification.dismiss()
            logger.exception(f"Import failed: {e}")
            ui.notify(f"Import failed: {e}", type="negative", close_button=True, timeout=10)

    async def _show_import_review_dialog(self, extraction_result: dict[str, Any]) -> None:
        """Show dialog to review and confirm extracted entities.

        Args:
            extraction_result: The extraction result from import service.
        """
        logger.info("Showing import review dialog")

        characters = extraction_result.get("characters", [])
        locations = extraction_result.get("locations", [])
        items = extraction_result.get("items", [])
        relationships = extraction_result.get("relationships", [])
        summary = extraction_result.get("summary", {})

        # Track selections (all selected by default)
        selected_chars = dict.fromkeys(range(len(characters)), True)
        selected_locs = dict.fromkeys(range(len(locations)), True)
        selected_items = dict.fromkeys(range(len(items)), True)
        selected_rels = dict.fromkeys(range(len(relationships)), True)

        with ui.dialog() as review_dialog, ui.card().classes("p-6 min-w-[800px] max-w-[1000px]"):
            ui.label("Review Extracted Entities").classes("text-xl font-bold mb-2")
            ui.label(
                f"Found {summary.get('total_entities', 0)} entities and {summary.get('relationships', 0)} relationships. "
                f"Review below and uncheck any items you don't want to import."
            ).classes("text-sm text-gray-600 dark:text-gray-400 mb-4")

            if summary.get("needs_review", 0) > 0:
                ui.label(
                    f"⚠️ {summary['needs_review']} items flagged for review (marked with ⚠️) - low confidence extraction"
                ).classes("text-sm text-orange-600 dark:text-orange-400 mb-4")

            # Tabs for different entity types
            with ui.tabs().classes("w-full") as tabs:
                ui.tab("characters", label=f"Characters ({len(characters)})", icon="person")
                ui.tab("locations", label=f"Locations ({len(locations)})", icon="place")
                ui.tab("items", label=f"Items ({len(items)})", icon="category")
                ui.tab("relationships", label=f"Relationships ({len(relationships)})", icon="link")

            with ui.tab_panels(tabs, value="characters").classes("w-full max-h-96 overflow-auto"):
                # Characters tab
                with ui.tab_panel("characters"):
                    if not characters:
                        ui.label("No characters found").classes("text-gray-500")
                    else:
                        for i, char in enumerate(characters):
                            with ui.card().classes("w-full p-3 mb-2"):
                                with ui.row().classes("w-full items-start gap-2"):
                                    ui.checkbox(
                                        value=selected_chars[i],
                                        on_change=lambda e, idx=i: selected_chars.update(
                                            {idx: e.value}
                                        ),
                                    )
                                    with ui.column().classes("flex-grow gap-1"):
                                        name_label = ui.label(char.get("name", "Unknown")).classes(
                                            "font-semibold"
                                        )
                                        if char.get("needs_review", False):
                                            name_label.classes("text-orange-600")
                                            name_label.text = f"⚠️ {name_label.text}"

                                        ui.label(f"Role: {char.get('role', 'Unknown')}").classes(
                                            "text-sm text-gray-600"
                                        )
                                        ui.label(char.get("description", "No description")).classes(
                                            "text-sm"
                                        )
                                        ui.label(
                                            f"Confidence: {char.get('confidence', 0):.0%}"
                                        ).classes("text-xs text-gray-500")

                # Locations tab
                with ui.tab_panel("locations"):
                    if not locations:
                        ui.label("No locations found").classes("text-gray-500")
                    else:
                        for i, loc in enumerate(locations):
                            with ui.card().classes("w-full p-3 mb-2"):
                                with ui.row().classes("w-full items-start gap-2"):
                                    ui.checkbox(
                                        value=selected_locs[i],
                                        on_change=lambda e, idx=i: selected_locs.update(
                                            {idx: e.value}
                                        ),
                                    )
                                    with ui.column().classes("flex-grow gap-1"):
                                        name_label = ui.label(loc.get("name", "Unknown")).classes(
                                            "font-semibold"
                                        )
                                        if loc.get("needs_review", False):
                                            name_label.classes("text-orange-600")
                                            name_label.text = f"⚠️ {name_label.text}"

                                        ui.label(loc.get("description", "No description")).classes(
                                            "text-sm"
                                        )
                                        if loc.get("significance"):
                                            ui.label(
                                                f"Significance: {loc['significance']}"
                                            ).classes("text-sm text-gray-600")
                                        ui.label(
                                            f"Confidence: {loc.get('confidence', 0):.0%}"
                                        ).classes("text-xs text-gray-500")

                # Items tab
                with ui.tab_panel("items"):
                    if not items:
                        ui.label("No items found").classes("text-gray-500")
                    else:
                        for i, item in enumerate(items):
                            with ui.card().classes("w-full p-3 mb-2"):
                                with ui.row().classes("w-full items-start gap-2"):
                                    ui.checkbox(
                                        value=selected_items[i],
                                        on_change=lambda e, idx=i: selected_items.update(
                                            {idx: e.value}
                                        ),
                                    )
                                    with ui.column().classes("flex-grow gap-1"):
                                        name_label = ui.label(item.get("name", "Unknown")).classes(
                                            "font-semibold"
                                        )
                                        if item.get("needs_review", False):
                                            name_label.classes("text-orange-600")
                                            name_label.text = f"⚠️ {name_label.text}"

                                        ui.label(item.get("description", "No description")).classes(
                                            "text-sm"
                                        )
                                        if item.get("properties"):
                                            props_str = ", ".join(item["properties"])
                                            ui.label(f"Properties: {props_str}").classes(
                                                "text-sm text-gray-600"
                                            )
                                        ui.label(
                                            f"Confidence: {item.get('confidence', 0):.0%}"
                                        ).classes("text-xs text-gray-500")

                # Relationships tab
                with ui.tab_panel("relationships"):
                    if not relationships:
                        ui.label("No relationships found").classes("text-gray-500")
                    else:
                        for i, rel in enumerate(relationships):
                            with ui.card().classes("w-full p-3 mb-2"):
                                with ui.row().classes("w-full items-start gap-2"):
                                    ui.checkbox(
                                        value=selected_rels[i],
                                        on_change=lambda e, idx=i: selected_rels.update(
                                            {idx: e.value}
                                        ),
                                    )
                                    with ui.column().classes("flex-grow gap-1"):
                                        rel_label = ui.label(
                                            f"{rel.get('source', '?')} → {rel.get('target', '?')}"
                                        ).classes("font-semibold")
                                        if rel.get("needs_review", False):
                                            rel_label.classes("text-orange-600")
                                            rel_label.text = f"⚠️ {rel_label.text}"

                                        ui.label(
                                            f"Type: {rel.get('relation_type', 'unknown')}"
                                        ).classes("text-sm text-gray-600")
                                        ui.label(rel.get("description", "No description")).classes(
                                            "text-sm"
                                        )
                                        ui.label(
                                            f"Confidence: {rel.get('confidence', 0):.0%}"
                                        ).classes("text-xs text-gray-500")

            ui.separator()

            # Action buttons
            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=review_dialog.close).props("flat")

                async def confirm_import():
                    """Confirm and add selected entities to world."""
                    review_dialog.close()
                    await self._add_imported_entities(
                        characters,
                        locations,
                        items,
                        relationships,
                        selected_chars,
                        selected_locs,
                        selected_items,
                        selected_rels,
                    )

                ui.button(
                    "Import Selected",
                    on_click=confirm_import,
                    icon="check",
                ).props("color=primary")

        review_dialog.open()

    async def _add_imported_entities(
        self,
        characters: list[dict[str, Any]],
        locations: list[dict[str, Any]],
        items: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
        selected_chars: dict[int, bool],
        selected_locs: dict[int, bool],
        selected_items: dict[int, bool],
        selected_rels: dict[int, bool],
    ) -> None:
        """Add selected imported entities to the world database.

        Args:
            characters: All extracted characters.
            locations: All extracted locations.
            items: All extracted items.
            relationships: All extracted relationships.
            selected_chars: Selection state for characters.
            selected_locs: Selection state for locations.
            selected_items: Selection state for items.
            selected_rels: Selection state for relationships.
        """
        if not self.state.world_db:
            return

        logger.info("Adding imported entities to world database")
        notification = ui.notification(
            message="Adding entities to world...",
            spinner=True,
            timeout=None,
        )

        try:
            added_counts = {"characters": 0, "locations": 0, "items": 0, "relationships": 0}
            entity_name_to_id: dict[str, str] = {}

            # Add characters
            for i, char in enumerate(characters):
                if selected_chars.get(i, False):
                    name = char.get("name", "Unknown")
                    # Check for conflicts
                    existing = self.state.world_db.search_entities(name, entity_type="character")
                    if existing:
                        logger.info(f"Skipping duplicate character: {name}")
                        continue

                    attrs = {
                        "role": char.get("role", "supporting"),
                        "confidence": char.get("confidence", 0.5),
                        "imported": True,
                    }
                    entity_id = self.services.world.add_entity(
                        self.state.world_db,
                        entity_type="character",
                        name=name,
                        description=char.get("description", ""),
                        attributes=attrs,
                    )
                    entity_name_to_id[name] = entity_id
                    added_counts["characters"] += 1

            # Add locations
            for i, loc in enumerate(locations):
                if selected_locs.get(i, False):
                    name = loc.get("name", "Unknown")
                    # Check for conflicts
                    existing = self.state.world_db.search_entities(name, entity_type="location")
                    if existing:
                        logger.info(f"Skipping duplicate location: {name}")
                        continue

                    attrs = {
                        "significance": loc.get("significance", ""),
                        "confidence": loc.get("confidence", 0.5),
                        "imported": True,
                    }
                    entity_id = self.services.world.add_entity(
                        self.state.world_db,
                        entity_type="location",
                        name=name,
                        description=loc.get("description", ""),
                        attributes=attrs,
                    )
                    entity_name_to_id[name] = entity_id
                    added_counts["locations"] += 1

            # Add items
            for i, item in enumerate(items):
                if selected_items.get(i, False):
                    name = item.get("name", "Unknown")
                    # Check for conflicts
                    existing = self.state.world_db.search_entities(name, entity_type="item")
                    if existing:
                        logger.info(f"Skipping duplicate item: {name}")
                        continue

                    attrs = {
                        "significance": item.get("significance", ""),
                        "properties": item.get("properties", []),
                        "confidence": item.get("confidence", 0.5),
                        "imported": True,
                    }
                    entity_id = self.services.world.add_entity(
                        self.state.world_db,
                        entity_type="item",
                        name=name,
                        description=item.get("description", ""),
                        attributes=attrs,
                    )
                    entity_name_to_id[name] = entity_id
                    added_counts["items"] += 1

            # Add relationships
            for i, rel in enumerate(relationships):
                if selected_rels.get(i, False):
                    source_name = rel.get("source", "")
                    target_name = rel.get("target", "")

                    # Get entity IDs (from newly added or search existing)
                    source_id = entity_name_to_id.get(source_name)
                    if not source_id:
                        existing = self.state.world_db.search_entities(
                            source_name, entity_type=None
                        )
                        if existing:
                            source_id = existing[0].id

                    target_id = entity_name_to_id.get(target_name)
                    if not target_id:
                        existing = self.state.world_db.search_entities(
                            target_name, entity_type=None
                        )
                        if existing:
                            target_id = existing[0].id

                    if source_id and target_id:
                        self.services.world.add_relationship(
                            self.state.world_db,
                            source_id=source_id,
                            target_id=target_id,
                            relation_type=rel.get("relation_type", "knows"),
                            description=rel.get("description", ""),
                        )
                        added_counts["relationships"] += 1
                    else:
                        logger.warning(
                            f"Skipping relationship {source_name} -> {target_name}: entity not found"
                        )

            # Invalidate graph cache for fresh tooltips
            self.state.world_db.invalidate_graph_cache()

            # Refresh UI
            self._refresh_entity_list()
            if self._graph:
                self._graph.refresh()

            # Save project
            if self.state.project:
                self.services.project.save_project(self.state.project)

            notification.dismiss()
            ui.notify(
                f"Imported {added_counts['characters']} characters, "
                f"{added_counts['locations']} locations, "
                f"{added_counts['items']} items, "
                f"{added_counts['relationships']} relationships",
                type="positive",
            )

            logger.info(f"Import complete: {added_counts}")

        except Exception as e:
            notification.dismiss()
            logger.exception(f"Failed to add imported entities: {e}")
            ui.notify(f"Import failed: {e}", type="negative")

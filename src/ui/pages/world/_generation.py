"""World Builder page - generation mixin with all generation dialogs and methods."""

import logging
import random
import threading
from collections.abc import Callable
from typing import Any

from nicegui import ui

from src.memory.world_quality import RefinementConfig
from src.services.world_quality import EntityGenerationProgress
from src.ui.components.build_dialog import show_build_structure_dialog
from src.ui.pages.world._page import WorldPageBase
from src.utils.exceptions import WorldGenerationError

logger = logging.getLogger(__name__)


class GenerationMixin(WorldPageBase):
    """Mixin providing generation dialogs and methods for WorldPage."""

    def _notify_partial_failure(
        self,
        results_count: int,
        requested_count: int,
        entity_type: str,
        should_cancel: Callable[[], bool],
    ) -> None:
        """Notify user of partial generation failure with cancel awareness.

        Args:
            results_count: Number of entities successfully generated.
            requested_count: Number of entities originally requested.
            entity_type: Type of entity (e.g., "characters", "locations").
            should_cancel: Callable that returns True if generation was cancelled.
        """
        if results_count >= requested_count:
            return
        failed_count = requested_count - results_count
        if should_cancel():
            ui.notify(
                f"Generation cancelled. Generated {results_count} of {requested_count} {entity_type}.",
                type="info",
                timeout=5000,
            )
        else:
            ui.notify(
                f"ERROR: {failed_count} of {requested_count} {entity_type} FAILED to generate! "
                "Check logs for details.",
                type="negative",
                timeout=10000,
                close_button=True,
            )

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
                    logger.info(f"User requested cancellation of {entity_type} generation")
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
                        total_secs = int(progress.estimated_remaining_seconds)
                        if total_secs >= 3600:
                            hours, remainder = divmod(total_secs, 3600)
                            mins, secs = divmod(remainder, 60)
                            eta_label.text = f"~{hours}:{mins:02d}:{secs:02d} remaining"
                        else:
                            mins, secs = divmod(total_secs, 60)
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

                    # Check for partial failure and notify user
                    self._notify_partial_failure(len(results), count, "characters", should_cancel)
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

                    # Check for partial failure and notify user
                    self._notify_partial_failure(
                        len(loc_results), count, "locations", should_cancel
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

                    # Check for partial failure and notify user
                    self._notify_partial_failure(
                        len(faction_results), count, "factions", should_cancel
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

                    # Check for partial failure and notify user
                    self._notify_partial_failure(len(item_results), count, "items", should_cancel)
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

                    # Check for partial failure and notify user
                    self._notify_partial_failure(
                        len(concept_results), count, "concepts", should_cancel
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

                    # Check for partial failure and notify user
                    self._notify_partial_failure(
                        len(rel_results), count, "relationships", should_cancel
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

        # Create cancellation infrastructure for quality generation
        self._generation_cancel_event = threading.Event()

        def should_cancel() -> bool:
            """Check if generation should be cancelled."""
            return (
                self._generation_cancel_event is not None and self._generation_cancel_event.is_set()
            )

        if use_quality:
            # Create progress dialog with cancel button
            self._generation_dialog = ui.dialog().props("persistent")
            progress_label: ui.label | None = None
            progress_bar: ui.linear_progress | None = None
            eta_label: ui.label | None = None
            cancel_btn: ui.button | None = None

            with self._generation_dialog, ui.card().classes("w-96 p-4"):
                ui.label("Generating Relationships").classes("text-lg font-bold")
                progress_label = ui.label(f"Starting generation of {total_count} relationships...")
                progress_bar = ui.linear_progress(value=0).classes("w-full my-2")
                eta_label = ui.label("Calculating...").classes("text-sm text-gray-500")

                def do_cancel() -> None:
                    """Handle cancel button click."""
                    logger.info("User requested cancellation of relationship generation")
                    if self._generation_cancel_event:
                        self._generation_cancel_event.set()
                    if cancel_btn:
                        cancel_btn.disable()
                    if progress_label:
                        progress_label.text = "Cancelling after current relationship..."

                cancel_btn = ui.button("Cancel", on_click=do_cancel).props("flat color=negative")

            self._generation_dialog.open()

            def update_progress(progress: EntityGenerationProgress) -> None:
                """Update dialog with generation progress."""
                if progress_label:
                    if progress.entity_name:
                        progress_label.text = f"Generated: {progress.entity_name}"
                    else:
                        progress_label.text = (
                            f"Generating relationship {progress.current}/{progress.total}..."
                        )

                if progress_bar:
                    progress_bar.value = progress.progress_fraction

                if eta_label:
                    if progress.estimated_remaining_seconds is not None:
                        total_secs = int(progress.estimated_remaining_seconds)
                        if total_secs >= 3600:
                            hours, remainder = divmod(total_secs, 3600)
                            mins, secs = divmod(remainder, 60)
                            eta_label.text = f"~{hours}:{mins:02d}:{secs:02d} remaining"
                        else:
                            mins, secs = divmod(total_secs, 60)
                            eta_label.text = f"~{mins}:{secs:02d} remaining"
                    elif progress.current > 1:
                        eta_label.text = "Calculating..."

            notification = None  # No notification when using dialog
        else:
            notification = ui.notification(
                message=f"Generating relationships for {len(entity_names)} entities...",
                spinner=True,
                timeout=None,
            )
            update_progress = None  # type: ignore[assignment]

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
                    should_cancel,
                    update_progress,
                )

                # Check for partial failure and notify user
                self._notify_partial_failure(
                    len(results), total_count, "relationships", should_cancel
                )

                if len(results) == 0:
                    if self._generation_dialog:
                        self._generation_dialog.close()
                    elif notification:
                        notification.dismiss()
                    ui.notify("Failed to generate any relationships", type="negative")
                    return

                if self._generation_dialog:
                    self._generation_dialog.close()
                elif notification:
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
                if notification:
                    notification.dismiss()
                ui.notify(
                    "Relationship generation requires quality refinement to be enabled",
                    type="warning",
                )

        except Exception as e:
            if self._generation_dialog:
                self._generation_dialog.close()
            elif notification:
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

    # Methods to be implemented by other mixins
    def _refresh_entity_list(self) -> None:
        """Refresh the entity list display - implemented by BrowserMixin."""
        raise NotImplementedError

    def _get_all_entity_names(self) -> list[str]:
        """Get all entity names - implemented by base class."""
        raise NotImplementedError

    def _show_import_wizard(self) -> None:
        """Show import wizard - implemented by ImportMixin."""
        raise NotImplementedError

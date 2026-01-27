"""World Builder page - generation mixin composing toolbar, dialogs, and operations."""

import logging
import random
from collections.abc import Callable

from nicegui import ui

from src.memory.world_quality import RefinementConfig
from src.ui.pages.world._gen_dialogs import GenDialogsMixin
from src.ui.pages.world._gen_operations import GenOperationsMixin
from src.ui.pages.world._page import WorldPageBase

logger = logging.getLogger(__name__)


class GenerationMixin(GenDialogsMixin, GenOperationsMixin, WorldPageBase):
    """Mixin providing generation toolbar, quality settings, and utility methods for WorldPage."""

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

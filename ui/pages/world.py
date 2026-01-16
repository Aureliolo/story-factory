"""World Builder page - entity and relationship management."""

import logging
import random

from nicegui import ui
from nicegui.elements.button import Button
from nicegui.elements.column import Column
from nicegui.elements.html import Html
from nicegui.elements.input import Input
from nicegui.elements.select import Select
from nicegui.elements.textarea import Textarea

from memory.entities import Entity
from memory.world_quality import RefinementConfig
from services import ServiceContainer
from ui.components.entity_card import entity_list_item
from ui.components.graph import GraphComponent
from ui.graph_renderer import (
    render_centrality_result,
    render_communities_result,
    render_path_result,
)
from ui.state import ActionType, AppState, UndoAction
from ui.theme import ENTITY_COLORS
from utils.exceptions import WorldGenerationError

logger = logging.getLogger(__name__)


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

        # UI references
        self._graph: GraphComponent | None = None
        self._entity_list: Column | None = None
        self._editor_container: Column | None = None
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
        self._entity_attrs: dict = {}
        self._rel_source_select: Select | None = None
        self._rel_type_select: Select | None = None
        self._rel_target_select: Select | None = None
        self._analysis_result: Html | None = None
        self._undo_btn: Button | None = None
        self._redo_btn: Button | None = None

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
        with ui.row().classes("w-full h-full gap-4 p-4 flex-wrap lg:flex-nowrap"):
            # Left panel - Entity browser (full width on mobile, 20% on desktop)
            with ui.column().classes("w-full lg:w-1/5 gap-4 min-w-[250px]"):
                self._build_entity_browser()

            # Center panel - Graph visualization (full width on mobile, 60% on desktop)
            with ui.column().classes("w-full lg:w-3/5 gap-4 min-w-[300px]"):
                self._build_graph_section()

            # Right panel - Entity editor (full width on mobile, 20% on desktop)
            self._editor_container = ui.column().classes("w-full lg:w-1/5 gap-4 min-w-[250px]")
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
                    on_click=lambda: self._generate_more("characters"),
                    icon="person_add",
                ).props("outline dense").classes("text-green-600").tooltip("Add more characters")

                ui.button(
                    "Locations",
                    on_click=lambda: self._generate_more("locations"),
                    icon="add_location",
                ).props("outline dense").classes("text-blue-600").tooltip("Add more locations")

                ui.button(
                    "Factions",
                    on_click=lambda: self._generate_more("factions"),
                    icon="groups",
                ).props("outline dense").classes("text-amber-600").tooltip(
                    "Add factions/organizations"
                )

                ui.button(
                    "Items",
                    on_click=lambda: self._generate_more("items"),
                    icon="category",
                ).props("outline dense").classes("text-cyan-600").tooltip("Add significant items")

                ui.button(
                    "Concepts",
                    on_click=lambda: self._generate_more("concepts"),
                    icon="lightbulb",
                ).props("outline dense").classes("text-pink-600").tooltip("Add thematic concepts")

                ui.button(
                    "Relationships",
                    on_click=lambda: self._generate_more("relationships"),
                    icon="link",
                ).props("outline dense").classes("text-purple-600").tooltip("Add relationships")

            ui.separator().props("vertical")

            # Regenerate button (dangerous action)
            ui.button(
                "Rebuild World",
                on_click=self._confirm_regenerate,
                icon="refresh",
            ).props("outline color=negative")

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

                def save_settings():
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
            Random integer between min and max from settings.
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

    async def _generate_more(self, entity_type: str) -> None:
        """Generate more entities of a specific type.

        Args:
            entity_type: Type of entities to generate (characters, locations, factions, items, concepts, relationships)
        """
        logger.info(f"Generate more clicked: entity_type={entity_type}")

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

        # Get count from settings (randomized within range)
        count = self._get_random_count(entity_type)
        logger.info(f"Will generate {count} {entity_type}")

        # Get ALL existing entity names to avoid duplicates
        all_existing_names = self._get_all_entity_names()
        logger.info(f"Found {len(all_existing_names)} existing entities to avoid duplicates")

        # Use notification that can be dismissed
        quality_msg = " with quality refinement" if use_quality else ""
        notification = ui.notification(
            message=f"Generating {count} {entity_type}{quality_msg}...",
            spinner=True,
            timeout=None,
        )

        try:
            from nicegui import run

            if entity_type == "characters":
                if use_quality:
                    # Generate characters with quality refinement
                    logger.info("Calling world quality service to generate characters...")
                    results = await run.io_bound(
                        self.services.world_quality.generate_characters_with_quality,
                        self.state.project,
                        all_existing_names,
                        count,
                    )
                    logger.info(f"Generated {len(results)} characters with quality refinement")

                    # Generate mini descriptions for hover tooltips
                    notification.message = "Generating hover summaries..."
                    entity_data = [
                        {"name": c.name, "type": "character", "description": c.description}
                        for c, _ in results
                    ]
                    mini_descs = await run.io_bound(
                        self.services.world_quality.generate_mini_descriptions_batch,
                        entity_data,
                    )

                    # Add to world database with quality scores and mini descriptions
                    for char, scores in results:
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
                    notification.dismiss()
                    avg_quality = (
                        sum(s.average for _, s in results) / len(results) if results else 0
                    )
                    ui.notify(
                        f"Added {len(results)} characters (avg quality: {avg_quality:.1f})",
                        type="positive",
                    )
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
                    )
                    logger.info(f"Generated {len(loc_results)} locations with quality refinement")

                    # Generate mini descriptions for hover tooltips
                    notification.message = "Generating hover summaries..."
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

                    # Add to world database with quality scores and mini descriptions
                    for loc, scores in loc_results:  # type: ignore[assignment]
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
                    notification.dismiss()
                    avg_quality = (
                        sum(s.average for _, s in loc_results) / len(loc_results)
                        if loc_results
                        else 0
                    )
                    ui.notify(
                        f"Added {len(loc_results)} locations (avg quality: {avg_quality:.1f})",
                        type="positive",
                    )
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
                    notification.dismiss()
                    ui.notify(f"Added {added_count} new locations!", type="positive")

            elif entity_type == "factions":
                if use_quality:
                    # Generate factions with quality refinement
                    logger.info("Calling world quality service to generate factions...")
                    faction_results = await run.io_bound(
                        self.services.world_quality.generate_factions_with_quality,
                        self.state.project,
                        all_existing_names,
                        count,
                    )
                    logger.info(
                        f"Generated {len(faction_results)} factions with quality refinement"
                    )

                    # Generate mini descriptions for hover tooltips
                    notification.message = "Generating hover summaries..."
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

                    # Add to world database with quality scores and mini descriptions
                    for faction, scores in faction_results:  # type: ignore[assignment]
                        if isinstance(faction, dict) and "name" in faction:
                            attrs = {
                                "leader": faction.get("leader", ""),
                                "goals": faction.get("goals", []),
                                "values": faction.get("values", []),
                                "quality_scores": scores.to_dict(),
                            }
                            if faction["name"] in mini_descs:
                                attrs["mini_description"] = mini_descs[faction["name"]]
                            self.services.world.add_entity(
                                self.state.world_db,
                                name=faction["name"],
                                entity_type="faction",
                                description=faction.get("description", ""),
                                attributes=attrs,
                            )
                    notification.dismiss()
                    avg_quality = (
                        sum(s.average for _, s in faction_results) / len(faction_results)
                        if faction_results
                        else 0
                    )
                    ui.notify(
                        f"Added {len(faction_results)} factions (avg quality: {avg_quality:.1f})",
                        type="positive",
                    )
                else:
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
                    )
                    logger.info(f"Generated {len(item_results)} items with quality refinement")

                    # Generate mini descriptions for hover tooltips
                    notification.message = "Generating hover summaries..."
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

                    # Add to world database with quality scores and mini descriptions
                    for item, scores in item_results:  # type: ignore[assignment]
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
                    notification.dismiss()
                    avg_quality = (
                        sum(s.average for _, s in item_results) / len(item_results)
                        if item_results
                        else 0
                    )
                    ui.notify(
                        f"Added {len(item_results)} items (avg quality: {avg_quality:.1f})",
                        type="positive",
                    )
                else:
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
                    )
                    logger.info(
                        f"Generated {len(concept_results)} concepts with quality refinement"
                    )

                    # Generate mini descriptions for hover tooltips
                    notification.message = "Generating hover summaries..."
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

                    # Add to world database with quality scores and mini descriptions
                    for concept, scores in concept_results:  # type: ignore[assignment]
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
                    notification.dismiss()
                    avg_quality = (
                        sum(s.average for _, s in concept_results) / len(concept_results)
                        if concept_results
                        else 0
                    )
                    ui.notify(
                        f"Added {len(concept_results)} concepts (avg quality: {avg_quality:.1f})",
                        type="positive",
                    )
                else:
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
                    )
                    logger.info(
                        f"Generated {len(rel_results)} relationships with quality refinement"
                    )

                    # Add to world database with quality scores
                    added = 0
                    for rel_data, scores in rel_results:  # type: ignore[assignment]
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
                    notification.dismiss()
                    avg_quality = (
                        sum(s.average for _, s in rel_results) / len(rel_results)
                        if rel_results
                        else 0
                    )
                    ui.notify(
                        f"Added {added} relationships (avg quality: {avg_quality:.1f})",
                        type="positive",
                    )
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
            notification.dismiss()
            logger.error(f"World generation failed for {entity_type}: {e}")
            ui.notify(f"Generation failed: {e}", type="negative", close_button=True, timeout=10)
        except Exception as e:
            notification.dismiss()
            logger.exception(f"Unexpected error generating {entity_type}: {e}")
            ui.notify(f"Error: {e}", type="negative")

    def _confirm_regenerate(self) -> None:
        """Show confirmation dialog before regenerating world."""
        logger.info("Rebuild World button clicked - showing confirmation dialog")

        # Check if there's existing story content
        has_chapters = (
            self.state.project
            and self.state.project.chapters
            and any(c.content for c in self.state.project.chapters)
        )
        logger.info(f"Has written chapters: {has_chapters}")

        warning_msg = (
            "This will rebuild the entire world from scratch, "
            "replacing all characters, locations, and relationships."
        )
        if has_chapters:
            warning_msg += (
                "\n\n⚠️ WARNING: You have written chapters. "
                "Regenerating the world may create inconsistencies!"
            )

        with ui.dialog() as dialog, ui.card().classes("p-4 min-w-[400px]"):
            ui.label("Rebuild World?").classes("text-lg font-bold")
            ui.label(warning_msg).classes("text-gray-600 dark:text-gray-400 whitespace-pre-line")

            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button(
                    "Rebuild",
                    on_click=lambda: self._do_regenerate(dialog),
                    icon="refresh",
                ).props("color=negative")

        dialog.open()

    async def _do_regenerate(self, dialog) -> None:
        """Execute world regeneration - builds complete world with locations and relationships."""
        logger.info("User confirmed rebuild - starting world regeneration")
        dialog.close()

        if not self.state.project or not self.state.world_db:
            logger.warning("Rebuild failed: no project or world_db available")
            ui.notify("No project available", type="negative")
            return

        logger.info(f"Starting world rebuild for project {self.state.project.id}")

        # Use notification context manager so it auto-dismisses
        notification = ui.notification(
            message="Rebuilding world... This may take a moment.",
            spinner=True,
            timeout=None,  # Don't auto-dismiss
        )

        try:
            from nicegui import run

            # Clear the world database - delete relationships first, then entities
            relationships = self.state.world_db.list_relationships()
            logger.info(f"Deleting {len(relationships)} existing relationships...")
            for rel in relationships:
                self.state.world_db.delete_relationship(rel.id)
            logger.info("All relationships deleted")

            entities = self.state.world_db.list_entities()
            logger.info(f"Deleting {len(entities)} existing entities...")
            for entity in entities:
                self.state.world_db.delete_entity(entity.id)
            logger.info("All entities deleted")

            # Step 1: Rebuild the story structure via service (this calls the architect)
            notification.message = "Step 1/4: Generating story structure..."
            logger.info("Calling rebuild_world via story service...")
            await run.io_bound(self.services.story.rebuild_world, self.state.project)
            logger.info(
                f"Story service rebuild complete. "
                f"Characters: {len(self.state.project.characters)}, "
                f"Chapters: {len(self.state.project.chapters)}"
            )

            # Step 2: Extract characters to world database
            notification.message = "Step 2/4: Adding characters to world..."
            if self.state.project.characters:
                logger.info(
                    f"Extracting {len(self.state.project.characters)} characters to world database..."
                )
                added_chars = 0
                for char in self.state.project.characters:
                    existing = self.state.world_db.search_entities(
                        char.name, entity_type="character"
                    )
                    if existing:
                        logger.debug(f"Character already exists: {char.name}")
                        continue

                    self.services.world.add_entity(
                        self.state.world_db,
                        name=char.name,
                        entity_type="character",
                        description=char.description,
                        attributes={
                            "role": char.role,
                            "personality_traits": char.personality_traits,
                            "goals": char.goals,
                            "arc_notes": char.arc_notes,
                        },
                    )
                    added_chars += 1
                logger.info(f"Added {added_chars} characters to world database")
            else:
                logger.warning("No characters generated by architect!")

            # Step 3: Generate locations for the world
            notification.message = "Step 3/4: Generating locations..."
            logger.info("Generating locations for the world...")
            try:
                locations = await run.io_bound(
                    self.services.story.generate_locations, self.state.project, 3
                )
                logger.info(f"Generated {len(locations)} locations from LLM")
                added_locs = 0
                for loc in locations:
                    if isinstance(loc, dict) and "name" in loc:
                        self.services.world.add_entity(
                            self.state.world_db,
                            name=loc["name"],
                            entity_type="location",
                            description=loc.get("description", ""),
                            attributes={"significance": loc.get("significance", "")},
                        )
                        added_locs += 1
                    else:
                        logger.warning(f"Skipping invalid location: {loc}")
                logger.info(f"Added {added_locs} locations to world database")
            except Exception as e:
                logger.exception(f"Failed to generate locations: {e}")

            # Step 4: Generate relationships between all entities
            notification.message = "Step 4/4: Generating relationships..."
            logger.info("Generating relationships between entities...")
            try:
                all_entities = self.state.world_db.list_entities()
                entity_names = [e.name for e in all_entities]
                logger.info(f"Generating relationships for {len(entity_names)} entities")

                if len(entity_names) >= 2:
                    generated_rels: list = await run.io_bound(
                        self.services.story.generate_relationships,
                        self.state.project,
                        entity_names,
                        [],  # No existing relationships
                        5,
                    )
                    logger.info(f"Generated {len(generated_rels)} relationships from LLM")

                    added_rels = 0
                    for rel_data in generated_rels:
                        if (
                            isinstance(rel_data, dict)
                            and "source" in rel_data
                            and "target" in rel_data
                        ):
                            source_entity = next(
                                (e for e in all_entities if e.name == rel_data["source"]), None
                            )
                            target_entity = next(
                                (e for e in all_entities if e.name == rel_data["target"]), None
                            )
                            if source_entity and target_entity:
                                self.services.world.add_relationship(
                                    self.state.world_db,
                                    source_entity.id,
                                    target_entity.id,
                                    rel_data.get("relation_type", "knows"),
                                    rel_data.get("description", ""),
                                )
                                added_rels += 1
                            else:
                                logger.warning(
                                    f"Skipping relationship: {rel_data['source']} -> {rel_data['target']} "
                                    "(entity not found)"
                                )
                        else:
                            logger.warning(f"Skipping invalid relationship: {rel_data}")
                    logger.info(f"Added {added_rels} relationships to world database")
                else:
                    logger.warning("Not enough entities to generate relationships")
            except Exception as e:
                logger.exception(f"Failed to generate relationships: {e}")

            # Step 5: Generate mini descriptions for tooltips
            notification.message = "Finalizing: Generating hover summaries..."
            logger.info("Generating mini descriptions for entities...")
            try:
                all_entities = self.state.world_db.list_entities()
                entity_data = [
                    {"name": e.name, "type": e.type, "description": e.description}
                    for e in all_entities
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
            except Exception as e:
                logger.exception(f"Failed to generate mini descriptions: {e}")

            # Dismiss the loading notification
            notification.dismiss()

            # Save the project
            if self.state.project:
                logger.info(f"Saving project {self.state.project.id}...")
                self.services.project.save_project(self.state.project)
                logger.info("Project saved successfully")

            # Log final stats
            final_entities = self.state.world_db.count_entities()
            final_rels = len(self.state.world_db.list_relationships())
            logger.info(
                f"World rebuild complete: {final_entities} entities, {final_rels} relationships"
            )
            ui.notify(
                f"World rebuilt: {final_entities} entities, {final_rels} relationships",
                type="positive",
            )

            # Force page refresh to ensure all components update correctly
            logger.info("Triggering page refresh after world rebuild...")
            ui.navigate.reload()

        except WorldGenerationError as e:
            notification.dismiss()
            logger.error(f"World rebuild generation failed: {e}")
            ui.notify(
                f"World rebuild failed: {e}",
                type="negative",
                close_button=True,
                timeout=10,
            )
        except Exception as e:
            notification.dismiss()
            logger.exception(f"Error rebuilding world: {e}")
            ui.notify(f"Error: {e}", type="negative")

    def _build_entity_browser(self) -> None:
        """Build the entity browser panel."""
        with ui.card().classes("w-full h-full"):
            ui.label("Entity Browser").classes("text-lg font-semibold")

            # Type filter
            ui.label("Filter by Type").classes(
                "text-sm font-medium text-gray-600 dark:text-gray-400 mt-2"
            )

            for entity_type in ["character", "location", "item", "faction", "concept"]:
                color = ENTITY_COLORS[entity_type]
                ui.checkbox(
                    entity_type.title(),
                    value=entity_type in self.state.entity_filter_types,
                    on_change=lambda e, t=entity_type: self._toggle_type_filter(t, e.value),
                ).props(f'color="{color}"')

            ui.separator()

            # Search
            ui.input(
                placeholder="Search entities...",
                on_change=self._on_search,
            ).classes("w-full").props("outlined dense")

            # Entity list with proper styling
            self._entity_list = ui.column().classes(
                "w-full gap-1 overflow-auto max-h-64 p-2 bg-gray-50 dark:bg-gray-800 rounded-lg"
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

    def _build_graph_section(self) -> None:
        """Build the graph visualization section."""
        self._graph = GraphComponent(
            world_db=self.state.world_db,
            settings=self.services.settings,
            on_node_select=self._on_node_select,
            on_edge_select=self._on_edge_select,
            height=450,
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
                    "Delete",
                    on_click=self._confirm_delete_entity,
                    icon="delete",
                ).props("color=negative outline")

    def _build_type_specific_fields(self, entity: Entity) -> None:
        """Build attribute fields specific to entity type."""
        attrs = entity.attributes or {}

        with ui.expansion("Attributes", icon="list", value=True).classes("w-full"):
            if entity.type == "character":
                # Role
                self._attr_role_select = ui.select(
                    label="Role",
                    options=[
                        "protagonist",
                        "antagonist",
                        "love_interest",
                        "supporting",
                        "mentor",
                        "sidekick",
                    ],
                    value=attrs.get("role", "supporting"),
                ).classes("w-full")

                # Traits (comma-separated)
                traits = attrs.get("traits") or attrs.get("personality_traits") or []
                traits_str = ", ".join(traits) if isinstance(traits, list) else str(traits)
                self._attr_traits_input = (
                    ui.input(
                        label="Traits (comma-separated)",
                        value=traits_str,
                    )
                    .classes("w-full")
                    .props("hint='e.g., brave, stubborn, loyal'")
                )

                # Goals (comma-separated)
                goals = attrs.get("goals") or []
                goals_str = ", ".join(goals) if isinstance(goals, list) else str(goals)
                self._attr_goals_input = ui.input(
                    label="Goals (comma-separated)",
                    value=goals_str,
                ).classes("w-full")

                # Arc notes
                self._attr_arc_input = (
                    ui.textarea(
                        label="Character Arc",
                        value=attrs.get("arc") or attrs.get("arc_notes") or "",
                    )
                    .classes("w-full")
                    .props("rows=2")
                )

            elif entity.type == "location":
                # Significance
                self._attr_significance_input = (
                    ui.textarea(
                        label="Significance",
                        value=attrs.get("significance", ""),
                    )
                    .classes("w-full")
                    .props("rows=3 hint='Why is this location important to the story?'")
                )

            elif entity.type == "faction":
                # Leader
                self._attr_leader_input = ui.input(
                    label="Leader",
                    value=attrs.get("leader", ""),
                ).classes("w-full")

                # Goals (comma-separated)
                goals = attrs.get("goals") or []
                goals_str = ", ".join(goals) if isinstance(goals, list) else str(goals)
                self._attr_goals_input = ui.input(
                    label="Goals (comma-separated)",
                    value=goals_str,
                ).classes("w-full")

                # Values (comma-separated)
                values = attrs.get("values") or []
                values_str = ", ".join(values) if isinstance(values, list) else str(values)
                self._attr_values_input = ui.input(
                    label="Values (comma-separated)",
                    value=values_str,
                ).classes("w-full")

            elif entity.type == "item":
                # Significance
                self._attr_significance_input = (
                    ui.textarea(
                        label="Significance",
                        value=attrs.get("significance", ""),
                    )
                    .classes("w-full")
                    .props("rows=2 hint='Why is this item important?'")
                )

                # Properties (comma-separated)
                properties = attrs.get("properties") or []
                props_str = (
                    ", ".join(properties) if isinstance(properties, list) else str(properties)
                )
                self._attr_properties_input = (
                    ui.input(
                        label="Properties (comma-separated)",
                        value=props_str,
                    )
                    .classes("w-full")
                    .props("hint='e.g., indestructible, glowing, cursed'")
                )

            elif entity.type == "concept":
                # Manifestations
                self._attr_manifestations_input = (
                    ui.textarea(
                        label="Manifestations",
                        value=attrs.get("manifestations", ""),
                    )
                    .classes("w-full")
                    .props("rows=3 hint='How does this concept appear in the story?'")
                )

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
        """Build the graph analysis section."""
        with ui.expansion("Analysis Tools", icon="analytics", value=False).classes("w-full"):
            with ui.tabs().classes("w-full") as tabs:
                ui.tab("path", label="Find Path")
                ui.tab("centrality", label="Most Connected")
                ui.tab("communities", label="Communities")

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

        # Filter by search
        if self.state.entity_search_query:
            query = self.state.entity_search_query.lower()
            entities = [
                e for e in entities if query in e.name.lower() or query in e.description.lower()
            ]

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

    def _on_search(self, e) -> None:
        """Handle search input change."""
        self.state.entity_search_query = e.value
        self._refresh_entity_list()

        # Highlight matching nodes in the graph
        if self._graph:
            self._graph.highlight_search(e.value)

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
            rel_desc_input = ui.textarea(
                "Description",
                value=rel.description,
            ).classes("w-full")

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

                def save_relationship():
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

                def delete_relationship():
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
            desc_input = ui.textarea(label="Description").classes("w-full")

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button(
                    "Add",
                    on_click=lambda: self._add_entity(
                        dialog, name_input.value, type_select.value, desc_input.value
                    ),
                ).props("color=primary")

        dialog.open()

    def _add_entity(self, dialog, name: str, entity_type: str, description: str) -> None:
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

    def _collect_attrs_from_form(self, entity_type: str) -> dict:
        """Collect attributes from type-specific form fields.

        Args:
            entity_type: The type of entity being edited.

        Returns:
            Dictionary of attributes collected from form fields.
        """
        # Start with existing attrs to preserve fields not shown in form
        attrs = self._entity_attrs.copy() if self._entity_attrs else {}

        if entity_type == "character":
            if self._attr_role_select:
                attrs["role"] = self._attr_role_select.value
            if self._attr_traits_input and self._attr_traits_input.value:
                traits_list = [
                    t.strip() for t in self._attr_traits_input.value.split(",") if t.strip()
                ]
                attrs["traits"] = traits_list
            if self._attr_goals_input and self._attr_goals_input.value:
                goals_list = [
                    g.strip() for g in self._attr_goals_input.value.split(",") if g.strip()
                ]
                attrs["goals"] = goals_list
            if self._attr_arc_input:
                attrs["arc"] = self._attr_arc_input.value

        elif entity_type == "location":
            if self._attr_significance_input:
                attrs["significance"] = self._attr_significance_input.value

        elif entity_type == "faction":
            if self._attr_leader_input:
                attrs["leader"] = self._attr_leader_input.value
            if self._attr_goals_input and self._attr_goals_input.value:
                goals_list = [
                    g.strip() for g in self._attr_goals_input.value.split(",") if g.strip()
                ]
                attrs["goals"] = goals_list
            if self._attr_values_input and self._attr_values_input.value:
                values_list = [
                    v.strip() for v in self._attr_values_input.value.split(",") if v.strip()
                ]
                attrs["values"] = values_list

        elif entity_type == "item":
            if self._attr_significance_input:
                attrs["significance"] = self._attr_significance_input.value
            if self._attr_properties_input and self._attr_properties_input.value:
                props_list = [
                    p.strip() for p in self._attr_properties_input.value.split(",") if p.strip()
                ]
                attrs["properties"] = props_list

        elif entity_type == "concept":
            if self._attr_manifestations_input:
                attrs["manifestations"] = self._attr_manifestations_input.value

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
        """Show community detection results."""
        if not self.state.world_db:
            return

        if self._analysis_result:
            self._analysis_result.content = render_communities_result(self.state.world_db)

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

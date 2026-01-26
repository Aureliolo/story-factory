"""Shared build structure dialog component.

Provides a dialog for building/rebuilding story structure that can be
used from both the Interview (write) and World pages.
"""

import logging
import threading
from collections.abc import Awaitable, Callable

from nicegui import run, ui

from src.memory.templates import WorldTemplate
from src.services import ServiceContainer
from src.services.world_service import WorldBuildOptions
from src.ui.state import AppState
from src.utils.exceptions import GenerationCancelledError, WorldGenerationError

logger = logging.getLogger(__name__)


async def show_build_structure_dialog(
    state: AppState,
    services: ServiceContainer,
    rebuild: bool = False,
    on_complete: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """Show the build/rebuild structure dialog.

    Args:
        state: The application state with project and world_db.
        services: The service container for accessing world/project services.
        rebuild: If True, shows rebuild warning and uses full_rebuild options.
                 If False, builds without clearing (uses full options).
        on_complete: Optional async callback to run after successful build
                     (e.g., to refresh UI elements).
    """
    mode = "rebuild" if rebuild else "build"
    logger.info(f"Opening build structure dialog (mode={mode})")

    if not state.project or not state.world_db:
        logger.warning("Build dialog failed: no project or world_db available")
        ui.notify("No project available", type="negative")
        return

    brief = state.project.brief
    if not brief:
        ui.notify("No story brief found. Complete the interview first.", type="warning")
        return

    # Create cancellation event
    cancel_event = threading.Event()

    # Create progress dialog
    dialog = ui.dialog().props("persistent")
    is_building = False

    # UI element references (populated during dialog build)
    progress_label: ui.label
    progress_bar: ui.linear_progress
    cancel_btn: ui.button
    build_btn: ui.button

    # Input references for settings
    chapter_input: ui.number
    char_min_input: ui.number
    char_max_input: ui.number
    loc_min_input: ui.number
    loc_max_input: ui.number
    fac_min_input: ui.number
    fac_max_input: ui.number
    item_min_input: ui.number
    item_max_input: ui.number
    concept_min_input: ui.number
    concept_max_input: ui.number

    def do_cancel() -> None:
        """Handle cancel button click."""
        nonlocal is_building
        if is_building:
            cancel_event.set()
            progress_label.text = "Cancelling..."
            cancel_btn.disable()
        else:
            dialog.close()

    async def do_build(world_template: WorldTemplate | None = None) -> None:
        """Execute the build with progress updates.

        Args:
            world_template: Optional world template to use for generation.
        """
        nonlocal is_building
        if not state.project or not state.world_db:
            dialog.close()
            return

        is_building = True
        build_btn.disable()
        cancel_btn.text = "Stop"

        logger.info(f"Starting structure {mode} for project {state.project.id}")

        try:
            # Progress callback to update dialog
            def on_progress(progress) -> None:
                """Update the dialog label and progress bar with current build progress."""
                progress_label.text = progress.message
                progress_bar.value = progress.step / progress.total_steps

            # Use the appropriate build options based on rebuild flag
            if rebuild:
                build_options = WorldBuildOptions.full_rebuild(
                    cancellation_event=cancel_event,
                    world_template=world_template,
                )
            else:
                build_options = WorldBuildOptions.full(
                    cancellation_event=cancel_event,
                    world_template=world_template,
                )

            # Use the unified world build method with cancellation support
            counts = await run.io_bound(
                services.world.build_world,
                state.project,
                state.world_db,
                services,
                build_options,
                on_progress,
            )
            logger.info(f"Structure {mode} counts: {counts}")

            # Save the project
            progress_label.text = "Saving project..."
            progress_bar.value = 0.95
            if state.project:
                logger.info(f"Saving project {state.project.id}...")
                services.project.save_project(state.project)
                logger.info("Project saved successfully")

            progress_bar.value = 1.0
            dialog.close()

            # Log final stats
            chapters = len(state.project.chapters)
            chars = len(state.project.characters)
            action = "rebuilt" if rebuild else "built"
            logger.info(f"Structure {mode} complete: {chapters} chapters, {chars} characters")
            ui.notify(
                f"Story structure {action}: {chapters} chapters, {chars} characters",
                type="positive",
            )

            # Run completion callback if provided
            if on_complete:
                await on_complete()

        except GenerationCancelledError:
            logger.info(f"Structure {mode} cancelled by user")
            dialog.close()
            cancel_msg = "Rebuild cancelled." if rebuild else "Build cancelled."
            ui.notify(cancel_msg, type="warning")

        except WorldGenerationError as e:
            dialog.close()
            logger.error(f"Structure {mode} generation failed: {e}")
            ui.notify(
                f"Structure {mode} failed: {e}",
                type="negative",
                close_button=True,
                timeout=10,
            )
        except Exception as e:
            dialog.close()
            logger.exception(f"Error during structure {mode}: {e}")
            ui.notify(f"Error: {e}", type="negative")

    # Use state-based dark mode styling
    card_bg = "#1f2937" if state.dark_mode else "#ffffff"
    inner_card_bg = "#374151" if state.dark_mode else "#f9fafb"

    # Responsive dialog: wider on larger screens, max 800px
    with (
        dialog,
        ui.card().classes("w-[95vw] max-w-[800px] p-6").style(f"background-color: {card_bg}"),
    ):
        title = "Rebuild World" if rebuild else "Building Story Structure"
        ui.label(title).classes("text-xl font-bold mb-4")

        # Show warning card for rebuild mode
        if rebuild:
            with (
                ui.card()
                .classes("w-full mb-4 border-l-4 border-orange-500")
                .style(f"background-color: {inner_card_bg}")
            ):
                ui.label("This will clear and rebuild the entire world:").classes(
                    "font-medium text-orange-600 dark:text-orange-400"
                )
                ui.label("All existing entities will be deleted").classes("text-sm ml-2")
                ui.label("All relationships will be removed").classes("text-sm ml-2")
                ui.label("New characters, locations, and plot will be generated").classes(
                    "text-sm ml-2"
                )

        # Two-column layout for overview and AI actions
        with ui.row().classes("w-full gap-6 mb-4 flex-wrap"):
            # Story Overview
            with (
                ui.card()
                .classes("flex-1 min-w-[250px]")
                .style(f"background-color: {inner_card_bg}")
            ):
                ui.label("Story Overview:").classes("font-medium mb-2")
                ui.label(f"Genre: {brief.genre}").classes(
                    "text-sm text-gray-600 dark:text-gray-400"
                )
                ui.label(f"Tone: {brief.tone}").classes("text-sm text-gray-600 dark:text-gray-400")
                ui.label(f"Length: {brief.target_length.replace('_', ' ').title()}").classes(
                    "text-sm text-gray-600 dark:text-gray-400"
                )
                premise_text = (
                    brief.premise[:120] + "..." if len(brief.premise) > 120 else brief.premise
                )
                ui.label(f"Premise: {premise_text}").classes(
                    "text-sm text-gray-600 dark:text-gray-400 mt-2"
                )

            # AI Actions
            with (
                ui.card()
                .classes("flex-1 min-w-[250px]")
                .style(f"background-color: {inner_card_bg}")
            ):
                ui.label("The AI will:").classes("font-medium mb-2")
                ui.label("Create detailed world description").classes("text-sm")
                ui.label("Design main characters with backstories").classes("text-sm")
                ui.label("Outline chapter structure and plot points").classes("text-sm")
                ui.label("Establish story rules and timeline").classes("text-sm")

        # World Template selector
        ui.separator().classes("my-2")
        ui.label("World Template").classes("font-medium mb-2")

        # Get available templates
        world_templates = services.world_template.list_templates()
        template_options = {t.id: f"{t.name} - {t.description[:50]}..." for t in world_templates}
        template_options[""] = "None (use story brief only)"

        selected_template: WorldTemplate | None = None
        if state.project.world_template_id:
            selected_template = services.world_template.get_template(
                state.project.world_template_id
            )
            # Validate that the saved template ID still exists
            if not selected_template:
                logger.warning(
                    f"Previously selected world template not found: "
                    f"{state.project.world_template_id}. Clearing selection."
                )
                state.project.world_template_id = None

        def on_template_change(e) -> None:
            """Handle template selection change."""
            nonlocal selected_template
            if e.value:
                selected_template = services.world_template.get_template(e.value)
                if selected_template:
                    logger.debug(f"Selected world template: {selected_template.id}")
            else:
                selected_template = None
                logger.debug("Cleared world template selection")

        template_select = (
            ui.select(
                options=template_options,
                value=state.project.world_template_id or "",
                on_change=on_template_change,
            )
            .props("outlined dense")
            .classes("w-full mb-4")
        )

        # Generation settings section
        ui.separator().classes("my-2")
        ui.label("Generation Settings").classes("font-medium mb-2")

        settings = services.settings
        project = state.project

        # Calculate default chapters based on length
        length_map = {
            "short_story": settings.chapters_short_story,
            "novella": settings.chapters_novella,
            "novel": settings.chapters_novel,
        }
        default_chapters = length_map.get(brief.target_length, settings.chapters_novella)

        # Helper to create min-max input pair
        def create_minmax_input(
            label: str,
            project_min: int | None,
            project_max: int | None,
            default_min: int,
            default_max: int,
            max_val: int = 50,
        ) -> tuple[ui.number, ui.number]:
            """Create a labeled min-max number input pair for generation settings."""
            with ui.column().classes("gap-1"):
                ui.label(f"{label} (min-max)").classes("text-xs text-gray-500")
                with ui.row().classes("gap-1 items-center"):
                    min_input = (
                        ui.number(
                            value=project_min or default_min,
                            min=0,
                            max=max_val,
                            step=1,
                        )
                        .props("dense outlined")
                        .classes("w-16")
                    )
                    ui.label("-").classes("text-gray-400")
                    max_input = (
                        ui.number(
                            value=project_max or default_max,
                            min=0,
                            max=max_val,
                            step=1,
                        )
                        .props("dense outlined")
                        .classes("w-16")
                    )
                ui.label(f"Default: {default_min}-{default_max}").classes("text-xs text-gray-400")
            return min_input, max_input

        # Chapter count (standalone)
        with ui.element("div").classes("mb-4"):
            with ui.column().classes("gap-1"):
                ui.label("Chapters").classes("text-xs text-gray-500")
                chapter_input = (
                    ui.number(
                        value=project.target_chapters or default_chapters,
                        min=1,
                        max=100,
                        step=1,
                    )
                    .props("dense outlined")
                    .classes("w-20")
                )
                ui.label(f"Default: {default_chapters}").classes("text-xs text-gray-400")

        # All entity settings in one grid with consistent horizontal layout
        with ui.element("div").classes("grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4 mb-4"):
            # Character count range
            char_min_input, char_max_input = create_minmax_input(
                "Characters",
                project.target_characters_min,
                project.target_characters_max,
                settings.world_gen_characters_min,
                settings.world_gen_characters_max,
            )

            # Locations count range
            loc_min_input, loc_max_input = create_minmax_input(
                "Locations",
                project.target_locations_min,
                project.target_locations_max,
                settings.world_gen_locations_min,
                settings.world_gen_locations_max,
            )

            # Factions count range
            fac_min_input, fac_max_input = create_minmax_input(
                "Factions",
                project.target_factions_min,
                project.target_factions_max,
                settings.world_gen_factions_min,
                settings.world_gen_factions_max,
            )

            # Items count range
            item_min_input, item_max_input = create_minmax_input(
                "Items",
                project.target_items_min,
                project.target_items_max,
                settings.world_gen_items_min,
                settings.world_gen_items_max,
            )

            # Concepts count range
            concept_min_input, concept_max_input = create_minmax_input(
                "Concepts",
                project.target_concepts_min,
                project.target_concepts_max,
                settings.world_gen_concepts_min,
                settings.world_gen_concepts_max,
            )

        # Function to save settings before building
        async def save_settings_and_build() -> None:
            """Save the generation settings to the project and start the build."""
            nonlocal selected_template
            # Update project with dialog values
            project.target_chapters = int(chapter_input.value) if chapter_input.value else None
            project.target_characters_min = (
                int(char_min_input.value) if char_min_input.value else None
            )
            project.target_characters_max = (
                int(char_max_input.value) if char_max_input.value else None
            )
            project.target_locations_min = int(loc_min_input.value) if loc_min_input.value else None
            project.target_locations_max = int(loc_max_input.value) if loc_max_input.value else None
            project.target_factions_min = int(fac_min_input.value) if fac_min_input.value else None
            project.target_factions_max = int(fac_max_input.value) if fac_max_input.value else None
            project.target_items_min = int(item_min_input.value) if item_min_input.value else None
            project.target_items_max = int(item_max_input.value) if item_max_input.value else None
            project.target_concepts_min = (
                int(concept_min_input.value) if concept_min_input.value else None
            )
            project.target_concepts_max = (
                int(concept_max_input.value) if concept_max_input.value else None
            )
            # Save world template selection
            project.world_template_id = template_select.value if template_select.value else None
            services.project.save_project(project)
            logger.info(
                f"Updated generation settings: chapters={project.target_chapters}, "
                f"chars={project.target_characters_min}-{project.target_characters_max}, "
                f"locs={project.target_locations_min}-{project.target_locations_max}, "
                f"facs={project.target_factions_min}-{project.target_factions_max}, "
                f"items={project.target_items_min}-{project.target_items_max}, "
                f"concepts={project.target_concepts_min}-{project.target_concepts_max}, "
                f"world_template={project.world_template_id}"
            )
            # Now start the build
            await do_build(selected_template)

        ready_text = "Ready to rebuild..." if rebuild else "Ready to build..."
        progress_label = ui.label(ready_text).classes(
            "text-sm text-gray-500 dark:text-gray-400 mb-2"
        )
        progress_bar = ui.linear_progress(value=0, show_value=False).classes("w-full mb-4")

        with ui.row().classes("w-full justify-end gap-2"):
            cancel_btn = ui.button("Cancel", on_click=do_cancel).props("flat")
            if rebuild:
                build_btn = ui.button(
                    "Rebuild World", on_click=save_settings_and_build, icon="refresh"
                ).props("color=negative")
            else:
                build_btn = ui.button("Build Structure", on_click=save_settings_and_build).props(
                    "color=primary"
                )

    dialog.open()

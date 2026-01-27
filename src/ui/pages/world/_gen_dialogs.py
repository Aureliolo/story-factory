"""World Builder page - generation dialog mixins for entity preview and relationship prompts."""

import logging
from typing import Any

from nicegui import ui

from src.ui.pages.world._page import WorldPageBase

logger = logging.getLogger(__name__)


class GenDialogsMixin(WorldPageBase):
    """Mixin providing generation dialog UI methods for WorldPage."""

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

    # Methods to be implemented by other mixins
    async def _generate_more(
        self, entity_type: str, count: int | None = None, custom_instructions: str | None = None
    ) -> None:
        """Generate more entities - implemented by GenOperationsMixin."""
        raise NotImplementedError

    async def _generate_relationships_for_entities(
        self, entity_names: list[str], count_per_entity: int
    ) -> None:
        """Generate relationships for entities - implemented by GenWorldOpsMixin."""
        raise NotImplementedError

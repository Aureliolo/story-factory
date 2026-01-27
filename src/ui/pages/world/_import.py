"""World Builder page - import mixin with import methods."""

import logging
from typing import Any

from nicegui import ui

from src.ui.pages.world._page import WorldPageBase

logger = logging.getLogger(__name__)


class ImportMixin(WorldPageBase):
    """Mixin providing import methods for WorldPage."""

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

    # Methods to be implemented by other mixins
    def _refresh_entity_list(self) -> None:
        """Refresh the entity list display - implemented by BrowserMixin."""
        raise NotImplementedError

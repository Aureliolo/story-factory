"""Settings page - World generation and story structure section mixins."""

from nicegui import ui

from src.ui.pages.settings._page import SettingsPageBase


class WorldGenMixin(SettingsPageBase):
    """Mixin providing world generation and story structure settings functionality."""

    def _build_world_gen_section(self) -> None:
        """Build world generation settings section."""
        with ui.card().classes("w-full h-full"):
            self._section_header(
                "World Generation",
                "public",
                "Configure entity counts for world building. "
                "Actual counts are randomized between min and max values.",
            )

            # Store inputs for saving
            self._world_gen_inputs: dict[str, tuple[ui.number, ui.number]] = {}

            entity_configs = [
                ("characters", "Characters", "people", 1, 20),
                ("locations", "Locations", "place", 1, 15),
                ("factions", "Factions", "groups", 0, 10),
                ("items", "Items", "inventory", 0, 15),
                ("concepts", "Concepts", "lightbulb", 0, 10),
                ("relationships", "Relationships", "share", 1, 40),
            ]

            # Table-style layout with headers
            with ui.element("div").classes("w-full"):
                # Header row
                with ui.row().classes("items-center gap-2 mb-2 text-xs text-gray-500"):
                    ui.element("div").classes("w-28")  # Spacer for label column
                    ui.label("Min").classes("w-14 text-center")
                    ui.label("Max").classes("w-14 text-center")

                # Entity rows
                for key, label, icon, abs_min, abs_max in entity_configs:
                    min_attr = f"world_gen_{key}_min"
                    max_attr = f"world_gen_{key}_max"
                    current_min = getattr(self.settings, min_attr)
                    current_max = getattr(self.settings, max_attr)

                    with ui.row().classes("items-center gap-2"):
                        ui.icon(icon, size="xs").classes("text-gray-500 w-5")
                        ui.label(label).classes("text-sm w-24")
                        min_input = (
                            ui.number(value=current_min, min=abs_min, max=abs_max, step=1)
                            .props("outlined dense")
                            .classes("w-14")
                        )
                        max_input = (
                            ui.number(value=current_max, min=abs_min, max=abs_max, step=1)
                            .props("outlined dense")
                            .classes("w-14")
                        )

                        self._world_gen_inputs[key] = (min_input, max_input)

            # Quality refinement settings (subsection)
            ui.separator().classes("my-3")
            with ui.row().classes("items-center gap-2 mb-2"):
                ui.icon("auto_fix_high", size="xs").classes("text-gray-500")
                ui.label("Quality Refinement").classes("text-sm font-medium")

            # Quality threshold and iterations in a row
            with ui.row().classes("items-center gap-4"):
                with ui.column().classes("gap-1"):
                    ui.label("Threshold").classes("text-xs text-gray-500")
                    self._quality_threshold_input = (
                        ui.number(
                            value=self.settings.world_quality_threshold,
                            min=0.0,
                            max=10.0,
                            step=0.5,
                        )
                        .props("outlined dense")
                        .classes("w-16")
                        .tooltip("Minimum quality score (0-10) to accept entity")
                    )

                with ui.column().classes("gap-1"):
                    ui.label("Max Iter.").classes("text-xs text-gray-500")
                    self._quality_max_iterations_input = (
                        ui.number(
                            value=self.settings.world_quality_max_iterations,
                            min=1,
                            max=10,
                            step=1,
                        )
                        .props("outlined dense")
                        .classes("w-16")
                        .tooltip("Maximum refinement iterations per entity")
                    )

                with ui.column().classes("gap-1"):
                    ui.label("Patience").classes("text-xs text-gray-500")
                    self._quality_patience_input = (
                        ui.number(
                            value=self.settings.world_quality_early_stopping_patience,
                            min=1,
                            max=10,
                            step=1,
                        )
                        .props("outlined dense")
                        .classes("w-16")
                        .tooltip(
                            "Stop early after N consecutive score degradations. "
                            "Saves compute when quality isn't improving."
                        )
                    )

    def _build_story_structure_section(self) -> None:
        """Build story structure settings section."""
        with ui.card().classes("w-full h-full"):
            self._section_header(
                "Story Structure",
                "menu_book",
                "Default chapter counts for different story lengths. "
                "Projects can override these values individually.",
            )

            # Store inputs for saving
            self._chapter_inputs: dict[str, ui.number] = {}

            length_configs = [
                ("short_story", "Short Story", "Quick reads", 1, 5),
                ("novella", "Novella", "Medium length", 3, 15),
                ("novel", "Novel", "Full length", 10, 50),
            ]

            with ui.column().classes("w-full gap-4"):
                for key, label, hint, min_val, max_val in length_configs:
                    attr_name = f"chapters_{key}"
                    current_val = getattr(self.settings, attr_name)

                    with ui.row().classes("w-full items-center gap-3"):
                        with ui.column().classes("flex-grow"):
                            ui.label(label).classes("text-sm font-medium")
                            ui.label(hint).classes("text-xs text-gray-500")

                        self._chapter_inputs[key] = (
                            ui.number(
                                value=current_val,
                                min=min_val,
                                max=max_val,
                                step=1,
                            )
                            .props("outlined dense")
                            .classes("w-20")
                        )

                ui.separator().classes("my-2")

                # Info about per-project overrides
                with ui.row().classes("items-center gap-2"):
                    ui.icon("info", size="xs").classes("text-blue-500")
                    ui.label(
                        "Individual projects can override these in 'Generation Settings'"
                    ).classes("text-xs text-gray-500 dark:text-gray-400")

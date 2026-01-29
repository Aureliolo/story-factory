"""Settings page - application configuration.

Package structure:
    __init__.py      - Main SettingsPage class with build(), save, undo/redo
    _connection.py   - Connection section UI and test
    _models.py       - Model selection and temperature sections
    _interaction.py  - Interaction mode and context sections
    _modes.py        - Generation mode and adaptive learning sections
    _advanced.py     - Advanced LLM, world gen, story structure, data integrity, relationships
    _persistence.py  - Save, snapshot capture, restore, and UI refresh for undo/redo
"""

import logging

from nicegui import ui

from src.services import ServiceContainer
from src.ui.pages.settings._advanced import (
    build_advanced_llm_section,
    build_data_integrity_section,
    build_relationship_validation_section,
    build_story_structure_section,
    build_world_gen_section,
)
from src.ui.pages.settings._connection import (
    build_connection_section,
    test_connection,
)
from src.ui.pages.settings._interaction import (
    build_context_section,
    build_interaction_section,
)
from src.ui.pages.settings._models import (
    build_model_section,
    build_temperature_section,
)
from src.ui.pages.settings._modes import (
    build_learning_section,
    build_mode_section,
)
from src.ui.pages.settings._persistence import (
    capture_settings_snapshot,
    refresh_ui_from_settings,
    restore_settings_snapshot,
    save_settings,
)
from src.ui.state import AppState

logger = logging.getLogger(__name__)

__all__ = ["SettingsPage"]


class SettingsPage:
    """Settings page for application configuration.

    Features:
    - Ollama connection settings
    - Model selection (per-agent or global)
    - Temperature settings
    - Interaction mode
    - Context limits
    - Generation modes (presets for model combinations)
    - Adaptive learning settings (autonomy, triggers, thresholds)
    - Collapsible category groups with expand/collapse all
    """

    def __init__(self, state: AppState, services: ServiceContainer):
        """Initialize settings page.

        Args:
            state: Application state.
            services: Service container.
        """
        self.state = state
        self.services = services

        # Settings reference
        self.settings = services.settings

        # Store expansion refs for expand/collapse all
        self._expansions: list[ui.expansion] = []

        # Register undo/redo handlers for this page
        self.state.on_undo(self._do_undo)
        self.state.on_redo(self._do_redo)

    def build(self) -> None:
        """Build the settings page UI with collapsible category groups."""
        # Max-width centered container
        with ui.element("div").classes("w-full max-w-[1600px] mx-auto p-4"):
            # Header with Expand/Collapse All buttons
            with ui.row().classes("w-full items-center justify-between mb-4"):
                ui.label("Settings").classes("text-2xl font-bold")
                with ui.row().classes("gap-2"):
                    ui.button("Expand All", on_click=self._expand_all, icon="unfold_more").props(
                        "flat dense"
                    )
                    ui.button(
                        "Collapse All", on_click=self._collapse_all, icon="unfold_less"
                    ).props("flat dense")

            # Clear expansions list for fresh build
            self._expansions = []

            # Group 1: Connection & Models
            exp, grid = self._collapsible_group("Connection & Models", "dns")
            self._expansions.append(exp)
            with grid:
                self._build_connection_section()
                self._build_model_section()
                self._build_temperature_section()

            # Group 2: Workflow & Learning
            exp, grid = self._collapsible_group("Workflow & Learning", "route")
            self._expansions.append(exp)
            with grid:
                self._build_interaction_section()
                self._build_mode_section()
                self._build_learning_section()

            # Group 3: Story & World
            exp, grid = self._collapsible_group("Story & World", "auto_stories")
            self._expansions.append(exp)
            with grid:
                self._build_story_structure_section()
                self._build_world_gen_section()
                self._build_relationship_validation_section()
                self._build_context_section()

            # Group 4: System & Reliability
            exp, grid = self._collapsible_group("System & Reliability", "security")
            self._expansions.append(exp)
            with grid:
                self._build_data_integrity_section()
                self._build_advanced_llm_section()

            # Save button
            ui.button(
                "Save Settings",
                on_click=self._save_settings,
                icon="save",
            ).props("color=primary").classes("mt-4")

        logger.debug("Settings page built with %d collapsible groups", len(self._expansions))

    def _collapsible_group(
        self, title: str, icon: str, expanded: bool = True
    ) -> tuple[ui.expansion, ui.element]:
        """Create a collapsible group with grid container inside.

        Args:
            title: Group title displayed in the expansion header.
            icon: Material icon name for the group.
            expanded: Whether the group starts expanded (default True).

        Returns:
            Tuple of (expansion element, grid container element).
        """
        expansion = ui.expansion(title, icon=icon, value=expanded).classes(
            "w-full bg-gray-50 dark:bg-gray-800 rounded-lg mb-4"
        )
        with expansion:
            grid = ui.element("div").style(
                "display: grid; "
                "grid-template-columns: repeat(auto-fit, minmax(300px, 400px)); "
                "gap: 1rem; "
                "align-items: start;"
            )
        return expansion, grid

    def _expand_all(self) -> None:
        """Expand all collapsible groups."""
        for exp in self._expansions:
            exp.value = True
        logger.debug("Expanded all %d groups", len(self._expansions))

    def _collapse_all(self) -> None:
        """Collapse all collapsible groups."""
        for exp in self._expansions:
            exp.value = False
        logger.debug("Collapsed all %d groups", len(self._expansions))

    # ── UI utilities ──────────────────────────────────────────────────────

    def _section_header(self, title: str, icon: str, tooltip: str) -> None:
        """Build a section header with help icon."""
        with ui.row().classes("items-center gap-2 mb-4"):
            ui.icon(icon).classes("text-blue-500")
            ui.label(title).classes("text-lg font-semibold")
            ui.icon("help_outline", size="xs").classes(
                "text-gray-400 dark:text-gray-500 cursor-help"
            ).tooltip(tooltip)

    def _build_number_input(
        self,
        value: int | float,
        min_val: int | float,
        max_val: int | float,
        step: int | float,
        tooltip_text: str,
        width: str = "w-20",
    ) -> ui.number:
        """Build a standardized number input with common styling.

        Args:
            value: Initial value for the input.
            min_val: Minimum allowed value.
            max_val: Maximum allowed value.
            step: Step increment for the input.
            tooltip_text: Tooltip text to display on hover.
            width: Tailwind CSS width class (default: "w-20").

        Returns:
            Configured ui.number element.
        """
        return (
            ui.number(value=value, min=min_val, max=max_val, step=step)
            .props("outlined dense")
            .classes(width)
            .tooltip(tooltip_text)
        )

    # ── Section delegation ────────────────────────────────────────────────

    def _build_connection_section(self) -> None:
        """Build Ollama connection settings."""
        build_connection_section(self)

    def _build_model_section(self) -> None:
        """Build model selection settings."""
        build_model_section(self)

    def _build_temperature_section(self) -> None:
        """Build temperature settings."""
        build_temperature_section(self)

    def _build_interaction_section(self) -> None:
        """Build interaction mode settings."""
        build_interaction_section(self)

    def _build_context_section(self) -> None:
        """Build memory and context settings."""
        build_context_section(self)

    def _build_mode_section(self) -> None:
        """Build generation mode settings."""
        build_mode_section(self)

    def _build_learning_section(self) -> None:
        """Build adaptive learning settings."""
        build_learning_section(self)

    def _build_world_gen_section(self) -> None:
        """Build world generation settings."""
        build_world_gen_section(self)

    def _build_story_structure_section(self) -> None:
        """Build story structure settings."""
        build_story_structure_section(self)

    def _build_data_integrity_section(self) -> None:
        """Build data integrity settings."""
        build_data_integrity_section(self)

    def _build_advanced_llm_section(self) -> None:
        """Build advanced LLM settings."""
        build_advanced_llm_section(self)

    def _build_relationship_validation_section(self) -> None:
        """Build relationship validation settings."""
        build_relationship_validation_section(self)

    # ── Connection test ───────────────────────────────────────────────────

    async def _test_connection(self) -> None:
        """Test Ollama connection."""
        await test_connection(self)

    # ── Save / snapshot / undo / redo ─────────────────────────────────────

    def _save_settings(self) -> None:
        """Persist current UI-configured settings and record an undo snapshot."""
        save_settings(self)

    def _capture_settings_snapshot(self) -> dict:
        """Create a serializable snapshot of current settings suitable for undo/redo."""
        return capture_settings_snapshot(self)

    def _restore_settings_snapshot(self, snapshot: dict) -> None:
        """Restore the SettingsPage state from a snapshot and persist the restored values."""
        restore_settings_snapshot(self, snapshot)

    def _refresh_ui_from_settings(self) -> None:
        """Refresh all UI input elements from current settings values."""
        refresh_ui_from_settings(self)

    def _do_undo(self) -> None:
        """Handle undo for settings changes."""
        from src.ui.state import ActionType

        action = self.state.undo()
        if not action:
            return

        try:
            if action.action_type == ActionType.UPDATE_SETTINGS:
                # Restore old settings
                self._restore_settings_snapshot(action.inverse_data)
                logger.debug("Undone settings change")
        except Exception as e:
            logger.exception("Undo failed for settings")
            ui.notify(f"Undo failed: {e}", type="negative")

    def _do_redo(self) -> None:
        """Handle redo for settings changes."""
        from src.ui.state import ActionType

        action = self.state.redo()
        if not action:
            return

        try:
            if action.action_type == ActionType.UPDATE_SETTINGS:
                # Restore new settings
                self._restore_settings_snapshot(action.data)
                logger.debug("Redone settings change")
        except Exception as e:
            logger.exception("Redo failed for settings")
            ui.notify(f"Redo failed: {e}", type="negative")

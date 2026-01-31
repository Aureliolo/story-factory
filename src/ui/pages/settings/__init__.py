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
from collections.abc import Callable

from nicegui import ui

from src.services import ServiceContainer
from src.ui.pages.settings._advanced import (
    build_calendar_temporal_section,
    build_circuit_breaker_section,
    build_circular_types_section,
    build_data_integrity_section,
    build_duplicate_detection_section,
    build_judge_consistency_section,
    build_refinement_stopping_section,
    build_relationship_minimums_section,
    build_retry_strategy_section,
    build_story_structure_section,
    build_validation_rules_section,
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
        """Build the settings page UI with masonry layout in collapsible groups."""
        # Load Masonry.js for true masonry packing (items placed into shortest column)
        ui.add_head_html(
            '<script src="https://unpkg.com/masonry-layout@4.2.2/dist/masonry.pkgd.min.js">'
            "</script>"
        )

        # Responsive column widths: 1 → 2 → 3 → 4 columns as viewport grows
        ui.add_css("""
            .masonry-sizer, .masonry-item { width: 100%; }
            .masonry-item-wide { width: 100%; }
            .masonry-item { box-sizing: border-box; padding: 8px; }

            @media (min-width: 700px) {
                .masonry-sizer, .masonry-item { width: 50%; }
                .masonry-item-wide { width: 100%; }
            }
            @media (min-width: 1050px) {
                .masonry-sizer, .masonry-item { width: 33.333%; }
                .masonry-item-wide { width: 66.666%; }
            }
            @media (min-width: 1400px) {
                .masonry-sizer, .masonry-item { width: 25%; }
                .masonry-item-wide { width: 50%; }
            }
        """)

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

            # Reset for page rebuilds (build() may be called more than once)
            self._expansions.clear()

            # Define setting groups: (title, icon, build_items)
            # Each build_item is either a function (normal width) or
            # a (function, "wide") tuple for double-width cards.
            _BuildItem = Callable[[], None] | tuple[Callable[[], None], str]
            setting_groups: list[tuple[str, str, list[_BuildItem]]] = [
                (
                    "Connection & Models",
                    "dns",
                    [
                        self._build_connection_section,
                        self._build_model_section,
                        (self._build_temperature_section, "wide"),
                    ],
                ),
                (
                    "Workflow & Learning",
                    "route",
                    [
                        self._build_interaction_section,
                        self._build_mode_section,
                        self._build_learning_section,
                    ],
                ),
                (
                    "Story & World",
                    "auto_stories",
                    [
                        self._build_story_structure_section,
                        self._build_world_gen_section,
                        self._build_validation_rules_section,
                        self._build_circular_types_section,
                        self._build_calendar_temporal_section,
                        self._build_relationship_minimums_section,
                        self._build_context_section,
                    ],
                ),
                (
                    "System & Reliability",
                    "security",
                    [
                        self._build_data_integrity_section,
                        self._build_circuit_breaker_section,
                        self._build_retry_strategy_section,
                        self._build_duplicate_detection_section,
                        self._build_refinement_stopping_section,
                        self._build_judge_consistency_section,
                    ],
                ),
            ]

            # Build groups from the data structure
            for title, icon, build_items in setting_groups:
                exp, container = self._collapsible_group(title, icon)
                self._expansions.append(exp)
                with container:
                    for item in build_items:
                        if isinstance(item, tuple):
                            func, _ = item
                            css = "masonry-item masonry-item-wide"
                        else:
                            func = item
                            css = "masonry-item"
                        with ui.element("div").classes(css):
                            func()

            # Save button
            ui.button(
                "Save Settings",
                on_click=self._save_settings,
                icon="save",
            ).props("color=primary").classes("mt-4")

        # Initialize masonry layout after DOM renders
        ui.timer(0.5, self._init_masonry, once=True)

        logger.debug("Settings page built with %d collapsible groups", len(self._expansions))

    def _collapsible_group(self, title: str, icon: str) -> tuple[ui.expansion, ui.element]:
        """Create a collapsible group with masonry container inside.

        Args:
            title: Group title displayed in the expansion header.
            icon: Material icon name for the group.

        Returns:
            Tuple of (expansion element, masonry container element).
        """
        expansion = ui.expansion(title, icon=icon, value=True).classes(
            "w-full bg-gray-800 rounded-lg mb-4"
        )
        with expansion:
            container = ui.element("div").classes("masonry-container w-full")
            with container:
                # Invisible sizer element — Masonry reads its width to set column size
                ui.element("div").classes("masonry-sizer")
        return expansion, container

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

    def _init_masonry(self) -> None:
        """Initialize Masonry.js on all containers and set up resize handling."""
        ui.run_javascript("""
            try {
                if (typeof Masonry === 'undefined') {
                    console.error('[Settings] Masonry.js not loaded');
                    return;
                }

                function initAllMasonry() {
                    document.querySelectorAll('.masonry-container').forEach(container => {
                        try {
                            if (container._msnry) {
                                container._msnry.layout();
                            } else {
                                container._msnry = new Masonry(container, {
                                    itemSelector: '.masonry-item',
                                    columnWidth: '.masonry-sizer',
                                    percentPosition: true,
                                    transitionDuration: '0.2s'
                                });
                            }
                        } catch (err) {
                            console.error('[Settings] Masonry init failed:', err);
                        }
                    });
                }

                // Clean up previous handlers before re-initializing (page rebuild safety)
                if (window._masonryObserver) {
                    window._masonryObserver.disconnect();
                    window._masonryObserver = null;
                }
                if (window._masonryResizeHandler) {
                    window.removeEventListener('resize', window._masonryResizeHandler);
                    window._masonryResizeHandler = null;
                }

                // Initial layout
                initAllMasonry();

                // Re-layout on window resize (debounced)
                window._masonryResizeHandler = () => {
                    clearTimeout(window._masonryResizeTimer);
                    window._masonryResizeTimer = setTimeout(initAllMasonry, 150);
                };
                window.addEventListener('resize', window._masonryResizeHandler);

                // Re-layout when expansion panels toggle (observe aria-expanded changes)
                window._masonryObserver = new MutationObserver((mutations) => {
                    const isExpansionToggle = mutations.some(m =>
                        m.type === 'attributes' && m.attributeName === 'aria-expanded'
                    );
                    if (isExpansionToggle) {
                        clearTimeout(window._masonryMutationTimer);
                        window._masonryMutationTimer = setTimeout(initAllMasonry, 300);
                    }
                });
                document.querySelectorAll('.q-expansion-item').forEach(exp => {
                    window._masonryObserver.observe(exp, {
                        attributes: true, attributeFilter: ['aria-expanded'],
                        subtree: true
                    });
                });
            } catch (err) {
                console.error('[Settings] Masonry setup failed:', err);
            }
        """)
        logger.debug("Masonry.js initialized on all containers")

    # ── UI utilities ──────────────────────────────────────────────────────

    def _section_header(self, title: str, icon: str, tooltip: str) -> None:
        """Build a section header with help icon."""
        with ui.row().classes("items-center gap-2 mb-4"):
            ui.icon(icon).classes("text-blue-500")
            ui.label(title).classes("text-lg font-semibold")
            ui.icon("help_outline", size="xs").classes("text-gray-500 cursor-help").tooltip(tooltip)

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

    def _build_circuit_breaker_section(self) -> None:
        """Build circuit breaker settings."""
        build_circuit_breaker_section(self)

    def _build_retry_strategy_section(self) -> None:
        """Build retry strategy settings."""
        build_retry_strategy_section(self)

    def _build_duplicate_detection_section(self) -> None:
        """Build duplicate detection settings."""
        build_duplicate_detection_section(self)

    def _build_refinement_stopping_section(self) -> None:
        """Build refinement and stopping settings."""
        build_refinement_stopping_section(self)

    def _build_judge_consistency_section(self) -> None:
        """Build judge consistency settings (multi-call averaging, outlier detection)."""
        build_judge_consistency_section(self)

    def _build_validation_rules_section(self) -> None:
        """Build validation rules settings."""
        build_validation_rules_section(self)

    def _build_circular_types_section(self) -> None:
        """Build circular types settings."""
        build_circular_types_section(self)

    def _build_calendar_temporal_section(self) -> None:
        """Build calendar and temporal settings."""
        build_calendar_temporal_section(self)

    def _build_relationship_minimums_section(self) -> None:
        """Build relationship minimums settings."""
        build_relationship_minimums_section(self)

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

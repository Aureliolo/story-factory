"""Keyboard shortcut manager for the application.

Provides centralized keyboard shortcut handling with help dialog.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from nicegui import ui

logger = logging.getLogger(__name__)


@dataclass
class Shortcut:
    """A keyboard shortcut definition."""

    key: str  # The key code (e.g., "s", "z", "F1")
    modifiers: list[str]  # Modifiers like "ctrl", "shift", "alt"
    description: str  # Human-readable description
    action: Callable[[], None] | None = None  # Optional action callback
    category: str = "General"  # Category for help dialog grouping


class ShortcutManager:
    """Manages keyboard shortcuts for the application.

    Features:
    - Centralized shortcut registration
    - Help dialog with all shortcuts
    - Conflict detection
    - Page-specific shortcuts
    """

    def __init__(self):
        """Initialize the shortcut manager."""
        self._shortcuts: dict[str, Shortcut] = {}
        self._help_dialog: ui.dialog | None = None

    def _get_shortcut_key(self, key: str, modifiers: list[str]) -> str:
        """Generate a unique key for the shortcut registry."""
        mod_str = "+".join(sorted(modifiers)) if modifiers else ""
        return f"{mod_str}+{key}" if mod_str else key

    def register(
        self,
        key: str,
        modifiers: list[str] | None = None,
        description: str = "",
        action: Callable[[], None] | None = None,
        category: str = "General",
    ) -> None:
        """Register a keyboard shortcut.

        Args:
            key: The key code (e.g., "s", "z", "F1").
            modifiers: Modifiers like ["ctrl"], ["ctrl", "shift"].
            description: Human-readable description.
            action: Optional callback function.
            category: Category for grouping in help dialog.
        """
        modifiers = modifiers or []
        shortcut_key = self._get_shortcut_key(key, modifiers)

        if shortcut_key in self._shortcuts:
            logger.warning(f"Overwriting existing shortcut: {shortcut_key}")

        self._shortcuts[shortcut_key] = Shortcut(
            key=key,
            modifiers=modifiers,
            description=description,
            action=action,
            category=category,
        )
        logger.debug(f"Registered shortcut: {shortcut_key} - {description}")

    def unregister(self, key: str, modifiers: list[str] | None = None) -> bool:
        """Unregister a keyboard shortcut.

        Args:
            key: The key code.
            modifiers: The modifiers.

        Returns:
            True if the shortcut was removed, False if not found.
        """
        modifiers = modifiers or []
        shortcut_key = self._get_shortcut_key(key, modifiers)

        if shortcut_key in self._shortcuts:
            del self._shortcuts[shortcut_key]
            logger.debug(f"Unregistered shortcut: {shortcut_key}")
            return True
        return False

    def get_shortcut_display(self, shortcut: Shortcut) -> str:
        """Get a human-readable display string for a shortcut.

        Args:
            shortcut: The shortcut to display.

        Returns:
            Formatted string like "Ctrl+S" or "F1".
        """
        parts = [mod.capitalize() for mod in shortcut.modifiers]
        parts.append(shortcut.key.upper())
        return "+".join(parts)

    def show_help_dialog(self) -> None:
        """Show the keyboard shortcuts help dialog."""
        if self._help_dialog:
            self._help_dialog.open()
            return

        with ui.dialog() as self._help_dialog:
            with ui.card().classes("w-96"):
                ui.label("Keyboard Shortcuts").classes("text-xl font-bold mb-4")

                # Group shortcuts by category
                categories: dict[str, list[Shortcut]] = {}
                for shortcut in self._shortcuts.values():
                    if shortcut.category not in categories:
                        categories[shortcut.category] = []
                    categories[shortcut.category].append(shortcut)

                for category, shortcuts in sorted(categories.items()):
                    ui.label(category).classes("text-lg font-semibold mt-3 mb-2")
                    for shortcut in shortcuts:
                        with ui.row().classes("w-full justify-between items-center py-1"):
                            ui.label(shortcut.description).classes("text-sm")
                            ui.label(self.get_shortcut_display(shortcut)).classes(
                                "text-sm font-mono bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded"
                            )

                with ui.row().classes("w-full justify-end mt-4"):
                    ui.button("Close", on_click=self._help_dialog.close)

        self._help_dialog.open()

    def setup_global_shortcuts(
        self,
        on_save: Callable[[], None] | None = None,
        on_undo: Callable[[], None] | None = None,
        on_redo: Callable[[], None] | None = None,
    ) -> None:
        """Setup common global keyboard shortcuts.

        Args:
            on_save: Callback for Ctrl+S.
            on_undo: Callback for Ctrl+Z.
            on_redo: Callback for Ctrl+Y or Ctrl+Shift+Z.
        """
        # Register shortcuts
        self.register(
            key="s",
            modifiers=["ctrl"],
            description="Save current work",
            action=on_save,
            category="General",
        )
        self.register(
            key="z",
            modifiers=["ctrl"],
            description="Undo last action",
            action=on_undo,
            category="Editing",
        )
        self.register(
            key="y",
            modifiers=["ctrl"],
            description="Redo last action",
            action=on_redo,
            category="Editing",
        )
        self.register(
            key="F1",
            modifiers=[],
            description="Show help / shortcuts",
            action=self.show_help_dialog,
            category="Help",
        )
        self.register(
            key="Escape",
            modifiers=[],
            description="Close dialog / cancel",
            action=None,  # Handled by NiceGUI dialogs
            category="Navigation",
        )

        # Add JavaScript keyboard event handler
        ui.add_body_html("""
        <script>
        document.addEventListener('keydown', function(e) {
            // Ctrl+S - Save
            if (e.ctrlKey && e.key === 's') {
                e.preventDefault();
                if (window.appShortcuts && window.appShortcuts.onSave) {
                    window.appShortcuts.onSave();
                }
            }
            // Ctrl+Z - Undo
            if (e.ctrlKey && !e.shiftKey && e.key === 'z') {
                // Let browser handle in input fields
                if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
                    e.preventDefault();
                    if (window.appShortcuts && window.appShortcuts.onUndo) {
                        window.appShortcuts.onUndo();
                    }
                }
            }
            // Ctrl+Y or Ctrl+Shift+Z - Redo
            if ((e.ctrlKey && e.key === 'y') || (e.ctrlKey && e.shiftKey && e.key === 'z')) {
                if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
                    e.preventDefault();
                    if (window.appShortcuts && window.appShortcuts.onRedo) {
                        window.appShortcuts.onRedo();
                    }
                }
            }
            // F1 - Help
            if (e.key === 'F1') {
                e.preventDefault();
                if (window.appShortcuts && window.appShortcuts.onHelp) {
                    window.appShortcuts.onHelp();
                }
            }
        });

        // Initialize shortcut callbacks
        window.appShortcuts = window.appShortcuts || {};
        </script>
        """)

        logger.info("Global keyboard shortcuts initialized")


# Global shortcut manager instance
shortcut_manager = ShortcutManager()

"""Tests for ui/shortcuts.py keyboard shortcut manager."""

from src.ui.shortcuts import Shortcut, ShortcutManager


class TestShortcut:
    """Tests for Shortcut dataclass."""

    def test_shortcut_creation_minimal(self):
        """Test creating shortcut with minimal fields."""
        shortcut = Shortcut(key="s", modifiers=["ctrl"], description="Save")
        assert shortcut.key == "s"
        assert shortcut.modifiers == ["ctrl"]
        assert shortcut.description == "Save"
        assert shortcut.action is None
        assert shortcut.category == "General"

    def test_shortcut_creation_full(self):
        """Test creating shortcut with all fields."""

        def noop_action() -> None:
            """Do nothing, used to verify action assignment."""
            pass

        shortcut = Shortcut(
            key="z",
            modifiers=["ctrl", "shift"],
            description="Redo",
            action=noop_action,
            category="Editing",
        )
        assert shortcut.key == "z"
        assert shortcut.modifiers == ["ctrl", "shift"]
        assert shortcut.description == "Redo"
        assert shortcut.action is noop_action
        assert shortcut.category == "Editing"


class TestShortcutManager:
    """Tests for ShortcutManager class."""

    def test_register_shortcut(self):
        """Test registering a shortcut."""
        manager = ShortcutManager()
        manager.register(
            key="s",
            modifiers=["ctrl"],
            description="Save",
            category="General",
        )
        assert "ctrl+s" in manager._shortcuts

    def test_register_shortcut_no_modifiers(self):
        """Test registering a shortcut without modifiers."""
        manager = ShortcutManager()
        manager.register(key="F1", description="Help", category="Help")
        assert "F1" in manager._shortcuts

    def test_register_overwrites_existing(self):
        """Test registering overwrites existing shortcut."""
        manager = ShortcutManager()
        manager.register(key="s", modifiers=["ctrl"], description="Save v1")
        manager.register(key="s", modifiers=["ctrl"], description="Save v2")
        assert manager._shortcuts["ctrl+s"].description == "Save v2"

    def test_unregister_shortcut(self):
        """Test unregistering a shortcut."""
        manager = ShortcutManager()
        manager.register(key="s", modifiers=["ctrl"], description="Save")
        result = manager.unregister(key="s", modifiers=["ctrl"])
        assert result is True
        assert "ctrl+s" not in manager._shortcuts

    def test_unregister_nonexistent(self):
        """Test unregistering nonexistent shortcut returns False."""
        manager = ShortcutManager()
        result = manager.unregister(key="x", modifiers=["ctrl"])
        assert result is False

    def test_get_shortcut_display_with_modifiers(self):
        """Test display string for shortcut with modifiers."""
        manager = ShortcutManager()
        shortcut = Shortcut(key="s", modifiers=["ctrl"], description="Save")
        display = manager.get_shortcut_display(shortcut)
        assert display == "Ctrl+S"

    def test_get_shortcut_display_multiple_modifiers(self):
        """Test display string for shortcut with multiple modifiers."""
        manager = ShortcutManager()
        shortcut = Shortcut(key="z", modifiers=["ctrl", "shift"], description="Redo")
        display = manager.get_shortcut_display(shortcut)
        assert "Ctrl" in display
        assert "Shift" in display
        assert "Z" in display

    def test_get_shortcut_display_no_modifiers(self):
        """Test display string for shortcut without modifiers."""
        manager = ShortcutManager()
        shortcut = Shortcut(key="F1", modifiers=[], description="Help")
        display = manager.get_shortcut_display(shortcut)
        assert display == "F1"

    def test_shortcut_key_generation(self):
        """Test shortcut key generation is consistent."""
        manager = ShortcutManager()
        # Order of modifiers shouldn't matter
        key1 = manager._get_shortcut_key("s", ["ctrl", "shift"])
        key2 = manager._get_shortcut_key("s", ["shift", "ctrl"])
        assert key1 == key2

    def test_register_with_action(self):
        """Test registering shortcut with action callback."""
        manager = ShortcutManager()
        called = []

        def action():
            """Record that the action callback was invoked."""
            called.append(True)

        manager.register(
            key="s",
            modifiers=["ctrl"],
            description="Save",
            action=action,
        )

        # Call the action
        shortcut = manager._shortcuts["ctrl+s"]
        assert shortcut.action is not None
        shortcut.action()
        assert len(called) == 1

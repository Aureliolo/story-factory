"""Tests for advanced settings persistence (save/refresh logic).

Covers save_to_settings and refresh_from_settings using mock page objects
to avoid NiceGUI widget dependencies.
"""

from types import SimpleNamespace

from src.ui.pages.settings._advanced_persistence import refresh_from_settings, save_to_settings


def _make_page(settings, **overrides):
    """Create a mock SettingsPage with the minimal attributes needed.

    Uses SimpleNamespace so hasattr() returns False for missing attributes,
    matching the real page's behavior in save_to_settings/refresh_from_settings.
    """
    page = SimpleNamespace(settings=settings, **overrides)
    return page


def _input_widget(value):
    """Create a simple mock UI input widget with a .value attribute."""
    return SimpleNamespace(value=value)


class TestSaveToSettings:
    """Tests for save_to_settings."""

    def test_saves_world_gen_min_max(self):
        """World gen min/max values are saved correctly."""
        settings = SimpleNamespace(world_gen_characters_min=1, world_gen_characters_max=5)
        page = _make_page(
            settings,
            _world_gen_inputs={"characters": (_input_widget(2), _input_widget(6))},
        )

        save_to_settings(page)

        assert settings.world_gen_characters_min == 2
        assert settings.world_gen_characters_max == 6

    def test_saves_world_gen_clamps_min_over_max(self):
        """When min > max, max is clamped to min."""
        settings = SimpleNamespace(world_gen_items_min=1, world_gen_items_max=5)
        page = _make_page(
            settings,
            _world_gen_inputs={"items": (_input_widget(10), _input_widget(3))},
        )

        save_to_settings(page)

        assert settings.world_gen_items_min == 10
        assert settings.world_gen_items_max == 10

    def test_saves_quality_thresholds(self):
        """Per-entity quality thresholds are saved and legacy threshold updated."""
        settings = SimpleNamespace(
            world_quality_thresholds={"character": 8.0, "location": 7.5},
            world_quality_threshold=8.0,
        )
        page = _make_page(
            settings,
            _world_gen_inputs={},
            _quality_threshold_inputs={
                "character": _input_widget(8.5),
                "location": _input_widget(7.0),
            },
        )

        save_to_settings(page)

        assert settings.world_quality_thresholds["character"] == 8.5
        assert settings.world_quality_thresholds["location"] == 7.0
        assert settings.world_quality_threshold == 8.5  # max of thresholds

    def test_saves_chapter_inputs(self):
        """Chapter settings are saved via setattr."""
        settings = SimpleNamespace(chapters_min=3, chapters_max=10)
        page = _make_page(
            settings,
            _world_gen_inputs={},
            _chapter_inputs={"min": _input_widget(4), "max": _input_widget(8)},
        )

        save_to_settings(page)

        assert settings.chapters_min == 4
        assert settings.chapters_max == 8

    def test_saves_advanced_llm_settings_with_type_conversion(self):
        """Advanced LLM settings are saved with proper type conversion."""
        settings = SimpleNamespace(
            circuit_breaker_failure_threshold=3,
            circuit_breaker_timeout=30.0,
            circuit_breaker_enabled=True,
            model_health_cache_ttl=30.0,
            world_quality_hail_mary_min_attempts=5,
        )
        page = _make_page(
            settings,
            _world_gen_inputs={},
            _circuit_breaker_failure_threshold_input=_input_widget("5"),
            _circuit_breaker_timeout_input=_input_widget("60.0"),
            _circuit_breaker_enabled_switch=_input_widget(False),
            _model_health_cache_ttl_input=_input_widget("45.0"),
            _hail_mary_min_attempts_input=_input_widget("10"),
        )

        save_to_settings(page)

        assert settings.circuit_breaker_failure_threshold == 5
        assert isinstance(settings.circuit_breaker_failure_threshold, int)
        assert settings.circuit_breaker_timeout == 60.0
        assert isinstance(settings.circuit_breaker_timeout, float)
        assert settings.circuit_breaker_enabled is False
        assert settings.model_health_cache_ttl == 45.0
        assert settings.world_quality_hail_mary_min_attempts == 10

    def test_saves_data_integrity_settings(self):
        """Entity version retention and backup verify settings are saved."""
        settings = SimpleNamespace(entity_version_retention=5, backup_verify_on_restore=True)
        page = _make_page(
            settings,
            _world_gen_inputs={},
            _entity_version_retention_input=_input_widget(10),
            _backup_verify_on_restore_switch=_input_widget(False),
        )

        save_to_settings(page)

        assert settings.entity_version_retention == 10
        assert settings.backup_verify_on_restore is False

    def test_skips_missing_optional_sections(self):
        """Missing optional page attributes don't cause errors."""
        settings = SimpleNamespace()
        page = _make_page(settings, _world_gen_inputs={})

        # Should not raise even though optional attributes are missing
        save_to_settings(page)


class TestRefreshFromSettings:
    """Tests for refresh_from_settings."""

    def test_refreshes_world_gen_inputs(self):
        """World gen min/max inputs are refreshed from settings."""
        settings = SimpleNamespace(world_gen_locations_min=3, world_gen_locations_max=8)
        min_w = _input_widget(0)
        max_w = _input_widget(0)
        page = _make_page(settings, _world_gen_inputs={"locations": (min_w, max_w)})

        refresh_from_settings(page)

        assert min_w.value == 3
        assert max_w.value == 8

    def test_refreshes_quality_threshold_inputs(self):
        """Per-entity quality threshold inputs are refreshed."""
        settings = SimpleNamespace(world_quality_thresholds={"character": 8.5})
        char_w = _input_widget(0.0)
        page = _make_page(settings, _quality_threshold_inputs={"character": char_w})

        refresh_from_settings(page)

        assert char_w.value == 8.5

    def test_refreshes_advanced_llm_settings(self):
        """Advanced LLM settings are refreshed from settings values."""
        settings = SimpleNamespace(
            circuit_breaker_enabled=False,
            model_health_cache_ttl=45.0,
            world_quality_hail_mary_min_attempts=10,
        )
        switch_w = _input_widget(True)
        ttl_w = _input_widget(30.0)
        hm_w = _input_widget(5)
        page = _make_page(
            settings,
            _circuit_breaker_enabled_switch=switch_w,
            _model_health_cache_ttl_input=ttl_w,
            _hail_mary_min_attempts_input=hm_w,
        )

        refresh_from_settings(page)

        assert switch_w.value is False
        assert ttl_w.value == 45.0
        assert hm_w.value == 10

    def test_refreshes_chapter_inputs(self):
        """Chapter inputs are refreshed from settings."""
        settings = SimpleNamespace(chapters_min=5, chapters_max=12)
        min_w = _input_widget(0)
        max_w = _input_widget(0)
        page = _make_page(settings, _chapter_inputs={"min": min_w, "max": max_w})

        refresh_from_settings(page)

        assert min_w.value == 5
        assert max_w.value == 12

    def test_refreshes_data_integrity_settings(self):
        """Data integrity inputs are refreshed from settings."""
        settings = SimpleNamespace(entity_version_retention=10, backup_verify_on_restore=False)
        retention_w = _input_widget(5)
        verify_w = _input_widget(True)
        page = _make_page(
            settings,
            _entity_version_retention_input=retention_w,
            _backup_verify_on_restore_switch=verify_w,
        )

        refresh_from_settings(page)

        assert retention_w.value == 10
        assert verify_w.value is False

    def test_skips_missing_optional_sections(self):
        """Missing optional page attributes don't cause errors."""
        settings = SimpleNamespace()
        page = _make_page(settings)

        # Should not raise
        refresh_from_settings(page)


class TestSaveRefreshRoundTrip:
    """Tests that save then refresh produces consistent state."""

    def test_round_trip_preserves_values(self):
        """Values saved are faithfully round-tripped through refresh."""
        settings = SimpleNamespace(
            world_gen_factions_min=1,
            world_gen_factions_max=5,
            world_quality_thresholds={"faction": 8.0},
            world_quality_threshold=8.0,
            world_quality_max_iterations=3,
            world_quality_early_stopping_patience=2,
            entity_version_retention=5,
            backup_verify_on_restore=True,
        )

        # Save phase: set new values
        min_w = _input_widget(3)
        max_w = _input_widget(7)
        threshold_w = _input_widget(8.5)
        max_iter_w = _input_widget(5)
        patience_w = _input_widget(3)
        retention_w = _input_widget(10)
        verify_w = _input_widget(False)

        page = _make_page(
            settings,
            _world_gen_inputs={"factions": (min_w, max_w)},
            _quality_threshold_inputs={"faction": threshold_w},
            _quality_max_iterations_input=max_iter_w,
            _quality_patience_input=patience_w,
            _entity_version_retention_input=retention_w,
            _backup_verify_on_restore_switch=verify_w,
        )

        save_to_settings(page)

        # Refresh phase: reset widgets and load back
        min_w.value = 0
        max_w.value = 0
        threshold_w.value = 0.0
        max_iter_w.value = 0
        patience_w.value = 0
        retention_w.value = 0
        verify_w.value = True

        refresh_from_settings(page)

        assert min_w.value == 3
        assert max_w.value == 7
        assert threshold_w.value == 8.5
        assert max_iter_w.value == 5
        assert patience_w.value == 3
        assert retention_w.value == 10
        assert verify_w.value is False

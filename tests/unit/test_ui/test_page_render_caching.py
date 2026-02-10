"""Tests for per-render data sharing to eliminate duplicate API calls (#265).

Verifies that:
- Settings page calls check_health() and list_installed() exactly once per render
- World page calls list_entities() exactly once per render for entity options
- Cached data is consistent across child components
"""

from unittest.mock import MagicMock, patch


class TestSettingsPageCaching:
    """Tests for Settings page per-render data caching."""

    def _make_settings_page(self):
        """Create a SettingsPage instance with mocked dependencies."""
        from src.settings import Settings
        from src.ui.state import AppState

        state = MagicMock(spec=AppState)
        state.on_undo = MagicMock()
        state.on_redo = MagicMock()

        services = MagicMock()
        settings = Settings()
        settings.validate()
        services.settings = settings

        # Mock health check
        health = MagicMock()
        health.is_healthy = True
        health.available_vram = 8.0
        health.message = "OK"
        services.model.check_health.return_value = health

        # Mock installed models
        services.model.list_installed.return_value = [
            "test-model:8b",
            "test-embed:latest",
        ]

        from src.ui.pages.settings import SettingsPage

        page = SettingsPage(state, services)
        return page, services

    def test_check_health_called_once_per_build(self):
        """check_health() should be called exactly once during build()."""
        page, services = self._make_settings_page()

        with patch("src.ui.pages.settings.ui"):
            page.build()

        assert services.model.check_health.call_count == 1, (
            f"check_health() called {services.model.check_health.call_count} times, expected 1"
        )

    def test_list_installed_called_once_per_build(self):
        """list_installed() should be called exactly once during build()."""
        page, services = self._make_settings_page()

        with patch("src.ui.pages.settings.ui"):
            page.build()

        assert services.model.list_installed.call_count == 1, (
            f"list_installed() called {services.model.list_installed.call_count} times, expected 1"
        )

    def test_cached_health_shared_with_connection_section(self):
        """Connection section should use cached health from parent build()."""
        page, services = self._make_settings_page()

        with patch("src.ui.pages.settings.ui"):
            page.build()

        # Verify the cached health object is stored
        assert hasattr(page, "_cached_health")
        assert page._cached_health is services.model.check_health.return_value

    def test_cached_installed_models_shared_with_model_section(self):
        """Model section should use cached installed_models from parent build()."""
        page, _services = self._make_settings_page()

        with patch("src.ui.pages.settings.ui"):
            page.build()

        # Verify the cached models list is stored
        assert hasattr(page, "_cached_installed_models")
        assert page._cached_installed_models == ["test-model:8b", "test-embed:latest"]

    def test_cached_data_consistent_across_sections(self):
        """Both model section and advanced section should see the same installed models."""
        page, services = self._make_settings_page()

        # Track what list_installed returns to verify consistency
        installed_models = ["test-model:8b", "test-embed:latest"]
        services.model.list_installed.return_value = installed_models

        with patch("src.ui.pages.settings.ui"):
            page.build()

        # The cached list should be exactly the same object
        assert page._cached_installed_models is installed_models

    def test_connection_section_uses_cached_health(self):
        """build_connection_section should read from page._cached_health, not call check_health()."""
        from src.ui.pages.settings._connection import build_connection_section

        page = MagicMock()
        page.settings = MagicMock()
        page.settings.ollama_url = "http://localhost:11434"
        page.settings.log_level = "INFO"

        health = MagicMock()
        health.is_healthy = True
        health.available_vram = 8.0
        page._cached_health = health

        with patch("src.ui.pages.settings._connection.ui"):
            build_connection_section(page)

        # check_health should NOT have been called on the services
        page.services.model.check_health.assert_not_called()

    def test_model_section_uses_cached_installed_models(self):
        """build_model_section should read from page._cached_installed_models."""
        from src.ui.pages.settings._models import build_model_section

        page = MagicMock()
        page.settings = MagicMock()
        page.settings.default_model = "auto"
        page.settings.use_per_agent_models = False
        page.settings.agent_models = {}
        page._cached_installed_models = ["test-model:8b"]

        with patch("src.ui.pages.settings._models.ui"):
            build_model_section(page)

        page.services.model.list_installed.assert_not_called()

    def test_advanced_section_uses_cached_installed_models(self):
        """build_duplicate_detection_section should read from page._cached_installed_models."""
        from src.ui.pages.settings._advanced import build_duplicate_detection_section

        page = MagicMock()
        page.settings = MagicMock()
        page.settings.semantic_duplicate_enabled = True
        page.settings.semantic_duplicate_threshold = 0.85
        page.settings.embedding_model = ""
        page._cached_installed_models = ["test-model:8b"]

        with patch("src.ui.pages.settings._advanced.ui"):
            build_duplicate_detection_section(page)

        page.services.model.list_installed.assert_not_called()


class TestWorldPageCaching:
    """Tests for World page per-render entity options caching."""

    def _make_world_page(self):
        """Create a WorldPage instance with mocked dependencies."""
        from src.ui.state import AppState

        state = MagicMock(spec=AppState)
        state.on_undo = MagicMock()
        state.on_redo = MagicMock()
        state.has_project = True
        state.interview_complete = True
        state.world_db = MagicMock()
        state.selected_entity_id = None
        state.entity_search_query = ""
        state.entity_search_names = True
        state.entity_search_descriptions = True
        state.entity_filter_types = []
        state.entity_quality_filter = "all"
        state.entity_sort_by = "name"
        state.entity_sort_descending = False

        services = MagicMock()

        # Create mock entities
        entity1 = MagicMock()
        entity1.id = "e1"
        entity1.name = "Alice"
        entity1.type = "character"
        entity1.description = "A brave hero"
        entity1.attributes = {}

        entity2 = MagicMock()
        entity2.id = "e2"
        entity2.name = "Castle"
        entity2.type = "location"
        entity2.description = "A dark castle"
        entity2.attributes = {}

        services.world.list_entities.return_value = [entity1, entity2]

        from src.ui.pages.world import WorldPage

        page = WorldPage(state, services)
        return page, services, [entity1, entity2]

    def _patch_world_page_build_children(self, page):
        """Patch all child build methods on a WorldPage instance to no-ops.

        Returns a context manager that patches instance methods to prevent
        NiceGUI UI calls during testing.
        """
        from contextlib import contextmanager

        @contextmanager
        def patches():
            """Patch all child build methods and ui module."""
            with (
                patch.object(page, "_build_generation_toolbar"),
                patch.object(page, "_build_entity_browser"),
                patch.object(page, "_build_graph_section"),
                patch.object(page, "_build_entity_editor"),
                patch.object(page, "_build_health_section"),
                patch.object(page, "_build_relationships_section"),
                patch.object(page, "_build_analysis_section"),
                patch("src.ui.pages.world.ui"),
            ):
                yield

        return patches()

    def test_list_entities_called_once_during_build(self):
        """list_entities() should be called exactly once during build() for entity options."""
        page, services, _entities = self._make_world_page()

        with self._patch_world_page_build_children(page):
            page.build()

        # list_entities called once in build() for caching
        assert services.world.list_entities.call_count == 1

    def test_cached_entity_options_populated_during_build(self):
        """_cached_entity_options should be populated with entity id->name mapping during build."""
        page, _services, _entities = self._make_world_page()

        # Track the cache value observed during child builds via a side-effect
        observed_cache = {}

        def capture_cache(*args, **kwargs):
            """Record the current cache value when a child build method is called."""
            observed_cache["value"] = page._cached_entity_options

        with (
            patch.object(page, "_build_generation_toolbar", side_effect=capture_cache),
            patch.object(page, "_build_entity_browser"),
            patch.object(page, "_build_graph_section"),
            patch.object(page, "_build_entity_editor"),
            patch.object(page, "_build_health_section"),
            patch.object(page, "_build_relationships_section"),
            patch.object(page, "_build_analysis_section"),
            patch("src.ui.pages.world.ui"),
        ):
            page.build()

        # During build, the cache should have been populated with entity id->name mapping
        assert observed_cache["value"] == {"e1": "Alice", "e2": "Castle"}
        # After build completes, cache should be cleared for freshness
        assert page._cached_entity_options is None

    def test_cached_entity_options_cleared_after_build(self):
        """_cached_entity_options should be None after build() completes."""
        page, _services, _entities = self._make_world_page()

        with self._patch_world_page_build_children(page):
            page.build()

        # Cache is cleared after build so post-build interactions use fresh data
        assert page._cached_entity_options is None

    def test_get_entity_options_uses_cache_when_available(self):
        """get_entity_options() should return cached data during build."""
        from src.ui.pages.world._graph import get_entity_options

        page = MagicMock()
        page.state.world_db = MagicMock()
        page._cached_entity_options = {"e1": "Alice", "e2": "Castle"}

        result = get_entity_options(page)

        assert result == {"e1": "Alice", "e2": "Castle"}
        # Should NOT call list_entities when cache is available
        page.services.world.list_entities.assert_not_called()

    def test_get_entity_options_falls_back_when_no_cache(self):
        """get_entity_options() should call list_entities() when cache is None."""
        from src.ui.pages.world._graph import get_entity_options

        page = MagicMock()
        page.state.world_db = MagicMock()
        page._cached_entity_options = None

        entity = MagicMock()
        entity.id = "e1"
        entity.name = "Alice"
        page.services.world.list_entities.return_value = [entity]

        result = get_entity_options(page)

        assert result == {"e1": "Alice"}
        page.services.world.list_entities.assert_called_once()

    def test_get_entity_options_returns_copy_of_cache(self):
        """get_entity_options() should return a copy so mutations don't affect cache."""
        from src.ui.pages.world._graph import get_entity_options

        page = MagicMock()
        page.state.world_db = MagicMock()
        original_cache = {"e1": "Alice", "e2": "Castle"}
        page._cached_entity_options = original_cache

        result = get_entity_options(page)

        # Mutating the result should not affect the cache
        result["e3"] = "Bob"
        assert "e3" not in page._cached_entity_options

    def test_build_without_project_skips_caching(self):
        """build() should not fetch entities when no project is selected."""
        page, services, _ = self._make_world_page()
        page.state.has_project = False

        with patch("src.ui.pages.world.ui"):
            page.build()

        services.world.list_entities.assert_not_called()

    def test_build_without_interview_skips_caching(self):
        """build() should not fetch entities when interview is not complete."""
        page, services, _ = self._make_world_page()
        page.state.interview_complete = False

        with patch("src.ui.pages.world.ui"):
            page.build()

        services.world.list_entities.assert_not_called()

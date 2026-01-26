"""Component tests for the settings page.

Tests UI rendering and interactions for the Settings page,
including the Advanced LLM Settings section (WP1/WP2 settings).
"""

import pytest
from nicegui import ui
from nicegui.testing import User


@pytest.mark.component
class TestSettingsPage:
    """Tests for the SettingsPage class."""

    async def test_settings_page_builds_successfully(
        self, user: User, test_app_state, test_services
    ):
        """Settings page builds without errors."""
        from src.ui.pages.settings import SettingsPage

        @ui.page("/test-settings")
        def test_page():
            """Build test page with settings."""
            page = SettingsPage(test_app_state, test_services)
            page.build()

        await user.open("/test-settings")
        # If we get here without exception, the build succeeded

    async def test_advanced_llm_section_exists(self, user: User, test_app_state, test_services):
        """Advanced LLM Settings expansion panel is present."""
        from src.ui.pages.settings import SettingsPage

        page_ref = []

        @ui.page("/test-settings-advanced")
        def test_page():
            """Build test page with settings to check advanced section."""
            page = SettingsPage(test_app_state, test_services)
            page.build()
            page_ref.append(page)

        await user.open("/test-settings-advanced")

        # Verify the section was built by checking for circuit breaker switch
        page = page_ref[0]
        assert hasattr(page, "_circuit_breaker_enabled_switch")
        assert page._circuit_breaker_enabled_switch is not None

    async def test_circuit_breaker_controls_visible(
        self, user: User, test_app_state, test_services
    ):
        """Circuit breaker controls exist."""
        from src.ui.pages.settings import SettingsPage

        page_ref = []

        @ui.page("/test-settings-circuit-breaker")
        def test_page():
            """Build test page with settings to check circuit breaker controls."""
            page = SettingsPage(test_app_state, test_services)
            page.build()
            page_ref.append(page)

        await user.open("/test-settings-circuit-breaker")

        page = page_ref[0]
        # Check all circuit breaker controls exist
        assert hasattr(page, "_circuit_breaker_enabled_switch")
        assert hasattr(page, "_circuit_breaker_failure_threshold_input")
        assert hasattr(page, "_circuit_breaker_success_threshold_input")
        assert hasattr(page, "_circuit_breaker_timeout_input")

        # Verify initial values match settings
        assert (
            page._circuit_breaker_enabled_switch.value
            == test_services.settings.circuit_breaker_enabled
        )
        assert (
            page._circuit_breaker_failure_threshold_input.value
            == test_services.settings.circuit_breaker_failure_threshold
        )
        assert (
            page._circuit_breaker_success_threshold_input.value
            == test_services.settings.circuit_breaker_success_threshold
        )
        assert (
            page._circuit_breaker_timeout_input.value
            == test_services.settings.circuit_breaker_timeout
        )

    async def test_retry_strategy_controls_visible(self, user: User, test_app_state, test_services):
        """Retry strategy controls exist."""
        from src.ui.pages.settings import SettingsPage

        page_ref = []

        @ui.page("/test-settings-retry")
        def test_page():
            """Build test page with settings to check retry controls."""
            page = SettingsPage(test_app_state, test_services)
            page.build()
            page_ref.append(page)

        await user.open("/test-settings-retry")

        page = page_ref[0]
        # Check retry strategy controls exist
        assert hasattr(page, "_retry_temp_increase_input")
        assert hasattr(page, "_retry_simplify_on_attempt_input")

        # Verify initial values match settings
        assert page._retry_temp_increase_input.value == test_services.settings.retry_temp_increase
        assert (
            page._retry_simplify_on_attempt_input.value
            == test_services.settings.retry_simplify_on_attempt
        )

    async def test_semantic_duplicate_controls_visible(
        self, user: User, test_app_state, test_services
    ):
        """Semantic duplicate detection controls exist."""
        from src.ui.pages.settings import SettingsPage

        page_ref = []

        @ui.page("/test-settings-duplicate")
        def test_page():
            """Build test page with settings to check duplicate detection controls."""
            page = SettingsPage(test_app_state, test_services)
            page.build()
            page_ref.append(page)

        await user.open("/test-settings-duplicate")

        page = page_ref[0]
        # Check duplicate detection controls exist
        assert hasattr(page, "_semantic_duplicate_enabled_switch")
        assert hasattr(page, "_semantic_duplicate_threshold_input")
        assert hasattr(page, "_embedding_model_select")

        # Verify initial values match settings
        assert (
            page._semantic_duplicate_enabled_switch.value
            == test_services.settings.semantic_duplicate_enabled
        )
        assert (
            page._semantic_duplicate_threshold_input.value
            == test_services.settings.semantic_duplicate_threshold
        )
        assert page._embedding_model_select.value == test_services.settings.embedding_model

    async def test_refinement_temperature_controls_visible(
        self, user: User, test_app_state, test_services
    ):
        """Refinement temperature controls exist."""
        from src.ui.pages.settings import SettingsPage

        page_ref = []

        @ui.page("/test-settings-temp")
        def test_page():
            """Build test page with settings to check temperature controls."""
            page = SettingsPage(test_app_state, test_services)
            page.build()
            page_ref.append(page)

        await user.open("/test-settings-temp")

        page = page_ref[0]
        # Check temperature controls exist
        assert hasattr(page, "_refinement_temp_start_input")
        assert hasattr(page, "_refinement_temp_end_input")
        assert hasattr(page, "_refinement_temp_decay_select")

        # Verify initial values match settings
        assert (
            page._refinement_temp_start_input.value
            == test_services.settings.world_quality_refinement_temp_start
        )
        assert (
            page._refinement_temp_end_input.value
            == test_services.settings.world_quality_refinement_temp_end
        )
        assert (
            page._refinement_temp_decay_select.value
            == test_services.settings.world_quality_refinement_temp_decay
        )

    async def test_early_stopping_controls_visible(self, user: User, test_app_state, test_services):
        """Early stopping controls exist."""
        from src.ui.pages.settings import SettingsPage

        page_ref = []

        @ui.page("/test-settings-early-stop")
        def test_page():
            """Build test page with settings to check early stopping controls."""
            page = SettingsPage(test_app_state, test_services)
            page.build()
            page_ref.append(page)

        await user.open("/test-settings-early-stop")

        page = page_ref[0]
        # Check early stopping controls exist
        assert hasattr(page, "_early_stopping_min_iterations_input")
        assert hasattr(page, "_early_stopping_variance_tolerance_input")

        # Verify initial values match settings
        assert (
            page._early_stopping_min_iterations_input.value
            == test_services.settings.world_quality_early_stopping_min_iterations
        )
        assert (
            page._early_stopping_variance_tolerance_input.value
            == test_services.settings.world_quality_early_stopping_variance_tolerance
        )

    async def test_settings_save_updates_values(self, user: User, test_app_state, test_services):
        """Saving settings persists new values."""
        from src.ui.pages.settings import SettingsPage

        page_ref = []

        @ui.page("/test-settings-save")
        def test_page():
            """Build test page with settings for save test."""
            page = SettingsPage(test_app_state, test_services)
            page.build()
            page_ref.append(page)

        await user.open("/test-settings-save")

        page = page_ref[0]

        # Change some values
        original_failure_threshold = test_services.settings.circuit_breaker_failure_threshold
        new_failure_threshold = 10

        page._circuit_breaker_failure_threshold_input.value = new_failure_threshold
        page._retry_temp_increase_input.value = 0.25
        page._early_stopping_min_iterations_input.value = 3

        # Save settings
        page._save_settings()

        # Verify settings were updated
        assert test_services.settings.circuit_breaker_failure_threshold == new_failure_threshold
        assert test_services.settings.retry_temp_increase == 0.25
        assert test_services.settings.world_quality_early_stopping_min_iterations == 3

        # Restore original value for test isolation
        test_services.settings.circuit_breaker_failure_threshold = original_failure_threshold

    async def test_settings_snapshot_includes_advanced_llm_settings(
        self, user: User, test_app_state, test_services
    ):
        """Settings snapshot includes all advanced LLM settings."""
        from src.ui.pages.settings import SettingsPage

        page_ref = []

        @ui.page("/test-settings-snapshot")
        def test_page():
            """Build test page with settings for snapshot test."""
            page = SettingsPage(test_app_state, test_services)
            page.build()
            page_ref.append(page)

        await user.open("/test-settings-snapshot")

        page = page_ref[0]
        snapshot = page._capture_settings_snapshot()

        # Verify all advanced LLM settings are in the snapshot
        assert "circuit_breaker_enabled" in snapshot
        assert "circuit_breaker_failure_threshold" in snapshot
        assert "circuit_breaker_success_threshold" in snapshot
        assert "circuit_breaker_timeout" in snapshot
        assert "retry_temp_increase" in snapshot
        assert "retry_simplify_on_attempt" in snapshot
        assert "semantic_duplicate_enabled" in snapshot
        assert "semantic_duplicate_threshold" in snapshot
        assert "embedding_model" in snapshot
        assert "world_quality_refinement_temp_start" in snapshot
        assert "world_quality_refinement_temp_end" in snapshot
        assert "world_quality_refinement_temp_decay" in snapshot
        assert "world_quality_early_stopping_min_iterations" in snapshot
        assert "world_quality_early_stopping_variance_tolerance" in snapshot

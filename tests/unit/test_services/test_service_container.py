"""Tests for ServiceContainer initialization."""

import logging
from unittest.mock import patch

from src.services import ServiceContainer
from src.settings import Settings


class TestServiceContainer:
    """Tests for ServiceContainer class."""

    def test_init_with_provided_settings(self):
        """Test ServiceContainer initialization with provided settings."""
        settings = Settings()
        container = ServiceContainer(settings)

        # Verify all services are initialized
        assert container.settings is settings
        assert container.project is not None
        assert container.world is not None
        assert container.model is not None
        assert container.export is not None
        assert container.mode is not None
        assert container.scoring is not None
        assert container.story is not None
        assert container.world_quality is not None
        assert container.suggestion is not None
        assert container.template is not None
        assert container.backup is not None
        assert container.import_svc is not None
        assert container.comparison is not None
        assert container.timeline is not None
        assert container.conflict_analysis is not None
        assert container.world_template is not None
        assert container.content_guidelines is not None

    def test_init_loads_settings_if_not_provided(self):
        """Test ServiceContainer loads settings when None is passed."""
        with patch("src.services.Settings.load") as mock_load:
            mock_settings = Settings()
            mock_load.return_value = mock_settings

            container = ServiceContainer(None)

            mock_load.assert_called_once()
            assert container.settings is mock_settings

    def test_init_logs_timing(self, caplog):
        """Test ServiceContainer logs initialization timing at INFO level."""
        settings = Settings()
        with caplog.at_level(logging.INFO, logger="src.services"):
            container = ServiceContainer(settings)

        assert any("Initializing ServiceContainer" in r.message for r in caplog.records)
        # Service count excludes 'settings' and 'mode_db' (not a service)
        expected_count = len(container.__class__.__annotations__) - 2
        assert any(
            "ServiceContainer initialized" in r.message
            and f"{expected_count} services" in r.message
            for r in caplog.records
        )

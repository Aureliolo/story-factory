"""Tests for VRAM management functions."""

from unittest.mock import MagicMock, patch

from src.memory.mode_models import VramStrategy
from src.services.model_mode_service import _vram
from src.settings import Settings


class TestPrepareModelShortCircuit:
    """Tests for the prepare_model short-circuit when same model is already prepared."""

    def _make_service(self) -> MagicMock:
        """Create a mock service with VRAM settings for testing."""
        mock_settings = MagicMock(spec=Settings)
        mock_settings.ollama_url = "http://localhost:11434"
        mock_settings.vram_strategy = VramStrategy.SEQUENTIAL.value
        svc = MagicMock()
        svc.settings = mock_settings
        svc._loaded_models = set()
        svc._ollama_client = MagicMock()
        return svc

    def setup_method(self) -> None:
        """Reset the module-level cache before each test."""
        with _vram._last_prepared_model_lock:
            _vram._last_prepared_model_id = None

    @patch("src.settings.get_available_vram", return_value=24.0)
    @patch("src.settings.get_installed_models_with_sizes", return_value={"test-model:8b": 4.0})
    @patch("src.settings.get_model_info")
    def test_same_model_skips_preparation(
        self,
        _mock_info: MagicMock,
        _mock_installed: MagicMock,
        _mock_vram: MagicMock,
    ) -> None:
        """Calling prepare_model twice with the same model skips on second call."""
        svc = self._make_service()

        # First call — full preparation
        _vram.prepare_model(svc, "test-model:8b")
        assert "test-model:8b" in svc._loaded_models

        # Second call — should short-circuit (no unload_all_except call)
        svc._loaded_models = {"test-model:8b"}
        with patch.object(_vram, "unload_all_except") as mock_unload:
            _vram.prepare_model(svc, "test-model:8b")
            mock_unload.assert_not_called()

    @patch("src.settings.get_available_vram", return_value=24.0)
    @patch(
        "src.settings.get_installed_models_with_sizes",
        return_value={"model-a:8b": 4.0, "model-b:8b": 4.0},
    )
    @patch("src.settings.get_model_info")
    def test_different_model_does_not_skip(
        self,
        _mock_info: MagicMock,
        _mock_installed: MagicMock,
        _mock_vram: MagicMock,
    ) -> None:
        """Switching models runs full preparation."""
        svc = self._make_service()

        _vram.prepare_model(svc, "model-a:8b")
        _vram.prepare_model(svc, "model-b:8b")

        # Both should have been loaded
        assert "model-b:8b" in svc._loaded_models


class TestPrepareModelResidencyCheck:
    """Tests for GPU residency check in prepare_model."""

    def setup_method(self) -> None:
        """Reset the module-level cache before each test."""
        with _vram._last_prepared_model_lock:
            _vram._last_prepared_model_id = None

    @patch("src.settings.get_available_vram", side_effect=RuntimeError("GPU info unavailable"))
    @patch("src.settings.get_installed_models_with_sizes", return_value={"test-model:8b": 4.0})
    @patch("src.settings.get_model_info")
    def test_residency_check_exception_logged(
        self,
        _mock_info: MagicMock,
        _mock_installed: MagicMock,
        _mock_vram: MagicMock,
    ) -> None:
        """Exception in residency check is caught and logged at debug level."""
        svc = MagicMock()
        svc.settings = MagicMock(spec=Settings)
        svc.settings.vram_strategy = VramStrategy.SEQUENTIAL.value
        svc._loaded_models = set()
        svc._ollama_client = MagicMock()

        # Should not raise — the exception is caught inside prepare_model
        _vram.prepare_model(svc, "test-model:8b")
        assert "test-model:8b" in svc._loaded_models


class TestUnloadClearsLastPreparedCache:
    """Tests that unloading models clears the last-prepared cache."""

    def setup_method(self) -> None:
        """Reset the module-level cache before each test."""
        with _vram._last_prepared_model_lock:
            _vram._last_prepared_model_id = None

    def test_unload_clears_cache_for_evicted_model(self) -> None:
        """When a model is evicted, the last-prepared cache is cleared."""
        svc = MagicMock()
        svc._loaded_models = {"model-a:8b", "model-b:8b"}
        svc._ollama_client = MagicMock()

        # Simulate model-a was last prepared
        with _vram._last_prepared_model_lock:
            _vram._last_prepared_model_id = "model-a:8b"

        _vram.unload_all_except(svc, "model-b:8b")

        with _vram._last_prepared_model_lock:
            assert _vram._last_prepared_model_id is None

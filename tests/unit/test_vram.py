"""Tests for VRAM management functions."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.memory.mode_models import VramStrategy
from src.services.model_mode_service import _vram
from src.settings import Settings
from src.utils.exceptions import VRAMAllocationError


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
            _vram._last_prepared_models.clear()

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
            _vram._last_prepared_models.clear()

    @patch("src.settings.get_available_vram", side_effect=RuntimeError("GPU info unavailable"))
    @patch("src.settings.get_installed_models_with_sizes", return_value={"test-model:8b": 4.0})
    @patch("src.settings.get_model_info")
    def test_residency_check_expected_exception_logged_at_debug(
        self,
        _mock_info: MagicMock,
        _mock_installed: MagicMock,
        _mock_vram: MagicMock,
        caplog,
    ) -> None:
        """Expected exceptions (RuntimeError, etc.) in residency check are logged at DEBUG."""
        svc = MagicMock()
        svc.settings = MagicMock(spec=Settings)
        svc.settings.vram_strategy = VramStrategy.SEQUENTIAL.value
        svc._loaded_models = set()
        svc._ollama_client = MagicMock()

        with caplog.at_level(logging.DEBUG, logger="src.services.model_mode_service._vram"):
            _vram.prepare_model(svc, "test-model:8b")

        assert "test-model:8b" in svc._loaded_models
        assert any(
            "Could not check GPU residency" in r.message and r.levelno == logging.DEBUG
            for r in caplog.records
        )

    @patch(
        "src.settings.get_available_vram",
        side_effect=MemoryError("unexpected"),
    )
    @patch("src.settings.get_installed_models_with_sizes", return_value={"test-model:8b": 4.0})
    @patch("src.settings.get_model_info")
    def test_residency_check_unexpected_exception_logged_at_warning(
        self,
        _mock_info: MagicMock,
        _mock_installed: MagicMock,
        _mock_vram: MagicMock,
        caplog,
    ) -> None:
        """Unexpected exceptions in residency check are logged at WARNING and model is NOT tracked."""
        svc = MagicMock()
        svc.settings = MagicMock(spec=Settings)
        svc.settings.vram_strategy = VramStrategy.SEQUENTIAL.value
        svc._loaded_models = set()
        svc._ollama_client = MagicMock()

        with caplog.at_level(logging.DEBUG, logger="src.services.model_mode_service._vram"):
            _vram.prepare_model(svc, "test-model:8b")

        # Model should NOT be added to _loaded_models after unexpected error
        assert "test-model:8b" not in svc._loaded_models
        assert any(
            "Unexpected error checking GPU residency" in r.message and r.levelno == logging.WARNING
            for r in caplog.records
        )

    @patch("src.settings.get_available_vram", return_value=6.0)
    @patch("src.settings.get_installed_models_with_sizes", return_value={"big-model:70b": 20.0})
    @patch("src.settings.get_model_info")
    def test_residency_below_threshold_raises_vram_error(
        self,
        _mock_info: MagicMock,
        _mock_installed: MagicMock,
        _mock_vram: MagicMock,
    ) -> None:
        """GPU residency below MIN_GPU_RESIDENCY raises VRAMAllocationError."""
        svc = MagicMock()
        svc.settings = MagicMock(spec=Settings)
        svc.settings.vram_strategy = VramStrategy.SEQUENTIAL.value
        svc._loaded_models = set()
        svc._ollama_client = MagicMock()

        with pytest.raises(VRAMAllocationError) as exc_info:
            _vram.prepare_model(svc, "big-model:70b")

        assert exc_info.value.model_id == "big-model:70b"
        assert "big-model:70b" not in svc._loaded_models


class TestTwoSlotRoleCache:
    """Tests for the two-slot role-based prepare cache (creator/judge)."""

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
            _vram._last_prepared_models.clear()

    @patch("src.settings.get_available_vram", return_value=24.0)
    @patch(
        "src.settings.get_installed_models_with_sizes",
        return_value={"model-a:8b": 4.0, "model-b:8b": 5.0},
    )
    @patch("src.settings.get_model_info")
    def test_role_specific_cache_entries(
        self,
        _mock_info: MagicMock,
        _mock_installed: MagicMock,
        _mock_vram: MagicMock,
    ) -> None:
        """Each role stores its cache entry under its own key (not a single slot)."""
        svc = self._make_service()

        _vram.prepare_model(svc, "model-a:8b", role="creator")

        # After preparing creator, its cache entry exists
        with _vram._last_prepared_model_lock:
            assert _vram._last_prepared_models["creator"] == ("model-a:8b", "sequential")

        # Preparing judge clears entire cache (sequential strategy calls unload_all_except)
        # then re-caches only the judge entry
        _vram.prepare_model(svc, "model-b:8b", role="judge")
        with _vram._last_prepared_model_lock:
            assert _vram._last_prepared_models["judge"] == ("model-b:8b", "sequential")

    @patch("src.settings.get_available_vram", return_value=24.0)
    @patch(
        "src.settings.get_installed_models_with_sizes",
        return_value={"model-a:8b": 4.0},
    )
    @patch("src.settings.get_model_info")
    def test_same_role_same_model_short_circuits(
        self,
        _mock_info: MagicMock,
        _mock_installed: MagicMock,
        _mock_vram: MagicMock,
    ) -> None:
        """Consecutive prepare for the same role+model skips full preparation."""
        svc = self._make_service()

        # First call — full preparation
        _vram.prepare_model(svc, "model-a:8b", role="creator")

        # Second call to same role+model — should short-circuit
        with patch.object(_vram, "unload_all_except") as mock_unload:
            _vram.prepare_model(svc, "model-a:8b", role="creator")
            mock_unload.assert_not_called()


class TestUnloadPartialFailure:
    """Tests for unload_all_except when some evictions fail."""

    def test_partial_failure_keeps_failed_model(self) -> None:
        """Models that fail to unload remain in _loaded_models."""
        import ollama

        svc = MagicMock()
        svc._loaded_models = {"model-a:8b", "model-b:8b", "model-c:8b"}
        svc._ollama_client = MagicMock()

        def side_effect(model: str, keep_alive: int) -> None:
            """Simulate model-b failing to unload."""
            if model == "model-b:8b":
                raise ollama.ResponseError("not found")

        svc._ollama_client.generate.side_effect = side_effect

        _vram.unload_all_except(svc, "model-c:8b")

        # model-b failed to unload — must remain tracked
        assert "model-b:8b" in svc._loaded_models
        # model-a succeeded — must be removed
        assert "model-a:8b" not in svc._loaded_models
        # model-c was kept — must remain
        assert "model-c:8b" in svc._loaded_models


class TestUnloadClearsLastPreparedCache:
    """Tests that unloading models clears the last-prepared cache."""

    def setup_method(self) -> None:
        """Reset the module-level cache before each test."""
        with _vram._last_prepared_model_lock:
            _vram._last_prepared_models.clear()

    def test_unload_clears_cache_for_evicted_model(self) -> None:
        """When a model is evicted, the last-prepared cache is cleared."""
        svc = MagicMock()
        svc._loaded_models = {"model-a:8b", "model-b:8b"}
        svc._ollama_client = MagicMock()

        # Simulate model-a was last prepared for "creator" role
        with _vram._last_prepared_model_lock:
            _vram._last_prepared_models["creator"] = ("model-a:8b", "sequential")

        _vram.unload_all_except(svc, "model-b:8b")

        with _vram._last_prepared_model_lock:
            assert _vram._last_prepared_models == {}

    def test_unload_clears_cache_unconditionally(self) -> None:
        """Cache is always cleared when models are evicted, even for the kept model."""
        svc = MagicMock()
        svc._loaded_models = {"model-a:8b", "model-b:8b"}
        svc._ollama_client = MagicMock()

        # Simulate model-b was last prepared — it's also the kept model
        with _vram._last_prepared_model_lock:
            _vram._last_prepared_models["judge"] = ("model-b:8b", "sequential")

        _vram.unload_all_except(svc, "model-b:8b")

        # Cache should be cleared after successful eviction of model-a
        with _vram._last_prepared_model_lock:
            assert _vram._last_prepared_models == {}

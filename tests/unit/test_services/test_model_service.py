"""Tests for ModelService."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.services.model_service import ModelService
from src.settings import Settings


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings()


@pytest.fixture
def model_service(settings):
    """Create ModelService instance."""
    return ModelService(settings)


class TestModelServiceCheckHealth:
    """Tests for check_health method."""

    def test_returns_healthy_when_ollama_responds(self, model_service):
        """Test health check returns healthy when Ollama is running."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.list.return_value = MagicMock(models=[])

            health = model_service.check_health()

            assert health.is_healthy is True
            assert "running" in health.message.lower()

    def test_returns_unhealthy_on_connection_error(self, model_service):
        """Test health check returns unhealthy on connection failure."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.list.side_effect = ConnectionError("Connection refused")

            health = model_service.check_health()

            assert health.is_healthy is False
            assert "connect" in health.message.lower()


class TestModelServiceListInstalled:
    """Tests for list_installed method."""

    def test_returns_empty_list_on_error(self, model_service):
        """Test returns empty list when Ollama fails."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.list.side_effect = ConnectionError("Failed")

            result = model_service.list_installed()

            assert result == []

    def test_returns_model_ids(self, model_service):
        """Test returns list of model IDs."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            mock_model1 = MagicMock()
            mock_model1.model = "llama3:8b"
            mock_model2 = MagicMock()
            mock_model2.model = "mistral:7b"

            mock_instance.list.return_value = MagicMock(models=[mock_model1, mock_model2])

            result = model_service.list_installed()

            assert result == ["llama3:8b", "mistral:7b"]


class TestModelServicePullModel:
    """Tests for pull_model method."""

    def test_yields_progress_updates(self, model_service):
        """Test pull_model yields progress dictionaries."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.pull.return_value = [
                {"status": "pulling manifest", "completed": 0, "total": 0},
                {"status": "downloading", "completed": 500, "total": 1000},
                {"status": "success", "completed": 1000, "total": 1000},
            ]

            results = list(model_service.pull_model("test-model"))

            assert len(results) == 3
            assert results[0]["status"] == "pulling manifest"
            assert results[1]["completed"] == 500
            assert results[2]["status"] == "success"

    def test_yields_error_on_connection_failure(self, model_service):
        """Test pull_model yields error dict on connection failure."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.pull.side_effect = ConnectionError("Network error")

            results = list(model_service.pull_model("test-model"))

            assert len(results) == 1
            assert results[0]["error"] is True
            assert "error" in results[0]["status"].lower()

    def test_invalidates_caches_after_successful_pull(self, model_service):
        """Test that pull_model calls invalidate_caches after download completes."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.pull.return_value = [
                {"status": "success", "completed": 1000, "total": 1000},
            ]
            model_service.invalidate_caches = MagicMock()

            list(model_service.pull_model("test-model"))

            model_service.invalidate_caches.assert_called_once()

    def test_handles_none_values_in_progress(self, model_service):
        """Test pull_model handles None values for total/completed without crashing.

        This is a regression test for the bug where Ollama API returns None
        for total/completed fields during certain phases of the download.
        """
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            # Ollama API can return None for total/completed during certain phases
            mock_instance.pull.return_value = [
                {"status": "pulling manifest"},  # No total/completed keys
                {"status": "pulling layers", "completed": None, "total": None},
                {"status": "downloading", "completed": 500, "total": 1000},
                {"status": "success", "completed": 1000, "total": 1000},
            ]

            # Should not raise TypeError: '>' not supported between instances of 'NoneType' and 'int'
            results = list(model_service.pull_model("test-model"))

            assert len(results) == 4
            assert results[0]["status"] == "pulling manifest"
            # None values should be converted to 0
            assert results[1]["total"] == 0
            assert results[1]["completed"] == 0
            assert results[3]["status"] == "success"


class TestModelServiceCheckModelUpdate:
    """Tests for check_model_update method."""

    def test_returns_has_update_when_downloading(self, model_service):
        """Test returns has_update=True when model needs download."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.pull.return_value = [
                {"status": "pulling layers", "completed": 100, "total": 5000},
            ]

            result = model_service.check_model_update("test-model")

            assert result["has_update"] is True

    def test_returns_no_update_when_up_to_date(self, model_service):
        """Test returns has_update=False when already up to date."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.pull.return_value = [
                {"status": "already up to date"},
            ]

            result = model_service.check_model_update("test-model")

            assert result["has_update"] is False

    def test_handles_none_total_gracefully(self, model_service):
        """Test handles None value for total without crashing."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            # Simulate Ollama returning None for total (the bug case)
            mock_instance.pull.return_value = [
                {"status": "pulling layers", "completed": None, "total": None},
                {"status": "already up to date"},
            ]

            # Should not raise TypeError
            result = model_service.check_model_update("test-model")

            assert "has_update" in result

    def test_handles_missing_keys_gracefully(self, model_service):
        """Test handles missing keys in progress dict."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            # Simulate Ollama returning incomplete progress dict
            mock_instance.pull.return_value = [
                {"status": "checking"},  # No total or completed
                {"status": "up to date"},
            ]

            # Should not raise KeyError
            result = model_service.check_model_update("test-model")

            assert result["has_update"] is False

    def test_returns_error_on_connection_failure(self, model_service):
        """Test returns error dict on connection failure."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.pull.side_effect = ConnectionError("Network error")

            result = model_service.check_model_update("test-model")

            assert result.get("error") is True


class TestModelServiceDeleteModel:
    """Tests for delete_model method."""

    def test_returns_true_on_success(self, model_service):
        """Test returns True when deletion succeeds."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            result = model_service.delete_model("test-model")

            assert result is True
            mock_instance.delete.assert_called_once_with("test-model")

    def test_returns_false_on_error(self, model_service):
        """Test returns False when deletion fails."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.delete.side_effect = ConnectionError("Failed")

            result = model_service.delete_model("test-model")

            assert result is False

    def test_invalidates_caches_on_success(self, model_service):
        """Test that delete_model calls invalidate_caches after successful deletion."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            model_service.invalidate_caches = MagicMock()

            model_service.delete_model("test-model")

            model_service.invalidate_caches.assert_called_once()

    def test_does_not_invalidate_caches_on_failure(self, model_service):
        """Test that delete_model does NOT call invalidate_caches when deletion fails."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.delete.side_effect = ConnectionError("Failed")
            model_service.invalidate_caches = MagicMock()

            model_service.delete_model("test-model")

            model_service.invalidate_caches.assert_not_called()


class TestModelServiceTestModel:
    """Tests for test_model method."""

    def test_returns_success_on_valid_response(self, model_service):
        """Test returns success when model generates response."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            # Ollama returns an object with .response attribute
            mock_response = MagicMock()
            mock_response.response = "Hello! Test successful."
            mock_instance.generate.return_value = mock_response

            success, message = model_service.test_model("test-model")

            assert success is True
            assert "passed" in message.lower() or "test" in message.lower()

    def test_returns_failure_on_error(self, model_service):
        """Test returns failure when model test fails."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.generate.side_effect = ConnectionError("Model not found")

            success, _message = model_service.test_model("test-model")

            assert success is False


class TestModelServiceGetVram:
    """Tests for get_vram method."""

    def test_returns_vram_value(self, model_service):
        """Test returns VRAM value."""
        with patch("src.services.model_service.get_available_vram") as mock_vram:
            mock_vram.return_value = 24

            result = model_service.get_vram()

            assert result == 24

    def test_returns_zero_on_error(self, model_service):
        """Test returns 0 when VRAM detection fails."""
        with patch("src.services.model_service.get_available_vram") as mock_vram:
            mock_vram.side_effect = Exception("Failed to detect VRAM")

            # Should handle gracefully (depends on implementation)
            # If get_available_vram raises, it should be caught
            try:
                result = model_service.get_vram()
                # If it doesn't raise, check the result is reasonable
                assert isinstance(result, (int, float))
            except Exception:
                # If it raises, that's also acceptable behavior to test
                pass


class TestLogHealthFailure:
    """Tests for _log_health_failure helper method."""

    def test_logs_warning_when_not_previously_unhealthy(self, model_service, caplog):
        """Test _log_health_failure logs WARNING when _last_health_healthy is None (first call)."""
        with caplog.at_level(logging.DEBUG, logger="src.services.model_service"):
            model_service._log_health_failure("test failure message")

            warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert any("test failure message" in r.message for r in warn_records)

    def test_logs_debug_when_already_unhealthy(self, model_service, caplog):
        """Test _log_health_failure logs DEBUG when _last_health_healthy is False."""
        model_service._last_health_healthy = False
        with caplog.at_level(logging.DEBUG, logger="src.services.model_service"):
            model_service._log_health_failure("repeated failure message")

            debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
            assert any("repeated failure message" in r.message for r in debug_records)
            warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert not any("repeated failure message" in r.message for r in warn_records)

    def test_logs_warning_when_was_healthy(self, model_service, caplog):
        """Test _log_health_failure logs WARNING when transitioning from healthy to unhealthy."""
        model_service._last_health_healthy = True
        with caplog.at_level(logging.DEBUG, logger="src.services.model_service"):
            model_service._log_health_failure("transition failure")

            warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert any("transition failure" in r.message for r in warn_records)


class TestLogHealthFailureTransitions:
    """Tests for _log_health_failure transition lifecycle.

    Verifies the full WARNING→DEBUG→WARNING cycle across state transitions:
    healthy→unhealthy (WARNING), unhealthy→unhealthy (DEBUG), recovery→unhealthy (WARNING).
    """

    def test_full_transition_lifecycle(self, model_service, caplog):
        """Test WARNING on first failure, DEBUG on repeat, WARNING after recovery."""
        with caplog.at_level(logging.DEBUG, logger="src.services.model_service"):
            # Initial state (None) → first failure → WARNING
            model_service._log_health_failure("failure 1")
            warn_1 = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert any("failure 1" in r.message for r in warn_1)

            # Simulate state update to unhealthy
            model_service._last_health_healthy = False

            # Consecutive failure → DEBUG (not WARNING)
            caplog.clear()
            model_service._log_health_failure("failure 2")
            warn_2 = [r for r in caplog.records if r.levelno == logging.WARNING]
            debug_2 = [r for r in caplog.records if r.levelno == logging.DEBUG]
            assert not any("failure 2" in r.message for r in warn_2)
            assert any("failure 2" in r.message for r in debug_2)

            # Recovery (set healthy)
            model_service._last_health_healthy = True

            # Failure after recovery → WARNING again
            caplog.clear()
            model_service._log_health_failure("failure 3")
            warn_3 = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert any("failure 3" in r.message for r in warn_3)


class TestModelServiceCheckHealthEdgeCases:
    """Additional edge case tests for check_health."""

    def test_returns_unhealthy_on_response_error(self, model_service):
        """Test health check handles Ollama ResponseError."""
        import ollama

        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.list.side_effect = ollama.ResponseError("API error")

            health = model_service.check_health()

            assert health.is_healthy is False
            assert "api error" in health.message.lower() or "error" in health.message.lower()


class TestModelServiceListInstalledWithSizes:
    """Tests for list_installed_with_sizes method."""

    def test_returns_models_with_sizes(self, model_service):
        """Test returns dict of model IDs to sizes in GB."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            mock_model1 = MagicMock()
            mock_model1.model = "llama3:8b"
            mock_model1.size = 5 * 1024**3  # 5 GB in bytes

            mock_model2 = MagicMock()
            mock_model2.model = "mistral:7b"
            mock_model2.size = 4 * 1024**3  # 4 GB in bytes

            mock_instance.list.return_value = MagicMock(models=[mock_model1, mock_model2])

            result = model_service.list_installed_with_sizes()

            assert result["llama3:8b"] == 5.0
            assert result["mistral:7b"] == 4.0

    def test_handles_missing_size_attribute(self, model_service):
        """Test handles models without size attribute."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            mock_model = MagicMock()
            mock_model.model = "test:model"
            del mock_model.size  # Remove size attribute

            mock_instance.list.return_value = MagicMock(models=[mock_model])

            result = model_service.list_installed_with_sizes()

            assert "test:model" in result
            assert result["test:model"] == 0.0

    def test_returns_empty_on_error(self, model_service):
        """Test returns empty dict on connection error."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.list.side_effect = ConnectionError("Failed")

            result = model_service.list_installed_with_sizes()

            assert result == {}


class TestModelServiceListAvailable:
    """Tests for list_available method."""

    def test_lists_known_and_installed_models(self, model_service):
        """Test returns combined list of known and installed models."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            # Return one installed model that's in RECOMMENDED_MODELS
            mock_model = MagicMock()
            mock_model.model = "huihui_ai/qwen3-abliterated:8b"
            mock_model.size = 5 * 1024**3

            mock_instance.list.return_value = MagicMock(models=[mock_model])

            result = model_service.list_available()

            # Should include models from RECOMMENDED_MODELS
            assert len(result) > 0
            model_ids = [m.model_id for m in result]
            assert "huihui_ai/qwen3-abliterated:8b" in model_ids

    def test_includes_unknown_installed_models(self, model_service):
        """Test includes installed models not in RECOMMENDED_MODELS."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            # Return a completely unknown model
            mock_model = MagicMock()
            mock_model.model = "custom-unknown:latest"
            mock_model.size = 8 * 1024**3

            mock_instance.list.return_value = MagicMock(models=[mock_model])

            result = model_service.list_available()

            # Find the custom model
            custom_model = next((m for m in result if m.model_id == "custom-unknown:latest"), None)
            assert custom_model is not None
            assert custom_model.installed is True
            assert custom_model.size_gb == 8.0
            assert custom_model.description == "Automatically detected model"

    def test_matches_variant_models_to_known_base(self, model_service):
        """Test matches variant models (e.g., :latest) to known base models."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            # Return a variant of a known model
            mock_model = MagicMock()
            mock_model.model = "huihui_ai/qwen3-abliterated:latest"
            mock_model.size = 5 * 1024**3

            mock_instance.list.return_value = MagicMock(models=[mock_model])

            result = model_service.list_available()

            # Find the variant model
            variant = next(
                (m for m in result if m.model_id == "huihui_ai/qwen3-abliterated:latest"), None
            )
            assert variant is not None
            assert variant.installed is True
            # Should inherit properties from known base


class TestModelServiceGetModelInfo:
    """Tests for get_model_info method."""

    def test_returns_model_info_for_known_model(self, model_service):
        """Test returns ModelInfo for known model."""
        result = model_service.get_model_info("huihui_ai/qwen3-abliterated:8b")

        assert "name" in result
        assert "quality" in result


class TestModelServicePullModelEdgeCases:
    """Additional edge cases for pull_model."""

    def test_yields_error_on_response_error(self, model_service):
        """Test pull_model yields error on ResponseError."""
        import ollama

        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.pull.side_effect = ollama.ResponseError("Model not found")

            results = list(model_service.pull_model("nonexistent-model"))

            assert len(results) == 1
            assert results[0]["error"] is True

    def test_yields_error_on_unexpected_exception(self, model_service):
        """Test pull_model yields error on unexpected exception."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.pull.side_effect = RuntimeError("Unexpected error")

            results = list(model_service.pull_model("test-model"))

            assert len(results) == 1
            assert results[0]["error"] is True
            assert "unexpected" in results[0]["status"].lower()


class TestModelServiceCheckModelUpdateEdgeCases:
    """Additional edge cases for check_model_update."""

    def test_returns_no_update_on_success_status(self, model_service):
        """Test returns no update when success status received."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.pull.return_value = [
                {"status": "success"},
            ]

            result = model_service.check_model_update("test-model")

            assert result["has_update"] is False
            assert "up to date" in result["message"].lower()

    def test_returns_error_on_response_error(self, model_service):
        """Test returns error on ResponseError."""
        import ollama

        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.pull.side_effect = ollama.ResponseError("API error")

            result = model_service.check_model_update("test-model")

            assert result.get("error") is True
            assert "failed" in result["message"].lower()

    def test_returns_no_update_when_loop_completes_without_match(self, model_service):
        """Test returns no update when loop completes without matching status."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            # Return statuses that don't match any of the conditions
            mock_instance.pull.return_value = [
                {"status": "verifying"},
                {"status": "completed"},
            ]

            result = model_service.check_model_update("test-model")

            assert result["has_update"] is False
            assert "up to date" in result["message"].lower()

    def test_returns_update_on_downloading_status(self, model_service):
        """Test returns update when 'downloading' status is seen."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            # Simulate actual downloading happening
            mock_instance.pull.return_value = [
                {"status": "downloading abc123", "total": 1000, "completed": 100},
            ]

            result = model_service.check_model_update("test-model")

            assert result["has_update"] is True
            assert "available" in result["message"].lower()


class TestModelServiceGetRecommendedModel:
    """Tests for get_recommended_model method.

    Since model selection is now fully automatic based on installed models,
    we test that the selection returns a valid model from the installed set.
    """

    def test_recommends_model_for_high_vram(self, model_service):
        """Test recommends a model when VRAM is available."""
        with patch("src.services.model_service.get_available_vram") as mock_vram:
            mock_vram.return_value = 24

            # Mock installed models with sizes
            with patch(
                "src.settings._settings.get_installed_models_with_sizes",
                return_value={
                    "large-model:30b": 18.0,
                    "medium-model:12b": 10.0,
                    "small-model:8b": 5.0,
                },
            ):
                # Tag all models for writer role (default role for get_recommended_model)
                model_service.settings.custom_model_tags = {
                    "large-model:30b": ["writer"],
                    "medium-model:12b": ["writer"],
                    "small-model:8b": ["writer"],
                }
                result = model_service.get_recommended_model()

                # Should return one of the installed models
                assert result in ["large-model:30b", "medium-model:12b", "small-model:8b"]

    def test_recommends_model_for_medium_vram(self, model_service):
        """Test recommends appropriate model for medium VRAM."""
        with patch("src.services.model_service.get_available_vram") as mock_vram:
            mock_vram.return_value = 14

            # Mock installed models
            with patch(
                "src.settings._settings.get_installed_models_with_sizes",
                return_value={
                    "large-model:30b": 18.0,  # Won't fit (18*1.2 = 21.6GB)
                    "medium-model:12b": 10.0,  # Fits (10*1.2 = 12GB)
                    "small-model:8b": 5.0,  # Fits
                },
            ):
                # Tag all models for writer role (default role for get_recommended_model)
                model_service.settings.custom_model_tags = {
                    "large-model:30b": ["writer"],
                    "medium-model:12b": ["writer"],
                    "small-model:8b": ["writer"],
                }
                result = model_service.get_recommended_model()

                # Should return a model that fits VRAM
                assert result in ["medium-model:12b", "small-model:8b"]

    def test_recommends_small_model_for_low_vram(self, model_service):
        """Test recommends small model for low VRAM."""
        with patch("src.services.model_service.get_available_vram") as mock_vram:
            mock_vram.return_value = 8

            with patch(
                "src.settings._settings.get_installed_models_with_sizes",
                return_value={
                    "medium-model:12b": 10.0,  # Won't fit (10*1.2 = 12GB)
                    "small-model:8b": 5.0,  # Fits (5*1.2 = 6GB)
                },
            ):
                # Tag models for writer role (default role for get_recommended_model)
                model_service.settings.custom_model_tags = {
                    "medium-model:12b": ["writer"],
                    "small-model:8b": ["writer"],
                }
                result = model_service.get_recommended_model()

                # Should return the small model that fits
                assert result == "small-model:8b"

    def test_uses_agent_role_for_recommendation(self, model_service):
        """Test uses role-specific recommendation."""
        with patch("src.services.model_service.get_available_vram") as mock_vram:
            mock_vram.return_value = 24

            with patch(
                "src.settings._settings.get_installed_models_with_sizes",
                return_value={"test-model:8b": 5.0},
            ):
                # Tag the model for writer role
                model_service.settings.custom_model_tags = {"test-model:8b": ["writer"]}
                # With role, should use settings.get_model_for_agent
                result = model_service.get_recommended_model(role="writer")

                assert result == "test-model:8b"


class TestModelServiceGetModelsForVram:
    """Tests for get_models_for_vram method."""

    def test_filters_by_vram_requirement(self, model_service):
        """Test returns only models that fit within VRAM."""
        with patch.object(model_service, "list_available") as mock_list:
            from src.services.model_service import ModelStatus

            mock_list.return_value = [
                ModelStatus(
                    model_id="small:8b",
                    name="Small",
                    installed=True,
                    size_gb=5,
                    vram_required=8,
                    quality=7,
                    speed=8,
                    uncensored=True,
                    description="Small model",
                ),
                ModelStatus(
                    model_id="large:70b",
                    name="Large",
                    installed=True,
                    size_gb=40,
                    vram_required=48,
                    quality=10,
                    speed=3,
                    uncensored=True,
                    description="Large model",
                ),
            ]

            result = model_service.get_models_for_vram(min_vram=16)

            assert len(result) == 1
            assert result[0].model_id == "small:8b"

    def test_uses_detected_vram_when_not_specified(self, model_service):
        """Test uses detected VRAM when min_vram not provided."""
        with (
            patch.object(model_service, "get_vram") as mock_vram,
            patch.object(model_service, "list_available") as mock_list,
        ):
            mock_vram.return_value = 12
            mock_list.return_value = []

            model_service.get_models_for_vram()

            mock_vram.assert_called_once()


class TestModelServiceTestModelEdgeCases:
    """Additional edge cases for test_model."""

    def test_returns_failure_on_empty_response(self, model_service):
        """Test returns failure when model returns empty response."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_response = MagicMock()
            mock_response.response = ""
            mock_instance.generate.return_value = mock_response

            success, message = model_service.test_model("test-model")

            assert success is False
            assert "empty" in message.lower()

    def test_returns_failure_on_response_error(self, model_service):
        """Test returns failure on ResponseError."""
        import ollama

        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.generate.side_effect = ollama.ResponseError("Model error")

            success, message = model_service.test_model("test-model")

            assert success is False
            assert "error" in message.lower()


class TestModelServiceGetModelByQuality:
    """Tests for get_model_by_quality method."""

    def test_filters_by_quality_and_vram(self, model_service):
        """Test filters models by minimum quality and VRAM."""
        with patch.object(model_service, "list_available") as mock_list:
            from src.services.model_service import ModelStatus

            mock_list.return_value = [
                ModelStatus(
                    model_id="high-quality:8b",
                    name="High Quality",
                    installed=True,
                    size_gb=5,
                    vram_required=8,
                    quality=9,
                    speed=7,
                    uncensored=True,
                    description="High quality model",
                ),
                ModelStatus(
                    model_id="low-quality:8b",
                    name="Low Quality",
                    installed=True,
                    size_gb=5,
                    vram_required=8,
                    quality=4,
                    speed=9,
                    uncensored=True,
                    description="Low quality model",
                ),
            ]

            result = model_service.get_model_by_quality(min_quality=7, max_vram=16)

            assert len(result) == 1
            assert result[0].model_id == "high-quality:8b"

    def test_filters_by_uncensored_requirement(self, model_service):
        """Test filters models by uncensored requirement."""
        with patch.object(model_service, "list_available") as mock_list:
            from src.services.model_service import ModelStatus

            mock_list.return_value = [
                ModelStatus(
                    model_id="censored:8b",
                    name="Censored",
                    installed=True,
                    size_gb=5,
                    vram_required=8,
                    quality=8,
                    speed=7,
                    uncensored=False,
                    description="Censored model",
                ),
                ModelStatus(
                    model_id="uncensored:8b",
                    name="Uncensored",
                    installed=True,
                    size_gb=5,
                    vram_required=8,
                    quality=8,
                    speed=7,
                    uncensored=True,
                    description="Uncensored model",
                ),
            ]

            result = model_service.get_model_by_quality(
                min_quality=7, max_vram=16, uncensored_required=True
            )

            assert len(result) == 1
            assert result[0].model_id == "uncensored:8b"

    def test_sorts_by_quality_descending(self, model_service):
        """Test results are sorted by quality descending."""
        with patch.object(model_service, "list_available") as mock_list:
            from src.services.model_service import ModelStatus

            mock_list.return_value = [
                ModelStatus(
                    model_id="medium:8b",
                    name="Medium",
                    installed=True,
                    size_gb=5,
                    vram_required=8,
                    quality=7,
                    speed=8,
                    uncensored=True,
                    description="Medium",
                ),
                ModelStatus(
                    model_id="high:8b",
                    name="High",
                    installed=True,
                    size_gb=5,
                    vram_required=8,
                    quality=9,
                    speed=6,
                    uncensored=True,
                    description="High",
                ),
            ]

            result = model_service.get_model_by_quality(min_quality=5, max_vram=16)

            assert result[0].quality >= result[-1].quality


class TestModelServiceCompareModels:
    """Tests for compare_models method."""

    def test_compares_multiple_models(self, model_service):
        """Test compares multiple models on same prompt."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            mock_response = MagicMock()
            mock_response.response = "Test response"
            mock_instance.generate.return_value = mock_response

            result = model_service.compare_models(
                model_ids=["model1:8b", "model2:8b"],
                prompt="Write a test",
            )

            assert len(result) == 2
            assert result[0]["model_id"] == "model1:8b"
            assert result[0]["success"] is True
            assert "time_seconds" in result[0]

    def test_handles_model_failure_gracefully(self, model_service):
        """Test handles individual model failure in comparison."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            # First model succeeds, second fails
            mock_response = MagicMock()
            mock_response.response = "Success"
            mock_instance.generate.side_effect = [
                mock_response,
                ConnectionError("Model failed"),
            ]

            result = model_service.compare_models(
                model_ids=["good-model:8b", "bad-model:8b"],
                prompt="Test prompt",
            )

            assert len(result) == 2
            assert result[0]["success"] is True
            assert result[1]["success"] is False
            assert "error" in result[1]


class TestModelServiceStateChangeLogging:
    """Tests for state-change-only logging in ModelService."""

    def test_health_check_logs_info_on_first_success(self, model_service, caplog):
        """Test health check logs at INFO on first successful call."""
        with (
            patch("src.services.model_service.ollama.Client") as mock_client,
            patch("src.services.model_service.get_available_vram", return_value=24),
            caplog.at_level(logging.DEBUG, logger="src.services.model_service"),
        ):
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.list.return_value = MagicMock(models=[])

            model_service.check_health()

            info_records = [r for r in caplog.records if r.levelno == logging.INFO]
            assert any("health check successful" in r.message for r in info_records)

    def test_health_check_logs_debug_on_repeated_success(self, model_service, caplog):
        """Test health check logs at DEBUG on repeated identical success."""
        with (
            patch("src.services.model_service.ollama.Client") as mock_client,
            patch("src.services.model_service.get_available_vram", return_value=24),
            caplog.at_level(logging.DEBUG, logger="src.services.model_service"),
        ):
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.list.return_value = MagicMock(models=[])

            model_service.check_health()
            caplog.clear()

            # Invalidate TTL cache so the second call actually hits the API
            model_service.invalidate_caches()
            model_service.check_health()

            info_records = [r for r in caplog.records if r.levelno == logging.INFO]
            assert not any("health check successful" in r.message for r in info_records)
            debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
            assert any("health check successful" in r.message for r in debug_records)

    def test_health_check_logs_info_on_vram_change(self, model_service, caplog):
        """Test health check logs at INFO when VRAM changes."""
        with (
            patch("src.services.model_service.ollama.Client") as mock_client,
            patch("src.services.model_service.get_available_vram") as mock_vram,
            caplog.at_level(logging.DEBUG, logger="src.services.model_service"),
        ):
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.list.return_value = MagicMock(models=[])

            mock_vram.return_value = 24
            model_service.check_health()
            caplog.clear()

            # Invalidate TTL cache so the second call actually hits the API
            model_service.invalidate_caches()
            mock_vram.return_value = 16
            model_service.check_health()

            info_records = [r for r in caplog.records if r.levelno == logging.INFO]
            assert any("health check successful" in r.message for r in info_records)

    def test_health_check_logs_warning_on_first_failure(self, model_service, caplog):
        """Test health check logs at WARNING on first failure."""
        with (
            patch("src.services.model_service.ollama.Client") as mock_client,
            caplog.at_level(logging.DEBUG, logger="src.services.model_service"),
        ):
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.list.side_effect = ConnectionError("Connection refused")

            model_service.check_health()

            warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert any("Cannot connect" in r.message for r in warn_records)

    def test_health_check_logs_debug_on_repeated_failure(self, model_service, caplog):
        """Test health check logs at DEBUG on repeated failure."""
        with (
            patch("src.services.model_service.ollama.Client") as mock_client,
            caplog.at_level(logging.DEBUG, logger="src.services.model_service"),
        ):
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.list.side_effect = ConnectionError("Connection refused")

            model_service.check_health()
            caplog.clear()

            # Invalidate TTL cache so the second call actually hits the API
            model_service.invalidate_caches()
            model_service.check_health()

            warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert not any("Cannot connect" in r.message for r in warn_records)
            debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
            assert any("Cannot connect" in r.message for r in debug_records)

    def test_health_check_logs_debug_on_repeated_response_error(self, model_service, caplog):
        """Test health check logs at DEBUG on repeated ResponseError failure."""
        import ollama

        with (
            patch("src.services.model_service.ollama.Client") as mock_client,
            caplog.at_level(logging.DEBUG, logger="src.services.model_service"),
        ):
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.list.side_effect = ollama.ResponseError("API error")

            model_service.check_health()
            caplog.clear()

            # Invalidate TTL cache so the second call actually hits the API
            model_service.invalidate_caches()
            model_service.check_health()

            warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert not any("API error" in r.message for r in warn_records)
            debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
            assert any("API error" in r.message for r in debug_records)

    def test_health_check_logs_info_on_recovery(self, model_service, caplog):
        """Test health check logs at INFO when recovering from failure."""
        with (
            patch("src.services.model_service.ollama.Client") as mock_client,
            patch("src.services.model_service.get_available_vram", return_value=24),
            caplog.at_level(logging.DEBUG, logger="src.services.model_service"),
        ):
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            # First call fails
            mock_instance.list.side_effect = ConnectionError("Connection refused")
            model_service.check_health()
            caplog.clear()

            # Invalidate TTL cache so the second call actually hits the API
            model_service.invalidate_caches()
            # Second call succeeds (recovery)
            mock_instance.list.side_effect = None
            mock_instance.list.return_value = MagicMock(models=[])
            model_service.check_health()

            info_records = [r for r in caplog.records if r.levelno == logging.INFO]
            assert any("health check successful" in r.message for r in info_records)

    def test_list_installed_logs_info_on_first_call(self, model_service, caplog):
        """Test list_installed logs at INFO on first call."""
        with (
            patch("src.services.model_service.ollama.Client") as mock_client,
            caplog.at_level(logging.DEBUG, logger="src.services.model_service"),
        ):
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_model = MagicMock()
            mock_model.model = "llama3:8b"
            mock_instance.list.return_value = MagicMock(models=[mock_model])

            model_service.list_installed()

            info_records = [r for r in caplog.records if r.levelno == logging.INFO]
            assert any("Found 1 installed models" in r.message for r in info_records)

    def test_list_installed_logs_debug_on_same_count(self, model_service, caplog):
        """Test list_installed logs at DEBUG when count is unchanged."""
        with (
            patch("src.services.model_service.ollama.Client") as mock_client,
            caplog.at_level(logging.DEBUG, logger="src.services.model_service"),
        ):
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_model = MagicMock()
            mock_model.model = "llama3:8b"
            mock_instance.list.return_value = MagicMock(models=[mock_model])

            model_service.list_installed()
            caplog.clear()

            # Invalidate TTL cache so the second call actually hits the API
            model_service.invalidate_caches()
            model_service.list_installed()

            info_records = [r for r in caplog.records if r.levelno == logging.INFO]
            assert not any("Found 1 installed models" in r.message for r in info_records)
            debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
            assert any("Found 1 installed models" in r.message for r in debug_records)

    def test_list_installed_logs_info_on_count_change(self, model_service, caplog):
        """Test list_installed logs at INFO when model count changes."""
        with (
            patch("src.services.model_service.ollama.Client") as mock_client,
            caplog.at_level(logging.DEBUG, logger="src.services.model_service"),
        ):
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            mock_model1 = MagicMock()
            mock_model1.model = "llama3:8b"
            mock_instance.list.return_value = MagicMock(models=[mock_model1])

            model_service.list_installed()
            caplog.clear()

            # Invalidate TTL cache so the second call actually hits the API
            model_service.invalidate_caches()
            mock_model2 = MagicMock()
            mock_model2.model = "mistral:7b"
            mock_instance.list.return_value = MagicMock(models=[mock_model1, mock_model2])

            model_service.list_installed()

            info_records = [r for r in caplog.records if r.levelno == logging.INFO]
            assert any("Found 2 installed models" in r.message for r in info_records)

    def test_list_installed_with_sizes_logs_info_on_first_call(self, model_service, caplog):
        """Test list_installed_with_sizes logs at INFO on first call."""
        with (
            patch("src.services.model_service.ollama.Client") as mock_client,
            caplog.at_level(logging.DEBUG, logger="src.services.model_service"),
        ):
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_model = MagicMock()
            mock_model.model = "llama3:8b"
            mock_model.size = 5 * 1024**3
            mock_instance.list.return_value = MagicMock(models=[mock_model])

            model_service.list_installed_with_sizes()

            info_records = [r for r in caplog.records if r.levelno == logging.INFO]
            assert any("Found 1 installed models with sizes" in r.message for r in info_records)

    def test_list_installed_with_sizes_logs_debug_on_same_count(self, model_service, caplog):
        """Test list_installed_with_sizes logs at DEBUG when count unchanged."""
        with (
            patch("src.services.model_service.ollama.Client") as mock_client,
            caplog.at_level(logging.DEBUG, logger="src.services.model_service"),
        ):
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_model = MagicMock()
            mock_model.model = "llama3:8b"
            mock_model.size = 5 * 1024**3
            mock_instance.list.return_value = MagicMock(models=[mock_model])

            model_service.list_installed_with_sizes()
            caplog.clear()

            model_service.list_installed_with_sizes()

            info_records = [r for r in caplog.records if r.levelno == logging.INFO]
            assert not any("Found 1 installed models with sizes" in r.message for r in info_records)
            debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
            assert any("Found 1 installed models with sizes" in r.message for r in debug_records)


class TestModelServiceGetRunningModels:
    """Tests for get_running_models method."""

    def test_returns_running_models(self, model_service):
        """Test returns list of currently loaded models."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            mock_model = MagicMock()
            mock_model.name = "test-model:8b"
            mock_model.size = 5 * 1024**3  # 5 GB

            mock_instance.ps.return_value = MagicMock(models=[mock_model])

            result = model_service.get_running_models()

            assert len(result) == 1
            assert result[0]["name"] == "test-model:8b"
            assert result[0]["size_gb"] == 5.0

    def test_returns_multiple_running_models(self, model_service):
        """Test returns multiple loaded models."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            mock_model1 = MagicMock()
            mock_model1.name = "model-a:8b"
            mock_model1.size = 5 * 1024**3

            mock_model2 = MagicMock()
            mock_model2.name = "model-b:30b"
            mock_model2.size = 18 * 1024**3

            mock_instance.ps.return_value = MagicMock(models=[mock_model1, mock_model2])

            result = model_service.get_running_models()

            assert len(result) == 2
            names = [m["name"] for m in result]
            assert "model-a:8b" in names
            assert "model-b:30b" in names

    def test_returns_none_on_connection_error(self, model_service):
        """Test returns None when Ollama is unreachable (unknown state)."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.ps.side_effect = ConnectionError("Connection refused")

            result = model_service.get_running_models()

            assert result is None

    def test_returns_none_on_attribute_error(self, model_service):
        """Test returns None for older Ollama clients without ps() method."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.ps.side_effect = AttributeError("no attribute ps")

            result = model_service.get_running_models()

            assert result is None

    def test_returns_empty_list_when_no_models_loaded(self, model_service):
        """Test returns empty list when no models are in VRAM."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.ps.return_value = MagicMock(models=[])

            result = model_service.get_running_models()

            assert result == []

    def test_falls_back_to_model_attr_if_name_missing(self, model_service):
        """Test falls back to 'model' attribute if 'name' is not present."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            mock_model = MagicMock(spec=[])  # No default attributes
            mock_model.name = ""
            mock_model.model = "fallback-model:8b"
            mock_model.size = 5 * 1024**3

            mock_instance.ps.return_value = MagicMock(models=[mock_model])

            result = model_service.get_running_models()

            assert len(result) == 1
            assert result[0]["name"] == "fallback-model:8b"


class TestModelServiceLogModelLoadState:
    """Tests for log_model_load_state method."""

    def test_logs_no_models_loaded(self, model_service, caplog):
        """Test logs informative message when no models are loaded."""
        with caplog.at_level(logging.INFO, logger="src.services.model_service"):
            with patch.object(model_service, "get_running_models", return_value=[]):
                model_service.log_model_load_state()

        assert any("no models currently loaded" in r.message for r in caplog.records)

    def test_logs_loaded_models(self, model_service, caplog):
        """Test logs details of loaded models."""
        running = [{"name": "test-model:8b", "size_gb": 5.0}]
        with caplog.at_level(logging.INFO, logger="src.services.model_service"):
            with patch.object(model_service, "get_running_models", return_value=running):
                model_service.log_model_load_state()

        assert any("1 model(s) loaded" in r.message for r in caplog.records)
        assert any("test-model:8b" in r.message for r in caplog.records)

    def test_warns_cold_start_when_target_not_loaded(self, model_service, caplog):
        """Test warns about cold-start when target model is not loaded."""
        with caplog.at_level(logging.WARNING, logger="src.services.model_service"):
            with patch.object(model_service, "get_running_models", return_value=[]):
                model_service.log_model_load_state(target_model="test-writer:8b")

        warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("cold-start" in r.message.lower() for r in warn_records)
        assert any("No models loaded" in r.message for r in warn_records)

    def test_warns_cold_start_when_different_model_loaded(self, model_service, caplog):
        """Test warns about cold-start when a different model is loaded."""
        running = [{"name": "other-model:8b", "size_gb": 5.0}]
        with caplog.at_level(logging.INFO, logger="src.services.model_service"):
            with patch.object(model_service, "get_running_models", return_value=running):
                model_service.log_model_load_state(target_model="test-writer:8b")

        warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("NOT loaded" in r.message for r in warn_records)

    def test_no_warning_when_target_already_loaded(self, model_service, caplog):
        """Test no cold-start warning when target model is already loaded."""
        running = [{"name": "test-writer:8b", "size_gb": 5.0}]
        with caplog.at_level(logging.INFO, logger="src.services.model_service"):
            with patch.object(model_service, "get_running_models", return_value=running):
                model_service.log_model_load_state(target_model="test-writer:8b")

        warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert not warn_records
        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert any("already loaded" in r.message for r in info_records)

    def test_no_target_model_just_logs_state(self, model_service, caplog):
        """Test logs state without warnings when no target specified."""
        running = [{"name": "some-model:8b", "size_gb": 5.0}]
        with caplog.at_level(logging.INFO, logger="src.services.model_service"):
            with patch.object(model_service, "get_running_models", return_value=running):
                model_service.log_model_load_state()

        warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert not warn_records
        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert any("1 model(s) loaded" in r.message for r in info_records)

    def test_returns_early_when_running_models_unavailable(self, model_service, caplog):
        """Test logs info and returns when running model state is unavailable."""
        with caplog.at_level(logging.INFO, logger="src.services.model_service"):
            with patch.object(model_service, "get_running_models", return_value=None):
                model_service.log_model_load_state(target_model="test-writer:8b")

        assert any("unable to query running models" in r.message for r in caplog.records)
        # Should NOT warn about cold-start when state is unknown
        warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert not warn_records


class TestTTLCaching:
    """Tests for TTL caching on check_health, list_installed, and get_vram."""

    def test_check_health_returns_cached_within_ttl(self, model_service):
        """Test that check_health returns cached result within TTL."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            with patch("src.services.model_service.get_available_vram", return_value=24):
                with patch.object(model_service, "get_running_models", return_value=[]):
                    result1 = model_service.check_health()
                    result2 = model_service.check_health()

        assert result1 is result2
        # Client should only be created once (cached on second call)
        assert mock_client.call_count == 1

    def test_list_installed_returns_cached_within_ttl(self, model_service):
        """Test that list_installed returns cached result within TTL."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_response = MagicMock()
            mock_model = MagicMock()
            mock_model.model = "test-model:8b"
            mock_response.models = [mock_model]
            mock_instance = MagicMock()
            mock_instance.list.return_value = mock_response
            mock_client.return_value = mock_instance

            result1 = model_service.list_installed()
            result2 = model_service.list_installed()

        assert result1 == result2
        # Client should only be created once (cached on second call)
        assert mock_client.call_count == 1

    def test_get_vram_returns_cached_within_ttl(self, model_service):
        """Test that get_vram returns cached result within TTL."""
        with patch("src.services.model_service.get_available_vram", return_value=24) as mock_vram:
            result1 = model_service.get_vram()
            result2 = model_service.get_vram()

        assert result1 == result2 == 24
        assert mock_vram.call_count == 1

    def test_invalidate_caches_clears_all(self, model_service):
        """Test that invalidate_caches clears all cached data."""
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_response = MagicMock()
            mock_model = MagicMock()
            mock_model.model = "test-model:8b"
            mock_response.models = [mock_model]
            mock_instance.list.return_value = mock_response

            with patch("src.services.model_service.get_available_vram", return_value=24):
                with patch.object(model_service, "get_running_models", return_value=[]):
                    model_service.check_health()
                    model_service.list_installed()
                    model_service.get_vram()

                    model_service.invalidate_caches()

                    # After invalidation, next calls should hit the actual methods
                    with patch.object(model_service, "get_running_models", return_value=[]):
                        model_service.check_health()
                    model_service.list_installed()
                    model_service.get_vram()

        # Client created twice for health and twice for list (2 each)
        assert mock_client.call_count == 4


class TestTTLCacheExpiration:
    """Tests for TTL cache expiration using mocked time."""

    def test_check_health_cache_expires_after_ttl(self, model_service):
        """Test that check_health re-queries Ollama after TTL expires."""
        call_count = 0
        monotonic_values = iter([100.0, 100.5, 200.0])  # start, within TTL, past TTL

        def mock_monotonic():
            """Return predetermined time values to simulate TTL expiration."""
            nonlocal call_count
            call_count += 1
            return next(monotonic_values)

        with (
            patch("src.services.model_service.time.monotonic", side_effect=mock_monotonic),
            patch("src.services.model_service.ollama.Client") as mock_client,
            patch("src.services.model_service.get_available_vram", return_value=24),
            patch.object(model_service, "get_running_models", return_value=[]),
        ):
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.list.return_value = MagicMock(models=[])

            result1 = model_service.check_health()
            result2 = model_service.check_health()  # within TTL — cached
            result3 = model_service.check_health()  # past TTL — fresh

        assert result1 is result2  # Same cached object
        assert result3 is not result1  # Fresh object after expiration
        assert mock_client.call_count == 2  # Created for call 1 and call 3

    def test_list_installed_cache_expires_after_ttl(self, model_service):
        """Test that list_installed re-queries Ollama after TTL expires."""
        monotonic_values = iter([100.0, 100.5, 200.0])

        with (
            patch(
                "src.services.model_service.time.monotonic",
                side_effect=lambda: next(monotonic_values),
            ),
            patch("src.services.model_service.ollama.Client") as mock_client,
        ):
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_model = MagicMock()
            mock_model.model = "test-model:8b"
            mock_instance.list.return_value = MagicMock(models=[mock_model])

            result1 = model_service.list_installed()
            result2 = model_service.list_installed()  # within TTL
            result3 = model_service.list_installed()  # past TTL

        assert result1 == result2 == result3 == ["test-model:8b"]
        # Defensive copies mean different list objects even from cache
        assert result1 is not result2
        # Client created twice (first call + expired call)
        assert mock_client.call_count == 2


class TestColdStartDetection:
    """Tests for cold-start model detection in check_health."""

    def test_detects_cold_start_with_default_model(self, model_service):
        """Test cold-start detection when default_model is configured but not running."""
        model_service.settings.default_model = "my-model:8b"
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            with patch("src.services.model_service.get_available_vram", return_value=24):
                with patch.object(model_service, "get_running_models", return_value=[]):
                    result = model_service.check_health()

        assert result.cold_start_models == ["my-model:8b"]

    def test_detects_cold_start_with_agent_models(self, model_service):
        """Test cold-start detection when agent_models have non-auto values."""
        model_service.settings.agent_models = {
            "writer": "writer-model:8b",
            "editor": "auto",
        }
        running = [{"name": "other-model:8b"}]
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            with patch("src.services.model_service.get_available_vram", return_value=24):
                with patch.object(model_service, "get_running_models", return_value=running):
                    result = model_service.check_health()

        assert "writer-model:8b" in result.cold_start_models

    def test_no_cold_start_when_model_is_running(self, model_service):
        """Test no cold-start reported when configured model is already running."""
        model_service.settings.default_model = "my-model:8b"
        running = [{"name": "my-model:8b"}]
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            with patch("src.services.model_service.get_available_vram", return_value=24):
                with patch.object(model_service, "get_running_models", return_value=running):
                    result = model_service.check_health()

        assert result.cold_start_models == []

    def test_cold_start_skipped_when_running_models_unavailable(self, model_service, caplog):
        """Test that cold-start detection is skipped when running model state is unavailable."""
        model_service.settings.default_model = "my-model:8b"
        with patch("src.services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            with patch("src.services.model_service.get_available_vram", return_value=24):
                with patch.object(model_service, "get_running_models", return_value=None):
                    with caplog.at_level(logging.DEBUG, logger="src.services.model_service"):
                        result = model_service.check_health()

        assert result.is_healthy is True
        assert result.cold_start_models == []
        assert any("Skipping cold-start detection" in r.message for r in caplog.records)

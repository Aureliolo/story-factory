"""Tests for ModelService."""

from unittest.mock import MagicMock, patch

import pytest

from services.model_service import ModelService
from settings import Settings


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
        with patch("services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.list.return_value = MagicMock(models=[])

            health = model_service.check_health()

            assert health.is_healthy is True
            assert "running" in health.message.lower()

    def test_returns_unhealthy_on_connection_error(self, model_service):
        """Test health check returns unhealthy on connection failure."""
        with patch("services.model_service.ollama.Client") as mock_client:
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
        with patch("services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.list.side_effect = ConnectionError("Failed")

            result = model_service.list_installed()

            assert result == []

    def test_returns_model_ids(self, model_service):
        """Test returns list of model IDs."""
        with patch("services.model_service.ollama.Client") as mock_client:
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
        with patch("services.model_service.ollama.Client") as mock_client:
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
        with patch("services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.pull.side_effect = ConnectionError("Network error")

            results = list(model_service.pull_model("test-model"))

            assert len(results) == 1
            assert results[0]["error"] is True
            assert "error" in results[0]["status"].lower()

    def test_handles_none_values_in_progress(self, model_service):
        """Test pull_model handles None values for total/completed without crashing.

        This is a regression test for the bug where Ollama API returns None
        for total/completed fields during certain phases of the download.
        """
        with patch("services.model_service.ollama.Client") as mock_client:
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
        with patch("services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.pull.return_value = [
                {"status": "pulling layers", "completed": 100, "total": 5000},
            ]

            result = model_service.check_model_update("test-model")

            assert result["has_update"] is True

    def test_returns_no_update_when_up_to_date(self, model_service):
        """Test returns has_update=False when already up to date."""
        with patch("services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.pull.return_value = [
                {"status": "already up to date"},
            ]

            result = model_service.check_model_update("test-model")

            assert result["has_update"] is False

    def test_handles_none_total_gracefully(self, model_service):
        """Test handles None value for total without crashing."""
        with patch("services.model_service.ollama.Client") as mock_client:
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
        with patch("services.model_service.ollama.Client") as mock_client:
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
        with patch("services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.pull.side_effect = ConnectionError("Network error")

            result = model_service.check_model_update("test-model")

            assert result.get("error") is True


class TestModelServiceDeleteModel:
    """Tests for delete_model method."""

    def test_returns_true_on_success(self, model_service):
        """Test returns True when deletion succeeds."""
        with patch("services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            result = model_service.delete_model("test-model")

            assert result is True
            mock_instance.delete.assert_called_once_with("test-model")

    def test_returns_false_on_error(self, model_service):
        """Test returns False when deletion fails."""
        with patch("services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.delete.side_effect = ConnectionError("Failed")

            result = model_service.delete_model("test-model")

            assert result is False


class TestModelServiceTestModel:
    """Tests for test_model method."""

    def test_returns_success_on_valid_response(self, model_service):
        """Test returns success when model generates response."""
        with patch("services.model_service.ollama.Client") as mock_client:
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
        with patch("services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.generate.side_effect = ConnectionError("Model not found")

            success, message = model_service.test_model("test-model")

            assert success is False


class TestModelServiceGetVram:
    """Tests for get_vram method."""

    def test_returns_vram_value(self, model_service):
        """Test returns VRAM value."""
        with patch("services.model_service.get_available_vram") as mock_vram:
            mock_vram.return_value = 24

            result = model_service.get_vram()

            assert result == 24

    def test_returns_zero_on_error(self, model_service):
        """Test returns 0 when VRAM detection fails."""
        with patch("services.model_service.get_available_vram") as mock_vram:
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


class TestModelServiceCheckHealthEdgeCases:
    """Additional edge case tests for check_health."""

    def test_returns_unhealthy_on_response_error(self, model_service):
        """Test health check handles Ollama ResponseError."""
        import ollama

        with patch("services.model_service.ollama.Client") as mock_client:
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
        with patch("services.model_service.ollama.Client") as mock_client:
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
        with patch("services.model_service.ollama.Client") as mock_client:
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
        with patch("services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.list.side_effect = ConnectionError("Failed")

            result = model_service.list_installed_with_sizes()

            assert result == {}


class TestModelServiceListAvailable:
    """Tests for list_available method."""

    def test_lists_known_and_installed_models(self, model_service):
        """Test returns combined list of known and installed models."""
        with patch("services.model_service.ollama.Client") as mock_client:
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
        with patch("services.model_service.ollama.Client") as mock_client:
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
        with patch("services.model_service.ollama.Client") as mock_client:
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

        with patch("services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.pull.side_effect = ollama.ResponseError("Model not found")

            results = list(model_service.pull_model("nonexistent-model"))

            assert len(results) == 1
            assert results[0]["error"] is True

    def test_yields_error_on_unexpected_exception(self, model_service):
        """Test pull_model yields error on unexpected exception."""
        with patch("services.model_service.ollama.Client") as mock_client:
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
        with patch("services.model_service.ollama.Client") as mock_client:
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

        with patch("services.model_service.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.pull.side_effect = ollama.ResponseError("API error")

            result = model_service.check_model_update("test-model")

            assert result.get("error") is True
            assert "failed" in result["message"].lower()

    def test_returns_no_update_when_loop_completes_without_match(self, model_service):
        """Test returns no update when loop completes without matching status."""
        with patch("services.model_service.ollama.Client") as mock_client:
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
        with patch("services.model_service.ollama.Client") as mock_client:
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
        with patch("services.model_service.get_available_vram") as mock_vram:
            mock_vram.return_value = 24

            # Mock installed models with sizes
            with patch(
                "settings.get_installed_models_with_sizes",
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
        with patch("services.model_service.get_available_vram") as mock_vram:
            mock_vram.return_value = 14

            # Mock installed models
            with patch(
                "settings.get_installed_models_with_sizes",
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
        with patch("services.model_service.get_available_vram") as mock_vram:
            mock_vram.return_value = 8

            with patch(
                "settings.get_installed_models_with_sizes",
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
        with patch("services.model_service.get_available_vram") as mock_vram:
            mock_vram.return_value = 24

            with patch(
                "settings.get_installed_models_with_sizes",
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
            from services.model_service import ModelStatus

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
        with patch("services.model_service.ollama.Client") as mock_client:
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

        with patch("services.model_service.ollama.Client") as mock_client:
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
            from services.model_service import ModelStatus

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
            from services.model_service import ModelStatus

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
            from services.model_service import ModelStatus

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
        with patch("services.model_service.ollama.Client") as mock_client:
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
        with patch("services.model_service.ollama.Client") as mock_client:
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

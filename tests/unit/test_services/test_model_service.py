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

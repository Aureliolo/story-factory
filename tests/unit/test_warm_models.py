"""Tests for model warm-up at world build start."""

import logging
from unittest.mock import MagicMock, patch

import ollama
import pytest

from src.services.world_service._warmup import _warm_models


@pytest.fixture
def mock_services():
    """Create a mock ServiceContainer with world_quality service."""
    services = MagicMock()
    services.world_quality._get_creator_model.return_value = "test-creator:8b"
    services.world_quality._get_judge_model.return_value = "test-judge:12b"
    services.world_quality.settings.ollama_url = "http://localhost:11434"
    services.world_quality.settings.ollama_generate_timeout = 120.0
    return services


class TestWarmModels:
    """Tests for _warm_models helper function."""

    def test_warms_both_models(self, mock_services):
        """Test that both creator and judge models get warmed."""
        with patch("src.services.world_service._warmup.get_ollama_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            _warm_models(mock_services)

            assert mock_client.chat.call_count == 2
            calls = mock_client.chat.call_args_list
            model_names = {call.kwargs["model"] for call in calls}
            assert model_names == {"test-creator:8b", "test-judge:12b"}

    def test_deduplicates_same_model(self, mock_services):
        """Test that same model for creator and judge only warms once."""
        mock_services.world_quality._get_creator_model.return_value = "shared-model:8b"
        mock_services.world_quality._get_judge_model.return_value = "shared-model:8b"

        with patch("src.services.world_service._warmup.get_ollama_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            _warm_models(mock_services)

            assert mock_client.chat.call_count == 1
            mock_client.chat.assert_called_once_with(
                model="shared-model:8b",
                messages=[{"role": "user", "content": "hi"}],
                options={"num_predict": 1, "num_ctx": 512},
            )

    def test_warm_failure_ollama_response_error_is_nonfatal(self, mock_services):
        """Test that ollama.ResponseError during warm-up does not raise."""
        with patch("src.services.world_service._warmup.get_ollama_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.side_effect = ollama.ResponseError("model not found")
            mock_get_client.return_value = mock_client

            # Should not raise
            _warm_models(mock_services)

    def test_warm_failure_connection_error_is_nonfatal(self, mock_services):
        """Test that ConnectionError (Ollama not running) does not raise."""
        with patch("src.services.world_service._warmup.get_ollama_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.side_effect = ConnectionError("Connection refused")
            mock_get_client.return_value = mock_client

            _warm_models(mock_services)

    def test_warm_failure_timeout_error_is_nonfatal(self, mock_services):
        """Test that TimeoutError during warm-up does not raise."""
        with patch("src.services.world_service._warmup.get_ollama_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.side_effect = TimeoutError("Read timed out")
            mock_get_client.return_value = mock_client

            _warm_models(mock_services)

    def test_warm_uses_num_predict_1(self, mock_services):
        """Test that warm-up uses num_predict=1 for minimal overhead."""
        with patch("src.services.world_service._warmup.get_ollama_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            _warm_models(mock_services)

            for call in mock_client.chat.call_args_list:
                assert call.kwargs["options"]["num_predict"] == 1
                assert call.kwargs["options"]["num_ctx"] == 512

    def test_uses_get_ollama_client_with_model_id(self, mock_services):
        """Test that get_ollama_client is called with settings and model_id."""
        with patch("src.services.world_service._warmup.get_ollama_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            _warm_models(mock_services)

            # Verify get_ollama_client was called with correct settings and model_id
            for call in mock_get_client.call_args_list:
                assert call.args[0] is mock_services.world_quality.settings

            model_ids = {
                call.kwargs.get("model_id") or call.args[1]
                for call in mock_get_client.call_args_list
            }
            assert model_ids == {"test-creator:8b", "test-judge:12b"}

    def test_partial_failure_first_succeeds_second_fails(self, mock_services):
        """Test that partial failure (first model ok, second fails) doesn't raise."""
        with patch("src.services.world_service._warmup.get_ollama_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.side_effect = [
                MagicMock(),  # first model succeeds
                ollama.ResponseError("model not found"),  # second model fails
            ]
            mock_get_client.return_value = mock_client

            # Should not raise — first model still warmed
            _warm_models(mock_services)

    def test_model_resolution_failure_is_nonfatal(self, mock_services):
        """Test that model resolution failure does not abort the build."""
        mock_services.world_quality._get_creator_model.side_effect = ValueError(
            "Unknown agent role"
        )

        # Should not raise
        _warm_models(mock_services)

    def test_logs_info_on_success(self, mock_services, caplog):
        """Test that successful warm-up emits an info log."""
        with patch("src.services.world_service._warmup.get_ollama_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            with caplog.at_level(logging.INFO, logger="src.services.world_service._warmup"):
                _warm_models(mock_services)

            warmed_msgs = [r for r in caplog.records if "Warmed model" in r.message]
            assert len(warmed_msgs) == 2

    def test_logs_warning_on_failure(self, mock_services, caplog):
        """Test that warm-up failure emits a warning log."""
        with patch("src.services.world_service._warmup.get_ollama_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.side_effect = ConnectionError("Connection refused")
            mock_get_client.return_value = mock_client

            with caplog.at_level(logging.WARNING, logger="src.services.world_service._warmup"):
                _warm_models(mock_services)

            warning_msgs = [r for r in caplog.records if "Failed to warm model" in r.message]
            assert len(warning_msgs) == 2

    def test_logs_warning_on_model_resolution_failure(self, mock_services, caplog):
        """Test that model resolution failure emits a warning log."""
        mock_services.world_quality._get_creator_model.side_effect = ValueError(
            "Unknown agent role"
        )

        with caplog.at_level(logging.WARNING, logger="src.services.world_service._warmup"):
            _warm_models(mock_services)

        resolution_msgs = [
            r for r in caplog.records if "Failed to resolve models for warm-up" in r.message
        ]
        assert len(resolution_msgs) == 1

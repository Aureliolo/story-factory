"""Tests for model warm-up at world build start."""

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
        with patch("src.services.world_service._warmup.ollama.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            _warm_models(mock_services)

            assert mock_client.chat.call_count == 2
            calls = mock_client.chat.call_args_list
            model_names = {call.kwargs["model"] for call in calls}
            assert model_names == {"test-creator:8b", "test-judge:12b"}

    def test_deduplicates_same_model(self, mock_services):
        """Test that same model for creator and judge only warms once."""
        mock_services.world_quality._get_creator_model.return_value = "shared-model:8b"
        mock_services.world_quality._get_judge_model.return_value = "shared-model:8b"

        with patch("src.services.world_service._warmup.ollama.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            _warm_models(mock_services)

            assert mock_client.chat.call_count == 1
            mock_client.chat.assert_called_once_with(
                model="shared-model:8b",
                messages=[{"role": "user", "content": "hi"}],
                options={"num_predict": 1, "num_ctx": 512},
            )

    def test_warm_failure_is_nonfatal(self, mock_services):
        """Test that warm-up failure does not raise."""
        with patch("src.services.world_service._warmup.ollama.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.side_effect = ollama.ResponseError("model not found")
            mock_client_cls.return_value = mock_client

            # Should not raise
            _warm_models(mock_services)

    def test_warm_uses_num_predict_1(self, mock_services):
        """Test that warm-up uses num_predict=1 for minimal overhead."""
        with patch("src.services.world_service._warmup.ollama.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            _warm_models(mock_services)

            for call in mock_client.chat.call_args_list:
                assert call.kwargs["options"]["num_predict"] == 1
                assert call.kwargs["options"]["num_ctx"] == 512

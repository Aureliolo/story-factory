"""Tests for prompt metrics collection."""

from datetime import datetime, timedelta

import pytest

from prompts.metrics import MetricsCollector, PromptMetrics


class TestPromptMetrics:
    """Tests for PromptMetrics dataclass."""

    def test_create_metrics(self):
        """Should create PromptMetrics instance."""
        metrics = PromptMetrics(
            prompt_hash="abc123",
            agent="writer",
            task="write_chapter",
            model="test-model",
            temperature=0.7,
            timestamp=datetime.now(),
            latency_ms=1500.0,
            tokens_prompt=100,
            tokens_completion=500,
            validation_passed=True,
            retry_count=0,
            response_length=2000,
        )

        assert metrics.prompt_hash == "abc123"
        assert metrics.agent == "writer"
        assert metrics.latency_ms == 1500.0
        assert metrics.validation_passed is True


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    @pytest.fixture
    def storage_path(self, tmp_path):
        """Create temporary storage directory."""
        return tmp_path / "metrics"

    @pytest.fixture
    def sample_metrics(self):
        """Create sample metrics for testing."""
        now = datetime.now()
        return [
            PromptMetrics(
                prompt_hash="hash1",
                agent="writer",
                task="write_chapter",
                model="model1",
                temperature=0.7,
                timestamp=now - timedelta(hours=1),
                latency_ms=1000.0,
                validation_passed=True,
                retry_count=0,
                response_length=2000,
            ),
            PromptMetrics(
                prompt_hash="hash2",
                agent="writer",
                task="write_chapter",
                model="model1",
                temperature=0.7,
                timestamp=now - timedelta(hours=2),
                latency_ms=1500.0,
                validation_passed=True,
                retry_count=1,
                response_length=2500,
            ),
            PromptMetrics(
                prompt_hash="hash3",
                agent="editor",
                task="edit_chapter",
                model="model2",
                temperature=0.6,
                timestamp=now - timedelta(hours=3),
                latency_ms=800.0,
                validation_passed=False,
                retry_count=2,
                error="Validation failed",
                response_length=1500,
            ),
        ]

    def test_record_metrics(self, storage_path):
        """Should record metrics to daily log file."""
        collector = MetricsCollector(storage_path)

        metrics = PromptMetrics(
            prompt_hash="test123",
            agent="writer",
            task="test_task",
            model="test-model",
            temperature=0.7,
            timestamp=datetime.now(),
            latency_ms=1000.0,
        )

        collector.record(metrics)

        # Check file was created
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = storage_path / f"metrics_{today}.jsonl"
        assert log_file.exists()

        # Check content
        content = log_file.read_text()
        assert "test123" in content
        assert "writer" in content

    def test_get_agent_stats(self, storage_path, sample_metrics):
        """Should aggregate stats for an agent."""
        collector = MetricsCollector(storage_path)

        # Record sample metrics
        for m in sample_metrics:
            collector.record(m)

        # Get writer stats
        stats = collector.get_agent_stats("writer", days=7)

        assert stats["agent"] == "writer"
        assert stats["total_calls"] == 2
        assert stats["successful_calls"] == 2
        assert stats["success_rate"] == 1.0
        assert stats["avg_latency_ms"] == 1250.0  # (1000 + 1500) / 2
        assert stats["avg_retries"] == 0.5  # (0 + 1) / 2
        assert stats["error_rate"] == 0.0

    def test_get_agent_stats_with_errors(self, storage_path, sample_metrics):
        """Should correctly calculate error rates."""
        collector = MetricsCollector(storage_path)

        for m in sample_metrics:
            collector.record(m)

        stats = collector.get_agent_stats("editor", days=7)

        assert stats["total_calls"] == 1
        assert stats["error_rate"] == 1.0
        assert stats["validation_pass_rate"] == 0.0

    def test_get_agent_stats_no_data(self, storage_path):
        """Should handle agent with no metrics."""
        collector = MetricsCollector(storage_path)

        stats = collector.get_agent_stats("nonexistent", days=7)

        assert stats["total_calls"] == 0
        assert "note" in stats

    def test_get_task_stats(self, storage_path, sample_metrics):
        """Should get stats for specific agent/task combination."""
        collector = MetricsCollector(storage_path)

        for m in sample_metrics:
            collector.record(m)

        stats = collector.get_task_stats("writer", "write_chapter", days=7)

        assert stats["agent"] == "writer"
        assert stats["task"] == "write_chapter"
        assert stats["total_calls"] == 2
        assert stats["avg_latency_ms"] == 1250.0

    def test_get_model_comparison(self, storage_path, sample_metrics):
        """Should compare performance across models."""
        collector = MetricsCollector(storage_path)

        for m in sample_metrics:
            collector.record(m)

        comparison = collector.get_model_comparison(days=7)

        assert len(comparison) == 2
        # Should be sorted by total calls
        assert comparison[0]["model"] == "model1"
        assert comparison[0]["total_calls"] == 2
        assert comparison[1]["model"] == "model2"
        assert comparison[1]["total_calls"] == 1

    def test_cleanup_old_logs(self, storage_path):
        """Should delete old log files."""
        collector = MetricsCollector(storage_path)

        # Create old log files
        old_date = (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d")
        old_file = storage_path / f"metrics_{old_date}.jsonl"
        old_file.write_text('{"test": "data"}\n')

        # Create recent log file
        recent_date = datetime.now().strftime("%Y-%m-%d")
        recent_file = storage_path / f"metrics_{recent_date}.jsonl"
        recent_file.write_text('{"test": "data"}\n')

        # Cleanup (keep 30 days)
        collector.cleanup_old_logs(days_to_keep=30)

        # Old file should be deleted
        assert not old_file.exists()
        # Recent file should remain
        assert recent_file.exists()

    def test_load_recent_metrics(self, storage_path, sample_metrics):
        """Should load metrics from recent days."""
        collector = MetricsCollector(storage_path)

        # Record metrics
        for m in sample_metrics:
            collector.record(m)

        # Load recent metrics
        cutoff = datetime.now() - timedelta(days=1)
        metrics = collector._load_recent_metrics(cutoff)

        # Should only include metrics from last 24 hours
        assert len(metrics) >= 1
        assert all(m.timestamp >= cutoff for m in metrics)

    def test_percentile_calculation(self):
        """Should calculate percentiles correctly."""
        values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        p50 = MetricsCollector._percentile(values, 0.5)
        p95 = MetricsCollector._percentile(values, 0.95)

        assert p50 == 50
        assert p95 == 95

    def test_percentile_empty_list(self):
        """Should handle empty list."""
        result = MetricsCollector._percentile([], 0.5)
        assert result == 0.0

    def test_validation_errors(self, storage_path):
        """Should validate inputs."""
        collector = MetricsCollector(storage_path)

        with pytest.raises(ValueError, match="cannot be empty"):
            collector.get_agent_stats("", days=7)

        with pytest.raises(ValueError, match="must be positive"):
            collector.get_agent_stats("writer", days=0)

        with pytest.raises(ValueError, match="must be positive"):
            collector.cleanup_old_logs(days_to_keep=-1)

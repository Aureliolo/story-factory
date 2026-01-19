"""Metrics collection for prompt performance tracking."""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from utils.validation import validate_not_empty, validate_not_none, validate_positive

logger = logging.getLogger(__name__)


@dataclass
class PromptMetrics:
    """Metrics for a single prompt execution.

    Tracks performance, quality, and error information for analysis
    and optimization.
    """

    prompt_hash: str
    agent: str
    task: str
    model: str
    temperature: float
    timestamp: datetime

    # Performance metrics
    latency_ms: float
    tokens_prompt: int | None = None
    tokens_completion: int | None = None
    cache_hit: bool = False

    # Quality metrics
    validation_passed: bool = True
    retry_count: int = 0
    error: str | None = None

    # Response characteristics
    response_length: int = 0
    language_detected: str | None = None


class MetricsCollector:
    """Collects and analyzes prompt execution metrics.

    Stores metrics in daily JSONL files for efficient append-only logging
    and provides aggregation for analysis.
    """

    def __init__(self, storage_path: Path):
        """Initialize the metrics collector.

        Args:
            storage_path: Directory to store metric log files
        """
        validate_not_none(storage_path, "storage_path")

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def record(self, metrics: PromptMetrics):
        """Record metrics for a prompt execution.

        Args:
            metrics: The metrics to record
        """
        validate_not_none(metrics, "metrics")

        # Append to daily log file (JSONL format for efficient appending)
        date_str = metrics.timestamp.strftime("%Y-%m-%d")
        log_file = self.storage_path / f"metrics_{date_str}.jsonl"

        try:
            # Convert dataclass to dict, handle datetime serialization
            data = asdict(metrics)
            data["timestamp"] = metrics.timestamp.isoformat()

            with open(log_file, "a") as f:
                f.write(json.dumps(data) + "\n")

            logger.debug(f"Recorded metrics for {metrics.agent}/{metrics.task}")

        except Exception as e:
            logger.warning(f"Failed to record metrics: {e}")

    def get_agent_stats(self, agent: str, days: int = 7) -> dict[str, Any]:
        """Get aggregated statistics for an agent over recent days.

        Args:
            agent: Agent name to analyze
            days: Number of recent days to include

        Returns:
            Dictionary of aggregated statistics
        """
        validate_not_empty(agent, "agent")
        validate_positive(days, "days")

        # Load recent metrics
        cutoff = datetime.now() - timedelta(days=days)
        metrics = self._load_recent_metrics(cutoff)

        # Filter to this agent
        agent_metrics = [m for m in metrics if m.agent == agent]

        if not agent_metrics:
            return {
                "agent": agent,
                "days": days,
                "total_calls": 0,
                "note": "No data available for this period",
            }

        # Calculate statistics
        total = len(agent_metrics)
        successful = [m for m in agent_metrics if m.validation_passed and not m.error]

        return {
            "agent": agent,
            "days": days,
            "total_calls": total,
            "successful_calls": len(successful),
            "success_rate": len(successful) / total if total > 0 else 0,
            "avg_latency_ms": sum(m.latency_ms for m in agent_metrics) / total,
            "p95_latency_ms": self._percentile([m.latency_ms for m in agent_metrics], 0.95),
            "validation_pass_rate": (sum(1 for m in agent_metrics if m.validation_passed) / total),
            "avg_retries": sum(m.retry_count for m in agent_metrics) / total,
            "error_rate": sum(1 for m in agent_metrics if m.error) / total,
            "cache_hit_rate": sum(1 for m in agent_metrics if m.cache_hit) / total,
            "avg_response_length": (
                sum(m.response_length for m in agent_metrics) / total if total > 0 else 0
            ),
            "total_tokens_used": sum(
                (m.tokens_prompt or 0) + (m.tokens_completion or 0) for m in agent_metrics
            ),
        }

    def get_task_stats(self, agent: str, task: str, days: int = 7) -> dict[str, Any]:
        """Get statistics for a specific agent task.

        Args:
            agent: Agent name
            task: Task name
            days: Number of recent days to include

        Returns:
            Dictionary of task-specific statistics
        """
        validate_not_empty(agent, "agent")
        validate_not_empty(task, "task")

        cutoff = datetime.now() - timedelta(days=days)
        metrics = self._load_recent_metrics(cutoff)

        # Filter to this agent/task
        task_metrics = [m for m in metrics if m.agent == agent and m.task == task]

        if not task_metrics:
            return {
                "agent": agent,
                "task": task,
                "days": days,
                "total_calls": 0,
            }

        total = len(task_metrics)

        return {
            "agent": agent,
            "task": task,
            "days": days,
            "total_calls": total,
            "avg_latency_ms": sum(m.latency_ms for m in task_metrics) / total,
            "validation_pass_rate": sum(1 for m in task_metrics if m.validation_passed) / total,
            "error_rate": sum(1 for m in task_metrics if m.error) / total,
        }

    def get_model_comparison(self, days: int = 7) -> list[dict[str, Any]]:
        """Compare performance across different models.

        Args:
            days: Number of recent days to include

        Returns:
            List of per-model statistics, sorted by total usage
        """
        cutoff = datetime.now() - timedelta(days=days)
        metrics = self._load_recent_metrics(cutoff)

        # Group by model
        by_model: dict[str, list[PromptMetrics]] = {}
        for m in metrics:
            if m.model not in by_model:
                by_model[m.model] = []
            by_model[m.model].append(m)

        # Calculate stats per model
        results = []
        for model, model_metrics in by_model.items():
            total = len(model_metrics)
            results.append(
                {
                    "model": model,
                    "total_calls": total,
                    "avg_latency_ms": sum(m.latency_ms for m in model_metrics) / total,
                    "success_rate": (
                        sum(1 for m in model_metrics if m.validation_passed and not m.error) / total
                    ),
                    "avg_retries": sum(m.retry_count for m in model_metrics) / total,
                }
            )

        # Sort by total calls descending
        results.sort(key=lambda x: x["total_calls"], reverse=True)
        return results

    def _load_recent_metrics(self, cutoff: datetime) -> list[PromptMetrics]:
        """Load metrics from disk since cutoff date.

        Args:
            cutoff: Only load metrics after this datetime

        Returns:
            List of PromptMetrics objects
        """
        metrics = []

        # Find all metric files in date range
        for log_file in sorted(self.storage_path.glob("metrics_*.jsonl")):
            try:
                with open(log_file) as f:
                    for line in f:
                        data = json.loads(line.strip())

                        # Parse timestamp
                        timestamp = datetime.fromisoformat(data["timestamp"])
                        if timestamp < cutoff:
                            continue

                        # Reconstruct PromptMetrics object
                        metrics.append(
                            PromptMetrics(
                                prompt_hash=data["prompt_hash"],
                                agent=data["agent"],
                                task=data["task"],
                                model=data["model"],
                                temperature=data["temperature"],
                                timestamp=timestamp,
                                latency_ms=data["latency_ms"],
                                tokens_prompt=data.get("tokens_prompt"),
                                tokens_completion=data.get("tokens_completion"),
                                cache_hit=data.get("cache_hit", False),
                                validation_passed=data.get("validation_passed", True),
                                retry_count=data.get("retry_count", 0),
                                error=data.get("error"),
                                response_length=data.get("response_length", 0),
                                language_detected=data.get("language_detected"),
                            )
                        )
            except Exception as e:
                logger.warning(f"Failed to load metrics from {log_file}: {e}")

        return metrics

    @staticmethod
    def _percentile(values: list[float], p: float) -> float:
        """Calculate percentile of a list of values.

        Args:
            values: List of numeric values
            p: Percentile (0.0 to 1.0)

        Returns:
            Value at the given percentile
        """
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int(len(sorted_values) * p)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Delete metric log files older than specified days.

        Args:
            days_to_keep: Number of days of logs to retain
        """
        validate_positive(days_to_keep, "days_to_keep")

        cutoff = datetime.now() - timedelta(days=days_to_keep)
        cutoff_date_str = cutoff.strftime("%Y-%m-%d")

        deleted_count = 0
        for log_file in self.storage_path.glob("metrics_*.jsonl"):
            try:
                # Extract date from filename
                date_str = log_file.stem.split("_", 1)[1]
                if date_str < cutoff_date_str:
                    log_file.unlink()
                    deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to process log file {log_file}: {e}")

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old metric log files")

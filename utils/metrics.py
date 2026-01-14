"""Performance metrics tracking for Story Factory."""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class GenerationMetrics:
    """Metrics for a single generation operation."""
    operation: str  # "chapter", "interview", "outline", etc.
    agent_name: str
    model: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    tokens_generated: int = 0
    input_length: int = 0
    output_length: int = 0
    success: bool = True
    error: Optional[str] = None
    
    def finish(self, success: bool = True, error: Optional[str] = None, output_length: int = 0):
        """Mark the operation as complete."""
        self.end_time = datetime.now()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.success = success
        self.error = error
        self.output_length = output_length
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "operation": self.operation,
            "agent_name": self.agent_name,
            "model": self.model,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "input_length": self.input_length,
            "output_length": self.output_length,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class StoryMetrics:
    """Aggregated metrics for an entire story generation."""
    story_id: str
    created_at: datetime = field(default_factory=datetime.now)
    operations: list[GenerationMetrics] = field(default_factory=list)
    
    def add_operation(self, metric: GenerationMetrics):
        """Add a completed operation metric."""
        self.operations.append(metric)
        
        # Log the metric
        if metric.success:
            logger.info(
                f"Performance: {metric.agent_name} {metric.operation} completed in "
                f"{metric.duration_seconds:.2f}s ({metric.output_length} chars)",
                extra={'story_id': self.story_id, 'agent_name': metric.agent_name}
            )
        else:
            logger.warning(
                f"Performance: {metric.agent_name} {metric.operation} failed after "
                f"{metric.duration_seconds:.2f}s: {metric.error}",
                extra={'story_id': self.story_id, 'agent_name': metric.agent_name}
            )
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        if not self.operations:
            return {
                "total_operations": 0,
                "total_duration_seconds": 0,
                "average_duration_seconds": 0,
                "success_rate": 0,
            }
        
        total_duration = sum(op.duration_seconds for op in self.operations)
        successful = sum(1 for op in self.operations if op.success)
        
        # Group by agent
        by_agent: Dict[str, list[float]] = {}
        for op in self.operations:
            if op.agent_name not in by_agent:
                by_agent[op.agent_name] = []
            by_agent[op.agent_name].append(op.duration_seconds)
        
        agent_stats = {
            agent: {
                "count": len(times),
                "total_seconds": sum(times),
                "average_seconds": sum(times) / len(times),
                "min_seconds": min(times),
                "max_seconds": max(times),
            }
            for agent, times in by_agent.items()
        }
        
        return {
            "story_id": self.story_id,
            "total_operations": len(self.operations),
            "total_duration_seconds": total_duration,
            "average_duration_seconds": total_duration / len(self.operations),
            "success_rate": successful / len(self.operations),
            "by_agent": agent_stats,
        }
    
    def estimate_remaining_time(self, total_chapters: int, completed_chapters: int) -> Optional[timedelta]:
        """Estimate time remaining for story generation.
        
        Args:
            total_chapters: Total number of chapters in the story
            completed_chapters: Number of chapters completed so far
            
        Returns:
            Estimated time remaining, or None if not enough data
        """
        if completed_chapters == 0:
            return None
        
        # Get average time per chapter from completed operations
        chapter_ops = [op for op in self.operations if op.operation == "chapter" and op.success]
        if not chapter_ops:
            return None
        
        avg_time_per_chapter = sum(op.duration_seconds for op in chapter_ops) / len(chapter_ops)
        remaining_chapters = total_chapters - completed_chapters
        
        estimated_seconds = avg_time_per_chapter * remaining_chapters
        return timedelta(seconds=estimated_seconds)
    
    def save_to_file(self, directory: Path):
        """Save metrics to a JSON file."""
        filepath = directory / f"metrics_{self.story_id}.json"
        directory.mkdir(parents=True, exist_ok=True)
        
        data = {
            "story_id": self.story_id,
            "created_at": self.created_at.isoformat(),
            "operations": [op.to_dict() for op in self.operations],
            "summary": self.get_summary(),
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Metrics saved to {filepath}")
        except IOError as e:
            logger.warning(f"Failed to save metrics: {e}")


class PerformanceTracker:
    """Context manager for tracking operation performance."""
    
    def __init__(
        self,
        metrics: StoryMetrics,
        operation: str,
        agent_name: str,
        model: str,
        input_length: int = 0
    ):
        self.metrics = metrics
        self.operation_metric = GenerationMetrics(
            operation=operation,
            agent_name=agent_name,
            model=model,
            input_length=input_length,
        )
    
    def __enter__(self):
        return self.operation_metric
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Mark as complete (failed if exception occurred)
        success = exc_type is None
        error = str(exc_val) if exc_val else None
        self.operation_metric.finish(success=success, error=error)
        
        # Add to metrics
        self.metrics.add_operation(self.operation_metric)
        
        # Don't suppress exceptions
        return False


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string like "2m 30s" or "1h 15m"
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    
    if minutes < 60:
        return f"{minutes}m {remaining_seconds}s"
    
    hours = minutes // 60
    remaining_minutes = minutes % 60
    
    return f"{hours}h {remaining_minutes}m"

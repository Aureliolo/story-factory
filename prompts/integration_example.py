"""Example integration of prompt management system with BaseAgent.

This module shows how to integrate the new prompt template, caching,
and metrics systems with the existing BaseAgent architecture.
"""

import logging
import time
from datetime import datetime
from pathlib import Path

from agents.base import BaseAgent
from prompts import MetricsCollector, PromptMetrics, PromptTemplateManager, ResponseCache
from settings import Settings

logger = logging.getLogger(__name__)


class EnhancedBaseAgent(BaseAgent):
    """Enhanced BaseAgent with template, cache, and metrics support.

    This class demonstrates how to integrate the new prompt management
    features while maintaining backward compatibility.
    """

    # Class-level shared resources (singleton pattern)
    _template_manager: PromptTemplateManager | None = None
    _response_cache: ResponseCache | None = None
    _metrics_collector: MetricsCollector | None = None

    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        agent_role: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        settings: Settings | None = None,
        enable_cache: bool = True,
        enable_metrics: bool = True,
    ):
        """Initialize enhanced agent.

        Args:
            name: Display name
            role: Role description
            system_prompt: System prompt
            agent_role: Role identifier for auto model selection
            model: Override model
            temperature: Override temperature
            settings: Application settings
            enable_cache: Enable response caching
            enable_metrics: Enable metrics collection
        """
        super().__init__(name, role, system_prompt, agent_role, model, temperature, settings)

        self.enable_cache = enable_cache
        self.enable_metrics = enable_metrics

        # Initialize shared resources (only once)
        if EnhancedBaseAgent._template_manager is None:
            template_dir = Path(__file__).parent.parent / "prompts" / "templates"
            EnhancedBaseAgent._template_manager = PromptTemplateManager(template_dir)
            logger.info("Initialized shared template manager")

        if EnhancedBaseAgent._response_cache is None and enable_cache:
            cache_dir = Path(__file__).parent.parent / "output" / "cache"
            EnhancedBaseAgent._response_cache = ResponseCache(
                cache_dir,
                max_size=self.settings.prompt_cache_size,
                ttl_seconds=self.settings.prompt_cache_ttl,
            )
            logger.info("Initialized shared response cache")

        if EnhancedBaseAgent._metrics_collector is None and enable_metrics:
            metrics_dir = Path(__file__).parent.parent / "output" / "metrics"
            EnhancedBaseAgent._metrics_collector = MetricsCollector(metrics_dir)
            logger.info("Initialized shared metrics collector")

    def generate_from_template(
        self,
        task: str,
        template_version: str = "latest",
        context: str | None = None,
        temperature: float | None = None,
        **template_vars,
    ) -> str:
        """Generate using a prompt template with caching and metrics.

        Args:
            task: Task name (e.g., "write_chapter")
            template_version: Template version to use
            context: Optional context to include
            temperature: Override temperature
            **template_vars: Variables to render in template

        Returns:
            Generated response
        """
        # Load template
        template = self._template_manager.load(self.agent_role, task, template_version)

        # Render prompts
        system_prompt, user_prompt = template.render(**template_vars)

        # Add context if provided
        if context:
            user_prompt = f"CONTEXT:\n{context}\n\n{user_prompt}"

        # Check cache first
        use_temp = temperature if temperature is not None else self.temperature
        if self.enable_cache and self._response_cache:
            cached = self._response_cache.get(user_prompt, self.model, use_temp)
            if cached:
                logger.info(f"{self.name}: Cache hit for {task}")
                # Still record metrics for cache hits
                if self.enable_metrics and self._metrics_collector:
                    self._record_metrics(
                        user_prompt,
                        task,
                        use_temp,
                        latency_ms=0,
                        response=cached,
                        cache_hit=True,
                    )
                return cached

        # Generate with metrics tracking
        start_time = time.time()
        retry_count = 0

        try:
            # Call parent generate method
            response = self.generate(user_prompt, context=None, temperature=use_temp)

            duration_ms = (time.time() - start_time) * 1000

            # Cache successful response
            if self.enable_cache and self._response_cache:
                self._response_cache.put(user_prompt, self.model, use_temp, response)

            # Record metrics
            if self.enable_metrics and self._metrics_collector:
                self._record_metrics(
                    user_prompt,
                    task,
                    use_temp,
                    duration_ms,
                    response,
                    validation_passed=True,
                    retry_count=retry_count,
                )

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            # Record failure metrics
            if self.enable_metrics and self._metrics_collector:
                self._record_metrics(
                    user_prompt,
                    task,
                    use_temp,
                    duration_ms,
                    "",
                    validation_passed=False,
                    error=str(e),
                )

            raise

    def _record_metrics(
        self,
        prompt: str,
        task: str,
        temperature: float,
        latency_ms: float,
        response: str,
        validation_passed: bool = True,
        retry_count: int = 0,
        error: str | None = None,
        cache_hit: bool = False,
    ):
        """Record metrics for a generation.

        Args:
            prompt: The prompt text
            task: Task name
            temperature: Temperature used
            latency_ms: Generation latency in milliseconds
            response: The generated response
            validation_passed: Whether validation passed
            retry_count: Number of retries
            error: Error message if failed
            cache_hit: Whether this was a cache hit
        """
        if not self._metrics_collector:
            return

        import hashlib

        prompt_hash = hashlib.sha256(
            f"{prompt}|{self.model}|{temperature:.2f}".encode()
        ).hexdigest()[:16]

        metrics = PromptMetrics(
            prompt_hash=prompt_hash,
            agent=self.agent_role,
            task=task,
            model=self.model,
            temperature=temperature,
            timestamp=datetime.now(),
            latency_ms=latency_ms,
            cache_hit=cache_hit,
            validation_passed=validation_passed,
            retry_count=retry_count,
            error=error,
            response_length=len(response),
        )

        self._metrics_collector.record(metrics)

    @classmethod
    def get_cache_stats(cls) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary of cache stats
        """
        if cls._response_cache:
            return cls._response_cache.get_stats()
        return {"error": "Cache not initialized"}

    @classmethod
    def get_agent_metrics(cls, agent_role: str, days: int = 7) -> dict:
        """Get metrics for an agent.

        Args:
            agent_role: Agent role identifier
            days: Number of days to analyze

        Returns:
            Dictionary of aggregated metrics
        """
        if cls._metrics_collector:
            return cls._metrics_collector.get_agent_stats(agent_role, days)
        return {"error": "Metrics not initialized"}


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create enhanced agent
    agent = EnhancedBaseAgent(
        name="Writer",
        role="Prose Craftsman",
        system_prompt="You are a writer.",
        agent_role="writer",
    )

    # Generate using template
    response = agent.generate_from_template(
        task="write_chapter",
        chapter_number=1,
        chapter_title="The Beginning",
        language="English",
        outline="Hero discovers their power",
        context="Fantasy adventure story",
        genre="Fantasy",
        tone="Adventurous",
        content_rating="everyone",
        previous_chapter_context="",
        revision_notes="",
    )

    print(f"Generated: {response[:200]}...")

    # Check cache stats
    cache_stats = EnhancedBaseAgent.get_cache_stats()
    print(f"\nCache stats: {cache_stats}")

    # Check agent metrics
    metrics = EnhancedBaseAgent.get_agent_metrics("writer", days=7)
    print(f"\nAgent metrics: {metrics}")

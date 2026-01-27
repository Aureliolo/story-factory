"""Quality judging mixin for ModelModeService."""

import logging
from typing import Any

import src.services.llm_client as _llm_client
from src.memory.mode_models import QualityScores
from src.utils.validation import validate_not_empty

from ._base import ModelModeServiceBase

logger = logging.getLogger(__name__)


class QualityMixin(ModelModeServiceBase):
    """Mixin providing quality judging functionality."""

    def judge_quality(
        self,
        content: str,
        genre: str,
        tone: str,
        themes: list[str],
    ) -> QualityScores:
        """Use LLM to judge content quality.

        Args:
            content: The generated content to evaluate.
            genre: Story genre.
            tone: Story tone.
            themes: Story themes.

        Returns:
            QualityScores with prose_quality and instruction_following.
        """
        validate_not_empty(content, "content")
        validate_not_empty(genre, "genre")
        validate_not_empty(tone, "tone")
        # Use validator model or smallest available
        judge_model = self.get_model_for_agent("validator")
        logger.debug(f"Using {judge_model} to judge quality for {genre}/{tone}")

        # Limit content size for faster judging
        truncated_content = content[: self.settings.content_truncation_for_judgment]

        prompt = f"""You are evaluating the quality of AI-generated story content.

**Story Brief:**
Genre: {genre}
Tone: {tone}
Themes: {", ".join(themes)}

**Content to evaluate:**
{truncated_content}

Rate each dimension from 0-10:

1. prose_quality: Creativity, flow, engagement, vocabulary variety
2. instruction_following: Adherence to genre, tone, themes"""

        try:
            scores = _llm_client.generate_structured(
                settings=self.settings,
                model=judge_model,
                prompt=prompt,
                response_model=QualityScores,
                temperature=self.settings.temp_capability_check,
            )
            logger.info(
                f"Quality judged: prose={scores.prose_quality:.1f}, "
                f"instruction={scores.instruction_following:.1f}"
            )
            return scores
        except Exception as e:
            logger.error(f"Quality judgment failed: {e}", exc_info=True)

        # Return neutral scores on failure
        logger.warning("Returning neutral quality scores (5.0) due to judgment failure")
        return QualityScores(prose_quality=5.0, instruction_following=5.0)

    def calculate_consistency_score(self, issues: list[dict[str, Any]]) -> float:
        """Calculate consistency score from continuity issues.

        Args:
            issues: List of ContinuityIssue-like dicts with 'severity'.

        Returns:
            Score from 0-10 (10 = no issues).
        """
        if not issues:
            return 10.0

        # Weight by severity
        penalty = 0.0
        for issue in issues:
            severity = issue.get("severity", "minor")
            if severity == "critical":
                penalty += 3.0
            elif severity == "moderate":
                penalty += 1.5
            else:  # minor
                penalty += 0.5

        return max(0.0, 10.0 - penalty)

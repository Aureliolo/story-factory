"""Scoring service - collects and aggregates quality signals.

This service provides a simple interface for recording implicit quality signals
from user interactions (regenerate, edit, rating) and coordinating with the
ModelModeService for LLM-based quality judgment.
"""

import logging
from difflib import SequenceMatcher

from memory.mode_models import QualityScores
from services.model_mode_service import ModelModeService
from utils.validation import validate_not_empty, validate_positive

logger = logging.getLogger(__name__)


class ScoringService:
    """Service for collecting quality signals during story generation.

    This service tracks:
    - Implicit signals from user behavior (regenerate, edit, accept)
    - LLM-judged quality scores
    - Performance metrics from generation

    It works with ModelModeService for score persistence and analysis.
    """

    # Maximum number of chapters to track simultaneously
    MAX_TRACKED_CHAPTERS = 50

    def __init__(self, mode_service: ModelModeService):
        """Initialize scoring service.

        Args:
            mode_service: The ModelModeService for score persistence.
        """
        logger.debug("Initializing ScoringService")
        self._mode_service = mode_service

        # Track active score IDs for updates
        # Note: Use clear_chapter_tracking() when chapters are complete to prevent memory leaks
        self._active_scores: dict[str, int] = {}  # chapter_id:agent_role -> score_id

        # Track original content for edit distance calculation
        # Note: Use clear_chapter_tracking() when chapters are complete to prevent memory leaks
        self._original_content: dict[str, str] = {}  # chapter_id -> content
        logger.debug("ScoringService initialized successfully")

    def start_tracking(
        self,
        project_id: str,
        agent_role: str,
        model_id: str,
        chapter_id: str | None = None,
        genre: str | None = None,
    ) -> int:
        """Start tracking a generation.

        Call this before generation starts. Returns a score ID that can be
        used for subsequent updates.

        Args:
            project_id: The project ID.
            agent_role: The agent role (writer, editor, etc.).
            model_id: The model being used.
            chapter_id: Optional chapter ID.
            genre: Optional story genre.

        Returns:
            Score ID for this generation.
        """
        validate_not_empty(project_id, "project_id")
        validate_not_empty(agent_role, "agent_role")
        validate_not_empty(model_id, "model_id")
        logger.debug(
            f"start_tracking called: project_id={project_id}, agent_role={agent_role}, "
            f"model_id={model_id}, chapter_id={chapter_id}"
        )
        try:
            score_id = self._mode_service.record_generation(
                project_id=project_id,
                agent_role=agent_role,
                model_id=model_id,
                chapter_id=chapter_id,
                genre=genre,
            )

            if chapter_id:
                self._active_scores[f"{chapter_id}:{agent_role}"] = score_id

            logger.info(
                f"Started tracking: score_id={score_id}, agent={agent_role}, model={model_id}"
            )
            return score_id
        except Exception as e:
            logger.error(
                f"Failed to start tracking for {agent_role}/{model_id}: {e}", exc_info=True
            )
            raise

    def finish_tracking(
        self,
        score_id: int,
        content: str,
        tokens_generated: int,
        time_seconds: float,
        chapter_id: str | None = None,
    ) -> None:
        """Finish tracking a generation with final metrics.

        Args:
            score_id: The score ID from start_tracking.
            content: The generated content.
            tokens_generated: Number of tokens generated.
            time_seconds: Generation time.
            chapter_id: Optional chapter ID for edit tracking.
        """
        validate_positive(score_id, "score_id")
        validate_not_empty(content, "content")
        logger.debug(
            f"finish_tracking called: score_id={score_id}, content_length={len(content)}, "
            f"tokens={tokens_generated}, time={time_seconds:.1f}s"
        )
        try:
            # Store original content for later edit distance calculation
            if chapter_id:
                self._original_content[chapter_id] = content
                # Enforce maximum tracking limit to prevent unbounded memory growth
                self._enforce_tracking_limits()

            # Persist performance metrics to database
            self._mode_service.update_performance_metrics(
                score_id,
                tokens_generated=tokens_generated,
                time_seconds=time_seconds,
            )

            logger.info(
                f"Generation tracking complete: score_id={score_id}, "
                f"tokens={tokens_generated}, time={time_seconds:.1f}s"
            )
        except Exception as e:
            logger.error(f"Failed to finish tracking for score_id {score_id}: {e}", exc_info=True)
            raise

    def on_regenerate(
        self,
        chapter_id: str,
        agent_role: str = "writer",
    ) -> None:
        """Record that user clicked regenerate (negative signal).

        Args:
            chapter_id: The chapter that was regenerated.
            agent_role: The agent role being regenerated.
        """
        validate_not_empty(chapter_id, "chapter_id")
        key = f"{chapter_id}:{agent_role}"
        score_id = self._active_scores.get(key)

        if score_id:
            self._mode_service.record_implicit_signal(score_id, was_regenerated=True)
            logger.debug(f"Recorded regenerate signal for score {score_id}")

    def on_edit(
        self,
        chapter_id: str,
        edited_content: str,
    ) -> None:
        """Record that user manually edited content.

        Calculates edit distance from original to measure how much
        the user changed the output.

        Args:
            chapter_id: The chapter that was edited.
            edited_content: The content after user edits.
        """
        validate_not_empty(chapter_id, "chapter_id")
        validate_not_empty(edited_content, "edited_content")
        original = self._original_content.get(chapter_id)
        if not original:
            return

        # Calculate edit distance (character-level)
        edit_distance = self._calculate_edit_distance(original, edited_content)

        # Find the most recent score for this chapter
        for key, score_id in self._active_scores.items():
            if key.startswith(f"{chapter_id}:"):
                self._mode_service.record_implicit_signal(score_id, edit_distance=edit_distance)
                logger.debug(f"Recorded edit signal for score {score_id}: distance={edit_distance}")
                break

        # Update stored original
        self._original_content[chapter_id] = edited_content

    def on_accept(
        self,
        chapter_id: str,
    ) -> None:
        """Record that user accepted content without changes (positive signal).

        Args:
            chapter_id: The chapter that was accepted.
        """
        validate_not_empty(chapter_id, "chapter_id")
        # Find scores for this chapter and mark edit_distance as 0
        for key, score_id in self._active_scores.items():
            if key.startswith(f"{chapter_id}:"):
                self._mode_service.record_implicit_signal(score_id, edit_distance=0)
                logger.debug(f"Recorded accept signal for score {score_id}")

    def on_rate(
        self,
        chapter_id: str,
        rating: int,
    ) -> None:
        """Record user's explicit rating for a chapter.

        Args:
            chapter_id: The chapter being rated.
            rating: Rating from 1-5 stars.
        """
        validate_not_empty(chapter_id, "chapter_id")
        if not 1 <= rating <= 5:
            logger.warning(f"Invalid rating {rating}, must be 1-5")
            return

        # Update all scores for this chapter
        for key, score_id in self._active_scores.items():
            if key.startswith(f"{chapter_id}:"):
                self._mode_service.record_implicit_signal(score_id, user_rating=rating)
                logger.debug(f"Recorded rating {rating} for score {score_id}")

    def judge_chapter_quality(
        self,
        content: str,
        genre: str,
        tone: str,
        themes: list[str],
        score_id: int | None = None,
    ) -> QualityScores:
        """Run LLM quality judgment on chapter content.

        Args:
            content: The chapter content to evaluate.
            genre: Story genre.
            tone: Story tone.
            themes: Story themes.
            score_id: Optional score ID to update with results.

        Returns:
            QualityScores from the judgment.
        """
        validate_not_empty(content, "content")
        validate_not_empty(genre, "genre")
        validate_not_empty(tone, "tone")
        logger.debug(
            f"judge_chapter_quality called: content_length={len(content)}, "
            f"genre={genre}, score_id={score_id}"
        )
        try:
            scores = self._mode_service.judge_quality(
                content=content,
                genre=genre,
                tone=tone,
                themes=themes,
            )

            if score_id:
                self._mode_service.update_quality_scores(score_id, scores)

            prose_str = f"{scores.prose_quality:.1f}" if scores.prose_quality is not None else "N/A"
            instr_str = (
                f"{scores.instruction_following:.1f}"
                if scores.instruction_following is not None
                else "N/A"
            )
            logger.info(f"Quality judgment complete: prose={prose_str}, instruction={instr_str}")
            return scores
        except Exception as e:
            logger.error(f"Failed to judge quality: {e}", exc_info=True)
            raise

    def calculate_consistency_score(
        self,
        issues: list[dict],
        score_id: int | None = None,
    ) -> float:
        """Calculate consistency score from continuity issues.

        Args:
            issues: List of continuity issues with 'severity' field.
            score_id: Optional score ID to update.

        Returns:
            Score from 0-10 (10 = perfect consistency).
        """
        logger.debug(
            f"calculate_consistency_score called: issues={len(issues)}, score_id={score_id}"
        )
        try:
            score = self._mode_service.calculate_consistency_score(issues)

            if score_id:
                self._mode_service.update_quality_scores(
                    score_id,
                    QualityScores(consistency_score=score),
                )

            logger.info(f"Consistency score calculated: {score:.1f}/10 ({len(issues)} issues)")
            return score
        except Exception as e:
            logger.error(f"Failed to calculate consistency score: {e}", exc_info=True)
            raise

    def _calculate_edit_distance(self, original: str, edited: str) -> int:
        """Calculate edit distance between two strings.

        Uses a simplified character-level difference count rather than
        full Levenshtein for performance.

        Args:
            original: Original content.
            edited: Edited content.

        Returns:
            Number of characters changed.
        """
        # Use SequenceMatcher for a quick similarity ratio
        matcher = SequenceMatcher(None, original, edited)
        ratio = matcher.ratio()

        # Convert to edit distance (roughly)
        max_len = max(len(original), len(edited))
        changes = int(max_len * (1 - ratio))

        return changes

    def get_active_score_id(
        self,
        chapter_id: str,
        agent_role: str = "writer",
    ) -> int | None:
        """Get the active score ID for a chapter/role combination.

        Args:
            chapter_id: The chapter ID.
            agent_role: The agent role.

        Returns:
            Score ID or None if not found.
        """
        validate_not_empty(chapter_id, "chapter_id")
        return self._active_scores.get(f"{chapter_id}:{agent_role}")

    def clear_chapter_tracking(self, chapter_id: str) -> None:
        """Clear tracking data for a completed chapter.

        This should be called when a chapter is finalized to prevent memory leaks.

        Args:
            chapter_id: The chapter ID to clear.
        """
        validate_not_empty(chapter_id, "chapter_id")
        logger.debug(f"clear_chapter_tracking called: chapter_id={chapter_id}")
        # Remove from active scores
        keys_to_remove = [k for k in self._active_scores if k.startswith(f"{chapter_id}:")]
        for key in keys_to_remove:
            del self._active_scores[key]

        # Remove original content
        self._original_content.pop(chapter_id, None)

        logger.debug(f"Cleared tracking data for chapter {chapter_id}")

    def _enforce_tracking_limits(self) -> None:
        """Enforce maximum tracking limits to prevent unbounded memory growth.

        This is a safety mechanism that keeps the most recent chapters and
        removes older ones when limits are exceeded.
        """
        # Check active scores
        if len(self._active_scores) > self.MAX_TRACKED_CHAPTERS:
            # Build an ordered list of unique chapter IDs based on first appearance
            chapter_ids_in_order: list[str] = []
            seen_chapters: set[str] = set()
            for key in self._active_scores:
                chapter_id = key.split(":", 1)[0]
                if chapter_id not in seen_chapters:
                    seen_chapters.add(chapter_id)
                    chapter_ids_in_order.append(chapter_id)

            excess_count = len(chapter_ids_in_order) - self.MAX_TRACKED_CHAPTERS

            if excess_count > 0:
                # Remove oldest chapters based on insertion order of _active_scores
                for chapter_id in chapter_ids_in_order[:excess_count]:
                    self.clear_chapter_tracking(chapter_id)
                    logger.warning(
                        f"Cleared tracking for old chapter {chapter_id} "
                        f"(exceeded MAX_TRACKED_CHAPTERS={self.MAX_TRACKED_CHAPTERS})"
                    )

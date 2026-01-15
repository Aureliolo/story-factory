"""Story service - handles story generation workflow."""

import logging
from collections.abc import Generator
from typing import Any

from memory.story_state import StoryBrief, StoryState
from memory.world_database import WorldDatabase
from settings import Settings
from workflows.orchestrator import StoryOrchestrator, WorkflowEvent

logger = logging.getLogger(__name__)


class StoryService:
    """Story generation workflow service.

    This service wraps the StoryOrchestrator to provide a clean interface
    for the UI layer. It handles interview, structure building, and
    chapter writing workflows.
    """

    def __init__(self, settings: Settings):
        """Initialize story service.

        Args:
            settings: Application settings.
        """
        self.settings = settings
        self._orchestrators: dict[str, StoryOrchestrator] = {}

    def _get_orchestrator(self, state: StoryState) -> StoryOrchestrator:
        """Get or create an orchestrator for a story state.

        Each story gets its own orchestrator instance to maintain
        conversation history and agent state.

        Args:
            state: The story state.

        Returns:
            StoryOrchestrator for this story.
        """
        if state.id not in self._orchestrators:
            orchestrator = StoryOrchestrator(settings=self.settings)
            orchestrator.story_state = state
            self._orchestrators[state.id] = orchestrator
        return self._orchestrators[state.id]

    def _sync_state(self, orchestrator: StoryOrchestrator, state: StoryState) -> None:
        """Sync orchestrator state back to the provided state object.

        Args:
            orchestrator: The orchestrator with potentially updated state.
            state: The state object to update.
        """
        if orchestrator.story_state:
            # Copy relevant fields back
            state.brief = orchestrator.story_state.brief
            state.characters = orchestrator.story_state.characters
            state.chapters = orchestrator.story_state.chapters
            state.plot_summary = orchestrator.story_state.plot_summary
            state.plot_points = orchestrator.story_state.plot_points
            state.world_description = orchestrator.story_state.world_description
            state.world_rules = orchestrator.story_state.world_rules
            state.established_facts = orchestrator.story_state.established_facts
            state.timeline = orchestrator.story_state.timeline
            state.current_chapter = orchestrator.story_state.current_chapter
            state.status = orchestrator.story_state.status

    # ========== INTERVIEW PHASE ==========

    def start_interview(self, state: StoryState) -> str:
        """Start the interview process.

        Args:
            state: The story state.

        Returns:
            Initial interview questions from the interviewer agent.
        """
        orchestrator = self._get_orchestrator(state)
        questions = orchestrator.start_interview()

        # Store in interview history
        state.interview_history.append(
            {
                "role": "assistant",
                "content": questions,
            }
        )

        return questions

    def process_interview(self, state: StoryState, user_message: str) -> tuple[str, bool]:
        """Process a user response in the interview.

        Args:
            state: The story state.
            user_message: User's response to interview questions.

        Returns:
            Tuple of (response_text, is_complete).
            is_complete is True when the brief has been generated.
        """
        orchestrator = self._get_orchestrator(state)

        # Store user message in history
        state.interview_history.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        response, is_complete = orchestrator.process_interview_response(user_message)

        # Store assistant response
        state.interview_history.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

        # Sync state back
        self._sync_state(orchestrator, state)

        return response, is_complete

    def finalize_interview(self, state: StoryState) -> StoryBrief:
        """Force finalize the interview with current information.

        Args:
            state: The story state.

        Returns:
            The generated StoryBrief.
        """
        orchestrator = self._get_orchestrator(state)
        brief = orchestrator.finalize_interview()
        self._sync_state(orchestrator, state)
        return brief

    def continue_interview(self, state: StoryState, additional_info: str) -> str:
        """Continue an already-completed interview with additional information.

        This allows users to add clarifications or changes after the initial
        interview is complete. The interviewer should be apprehensive of big
        changes but allow small adjustments.

        Args:
            state: The story state (must have a brief already).
            additional_info: Additional information or changes from the user.

        Returns:
            Response from the interviewer acknowledging changes.
        """
        if not state.brief:
            raise ValueError("Cannot continue interview - no brief exists yet.")

        orchestrator = self._get_orchestrator(state)

        # Store user message
        state.interview_history.append(
            {
                "role": "user",
                "content": additional_info,
            }
        )

        # Process as a continuation
        response, _ = orchestrator.process_interview_response(additional_info)

        # Store response
        state.interview_history.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

        self._sync_state(orchestrator, state)
        return response

    # ========== STRUCTURE PHASE ==========

    def build_structure(self, state: StoryState, world_db: WorldDatabase) -> StoryState:
        """Build the story structure and extract entities to world database.

        Args:
            state: The story state with completed brief.
            world_db: WorldDatabase to populate with extracted entities.

        Returns:
            Updated StoryState with structure.
        """
        if not state.brief:
            raise ValueError("Cannot build structure - no brief exists.")

        orchestrator = self._get_orchestrator(state)
        orchestrator.build_story_structure()
        self._sync_state(orchestrator, state)

        # Extract entities to world database
        self._extract_entities_to_world(state, world_db)

        return state

    def _extract_entities_to_world(self, state: StoryState, world_db: WorldDatabase) -> None:
        """Extract characters and locations from story state to world database.

        Args:
            state: Story state with characters.
            world_db: WorldDatabase to populate.
        """
        # Extract characters
        for char in state.characters:
            # Check if already exists
            existing = world_db.search_entities(char.name, entity_type="character")
            if existing:
                continue

            attributes = {
                "role": char.role,
                "personality_traits": char.personality_traits,
                "goals": char.goals,
                "arc_notes": char.arc_notes,
            }

            entity_id = world_db.add_entity(
                entity_type="character",
                name=char.name,
                description=char.description,
                attributes=attributes,
            )

            # Add relationships from character data
            for related_name, relationship in char.relationships.items():
                # Find the related character
                related_entities = world_db.search_entities(related_name, entity_type="character")
                if related_entities:
                    world_db.add_relationship(
                        source_id=entity_id,
                        target_id=related_entities[0].id,
                        relation_type=relationship,
                    )

        logger.info(f"Extracted {len(state.characters)} characters to world database")

    def get_outline(self, state: StoryState) -> str:
        """Get a formatted story outline.

        Args:
            state: The story state.

        Returns:
            Human-readable outline string.
        """
        orchestrator = self._get_orchestrator(state)
        orchestrator.story_state = state
        return orchestrator.get_outline_summary()

    # ========== WRITING PHASE ==========

    def write_chapter(
        self,
        state: StoryState,
        chapter_num: int,
        feedback: str | None = None,
    ) -> Generator[WorkflowEvent, None, str]:
        """Write a single chapter with streaming events.

        Args:
            state: The story state.
            chapter_num: Chapter number to write.
            feedback: Optional feedback to incorporate.

        Yields:
            WorkflowEvent objects for progress updates.

        Returns:
            The completed chapter content.
        """
        orchestrator = self._get_orchestrator(state)
        orchestrator.story_state = state

        # Write the chapter
        content = ""
        for event in orchestrator.write_chapter(chapter_num):
            yield event
            # The generator returns the content at the end
            if event.event_type == "agent_complete" and event.agent_name == "System":
                # Get the chapter content
                chapter = next((c for c in state.chapters if c.number == chapter_num), None)
                if chapter:
                    content = chapter.content

        self._sync_state(orchestrator, state)
        return content

    def write_all_chapters(self, state: StoryState) -> Generator[WorkflowEvent]:
        """Write all chapters with streaming events.

        Args:
            state: The story state.

        Yields:
            WorkflowEvent objects for progress updates.
        """
        orchestrator = self._get_orchestrator(state)
        orchestrator.story_state = state

        yield from orchestrator.write_all_chapters()
        self._sync_state(orchestrator, state)

    def write_short_story(self, state: StoryState) -> Generator[WorkflowEvent, None, str]:
        """Write a short story (single chapter).

        Args:
            state: The story state.

        Yields:
            WorkflowEvent objects for progress updates.

        Returns:
            The completed story content.
        """
        orchestrator = self._get_orchestrator(state)
        orchestrator.story_state = state

        content = ""
        for event in orchestrator.write_short_story():
            yield event
            if event.event_type == "agent_complete" and event.agent_name == "System":
                if state.chapters:
                    content = state.chapters[0].content

        self._sync_state(orchestrator, state)
        return content

    def get_full_story(self, state: StoryState) -> str:
        """Get the complete story text.

        Args:
            state: The story state.

        Returns:
            Full story text with all chapters.
        """
        orchestrator = self._get_orchestrator(state)
        orchestrator.story_state = state
        return orchestrator.get_full_story()

    def get_chapter_content(self, state: StoryState, chapter_num: int) -> str | None:
        """Get content of a specific chapter.

        Args:
            state: The story state.
            chapter_num: Chapter number.

        Returns:
            Chapter content or None if not found/written.
        """
        chapter = next((c for c in state.chapters if c.number == chapter_num), None)
        return chapter.content if chapter else None

    def get_statistics(self, state: StoryState) -> dict[str, Any]:
        """Get story statistics.

        Args:
            state: The story state.

        Returns:
            Dictionary with statistics including word count, chapters, etc.
        """
        orchestrator = self._get_orchestrator(state)
        orchestrator.story_state = state
        return orchestrator.get_statistics()

    # ========== FEEDBACK & REVIEWS ==========

    def add_review(
        self, state: StoryState, review_type: str, content: str, chapter_num: int | None = None
    ) -> None:
        """Add a review or note to the story.

        Args:
            state: The story state.
            review_type: Type of review ("user_note", "ai_suggestion", "marked_for_review").
            content: The review content.
            chapter_num: Optional chapter number this review applies to.
        """
        state.reviews.append(
            {
                "type": review_type,
                "content": content,
                "chapter": chapter_num,
                "timestamp": str(state.updated_at),
            }
        )

    def get_reviews(
        self, state: StoryState, chapter_num: int | None = None
    ) -> list[dict[str, Any]]:
        """Get reviews for a story or specific chapter.

        Args:
            state: The story state.
            chapter_num: Optional chapter number to filter by.

        Returns:
            List of review dictionaries.
        """
        if chapter_num is None:
            return state.reviews
        return [r for r in state.reviews if r.get("chapter") == chapter_num]

    # ========== TITLE GENERATION ==========

    def generate_title_suggestions(self, state: StoryState) -> list[str]:
        """Generate AI-powered title suggestions.

        Args:
            state: The story state with brief.

        Returns:
            List of suggested titles.
        """
        orchestrator = self._get_orchestrator(state)
        orchestrator.story_state = state
        return orchestrator.generate_title_suggestions()

    # ========== CLEANUP ==========

    def cleanup_orchestrator(self, state: StoryState) -> None:
        """Clean up orchestrator for a story (free memory).

        Args:
            state: The story state.
        """
        if state.id in self._orchestrators:
            del self._orchestrators[state.id]
            logger.debug(f"Cleaned up orchestrator for story {state.id}")

"""Interview phase mixin for StoryService."""

import logging

from src.memory.story_state import StoryBrief, StoryState
from src.utils.validation import (
    validate_not_empty,
    validate_not_none,
    validate_type,
)

from ._base import StoryServiceBase

logger = logging.getLogger(__name__)


class InterviewMixin(StoryServiceBase):
    """Mixin providing interview phase functionality."""

    def start_interview(self, state: StoryState) -> str:
        """
        Begin the interview for the given story and record the assistant's initial questions in the state's interview history.

        Parameters:
            state (StoryState): The story state representing the project to interview.

        Returns:
            str: The assistant's initial interview questions.
        """
        validate_not_none(state, "state")
        validate_type(state, "state", StoryState)
        logger.debug(f"start_interview called: project_id={state.id}")
        try:
            orchestrator = self._get_orchestrator(state)
            questions = orchestrator.start_interview()

            # Store in interview history
            state.interview_history.append(
                {
                    "role": "assistant",
                    "content": questions,
                }
            )

            logger.info(f"Interview started for project {state.id}")
            return questions
        except Exception as e:
            logger.error(f"Failed to start interview for project {state.id}: {e}", exc_info=True)
            raise

    def process_interview(self, state: StoryState, user_message: str) -> tuple[str, bool]:
        """Process a user response in the interview.

        Args:
            state: The story state.
            user_message: User's response to interview questions.

        Returns:
            Tuple of (response_text, is_complete).
            is_complete is True when the brief has been generated.
        """
        validate_not_none(state, "state")
        validate_type(state, "state", StoryState)
        validate_not_empty(user_message, "user_message")
        logger.debug(
            f"process_interview called: project_id={state.id}, message_length={len(user_message)}"
        )
        try:
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

            if is_complete:
                logger.info(f"Interview completed for project {state.id}")
            else:
                logger.debug(f"Interview in progress for project {state.id}")

            return response, is_complete
        except Exception as e:
            logger.error(f"Failed to process interview for project {state.id}: {e}", exc_info=True)
            raise

    def finalize_interview(self, state: StoryState) -> StoryBrief:
        """Force finalize the interview with current information.

        Args:
            state: The story state.

        Returns:
            The generated StoryBrief.
        """
        validate_not_none(state, "state")
        validate_type(state, "state", StoryState)
        logger.debug(f"finalize_interview called: project_id={state.id}")
        try:
            orchestrator = self._get_orchestrator(state)
            brief = orchestrator.finalize_interview()
            self._sync_state(orchestrator, state)
            logger.info(
                f"Interview finalized for project {state.id}: genre={brief.genre}, tone={brief.tone}"
            )
            return brief
        except Exception as e:
            logger.error(f"Failed to finalize interview for project {state.id}: {e}", exc_info=True)
            raise

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

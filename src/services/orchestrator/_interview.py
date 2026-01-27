"""Interview phase mixin for StoryOrchestrator."""

import logging

from src.memory.story_state import StoryBrief
from src.services.orchestrator._base import StoryOrchestratorBase
from src.utils.message_analyzer import analyze_message, format_inference_context

logger = logging.getLogger(__name__)


class InterviewMixin(StoryOrchestratorBase):
    """Mixin providing interview phase functionality."""

    def start_interview(self) -> str:
        """Start the interview process."""
        self._set_phase("interview")
        self._emit("agent_start", "Interviewer", "Starting interview...")
        questions = self.interviewer.get_initial_questions()
        self._emit("agent_complete", "Interviewer", "Initial questions ready")
        return questions

    def process_interview_response(self, user_response: str) -> tuple[str, bool]:
        """Process user response and return next questions or indicate completion.

        Returns: (response_text, is_complete)
        """
        if not self.story_state:
            raise ValueError("No story state. Call create_new_story() first.")

        self._emit("agent_start", "Interviewer", "Processing your response...")

        # Analyze the user message to infer language and content rating
        analysis = analyze_message(user_response)
        context = format_inference_context(analysis)

        response = self.interviewer.process_response(user_response, context=context)

        # Check if a brief was generated
        brief = self.interviewer.extract_brief(response)
        if brief:
            self.story_state.brief = brief
            self.story_state.status = "outlining"
            self._emit("agent_complete", "Interviewer", "Story brief created!")
            return response, True

        self._emit("agent_complete", "Interviewer", "Follow-up questions ready")
        return response, False

    def finalize_interview(self) -> StoryBrief:
        """Force finalize the interview with current information."""
        if not self.story_state:
            raise ValueError("No story state. Call create_new_story() first.")

        history = "\n".join(
            f"{h['role']}: {h['content']}" for h in self.interviewer.conversation_history
        )
        brief = self.interviewer.finalize_brief(history)
        self.story_state.brief = brief
        self.story_state.status = "outlining"
        return brief

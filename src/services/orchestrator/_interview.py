"""Interview phase functions for StoryOrchestrator."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.memory.story_state import StoryBrief
from src.utils.message_analyzer import analyze_message, format_inference_context

if TYPE_CHECKING:
    from . import StoryOrchestrator

logger = logging.getLogger(__name__)


def start_interview(orc: StoryOrchestrator) -> str:
    """Start the interview process.

    Args:
        orc: StoryOrchestrator instance.

    Returns:
        The initial interview questions from the Interviewer agent.
    """
    orc._set_phase("interview")
    orc._emit("agent_start", "Interviewer", "Starting interview...")
    questions = orc.interviewer.get_initial_questions()
    orc._emit("agent_complete", "Interviewer", "Initial questions ready")
    return questions


def process_interview_response(orc: StoryOrchestrator, user_response: str) -> tuple[str, bool]:
    """Process user response and return next questions or indicate completion.

    Args:
        orc: StoryOrchestrator instance.
        user_response: The user's answer to interview questions.

    Returns:
        Tuple of (response_text, is_complete).
    """
    if not orc.story_state:
        raise ValueError("No story state. Call create_new_story() first.")

    orc._emit("agent_start", "Interviewer", "Processing your response...")

    # Analyze the user message to infer language and content rating
    analysis = analyze_message(user_response)
    context = format_inference_context(analysis)

    response = orc.interviewer.process_response(user_response, context=context)

    # Check if a brief was generated
    brief = orc.interviewer.extract_brief(response)
    if brief:
        orc.story_state.brief = brief
        orc.story_state.status = "outlining"
        orc._emit("agent_complete", "Interviewer", "Story brief created!")
        return response, True

    orc._emit("agent_complete", "Interviewer", "Follow-up questions ready")
    return response, False


def finalize_interview(orc: StoryOrchestrator) -> StoryBrief:
    """Force finalize the interview with current information.

    Args:
        orc: StoryOrchestrator instance.

    Returns:
        The finalized StoryBrief.
    """
    if not orc.story_state:
        raise ValueError("No story state. Call create_new_story() first.")

    history = "\n".join(
        f"{h['role']}: {h['content']}" for h in orc.interviewer.conversation_history
    )
    brief = orc.interviewer.finalize_brief(history)
    orc.story_state.brief = brief
    orc.story_state.status = "outlining"
    return brief

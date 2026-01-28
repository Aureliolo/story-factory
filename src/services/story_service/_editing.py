"""Editing, continuation, review, and world generation functions for StoryService."""

import logging
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

from src.agents.continuity import ContinuityIssue
from src.memory.story_state import Character, StoryState
from src.services.orchestrator import WorkflowEvent

if TYPE_CHECKING:
    from src.services.story_service import StoryService

logger = logging.getLogger(__name__)


def continue_chapter(
    svc: StoryService,
    state: StoryState,
    chapter_num: int,
    direction: str | None = None,
) -> Generator[WorkflowEvent, None, str]:
    """Continue writing a chapter from where it left off.

    Parameters:
        svc: The StoryService instance.
        state: The story state.
        chapter_num: Chapter number to continue.
        direction: Optional direction for where to take the scene.

    Yields:
        WorkflowEvent objects for progress updates.

    Returns:
        The continuation text.
    """
    orchestrator = svc._get_orchestrator(state)
    orchestrator.story_state = state

    continuation = ""
    for event in orchestrator.continue_chapter(chapter_num, direction):
        yield event
        if event.event_type == "agent_complete" and event.agent_name == "Writer":
            # Get updated chapter content
            chapter = next((c for c in state.chapters if c.number == chapter_num), None)
            if chapter:
                continuation = event.data.get("continuation", "") if event.data else ""

    svc._sync_state(orchestrator, state)
    return continuation


def edit_passage(
    svc: StoryService,
    state: StoryState,
    text: str,
    focus: str | None = None,
) -> Generator[WorkflowEvent, None, str]:
    """Edit a specific passage with optional focus area.

    Parameters:
        svc: The StoryService instance.
        state: The story state.
        text: The text passage to edit.
        focus: Optional focus area (e.g., "dialogue", "pacing").

    Yields:
        WorkflowEvent objects for progress updates.

    Returns:
        The edited passage.
    """
    orchestrator = svc._get_orchestrator(state)
    orchestrator.story_state = state

    edited = ""
    for event in orchestrator.edit_passage(text, focus):
        yield event
        if event.event_type == "agent_complete":
            # The edited text is returned from the generator
            pass

    # The generator returns the edited text at the end
    # We need to call it fully and capture the return
    return edited


def get_edit_suggestions(
    svc: StoryService,
    state: StoryState,
    text: str,
) -> Generator[WorkflowEvent, None, str]:
    """Get editing suggestions without making changes.

    Parameters:
        svc: The StoryService instance.
        state: The story state.
        text: The text to review.

    Yields:
        WorkflowEvent objects for progress updates.

    Returns:
        Suggestions for improving the text.
    """
    orchestrator = svc._get_orchestrator(state)

    suggestions = ""
    for event in orchestrator.get_edit_suggestions(text):
        yield event
        if event.event_type == "agent_complete" and event.data:
            suggestions = event.data.get("suggestions", "")

    return suggestions


def review_full_story(
    svc: StoryService,
    state: StoryState,
) -> Generator[WorkflowEvent, None, list[ContinuityIssue]]:
    """Perform a full story continuity review.

    Parameters:
        svc: The StoryService instance.
        state: The story state.

    Yields:
        WorkflowEvent objects for progress updates.

    Returns:
        List of ContinuityIssue objects (as dicts).
    """
    orchestrator = svc._get_orchestrator(state)
    orchestrator.story_state = state

    issues = []
    for event in orchestrator.review_full_story():
        yield event
        if event.event_type == "agent_complete" and event.data:
            issues = event.data.get("issues", [])

    return issues


def generate_more_characters(
    svc: StoryService, state: StoryState, count: int = 2
) -> list[Character]:
    """Generate additional characters for the story.

    Parameters:
        svc: The StoryService instance.
        state: The story state.
        count: Number of characters to generate.

    Returns:
        List of new Character objects.
    """
    orchestrator = svc._get_orchestrator(state)
    orchestrator.story_state = state
    new_chars = orchestrator.generate_more_characters(count)
    svc._sync_state(orchestrator, state)
    return new_chars


def generate_locations(svc: StoryService, state: StoryState, count: int = 3) -> list[Any]:
    """Generate locations for the story world.

    Parameters:
        svc: The StoryService instance.
        state: The story state.
        count: Number of locations to generate.

    Returns:
        List of location dictionaries.
    """
    orchestrator = svc._get_orchestrator(state)
    orchestrator.story_state = state
    locations = orchestrator.generate_locations(count)
    svc._sync_state(orchestrator, state)
    return locations


def generate_relationships(
    svc: StoryService,
    state: StoryState,
    entity_names: list[str],
    existing_rels: list[tuple[str, str]],
    count: int = 5,
) -> list[Any]:
    """Generate relationships between entities.

    Parameters:
        svc: The StoryService instance.
        state: The story state.
        entity_names: Names of all entities.
        existing_rels: Existing (source, target) relationships.
        count: Number of relationships to generate.

    Returns:
        List of relationship dictionaries.
    """
    orchestrator = svc._get_orchestrator(state)
    orchestrator.story_state = state
    relationships = orchestrator.generate_relationships(entity_names, existing_rels, count)
    svc._sync_state(orchestrator, state)
    return relationships


def rebuild_world(svc: StoryService, state: StoryState) -> StoryState:
    """Rebuild the entire world from scratch.

    Parameters:
        svc: The StoryService instance.
        state: The story state.

    Returns:
        Updated StoryState.
    """
    orchestrator = svc._get_orchestrator(state)
    orchestrator.story_state = state
    orchestrator.rebuild_world()
    svc._sync_state(orchestrator, state)
    return state

"""Story service package - handles story generation workflow.

This package provides the StoryService class composed from specialized mixins:
- StoryServiceBase: Core functionality, orchestrator management
- InterviewMixin: Interview phase functionality
- StructureMixin: Story structure and outline generation
- WritingMixin: Chapter writing functionality
- EditingMixin: Editing and review functionality
- GenerationMixin: Title and world generation
"""

from ._base import GenerationCancelled, StoryServiceBase
from ._editing import EditingMixin
from ._generation import GenerationMixin
from ._interview import InterviewMixin
from ._structure import StructureMixin
from ._writing import WritingMixin


class StoryService(
    InterviewMixin,
    StructureMixin,
    WritingMixin,
    EditingMixin,
    GenerationMixin,
    StoryServiceBase,
):
    """Story generation workflow service.

    This service wraps the StoryOrchestrator to provide a clean interface
    for the UI layer. It handles interview, structure building, and
    chapter writing workflows.

    Composed from:
    - StoryServiceBase: Core functionality and orchestrator management
    - InterviewMixin: start_interview(), process_interview(), finalize_interview(), continue_interview()
    - StructureMixin: build_structure(), generate_outline_variations(), select_variation(), etc.
    - WritingMixin: write_chapter(), write_all_chapters(), regenerate_chapter_with_feedback(), etc.
    - EditingMixin: continue_chapter(), edit_passage(), get_edit_suggestions(), review_full_story()
    - GenerationMixin: generate_title_suggestions(), generate_more_characters(), etc.
    """

    pass


__all__ = [
    "GenerationCancelled",
    "StoryService",
]

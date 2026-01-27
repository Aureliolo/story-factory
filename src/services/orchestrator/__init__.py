"""Orchestrator package - coordinates all agents for story generation.

This package provides the StoryOrchestrator class composed from specialized mixins:
- StoryOrchestratorBase: Core functionality, agents, progress tracking
- InterviewMixin: Interview phase functionality
- StructureMixin: Architecture/structure phase functionality
- WritingMixin: Writing phase functionality
- EditingMixin: Editing phase functionality
- ExportMixin: Export functionality
- PersistenceMixin: Save/load functionality
"""

from ._base import StoryOrchestratorBase, WorkflowEvent
from ._editing import EditingMixin
from ._export import ExportMixin
from ._interview import InterviewMixin
from ._persistence import PersistenceMixin
from ._structure import StructureMixin
from ._writing import WritingMixin


class StoryOrchestrator(
    InterviewMixin,
    StructureMixin,
    WritingMixin,
    EditingMixin,
    ExportMixin,
    PersistenceMixin,
    StoryOrchestratorBase,
):
    """Orchestrates the story generation workflow.

    Composed from:
    - StoryOrchestratorBase: Core functionality, agents, progress tracking
    - InterviewMixin: start_interview(), process_interview_response(), finalize_interview()
    - StructureMixin: build_story_structure(), generate_more_characters(), etc.
    - WritingMixin: write_chapter(), write_all_chapters(), write_short_story()
    - EditingMixin: continue_chapter(), edit_passage(), get_edit_suggestions()
    - ExportMixin: export_to_markdown(), export_to_epub(), export_to_pdf(), etc.
    - PersistenceMixin: save_story(), load_story(), autosave()
    """

    pass


__all__ = [
    "StoryOrchestrator",
    "WorkflowEvent",
]

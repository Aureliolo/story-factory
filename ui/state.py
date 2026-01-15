"""Centralized UI state management."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from memory.story_state import StoryState
from memory.world_database import WorldDatabase


@dataclass
class AppState:
    """Centralized UI state.

    This class holds all UI state in one place, making it easy to:
    - Track what's currently selected/active
    - Pass state between components
    - Update UI reactively

    Usage:
        state = AppState()
        state.project_id = "abc-123"
        state.project, state.world_db = services.project.load_project(state.project_id)
    """

    # ========== Current Project ==========
    project_id: str | None = None
    project: StoryState | None = None
    world_db: WorldDatabase | None = None

    # ========== Interview State ==========
    interview_history: list[dict[str, str]] = field(default_factory=list)
    interview_complete: bool = False
    interview_processing: bool = False  # True while waiting for AI response

    # ========== Writing State ==========
    current_chapter: int = 0
    is_writing: bool = False  # True while generating content
    writing_progress: str = ""  # Current progress message

    # ========== UI Navigation ==========
    active_tab: str = "write"  # write, world, projects, settings, models
    active_sub_tab: str = "fundamentals"  # fundamentals, writing (for write tab)

    # ========== World Builder State ==========
    selected_entity_id: str | None = None
    entity_filter_types: list[str] = field(default_factory=lambda: ["character", "location"])
    graph_layout: str = "force-directed"  # force-directed, hierarchical, circular
    entity_search_query: str = ""

    # ========== Feedback Settings ==========
    feedback_mode: str = "per-chapter"  # per-chapter, mid-chapter, on-demand

    # ========== Callbacks ==========
    # These are called when certain state changes occur
    _on_project_change: Callable[[], None] | None = None
    _on_entity_select: Callable[[str], None] | None = None
    _on_chapter_change: Callable[[int], None] | None = None

    def set_project(
        self,
        project_id: str,
        project: StoryState,
        world_db: WorldDatabase,
    ) -> None:
        """Set the current project and trigger callbacks.

        Args:
            project_id: Project UUID.
            project: StoryState instance.
            world_db: WorldDatabase instance.
        """
        self.project_id = project_id
        self.project = project
        self.world_db = world_db

        # Sync interview state
        self.interview_history = project.interview_history.copy()
        self.interview_complete = project.status != "interview"

        # Sync chapter state
        self.current_chapter = project.current_chapter

        # Trigger callback
        if self._on_project_change:
            self._on_project_change()

    def clear_project(self) -> None:
        """Clear the current project."""
        self.project_id = None
        self.project = None
        self.world_db = None
        self.interview_history = []
        self.interview_complete = False
        self.current_chapter = 0
        self.selected_entity_id = None

        if self._on_project_change:
            self._on_project_change()

    def select_entity(self, entity_id: str | None) -> None:
        """Select an entity and trigger callback.

        Args:
            entity_id: Entity ID to select, or None to deselect.
        """
        self.selected_entity_id = entity_id
        if self._on_entity_select and entity_id:
            self._on_entity_select(entity_id)

    def select_chapter(self, chapter_num: int) -> None:
        """Select a chapter and trigger callback.

        Args:
            chapter_num: Chapter number to select.
        """
        self.current_chapter = chapter_num
        if self._on_chapter_change:
            self._on_chapter_change(chapter_num)

    def add_interview_message(self, role: str, content: str) -> None:
        """Add a message to interview history.

        Args:
            role: 'user' or 'assistant'.
            content: Message content.
        """
        self.interview_history.append({"role": role, "content": content})

        # Sync to project if exists
        if self.project:
            self.project.interview_history = self.interview_history.copy()

    def on_project_change(self, callback: Callable[[], None]) -> None:
        """Register callback for project changes.

        Args:
            callback: Function to call when project changes.
        """
        self._on_project_change = callback

    def on_entity_select(self, callback: Callable[[str], None]) -> None:
        """Register callback for entity selection.

        Args:
            callback: Function to call with entity ID when selected.
        """
        self._on_entity_select = callback

    def on_chapter_change(self, callback: Callable[[int], None]) -> None:
        """Register callback for chapter changes.

        Args:
            callback: Function to call with chapter number.
        """
        self._on_chapter_change = callback

    @property
    def has_project(self) -> bool:
        """Check if a project is currently loaded."""
        return self.project is not None

    @property
    def can_write(self) -> bool:
        """Check if writing is possible (structure exists)."""
        return (
            self.project is not None
            and self.project.brief is not None
            and len(self.project.chapters) > 0
        )

    @property
    def project_name(self) -> str:
        """Get current project name or default."""
        if self.project:
            return self.project.project_name
        return "No Project Selected"

    @property
    def project_status(self) -> str:
        """Get current project status."""
        if self.project:
            return self.project.status
        return "none"

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary (for debugging/logging).

        Returns:
            Dictionary representation of state.
        """
        return {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "project_status": self.project_status,
            "interview_complete": self.interview_complete,
            "current_chapter": self.current_chapter,
            "active_tab": self.active_tab,
            "active_sub_tab": self.active_sub_tab,
            "selected_entity_id": self.selected_entity_id,
            "feedback_mode": self.feedback_mode,
        }

"""Centralized UI state management."""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.memory.story_state import StoryState
from src.memory.world_database import WorldDatabase
from src.utils.exceptions import BackgroundTaskActiveError

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of undoable actions."""

    # World/Entity actions
    ADD_ENTITY = "add_entity"
    DELETE_ENTITY = "delete_entity"
    UPDATE_ENTITY = "update_entity"
    ADD_RELATIONSHIP = "add_relationship"
    DELETE_RELATIONSHIP = "delete_relationship"
    UPDATE_RELATIONSHIP = "update_relationship"

    # Write/Chapter actions
    UPDATE_CHAPTER_CONTENT = "update_chapter_content"
    DELETE_CHAPTER = "delete_chapter"
    ADD_CHAPTER = "add_chapter"
    UPDATE_CHAPTER_FEEDBACK = "update_chapter_feedback"

    # Settings actions
    UPDATE_SETTINGS = "update_settings"


@dataclass
class UndoAction:
    """Represents an undoable action."""

    action_type: ActionType
    data: dict[str, Any]
    inverse_data: dict[str, Any]  # Data needed to reverse the action


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
    generation_cancel_requested: bool = False  # Request cancellation of current generation
    generation_pause_requested: bool = False  # Request pause of current generation
    generation_is_paused: bool = False  # Generation is currently paused
    generation_can_resume: bool = False  # Generation can be resumed

    # ========== UI Navigation ==========
    active_tab: str = "write"  # write, world, projects, settings, models
    active_sub_tab: str = "fundamentals"  # fundamentals, writing (for write tab)

    # ========== World Builder State ==========
    selected_entity_id: str | None = None
    entity_filter_types: list[str] = field(
        default_factory=lambda: ["character", "location", "item", "faction", "concept"]
    )
    graph_layout: str = "force-directed"  # force-directed, hierarchical, circular
    entity_search_query: str = ""
    quality_refinement_enabled: bool = True  # Whether to use quality refinement for generation

    # Enhanced entity browser filters (Issue #182)
    entity_quality_filter: str = "all"  # all, high, medium, low
    entity_sort_by: str = "type"  # name, type, quality, created, relationships
    entity_sort_descending: bool = False
    entity_search_names: bool = True
    entity_search_descriptions: bool = True

    # ========== Feedback Settings ==========
    feedback_mode: str = "per-chapter"  # per-chapter, mid-chapter, on-demand

    # ========== Undo/Redo History ==========
    _undo_stack: list[UndoAction] = field(default_factory=list)
    _redo_stack: list[UndoAction] = field(default_factory=list)
    _max_undo_history: int = 50

    # ========== Background Task Tracking ==========
    _background_task_count: int = 0
    _background_task_lock: threading.Lock = field(default_factory=threading.Lock)

    # ========== Project List Cache ==========
    _project_list_cache: list[Any] | None = field(default=None, repr=False)
    _project_list_cache_time: float = 0.0

    # ========== Callbacks ==========
    # These are called when certain state changes occur
    _on_project_change: Callable[[], None] | None = None
    _on_entity_select: Callable[[str], None] | None = None
    _on_chapter_change: Callable[[int], None] | None = None
    _on_undo: Callable[[], None] | None = None  # Called when undo is requested
    _on_redo: Callable[[], None] | None = None  # Called when redo is requested

    def begin_background_task(self, task_name: str) -> None:
        """Register a background task as active.

        Args:
            task_name: Human-readable name for logging.
        """
        with self._background_task_lock:
            self._background_task_count += 1
            logger.debug(
                "Background task started: %s (active: %d)",
                task_name,
                self._background_task_count,
            )

    def end_background_task(self, task_name: str) -> None:
        """Mark a background task as finished.

        Args:
            task_name: Human-readable name for logging.
        """
        with self._background_task_lock:
            if self._background_task_count == 0:
                logger.warning(
                    "end_background_task called with no active tasks: %s",
                    task_name,
                )
            else:
                self._background_task_count -= 1
            logger.debug(
                "Background task ended: %s (active: %d)",
                task_name,
                self._background_task_count,
            )

    @property
    def is_busy(self) -> bool:
        """Check if any background tasks are currently running."""
        with self._background_task_lock:
            return self._background_task_count > 0

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

        Raises:
            BackgroundTaskActiveError: If background tasks are still running.
        """
        if self.is_busy:
            raise BackgroundTaskActiveError(
                "Cannot switch projects while background tasks are running. "
                "Wait for builds/generation to finish."
            )

        # Close existing database connection before switching
        if self.world_db is not None:
            self.world_db.close()

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
        """Clear the current project.

        Raises:
            BackgroundTaskActiveError: If background tasks are still running.
        """
        if self.is_busy:
            raise BackgroundTaskActiveError(
                "Cannot clear project while background tasks are running. "
                "Wait for builds/generation to finish."
            )

        # Close database connection before clearing
        if self.world_db is not None:
            self.world_db.close()

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

    def on_undo(self, callback: Callable[[], None]) -> None:
        """Register callback for undo requests.

        Args:
            callback: Function to call when undo is triggered.
        """
        self._on_undo = callback

    def on_redo(self, callback: Callable[[], None]) -> None:
        """Register callback for redo requests.

        Args:
            callback: Function to call when redo is triggered.
        """
        self._on_redo = callback

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

    # ========== Undo/Redo Methods ==========

    def record_action(self, action: UndoAction) -> None:
        """Record an action for undo/redo.

        Args:
            action: The action to record.
        """
        self._undo_stack.append(action)
        self._redo_stack.clear()  # Clear redo on new action

        # Limit history size
        if len(self._undo_stack) > self._max_undo_history:
            self._undo_stack.pop(0)

        logger.debug(f"Recorded action: {action.action_type.value}")

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self._redo_stack) > 0

    def undo(self) -> UndoAction | None:
        """Pop the last action from undo stack.

        Returns:
            The action to undo, or None if stack is empty.
        """
        if not self._undo_stack:
            return None

        action = self._undo_stack.pop()
        self._redo_stack.append(action)
        logger.debug(f"Undo: {action.action_type.value}")
        return action

    def redo(self) -> UndoAction | None:
        """Pop the last action from redo stack.

        Returns:
            The action to redo, or None if stack is empty.
        """
        if not self._redo_stack:
            return None

        action = self._redo_stack.pop()
        self._undo_stack.append(action)
        logger.debug(f"Redo: {action.action_type.value}")
        return action

    def clear_history(self) -> None:
        """Clear undo/redo history."""
        self._undo_stack.clear()
        self._redo_stack.clear()

    def trigger_undo(self) -> None:
        """Trigger undo action by calling registered callback.

        This is called by global keyboard shortcuts.
        """
        if self._on_undo and self.can_undo():
            self._on_undo()
            logger.debug("Undo triggered via global shortcut")

    def trigger_redo(self) -> None:
        """Trigger redo action by calling registered callback.

        This is called by global keyboard shortcuts.
        """
        if self._on_redo and self.can_redo():
            self._on_redo()
            logger.debug("Redo triggered via global shortcut")

    # ========== Generation Control Methods ==========

    def request_cancel_generation(self) -> None:
        """Request cancellation of the current generation."""
        self.generation_cancel_requested = True
        logger.info("Generation cancellation requested")

    def request_pause_generation(self) -> None:
        """Request pause of the current generation."""
        self.generation_pause_requested = True
        logger.info("Generation pause requested")

    def resume_generation(self) -> None:
        """Resume a paused generation."""
        self.generation_is_paused = False
        self.generation_pause_requested = False
        logger.info("Generation resumed")

    def reset_generation_flags(self) -> None:
        """Reset all generation control flags."""
        self.generation_cancel_requested = False
        self.generation_pause_requested = False
        self.generation_is_paused = False
        self.generation_can_resume = False
        logger.debug("Generation flags reset")

    # ========== Project List Cache Methods ==========

    def get_cached_projects(self, fetch_fn: Callable[[], list]) -> list:
        """Get projects from cache or fetch if stale (>2s).

        Args:
            fetch_fn: Function to call to fetch fresh project list.

        Returns:
            List of projects (cached or freshly fetched).
        """
        now = time.time()
        if self._project_list_cache is not None and (now - self._project_list_cache_time) < 2.0:
            logger.debug(
                "Returning cached project list (%d projects)", len(self._project_list_cache)
            )
            return self._project_list_cache

        self._project_list_cache = fetch_fn()
        self._project_list_cache_time = now
        logger.debug("Refreshed project list cache with %d projects", len(self._project_list_cache))
        return self._project_list_cache

    def invalidate_project_cache(self) -> None:
        """Invalidate project list cache after mutations."""
        self._project_list_cache = None
        logger.debug("Project list cache invalidated")

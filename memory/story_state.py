"""Story state management - maintains context across the generation process."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Character(BaseModel):
    """A character in the story."""

    name: str
    role: str  # protagonist, antagonist, supporting, etc.
    description: str
    personality_traits: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    relationships: dict[str, str] = Field(default_factory=dict)  # character_name -> relationship
    arc_notes: str = ""  # How the character should develop
    arc_progress: dict[int, str] = Field(default_factory=dict)  # chapter_number -> arc state

    def update_arc(self, chapter_number: int, state: str):
        """Update character arc progress for a chapter."""
        self.arc_progress[chapter_number] = state

    def get_arc_summary(self) -> str:
        """Get a summary of character arc progression."""
        if not self.arc_progress:
            return f"Arc: {self.arc_notes}" if self.arc_notes else ""

        summary_parts = [f"Arc plan: {self.arc_notes}"] if self.arc_notes else []
        for chapter, state in sorted(self.arc_progress.items()):
            summary_parts.append(f"  Ch{chapter}: {state}")
        return "\n".join(summary_parts)


class PlotPoint(BaseModel):
    """A key plot point in the story."""

    description: str
    chapter: int | None = None
    completed: bool = False
    foreshadowing_planted: bool = False


class Scene(BaseModel):
    """A scene within a chapter.

    Note on similar fields:
    - `order` is the canonical position field (use for reordering)
    - `goal` is a one-sentence purpose summary (for outline display)
    - `goals` is a list of specific checkpoints (for progress tracking)
    """

    id: str  # Unique identifier for the scene
    title: str
    outline: str = ""  # What happens in this scene
    goal: str = ""  # One-sentence scene purpose (for outline display)
    content: str = ""  # Actual prose content
    word_count: int = 0
    pov_character: str = ""  # Point of view character name
    location: str = ""  # Where the scene takes place
    beats: list[str] = Field(default_factory=list)  # Key story beats/events in the scene
    goals: list[str] = Field(default_factory=list)  # Specific checkpoints to accomplish
    order: int = 0  # Position within the chapter (for drag-drop reordering)
    status: str = "pending"  # pending, drafted, edited, final

    def update_word_count(self) -> None:
        """Update word count from content."""
        if self.content:
            self.word_count = len(self.content.split())
        else:
            self.word_count = 0


class Chapter(BaseModel):
    """A chapter in the story."""

    number: int
    title: str
    outline: str
    content: str = ""
    word_count: int = 0
    status: str = "pending"  # pending, drafted, edited, reviewed, final
    revision_notes: list[str] = Field(default_factory=list)
    scenes: list[Scene] = Field(default_factory=list)  # Scenes within this chapter

    def add_scene(self, scene: Scene) -> None:
        """Add a scene to the chapter.

        Args:
            scene: Scene to add.
        """
        scene.order = len(self.scenes)
        self.scenes.append(scene)

    def remove_scene(self, scene_id: str) -> bool:
        """Remove a scene from the chapter.

        Args:
            scene_id: ID of the scene to remove.

        Returns:
            True if scene was removed, False if not found.
        """
        for i, scene in enumerate(self.scenes):
            if scene.id == scene_id:
                self.scenes.pop(i)
                # Reorder remaining scenes
                for j, remaining_scene in enumerate(self.scenes):
                    remaining_scene.order = j
                return True
        return False

    def reorder_scenes(self, scene_ids: list[str]) -> None:
        """Reorder scenes based on a list of scene IDs.

        Args:
            scene_ids: List of scene IDs in desired order.
        """
        # Create a mapping of scene_id to scene
        scene_map = {scene.id: scene for scene in self.scenes}

        # Reorder scenes and update order field
        self.scenes = []
        for i, scene_id in enumerate(scene_ids):
            if scene_id in scene_map:
                scene = scene_map[scene_id]
                scene.order = i
                self.scenes.append(scene)

    def get_scene_by_id(self, scene_id: str) -> Scene | None:
        """Get a scene by its ID.

        Args:
            scene_id: Scene ID to find.

        Returns:
            Scene if found, None otherwise.
        """
        for scene in self.scenes:
            if scene.id == scene_id:
                return scene
        return None

    def update_chapter_word_count(self) -> None:
        """Update chapter word count from scenes or direct content."""
        if self.scenes:
            # Calculate from scenes
            total = sum(scene.word_count for scene in self.scenes)
            self.word_count = total
        elif self.content:
            # Fallback to direct content word count
            self.word_count = len(self.content.split())
        else:
            self.word_count = 0


class StoryBrief(BaseModel):
    """The initial story brief from the interviewer."""

    premise: str
    genre: str
    subgenres: list[str] = Field(default_factory=list)
    tone: str
    themes: list[str] = Field(default_factory=list)
    setting_time: str
    setting_place: str
    target_length: str  # short_story, novella, novel
    language: str = "English"  # Output language for all content
    content_rating: str  # none, mild, moderate, explicit
    content_preferences: list[str] = Field(default_factory=list)  # What to include
    content_avoid: list[str] = Field(default_factory=list)  # What to avoid
    additional_notes: str = ""


class StoryState(BaseModel):
    """Complete state of a story in progress."""

    id: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Project metadata
    project_name: str = ""  # User-editable title
    project_description: str = ""  # Optional notes
    last_saved: datetime | None = None  # Track last save time

    # World database reference (SQLite file path)
    world_db_path: str = ""

    # Interview history (for displaying and continuing conversations)
    interview_history: list[dict[str, str]] = Field(default_factory=list)

    # Reviews and notes from user/AI
    reviews: list[dict[str, Any]] = Field(default_factory=list)

    # Story brief
    brief: StoryBrief | None = None

    # World building (kept for backward compatibility - primary storage is WorldDatabase)
    world_description: str = ""
    world_rules: list[str] = Field(default_factory=list)

    # Characters
    characters: list[Character] = Field(default_factory=list)

    # Plot
    plot_summary: str = ""
    plot_points: list[PlotPoint] = Field(default_factory=list)

    # Structure
    chapters: list[Chapter] = Field(default_factory=list)
    current_chapter: int = 0

    # Continuity tracking
    timeline: list[str] = Field(default_factory=list)  # Key events in order
    established_facts: list[str] = Field(default_factory=list)  # Things that are now canon

    # Status
    status: str = "interview"  # interview, outlining, writing, editing, complete

    def get_context_summary(self) -> str:
        """Generate a compressed context summary for agents."""
        summary_parts = []

        if self.brief:
            summary_parts.append(f"PREMISE: {self.brief.premise}")
            summary_parts.append(f"GENRE: {self.brief.genre} | TONE: {self.brief.tone}")
            summary_parts.append(f"SETTING: {self.brief.setting_place}, {self.brief.setting_time}")

        if self.characters:
            char_summary = "CHARACTERS: " + ", ".join(
                f"{c.name} ({c.role})" for c in self.characters
            )
            summary_parts.append(char_summary)

        if self.plot_points:
            completed = [p for p in self.plot_points if p.completed]
            pending = [p for p in self.plot_points if not p.completed]
            if completed:
                summary_parts.append(f"COMPLETED PLOT POINTS: {len(completed)}")
            if pending:
                summary_parts.append(f"UPCOMING: {pending[0].description if pending else 'None'}")

        if self.established_facts:
            recent_facts = self.established_facts[-30:]  # Last 30 facts for better context
            summary_parts.append(f"RECENT FACTS: {'; '.join(recent_facts)}")

        return "\n".join(summary_parts)

    def add_established_fact(self, fact: str):
        """Add a new established fact."""
        self.established_facts.append(fact)
        self.updated_at = datetime.now()

    def get_character_by_name(self, name: str) -> Character | None:
        """Find a character by name."""
        for char in self.characters:
            if char.name.lower() == name.lower():
                return char
        return None

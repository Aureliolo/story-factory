"""Story state management - maintains context across the generation process."""

import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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
    """A scene within a chapter."""

    number: int
    title: str
    goal: str  # What this scene aims to accomplish
    pov_character: str = ""  # Point of view character for this scene
    location: str = ""  # Where the scene takes place
    beats: list[str] = Field(default_factory=list)  # Key story beats/events in the scene
    content: str = ""  # The actual prose content of the scene


class Chapter(BaseModel):
    """A chapter in the story."""

    number: int
    title: str
    outline: str
    content: str = ""
    word_count: int = 0
    status: str = "pending"  # pending, drafted, edited, reviewed, final
    revision_notes: list[str] = Field(default_factory=list)
    scenes: list[Scene] = Field(default_factory=list)  # Optional scene-level breakdown


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


class OutlineVariation(BaseModel):
    """A variation of the story outline with different plot/character/chapter choices."""

    id: str  # Unique identifier for this variation
    created_at: datetime = Field(default_factory=datetime.now)
    name: str = ""  # User-friendly name (e.g., "Variation 1", "Dark Ending")

    # Core story structure for this variation
    world_description: str = ""
    world_rules: list[str] = Field(default_factory=list)
    characters: list[Character] = Field(default_factory=list)
    plot_summary: str = ""
    plot_points: list[PlotPoint] = Field(default_factory=list)
    chapters: list[Chapter] = Field(default_factory=list)

    # Metadata
    user_rating: int = 0  # 0-5 star rating
    user_notes: str = ""  # User feedback on this variation
    is_favorite: bool = False  # User-marked favorite
    ai_rationale: str = ""  # Why this variation was generated

    # Selection tracking
    selected_elements: dict[str, bool] = Field(default_factory=dict)  # element_id -> selected

    def get_summary(self) -> str:
        """Get a brief summary of this variation."""
        parts = []
        if self.name:
            parts.append(f"**{self.name}**")
        if self.plot_summary:
            summary = (
                self.plot_summary[:150] + "..."
                if len(self.plot_summary) > 150
                else self.plot_summary
            )
            parts.append(summary)
        parts.append(f"{len(self.characters)} characters, {len(self.chapters)} chapters")
        if self.user_rating > 0:
            parts.append(f"Rating: {'⭐' * self.user_rating}")
        return " | ".join(parts)


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

    # Outline Variations
    outline_variations: list[OutlineVariation] = Field(default_factory=list)
    selected_variation_id: str | None = None  # The variation selected as canonical
    variation_generation_count: int = 3  # How many variations to generate (3-5)

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

    def add_outline_variation(self, variation: OutlineVariation) -> None:
        """Add a new outline variation.

        Args:
            variation: The variation to add.
        """
        self.outline_variations.append(variation)
        self.updated_at = datetime.now()
        logger.debug(f"Added outline variation: {variation.name} (id={variation.id})")

    def get_variation_by_id(self, variation_id: str) -> OutlineVariation | None:
        """Find a variation by ID.

        Args:
            variation_id: The variation ID to find.

        Returns:
            The variation if found, None otherwise.
        """
        for variation in self.outline_variations:
            if variation.id == variation_id:
                return variation
        return None

    def select_variation_as_canonical(self, variation_id: str) -> bool:
        """Select a variation as the canonical outline.

        This copies the variation's structure to the main story state.

        Args:
            variation_id: The ID of the variation to make canonical.

        Returns:
            True if successful, False if variation not found.
        """
        variation = self.get_variation_by_id(variation_id)
        if not variation:
            logger.warning(f"Variation {variation_id} not found")
            return False

        # Copy variation data to main state
        self.world_description = variation.world_description
        self.world_rules = variation.world_rules.copy()
        self.characters = [char.model_copy(deep=True) for char in variation.characters]
        self.plot_summary = variation.plot_summary
        self.plot_points = [pp.model_copy(deep=True) for pp in variation.plot_points]
        self.chapters = [ch.model_copy(deep=True) for ch in variation.chapters]
        self.selected_variation_id = variation_id
        self.updated_at = datetime.now()

        logger.info(f"Selected variation {variation.name} as canonical")
        return True

    def create_merged_variation(
        self,
        name: str,
        source_variations: dict[str, list[str]],
    ) -> OutlineVariation:
        """Create a new variation by merging elements from multiple variations.

        Args:
            name: Name for the merged variation.
            source_variations: Dict mapping variation_id to list of element types
                              e.g., {"var1": ["characters", "world"], "var2": ["plot", "chapters"]}

        Returns:
            A new OutlineVariation with merged elements.
        """
        import uuid

        merged = OutlineVariation(
            id=str(uuid.uuid4()),
            name=name,
            ai_rationale=f"Merged from {len(source_variations)} variations",
        )

        # Merge elements from each source
        for var_id, elements in source_variations.items():
            source = self.get_variation_by_id(var_id)
            if not source:
                logger.warning(f"Source variation {var_id} not found, skipping")
                continue

            for element_type in elements:
                if element_type == "world":
                    merged.world_description = source.world_description
                    merged.world_rules = source.world_rules.copy()
                elif element_type == "characters":
                    merged.characters = [char.model_copy(deep=True) for char in source.characters]
                elif element_type == "plot":
                    merged.plot_summary = source.plot_summary
                    merged.plot_points = [pp.model_copy(deep=True) for pp in source.plot_points]
                elif element_type == "chapters":
                    merged.chapters = [ch.model_copy(deep=True) for ch in source.chapters]

        self.add_outline_variation(merged)
        logger.info(f"Created merged variation: {name}")
        return merged

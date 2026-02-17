"""Story state management - maintains context across the generation process."""

import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator, model_validator

from src.memory.content_guidelines import ContentProfile
from src.memory.templates import PersonalityTrait, TargetLength, normalize_traits

if TYPE_CHECKING:
    from src.memory._chapter_versions import ChapterVersionManager

logger = logging.getLogger(__name__)


class Character(BaseModel):
    """A character in the story."""

    name: str
    role: str  # protagonist, antagonist, supporting, etc.
    description: str
    personality_traits: list[PersonalityTrait] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    relationships: dict[str, str] = Field(default_factory=dict)  # character_name -> relationship
    arc_notes: str = ""  # How the character should develop
    arc_type: str | None = None  # Reference to arc template ID (e.g., "hero_journey")
    arc_progress: dict[int, str] = Field(default_factory=dict)  # chapter_number -> arc state

    @field_validator("personality_traits", mode="before")
    @classmethod
    def normalize_personality_traits(cls, v: Any) -> Any:
        """Normalize plain-string personality traits from old project JSONs."""
        return normalize_traits(v)

    @field_validator("arc_progress", mode="before")
    @classmethod
    def clean_arc_progress(cls, v: Any) -> dict[int, str]:
        """Clean arc_progress if LLM returns invalid format.

        LLMs sometimes return string keys like {"Embracing Power": "..."} instead of
        integer chapter numbers {1: "..."}. Since arc_progress is filled during writing,
        we just clear invalid data rather than fail character creation.

        This validator also serves as a safety net when loading old project JSON files
        that may contain invalid arc_progress data.

        Using field_validator instead of model_validator ensures this runs before
        Pydantic validates dict[int, str] keys, which is critical for instructor
        library compatibility.
        """
        if not isinstance(v, dict):
            return {}

        # Check if keys are valid integers
        cleaned: dict[int, str] = {}
        for key, value in v.items():
            try:
                int_key = int(key)
                cleaned[int_key] = str(value)
            except ValueError, TypeError:
                # Invalid key (like "Embracing Corruption") - skip this entry
                logger.warning("Skipping invalid arc_progress key: %r (expected integer)", key)
        return cleaned

    @property
    def trait_names(self) -> list[str]:
        """Get flat list of trait names for prompt building."""
        return [t.trait for t in self.personality_traits]

    def traits_by_category(self, category: str) -> list[str]:
        """Get trait names filtered by category.

        Args:
            category: One of "core", "flaw", "quirk".

        Returns:
            List of trait names matching the category.
        """
        return [t.trait for t in self.personality_traits if t.category == category]

    def update_arc(self, chapter_number: int, state: str) -> None:
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


class Faction(BaseModel):
    """A faction or organization in the story world."""

    name: str = Field(description="Name of the faction")
    type: str = Field(default="faction", description="Entity type (always 'faction')")
    description: str = Field(description="Description of the faction, its history, and purpose")
    leader: str = Field(default="", description="Name or title of leader (if any)")
    goals: list[str] = Field(default_factory=list, description="Primary and secondary goals")
    values: list[str] = Field(default_factory=list, description="Core values of the faction")
    base_location: str = Field(default="", description="Headquarters or territory location")


class Item(BaseModel):
    """A significant item or object in the story world."""

    name: str = Field(description="Name of the item")
    type: str = Field(default="item", description="Entity type (always 'item')")
    description: str = Field(description="Physical description and history of the item")
    significance: str = Field(default="", description="Why this item matters to the story")
    properties: list[str] = Field(
        default_factory=list, description="Special properties or abilities"
    )


class Concept(BaseModel):
    """A thematic concept or idea in the story world."""

    name: str = Field(description="Name of the concept")
    type: str = Field(default="concept", description="Entity type (always 'concept')")
    description: str = Field(description="What this concept means in the story context")
    manifestations: str = Field(
        default="", description="How this concept appears through events, characters, or symbols"
    )


class Location(BaseModel):
    """A location in the story world."""

    name: str = Field(description="Name of the location")
    type: str = Field(default="location", description="Entity type (always 'location')")
    description: str = Field(description="Detailed description with sensory details")
    significance: str = Field(default="", description="Why this place matters to the story")


class Relationship(BaseModel):
    """A relationship between two entities in the story world."""

    source: str = Field(description="Name of the source entity")
    target: str = Field(description="Name of the target entity")
    relation_type: str = Field(
        description="Type of relationship (e.g., knows, loves, hates, allies_with, "
        "enemies_with, located_in, owns, member_of)"
    )
    description: str = Field(
        description="Description of the relationship with history and dynamics"
    )


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


class ChapterVersion(BaseModel):
    """A saved version of a chapter with metadata."""

    id: str  # Unique identifier for this version
    created_at: datetime = Field(default_factory=datetime.now)
    content: str  # The actual prose content
    word_count: int = 0
    feedback: str = ""  # User feedback that prompted this version (if regenerated)
    version_number: int = 1  # Sequential version number
    is_current: bool = False  # True if this is the active version


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

    # Version history for regeneration and rollback
    versions: list[ChapterVersion] = Field(default_factory=list)
    current_version_id: str | None = None

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
        """
        Update the chapter's word_count based on available data.

        If the chapter has scenes, sets word_count to the sum of their word_count values. If there are no scenes but content is present, sets word_count to the number of words in the content. If neither scenes nor content are present, sets word_count to 0.
        """
        if self.scenes:
            # Calculate from scenes
            total = sum(scene.word_count for scene in self.scenes)
            self.word_count = total
        elif self.content:
            # Fallback to direct content word count
            self.word_count = len(self.content.split())
        else:
            self.word_count = 0

    @property
    def version_manager(self) -> ChapterVersionManager:
        """
        Lazily create and return a ChapterVersionManager bound to this chapter.

        Returns:
            ChapterVersionManager: The cached manager instance used for chapter version operations.
        """
        # Cache the manager to avoid creating new instances on every access
        cache_key = "_version_manager_cache"
        if cache_key not in self.__dict__:
            from src.memory._chapter_versions import ChapterVersionManager

            self.__dict__[cache_key] = ChapterVersionManager(self)
        manager: ChapterVersionManager = self.__dict__[cache_key]
        return manager

    def save_current_as_version(self, feedback: str = "") -> str:
        """
        Save the chapter's current content as a new version.

        Parameters:
            feedback (str): Optional feedback associated with this version.

        Returns:
            str: The ID of the newly created version.
        """
        return self.version_manager.save_version(feedback)

    def rollback_to_version(self, version_id: str) -> bool:
        """
        Restore the chapter to the specified saved version.

        Parameters:
            version_id (str): ID of the saved version to restore.

        Returns:
            bool: `True` if the rollback succeeded, `False` if the specified version was not found.
        """
        return self.version_manager.rollback(version_id)

    def get_version_by_id(self, version_id: str) -> ChapterVersion | None:
        """Get a version by its ID.

        Args:
            version_id: The version ID to find.

        Returns:
            The version if found, None otherwise.
        """
        return self.version_manager.get_version(version_id)

    def get_current_version(self) -> ChapterVersion | None:
        """
        Retrieve the chapter's currently active version.

        Returns:
            The active ChapterVersion if present, otherwise None.
        """
        return self.version_manager.get_current()

    def compare_versions(self, version_id_a: str, version_id_b: str) -> dict[str, Any]:
        """
        Produce a comparison report between two saved chapter versions.

        Parameters:
            version_id_a (str): ID of the first version to compare.
            version_id_b (str): ID of the second version to compare.

        Returns:
            comparison (dict[str, Any]): Mapping containing comparison details such as 'word_count_a', 'word_count_b', 'word_count_diff', and any content or metadata diffs.
        """
        return self.version_manager.compare(version_id_a, version_id_b)


class StoryBrief(BaseModel):
    """The initial story brief from the interviewer."""

    premise: str
    genre: str
    subgenres: list[str] = Field(default_factory=list)
    tone: str
    themes: list[str] = Field(default_factory=list)
    setting_time: str
    setting_place: str
    target_length: TargetLength  # short_story, novella, novel
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

    # World template reference
    world_template_id: str | None = None  # ID of world template used (e.g., "high_fantasy")

    # Content guidelines profile
    content_profile: ContentProfile | None = None  # Content appropriateness settings

    # Project-specific generation settings (None = use global settings defaults)
    target_chapters: int | None = None  # Override chapter count for this project
    target_characters_min: int | None = None  # Override min characters for this project
    target_characters_max: int | None = None  # Override max characters for this project
    target_locations_min: int | None = None  # Override min locations for this project
    target_locations_max: int | None = None  # Override max locations for this project
    target_factions_min: int | None = None  # Override min factions for this project
    target_factions_max: int | None = None  # Override max factions for this project
    target_items_min: int | None = None  # Override min items for this project
    target_items_max: int | None = None  # Override max items for this project
    target_concepts_min: int | None = None  # Override min concepts for this project
    target_concepts_max: int | None = None  # Override max concepts for this project

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

    def add_established_fact(self, fact: str) -> None:
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
        logger.debug(
            f"Variation {variation_id} not found in {len(self.outline_variations)} variations"
        )
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

        Note:
            If the same element type appears in multiple sources, later sources
            will overwrite earlier ones. A warning is logged when this occurs.
        """
        merged = OutlineVariation(
            id=str(uuid.uuid4()),
            name=name,
            ai_rationale=f"Merged from {len(source_variations)} variations",
        )

        # Track seen element types to warn on duplicates
        seen_elements: set[str] = set()

        # Merge elements from each source
        for var_id, elements in source_variations.items():
            source = self.get_variation_by_id(var_id)
            if not source:
                logger.warning(f"Source variation {var_id} not found, skipping")
                continue

            for element_type in elements:
                if element_type in seen_elements:
                    logger.warning(
                        f"Element type '{element_type}' already merged from another variation, "
                        f"overwriting with data from {var_id}"
                    )
                seen_elements.add(element_type)

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


# ============================================================================
# List Wrapper Models for Instructor Integration
# ============================================================================
# These wrapper models are used with the Instructor library to enforce
# JSON schema validation when generating lists of Pydantic models.
# Instructor requires a single Pydantic model, so we wrap lists.


class CharacterList(BaseModel):
    """Wrapper model for a list of characters.

    Used with generate_structured() to get validated character lists from LLM.
    Handles LLMs returning a single object instead of a wrapped list.
    """

    characters: list[Character]

    @model_validator(mode="before")
    @classmethod
    def wrap_single_object(cls, data: Any) -> Any:
        """Wrap a single Character object in a list if needed."""
        if isinstance(data, dict) and "characters" not in data:
            # LLM returned a single object, wrap it
            if "name" in data and "role" in data:
                logger.debug("Wrapping single Character object in CharacterList")
                return {"characters": [data]}
        return data


class CharacterCreation(BaseModel):
    """A character as created by the LLM, without runtime-only fields.

    Excludes arc_progress and arc_type which are populated at runtime during
    the write loop, not during character creation. This prevents grammar-constrained
    output from wasting tokens on empty/invalid arc_progress fields.
    """

    name: str
    role: str
    description: str
    personality_traits: list[PersonalityTrait] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    relationships: dict[str, str] = Field(default_factory=dict)
    arc_notes: str = ""

    @field_validator("personality_traits", mode="before")
    @classmethod
    def normalize_personality_traits(cls, v: Any) -> Any:
        """Normalize plain-string personality traits from old project JSONs."""
        return normalize_traits(v)

    def to_character(self) -> Character:
        """Convert to a full Character with default runtime fields."""
        logger.debug("Converting CharacterCreation '%s' to Character", self.name)
        return Character(**self.model_dump())


class CharacterCreationList(BaseModel):
    """Wrapper for a list of CharacterCreation models.

    Used with generate_structured() during character creation to exclude
    runtime-only fields (arc_progress, arc_type) from the grammar constraint.
    Handles LLMs returning a single object instead of a wrapped list.
    """

    characters: list[CharacterCreation]

    @model_validator(mode="before")
    @classmethod
    def wrap_single_object(cls, data: Any) -> Any:
        """Wrap a single CharacterCreation object in a list if needed."""
        if isinstance(data, dict) and "characters" not in data:
            if "name" in data and "role" in data:
                logger.debug("Wrapping single CharacterCreation object in CharacterCreationList")
                return {"characters": [data]}
        return data

    def to_characters(self) -> list[Character]:
        """Convert all CharacterCreation items to full Character objects."""
        logger.debug("Converting %d CharacterCreation items to Characters", len(self.characters))
        return [c.to_character() for c in self.characters]


class PlotPointList(BaseModel):
    """Wrapper model for a list of plot points.

    Used with generate_structured() to get validated plot point lists from LLM.
    Handles LLMs returning a single object instead of a wrapped list.
    """

    plot_points: list[PlotPoint]

    @model_validator(mode="before")
    @classmethod
    def wrap_single_object(cls, data: Any) -> Any:
        """Wrap a single PlotPoint object in a list if needed."""
        if isinstance(data, dict) and "plot_points" not in data:
            # LLM returned a single object, wrap it
            if "description" in data:
                logger.debug("Wrapping single PlotPoint object in PlotPointList")
                return {"plot_points": [data]}
        return data


class PlotOutline(BaseModel):
    """Complete plot outline with summary and plot points.

    Used with generate_structured() for the architect's create_plot_outline method.
    """

    plot_summary: str
    plot_points: list[PlotPoint]


class ChapterList(BaseModel):
    """Wrapper model for a list of chapters.

    Used with generate_structured() to get validated chapter lists from LLM.
    Handles LLMs returning a single object instead of a wrapped list.
    """

    chapters: list[Chapter]

    @model_validator(mode="before")
    @classmethod
    def wrap_single_object(cls, data: Any) -> Any:
        """Wrap a single Chapter object in a list if needed."""
        if isinstance(data, dict) and "chapters" not in data:
            # LLM returned a single object, wrap it
            if "number" in data and "title" in data:
                logger.debug("Wrapping single Chapter object in ChapterList")
                return {"chapters": [data]}
        return data

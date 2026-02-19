"""Entity-specific quality score models.

These models define the quality scoring dimensions for world-building entities:
characters, locations, factions, items, concepts, events, and calendars.
"""

from pydantic import Field

from src.memory.world_quality._models import BaseQualityScores


class CharacterQualityScores(BaseQualityScores):
    """Quality scores for a character (0-10 scale).

    All score fields are required - no defaults. This ensures that the LLM
    must provide explicit scores or the parsing will fail, preventing silent
    fallbacks to meaningless default values.
    """

    depth: float = Field(ge=0.0, le=10.0, description="Psychological complexity")
    goals: float = Field(
        alias="goal_clarity", ge=0.0, le=10.0, description="Clarity and story relevance"
    )
    flaws: float = Field(ge=0.0, le=10.0, description="Meaningful vulnerabilities")
    uniqueness: float = Field(ge=0.0, le=10.0, description="Distinctiveness")
    arc_potential: float = Field(ge=0.0, le=10.0, description="Room for transformation")
    temporal_plausibility: float = Field(
        ge=0.0, le=10.0, description="Timeline consistency and era-appropriate placement"
    )

    @property
    def average(self) -> float:
        """
        Mean of the character's six quality dimensions.

        Returns:
            average (float): Mean of `depth`, `goal_clarity` (alias `goals`), `flaws`,
                `uniqueness`, `arc_potential`, and `temporal_plausibility`.
        """
        return (
            self.depth
            + self.goals
            + self.flaws
            + self.uniqueness
            + self.arc_potential
            + self.temporal_plausibility
        ) / 6.0

    def to_dict(self) -> dict[str, float | str]:
        """
        Serialize the character quality scores into a dictionary formatted for storage in entity attributes.

        Returns:
            dict[str, float | str]: Mapping with keys:
                - "depth": float score for depth
                - "goal_clarity": float score for goals
                - "flaws": float score for flaws
                - "uniqueness": float score for uniqueness
                - "arc_potential": float score for arc potential
                - "temporal_plausibility": float score for temporal plausibility
                - "average": float mean of the scored dimensions
                - "feedback": string feedback associated with the scores
        """
        return {
            "depth": self.depth,
            "goal_clarity": self.goals,
            "flaws": self.flaws,
            "uniqueness": self.uniqueness,
            "arc_potential": self.arc_potential,
            "temporal_plausibility": self.temporal_plausibility,
            "average": self.average,
            "feedback": self.feedback,
        }

    def weak_dimensions(self, threshold: float = 7.0) -> list[str]:
        """
        Identify character quality dimensions whose scores are below the given threshold.

        Parameters:
                threshold (float): Score threshold; dimensions with values less than this are considered weak. Defaults to 7.0.

        Returns:
                weak_dimensions (list[str]): List of dimension names among "depth", "goal_clarity",
                    "flaws", "uniqueness", "arc_potential", and "temporal_plausibility" whose scores
                    are below the threshold.
        """
        weak = []
        if self.depth < threshold:
            weak.append("depth")
        if self.goals < threshold:
            weak.append("goal_clarity")
        if self.flaws < threshold:
            weak.append("flaws")
        if self.uniqueness < threshold:
            weak.append("uniqueness")
        if self.arc_potential < threshold:
            weak.append("arc_potential")
        if self.temporal_plausibility < threshold:
            weak.append("temporal_plausibility")
        return weak


class LocationQualityScores(BaseQualityScores):
    """Quality scores for a location (0-10 scale).

    All score fields are required - no defaults.
    """

    atmosphere: float = Field(ge=0.0, le=10.0, description="Sensory richness, mood")
    significance: float = Field(
        alias="narrative_significance", ge=0.0, le=10.0, description="Plot/symbolic meaning"
    )
    story_relevance: float = Field(ge=0.0, le=10.0, description="Theme/character links")
    distinctiveness: float = Field(ge=0.0, le=10.0, description="Memorable qualities")
    temporal_plausibility: float = Field(
        ge=0.0, le=10.0, description="Timeline consistency and era-appropriate placement"
    )

    @property
    def average(self) -> float:
        """
        Compute the mean of the location's quality dimensions.

        Returns:
            average (float): Mean of `atmosphere`, `significance`, `story_relevance`,
                `distinctiveness`, and `temporal_plausibility`.
        """
        return (
            self.atmosphere
            + self.significance
            + self.story_relevance
            + self.distinctiveness
            + self.temporal_plausibility
        ) / 5.0

    def to_dict(self) -> dict[str, float | str]:
        """
        Produce a dictionary of location quality dimensions and associated metadata keyed for storage.

        Returns:
            dict[str, float | str]: Dictionary containing 'atmosphere', 'narrative_significance',
                'story_relevance', 'distinctiveness', 'temporal_plausibility', 'average', and 'feedback'.
        """
        return {
            "atmosphere": self.atmosphere,
            "narrative_significance": self.significance,
            "story_relevance": self.story_relevance,
            "distinctiveness": self.distinctiveness,
            "temporal_plausibility": self.temporal_plausibility,
            "average": self.average,
            "feedback": self.feedback,
        }

    def weak_dimensions(self, threshold: float = 7.0) -> list[str]:
        """
        Identify location quality dimensions whose scores are below the given threshold.

        Parameters:
            threshold (float): The cutoff value; a dimension is considered weak if its score
                is less than this threshold.

        Returns:
            list[str]: Names of dimensions with scores below `threshold`.
        """
        weak = []
        if self.atmosphere < threshold:
            weak.append("atmosphere")
        if self.significance < threshold:
            weak.append("narrative_significance")
        if self.story_relevance < threshold:
            weak.append("story_relevance")
        if self.distinctiveness < threshold:
            weak.append("distinctiveness")
        if self.temporal_plausibility < threshold:
            weak.append("temporal_plausibility")
        return weak


class FactionQualityScores(BaseQualityScores):
    """Quality scores for a faction (0-10 scale).

    All score fields are required - no defaults.
    """

    coherence: float = Field(ge=0.0, le=10.0, description="Internal consistency")
    influence: float = Field(ge=0.0, le=10.0, description="World impact")
    conflict_potential: float = Field(ge=0.0, le=10.0, description="Story conflict potential")
    distinctiveness: float = Field(ge=0.0, le=10.0, description="Memorable qualities")
    temporal_plausibility: float = Field(
        ge=0.0, le=10.0, description="Timeline consistency and era-appropriate placement"
    )

    @property
    def average(self) -> float:
        """
        Compute the mean of the faction's five quality dimensions.

        Returns:
            average (float): Mean of coherence, influence, conflict_potential,
                distinctiveness, and temporal_plausibility.
        """
        return (
            self.coherence
            + self.influence
            + self.conflict_potential
            + self.distinctiveness
            + self.temporal_plausibility
        ) / 5.0

    def to_dict(self) -> dict[str, float | str]:
        """
        Return a mapping of this faction's quality dimensions, computed average, and feedback suitable for storage.

        Returns:
            dict[str, float | str]: Dictionary containing 'coherence', 'influence',
                'conflict_potential', 'distinctiveness', 'temporal_plausibility',
                'average', and 'feedback'.
        """
        return {
            "coherence": self.coherence,
            "influence": self.influence,
            "conflict_potential": self.conflict_potential,
            "distinctiveness": self.distinctiveness,
            "temporal_plausibility": self.temporal_plausibility,
            "average": self.average,
            "feedback": self.feedback,
        }

    def weak_dimensions(self, threshold: float = 7.0) -> list[str]:
        """
        Identify which faction quality dimensions score below a given threshold.

        Parameters:
            threshold (float): Numeric cutoff (0-10) used to classify a dimension as weak.
                Defaults to 7.0.

        Returns:
            list[str]: Names of dimensions whose values are less than `threshold`.
        """
        weak = []
        if self.coherence < threshold:
            weak.append("coherence")
        if self.influence < threshold:
            weak.append("influence")
        if self.conflict_potential < threshold:
            weak.append("conflict_potential")
        if self.distinctiveness < threshold:
            weak.append("distinctiveness")
        if self.temporal_plausibility < threshold:
            weak.append("temporal_plausibility")
        return weak


class ItemQualityScores(BaseQualityScores):
    """Quality scores for an item (0-10 scale).

    All score fields are required - no defaults.
    """

    significance: float = Field(
        alias="story_significance", ge=0.0, le=10.0, description="Story importance"
    )
    uniqueness: float = Field(ge=0.0, le=10.0, description="Distinctive qualities")
    narrative_potential: float = Field(ge=0.0, le=10.0, description="Plot opportunities")
    integration: float = Field(ge=0.0, le=10.0, description="Fits world")
    temporal_plausibility: float = Field(
        ge=0.0, le=10.0, description="Timeline consistency and era-appropriate placement"
    )

    @property
    def average(self) -> float:
        """
        Compute the mean quality score across the item's five dimensions.

        Returns:
            float: Mean of `significance`, `uniqueness`, `narrative_potential`,
                `integration`, and `temporal_plausibility`.
        """
        return (
            self.significance
            + self.uniqueness
            + self.narrative_potential
            + self.integration
            + self.temporal_plausibility
        ) / 5.0

    def to_dict(self) -> dict[str, float | str]:
        """
        Serialize the item's quality scores into a dictionary formatted for storing on an entity.

        Returns:
            dict[str, float | str]: Mapping containing 'story_significance', 'uniqueness',
                'narrative_potential', 'integration', 'temporal_plausibility',
                'average', and 'feedback'.
        """
        return {
            "story_significance": self.significance,
            "uniqueness": self.uniqueness,
            "narrative_potential": self.narrative_potential,
            "integration": self.integration,
            "temporal_plausibility": self.temporal_plausibility,
            "average": self.average,
            "feedback": self.feedback,
        }

    def weak_dimensions(self, threshold: float = 7.0) -> list[str]:
        """
        List which item quality dimensions fall below a threshold.

        Parameters:
                threshold (float): Cutoff value; any dimension with a score less than this
                    is considered weak. Defaults to 7.0.

        Returns:
                weak (list[str]): Names of dimensions whose scores are below `threshold`.
        """
        weak = []
        if self.significance < threshold:
            weak.append("story_significance")
        if self.uniqueness < threshold:
            weak.append("uniqueness")
        if self.narrative_potential < threshold:
            weak.append("narrative_potential")
        if self.integration < threshold:
            weak.append("integration")
        if self.temporal_plausibility < threshold:
            weak.append("temporal_plausibility")
        return weak


class ConceptQualityScores(BaseQualityScores):
    """Quality scores for a concept (0-10 scale).

    All score fields are required - no defaults.
    """

    relevance: float = Field(ge=0.0, le=10.0, description="Theme alignment")
    depth: float = Field(ge=0.0, le=10.0, description="Philosophical richness")
    manifestation: float = Field(ge=0.0, le=10.0, description="How it appears in story")
    resonance: float = Field(ge=0.0, le=10.0, description="Emotional impact")
    temporal_plausibility: float = Field(
        ge=0.0, le=10.0, description="Timeline consistency and era-appropriate placement"
    )

    @property
    def average(self) -> float:
        """
        Return the mean of the concept's five quality dimensions.

        Returns:
            The average score (0 to 10) across relevance, depth, manifestation,
            resonance, and temporal_plausibility.
        """
        return (
            self.relevance
            + self.depth
            + self.manifestation
            + self.resonance
            + self.temporal_plausibility
        ) / 5.0

    def to_dict(self) -> dict[str, float | str]:
        """
        Return a mapping of concept quality dimensions and metadata suitable for storing on an entity.

        Returns:
            dict[str, float | str]: Dictionary containing 'relevance', 'depth',
                'manifestation', 'resonance', 'temporal_plausibility',
                'average', and 'feedback'.
        """
        return {
            "relevance": self.relevance,
            "depth": self.depth,
            "manifestation": self.manifestation,
            "resonance": self.resonance,
            "temporal_plausibility": self.temporal_plausibility,
            "average": self.average,
            "feedback": self.feedback,
        }

    def weak_dimensions(self, threshold: float = 7.0) -> list[str]:
        """
        Identify concept quality dimensions whose scores are below the given threshold.

        Parameters:
            threshold (float): Cutoff value; any dimension with a score less than this
                is considered weak. Defaults to 7.0.

        Returns:
            weak_dimensions (list[str]): List of dimension names whose values are below `threshold`.
        """
        weak = []
        if self.relevance < threshold:
            weak.append("relevance")
        if self.depth < threshold:
            weak.append("depth")
        if self.manifestation < threshold:
            weak.append("manifestation")
        if self.resonance < threshold:
            weak.append("resonance")
        if self.temporal_plausibility < threshold:
            weak.append("temporal_plausibility")
        return weak


class EventQualityScores(BaseQualityScores):
    """Quality scores for a world event (0-10 scale).

    All score fields are required - no defaults.
    """

    significance: float = Field(ge=0.0, le=10.0, description="How world-shaping is this event")
    temporal_plausibility: float = Field(ge=0.0, le=10.0, description="Calendar timeline fit")
    causal_coherence: float = Field(ge=0.0, le=10.0, description="Logical causes and consequences")
    narrative_potential: float = Field(ge=0.0, le=10.0, description="Story opportunity creation")
    entity_integration: float = Field(ge=0.0, le=10.0, description="Participant roles make sense")

    @property
    def average(self) -> float:
        """Compute the mean of the event's five quality dimensions.

        Returns:
            average (float): Mean of significance, temporal_plausibility,
                causal_coherence, narrative_potential, and entity_integration.
        """
        return (
            self.significance
            + self.temporal_plausibility
            + self.causal_coherence
            + self.narrative_potential
            + self.entity_integration
        ) / 5.0

    def to_dict(self) -> dict[str, float | str]:
        """Serialize the event quality scores into a dictionary for storage.

        Returns:
            dict[str, float | str]: Dictionary with keys 'significance',
                'temporal_plausibility', 'causal_coherence', 'narrative_potential',
                'entity_integration', 'average', and 'feedback'.
        """
        return {
            "significance": self.significance,
            "temporal_plausibility": self.temporal_plausibility,
            "causal_coherence": self.causal_coherence,
            "narrative_potential": self.narrative_potential,
            "entity_integration": self.entity_integration,
            "average": self.average,
            "feedback": self.feedback,
        }

    def weak_dimensions(self, threshold: float = 7.0) -> list[str]:
        """Identify event quality dimensions whose scores are below the given threshold.

        Parameters:
            threshold (float): Cutoff value; dimensions below this are considered weak.
                Defaults to 7.0.

        Returns:
            list[str]: Names of dimensions with scores below `threshold`.
        """
        weak = []
        if self.significance < threshold:
            weak.append("significance")
        if self.temporal_plausibility < threshold:
            weak.append("temporal_plausibility")
        if self.causal_coherence < threshold:
            weak.append("causal_coherence")
        if self.narrative_potential < threshold:
            weak.append("narrative_potential")
        if self.entity_integration < threshold:
            weak.append("entity_integration")
        return weak


class CalendarQualityScores(BaseQualityScores):
    """Quality scores for a calendar system (0-10 scale).

    All score fields are required - no defaults.
    """

    internal_consistency: float = Field(
        ge=0.0, le=10.0, description="Internal logic and coherence of the calendar system"
    )
    thematic_fit: float = Field(
        ge=0.0, le=10.0, description="How well the calendar fits the story's genre and setting"
    )
    completeness: float = Field(
        ge=0.0, le=10.0, description="Thoroughness of months, eras, and day names"
    )
    uniqueness: float = Field(
        ge=0.0, le=10.0, description="Distinctiveness from real-world or generic calendars"
    )

    @property
    def average(self) -> float:
        """Compute the mean of the calendar's four quality dimensions.

        Returns:
            average (float): Mean of `internal_consistency`, `thematic_fit`,
                `completeness`, and `uniqueness`.
        """
        return (
            self.internal_consistency + self.thematic_fit + self.completeness + self.uniqueness
        ) / 4.0

    def to_dict(self) -> dict[str, float | str]:
        """Serialize the calendar quality scores into a dictionary for storage.

        Returns:
            dict[str, float | str]: Dictionary with keys 'internal_consistency',
                'thematic_fit', 'completeness', 'uniqueness', 'average', and 'feedback'.
        """
        return {
            "internal_consistency": self.internal_consistency,
            "thematic_fit": self.thematic_fit,
            "completeness": self.completeness,
            "uniqueness": self.uniqueness,
            "average": self.average,
            "feedback": self.feedback,
        }

    def weak_dimensions(self, threshold: float = 7.0) -> list[str]:
        """Identify calendar quality dimensions whose scores are below the given threshold.

        Parameters:
            threshold (float): Cutoff value; dimensions below this are considered weak.
                Defaults to 7.0.

        Returns:
            list[str]: Names of dimensions with scores below `threshold`.
        """
        weak = []
        if self.internal_consistency < threshold:
            weak.append("internal_consistency")
        if self.thematic_fit < threshold:
            weak.append("thematic_fit")
        if self.completeness < threshold:
            weak.append("completeness")
        if self.uniqueness < threshold:
            weak.append("uniqueness")
        return weak

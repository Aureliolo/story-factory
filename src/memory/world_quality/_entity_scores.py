"""Entity-specific quality score models.

These models define the quality scoring dimensions for world-building entities:
characters, locations, factions, items, and concepts.
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

    @property
    def average(self) -> float:
        """
        Mean of the character's five quality dimensions.
        
        Returns:
            average (float): Mean of `depth`, `goal_clarity` (alias `goals`), `flaws`, `uniqueness`, and `arc_potential`.
        """
        return (self.depth + self.goals + self.flaws + self.uniqueness + self.arc_potential) / 5.0

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
                - "average": float mean of the scored dimensions
                - "feedback": string feedback associated with the scores
        """
        return {
            "depth": self.depth,
            "goal_clarity": self.goals,
            "flaws": self.flaws,
            "uniqueness": self.uniqueness,
            "arc_potential": self.arc_potential,
            "average": self.average,
            "feedback": self.feedback,
        }

    def weak_dimensions(self, threshold: float = 7.0) -> list[str]:
        """
        Identify character quality dimensions whose scores are below the given threshold.
        
        Parameters:
        	threshold (float): Score threshold; dimensions with values less than this are considered weak. Defaults to 7.0.
        
        Returns:
        	weak_dimensions (list[str]): List of dimension names among "depth", "goal_clarity", "flaws", "uniqueness", and "arc_potential" whose scores are below the threshold.
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

    @property
    def average(self) -> float:
        """
        Compute the mean of the location's quality dimensions.
        
        Returns:
            average (float): Mean of `atmosphere`, `significance`, `story_relevance`, and `distinctiveness`.
        """
        return (
            self.atmosphere + self.significance + self.story_relevance + self.distinctiveness
        ) / 4.0

    def to_dict(self) -> dict[str, float | str]:
        """
        Produce a dictionary of location quality dimensions and associated metadata keyed for storage.
        
        Returns:
            dict[str, float | str]: Dictionary containing 'atmosphere', 'narrative_significance' (alias for significance), 'story_relevance', 'distinctiveness', 'average', and 'feedback'.
        """
        return {
            "atmosphere": self.atmosphere,
            "narrative_significance": self.significance,
            "story_relevance": self.story_relevance,
            "distinctiveness": self.distinctiveness,
            "average": self.average,
            "feedback": self.feedback,
        }

    def weak_dimensions(self, threshold: float = 7.0) -> list[str]:
        """
        Identify location quality dimensions whose scores are below the given threshold.
        
        Parameters:
            threshold (float): The cutoff value; a dimension is considered weak if its score is less than this threshold.
        
        Returns:
            list[str]: Names of dimensions with scores below `threshold`. Possible names: "atmosphere", "narrative_significance", "story_relevance", "distinctiveness".
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
        return weak


class FactionQualityScores(BaseQualityScores):
    """Quality scores for a faction (0-10 scale).

    All score fields are required - no defaults.
    """

    coherence: float = Field(ge=0.0, le=10.0, description="Internal consistency")
    influence: float = Field(ge=0.0, le=10.0, description="World impact")
    conflict_potential: float = Field(ge=0.0, le=10.0, description="Story conflict potential")
    distinctiveness: float = Field(ge=0.0, le=10.0, description="Memorable qualities")

    @property
    def average(self) -> float:
        """
        Compute the mean of the faction's four quality dimensions.
        
        Returns:
            average (float): Mean of coherence, influence, conflict_potential, and distinctiveness.
        """
        return (
            self.coherence + self.influence + self.conflict_potential + self.distinctiveness
        ) / 4.0

    def to_dict(self) -> dict[str, float | str]:
        """
        Return a mapping of this faction's quality dimensions, computed average, and feedback suitable for storage.
        
        Returns:
            dict[str, float | str]: Dictionary with keys:
                - "coherence": coherence score (0–10)
                - "influence": influence score (0–10)
                - "conflict_potential": conflict potential score (0–10)
                - "distinctiveness": distinctiveness score (0–10)
                - "average": mean of the four dimension scores
                - "feedback": human-readable feedback string
        """
        return {
            "coherence": self.coherence,
            "influence": self.influence,
            "conflict_potential": self.conflict_potential,
            "distinctiveness": self.distinctiveness,
            "average": self.average,
            "feedback": self.feedback,
        }

    def weak_dimensions(self, threshold: float = 7.0) -> list[str]:
        """
        Identify which faction quality dimensions score below a given threshold.
        
        Parameters:
            threshold (float): Numeric cutoff (0-10) used to classify a dimension as weak. Defaults to 7.0.
        
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

    @property
    def average(self) -> float:
        """
        Compute the mean quality score across the item's four dimensions.
        
        Returns:
            float: Mean of `significance`, `uniqueness`, `narrative_potential`, and `integration`.
        """
        return (
            self.significance + self.uniqueness + self.narrative_potential + self.integration
        ) / 4.0

    def to_dict(self) -> dict[str, float | str]:
        """
        Serialize the item's quality scores into a dictionary formatted for storing on an entity.
        
        Returns:
            dict[str, float | str]: Mapping with keys "story_significance", "uniqueness", "narrative_potential",
            "integration", "average", and "feedback" containing the corresponding score values.
        """
        return {
            "story_significance": self.significance,
            "uniqueness": self.uniqueness,
            "narrative_potential": self.narrative_potential,
            "integration": self.integration,
            "average": self.average,
            "feedback": self.feedback,
        }

    def weak_dimensions(self, threshold: float = 7.0) -> list[str]:
        """
        List which item quality dimensions fall below a threshold.
        
        Parameters:
        	threshold (float): Cutoff value; any dimension with a score less than this is considered weak. Defaults to 7.0.
        
        Returns:
        	weak (list[str]): Names of dimensions whose scores are below `threshold`. Possible names: "story_significance", "uniqueness", "narrative_potential", "integration".
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
        return weak


class ConceptQualityScores(BaseQualityScores):
    """Quality scores for a concept (0-10 scale).

    All score fields are required - no defaults.
    """

    relevance: float = Field(ge=0.0, le=10.0, description="Theme alignment")
    depth: float = Field(ge=0.0, le=10.0, description="Philosophical richness")
    manifestation: float = Field(ge=0.0, le=10.0, description="How it appears in story")
    resonance: float = Field(ge=0.0, le=10.0, description="Emotional impact")

    @property
    def average(self) -> float:
        """
        Return the mean of the concept's relevance, depth, manifestation, and resonance scores.
        
        Returns:
            The average score (0 to 10) across relevance, depth, manifestation, and resonance.
        """
        return (self.relevance + self.depth + self.manifestation + self.resonance) / 4.0

    def to_dict(self) -> dict[str, float | str]:
        """
        Return a mapping of concept quality dimensions and metadata suitable for storing on an entity.
        
        Returns:
            dict[str, float | str]: Dictionary with keys 'relevance', 'depth', 'manifestation', 'resonance', 'average', and 'feedback' mapping to their corresponding values.
        """
        return {
            "relevance": self.relevance,
            "depth": self.depth,
            "manifestation": self.manifestation,
            "resonance": self.resonance,
            "average": self.average,
            "feedback": self.feedback,
        }

    def weak_dimensions(self, threshold: float = 7.0) -> list[str]:
        """
        Identify concept quality dimensions whose scores are below the given threshold.
        
        Parameters:
            threshold (float): Cutoff value; any dimension with a score less than this is considered weak. Defaults to 7.0.
        
        Returns:
            weak_dimensions (list[str]): List of dimension names ("relevance", "depth", "manifestation", "resonance") whose values are below `threshold`.
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
        return weak
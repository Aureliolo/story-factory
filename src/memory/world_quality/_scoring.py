"""Quality score models for world building entities.

Each entity type has its own quality score model with specific dimensions.
All scores use a 0-10 scale and provide average calculation, weak dimension
detection, and dict serialization.
"""

from pydantic import BaseModel, Field


class CharacterQualityScores(BaseModel):
    """Quality scores for a character (0-10 scale).

    All score fields are required - no defaults. This ensures that the LLM
    must provide explicit scores or the parsing will fail, preventing silent
    fallbacks to meaningless default values.
    """

    depth: float = Field(ge=0.0, le=10.0, description="Psychological complexity")
    goals: float = Field(ge=0.0, le=10.0, description="Clarity and story relevance")
    flaws: float = Field(ge=0.0, le=10.0, description="Meaningful vulnerabilities")
    uniqueness: float = Field(ge=0.0, le=10.0, description="Distinctiveness")
    arc_potential: float = Field(ge=0.0, le=10.0, description="Room for transformation")
    feedback: str = ""

    @property
    def average(self) -> float:
        """Calculate average score across all dimensions."""
        return (self.depth + self.goals + self.flaws + self.uniqueness + self.arc_potential) / 5.0

    def to_dict(self) -> dict[str, float | str]:
        """Convert to dictionary for storage in entity attributes."""
        return {
            "depth": self.depth,
            "goals": self.goals,
            "flaws": self.flaws,
            "uniqueness": self.uniqueness,
            "arc_potential": self.arc_potential,
            "average": self.average,
            "feedback": self.feedback,
        }

    def weak_dimensions(self, threshold: float = 7.0) -> list[str]:
        """Return list of dimensions below threshold."""
        weak = []
        if self.depth < threshold:
            weak.append("depth")
        if self.goals < threshold:
            weak.append("goals")
        if self.flaws < threshold:
            weak.append("flaws")
        if self.uniqueness < threshold:
            weak.append("uniqueness")
        if self.arc_potential < threshold:
            weak.append("arc_potential")
        return weak


class LocationQualityScores(BaseModel):
    """Quality scores for a location (0-10 scale).

    All score fields are required - no defaults.
    """

    atmosphere: float = Field(ge=0.0, le=10.0, description="Sensory richness, mood")
    significance: float = Field(ge=0.0, le=10.0, description="Plot/symbolic meaning")
    story_relevance: float = Field(ge=0.0, le=10.0, description="Theme/character links")
    distinctiveness: float = Field(ge=0.0, le=10.0, description="Memorable qualities")
    feedback: str = ""

    @property
    def average(self) -> float:
        """Calculate average score across all dimensions."""
        return (
            self.atmosphere + self.significance + self.story_relevance + self.distinctiveness
        ) / 4.0

    def to_dict(self) -> dict[str, float | str]:
        """Convert to dictionary for storage in entity attributes."""
        return {
            "atmosphere": self.atmosphere,
            "significance": self.significance,
            "story_relevance": self.story_relevance,
            "distinctiveness": self.distinctiveness,
            "average": self.average,
            "feedback": self.feedback,
        }

    def weak_dimensions(self, threshold: float = 7.0) -> list[str]:
        """Return list of dimensions below threshold."""
        weak = []
        if self.atmosphere < threshold:
            weak.append("atmosphere")
        if self.significance < threshold:
            weak.append("significance")
        if self.story_relevance < threshold:
            weak.append("story_relevance")
        if self.distinctiveness < threshold:
            weak.append("distinctiveness")
        return weak


class RelationshipQualityScores(BaseModel):
    """Quality scores for a relationship (0-10 scale).

    All score fields are required - no defaults.
    """

    tension: float = Field(ge=0.0, le=10.0, description="Conflict potential")
    dynamics: float = Field(ge=0.0, le=10.0, description="Complexity, power balance")
    story_potential: float = Field(ge=0.0, le=10.0, description="Scene opportunities")
    authenticity: float = Field(ge=0.0, le=10.0, description="Believability")
    feedback: str = ""

    @property
    def average(self) -> float:
        """Calculate average score across all dimensions."""
        return (self.tension + self.dynamics + self.story_potential + self.authenticity) / 4.0

    def to_dict(self) -> dict[str, float | str]:
        """Convert to dictionary for storage in entity attributes."""
        return {
            "tension": self.tension,
            "dynamics": self.dynamics,
            "story_potential": self.story_potential,
            "authenticity": self.authenticity,
            "average": self.average,
            "feedback": self.feedback,
        }

    def weak_dimensions(self, threshold: float = 7.0) -> list[str]:
        """Return list of dimensions below threshold."""
        weak = []
        if self.tension < threshold:
            weak.append("tension")
        if self.dynamics < threshold:
            weak.append("dynamics")
        if self.story_potential < threshold:
            weak.append("story_potential")
        if self.authenticity < threshold:
            weak.append("authenticity")
        return weak


class FactionQualityScores(BaseModel):
    """Quality scores for a faction (0-10 scale).

    All score fields are required - no defaults.
    """

    coherence: float = Field(ge=0.0, le=10.0, description="Internal consistency")
    influence: float = Field(ge=0.0, le=10.0, description="World impact")
    conflict_potential: float = Field(ge=0.0, le=10.0, description="Story conflict potential")
    distinctiveness: float = Field(ge=0.0, le=10.0, description="Memorable qualities")
    feedback: str = ""

    @property
    def average(self) -> float:
        """Calculate average score across all dimensions."""
        return (
            self.coherence + self.influence + self.conflict_potential + self.distinctiveness
        ) / 4.0

    def to_dict(self) -> dict[str, float | str]:
        """Convert to dictionary for storage in entity attributes."""
        return {
            "coherence": self.coherence,
            "influence": self.influence,
            "conflict_potential": self.conflict_potential,
            "distinctiveness": self.distinctiveness,
            "average": self.average,
            "feedback": self.feedback,
        }

    def weak_dimensions(self, threshold: float = 7.0) -> list[str]:
        """Return list of dimensions below threshold."""
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


class ItemQualityScores(BaseModel):
    """Quality scores for an item (0-10 scale).

    All score fields are required - no defaults.
    """

    significance: float = Field(ge=0.0, le=10.0, description="Story importance")
    uniqueness: float = Field(ge=0.0, le=10.0, description="Distinctive qualities")
    narrative_potential: float = Field(ge=0.0, le=10.0, description="Plot opportunities")
    integration: float = Field(ge=0.0, le=10.0, description="Fits world")
    feedback: str = ""

    @property
    def average(self) -> float:
        """Calculate average score across all dimensions."""
        return (
            self.significance + self.uniqueness + self.narrative_potential + self.integration
        ) / 4.0

    def to_dict(self) -> dict[str, float | str]:
        """Convert to dictionary for storage in entity attributes."""
        return {
            "significance": self.significance,
            "uniqueness": self.uniqueness,
            "narrative_potential": self.narrative_potential,
            "integration": self.integration,
            "average": self.average,
            "feedback": self.feedback,
        }

    def weak_dimensions(self, threshold: float = 7.0) -> list[str]:
        """Return list of dimensions below threshold."""
        weak = []
        if self.significance < threshold:
            weak.append("significance")
        if self.uniqueness < threshold:
            weak.append("uniqueness")
        if self.narrative_potential < threshold:
            weak.append("narrative_potential")
        if self.integration < threshold:
            weak.append("integration")
        return weak


class ConceptQualityScores(BaseModel):
    """Quality scores for a concept (0-10 scale).

    All score fields are required - no defaults.
    """

    relevance: float = Field(ge=0.0, le=10.0, description="Theme alignment")
    depth: float = Field(ge=0.0, le=10.0, description="Philosophical richness")
    manifestation: float = Field(ge=0.0, le=10.0, description="How it appears in story")
    resonance: float = Field(ge=0.0, le=10.0, description="Emotional impact")
    feedback: str = ""

    @property
    def average(self) -> float:
        """Calculate average score across all dimensions."""
        return (self.relevance + self.depth + self.manifestation + self.resonance) / 4.0

    def to_dict(self) -> dict[str, float | str]:
        """Convert to dictionary for storage in entity attributes."""
        return {
            "relevance": self.relevance,
            "depth": self.depth,
            "manifestation": self.manifestation,
            "resonance": self.resonance,
            "average": self.average,
            "feedback": self.feedback,
        }

    def weak_dimensions(self, threshold: float = 7.0) -> list[str]:
        """Return list of dimensions below threshold."""
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

"""Story-related quality score models.

These models define the quality scoring dimensions for story elements:
relationships, plots, and chapters.
"""

from pydantic import Field

from src.memory.world_quality._models import BaseQualityScores


class RelationshipQualityScores(BaseQualityScores):
    """Quality scores for a relationship (0-10 scale).

    All score fields are required - no defaults.
    """

    tension: float = Field(ge=0.0, le=10.0, description="Conflict potential")
    dynamics: float = Field(ge=0.0, le=10.0, description="Complexity, power balance")
    story_potential: float = Field(ge=0.0, le=10.0, description="Scene opportunities")
    authenticity: float = Field(ge=0.0, le=10.0, description="Believability of earned connection")

    @property
    def average(self) -> float:
        """
        Compute the arithmetic mean of the relationship quality dimensions.

        Returns:
            float: The average of `tension`, `dynamics`, `story_potential`, and `authenticity`.
        """
        return (self.tension + self.dynamics + self.story_potential + self.authenticity) / 4.0

    def to_dict(self) -> dict[str, float | str]:
        """
        Serialize relationship quality scores and feedback into a dictionary for storage.

        Returns:
            dict: Mapping with keys "tension", "dynamics", "story_potential", "authenticity" (float scores), "average" (float mean), and "feedback" (str).
        """
        return {
            "tension": self.tension,
            "dynamics": self.dynamics,
            "story_potential": self.story_potential,
            "authenticity": self.authenticity,
            "average": self.average,
            "feedback": self.feedback,
        }

    def weak_dimensions(self, threshold: float = 7.0) -> list[str]:
        """
        Identify score dimensions with values less than the given threshold.

        Parameters:
            threshold (float): Cutoff value; any dimension with a score less than this value is considered weak. Defaults to 7.0.

        Returns:
            list[str]: Names of the dimensions whose scores are less than the threshold.
        """
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


class PlotQualityScores(BaseQualityScores):
    """Quality scores for a plot outline (0-10 scale).

    All score fields are required - no defaults.
    """

    coherence: float = Field(ge=0.0, le=10.0, description="Logical progression")
    tension_arc: float = Field(ge=0.0, le=10.0, description="Stakes and tension")
    character_integration: float = Field(ge=0.0, le=10.0, description="Character arc advancement")
    originality: float = Field(ge=0.0, le=10.0, description="Avoids cliches")

    @property
    def average(self) -> float:
        """
        Compute the mean of the plot quality dimensions.

        Returns:
            average (float): The average of coherence, tension_arc, character_integration, and originality.
        """
        return (
            self.coherence + self.tension_arc + self.character_integration + self.originality
        ) / 4.0

    def to_dict(self) -> dict[str, float | str]:
        """
        Serialize the plot quality scores and metadata into a dictionary for storage.

        Returns:
            dict[str, float | str]: Dictionary containing `coherence`, `tension_arc`, `character_integration`, `originality`, and `average` as floats, and `feedback` as a string.
        """
        return {
            "coherence": self.coherence,
            "tension_arc": self.tension_arc,
            "character_integration": self.character_integration,
            "originality": self.originality,
            "average": self.average,
            "feedback": self.feedback,
        }

    def weak_dimensions(self, threshold: float = 7.0) -> list[str]:
        """
        Identify plot quality dimensions with scores below a given threshold.

        Parameters:
            threshold (float): Score cutoff; any dimension with a value less than this threshold is considered weak.

        Returns:
            list[str]: Names of dimensions (from 'coherence', 'tension_arc', 'character_integration', 'originality') whose scores are less than `threshold`.
        """
        weak = []
        if self.coherence < threshold:
            weak.append("coherence")
        if self.tension_arc < threshold:
            weak.append("tension_arc")
        if self.character_integration < threshold:
            weak.append("character_integration")
        if self.originality < threshold:
            weak.append("originality")
        return weak


class ChapterQualityScores(BaseQualityScores):
    """Quality scores for a chapter outline (0-10 scale).

    All score fields are required - no defaults.
    """

    purpose: float = Field(ge=0.0, le=10.0, description="Plot/character advancement")
    pacing: float = Field(ge=0.0, le=10.0, description="Action/dialogue/reflection balance")
    hook: float = Field(ge=0.0, le=10.0, description="Opening grab and ending compulsion")
    coherence: float = Field(ge=0.0, le=10.0, description="Internal consistency and flow")

    @property
    def average(self) -> float:
        """
        Compute the mean score across the chapter's four quality dimensions.

        Returns:
            float: Arithmetic mean of `purpose`, `pacing`, `hook`, and `coherence`.
        """
        return (self.purpose + self.pacing + self.hook + self.coherence) / 4.0

    def to_dict(self) -> dict[str, float | str]:
        """
        Serialize the quality scores and feedback into a dictionary suitable for storage.

        Returns:
            dict[str, float | str]: Mapping with keys:
                - "purpose", "pacing", "hook", "coherence": individual dimension scores (float)
                - "average": mean score of the dimensions (float)
                - "feedback": textual feedback (str)
        """
        return {
            "purpose": self.purpose,
            "pacing": self.pacing,
            "hook": self.hook,
            "coherence": self.coherence,
            "average": self.average,
            "feedback": self.feedback,
        }

    def weak_dimensions(self, threshold: float = 7.0) -> list[str]:
        """
        Identify score dimensions with values below a given threshold.

        Parameters:
                threshold (float): Cutoff value; any dimension with a score less than this value is considered weak. Defaults to 7.0.

        Returns:
                weak_dimensions (list[str]): Names of dimensions whose scores are less than `threshold`.
        """
        weak = []
        if self.purpose < threshold:
            weak.append("purpose")
        if self.pacing < threshold:
            weak.append("pacing")
        if self.hook < threshold:
            weak.append("hook")
        if self.coherence < threshold:
            weak.append("coherence")
        return weak

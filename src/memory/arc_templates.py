"""Character arc templates for protagonist and antagonist development patterns."""

import logging
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ArcStage(BaseModel):
    """A stage in a character's arc progression."""

    name: str = Field(description="Name of this arc stage")
    description: str = Field(description="What happens to the character at this stage")
    percentage: int = Field(
        ge=0, le=100, description="Story position (0-100) where this stage typically occurs"
    )


class CharacterArcTemplate(BaseModel):
    """Template for a character arc pattern."""

    id: str = Field(description="Unique identifier for this arc template")
    name: str = Field(description="Display name of the arc")
    description: str = Field(description="Brief description of this arc pattern")
    arc_category: Literal["protagonist", "antagonist"] = Field(
        description="Whether this arc is for protagonists or antagonists"
    )
    stages: list[ArcStage] = Field(default_factory=list, description="Ordered list of arc stages")
    required_traits: list[str] = Field(
        default_factory=list, description="Character traits that work well with this arc"
    )
    recommended_relationships: list[str] = Field(
        default_factory=list, description="Relationship types that enhance this arc"
    )


# =============================================================================
# PROTAGONIST ARC TEMPLATES
# =============================================================================

HERO_JOURNEY = CharacterArcTemplate(
    id="hero_journey",
    name="Hero's Journey",
    description="Classic transformation from ordinary person to hero through trials and growth",
    arc_category="protagonist",
    stages=[
        ArcStage(
            name="Ordinary World",
            description="Character lives in their familiar, comfortable world",
            percentage=0,
        ),
        ArcStage(
            name="Call to Adventure",
            description="An event disrupts their world and presents a challenge",
            percentage=10,
        ),
        ArcStage(
            name="Refusal of the Call",
            description="Character hesitates or refuses the challenge due to fear",
            percentage=15,
        ),
        ArcStage(
            name="Meeting the Mentor",
            description="Character encounters a guide who provides wisdom or tools",
            percentage=20,
        ),
        ArcStage(
            name="Crossing the Threshold",
            description="Character commits to the adventure and enters the unknown",
            percentage=25,
        ),
        ArcStage(
            name="Tests, Allies, Enemies",
            description="Character faces challenges and builds relationships",
            percentage=40,
        ),
        ArcStage(
            name="Approach to the Inmost Cave",
            description="Character prepares for their greatest challenge",
            percentage=50,
        ),
        ArcStage(
            name="The Ordeal",
            description="Character faces their greatest fear and nearly fails",
            percentage=60,
        ),
        ArcStage(
            name="Reward",
            description="Character gains something valuable from their trials",
            percentage=70,
        ),
        ArcStage(
            name="The Road Back",
            description="Character begins the journey home, facing new challenges",
            percentage=80,
        ),
        ArcStage(
            name="Resurrection",
            description="Final test where character must apply all they've learned",
            percentage=90,
        ),
        ArcStage(
            name="Return with Elixir",
            description="Character returns transformed, bringing wisdom to share",
            percentage=100,
        ),
    ],
    required_traits=["courage", "determination", "humility", "compassion"],
    recommended_relationships=["mentor", "ally", "love_interest", "threshold_guardian"],
)

REDEMPTION = CharacterArcTemplate(
    id="redemption",
    name="Redemption Arc",
    description="Character overcomes past mistakes and moral failings to become better",
    arc_category="protagonist",
    stages=[
        ArcStage(
            name="Moral Corruption",
            description="Character is deeply flawed, having done terrible things",
            percentage=0,
        ),
        ArcStage(
            name="Wake-up Call",
            description="An event forces character to confront their wrongdoings",
            percentage=15,
        ),
        ArcStage(
            name="Denial/Resistance",
            description="Character resists change, making excuses or deflecting",
            percentage=25,
        ),
        ArcStage(
            name="Acknowledgment",
            description="Character finally admits their faults and feels genuine remorse",
            percentage=40,
        ),
        ArcStage(
            name="Seeking Forgiveness",
            description="Character attempts to make amends to those they've hurt",
            percentage=55,
        ),
        ArcStage(
            name="Test of Commitment",
            description="Character faces a temptation to return to old ways",
            percentage=70,
        ),
        ArcStage(
            name="Sacrifice",
            description="Character makes a significant sacrifice to prove change",
            percentage=85,
        ),
        ArcStage(
            name="Absolution",
            description="Character achieves redemption, though scars remain",
            percentage=100,
        ),
    ],
    required_traits=["guilt", "self-awareness", "capacity_for_change", "courage"],
    recommended_relationships=["victim", "mentor", "skeptic", "believer"],
)

COMING_OF_AGE = CharacterArcTemplate(
    id="coming_of_age",
    name="Coming of Age",
    description="Young character matures through experiences and gains adult understanding",
    arc_category="protagonist",
    stages=[
        ArcStage(
            name="Innocence",
            description="Character has a naive, simplified view of the world",
            percentage=0,
        ),
        ArcStage(
            name="First Challenge",
            description="Character encounters something that doesn't fit their worldview",
            percentage=15,
        ),
        ArcStage(
            name="Confusion",
            description="Character struggles to reconcile new information with old beliefs",
            percentage=30,
        ),
        ArcStage(
            name="Experimentation",
            description="Character tries new behaviors and makes mistakes",
            percentage=45,
        ),
        ArcStage(
            name="Disillusionment",
            description="Character loses faith in something they once believed in",
            percentage=60,
        ),
        ArcStage(
            name="Identity Crisis",
            description="Character must decide who they want to become",
            percentage=75,
        ),
        ArcStage(
            name="Maturity",
            description="Character achieves a more nuanced understanding of the world",
            percentage=100,
        ),
    ],
    required_traits=["curiosity", "adaptability", "emotional_depth", "idealism"],
    recommended_relationships=["mentor", "peer", "first_love", "family"],
)

TRAGEDY = CharacterArcTemplate(
    id="tragedy",
    name="Tragic Arc",
    description="Character's fatal flaw leads to their downfall despite admirable qualities",
    arc_category="protagonist",
    stages=[
        ArcStage(
            name="Greatness",
            description="Character is admirable and successful, with a hidden flaw",
            percentage=0,
        ),
        ArcStage(
            name="Hamartia Revealed",
            description="Character's fatal flaw begins to manifest",
            percentage=20,
        ),
        ArcStage(
            name="Rising Consequences",
            description="Flaw leads to increasingly serious problems",
            percentage=35,
        ),
        ArcStage(
            name="Moment of Choice",
            description="Character could still change course but chooses not to",
            percentage=50,
        ),
        ArcStage(
            name="Point of No Return",
            description="Character's actions become irreversible",
            percentage=65,
        ),
        ArcStage(
            name="Catastrophe",
            description="Character experiences devastating loss or failure",
            percentage=80,
        ),
        ArcStage(
            name="Anagnorisis",
            description="Character finally understands their role in their downfall",
            percentage=90,
        ),
        ArcStage(
            name="Catharsis",
            description="Resolution brings emotional release and tragic meaning",
            percentage=100,
        ),
    ],
    required_traits=["pride", "ambition", "blindness", "nobility"],
    recommended_relationships=["loyal_friend", "victim", "tempter", "warner"],
)


# =============================================================================
# ANTAGONIST ARC TEMPLATES
# =============================================================================

MIRROR = CharacterArcTemplate(
    id="mirror",
    name="Mirror Antagonist",
    description="Antagonist reflects what the protagonist could become without growth",
    arc_category="antagonist",
    stages=[
        ArcStage(
            name="Parallel Origins",
            description="Antagonist's backstory mirrors protagonist's in key ways",
            percentage=0,
        ),
        ArcStage(
            name="Different Choice",
            description="Reveal how antagonist made opposite choice at critical moment",
            percentage=25,
        ),
        ArcStage(
            name="Confrontation of Similarity",
            description="Protagonist recognizes themselves in the antagonist",
            percentage=50,
        ),
        ArcStage(
            name="Contrast Deepens",
            description="Their divergent paths become clearer through conflict",
            percentage=75,
        ),
        ArcStage(
            name="Final Mirror",
            description="Ultimate confrontation highlights what separates them",
            percentage=100,
        ),
    ],
    required_traits=["charisma", "intelligence", "wounded_past", "conviction"],
    recommended_relationships=["rival", "dark_mentor", "shared_history"],
)

FORCE_OF_NATURE = CharacterArcTemplate(
    id="force_of_nature",
    name="Force of Nature",
    description="Antagonist represents an unstoppable, implacable threat",
    arc_category="antagonist",
    stages=[
        ArcStage(
            name="Distant Threat",
            description="Antagonist is known but seems far away or theoretical",
            percentage=0,
        ),
        ArcStage(
            name="First Strike",
            description="Antagonist's power is demonstrated in a devastating way",
            percentage=20,
        ),
        ArcStage(
            name="Relentless Pursuit",
            description="Antagonist cannot be reasoned with, bargained with, or stopped",
            percentage=40,
        ),
        ArcStage(
            name="Seeming Defeat",
            description="Heroes think they've won, but antagonist returns",
            percentage=60,
        ),
        ArcStage(
            name="Full Power",
            description="Antagonist reaches peak threat level",
            percentage=80,
        ),
        ArcStage(
            name="Resolution",
            description="Antagonist is finally overcome through cleverness or sacrifice",
            percentage=100,
        ),
    ],
    required_traits=["implacability", "power", "singlemindedness", "terror"],
    recommended_relationships=["hunted", "collateral_victim", "failed_challenger"],
)

FALLEN_HERO = CharacterArcTemplate(
    id="fallen_hero",
    name="Fallen Hero",
    description="Antagonist was once good but was corrupted by circumstances or choices",
    arc_category="antagonist",
    stages=[
        ArcStage(
            name="Heroic Past",
            description="Antagonist's noble origins are hinted at or revealed",
            percentage=0,
        ),
        ArcStage(
            name="The Wound",
            description="Reveal the trauma or betrayal that started their fall",
            percentage=20,
        ),
        ArcStage(
            name="Corruption Deepens",
            description="Show how initial compromise led to greater evil",
            percentage=40,
        ),
        ArcStage(
            name="Moment of Humanity",
            description="Brief glimpse of the person they once were",
            percentage=60,
        ),
        ArcStage(
            name="Choice Point",
            description="Antagonist could still be redeemed but refuses",
            percentage=80,
        ),
        ArcStage(
            name="Final Fate",
            description="Antagonist meets end that reflects their journey",
            percentage=100,
        ),
    ],
    required_traits=["idealism_corrupted", "bitterness", "power", "tragic_nobility"],
    recommended_relationships=["former_ally", "disappointed_mentor", "betrayer"],
)

TRUE_BELIEVER = CharacterArcTemplate(
    id="true_believer",
    name="True Believer",
    description="Antagonist believes completely in their cause, making them dangerous",
    arc_category="antagonist",
    stages=[
        ArcStage(
            name="Introduction of Ideology",
            description="Antagonist's beliefs and goals are established",
            percentage=0,
        ),
        ArcStage(
            name="Justification",
            description="Antagonist explains why ends justify means",
            percentage=20,
        ),
        ArcStage(
            name="Zealous Action",
            description="Antagonist commits terrible acts with clear conscience",
            percentage=40,
        ),
        ArcStage(
            name="Challenge to Faith",
            description="Evidence that their cause may be flawed",
            percentage=60,
        ),
        ArcStage(
            name="Doubling Down",
            description="Antagonist becomes more extreme rather than reconsider",
            percentage=80,
        ),
        ArcStage(
            name="Reckoning",
            description="Antagonist faces consequences of their fanaticism",
            percentage=100,
        ),
    ],
    required_traits=["conviction", "intelligence", "charisma", "ruthlessness"],
    recommended_relationships=["follower", "doubter", "victim_of_cause", "ideological_opponent"],
)

MASTERMIND = CharacterArcTemplate(
    id="mastermind",
    name="Mastermind",
    description="Antagonist operates through elaborate plans and manipulation",
    arc_category="antagonist",
    stages=[
        ArcStage(
            name="Hidden Hand",
            description="Antagonist's influence is felt but they remain unseen",
            percentage=0,
        ),
        ArcStage(
            name="Reveal of Scope",
            description="Heroes realize how extensive the antagonist's plans are",
            percentage=20,
        ),
        ArcStage(
            name="Chess Game",
            description="Back-and-forth as heroes try to outmaneuver antagonist",
            percentage=40,
        ),
        ArcStage(
            name="Trap Sprung",
            description="Heroes fall into antagonist's carefully laid trap",
            percentage=60,
        ),
        ArcStage(
            name="Countermove",
            description="Heroes find unexpected way to disrupt the plan",
            percentage=80,
        ),
        ArcStage(
            name="Final Gambit",
            description="Ultimate confrontation of wits and will",
            percentage=100,
        ),
    ],
    required_traits=["genius", "patience", "manipulation", "ego"],
    recommended_relationships=["pawn", "rival_intellect", "unexpected_variable"],
)


# =============================================================================
# BUILT-IN TEMPLATES REGISTRY
# =============================================================================

BUILTIN_ARC_TEMPLATES: dict[str, CharacterArcTemplate] = {
    # Protagonist arcs
    "hero_journey": HERO_JOURNEY,
    "redemption": REDEMPTION,
    "coming_of_age": COMING_OF_AGE,
    "tragedy": TRAGEDY,
    # Antagonist arcs
    "mirror": MIRROR,
    "force_of_nature": FORCE_OF_NATURE,
    "fallen_hero": FALLEN_HERO,
    "true_believer": TRUE_BELIEVER,
    "mastermind": MASTERMIND,
}


def get_arc_template(arc_id: str) -> CharacterArcTemplate | None:
    """Get an arc template by ID.

    Args:
        arc_id: The arc template ID to look up.

    Returns:
        The arc template if found, None otherwise.
    """
    template = BUILTIN_ARC_TEMPLATES.get(arc_id)
    if template:
        logger.debug(f"Retrieved arc template: {arc_id}")
    else:
        logger.warning(f"Arc template not found: {arc_id}")
    return template


def list_arc_templates(
    category: Literal["protagonist", "antagonist"] | None = None,
) -> list[CharacterArcTemplate]:
    """List all available arc templates, optionally filtered by category.

    Args:
        category: Optional filter for protagonist or antagonist arcs.

    Returns:
        List of matching arc templates.
    """
    templates = list(BUILTIN_ARC_TEMPLATES.values())
    if category:
        templates = [t for t in templates if t.arc_category == category]
        logger.debug(f"Listed {len(templates)} {category} arc templates")
    else:
        logger.debug(f"Listed {len(templates)} arc templates (all categories)")
    return templates


def format_arc_guidance(arc_template: CharacterArcTemplate) -> str:
    """Format an arc template into prompt guidance text.

    Args:
        arc_template: The arc template to format.

    Returns:
        Formatted string describing the arc for prompt injection.
    """
    lines = [
        f"CHARACTER ARC: {arc_template.name}",
        f"Description: {arc_template.description}",
        "",
        "Arc Stages:",
    ]

    for stage in arc_template.stages:
        lines.append(f"  - {stage.name} ({stage.percentage}%): {stage.description}")

    if arc_template.required_traits:
        traits = ", ".join(arc_template.required_traits)
        lines.append(f"\nRecommended Traits: {traits}")

    if arc_template.recommended_relationships:
        rels = ", ".join(arc_template.recommended_relationships)
        lines.append(f"Recommended Relationships: {rels}")

    guidance = "\n".join(lines)
    logger.debug(f"Formatted arc guidance for {arc_template.id} ({len(guidance)} chars)")
    return guidance

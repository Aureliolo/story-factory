"""Built-in story templates and structure presets."""

from memory.templates import (
    CharacterTemplate,
    PlotPointTemplate,
    StoryTemplate,
    StructurePreset,
)

# ============================================================================
# STRUCTURE PRESETS
# ============================================================================

THREE_ACT_STRUCTURE = StructurePreset(
    id="three-act",
    name="Three-Act Structure",
    description="Classic three-act narrative structure with setup, confrontation, and resolution",
    acts=["Act 1: Setup", "Act 2: Confrontation", "Act 3: Resolution"],
    plot_points=[
        PlotPointTemplate(
            description="Opening Image - Establish the world and protagonist",
            act=1,
            percentage=0,
        ),
        PlotPointTemplate(
            description="Inciting Incident - Event that sets the story in motion",
            act=1,
            percentage=12,
        ),
        PlotPointTemplate(
            description="First Plot Point - Protagonist commits to the journey",
            act=1,
            percentage=25,
        ),
        PlotPointTemplate(
            description="Midpoint - Major revelation or shift in the story",
            act=2,
            percentage=50,
        ),
        PlotPointTemplate(
            description="Second Plot Point - All is lost moment",
            act=2,
            percentage=75,
        ),
        PlotPointTemplate(
            description="Climax - Final confrontation",
            act=3,
            percentage=90,
        ),
        PlotPointTemplate(
            description="Resolution - New equilibrium established",
            act=3,
            percentage=100,
        ),
    ],
    beats=[
        "Hook the reader",
        "Establish normal world",
        "Inciting incident",
        "Rising action",
        "Midpoint twist",
        "Complications escalate",
        "Dark night of the soul",
        "Climactic confrontation",
        "Resolution and epilogue",
    ],
)

HEROS_JOURNEY = StructurePreset(
    id="heros-journey",
    name="Hero's Journey",
    description="Joseph Campbell's monomyth - the classic hero's adventure structure",
    acts=["Departure", "Initiation", "Return"],
    plot_points=[
        PlotPointTemplate(
            description="Ordinary World - Hero in their normal environment",
            act=1,
            percentage=0,
        ),
        PlotPointTemplate(
            description="Call to Adventure - Challenge or quest presented",
            act=1,
            percentage=10,
        ),
        PlotPointTemplate(
            description="Refusal of the Call - Hero hesitates",
            act=1,
            percentage=15,
        ),
        PlotPointTemplate(
            description="Meeting the Mentor - Guidance received",
            act=1,
            percentage=20,
        ),
        PlotPointTemplate(
            description="Crossing the Threshold - Entering the special world",
            act=1,
            percentage=25,
        ),
        PlotPointTemplate(
            description="Tests, Allies, and Enemies - Learning the new world",
            act=2,
            percentage=40,
        ),
        PlotPointTemplate(
            description="Approach to the Inmost Cave - Preparing for the ordeal",
            act=2,
            percentage=55,
        ),
        PlotPointTemplate(
            description="The Ordeal - Greatest fear confronted",
            act=2,
            percentage=65,
        ),
        PlotPointTemplate(
            description="Reward - Prize obtained after ordeal",
            act=2,
            percentage=70,
        ),
        PlotPointTemplate(
            description="The Road Back - Return journey begins",
            act=3,
            percentage=80,
        ),
        PlotPointTemplate(
            description="Resurrection - Final test, hero reborn",
            act=3,
            percentage=90,
        ),
        PlotPointTemplate(
            description="Return with Elixir - Hero brings wisdom home",
            act=3,
            percentage=100,
        ),
    ],
    beats=[
        "Show ordinary world",
        "Present the call to adventure",
        "Hero refuses or hesitates",
        "Mentor appears with guidance",
        "Cross into special world",
        "Face tests and make allies",
        "Approach greatest challenge",
        "Survive the ordeal",
        "Claim the reward",
        "Journey homeward",
        "Final transformation",
        "Return changed with knowledge",
    ],
)

SAVE_THE_CAT = StructurePreset(
    id="save-the-cat",
    name="Save The Cat",
    description="Blake Snyder's beat sheet - proven structure for commercial storytelling",
    acts=["Act 1", "Act 2A", "Act 2B", "Act 3"],
    plot_points=[
        PlotPointTemplate(
            description="Opening Image - Snapshot of hero before journey",
            act=1,
            percentage=1,
        ),
        PlotPointTemplate(
            description="Theme Stated - What the story is about",
            act=1,
            percentage=5,
        ),
        PlotPointTemplate(
            description="Setup - Establish stakes and world",
            act=1,
            percentage=10,
        ),
        PlotPointTemplate(
            description="Catalyst - Event that changes everything",
            act=1,
            percentage=12,
        ),
        PlotPointTemplate(
            description="Debate - Hero hesitates, weighs options",
            act=1,
            percentage=20,
        ),
        PlotPointTemplate(
            description="Break into Two - Hero commits to new path",
            act=1,
            percentage=25,
        ),
        PlotPointTemplate(
            description="B Story - Secondary plot/relationship begins",
            act=2,
            percentage=30,
        ),
        PlotPointTemplate(
            description="Fun and Games - Promise of the premise",
            act=2,
            percentage=40,
        ),
        PlotPointTemplate(
            description="Midpoint - False victory or defeat",
            act=2,
            percentage=50,
        ),
        PlotPointTemplate(
            description="Bad Guys Close In - Pressure increases",
            act=2,
            percentage=60,
        ),
        PlotPointTemplate(
            description="All Is Lost - Lowest point",
            act=2,
            percentage=75,
        ),
        PlotPointTemplate(
            description="Dark Night of the Soul - Hero processes loss",
            act=2,
            percentage=80,
        ),
        PlotPointTemplate(
            description="Break into Three - Hero finds solution",
            act=3,
            percentage=85,
        ),
        PlotPointTemplate(
            description="Finale - Execute plan and triumph",
            act=3,
            percentage=95,
        ),
        PlotPointTemplate(
            description="Final Image - Opposite of opening image",
            act=3,
            percentage=100,
        ),
    ],
    beats=[
        "Opening snapshot",
        "State the theme",
        "Set up the world",
        "Catalyst strikes",
        "Hero debates",
        "Commit to change",
        "B story develops",
        "Fun and games",
        "Midpoint twist",
        "Opposition intensifies",
        "Lowest moment",
        "Dark reflection",
        "Discovery and decision",
        "Finale triumph",
        "New world snapshot",
    ],
)

# ============================================================================
# GENRE TEMPLATES
# ============================================================================

MYSTERY_TEMPLATE = StoryTemplate(
    id="mystery-detective",
    name="Mystery / Detective",
    description="Classic whodunit with clues, red herrings, and a satisfying resolution",
    genre="Mystery",
    subgenres=["Detective", "Crime"],
    tone="Suspenseful with moments of intrigue",
    themes=["Justice", "Truth", "Deception", "Morality"],
    setting_time="Contemporary",
    setting_place="Urban city",
    target_length="novel",
    structure_preset_id="three-act",
    world_description="A gritty urban environment where crime lurks beneath the surface of normalcy. Law enforcement and criminals engage in a constant game of cat and mouse.",
    world_rules=[
        "Every mystery has a logical solution",
        "Fair play with readers - all clues are presented",
        "Red herrings misdirect but don't cheat",
        "The detective cannot be the culprit",
    ],
    characters=[
        CharacterTemplate(
            name="Detective/Investigator",
            role="protagonist",
            description="Sharp-minded investigator with keen observation skills and troubled past",
            personality_traits=[
                "Analytical",
                "Perceptive",
                "Determined",
                "Flawed but capable",
            ],
            goals=[
                "Solve the central mystery",
                "Find justice for victims",
                "Overcome personal demons",
            ],
            arc_notes="Grows from skeptical loner to someone who trusts others and themselves",
        ),
        CharacterTemplate(
            name="The Victim",
            role="supporting",
            description="The person whose death or misfortune triggers the investigation",
            personality_traits=["Complex", "Had secrets", "Connected to many"],
            goals=["Justice (posthumously)"],
            arc_notes="Revealed through investigation",
        ),
        CharacterTemplate(
            name="Primary Suspect",
            role="antagonist",
            description="The most obvious culprit with motive and opportunity",
            personality_traits=["Defensive", "Hiding something", "Morally gray"],
            goals=["Protect their secret", "Avoid blame"],
            arc_notes="May or may not be the true culprit",
        ),
        CharacterTemplate(
            name="The Partner",
            role="supporting",
            description="Fellow investigator or assistant who provides contrast and support",
            personality_traits=["Loyal", "Different approach", "Grounding influence"],
            goals=["Support protagonist", "Solve the case"],
            arc_notes="Learns to trust protagonist's instincts",
        ),
    ],
    plot_points=[
        PlotPointTemplate(
            description="Discovery of the crime/mystery",
            act=1,
            percentage=5,
        ),
        PlotPointTemplate(
            description="Detective takes the case",
            act=1,
            percentage=15,
        ),
        PlotPointTemplate(
            description="First major clue discovered",
            act=1,
            percentage=25,
        ),
        PlotPointTemplate(
            description="Interview key witnesses/suspects",
            act=2,
            percentage=35,
        ),
        PlotPointTemplate(
            description="False solution or red herring exposed",
            act=2,
            percentage=50,
        ),
        PlotPointTemplate(
            description="Major revelation changes everything",
            act=2,
            percentage=65,
        ),
        PlotPointTemplate(
            description="Detective in danger/threatened",
            act=2,
            percentage=75,
        ),
        PlotPointTemplate(
            description="Final piece of puzzle falls into place",
            act=3,
            percentage=85,
        ),
        PlotPointTemplate(
            description="Confrontation with culprit and reveal",
            act=3,
            percentage=95,
        ),
    ],
    tags=["mystery", "detective", "crime", "thriller", "investigation"],
)

ROMANCE_TEMPLATE = StoryTemplate(
    id="romance-contemporary",
    name="Contemporary Romance",
    description="Modern love story with emotional depth and satisfying happily-ever-after",
    genre="Romance",
    subgenres=["Contemporary"],
    tone="Warm, emotional, with moments of humor and tension",
    themes=["Love", "Self-discovery", "Vulnerability", "Trust"],
    setting_time="Modern day",
    setting_place="Contemporary city or small town",
    target_length="novel",
    structure_preset_id="three-act",
    world_description="A realistic contemporary setting where two people navigate the complexities of modern relationships, careers, and personal growth.",
    world_rules=[
        "Emotional honesty drives the narrative",
        "Both protagonists must grow and change",
        "External conflicts serve internal character development",
        "Happy ending or happy-for-now is essential",
    ],
    characters=[
        CharacterTemplate(
            name="First Protagonist",
            role="protagonist",
            description="Compelling lead with relatable flaws and dreams",
            personality_traits=[
                "Passionate about something",
                "Wounded from past",
                "Guarded emotionally",
                "Capable of growth",
            ],
            goals=[
                "Achieve professional/personal goal",
                "Protect heart from hurt",
                "Eventually: open up to love",
            ],
            arc_notes="Learns to trust and be vulnerable, heals from past wounds",
        ),
        CharacterTemplate(
            name="Second Protagonist",
            role="protagonist",
            description="Equally compelling lead who complements first protagonist",
            personality_traits=[
                "Different but compatible",
                "Has own goals and life",
                "Patient or challenging as needed",
                "Emotionally available (or learns to be)",
            ],
            goals=[
                "Pursue own dreams",
                "Break through partner's walls",
                "Build genuine connection",
            ],
            arc_notes="Also grows and changes through relationship",
        ),
        CharacterTemplate(
            name="Best Friend/Confidant",
            role="supporting",
            description="Trusted friend who provides advice and comic relief",
            personality_traits=["Supportive", "Honest", "Perceptive", "Funny"],
            goals=["See protagonist happy", "Offer wisdom"],
            arc_notes="May have own subplot",
        ),
        CharacterTemplate(
            name="The Ex or Rival",
            role="antagonist",
            description="Past relationship or rival for affection that creates tension",
            personality_traits=["Represents past", "Attractive but wrong fit"],
            goals=["Win protagonist back", "Create obstacles"],
            arc_notes="Eventually exits gracefully or reveals true nature",
        ),
    ],
    plot_points=[
        PlotPointTemplate(
            description="Meet cute - protagonists first encounter",
            act=1,
            percentage=10,
        ),
        PlotPointTemplate(
            description="Initial attraction despite resistance",
            act=1,
            percentage=20,
        ),
        PlotPointTemplate(
            description="Forced proximity or reason to interact",
            act=1,
            percentage=25,
        ),
        PlotPointTemplate(
            description="Growing connection and chemistry",
            act=2,
            percentage=40,
        ),
        PlotPointTemplate(
            description="First kiss or declaration of feelings",
            act=2,
            percentage=50,
        ),
        PlotPointTemplate(
            description="Relationship deepens, walls come down",
            act=2,
            percentage=60,
        ),
        PlotPointTemplate(
            description="Black moment - big fight or separation",
            act=2,
            percentage=75,
        ),
        PlotPointTemplate(
            description="Protagonist realization and growth",
            act=3,
            percentage=85,
        ),
        PlotPointTemplate(
            description="Grand gesture or reconciliation",
            act=3,
            percentage=95,
        ),
        PlotPointTemplate(
            description="Happily ever after confirmed",
            act=3,
            percentage=100,
        ),
    ],
    tags=["romance", "love story", "contemporary", "relationship"],
)

SCIFI_TEMPLATE = StoryTemplate(
    id="scifi-space-opera",
    name="Science Fiction / Space Opera",
    description="Epic space adventure with advanced technology and interstellar stakes",
    genre="Science Fiction",
    subgenres=["Space Opera", "Adventure"],
    tone="Epic and adventurous with philosophical undertones",
    themes=["Exploration", "Technology vs Humanity", "Unity", "Survival"],
    setting_time="Far future",
    setting_place="Multiple planets and space stations",
    target_length="novel",
    structure_preset_id="heros-journey",
    world_description="A vast galaxy with multiple alien species, advanced technology, and interstellar politics. Faster-than-light travel connects distant worlds.",
    world_rules=[
        "Technology follows consistent scientific principles",
        "Different species have unique cultures and biology",
        "Actions have consequences across star systems",
        "Ancient mysteries hint at precursor civilizations",
    ],
    characters=[
        CharacterTemplate(
            name="The Captain/Pilot",
            role="protagonist",
            description="Skilled but reluctant leader thrust into galactic events",
            personality_traits=[
                "Resourceful",
                "Independent",
                "Skeptical of authority",
                "Natural leader",
            ],
            goals=[
                "Complete the mission",
                "Keep crew safe",
                "Uncover the truth",
            ],
            arc_notes="Grows from lone wolf to inspirational leader",
        ),
        CharacterTemplate(
            name="The Alien Ally",
            role="supporting",
            description="Non-human companion with unique perspective and abilities",
            personality_traits=["Logical or emotional", "Loyal", "Different worldview"],
            goals=["Bridge species divide", "Protect protagonist"],
            arc_notes="Learns to understand humanity",
        ),
        CharacterTemplate(
            name="The Scientist/Engineer",
            role="supporting",
            description="Brilliant mind who solves technical challenges",
            personality_traits=["Intelligent", "Curious", "Awkward socially"],
            goals=["Understand alien technology", "Make discoveries"],
            arc_notes="Gains confidence and courage",
        ),
        CharacterTemplate(
            name="The Admiral/Antagonist",
            role="antagonist",
            description="Powerful figure representing opposing ideology or faction",
            personality_traits=[
                "Authoritarian",
                "Believes end justifies means",
                "Charismatic",
            ],
            goals=["Control the galaxy", "Eliminate threats"],
            arc_notes="May have sympathetic motivation",
        ),
    ],
    plot_points=[
        PlotPointTemplate(
            description="Discovery of alien artifact or distress signal",
            act=1,
            percentage=10,
        ),
        PlotPointTemplate(
            description="Departure from known space",
            act=1,
            percentage=25,
        ),
        PlotPointTemplate(
            description="First contact with alien species",
            act=2,
            percentage=35,
        ),
        PlotPointTemplate(
            description="Revelation about true nature of threat",
            act=2,
            percentage=50,
        ),
        PlotPointTemplate(
            description="Major space battle or confrontation",
            act=2,
            percentage=65,
        ),
        PlotPointTemplate(
            description="Crew scattered or ship destroyed",
            act=2,
            percentage=75,
        ),
        PlotPointTemplate(
            description="Rally allies for final assault",
            act=3,
            percentage=85,
        ),
        PlotPointTemplate(
            description="Climactic battle for galaxy's fate",
            act=3,
            percentage=95,
        ),
    ],
    tags=["sci-fi", "space opera", "adventure", "aliens", "technology"],
)

FANTASY_TEMPLATE = StoryTemplate(
    id="fantasy-epic",
    name="Epic Fantasy",
    description="Grand fantasy adventure with magic, quests, and world-changing stakes",
    genre="Fantasy",
    subgenres=["Epic Fantasy", "High Fantasy"],
    tone="Epic and mythic with moments of wonder",
    themes=["Good vs Evil", "Power and Corruption", "Destiny", "Sacrifice"],
    setting_time="Medieval-inspired fantasy world",
    setting_place="Multiple kingdoms and mystical lands",
    target_length="novel",
    structure_preset_id="heros-journey",
    world_description="A richly detailed fantasy realm with diverse kingdoms, ancient magic systems, mythical creatures, and looming darkness threatening the land.",
    world_rules=[
        "Magic follows established rules and has costs",
        "Ancient prophecies shape events",
        "Different races have unique abilities and cultures",
        "Balance between light and darkness is crucial",
    ],
    characters=[
        CharacterTemplate(
            name="The Chosen One",
            role="protagonist",
            description="Unlikely hero destined for greatness",
            personality_traits=[
                "Humble origins",
                "Brave despite fear",
                "Questions destiny",
                "Compassionate",
            ],
            goals=[
                "Master their powers",
                "Defeat the darkness",
                "Save loved ones",
            ],
            arc_notes="Accepts destiny and becomes the leader needed",
        ),
        CharacterTemplate(
            name="The Wise Mentor",
            role="supporting",
            description="Experienced magic user or warrior who guides the hero",
            personality_traits=["Wise", "Secretive past", "Patient teacher"],
            goals=["Prepare hero for coming trials", "Atone for past failures"],
            arc_notes="May sacrifice themselves for hero's growth",
        ),
        CharacterTemplate(
            name="The Loyal Companion",
            role="supporting",
            description="Friend who stands by hero through everything",
            personality_traits=["Loyal", "Courageous", "Provides balance"],
            goals=["Support the hero", "Prove their worth"],
            arc_notes="Discovers own inner strength",
        ),
        CharacterTemplate(
            name="The Dark Lord",
            role="antagonist",
            description="Ancient evil seeking to conquer or destroy the world",
            personality_traits=[
                "Powerful",
                "Corrupted by dark magic",
                "Patient and cunning",
            ],
            goals=["Conquer all kingdoms", "Destroy the chosen one"],
            arc_notes="Tragic backstory may be revealed",
        ),
    ],
    plot_points=[
        PlotPointTemplate(
            description="Discovery of hero's special nature or destiny",
            act=1,
            percentage=15,
        ),
        PlotPointTemplate(
            description="Mentor appears with call to adventure",
            act=1,
            percentage=20,
        ),
        PlotPointTemplate(
            description="Leaving home after tragedy or threat",
            act=1,
            percentage=25,
        ),
        PlotPointTemplate(
            description="Training montage and skill development",
            act=2,
            percentage=35,
        ),
        PlotPointTemplate(
            description="First major victory against evil forces",
            act=2,
            percentage=45,
        ),
        PlotPointTemplate(
            description="Revelation about true scope of darkness",
            act=2,
            percentage=55,
        ),
        PlotPointTemplate(
            description="Mentor's death or great loss",
            act=2,
            percentage=70,
        ),
        PlotPointTemplate(
            description="Hero discovers final power or truth",
            act=3,
            percentage=85,
        ),
        PlotPointTemplate(
            description="Epic final battle with Dark Lord",
            act=3,
            percentage=95,
        ),
    ],
    tags=["fantasy", "epic", "magic", "quest", "medieval"],
)

THRILLER_TEMPLATE = StoryTemplate(
    id="thriller-action",
    name="Action Thriller",
    description="Fast-paced thriller with high stakes and relentless tension",
    genre="Thriller",
    subgenres=["Action", "Suspense"],
    tone="Tense and urgent with explosive moments",
    themes=["Survival", "Justice", "Corruption", "Redemption"],
    setting_time="Contemporary",
    setting_place="Multiple urban and remote locations",
    target_length="novel",
    structure_preset_id="save-the-cat",
    world_description="A dangerous contemporary world where conspiracies lurk beneath the surface and protagonists must rely on skills and wits to survive against powerful enemies.",
    world_rules=[
        "Ticking clock creates urgency",
        "Trust no one completely",
        "Resources are limited",
        "Escalation is constant",
    ],
    characters=[
        CharacterTemplate(
            name="The Operative/Agent",
            role="protagonist",
            description="Highly skilled individual with combat and tactical expertise",
            personality_traits=[
                "Resourceful",
                "Haunted by past",
                "Determined",
                "Lone wolf",
            ],
            goals=[
                "Stop the threat",
                "Protect innocents",
                "Clear their name",
            ],
            arc_notes="Learns to trust others and accept help",
        ),
        CharacterTemplate(
            name="The Civilian",
            role="supporting",
            description="Ordinary person caught up in events who must adapt quickly",
            personality_traits=["Resilient", "Quick learner", "Moral compass"],
            goals=["Survive", "Help protagonist", "Protect loved ones"],
            arc_notes="Discovers inner strength and courage",
        ),
        CharacterTemplate(
            name="The Handler/Ally",
            role="supporting",
            description="Contact who provides information and resources",
            personality_traits=["Connected", "Mysterious", "Loyal (possibly)"],
            goals=["Support mission", "May have hidden agenda"],
            arc_notes="True loyalties tested",
        ),
        CharacterTemplate(
            name="The Mastermind",
            role="antagonist",
            description="Powerful villain orchestrating events from shadows",
            personality_traits=[
                "Intelligent",
                "Ruthless",
                "Always one step ahead",
            ],
            goals=["Execute conspiracy", "Eliminate protagonist"],
            arc_notes="Revealed in layers throughout story",
        ),
    ],
    plot_points=[
        PlotPointTemplate(
            description="Explosive opening action sequence",
            act=1,
            percentage=1,
        ),
        PlotPointTemplate(
            description="Protagonist targeted or framed",
            act=1,
            percentage=15,
        ),
        PlotPointTemplate(
            description="On the run, must go dark",
            act=1,
            percentage=25,
        ),
        PlotPointTemplate(
            description="Uncover first layer of conspiracy",
            act=2,
            percentage=40,
        ),
        PlotPointTemplate(
            description="Major action setpiece and narrow escape",
            act=2,
            percentage=50,
        ),
        PlotPointTemplate(
            description="Betrayal or unexpected twist",
            act=2,
            percentage=65,
        ),
        PlotPointTemplate(
            description="Seemingly defeated, all assets lost",
            act=2,
            percentage=75,
        ),
        PlotPointTemplate(
            description="Final confrontation setup",
            act=3,
            percentage=85,
        ),
        PlotPointTemplate(
            description="Climactic showdown with mastermind",
            act=3,
            percentage=95,
        ),
    ],
    tags=["thriller", "action", "suspense", "conspiracy", "espionage"],
)

# ============================================================================
# REGISTRY
# ============================================================================

BUILTIN_STRUCTURE_PRESETS: dict[str, StructurePreset] = {
    "three-act": THREE_ACT_STRUCTURE,
    "heros-journey": HEROS_JOURNEY,
    "save-the-cat": SAVE_THE_CAT,
}

BUILTIN_STORY_TEMPLATES: dict[str, StoryTemplate] = {
    "mystery-detective": MYSTERY_TEMPLATE,
    "romance-contemporary": ROMANCE_TEMPLATE,
    "scifi-space-opera": SCIFI_TEMPLATE,
    "fantasy-epic": FANTASY_TEMPLATE,
    "thriller-action": THRILLER_TEMPLATE,
}

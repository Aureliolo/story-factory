"""Mock LLM responses for integration testing.

These responses simulate realistic AI outputs for different agents and scenarios.
"""

# Interview responses
INTERVIEW_INITIAL_QUESTIONS = """What kind of story would you like to create today? I'm here to help you develop your idea into a complete narrative.

Some questions to get us started:
1. What genre are you thinking about? (e.g., Fantasy, Mystery, Science Fiction, Romance)
2. Do you have a core premise or idea in mind?
3. What kind of tone are you aiming for? (e.g., Dark, Humorous, Epic, Intimate)

Feel free to share whatever you have in mind, and I'll help you develop it further!"""

INTERVIEW_FOLLOW_UP = """Great! A science fiction mystery sounds fascinating. Let me ask a few more questions to flesh out the story:

1. What is the setting? (e.g., distant planet, space station, future Earth)
2. Who is the main character? What's their role or profession?
3. What's the central mystery they need to solve?
4. What's at stake if they fail to solve it?

These details will help us create a compelling narrative."""

INTERVIEW_FINAL_RESPONSE = """Excellent! I have everything I need to create your story. Here's what we'll be working with:

**Genre:** Science Fiction Mystery
**Setting:** A mining colony on Mars in 2157
**Protagonist:** Detective Sarah Chen, a former Earth detective now working in the colony
**Central Mystery:** A series of strange disappearances in the mining tunnels
**Stakes:** The colony's survival depends on solving the mystery before more lives are lost

I'll now help you build the world, characters, and plot structure for this story."""

# Architect responses
ARCHITECT_WORLD_RESPONSE = """{
  "world_description": "Mars Colony Alpha, established 2145, is humanity's largest off-world settlement. The colony consists of pressurized domes connected by underground tunnels. Below the surface, vast mining operations extract rare minerals crucial for Earth's technology. The colony is home to 50,000 people from diverse backgrounds.",
  "world_rules": [
    "All surface activity requires environmental suits",
    "The colony runs on a strict air and water recycling system",
    "Communication with Earth has a 20-minute delay",
    "The mining tunnels extend deep into the planet, some areas unexplored",
    "A corporate hierarchy governs the colony, prioritizing profit over safety"
  ],
  "characters": [
    {
      "name": "Sarah Chen",
      "role": "protagonist",
      "description": "A 42-year-old detective who transferred to Mars after a controversial case on Earth",
      "personality_traits": ["determined", "intuitive", "haunted by past failures", "adaptable"],
      "goals": ["Solve the disappearances", "Prove herself in her new position", "Find redemption"]
    },
    {
      "name": "Marcus Webb",
      "role": "antagonist",
      "description": "Chief Operations Officer of the mining company, hiding dangerous secrets",
      "personality_traits": ["calculating", "charismatic", "ruthless", "ambitious"],
      "goals": ["Protect company profits", "Keep the truth hidden", "Maintain control"]
    },
    {
      "name": "Dr. Yuki Tanaka",
      "role": "supporting",
      "description": "Colonial physician and Sarah's only ally",
      "personality_traits": ["compassionate", "principled", "cautious", "intelligent"],
      "goals": ["Help the missing people", "Support Sarah's investigation", "Expose the truth"]
    }
  ],
  "plot_summary": "Detective Sarah Chen investigates mysterious disappearances in the Mars colony's mining tunnels. As she digs deeper, she uncovers a conspiracy involving illegal experiments and corporate cover-ups. Racing against time, she must expose the truth before more lives are lost.",
  "plot_points": [
    {
      "description": "First disappearance reported - a young miner vanishes without a trace",
      "chapter": 1
    },
    {
      "description": "Sarah discovers anomalous readings in the deep tunnels",
      "chapter": 2
    },
    {
      "description": "Dr. Tanaka shares medical records showing unusual symptoms in miners",
      "chapter": 3
    },
    {
      "description": "Sarah finds evidence of unauthorized experiments in abandoned sections",
      "chapter": 4
    },
    {
      "description": "Confrontation with Marcus Webb reveals the full conspiracy",
      "chapter": 5
    },
    {
      "description": "Sarah exposes the truth, saving the colony but at great personal cost",
      "chapter": 6
    }
  ],
  "chapters": [
    {
      "number": 1,
      "title": "Vanishing Point",
      "outline": "Sarah is called to investigate the first disappearance. She interviews witnesses and examines the last known location in the tunnels."
    },
    {
      "number": 2,
      "title": "Deeper Dark",
      "outline": "More disappearances occur. Sarah ventures into the deep tunnels and discovers strange equipment and readings."
    },
    {
      "number": 3,
      "title": "Hidden Symptoms",
      "outline": "Dr. Tanaka reveals medical anomalies. Sarah begins to suspect corporate involvement."
    },
    {
      "number": 4,
      "title": "Buried Secrets",
      "outline": "Sarah finds an abandoned lab with evidence of human experiments. The truth begins to emerge."
    },
    {
      "number": 5,
      "title": "Surface Tension",
      "outline": "Confrontation with Webb. Sarah learns the full scope of the conspiracy and the company's plans."
    },
    {
      "number": 6,
      "title": "Red Reckoning",
      "outline": "Sarah broadcasts the evidence colony-wide. Justice is served, but the cost is heavy."
    }
  ]
}"""

# Writer responses
WRITER_CHAPTER_CONTENT = """The call came through at 0342 hours, dragging Sarah Chen from a fitful sleep in her cramped hab-unit. Three months on Mars, and she still wasn't used to the silence—the oppressive, tomb-like quiet that pressed against the dome like the weight of a dead world.

"Detective Chen," the voice crackled through her comm. "We have a situation. Miner reported missing in Tunnel 7-Delta. No response to calls, suit tracker offline."

Sarah pulled on her clothes, the same utilitarian gray everyone wore in the colony. Another missing person. That made three this month. The official line was equipment malfunction—suits failing, people getting lost in the labyrinth below. But Sarah's instincts, honed over fifteen years on Earth, whispered something different.

The Security Chief, a burly man named Rodriguez, met her at the tunnel entrance. His face was grim behind his helmet visor.

"Name's Danny Park," Rodriguez said, pulling up a holographic file. "Twenty-three, been here six months. Good kid. Went down for the night shift, never came back up."

Sarah studied the 3D map of Tunnel 7-Delta, her eyes tracing the network of passages that extended deep into the Martian crust. "When was his last check-in?"

"Four hours ago. Routine status update. Then... nothing."

They descended into the tunnels, their helmet lights cutting through the darkness. The walls were rough-hewn rock, reinforced with carbon-fiber supports. Every sound echoed—the hiss of recycled air, the scrape of their boots, the distant rumble of machinery.

Sarah's mind was already working the problem. Three disappearances, all in the same sector. All during night shifts. All experienced miners who knew these tunnels like the backs of their hands.

"Rodriguez," she said, her voice low. "Show me exactly where the others went missing."

He hesitated. "Detective, the cases were closed—"

"Show me."

The holographic map flickered, marking two locations. Both within a kilometer of where they stood. Both in rarely-used side tunnels that dead-ended at old excavation sites.

Sarah's pulse quickened. Not random. Not accidents.

"We need to search those areas again," she said. "All of them. And I need access to all security footage, all sensor logs, everything."

Rodriguez's expression shifted from doubt to understanding. "You think someone's covering something up."

Sarah didn't answer. She was already moving deeper into the tunnel, following a hunch that whispered in the dark. On Earth, she'd learned to trust those whispers. On Mars, in this alien darkness, they might be the only thing keeping her alive.

The tunnel stretched ahead, deeper and darker, holding its secrets close."""

EDITOR_REFINED_CONTENT = """The call came through at 0342 hours, dragging Sarah Chen from a fitful sleep in her cramped hab-unit. Three months on Mars, and she still wasn't used to the silence—the oppressive, tomb-like quiet that pressed against the dome like the weight of a dead world.

"Detective Chen." The voice crackled through her comm, urgent and tinny. "We have a situation. Miner reported missing in Tunnel 7-Delta. No response to calls, suit tracker offline."

Sarah pulled on her clothes—the same utilitarian gray everyone wore in the colony—and tried to shake off the dream that still clung to her thoughts. Another missing person. That made three this month. The official line was equipment malfunction: suits failing, people getting lost in the labyrinth below. But Sarah's instincts, honed over fifteen years on Earth, whispered something different.

Security Chief Rodriguez met her at the tunnel entrance, his burly frame tense even in the lower gravity. His face was grim behind his helmet visor.

"Name's Danny Park," he said, pulling up a holographic file. "Twenty-three, been here six months. Good kid. Went down for the night shift, never came back up."

Sarah studied the 3D map of Tunnel 7-Delta, her eyes tracing the network of passages extending deep into the Martian crust. "When was his last check-in?"

"Four hours ago. Routine status update. Then... nothing."

They descended into the tunnels, their helmet lights carving tunnels of illumination through the absolute darkness. The walls were rough-hewn rock, reinforced with carbon-fiber supports that gleamed dully in their lights. Every sound echoed: the hiss of recycled air, the scrape of their boots, the distant rumble of machinery that never slept.

Sarah's mind was already working the problem. Three disappearances, all in the same sector. All during night shifts. All experienced miners who knew these tunnels intimately.

"Rodriguez," she said, keeping her voice low. "Show me exactly where the others went missing."

He hesitated. "Detective, the cases were closed—"

"Show me."

The holographic map flickered, marking two locations. Both within a kilometer of where they stood. Both in rarely-used side tunnels that dead-ended at old excavation sites.

Sarah's pulse quickened. Not random. Not accidents.

"We need to search those areas again," she said. "All of them. And I need access to all security footage, all sensor logs—everything."

Rodriguez's expression shifted from doubt to understanding. "You think someone's covering something up."

Sarah didn't answer. She was already moving deeper into the tunnel, following a hunch that whispered in the dark. On Earth, she'd learned to trust those whispers. On Mars, in this alien darkness, they might be the only thing keeping her alive.

The tunnel stretched ahead, deeper and darker, holding its secrets close. And Sarah Chen, haunted by her own past failures, was determined to uncover them."""

# Continuity checker responses
CONTINUITY_REVIEW_RESPONSE = """{
  "issues": [
    {
      "type": "timeline",
      "severity": "minor",
      "description": "Sarah has been on Mars for 'three months' but earlier mentioned she transferred 'recently' - clarify exact timeline",
      "chapter": 1,
      "suggestion": "Consider specifying exact transfer date in character background"
    },
    {
      "type": "character_consistency",
      "severity": "minor",
      "description": "Dr. Tanaka not yet introduced but should be present at colony",
      "chapter": 1,
      "suggestion": "Consider adding a brief mention or set up introduction for Chapter 3"
    }
  ],
  "strengths": [
    "Strong atmosphere and setting details are consistent",
    "Character voice for Sarah is well-established",
    "Mystery setup is compelling and well-paced",
    "Technical details about Mars colony are internally consistent"
  ],
  "overall_assessment": "The chapter is strong with only minor continuity issues. The world-building is solid and the mystery hook is effective. Address the timeline clarity and the pacing is excellent."
}"""

# Error scenario responses
INCOMPLETE_JSON_RESPONSE = """{
  "world_description": "A futuristic city",
  "world_rules": [
    "Rule 1"
  ],
  "characters": [
    {
      "name": "Test Character",
      "role": "protagonist"
"""  # Intentionally incomplete

INVALID_JSON_RESPONSE = """This is not valid JSON at all, just some text that an LLM might return if it gets confused or doesn't follow the format properly. Sometimes models will just chat instead of returning structured data."""

TIMEOUT_SIMULATION = ""  # Empty response to simulate timeout

# Title generation responses
TITLE_SUGGESTIONS_RESPONSE = """Based on the story's themes and plot, here are some title suggestions:

1. **Red Dust Reckoning** - Emphasizes the Mars setting and themes of justice
2. **The Colony's Shadow** - Focuses on the mystery and hidden secrets
3. **Beneath the Red Planet** - Highlights the underground setting and mystery
4. **Mars Declassified** - Corporate conspiracy and revelation themes
5. **Crimson Cover-Up** - Alludes to both Mars and the corporate conspiracy
6. **The Martian Protocol** - Science fiction feel with mystery elements
7. **Tunnel Vision** - Dual meaning: literal tunnels and focused investigation
8. **Gravity's Edge** - Mars setting with danger/suspense connotations
9. **The Red Silence** - Atmospheric, mysterious tone
10. **Colony Zero-Dark** - Military/thriller feel with sci-fi elements

My recommendation would be **"Red Dust Reckoning"** or **"The Colony's Shadow"** as they best capture both the setting and the central mystery."""

# Edit suggestions response
EDIT_SUGGESTIONS_RESPONSE = """Here are my suggestions for improving this passage:

**Strengths:**
- Excellent atmosphere and mood
- Strong opening hook
- Good character introduction
- Effective use of sensory details

**Suggested Improvements:**

1. **Pacing:** The middle section where they descend into the tunnels could be tightened. Consider combining some of the description with action.

2. **Character Development:** Add a brief internal thought from Sarah about why she transferred to Mars - this builds intrigue about her past.

3. **Dialogue:** Rodriguez's dialogue could be more distinctive. Give him a verbal quirk or pattern that makes him memorable.

4. **Show vs Tell:** Instead of stating "Sarah's mind was already working the problem," show her thought process through actions or observations.

5. **Sensory Details:** You have great visual details. Consider adding more about how the lower gravity feels, or the recycled air smells.

**Specific Line Edits:**

- "Three months on Mars" - Consider being more specific: "Ninety-three sols on Mars" (more authentic)
- "Sarah's instincts, honed over fifteen years on Earth" - Slightly awkward. Rephrase: "Sarah's fifteen years of detective work on Earth had taught her..."
- Last paragraph is very strong - keep as is.

Overall: This is solid work. With minor refinements, it will be excellent."""

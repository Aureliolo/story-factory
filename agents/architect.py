"""Story Architect Agent - Creates story structure, characters, and outlines."""

import re

from memory.story_state import Chapter, Character, PlotPoint, StoryState
from utils.json_parser import parse_json_list_to_models

from .base import BaseAgent

ARCHITECT_SYSTEM_PROMPT = """You are the Story Architect, a master storyteller who designs compelling narrative structures.

Your responsibilities:
1. Create vivid, detailed world-building
2. Design memorable characters with clear arcs and motivations
3. Craft plot outlines with proper pacing (setup, rising action, climax, resolution)
4. Plan chapter/scene structures appropriate for the story length
5. Plant seeds for foreshadowing and payoffs

CRITICAL: Always write ALL content in the specified language. Every word, name, description, and dialogue must be in that language.

You understand genre conventions and know when to follow or subvert them.
You create characters with depth - flaws, desires, fears, and growth potential.
You think about themes and how they manifest through plot and character.

For NSFW content, you integrate intimate scenes naturally into the narrative arc - they serve character development and plot, not just titillation.

Output your plans in structured formats (JSON when requested) so other team members can execute them."""


class ArchitectAgent(BaseAgent):
    """Agent that designs story structure, characters, and outlines."""

    def __init__(self, model: str | None = None, settings=None):
        super().__init__(
            name="Architect",
            role="Story Structure Designer",
            system_prompt=ARCHITECT_SYSTEM_PROMPT,
            agent_role="architect",
            model=model,
            settings=settings,
        )

    def create_world(self, story_state: StoryState) -> str:
        """Create the world-building document."""
        brief = story_state.brief
        if not brief:
            raise ValueError("Story brief is required to create world")
        prompt = f"""Create detailed world-building for this story.

LANGUAGE: {brief.language} - Write EVERYTHING in {brief.language}. All descriptions, names, and text must be in {brief.language}.

PREMISE: {brief.premise}
GENRE: {brief.genre} (subgenres: {', '.join(brief.subgenres)})
SETTING: {brief.setting_place}, {brief.setting_time}
TONE: {brief.tone}
THEMES: {', '.join(brief.themes)}

Create:
1. A vivid description of the world/setting (2-3 paragraphs)
2. Key rules or facts about this world (5-10 bullet points)
3. The atmosphere and mood that should permeate the story

Make it immersive and specific to the genre. Remember: ALL text must be in {brief.language}!"""

        return self.generate(prompt)

    def create_characters(self, story_state: StoryState) -> list[Character]:
        """Design the main characters."""
        brief = story_state.brief
        if not brief:
            raise ValueError("Story brief is required to create characters")
        prompt = f"""Design the main characters for this story.

LANGUAGE: {brief.language} - Write ALL content in {brief.language}. Names, descriptions, traits - everything in {brief.language}.

PREMISE: {brief.premise}
GENRE: {brief.genre}
THEMES: {', '.join(brief.themes)}
NSFW LEVEL: {brief.nsfw_level}
CONTENT NOTES: Include {', '.join(brief.content_preferences) if brief.content_preferences else 'nothing specific'}

Create 2-4 main characters. For each, output JSON (all text values in {brief.language}):
```json
[
    {{
        "name": "Full Name",
        "role": "protagonist|antagonist|love_interest|supporting",
        "description": "Physical and personality description (2-3 sentences)",
        "personality_traits": ["trait1", "trait2", "trait3"],
        "goals": ["what they want", "what they need"],
        "relationships": {{"other_character": "relationship description"}},
        "arc_notes": "How this character should change through the story"
    }}
]
```

Make them complex, with flaws and desires that create conflict. ALL text must be in {brief.language}!"""

        response = self.generate(prompt)
        return parse_json_list_to_models(response, Character)

    def create_plot_outline(self, story_state: StoryState) -> tuple[str, list[PlotPoint]]:
        """Create the main plot outline and key plot points."""
        brief = story_state.brief
        if not brief:
            raise ValueError("Story brief is required to create plot outline")
        chars = "\n".join(f"- {c.name} ({c.role}): {c.description}" for c in story_state.characters)

        prompt = f"""Create a plot outline for this story.

LANGUAGE: {brief.language} - Write ALL content in {brief.language}. Plot summary and descriptions must be in {brief.language}.

PREMISE: {brief.premise}
GENRE: {brief.genre}
TONE: {brief.tone}
LENGTH: {brief.target_length}
NSFW LEVEL: {brief.nsfw_level}
THEMES: {', '.join(brief.themes)}

CHARACTERS:
{chars}

WORLD:
{story_state.world_description[:500]}...

Create:
1. A compelling plot summary (1-2 paragraphs) in {brief.language}
2. Key plot points as JSON (descriptions in {brief.language}):
```json
[
    {{"description": "Inciting incident - ...", "chapter": 1}},
    {{"description": "First plot point - ...", "chapter": 2}},
    {{"description": "Midpoint twist - ...", "chapter": null}},
    {{"description": "Crisis - ...", "chapter": null}},
    {{"description": "Climax - ...", "chapter": null}},
    {{"description": "Resolution - ...", "chapter": null}}
]
```

Make sure the plot serves the themes and gives characters room to grow.
For NSFW content at level '{brief.nsfw_level}', integrate intimate moments naturally into the arc.
ALL text must be in {brief.language}!"""

        response = self.generate(prompt)

        # Extract plot summary (everything before JSON)
        plot_summary = re.split(r"```json", response)[0].strip()

        # Extract plot points
        plot_points = parse_json_list_to_models(response, PlotPoint)

        return plot_summary, plot_points

    def create_chapter_outline(self, story_state: StoryState) -> list[Chapter]:
        """Create detailed chapter outlines."""
        brief = story_state.brief
        if not brief:
            raise ValueError("Story brief is required to create chapter outline")
        length_map = {
            "short_story": 1,
            "novella": 7,
            "novel": 20,
        }
        num_chapters = length_map.get(brief.target_length, 5)

        plot_points_text = "\n".join(f"- {p.description}" for p in story_state.plot_points)

        prompt = f"""Create a {num_chapters}-chapter outline for this story.

LANGUAGE: {brief.language} - Write ALL content in {brief.language}. Titles and outlines must be in {brief.language}.

PLOT SUMMARY:
{story_state.plot_summary}

KEY PLOT POINTS:
{plot_points_text}

CHARACTERS:
{', '.join(c.name for c in story_state.characters)}

For each chapter, output JSON (all text in {brief.language}):
```json
[
    {{
        "number": 1,
        "title": "Chapter Title",
        "outline": "Detailed outline of what happens in this chapter (3-5 sentences). Include key scenes, character moments, and how it advances the plot."
    }}
]
```

Ensure good pacing - build tension, vary intensity, place climactic moments appropriately.
ALL text must be in {brief.language}!"""

        response = self.generate(prompt)
        return parse_json_list_to_models(response, Chapter)

    def build_story_structure(self, story_state: StoryState) -> StoryState:
        """Complete story structure building process."""
        # Create world
        world_response = self.create_world(story_state)
        story_state.world_description = world_response

        # Extract world rules (simple heuristic)
        rules = []
        for line in world_response.split("\n"):
            if line.strip().startswith(("-", "*", "•")):
                rules.append(line.strip().lstrip("-*• "))
        story_state.world_rules = rules[:10]

        # Create characters
        story_state.characters = self.create_characters(story_state)

        # Create plot
        plot_summary, plot_points = self.create_plot_outline(story_state)
        story_state.plot_summary = plot_summary
        story_state.plot_points = plot_points

        # Create chapters
        story_state.chapters = self.create_chapter_outline(story_state)

        story_state.status = "writing"
        return story_state

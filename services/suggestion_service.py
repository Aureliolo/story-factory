"""Suggestion service - provides AI-powered writing prompts and suggestions."""

import logging
import random

from agents.base import BaseAgent
from memory.story_state import StoryState
from settings import Settings
from utils.json_parser import extract_json

logger = logging.getLogger(__name__)


class SuggestionService:
    """Service for generating context-aware writing prompts and suggestions.

    This service helps writers overcome writer's block by generating
    relevant suggestions based on the current story state, characters,
    and plot progression.
    """

    def __init__(self, settings: Settings):
        """Initialize suggestion service.

        Args:
            settings: Application settings.
        """
        self.settings = settings
        self._agent: BaseAgent | None = None

    def _get_agent(self) -> BaseAgent:
        """Get or create the suggestion agent.

        Returns:
            BaseAgent configured for generating suggestions.
        """
        if self._agent is None:
            system_prompt = """You are a creative writing assistant specialized in helping writers overcome writer's block.

Your role is to generate contextually relevant writing prompts and suggestions based on the current story state. Your suggestions should:
- Be specific to the characters, plot, and world
- Offer creative possibilities without forcing a direction
- Help writers explore different narrative paths
- Maintain consistency with established story elements
- Be concise and actionable

Generate suggestions in the following categories:
1. Plot prompts - "What if" scenarios and complications
2. Character prompts - Character reactions and development
3. Scene prompts - Atmospheric and tension-building ideas
4. Transition prompts - Ways to move between scenes/chapters

Always return your response as valid JSON."""

            self._agent = BaseAgent(
                name="Suggestion Assistant",
                role="Creative Writing Assistant",
                system_prompt=system_prompt,
                agent_role="suggestion",
                settings=self.settings,
            )
            logger.debug("Initialized suggestion agent")

        return self._agent

    def generate_suggestions(
        self, state: StoryState, category: str | None = None
    ) -> dict[str, list[str]]:
        """Generate context-aware writing suggestions.

        Args:
            state: Current story state with context.
            category: Optional category to focus on (plot, character, scene, transition).
                     If None, generates suggestions for all categories.

        Returns:
            Dictionary with category keys and lists of suggestion strings.
            Example: {"plot": ["What if...", ...], "character": [...], ...}
        """
        logger.info(f"Generating suggestions for story '{state.project_name}'")

        # Build context from story state
        context = self._build_context(state)

        # Generate prompt
        if category:
            prompt = f"Generate 3-5 {category} prompts based on this story context:\n\n{context}"
            categories = [category]
        else:
            prompt = f"""Generate creative writing suggestions for this story across all categories.

Story Context:
{context}

Return a JSON object with these keys:
- "plot": Array of 3-4 "what if" scenarios or plot complications
- "character": Array of 3-4 character action or development suggestions
- "scene": Array of 3-4 scene atmosphere or tension ideas
- "transition": Array of 2-3 scene/chapter transition suggestions

Each suggestion should be 1-2 sentences, specific to this story."""
            categories = ["plot", "character", "scene", "transition"]

        try:
            # Call LLM
            agent = self._get_agent()
            response = agent.generate(prompt)

            logger.debug(f"Raw suggestion response: {response[:200]}...")

            # Parse JSON response
            suggestions_data = extract_json(response)

            if not suggestions_data:
                logger.warning("Failed to extract JSON from suggestions response")
                return self._fallback_suggestions(state, categories)

            # Validate structure
            if category:
                # Single category response - wrap in dict
                if isinstance(suggestions_data, list):
                    return {category: suggestions_data}
                elif isinstance(suggestions_data, dict) and category in suggestions_data:
                    return {category: suggestions_data[category]}
                else:
                    logger.warning(f"Unexpected response structure for category {category}")
                    return self._fallback_suggestions(state, categories)
            else:
                # Multi-category response
                if not isinstance(suggestions_data, dict):
                    logger.warning("Expected dict for multi-category suggestions")
                    return self._fallback_suggestions(state, categories)

                # Ensure all expected categories are present
                result = {}
                for cat in categories:
                    if cat in suggestions_data and isinstance(suggestions_data[cat], list):
                        result[cat] = suggestions_data[cat]
                    else:
                        logger.warning(f"Missing or invalid category '{cat}' in response")
                        result[cat] = self._fallback_suggestions(state, [cat])[cat]

                return result

        except Exception:
            logger.exception("Failed to generate suggestions")
            return self._fallback_suggestions(state, categories)

    def _build_context(self, state: StoryState) -> str:
        """Build context string from story state.

        Args:
            state: Story state to extract context from.

        Returns:
            Formatted context string for the LLM.
        """
        logger.debug(f"Building context for story state (id={state.id})")
        context_parts = []

        # Story brief
        if state.brief:
            logger.debug("Including story brief in context")
            context_parts.append(f"**Premise:** {state.brief.premise}")
            context_parts.append(f"**Genre:** {state.brief.genre} | **Tone:** {state.brief.tone}")
            context_parts.append(
                f"**Setting:** {state.brief.setting_place}, {state.brief.setting_time}"
            )

        # Characters
        if state.characters:
            logger.debug(f"Including {len(state.characters)} characters in context")
            char_info = []
            for char in state.characters[:5]:  # Limit to main characters
                char_desc = f"{char.name} ({char.role})"
                if char.goals:
                    # Ensure all goals are strings
                    char_desc += f" - Goals: {', '.join(str(goal) for goal in char.goals[:2])}"
                char_info.append(char_desc)
            context_parts.append(f"**Characters:** {'; '.join(char_info)}")

        # Current chapter/progress
        if state.current_chapter and state.chapters:
            logger.debug(f"Including chapter {state.current_chapter} in context")
            current_ch = next(
                (ch for ch in state.chapters if ch.number == state.current_chapter), None
            )
            if current_ch:
                context_parts.append(
                    f"**Current Chapter:** {current_ch.number}. {current_ch.title}"
                )
                context_parts.append(f"**Chapter Outline:** {current_ch.outline}")

                # Recent content
                if current_ch.content:
                    recent = current_ch.content[-500:]  # Last 500 chars
                    context_parts.append(f"**Recent Content:** ...{recent}")

        # Plot points
        if state.plot_points:
            upcoming = [p for p in state.plot_points if not p.completed]
            if upcoming:
                context_parts.append(f"**Upcoming Plot Point:** {upcoming[0].description}")

        # Recent facts
        if state.established_facts:
            logger.debug(f"Including {len(state.established_facts[-5:])} recent facts in context")
            recent_facts = state.established_facts[-5:]
            context_parts.append(f"**Recent Facts:** {'; '.join(recent_facts)}")

        context = "\n".join(context_parts)
        logger.debug(f"Built context with {len(context_parts)} parts ({len(context)} chars)")
        return context

    def generate_project_names(self, state: StoryState, count: int = 10) -> list[str]:
        """Generate creative project name suggestions based on story content.

        Uses the LLM to generate unique, creative title suggestions based on
        the story's premise, genre, characters, and plot points.

        Args:
            state: Story state with context for generating names.
            count: Number of name suggestions to generate (default 10).

        Returns:
            List of project name suggestions.
        """
        logger.info(f"Generating {count} project name suggestions for story '{state.project_name}'")

        # Build context from story state
        context_parts = []

        # Story brief
        if state.brief:
            context_parts.append(f"Premise: {state.brief.premise}")
            context_parts.append(f"Genre: {state.brief.genre}")
            context_parts.append(f"Tone: {state.brief.tone}")
            if state.brief.setting_place:
                context_parts.append(f"Setting: {state.brief.setting_place}")

        # Characters
        if state.characters:
            char_names = [char.name for char in state.characters[:5]]
            context_parts.append(f"Main Characters: {', '.join(char_names)}")

        # Plot points
        if state.plot_points:
            key_points = [p.description for p in state.plot_points[:3]]
            context_parts.append(f"Key Plot Points: {'; '.join(key_points)}")

        context = "\n".join(context_parts)

        prompt = f"""Based on this story context, generate {count} unique, creative title suggestions.

Story Context:
{context}

Requirements:
- Each title should be evocative and memorable
- Mix different styles: some short and punchy, some poetic, some mysterious
- Avoid generic titles like "The Journey" or "A New Beginning"
- Consider the genre and tone when crafting titles
- Each title should feel like it could be a published book title

Return ONLY a JSON array of strings, with {count} title suggestions.
Example format: ["Title One", "Title Two", "Title Three"]

Do not include any explanation, just the JSON array."""

        try:
            agent = self._get_agent()
            response = agent.generate(prompt)

            logger.debug(f"Raw name suggestions response: {response[:200]}...")

            # Parse JSON response
            suggestions = extract_json(response)

            if not suggestions:
                logger.warning("Failed to extract JSON from name suggestions response")
                return self._fallback_project_names(state, count)

            # Validate it's a list of strings
            if isinstance(suggestions, list):
                names = [str(s).strip() for s in suggestions if s and str(s).strip()]
                if names:
                    logger.info(f"Generated {len(names)} project name suggestions")
                    return names[:count]

            logger.warning("Unexpected response structure for name suggestions")
            return self._fallback_project_names(state, count)

        except Exception:
            logger.exception("Failed to generate project name suggestions")
            return self._fallback_project_names(state, count)

    def _fallback_project_names(self, state: StoryState, count: int) -> list[str]:
        """Generate fallback project name suggestions when LLM fails.

        Args:
            state: Story state for context.
            count: Number of suggestions to generate.

        Returns:
            List of simple fallback name suggestions.
        """
        logger.info("Using fallback project name suggestions")

        # Get some context for personalization
        genre = state.brief.genre if state.brief else "Story"
        char_name = state.characters[0].name if state.characters else "Hero"

        base_suggestions = [
            f"The {genre} Chronicles",
            f"{char_name}'s Journey",
            f"Echoes of {genre}",
            f"The Last {genre}",
            f"Beyond the {genre}",
            "Whispers of Fate",
            "The Forgotten Path",
            "Shadows and Light",
            "The Turning Point",
            "Untold Stories",
            "The Rising Dawn",
            "Threads of Destiny",
        ]

        return base_suggestions[:count]

    def _fallback_suggestions(
        self, state: StoryState, categories: list[str]
    ) -> dict[str, list[str]]:
        """Generate simple fallback suggestions when LLM fails.

        Args:
            state: Story state for context.
            categories: Categories to generate suggestions for.

        Returns:
            Dictionary with basic suggestions for each category.
        """
        logger.info(f"Using fallback suggestions for categories: {categories}")

        result = {}

        # Get character names for templates
        char_names = (
            [c.name for c in state.characters[:3]] if state.characters else ["the protagonist"]
        )

        for category in categories:
            if category == "plot":
                result["plot"] = [
                    f"What if {random.choice(char_names)} discovered a hidden secret?",
                    f"A sudden complication forces {random.choice(char_names)} to make a difficult choice.",
                    "An unexpected ally or enemy appears at a critical moment.",
                ]
            elif category == "character":
                result["character"] = [
                    f"How would {random.choice(char_names)} react under extreme pressure?",
                    f"Show {random.choice(char_names)}'s vulnerability or inner conflict.",
                    "Reveal a character's backstory through action rather than exposition.",
                ]
            elif category == "scene":
                result["scene"] = [
                    "Build tension through environmental details and atmosphere.",
                    "Use dialogue to reveal subtext and unspoken conflicts.",
                    "Contrast the mood with an unexpected sensory detail.",
                ]
            elif category == "transition":
                result["transition"] = [
                    "Jump forward in time to the next critical moment.",
                    "End the scene on a cliffhanger or unanswered question.",
                ]

        return result

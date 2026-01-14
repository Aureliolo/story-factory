"""Interviewer Agent - Gathers story requirements from the user."""

from memory.story_state import StoryBrief
from utils.json_parser import parse_json_to_model

from .base import BaseAgent

INTERVIEWER_SYSTEM_PROMPT = """You are the Interviewer, the first member of a creative writing team. Your job is to gather all the information needed to write a compelling story.

You ask thoughtful, specific questions to understand:
- The core premise and concept
- Genre and tone preferences
- Setting (time and place)
- Key characters the user has in mind
- Themes to explore
- Content preferences and boundaries (including NSFW comfort level)
- Story length preference

Be conversational and encouraging. Ask follow-up questions when answers are vague. Help users who aren't sure what they want by offering creative suggestions.

When you have enough information, output a structured story brief in JSON format.

IMPORTANT: For NSFW content levels, use these definitions:
- "none": Keep it clean, no sexual or graphic content
- "mild": Fade to black, implied intimacy, mild violence
- "moderate": Some explicit scenes, detailed violence, mature themes
- "explicit": Fully explicit content, no restrictions

Always be respectful and non-judgmental about content preferences."""


class InterviewerAgent(BaseAgent):
    """Agent that interviews the user to gather story requirements."""

    def __init__(self, model: str = None, settings=None):
        super().__init__(
            name="Interviewer",
            role="Story Requirements Gatherer",
            system_prompt=INTERVIEWER_SYSTEM_PROMPT,
            agent_role="interviewer",
            model=model,
            settings=settings,
        )
        self.conversation_history = []

    def get_initial_questions(self) -> str:
        """Generate the initial interview questions."""
        prompt = """Start the interview by warmly greeting the user and asking about their story idea.
Ask 3-4 initial questions covering:
1. What's the basic premise or concept they have in mind?
2. What genre are they interested in?
3. Any specific setting or time period?
4. How long should the story be (short story, novella, or novel)?

Keep it friendly and conversational."""

        return self.generate(prompt)

    def process_response(self, user_response: str, context: str = "") -> str:
        """Process a user response and generate follow-up questions or the final brief."""
        self.conversation_history.append({"role": "user", "content": user_response})

        history_text = "\n".join(
            f"{'User' if h['role'] == 'user' else 'You'}: {h['content']}"
            for h in self.conversation_history
        )

        prompt = f"""Based on the conversation so far:

{history_text}

Either:
1. Ask follow-up questions if you need more information about characters, themes, tone, NSFW comfort level, or content preferences
2. If you have enough information, output a complete story brief in this exact JSON format:

```json
{{
    "premise": "...",
    "genre": "...",
    "subgenres": ["...", "..."],
    "tone": "...",
    "themes": ["...", "..."],
    "setting_time": "...",
    "setting_place": "...",
    "target_length": "short_story|novella|novel",
    "nsfw_level": "none|mild|moderate|explicit",
    "content_preferences": ["things to include..."],
    "content_avoid": ["things to avoid..."],
    "additional_notes": "..."
}}
```

Make sure to explicitly ask about NSFW comfort level if not yet discussed."""

        response = self.generate(prompt, context)
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def extract_brief(self, response: str) -> StoryBrief | None:
        """Try to extract a StoryBrief from the response if it contains JSON."""
        # Fallback pattern for JSON without code block
        fallback = r'\{[^{}]*"premise"[^{}]*\}'
        return parse_json_to_model(response, StoryBrief, fallback_pattern=fallback)

    def finalize_brief(self, conversation_summary: str) -> StoryBrief:
        """Force generation of a final brief from the conversation."""
        prompt = f"""Based on this conversation, create a complete story brief:

{conversation_summary}

Output ONLY a JSON object in this exact format (no other text):
```json
{{
    "premise": "...",
    "genre": "...",
    "subgenres": ["...", "..."],
    "tone": "...",
    "themes": ["...", "..."],
    "setting_time": "...",
    "setting_place": "...",
    "target_length": "short_story|novella|novel",
    "nsfw_level": "none|mild|moderate|explicit",
    "content_preferences": ["..."],
    "content_avoid": ["..."],
    "additional_notes": "..."
}}
```"""

        response = self.generate(prompt, temperature=0.3)
        brief = self.extract_brief(response)

        if not brief:
            # Create a default brief if parsing fails
            brief = StoryBrief(
                premise="Story based on user conversation",
                genre="Fiction",
                tone="Engaging",
                setting_time="Contemporary",
                setting_place="Unspecified",
                target_length="short_story",
                nsfw_level="moderate",
            )

        return brief

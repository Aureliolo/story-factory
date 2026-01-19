"""Interviewer Agent - Gathers story requirements from the user."""

import logging

from memory.story_state import StoryBrief
from settings import Settings
from utils.json_parser import parse_json_to_model
from utils.validation import validate_not_empty

from .base import BaseAgent

logger = logging.getLogger(__name__)

INTERVIEWER_SYSTEM_PROMPT = """You are the Interviewer, the first member of a creative writing team. Your job is to gather all the information needed to write a compelling story.

You ask thoughtful, specific questions to understand:
- The core premise and concept
- Genre and tone preferences
- Setting (time and place)
- Key characters the user has in mind
- Themes to explore
- Content preferences and maturity level
- Story length preference

CRITICAL - INFER FROM CONTEXT, DON'T ASK OBVIOUS QUESTIONS:
1. LANGUAGE: If the user writes their message in English, assume English. If in German, assume German. etc.
   Only ask about language if it's unclear or you want to confirm a non-obvious choice.
2. CONTENT RATING: Infer from keywords in the user's message:
   - Words like "smut", "nsfw", "explicit", "erotic", "adult content" → "adult" rating
   - Words like "dark", "violence", "mature themes" → "mature" rating
   - Words like "teen", "young adult", "YA" → "teen" rating
   - Family-friendly, kids, children → "general" rating
   Don't ask about content rating if the user has already made it obvious!
3. GENRE: If the user says "fantasy" or "sci-fi" or "romance", don't ask again.

Be conversational and encouraging. Ask follow-up questions only when answers are genuinely vague or missing. Help users who aren't sure what they want by offering creative suggestions.

When you have enough information, output a structured story brief in JSON format.

Content rating definitions (for your reference, not to ask about):
- "general": Family-friendly content suitable for all audiences
- "teen": Some mild themes, suitable for teenagers
- "mature": Adult themes, some explicit content
- "adult": Fully explicit content, no restrictions

Always be respectful and non-judgmental about content preferences."""


class InterviewerAgent(BaseAgent):
    """Agent that interviews the user to gather story requirements."""

    def __init__(self, model: str | None = None, settings: Settings | None = None) -> None:
        """Initialize the Interviewer agent.

        Args:
            model: Override model to use. If None, uses settings-based model for interviewer.
            settings: Application settings. If None, loads default settings.
        """
        super().__init__(
            name="Interviewer",
            role="Story Requirements Gatherer",
            system_prompt=INTERVIEWER_SYSTEM_PROMPT,
            agent_role="interviewer",
            model=model,
            settings=settings,
        )
        self.conversation_history: list[dict[str, str]] = []

    def get_initial_questions(self) -> str:
        """Generate the initial interview questions."""
        logger.info("Generating initial interview questions")
        prompt = """Start the interview by warmly greeting the user and asking about their story idea.
Ask 3-4 initial questions covering:
1. What's the basic premise or concept they have in mind?
2. What genre are they interested in?
3. Any specific setting or time period?
4. How long should the story be (short story, novella, or novel)?

Keep it friendly and conversational. Do NOT ask about language upfront - you'll infer it from how they write."""

        response = self.generate(prompt)
        logger.debug(f"Generated initial questions ({len(response)} chars)")
        return response

    def process_response(self, user_response: str, context: str = "") -> str:
        """Process a user response and generate follow-up questions or the final brief.

        Args:
            user_response: The user's message.
            context: Optional inference context (detected language, content rating, etc.)
        """
        validate_not_empty(user_response, "user_response")
        logger.info(
            f"Processing user response ({len(user_response)} chars, "
            f"history: {len(self.conversation_history)} messages)"
        )
        self.conversation_history.append({"role": "user", "content": user_response})

        history_text = "\n".join(
            f"{'User' if h['role'] == 'user' else 'You'}: {h['content']}"
            for h in self.conversation_history
        )

        # Include inference context at the start of the prompt if provided
        context_section = (
            f"ALREADY DETERMINED (do NOT ask about these):\n{context}\n" if context else ""
        )

        prompt = f"""{context_section}Based on the conversation so far:

{history_text}

YOUR TASK:
1. If this is the user's FIRST detailed message about their story idea, ask 2-3 thoughtful follow-up questions about:
   - Characters they envision (main protagonist, love interests, antagonists)
   - Specific themes or elements they want emphasized
   - Any specific scenes, plot points, or moments they're excited about
   - Things they definitely want to AVOID

2. If you've already asked follow-up questions and the user has answered them, summarize what you understand and ASK FOR CONFIRMATION:
   "Here's what I understand about your story: [brief summary]. Is there anything you'd like to add, change, or clarify before I create the story brief?"

3. ONLY when the user explicitly confirms they're done (says something like "yes", "that's all", "looks good", "I'm ready", "nothing else", etc.), output the JSON brief:

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
    "language": "English|German|Spanish|French|etc.",
    "content_rating": "general|teen|mature|adult",
    "content_preferences": ["things to include..."],
    "content_avoid": ["things to avoid..."],
    "additional_notes": "..."
}}
```

IMPORTANT RULES:
- NEVER output the JSON brief immediately after the user's first message!
- ALWAYS ask at least one round of follow-up questions first
- ALWAYS ask for confirmation before generating the brief
- Use the ALREADY DETERMINED values above if provided - do NOT ask about them!
- Infer language from the language the user is writing in
- Infer content rating from keywords: "smut"/"nsfw"/"explicit" = adult
- Be conversational and enthusiastic about their ideas!"""

        response = self.generate(prompt)
        self.conversation_history.append({"role": "assistant", "content": response})
        logger.debug(f"Generated response ({len(response)} chars)")
        return response

    def extract_brief(self, response: str) -> StoryBrief | None:
        """Try to extract a StoryBrief from the response if it contains JSON."""
        validate_not_empty(response, "response")
        logger.debug("Attempting to extract story brief from response")
        # Fallback pattern for JSON without code block
        fallback = r'\{[^{}]*"premise"[^{}]*\}'
        # Use strict=False since brief extraction during interview is optional -
        # the finalize_brief method has fallback logic to create a default brief
        brief = parse_json_to_model(response, StoryBrief, fallback_pattern=fallback, strict=False)
        if brief:
            logger.info(f"Extracted story brief: genre={brief.genre}, length={brief.target_length}")
        else:
            logger.debug("No story brief found in response")
        return brief

    def finalize_brief(self, conversation_summary: str) -> StoryBrief:
        """Force generation of a final brief from the conversation."""
        validate_not_empty(conversation_summary, "conversation_summary")
        logger.info("Finalizing story brief from conversation")
        prompt = f"""Based on this conversation, create a complete story brief.

{conversation_summary}

Fill in ALL fields with appropriate values based on the conversation.
If something wasn't discussed, make a reasonable choice that fits the story."""

        try:
            brief = self.generate_structured(
                prompt,
                StoryBrief,
                temperature=self.settings.temp_brief_extraction,
            )
            logger.info(f"Finalized story brief: {brief.genre} / {brief.target_length}")
            return brief
        except Exception as e:
            # Create a default brief if generation fails
            logger.warning(f"Failed to generate story brief ({e}), using default values")
            return StoryBrief(
                premise="Story based on user conversation",
                genre="Fiction",
                tone="Engaging",
                setting_time="Contemporary",
                setting_place="Unspecified",
                target_length="short_story",
                language="English",
                content_rating="mature",
            )

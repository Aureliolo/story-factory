"""Editor Agent - Refines and polishes prose."""

import logging
from typing import TYPE_CHECKING

from memory.story_state import StoryState
from utils.validation import validate_not_empty, validate_not_none, validate_type

from .base import BaseAgent

if TYPE_CHECKING:
    from settings import Settings

logger = logging.getLogger(__name__)

EDITOR_SYSTEM_PROMPT = """You are the Editor, a meticulous craftsman who polishes prose to perfection.

CRITICAL: Keep ALL content in the original language. Do not translate or change the language.

Your responsibilities:
1. Improve clarity and flow
2. Strengthen weak sentences
3. Fix awkward phrasing
4. Ensure "show don't tell" is followed
5. Vary sentence structure and length
6. Enhance sensory details
7. Tighten dialogue - make it punchy and character-distinct
8. Improve pacing - know when to speed up or slow down
9. Strengthen scene transitions
10. Ensure consistent voice and tone

You do NOT:
- Change the plot or major story elements
- Remove content (including mature content) - only improve how it's written
- Add new scenes or characters
- Change the fundamental meaning
- Change the language of the text

You preserve the writer's voice while making it shine.
Output the improved version of the text, not a critique."""


class EditorAgent(BaseAgent):
    """Agent that refines and polishes prose."""

    def __init__(self, model: str | None = None, settings: "Settings | None" = None) -> None:
        super().__init__(
            name="Editor",
            role="Prose Polisher",
            system_prompt=EDITOR_SYSTEM_PROMPT,
            agent_role="editor",
            model=model,
            settings=settings,
        )

    def edit_chapter(self, story_state: StoryState, chapter_content: str) -> str:
        """Edit and polish a chapter."""
        validate_not_none(story_state, "story_state")
        validate_type(story_state, "story_state", StoryState)
        validate_not_empty(chapter_content, "chapter_content")

        logger.info(f"Editing chapter ({len(chapter_content)} chars)")
        brief = story_state.brief
        if not brief:
            raise ValueError("Story brief is required to edit a chapter")

        prompt = f"""Edit and polish this chapter.

LANGUAGE: {brief.language} - Keep ALL text in {brief.language}. Do not translate.

---
{chapter_content}
---

GENRE: {brief.genre}
TONE: {brief.tone}

Improve:
- Prose quality and flow
- Dialogue naturalness
- Sensory details
- Pacing
- Sentence variety
- Show don't tell

Preserve:
- All plot points and events
- Character voices
- The overall story content
- Mature content (improve writing quality, don't remove)
- The language ({brief.language})

Output ONLY the edited chapter text in {brief.language} - no commentary or notes."""

        edited = self.generate(prompt)
        logger.info(f"Chapter edited ({len(edited)} chars, was {len(chapter_content)} chars)")
        return edited

    def edit_passage(
        self, passage: str, focus: str | None = None, language: str = "English"
    ) -> str:
        """Edit a specific passage with optional focus area."""
        validate_not_empty(passage, "passage")
        validate_not_empty(language, "language")

        logger.info(
            f"Editing passage ({len(passage)} chars)" + (f" focus: {focus}" if focus else "")
        )
        focus_instruction = f"\n\nFOCUS ESPECIALLY ON: {focus}" if focus else ""

        prompt = f"""Edit and polish this passage.

LANGUAGE: {language} - Keep ALL text in {language}. Do not translate.

---
{passage}
---
{focus_instruction}

Output ONLY the edited text in {language} - no commentary."""

        edited = self.generate(prompt)
        logger.debug(f"Passage edited ({len(edited)} chars)")
        return edited

    def get_edit_suggestions(self, text: str) -> str:
        """Get editing suggestions without making changes."""
        validate_not_empty(text, "text")

        logger.info(f"Getting edit suggestions for text ({len(text)} chars)")
        prompt = f"""Review this text and provide specific editing suggestions:

---
{text[: self.settings.full_text_preview_chars]}
---

Identify:
1. Awkward phrasing that needs smoothing
2. Places where "show don't tell" could improve
3. Dialogue that sounds unnatural
4. Pacing issues
5. Missing sensory details
6. Repetitive sentence structures

Be specific - quote the problematic text and suggest improvements."""

        suggestions = self.generate(prompt, temperature=self.settings.temp_edit_suggestions)
        logger.debug(f"Generated edit suggestions ({len(suggestions)} chars)")
        return suggestions

    def ensure_consistency(
        self,
        new_content: str,
        previous_content: str,
        story_state: StoryState,
    ) -> str:
        """Edit new content to ensure consistency with previous content."""
        validate_not_empty(new_content, "new_content")
        validate_not_empty(previous_content, "previous_content")
        validate_not_none(story_state, "story_state")
        validate_type(story_state, "story_state", StoryState)

        logger.info(
            f"Checking consistency: {len(new_content)} chars new, "
            f"{len(previous_content)} chars previous"
        )
        brief = story_state.brief
        if not brief:
            raise ValueError("Story brief is required to ensure consistency")
        prompt = f"""Review this new content for consistency with what came before.

LANGUAGE: {brief.language} - Keep ALL text in {brief.language}. Do not translate.

PREVIOUS CONTENT (ending):
...{previous_content[-1000:]}

NEW CONTENT:
{new_content}

ESTABLISHED FACTS:
{chr(10).join(story_state.established_facts[-30:])}

Check for and fix:
- Tone/voice shifts
- Tense inconsistencies
- Character voice changes
- Contradictions with established facts
- Language consistency (must be in {brief.language})

Output the corrected version of the NEW CONTENT only, in {brief.language}."""

        corrected = self.generate(prompt)
        logger.debug(f"Consistency check complete ({len(corrected)} chars)")
        return corrected

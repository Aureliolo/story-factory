"""Editor Agent - Refines and polishes prose."""

from .base import BaseAgent
from memory.story_state import StoryState

EDITOR_SYSTEM_PROMPT = """You are the Editor, a meticulous craftsman who polishes prose to perfection.

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
- Remove content (including NSFW content) - only improve how it's written
- Add new scenes or characters
- Change the fundamental meaning

You preserve the writer's voice while making it shine.
Output the improved version of the text, not a critique."""


class EditorAgent(BaseAgent):
    """Agent that refines and polishes prose."""

    def __init__(self, model: str = None, settings=None):
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
        brief = story_state.brief

        prompt = f"""Edit and polish this chapter:

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
- NSFW content (improve writing quality, don't remove)

Output ONLY the edited chapter text - no commentary or notes."""

        return self.generate(prompt)

    def edit_passage(self, passage: str, focus: str = None) -> str:
        """Edit a specific passage with optional focus area."""
        focus_instruction = f"\n\nFOCUS ESPECIALLY ON: {focus}" if focus else ""

        prompt = f"""Edit and polish this passage:

---
{passage}
---
{focus_instruction}

Output ONLY the edited text - no commentary."""

        return self.generate(prompt)

    def get_edit_suggestions(self, text: str) -> str:
        """Get editing suggestions without making changes."""
        prompt = f"""Review this text and provide specific editing suggestions:

---
{text[:3000]}
---

Identify:
1. Awkward phrasing that needs smoothing
2. Places where "show don't tell" could improve
3. Dialogue that sounds unnatural
4. Pacing issues
5. Missing sensory details
6. Repetitive sentence structures

Be specific - quote the problematic text and suggest improvements."""

        return self.generate(prompt, temperature=0.5)

    def ensure_consistency(
        self,
        new_content: str,
        previous_content: str,
        story_state: StoryState,
    ) -> str:
        """Edit new content to ensure consistency with previous content."""
        prompt = f"""Review this new content for consistency with what came before:

PREVIOUS CONTENT (ending):
...{previous_content[-1000:]}

NEW CONTENT:
{new_content}

ESTABLISHED FACTS:
{chr(10).join(story_state.established_facts[-10:])}

Check for and fix:
- Tone/voice shifts
- Tense inconsistencies
- Character voice changes
- Contradictions with established facts

Output the corrected version of the NEW CONTENT only."""

        return self.generate(prompt)

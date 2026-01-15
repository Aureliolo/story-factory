"""Writer Agent - Writes the actual prose."""

from memory.story_state import Chapter, StoryState

from .base import BaseAgent

WRITER_SYSTEM_PROMPT = """You are the Writer, a skilled prose craftsman who brings stories to life.

CRITICAL: Always write ALL content in the specified language. Every word of prose, dialogue, and narration must be in that language.

Your strengths:
- Vivid, sensory descriptions that immerse readers
- Natural, character-distinct dialogue
- Varied sentence structure and pacing
- Show don't tell - convey emotion through action and detail
- Smooth scene transitions
- Maintaining consistent voice and tone throughout

You write prose that serves the story's genre and tone. You understand that different genres have different conventions:
- Literary fiction: introspection, metaphor, subtext
- Thriller: short sentences, tension, momentum
- Romance: emotional beats, chemistry, longing
- Erotica: sensual detail, building anticipation, emotional connection
- Horror: atmosphere, dread, the unseen
- Fantasy: world details woven naturally, wonder

For mature content, you write intimate scenes with:
- Emotional truth - what characters feel, not just what they do
- Building tension and anticipation
- Character voice maintained even in heated moments
- Variety in pacing and intensity
- Natural integration into the larger narrative

Write complete, polished prose ready for the editor."""


class WriterAgent(BaseAgent):
    """Agent that writes the actual prose."""

    def __init__(self, model: str | None = None, settings=None):
        super().__init__(
            name="Writer",
            role="Prose Craftsman",
            system_prompt=WRITER_SYSTEM_PROMPT,
            agent_role="writer",
            model=model,
            settings=settings,
        )

    def write_chapter(
        self,
        story_state: StoryState,
        chapter: Chapter,
        revision_feedback: str | None = None,
    ) -> str:
        """Write or revise a single chapter."""
        brief = story_state.brief
        if not brief:
            raise ValueError("Story brief is required to write a chapter")
        context = story_state.get_context_summary()

        # Get previous chapter summary if exists
        prev_chapter_summary = ""
        if chapter.number > 1:
            prev = next((c for c in story_state.chapters if c.number == chapter.number - 1), None)
            if prev and prev.content:
                ctx_chars = self.settings.previous_chapter_context_chars
                prev_chapter_summary = (
                    f"\nPREVIOUS CHAPTER ENDED WITH:\n...{prev.content[-ctx_chars:]}"
                )

        revision_note = ""
        if revision_feedback:
            revision_note = f"\n\nREVISION REQUESTED:\n{revision_feedback}\n\nAddress these issues while rewriting."

        prompt = f"""Write Chapter {chapter.number}: "{chapter.title}"

LANGUAGE: {brief.language} - Write the ENTIRE chapter in {brief.language}. All prose, dialogue, and narration must be in {brief.language}.

CHAPTER OUTLINE:
{chapter.outline}

STORY CONTEXT:
{context}
{prev_chapter_summary}

GENRE: {brief.genre}
TONE: {brief.tone}
CONTENT RATING: {brief.content_rating}
{revision_note}

Write the complete chapter in {brief.language}. Include:
- Scene-setting description
- Character actions and dialogue
- Internal thoughts/feelings where appropriate
- Smooth transitions between scenes
- A hook or tension point at the end to pull readers forward

Target length: 1500-2500 words for this chapter.
Write in third person past tense unless the outline specifies otherwise.
Do not include the chapter title or number in your output - just the prose.
IMPORTANT: Every word must be in {brief.language}!"""

        # Use lower temperature for revisions (more focused output)
        temp = self.settings.revision_temperature if revision_feedback else None
        return self.generate(prompt, context, temperature=temp)

    def write_short_story(
        self,
        story_state: StoryState,
        revision_feedback: str | None = None,
    ) -> str:
        """Write a complete short story in one pass."""
        brief = story_state.brief
        if not brief:
            raise ValueError("Story brief is required to write a short story")
        context = story_state.get_context_summary()

        chars = "\n".join(f"- {c.name}: {c.description}" for c in story_state.characters)

        revision_note = ""
        if revision_feedback:
            revision_note = f"\n\nREVISION REQUESTED:\n{revision_feedback}\n\nAddress these issues while rewriting."

        prompt = f"""Write a complete short story based on this premise:

LANGUAGE: {brief.language} - Write the ENTIRE story in {brief.language}. All prose, dialogue, and narration must be in {brief.language}.

{brief.premise}

GENRE: {brief.genre}
TONE: {brief.tone}
THEMES: {", ".join(brief.themes)}
CONTENT RATING: {brief.content_rating}
SETTING: {brief.setting_place}, {brief.setting_time}

CHARACTERS:
{chars}

PLOT OUTLINE:
{story_state.plot_summary}
{revision_note}

Write a complete, polished short story in {brief.language} (2000-4000 words).
Include a strong opening hook, rising tension, a satisfying climax, and resolution.
Show character growth and explore the themes naturally through the narrative.

Write only the story prose - no titles, headers, or meta-commentary.
IMPORTANT: Every word must be in {brief.language}!"""

        # Use lower temperature for revisions (more focused output)
        temp = self.settings.revision_temperature if revision_feedback else None
        return self.generate(prompt, context, temperature=temp)

    def continue_scene(
        self,
        story_state: StoryState,
        current_text: str,
        direction: str | None = None,
    ) -> str:
        """Continue writing from where the text left off."""
        brief = story_state.brief
        if not brief:
            raise ValueError("Story brief is required to continue scene")
        context = story_state.get_context_summary()

        prompt = f"""Continue this scene:

LANGUAGE: {brief.language} - Continue writing in {brief.language}. All prose must be in {brief.language}.

{current_text[-1500:]}

{"DIRECTION: " + direction if direction else "Continue naturally to the next beat."}

GENRE: {brief.genre}
TONE: {brief.tone}
CONTENT RATING: {brief.content_rating}

Continue seamlessly from where the text ends. Write 500-1000 more words in {brief.language}.
Maintain the same voice, tense, and style."""

        return self.generate(prompt, context)

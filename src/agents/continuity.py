"""Continuity Checker Agent - Detects plot holes and inconsistencies."""

import logging
from typing import Any

from pydantic import BaseModel, Field, model_validator

from src.memory.story_state import StoryState
from src.settings import Settings
from src.utils.exceptions import LLMGenerationError
from src.utils.validation import (
    validate_not_empty,
    validate_not_none,
    validate_positive,
    validate_type,
)

from .base import BaseAgent

logger = logging.getLogger(__name__)


class ContinuityIssue(BaseModel):
    """A detected continuity issue."""

    severity: str  # "critical", "moderate", "minor"
    category: str  # "language", "plot_hole", "character", "timeline", "setting", "logic", "voice"
    description: str
    location: str  # Where in the text
    suggestion: str  # How to fix


class ContinuityIssueList(BaseModel):
    """Wrapper for a list of continuity issues.

    Used with generate_structured() to get validated issue lists from LLM.
    Handles LLMs returning a single object instead of a wrapped list.
    """

    issues: list[ContinuityIssue] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def wrap_single_object(cls, data: Any) -> Any:
        """Wrap a single ContinuityIssue object in a list if needed."""
        if isinstance(data, dict) and "issues" not in data:
            # LLM returned a single object, wrap it
            if "severity" in data and "category" in data:
                logger.debug("Wrapping single ContinuityIssue object in ContinuityIssueList")
                return {"issues": [data]}
        return data


class DialoguePattern(BaseModel):
    """Character dialogue patterns for voice consistency."""

    character_name: str
    vocabulary_level: str  # formal, casual, colloquial, technical
    speech_patterns: list[str] = Field(default_factory=list)  # Characteristic phrases, verbal tics
    typical_words: list[str] = Field(default_factory=list)  # Words this character commonly uses
    sentence_structure: str  # simple, complex, fragmented, eloquent


class DialoguePatternList(BaseModel):
    """Wrapper for a list of dialogue patterns.

    Used with generate_structured() to get validated pattern lists from LLM.
    Handles LLMs returning a single object instead of a wrapped list.
    """

    patterns: list[DialoguePattern] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def wrap_single_object(cls, data: Any) -> Any:
        """Wrap a single DialoguePattern object in a list if needed."""
        if isinstance(data, dict) and "patterns" not in data:
            # LLM returned a single object, wrap it
            if "character_name" in data and "vocabulary_level" in data:
                logger.debug("Wrapping single DialoguePattern object in DialoguePatternList")
                return {"patterns": [data]}
        return data


CONTINUITY_SYSTEM_PROMPT = """You are the Continuity Checker, the guardian of story consistency.

CRITICAL: The story must be in the specified language. Flag ANY text that is in the wrong language as a critical issue.

Your job is to catch:
1. LANGUAGE VIOLATIONS - Any text not in the specified language (ALWAYS check this first!)
2. PLOT HOLES - Events that contradict earlier events, unresolved setups, impossible occurrences
3. CHARACTER INCONSISTENCIES - Out-of-character behavior, personality shifts, forgotten traits
4. VOICE INCONSISTENCIES - Characters speaking out of character, dialogue that doesn't match their established speech patterns
5. TIMELINE ERRORS - Impossible chronology, contradictory timeframes
6. SETTING MISTAKES - Location inconsistencies, world rule violations
7. LOGIC PROBLEMS - Things that don't make sense within the story's rules

You read carefully and cross-reference with established facts.
You flag issues with specific quotes and clear explanations.
You suggest fixes that preserve the author's intent.

Be thorough but not nitpicky - focus on issues readers would notice.
Don't flag stylistic choices as errors.
Understand that some genre conventions (like coincidences in romance) are acceptable."""


class ContinuityAgent(BaseAgent):
    """Agent that checks for plot holes and inconsistencies."""

    def __init__(self, model: str | None = None, settings: Settings | None = None) -> None:
        """Initialize the Continuity Checker agent.

        Args:
            model: Override model to use. If None, uses settings-based model for continuity.
            settings: Application settings. If None, loads default settings.
        """
        super().__init__(
            name="Continuity Checker",
            role="Consistency Guardian",
            system_prompt=CONTINUITY_SYSTEM_PROMPT,
            agent_role="continuity",
            model=model,
            settings=settings,
        )

    def check_chapter(
        self,
        story_state: StoryState,
        chapter_content: str,
        chapter_number: int,
        check_voice: bool = True,
        established_patterns: dict[str, DialoguePattern] | None = None,
        world_context: str = "",
    ) -> list[ContinuityIssue]:
        """Check a chapter for continuity issues.

        Args:
            story_state: Current story state
            chapter_content: Chapter text to check
            chapter_number: Chapter number being checked
            check_voice: Whether to perform voice consistency checks (default: True)
            established_patterns: Previously established dialogue patterns (optional)
            world_context: Optional RAG-retrieved world context for richer consistency checks.

        Returns:
            List of continuity issues found.
        """
        validate_not_none(story_state, "story_state")
        validate_type(story_state, "story_state", StoryState)
        validate_not_empty(chapter_content, "chapter_content")
        validate_positive(chapter_number, "chapter_number")

        # Build context from previous chapters
        ctx_chars = self.settings.previous_chapter_context_chars
        previous_content = ""
        for ch in story_state.chapters:
            if ch.number < chapter_number and ch.content:
                previous_content += f"\n[Chapter {ch.number}]\n{ch.content[-ctx_chars:]}\n"

        chars_summary = "\n".join(
            f"- {c.name}: {c.description} | Traits: {', '.join(c.trait_names)}"
            for c in story_state.characters
        )

        world_context_block = ""
        if world_context:
            logger.debug(
                "Injecting world context into continuity check prompt (%d chars)",
                len(world_context),
            )
            world_context_block = f"\nRETRIEVED WORLD CONTEXT:\n{world_context}\n"

        brief = story_state.brief
        if not brief:
            raise ValueError("Story brief is required to check chapter continuity")
        prompt = f"""Analyze this chapter for continuity issues:

REQUIRED LANGUAGE: {brief.language} - ALL story text MUST be in {brief.language}. Flag any text in wrong language as CRITICAL.

ESTABLISHED CHARACTERS:
{chars_summary}

ESTABLISHED FACTS:
{chr(10).join(story_state.established_facts[-30:])}

WORLD RULES:
{chr(10).join(story_state.world_rules)}
{world_context_block}
PREVIOUS CHAPTER ENDINGS:
{previous_content}

CURRENT CHAPTER (#{chapter_number}):
{chapter_content}

Find any:
- LANGUAGE VIOLATIONS (text not in {brief.language}) - mark as CRITICAL
- Plot holes or contradictions
- Character inconsistencies
- Timeline errors
- Setting/world rule violations
- Logic problems

Return a list of issues found. If no issues, return an empty list."""

        try:
            result = self.generate_structured(prompt, ContinuityIssueList)
            issues = result.issues
        except LLMGenerationError:  # pragma: no cover
            raise  # Re-raise LLM errors directly
        except Exception as e:
            logger.error(f"Structured generation failed for continuity check: {e}")
            raise LLMGenerationError(f"Failed to check continuity: {e}") from e

        # Perform voice consistency check if requested
        if check_voice and story_state.characters:
            logger.debug("Performing voice consistency check")
            voice_issues = self.check_character_voice(
                story_state, chapter_content, established_patterns
            )
            issues.extend(voice_issues)

        return issues

    def extract_dialogue_patterns(
        self,
        story_state: StoryState,
        chapter_content: str,
    ) -> dict[str, DialoguePattern]:
        """Extract dialogue patterns for each character from chapter content.

        Returns:
            Dict mapping character names to their dialogue patterns.
        """
        if not story_state.characters:
            logger.debug("No characters defined, skipping dialogue pattern extraction")
            return {}

        logger.debug("Extracting dialogue patterns for %d characters", len(story_state.characters))

        chars_summary = "\n".join(
            f"- {c.name}: {c.description} | Traits: {', '.join(c.trait_names)}"
            for c in story_state.characters
        )

        prompt = f"""Analyze the dialogue patterns for each character who speaks in this chapter:

CHAPTER CONTENT:
{chapter_content[: self.settings.chapter_analysis_chars]}

CHARACTERS:
{chars_summary}

For each character who speaks, identify their dialogue characteristics:
- Vocabulary level (formal/casual/colloquial/technical)
- Speech patterns (catchphrases, verbal tics, repetitions)
- Typical words they use frequently
- Sentence structure (simple/complex/fragmented/eloquent)

Only include characters who actually speak in this chapter."""

        try:
            result = self.generate_structured(
                prompt,
                DialoguePatternList,
                temperature=self.settings.temp_plot_checking,
            )
            patterns = {p.character_name: p for p in result.patterns if p.character_name}
            logger.debug("Extracted dialogue patterns for %d characters", len(patterns))
            return patterns
        except LLMGenerationError:  # pragma: no cover
            raise  # Re-raise LLM errors directly
        except Exception as e:
            logger.error(f"Structured generation failed for dialogue patterns: {e}")
            raise LLMGenerationError(f"Failed to extract dialogue patterns: {e}") from e

    def check_character_voice(
        self,
        story_state: StoryState,
        chapter_content: str,
        established_patterns: dict[str, DialoguePattern] | None = None,
    ) -> list[ContinuityIssue]:
        """Check for character voice inconsistencies in dialogue.

        Args:
            story_state: Current story state with character information
            chapter_content: Chapter text to analyze
            established_patterns: Previously established dialogue patterns (optional)

        Returns:
            List of voice consistency issues found.
        """
        if not story_state.characters:
            logger.debug("No characters defined, skipping voice consistency check")
            return []

        logger.debug("Checking voice consistency for %d characters", len(story_state.characters))

        char_lines = []
        for c in story_state.characters:
            parts = [f"- {c.name}: {c.description}"]
            categorized_found = False
            for category, label in [
                ("core", "Core Traits"),
                ("flaw", "Flaws"),
                ("quirk", "Quirks"),
            ]:
                traits = c.traits_by_category(category)
                if traits:
                    parts.append(f"  {label}: {', '.join(traits)}")
                    categorized_found = True
            if not categorized_found and c.trait_names:
                parts.append(f"  Personality: {', '.join(c.trait_names)}")
            char_lines.append("\n".join(parts))
        chars_summary = "\n".join(char_lines)

        # Build context from established patterns
        patterns_context = ""
        if established_patterns:
            patterns_parts = []
            for name, pattern in established_patterns.items():
                patterns_parts.append(
                    f"- {name}: {pattern.vocabulary_level} vocabulary, "
                    f"{pattern.sentence_structure} sentences"
                )
                if pattern.speech_patterns:
                    patterns_parts.append(f"  Patterns: {', '.join(pattern.speech_patterns)}")
                if pattern.typical_words:
                    patterns_parts.append(f"  Common words: {', '.join(pattern.typical_words[:5])}")
            patterns_context = "\n".join(patterns_parts)

        prompt = f"""Analyze dialogue for character voice consistency:

ESTABLISHED CHARACTERS:
{chars_summary}

{f"ESTABLISHED SPEECH PATTERNS:{chr(10)}{patterns_context}{chr(10)}" if patterns_context else ""}

CHAPTER CONTENT:
{chapter_content[: self.settings.chapter_analysis_chars]}

Check each character's dialogue for:
1. Does vocabulary match their education/background/personality?
2. Are speech patterns consistent with their character traits?
3. Do formal/informal language choices fit the character?
4. Are verbal tics or catchphrases used consistently?
5. Does sentence complexity match their intelligence/personality?

Return voice inconsistencies found. Set category to "voice". If no issues, return an empty list."""

        try:
            result = self.generate_structured(
                prompt,
                ContinuityIssueList,
                temperature=self.settings.temp_plot_checking,
            )
            return result.issues
        except LLMGenerationError:  # pragma: no cover
            raise  # Re-raise LLM errors directly
        except Exception as e:
            logger.error(f"Structured generation failed for voice check: {e}")
            raise LLMGenerationError(f"Failed to check character voice: {e}") from e

    def check_full_story(
        self, story_state: StoryState, check_voice: bool = True, world_context: str = ""
    ) -> list[ContinuityIssue]:
        """Check the entire story for continuity issues.

        Args:
            story_state: Current story state
            check_voice: Whether to perform voice consistency checks (default: True)
            world_context: Optional world context (RAG + temporal) for richer consistency checks.

        Returns:
            List of continuity issues found.
        """
        validate_not_none(story_state, "story_state")
        validate_type(story_state, "story_state", StoryState)

        full_content = "\n\n".join(
            f"[Chapter {ch.number}: {ch.title}]\n{ch.content}"
            for ch in story_state.chapters
            if ch.content
        )

        if not full_content:
            return []

        world_context_block = ""
        if world_context:
            logger.debug(
                "Injecting world context into full story continuity check prompt (%d chars)",
                len(world_context),
            )
            world_context_block = f"\nRETRIEVED WORLD CONTEXT:\n{world_context}\n"

        brief = story_state.brief
        if not brief:
            raise ValueError("Story brief is required to check full story continuity")
        prompt = f"""Analyze this complete story for continuity issues:

REQUIRED LANGUAGE: {brief.language} - ALL story text MUST be in {brief.language}. Flag any text in wrong language as CRITICAL.

STORY PREMISE:
{brief.premise}

CHARACTERS:
{chr(10).join(f"- {c.name} ({c.role})" for c in story_state.characters)}
{world_context_block}
FULL STORY:
{full_content[:8000]}

Check for:
- LANGUAGE VIOLATIONS (text not in {brief.language}) - mark as CRITICAL
- Unresolved plot threads
- Character arcs that don't complete
- Foreshadowing that never pays off
- Overall logic issues
- Timeline inconsistencies across chapters

Return a list of issues found. If no issues, return an empty list."""

        try:
            result = self.generate_structured(prompt, ContinuityIssueList)
            issues = result.issues
        except LLMGenerationError:  # pragma: no cover
            raise  # Re-raise LLM errors directly
        except Exception as e:
            logger.error(f"Structured generation failed for full story check: {e}")
            raise LLMGenerationError(f"Failed to check full story: {e}") from e

        # Perform voice consistency check if requested
        if check_voice and story_state.characters and full_content:
            logger.debug("Performing voice consistency check on full story")
            # Extract patterns from the full story first
            patterns = self.extract_dialogue_patterns(story_state, full_content)
            # Then check for voice inconsistencies using those patterns
            voice_issues = self.check_character_voice(story_state, full_content, patterns)
            issues.extend(voice_issues)

        return issues

    def validate_against_outline(
        self,
        story_state: StoryState,
        chapter_content: str,
        chapter_outline: str,
    ) -> list[ContinuityIssue]:
        """Check if a chapter fulfills its outline."""
        prompt = f"""Compare this chapter content against its outline:

OUTLINE:
{chapter_outline}

ACTUAL CONTENT:
{chapter_content}

Check if:
- All outlined events occur
- Character moments are included
- Plot points are addressed
- Nothing major was skipped or contradicted

Return only issues (not confirmations). Use category "plot_hole" for outline mismatches.
If no issues, return an empty list."""

        try:
            result = self.generate_structured(prompt, ContinuityIssueList)
            return result.issues
        except LLMGenerationError:  # pragma: no cover
            raise  # Re-raise LLM errors directly
        except Exception as e:
            logger.error(f"Structured generation failed for outline validation: {e}")
            raise LLMGenerationError(f"Failed to validate outline: {e}") from e

    def extract_new_facts(
        self,
        chapter_content: str,
        story_state: StoryState,
    ) -> list[str]:
        """Extract new established facts from a chapter."""
        prompt = f"""Extract important facts established in this chapter that should be remembered:

{chapter_content}

ALREADY KNOWN:
{chr(10).join(story_state.established_facts[-30:])}

List NEW facts only - things that are now canon:
- Character revelations
- Plot events that happened
- World details established
- Relationship changes

Output as a simple list, one fact per line, starting with "- "."""

        response = self.generate(prompt)

        facts = []
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("-"):
                facts.append(line[1:].strip())

        return facts

    def should_revise(self, issues: list[ContinuityIssue]) -> bool:
        """Determine if issues are severe enough to warrant revision."""
        critical_count = sum(1 for i in issues if i.severity == "critical")
        moderate_count = sum(1 for i in issues if i.severity == "moderate")

        # Revise if any critical issues or 3+ moderate issues
        return critical_count > 0 or moderate_count >= 3

    def format_revision_feedback(self, issues: list[ContinuityIssue]) -> str:
        """Format issues into feedback for the writer."""
        if not issues:
            return ""

        feedback_parts = ["The following continuity issues need to be addressed:\n"]

        for i, issue in enumerate(issues, 1):
            feedback_parts.append(
                f"{i}. [{issue.severity.upper()}] {issue.category}: {issue.description}\n"
                f"   Location: {issue.location}\n"
                f"   Suggestion: {issue.suggestion}\n"
            )

        return "\n".join(feedback_parts)

    def extract_character_arcs(
        self,
        chapter_content: str,
        story_state: StoryState,
        chapter_number: int,
    ) -> dict[str, str]:
        """Extract character arc updates from a chapter.

        Returns:
            Dict mapping character names to their arc state in this chapter.
        """
        char_names = [c.name for c in story_state.characters]
        if not char_names:
            return {}

        prompt = f"""Analyze how each character develops in this chapter:

CHAPTER CONTENT:
{chapter_content[: self.settings.chapter_analysis_chars]}

CHARACTERS TO ANALYZE:
{chr(10).join(f"- {c.name}: {c.arc_notes}" for c in story_state.characters)}

For each character who appears in this chapter, describe their current emotional/psychological state and any development or change they undergo.

Output as simple lines:
CHARACTER_NAME: Brief description of their state/development in this chapter

Only include characters who actually appear in this chapter."""

        response = self.generate(prompt, temperature=self.temperature)

        arcs = {}
        for line in response.split("\n"):
            line = line.strip()
            if ":" in line:
                parts = line.split(":", 1)
                name = parts[0].strip()
                state = parts[1].strip() if len(parts) > 1 else ""
                # Match to actual character names (case-insensitive)
                for char_name in char_names:
                    if name.lower() == char_name.lower() or name.lower() in char_name.lower():
                        arcs[char_name] = state
                        break

        return arcs

    def check_plot_points_completed(
        self,
        chapter_content: str,
        story_state: StoryState,
        chapter_number: int,
    ) -> list[int]:
        """Check which plot points were completed in this chapter.

        Returns:
            List of indices of completed plot points.
        """
        pending_points = [
            (i, p)
            for i, p in enumerate(story_state.plot_points)
            if not p.completed and (p.chapter is None or p.chapter == chapter_number)
        ]

        if not pending_points:
            return []

        points_text = "\n".join(f"{i}. {p.description}" for i, p in pending_points)

        prompt = f"""Check which plot points were addressed in this chapter:

CHAPTER CONTENT:
{chapter_content[: self.settings.chapter_analysis_chars]}

PENDING PLOT POINTS:
{points_text}

List ONLY the numbers of plot points that were CLEARLY addressed or completed in this chapter.
Output format: Just the numbers separated by commas (e.g., "0, 2, 5") or "none" if no plot points were addressed."""

        # Use min_response_length=1 because valid responses can be very short
        # (e.g., "0" for first plot point, "none" for no completions)
        response = self.generate(
            prompt, temperature=self.settings.temp_plot_checking, min_response_length=1
        )

        completed_indices = []
        response_clean = response.lower().strip()

        if response_clean == "none" or not response_clean:
            return []

        # Parse numbers from response
        import re

        numbers = re.findall(r"\d+", response)
        for num_str in numbers:
            idx = int(num_str)
            # Verify it's a valid pending point index
            valid_indices = [i for i, _ in pending_points]
            if idx in valid_indices:
                completed_indices.append(idx)

        return completed_indices

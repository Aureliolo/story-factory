"""Continuity Checker Agent - Detects plot holes and inconsistencies."""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from memory.story_state import StoryState
from utils.json_parser import extract_json_list
from utils.validation import validate_not_empty, validate_not_none, validate_positive, validate_type

from .base import BaseAgent

if TYPE_CHECKING:
    from settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class ContinuityIssue:
    """A detected continuity issue."""

    severity: str  # "critical", "moderate", "minor"
    category: str  # "language", "plot_hole", "character", "timeline", "setting", "logic", "voice"
    description: str
    location: str  # Where in the text
    suggestion: str  # How to fix


@dataclass
class DialoguePattern:
    """Character dialogue patterns for voice consistency."""

    character_name: str
    vocabulary_level: str  # formal, casual, colloquial, technical
    speech_patterns: list[str]  # Characteristic phrases, verbal tics
    typical_words: list[str]  # Words this character commonly uses
    sentence_structure: str  # simple, complex, fragmented, eloquent


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

    def __init__(self, model: str | None = None, settings: "Settings | None" = None) -> None:
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
    ) -> list[ContinuityIssue]:
        """Check a chapter for continuity issues.

        Args:
            story_state: Current story state
            chapter_content: Chapter text to check
            chapter_number: Chapter number being checked
            check_voice: Whether to perform voice consistency checks (default: True)
            established_patterns: Previously established dialogue patterns (optional)

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
            f"- {c.name}: {c.description} | Traits: {', '.join(c.personality_traits)}"
            for c in story_state.characters
        )

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

Output as JSON:
```json
[
    {{
        "severity": "critical|moderate|minor",
        "category": "language|plot_hole|character|timeline|setting|logic",
        "description": "What the issue is",
        "location": "Quote the problematic text",
        "suggestion": "How to fix it"
    }}
]
```

If no issues found, output: ```json
[]
```"""

        response = self.generate(prompt)
        issues = self._parse_issues(response)

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
            f"- {c.name}: {c.description} | Traits: {', '.join(c.personality_traits)}"
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

Output as JSON array:
```json
[
    {{
        "character_name": "Character Name",
        "vocabulary_level": "formal|casual|colloquial|technical",
        "speech_patterns": ["pattern1", "pattern2"],
        "typical_words": ["word1", "word2"],
        "sentence_structure": "simple|complex|fragmented|eloquent"
    }}
]
```

Only include characters who actually speak in this chapter."""

        response = self.generate(prompt, temperature=self.settings.temp_plot_checking)
        return self._parse_dialogue_patterns(response)

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

        chars_summary = "\n".join(
            f"- {c.name}: {c.description}\n  Personality: {', '.join(c.personality_traits)}"
            for c in story_state.characters
        )

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

Output voice inconsistencies as JSON:
```json
[
    {{
        "severity": "moderate|minor",
        "category": "voice",
        "description": "Character speaks out of character",
        "location": "Quote the problematic dialogue",
        "suggestion": "How to rewrite the dialogue to match their voice"
    }}
]
```

If no voice issues found, output: ```json
[]
```"""

        response = self.generate(prompt, temperature=self.settings.temp_plot_checking)
        return self._parse_issues(response)

    def _parse_dialogue_patterns(self, response: str) -> dict[str, DialoguePattern]:
        """Parse dialogue patterns from agent response."""
        data = extract_json_list(response)
        if not data:
            logger.debug("No dialogue patterns found in response")
            return {}

        patterns = {}
        for item in data:
            try:
                pattern = DialoguePattern(
                    character_name=item.get("character_name", ""),
                    vocabulary_level=item.get("vocabulary_level", "casual"),
                    speech_patterns=item.get("speech_patterns", []),
                    typical_words=item.get("typical_words", []),
                    sentence_structure=item.get("sentence_structure", "simple"),
                )
                if pattern.character_name:
                    patterns[pattern.character_name] = pattern
            except (TypeError, KeyError) as e:
                logger.debug(f"Skipping malformed dialogue pattern item: {e}")

        logger.debug("Extracted dialogue patterns for %d characters", len(patterns))
        return patterns

    def check_full_story(
        self, story_state: StoryState, check_voice: bool = True
    ) -> list[ContinuityIssue]:
        """Check the entire story for continuity issues.

        Args:
            story_state: Current story state
            check_voice: Whether to perform voice consistency checks (default: True)

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

        brief = story_state.brief
        if not brief:
            raise ValueError("Story brief is required to check full story continuity")
        prompt = f"""Analyze this complete story for continuity issues:

REQUIRED LANGUAGE: {brief.language} - ALL story text MUST be in {brief.language}. Flag any text in wrong language as CRITICAL.

STORY PREMISE:
{brief.premise}

CHARACTERS:
{chr(10).join(f"- {c.name} ({c.role})" for c in story_state.characters)}

FULL STORY:
{full_content[:8000]}  # Truncate for context limits

Check for:
- LANGUAGE VIOLATIONS (text not in {brief.language}) - mark as CRITICAL
- Unresolved plot threads
- Character arcs that don't complete
- Foreshadowing that never pays off
- Overall logic issues
- Timeline inconsistencies across chapters

Output as JSON:
```json
[
    {{
        "severity": "critical|moderate|minor",
        "category": "language|plot_hole|character|timeline|setting|logic",
        "description": "What the issue is",
        "location": "Which chapter(s) / quote",
        "suggestion": "How to fix it"
    }}
]
```"""

        response = self.generate(prompt)
        issues = self._parse_issues(response)

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

Output as JSON (only issues, not confirmations):
```json
[
    {{
        "severity": "critical|moderate|minor",
        "category": "plot_hole",
        "description": "What's missing or wrong",
        "location": "outline vs content",
        "suggestion": "What needs to be added/fixed"
    }}
]
```"""

        response = self.generate(prompt)
        return self._parse_issues(response)

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

    def _parse_issues(self, response: str) -> list[ContinuityIssue]:
        """Parse continuity issues from agent response."""
        data = extract_json_list(response)
        if not data:
            return []

        issues = []
        for item in data:
            try:
                issues.append(ContinuityIssue(**item))
            except (TypeError, KeyError) as e:
                logger.debug(f"Skipping malformed continuity issue item: {e}")

        return issues

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

        response = self.generate(prompt, temperature=self.settings.temp_plot_checking)

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

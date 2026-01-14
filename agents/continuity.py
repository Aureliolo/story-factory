"""Continuity Checker Agent - Detects plot holes and inconsistencies."""

from .base import BaseAgent
from memory.story_state import StoryState
import json
import re
from dataclasses import dataclass


@dataclass
class ContinuityIssue:
    """A detected continuity issue."""
    severity: str  # "critical", "moderate", "minor"
    category: str  # "plot_hole", "character", "timeline", "setting", "logic"
    description: str
    location: str  # Where in the text
    suggestion: str  # How to fix


CONTINUITY_SYSTEM_PROMPT = """You are the Continuity Checker, the guardian of story consistency.

Your job is to catch:
1. PLOT HOLES - Events that contradict earlier events, unresolved setups, impossible occurrences
2. CHARACTER INCONSISTENCIES - Out-of-character behavior, personality shifts, forgotten traits
3. TIMELINE ERRORS - Impossible chronology, contradictory timeframes
4. SETTING MISTAKES - Location inconsistencies, world rule violations
5. LOGIC PROBLEMS - Things that don't make sense within the story's rules

You read carefully and cross-reference with established facts.
You flag issues with specific quotes and clear explanations.
You suggest fixes that preserve the author's intent.

Be thorough but not nitpicky - focus on issues readers would notice.
Don't flag stylistic choices as errors.
Understand that some genre conventions (like coincidences in romance) are acceptable."""


class ContinuityAgent(BaseAgent):
    """Agent that checks for plot holes and inconsistencies."""

    def __init__(self, model: str = None, settings=None):
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
    ) -> list[ContinuityIssue]:
        """Check a chapter for continuity issues."""
        # Build context from previous chapters
        previous_content = ""
        for ch in story_state.chapters:
            if ch.number < chapter_number and ch.content:
                previous_content += f"\n[Chapter {ch.number}]\n{ch.content[-500:]}\n"

        chars_summary = "\n".join(
            f"- {c.name}: {c.description} | Traits: {', '.join(c.personality_traits)}"
            for c in story_state.characters
        )

        prompt = f"""Analyze this chapter for continuity issues:

ESTABLISHED CHARACTERS:
{chars_summary}

ESTABLISHED FACTS:
{chr(10).join(story_state.established_facts[-20:])}

WORLD RULES:
{chr(10).join(story_state.world_rules)}

PREVIOUS CHAPTER ENDINGS:
{previous_content}

CURRENT CHAPTER (#{chapter_number}):
{chapter_content}

Find any:
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
        "category": "plot_hole|character|timeline|setting|logic",
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
        return self._parse_issues(response)

    def check_full_story(self, story_state: StoryState) -> list[ContinuityIssue]:
        """Check the entire story for continuity issues."""
        full_content = "\n\n".join(
            f"[Chapter {ch.number}: {ch.title}]\n{ch.content}"
            for ch in story_state.chapters
            if ch.content
        )

        if not full_content:
            return []

        prompt = f"""Analyze this complete story for continuity issues:

STORY PREMISE:
{story_state.brief.premise if story_state.brief else 'N/A'}

CHARACTERS:
{chr(10).join(f"- {c.name} ({c.role})" for c in story_state.characters)}

FULL STORY:
{full_content[:8000]}  # Truncate for context limits

Check for:
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
        "category": "plot_hole|character|timeline|setting|logic",
        "description": "What the issue is",
        "location": "Which chapter(s) / quote",
        "suggestion": "How to fix it"
    }}
]
```"""

        response = self.generate(prompt)
        return self._parse_issues(response)

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
{chr(10).join(story_state.established_facts[-10:])}

List NEW facts only - things that are now canon:
- Character revelations
- Plot events that happened
- World details established
- Relationship changes

Output as a simple list, one fact per line, starting with "- "."""

        response = self.generate(prompt)

        facts = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('-'):
                facts.append(line[1:].strip())

        return facts

    def _parse_issues(self, response: str) -> list[ContinuityIssue]:
        """Parse continuity issues from agent response."""
        issues = []
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)

        if json_match:
            try:
                data = json.loads(json_match.group(1))
                for item in data:
                    issues.append(ContinuityIssue(**item))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

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

"""Tests for ContinuityAgent."""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.continuity import (
    ContinuityAgent,
    ContinuityIssue,
    ContinuityIssueList,
    DialoguePattern,
    DialoguePatternList,
)
from src.memory.story_state import Chapter, Character, PlotPoint, StoryBrief, StoryState
from src.settings import Settings


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings()


@pytest.fixture
def continuity(settings):
    """Create ContinuityAgent with mocked Ollama client."""
    with patch("src.agents.base.ollama.Client"):
        agent = ContinuityAgent(model="test-model", settings=settings)
        return agent


@pytest.fixture
def sample_story_state():
    """Create a sample story state with chapters."""
    brief = StoryBrief(
        premise="A group of friends uncover an ancient mystery",
        genre="Adventure",
        subgenres=["Mystery", "Supernatural"],
        tone="Thrilling",
        themes=["Friendship", "Discovery"],
        setting_time="Present Day",
        setting_place="Egypt",
        target_length="novella",
        language="English",
        content_rating="general",
        content_preferences=["Puzzles", "Ancient history"],
        content_avoid=["Gore"],
    )
    state = StoryState(
        id="test-story-001",
        project_name="The Lost Tomb",
        brief=brief,
        status="writing",
        characters=[
            Character(
                name="Sarah Chen",
                role="protagonist",
                description="Brilliant archaeologist",
                personality_traits=["intelligent", "brave", "stubborn"],
                goals=["Find the lost tomb", "Honor her mentor's legacy"],
                arc_notes="Learns to trust others",
            ),
            Character(
                name="Marcus Wells",
                role="supporting",
                description="Skeptical journalist",
                personality_traits=["cynical", "loyal", "curious"],
                goals=["Get the story", "Protect Sarah"],
                arc_notes="Becomes a believer",
            ),
        ],
        chapters=[
            Chapter(
                number=1,
                title="The Discovery",
                outline="Sarah finds a clue",
                content="Sarah brushed away the sand...",
            ),
            Chapter(number=2, title="The Journey", outline="They travel to Egypt", content=""),
        ],
        established_facts=[
            "Sarah's mentor died mysteriously",
            "The tomb is hidden in the Valley of Kings",
            "Marcus has brown eyes",
        ],
        world_rules=[
            "Ancient curses are real in this world",
            "The tomb was built by Queen Nefertiti",
        ],
    )
    return state


class TestContinuityAgentInit:
    """Tests for ContinuityAgent initialization."""

    def test_init_with_defaults(self, settings):
        """Test agent initializes with default settings."""
        with patch("src.agents.base.ollama.Client"):
            agent = ContinuityAgent(settings=settings)
            assert agent.name == "Continuity Checker"
            assert agent.role == "Consistency Guardian"

    def test_init_with_custom_model(self, settings):
        """Test agent initializes with custom model."""
        with patch("src.agents.base.ollama.Client"):
            agent = ContinuityAgent(model="continuity-model:7b", settings=settings)
            assert agent.model == "continuity-model:7b"


class TestContinuityCheckChapter:
    """Tests for check_chapter method."""

    def test_returns_issues_list(self, continuity, sample_story_state):
        """Test returns list of continuity issues."""
        mock_result = ContinuityIssueList(
            issues=[
                ContinuityIssue(
                    severity="moderate",
                    category="character",
                    description="Marcus is described as having blue eyes, but was established with brown eyes",
                    location="Paragraph 3: 'His blue eyes sparkled'",
                    suggestion="Change 'blue eyes' to 'brown eyes'",
                )
            ]
        )
        continuity.generate_structured = MagicMock(return_value=mock_result)

        issues = continuity.check_chapter(
            sample_story_state,
            "The chapter content with Marcus's blue eyes sparkling...",
            chapter_number=2,
            check_voice=False,  # Disable voice check for this test
        )

        assert len(issues) == 1
        assert issues[0].severity == "moderate"
        assert issues[0].category == "character"
        assert "brown eyes" in issues[0].suggestion

    def test_returns_empty_list_when_no_issues(self, continuity, sample_story_state):
        """Test returns empty list when no issues found."""
        mock_result = ContinuityIssueList(issues=[])
        continuity.generate_structured = MagicMock(return_value=mock_result)

        issues = continuity.check_chapter(
            sample_story_state, "Perfect chapter content...", chapter_number=1, check_voice=False
        )

        assert issues == []

    def test_includes_established_facts(self, continuity, sample_story_state):
        """Test prompt includes established facts."""
        mock_result = ContinuityIssueList(issues=[])
        continuity.generate_structured = MagicMock(return_value=mock_result)

        continuity.check_chapter(sample_story_state, "Content", 1, check_voice=False)

        call_args = continuity.generate_structured.call_args
        prompt = call_args[0][0]
        assert "ESTABLISHED FACTS" in prompt
        assert "mentor died" in prompt

    def test_raises_without_brief(self, continuity):
        """Test raises error when brief is missing."""
        state = StoryState(id="test", status="writing")

        with pytest.raises(ValueError, match="brief"):
            continuity.check_chapter(state, "Content", 1)

    def test_raises_on_generation_exception(self, continuity, sample_story_state):
        """Test raises LLMGenerationError when generate_structured fails."""
        from src.utils.exceptions import LLMGenerationError

        continuity.generate_structured = MagicMock(side_effect=Exception("LLM error"))

        with pytest.raises(LLMGenerationError, match="Failed to check continuity"):
            continuity.check_chapter(
                sample_story_state, "Content...", chapter_number=1, check_voice=False
            )


class TestContinuityCheckFullStory:
    """Tests for check_full_story method."""

    def test_checks_entire_story(self, continuity, sample_story_state):
        """Test checks the full story for issues."""
        sample_story_state.chapters[0].content = "Chapter 1 content..."
        sample_story_state.chapters[1].content = "Chapter 2 content..."

        mock_result = ContinuityIssueList(
            issues=[
                ContinuityIssue(
                    severity="critical",
                    category="plot_hole",
                    description="The ancient map mentioned in chapter 1 is never found",
                    location="Chapter 1 vs overall plot",
                    suggestion="Add a scene where Sarah finds the map",
                )
            ]
        )
        continuity.generate_structured = MagicMock(return_value=mock_result)

        issues = continuity.check_full_story(sample_story_state, check_voice=False)

        assert len(issues) == 1
        assert issues[0].severity == "critical"
        assert issues[0].category == "plot_hole"

    def test_returns_empty_when_no_content(self, continuity, sample_story_state):
        """Test returns empty list when chapters have no content."""
        sample_story_state.chapters[0].content = ""
        sample_story_state.chapters[1].content = ""
        continuity.generate_structured = MagicMock(return_value="Should not be called")

        issues = continuity.check_full_story(sample_story_state)

        assert issues == []
        continuity.generate_structured.assert_not_called()

    def test_raises_without_brief(self, continuity):
        """Test raises error when brief is missing."""
        state = StoryState(
            id="test",
            status="writing",
            chapters=[Chapter(number=1, title="Ch1", outline="", content="Some content")],
        )

        with pytest.raises(ValueError, match="brief"):
            continuity.check_full_story(state)

    def test_raises_on_generation_exception(self, continuity, sample_story_state):
        """Test raises LLMGenerationError when generate_structured fails."""
        from src.utils.exceptions import LLMGenerationError

        sample_story_state.chapters[0].content = "Chapter 1 content..."
        sample_story_state.chapters[1].content = "Chapter 2 content..."
        continuity.generate_structured = MagicMock(side_effect=Exception("LLM error"))

        with pytest.raises(LLMGenerationError, match="Failed to check full story"):
            continuity.check_full_story(sample_story_state, check_voice=False)


class TestContinuityValidateAgainstOutline:
    """Tests for validate_against_outline method."""

    def test_compares_content_to_outline(self, continuity, sample_story_state):
        """Test compares chapter content against outline."""
        mock_result = ContinuityIssueList(
            issues=[
                ContinuityIssue(
                    severity="moderate",
                    category="plot_hole",
                    description="The discovery of the ancient coin mentioned in outline is missing",
                    location="outline vs content",
                    suggestion="Add scene where Sarah finds the coin",
                )
            ]
        )
        continuity.generate_structured = MagicMock(return_value=mock_result)

        issues = continuity.validate_against_outline(
            sample_story_state,
            chapter_content="Sarah explored the site...",
            chapter_outline="Sarah finds an ancient coin that reveals the tomb location",
        )

        assert len(issues) == 1
        call_args = continuity.generate_structured.call_args
        prompt = call_args[0][0]
        assert "OUTLINE" in prompt
        assert "CONTENT" in prompt

    def test_raises_on_generation_exception(self, continuity, sample_story_state):
        """Test raises LLMGenerationError when generate_structured fails."""
        from src.utils.exceptions import LLMGenerationError

        continuity.generate_structured = MagicMock(side_effect=Exception("LLM error"))

        with pytest.raises(LLMGenerationError, match="Failed to validate outline"):
            continuity.validate_against_outline(
                sample_story_state,
                chapter_content="Sarah explored the site...",
                chapter_outline="Sarah finds an ancient coin",
            )


class TestContinuityExtractNewFacts:
    """Tests for extract_new_facts method."""

    def test_extracts_facts_from_chapter(self, continuity, sample_story_state):
        """Test extracts new established facts."""
        response = """New facts established in this chapter:
- Sarah can read ancient hieroglyphics
- The tomb entrance faces east
- Marcus carries a lucky coin from his grandfather"""

        continuity.generate = MagicMock(return_value=response)

        facts = continuity.extract_new_facts(
            "Chapter content where these facts are established...", sample_story_state
        )

        assert len(facts) == 3
        assert "Sarah can read ancient hieroglyphics" in facts
        assert "Marcus carries a lucky coin" in facts[2]

    def test_returns_empty_list_for_no_facts(self, continuity, sample_story_state):
        """Test returns empty list when no new facts."""
        continuity.generate = MagicMock(return_value="No new facts established.")

        facts = continuity.extract_new_facts("Simple chapter...", sample_story_state)

        assert facts == []


class TestContinuityIssueHelpers:
    """Tests for helper methods."""

    def test_should_revise_with_critical(self, continuity):
        """Test should_revise returns True for critical issues."""
        issues = [
            ContinuityIssue(
                severity="critical",
                category="language",
                description="Wrong language",
                location="Para 1",
                suggestion="Fix language",
            )
        ]

        assert continuity.should_revise(issues) is True

    def test_should_revise_with_multiple_moderate(self, continuity):
        """Test should_revise returns True for 3+ moderate issues."""
        issues = [
            ContinuityIssue(
                severity="moderate",
                category="character",
                description="Issue 1",
                location="",
                suggestion="",
            ),
            ContinuityIssue(
                severity="moderate",
                category="timeline",
                description="Issue 2",
                location="",
                suggestion="",
            ),
            ContinuityIssue(
                severity="moderate",
                category="setting",
                description="Issue 3",
                location="",
                suggestion="",
            ),
        ]

        assert continuity.should_revise(issues) is True

    def test_should_not_revise_minor_only(self, continuity):
        """Test should_revise returns False for minor issues only."""
        issues = [
            ContinuityIssue(
                severity="minor",
                category="logic",
                description="Minor issue",
                location="",
                suggestion="",
            ),
        ]

        assert continuity.should_revise(issues) is False

    def test_format_revision_feedback(self, continuity):
        """Test formats issues into feedback string."""
        issues = [
            ContinuityIssue(
                severity="critical",
                category="language",
                description="Text is in German instead of English",
                location="Paragraphs 2-5",
                suggestion="Rewrite in English",
            ),
            ContinuityIssue(
                severity="moderate",
                category="character",
                description="Eye color inconsistency",
                location="Paragraph 7",
                suggestion="Change blue to brown",
            ),
        ]

        feedback = continuity.format_revision_feedback(issues)

        assert "CRITICAL" in feedback
        assert "MODERATE" in feedback
        assert "German" in feedback
        assert "Eye color" in feedback

    def test_format_revision_feedback_empty(self, continuity):
        """Test returns empty string for no issues."""
        feedback = continuity.format_revision_feedback([])
        assert feedback == ""


class TestContinuityExtractCharacterArcs:
    """Tests for extract_character_arcs method."""

    def test_extracts_character_states(self, continuity, sample_story_state):
        """Test extracts character arc information."""
        response = """Sarah Chen: Begins to doubt her initial theories, showing growth from stubbornness to openness
Marcus Wells: Still skeptical but starting to show curiosity about the supernatural elements"""

        continuity.generate = MagicMock(return_value=response)

        arcs = continuity.extract_character_arcs(
            "Chapter content...", sample_story_state, chapter_number=3
        )

        assert "Sarah Chen" in arcs
        assert "doubt" in arcs["Sarah Chen"].lower() or "growth" in arcs["Sarah Chen"].lower()
        assert "Marcus Wells" in arcs

    def test_returns_empty_for_no_characters(self, continuity, sample_story_state):
        """Test returns empty dict when no characters found."""
        sample_story_state.characters = []

        arcs = continuity.extract_character_arcs("Content", sample_story_state, 1)

        assert arcs == {}


class TestContinuityCheckPlotPoints:
    """Tests for check_plot_points_completed method."""

    def test_identifies_completed_points(self, continuity, sample_story_state):
        """Test identifies which plot points were completed."""
        sample_story_state.plot_points = [
            PlotPoint(description="Sarah finds the first clue", chapter=1, completed=False),
            PlotPoint(description="They decode the map", chapter=1, completed=False),
            PlotPoint(description="Marcus saves Sarah", chapter=None, completed=False),
        ]

        continuity.generate = MagicMock(return_value="0, 1")

        completed = continuity.check_plot_points_completed(
            "Chapter where Sarah finds the clue and they decode the map...",
            sample_story_state,
            chapter_number=1,
        )

        assert 0 in completed
        assert 1 in completed

    def test_returns_empty_for_no_pending_points(self, continuity, sample_story_state):
        """Test returns empty list when no pending plot points."""
        sample_story_state.plot_points = [
            PlotPoint(description="Already done", chapter=1, completed=True),
        ]
        continuity.generate = MagicMock(return_value="Should not be called")

        completed = continuity.check_plot_points_completed("Content", sample_story_state, 1)

        assert completed == []
        continuity.generate.assert_not_called()

    def test_handles_none_response(self, continuity, sample_story_state):
        """Test handles 'none' response gracefully."""
        sample_story_state.plot_points = [
            PlotPoint(description="Pending point", chapter=1, completed=False),
        ]

        continuity.generate = MagicMock(return_value="none")

        completed = continuity.check_plot_points_completed("Content", sample_story_state, 1)

        assert completed == []


class TestContinuityParseIssues:
    """Tests for issue parsing edge cases."""

    def test_returns_valid_issues_from_structured_response(self, continuity, sample_story_state):
        """Test returns valid issues from structured response."""
        # With generate_structured, Pydantic handles validation
        mock_result = ContinuityIssueList(
            issues=[
                ContinuityIssue(
                    severity="moderate",
                    category="character",
                    description="Valid issue",
                    location="Para 1",
                    suggestion="Fix it",
                ),
                ContinuityIssue(
                    severity="minor",
                    category="logic",
                    description="Another valid issue",
                    location="Para 2",
                    suggestion="Fix that",
                ),
            ]
        )
        continuity.generate_structured = MagicMock(return_value=mock_result)

        issues = continuity.check_chapter(
            sample_story_state, "Content...", chapter_number=1, check_voice=False
        )

        # Should have 2 issues
        assert len(issues) == 2
        assert issues[0].description == "Valid issue"
        assert issues[1].description == "Another valid issue"


class TestContinuityVoiceConsistency:
    """Tests for voice consistency checking."""

    def test_extract_dialogue_patterns(self, continuity, sample_story_state):
        """Test extracts dialogue patterns from chapter."""
        mock_result = DialoguePatternList(
            patterns=[
                DialoguePattern(
                    character_name="Sarah Chen",
                    vocabulary_level="formal",
                    speech_patterns=["uses archaeological terms", "speaks precisely"],
                    typical_words=["exactly", "precisely", "artifact"],
                    sentence_structure="complex",
                ),
                DialoguePattern(
                    character_name="Marcus Wells",
                    vocabulary_level="casual",
                    speech_patterns=["uses contractions frequently", "informal"],
                    typical_words=["yeah", "gonna", "seriously"],
                    sentence_structure="simple",
                ),
            ]
        )
        continuity.generate_structured = MagicMock(return_value=mock_result)

        patterns = continuity.extract_dialogue_patterns(
            sample_story_state, "Chapter with dialogue..."
        )

        assert len(patterns) == 2
        assert "Sarah Chen" in patterns
        assert patterns["Sarah Chen"].vocabulary_level == "formal"
        assert any("archaeological" in p.lower() for p in patterns["Sarah Chen"].speech_patterns)
        assert "Marcus Wells" in patterns
        assert patterns["Marcus Wells"].vocabulary_level == "casual"

    def test_check_character_voice_finds_issues(self, continuity, sample_story_state):
        """Test checks character voice for inconsistencies."""
        mock_result = ContinuityIssueList(
            issues=[
                ContinuityIssue(
                    severity="moderate",
                    category="voice",
                    description="Sarah Chen speaks too casually for her character",
                    location="Sarah said, 'Yeah, whatever dude.'",
                    suggestion="Change to more formal language: 'I understand your point.'",
                )
            ]
        )
        continuity.generate_structured = MagicMock(return_value=mock_result)

        issues = continuity.check_character_voice(
            sample_story_state, "Chapter content with out-of-character dialogue..."
        )

        assert len(issues) == 1
        assert issues[0].category == "voice"
        assert "Sarah Chen" in issues[0].description
        assert "formal language" in issues[0].suggestion

    def test_check_character_voice_with_established_patterns(self, continuity, sample_story_state):
        """Test voice checking uses established patterns."""
        established_patterns = {
            "Sarah Chen": DialoguePattern(
                character_name="Sarah Chen",
                vocabulary_level="formal",
                speech_patterns=["uses archaeological terminology"],
                typical_words=["precisely", "artifact", "excavation"],
                sentence_structure="complex",
            )
        }

        mock_result = ContinuityIssueList(issues=[])
        continuity.generate_structured = MagicMock(return_value=mock_result)

        continuity.check_character_voice(
            sample_story_state, "Chapter content...", established_patterns
        )

        call_args = continuity.generate_structured.call_args
        prompt = call_args[0][0]
        assert "ESTABLISHED SPEECH PATTERNS" in prompt
        assert "formal vocabulary" in prompt

    def test_check_character_voice_with_categorized_traits(self, continuity, sample_story_state):
        """Test voice checking includes categorized trait sections (core/flaw/quirk)."""
        from src.memory.templates import PersonalityTrait

        # Replace characters with categorized traits to hit flaw/quirk branches
        sample_story_state.characters = [
            Character(
                name="Sarah Chen",
                role="protagonist",
                description="Brilliant archaeologist",
                personality_traits=[
                    PersonalityTrait(trait="intelligent", category="core"),
                    PersonalityTrait(trait="brave", category="core"),
                    PersonalityTrait(trait="stubborn", category="flaw"),
                    PersonalityTrait(trait="hums when nervous", category="quirk"),
                ],
                goals=["Find the lost tomb"],
            ),
        ]

        mock_result = ContinuityIssueList(issues=[])
        continuity.generate_structured = MagicMock(return_value=mock_result)

        continuity.check_character_voice(sample_story_state, "Chapter content...")

        call_args = continuity.generate_structured.call_args
        prompt = call_args[0][0]
        assert "Core: intelligent, brave" in prompt
        assert "Flaws: stubborn" in prompt
        assert "Quirks: hums when nervous" in prompt

    def test_check_character_voice_no_characters(self, continuity, sample_story_state):
        """Test returns empty list when no characters."""
        sample_story_state.characters = []

        issues = continuity.check_character_voice(sample_story_state, "Content...")

        assert issues == []

    def test_check_character_voice_raises_on_exception(self, continuity, sample_story_state):
        """Test raises LLMGenerationError when generate_structured fails."""
        from src.utils.exceptions import LLMGenerationError

        continuity.generate_structured = MagicMock(side_effect=Exception("LLM error"))

        with pytest.raises(LLMGenerationError, match="Failed to check character voice"):
            continuity.check_character_voice(sample_story_state, "Content...")

    def test_check_chapter_includes_voice_check(self, continuity, sample_story_state):
        """Test check_chapter includes voice consistency by default."""
        # Mock both the main check and voice check
        main_result = ContinuityIssueList(
            issues=[
                ContinuityIssue(
                    severity="minor",
                    category="timeline",
                    description="Timeline issue",
                    location="Para 1",
                    suggestion="Fix timeline",
                )
            ]
        )
        voice_result = ContinuityIssueList(
            issues=[
                ContinuityIssue(
                    severity="moderate",
                    category="voice",
                    description="Voice issue",
                    location="Para 2",
                    suggestion="Fix voice",
                )
            ]
        )

        # The generate_structured method will be called twice
        continuity.generate_structured = MagicMock(side_effect=[main_result, voice_result])

        issues = continuity.check_chapter(sample_story_state, "Content...", chapter_number=1)

        # Should have both timeline and voice issues
        assert len(issues) == 2
        assert any(i.category == "timeline" for i in issues)
        assert any(i.category == "voice" for i in issues)

    def test_check_chapter_can_skip_voice_check(self, continuity, sample_story_state):
        """Test check_chapter can skip voice checking."""
        main_result = ContinuityIssueList(
            issues=[
                ContinuityIssue(
                    severity="minor",
                    category="timeline",
                    description="Timeline issue",
                    location="Para 1",
                    suggestion="Fix timeline",
                )
            ]
        )
        continuity.generate_structured = MagicMock(return_value=main_result)

        issues = continuity.check_chapter(
            sample_story_state, "Content...", chapter_number=1, check_voice=False
        )

        # Should only call generate_structured once (no voice check)
        assert continuity.generate_structured.call_count == 1
        assert len(issues) == 1
        assert issues[0].category == "timeline"

    def test_check_full_story_includes_voice_check(self, continuity, sample_story_state):
        """Test check_full_story includes voice consistency by default."""
        sample_story_state.chapters[0].content = "Chapter 1 with dialogue..."
        sample_story_state.chapters[1].content = "Chapter 2 with more dialogue..."

        # Mock responses: main check, pattern extraction, voice check
        main_result = ContinuityIssueList(
            issues=[
                ContinuityIssue(
                    severity="moderate",
                    category="plot_hole",
                    description="Plot issue",
                    location="Ch1",
                    suggestion="Fix plot",
                )
            ]
        )
        pattern_result = DialoguePatternList(
            patterns=[
                DialoguePattern(
                    character_name="Sarah Chen",
                    vocabulary_level="formal",
                    speech_patterns=[],
                    typical_words=[],
                    sentence_structure="complex",
                )
            ]
        )
        voice_result = ContinuityIssueList(
            issues=[
                ContinuityIssue(
                    severity="moderate",
                    category="voice",
                    description="Voice issue",
                    location="Ch2",
                    suggestion="Fix voice",
                )
            ]
        )

        continuity.generate_structured = MagicMock(
            side_effect=[main_result, pattern_result, voice_result]
        )

        issues = continuity.check_full_story(sample_story_state)

        # Should have both plot and voice issues
        assert len(issues) == 2
        assert any(i.category == "plot_hole" for i in issues)
        assert any(i.category == "voice" for i in issues)

    def test_extract_dialogue_patterns_no_characters(self, continuity, sample_story_state):
        """Test extract_dialogue_patterns returns empty dict when no characters."""
        sample_story_state.characters = []

        patterns = continuity.extract_dialogue_patterns(
            sample_story_state, "Chapter with dialogue..."
        )

        assert patterns == {}

    def test_extract_dialogue_patterns_raises_on_exception(self, continuity, sample_story_state):
        """Test extract_dialogue_patterns raises LLMGenerationError on exception."""
        from src.utils.exceptions import LLMGenerationError

        continuity.generate_structured = MagicMock(side_effect=Exception("LLM error"))

        with pytest.raises(LLMGenerationError, match="Failed to extract dialogue patterns"):
            continuity.extract_dialogue_patterns(sample_story_state, "Chapter with dialogue...")


class TestContinuityWrapperValidators:
    """Tests for continuity wrapper models that handle single object wrapping."""

    def test_continuity_issue_list_wraps_single_object(self):
        """Test ContinuityIssueList wraps a single issue in a list."""
        single_issue = {
            "severity": "critical",
            "category": "plot_hole",
            "description": "Character dies then reappears",
            "location": "Chapter 3",
            "suggestion": "Add resurrection explanation",
        }
        result = ContinuityIssueList.model_validate(single_issue)

        assert len(result.issues) == 1
        assert result.issues[0].severity == "critical"
        assert result.issues[0].category == "plot_hole"

    def test_continuity_issue_list_accepts_proper_format(self):
        """Test ContinuityIssueList accepts properly formatted input."""
        proper_data = {
            "issues": [
                {
                    "severity": "minor",
                    "category": "timeline",
                    "description": "Date inconsistency",
                    "location": "Chapter 1",
                    "suggestion": "Fix the date",
                }
            ]
        }
        result = ContinuityIssueList.model_validate(proper_data)

        assert len(result.issues) == 1

    def test_dialogue_pattern_list_wraps_single_object(self):
        """Test DialoguePatternList wraps a single pattern in a list."""
        single_pattern = {
            "character_name": "Alice",
            "vocabulary_level": "formal",
            "speech_patterns": ["tends to use long sentences"],
            "typical_words": ["indeed", "precisely"],
            "sentence_structure": "complex",
        }
        result = DialoguePatternList.model_validate(single_pattern)

        assert len(result.patterns) == 1
        assert result.patterns[0].character_name == "Alice"
        assert result.patterns[0].vocabulary_level == "formal"

    def test_dialogue_pattern_list_accepts_proper_format(self):
        """Test DialoguePatternList accepts properly formatted input."""
        proper_data = {
            "patterns": [
                {
                    "character_name": "Bob",
                    "vocabulary_level": "casual",
                    "sentence_structure": "simple",
                }
            ]
        }
        result = DialoguePatternList.model_validate(proper_data)

        assert len(result.patterns) == 1
        assert result.patterns[0].character_name == "Bob"

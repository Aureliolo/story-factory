"""Tests for text analytics utilities."""

from src.utils.text_analytics import (
    PacingMetrics,
    ReadabilityScores,
    analyze_pacing,
    calculate_readability,
    count_syllables,
    get_readability_interpretation,
)


class TestSyllableCounter:
    """Tests for syllable counting."""

    def test_single_syllable_words(self):
        """Test counting single syllable words."""
        assert count_syllables("cat") == 1
        assert count_syllables("dog") == 1
        assert count_syllables("run") == 1
        assert count_syllables("go") == 1

    def test_multi_syllable_words(self):
        """Test counting multi-syllable words."""
        assert count_syllables("running") == 2
        assert count_syllables("beautiful") == 3
        assert count_syllables("relationship") == 4
        assert count_syllables("understanding") == 4

    def test_silent_e(self):
        """Test handling of silent 'e' at end."""
        assert count_syllables("make") == 1
        assert count_syllables("time") == 1
        assert count_syllables("phone") == 1

    def test_edge_cases(self):
        """Test edge cases in syllable counting."""
        assert count_syllables("a") == 1
        assert count_syllables("I") == 1
        assert count_syllables("") == 1
        assert count_syllables("123") == 1  # No alphabetic characters


class TestReadabilityCalculation:
    """Tests for readability calculation."""

    def test_simple_text(self):
        """Test readability of simple text."""
        text = "The cat sat. The dog ran. They played together."
        scores = calculate_readability(text)

        assert isinstance(scores, ReadabilityScores)
        assert scores.word_count == 9
        assert scores.sentence_count == 3
        assert scores.avg_sentence_length == 3.0
        assert scores.flesch_reading_ease > 80  # Should be easy to read
        assert scores.flesch_kincaid_grade < 5  # Elementary level

    def test_complex_text(self):
        """Test readability of complex text."""
        text = """The implementation of sophisticated methodologies
        necessitates comprehensive understanding of multifaceted theoretical frameworks."""
        scores = calculate_readability(text)

        assert scores.word_count > 10
        assert scores.flesch_reading_ease < 50  # More difficult
        assert scores.flesch_kincaid_grade > 10  # Higher grade level

    def test_empty_text(self):
        """Test handling of empty text."""
        scores = calculate_readability("")
        assert scores.word_count == 0
        assert scores.sentence_count == 0
        assert scores.flesch_reading_ease == 0.0

    def test_no_sentences(self):
        """Test handling of text without sentence endings."""
        scores = calculate_readability("just some words")
        assert scores.word_count == 3
        # Text without punctuation is treated as one sentence
        assert scores.sentence_count == 1

    def test_no_words(self):
        """Test handling of text without alphabetic words."""
        scores = calculate_readability("123 456 789.")
        assert scores.word_count == 0

    def test_narrative_text(self):
        """Test readability of narrative text."""
        text = """
        Sarah walked through the ancient forest, her footsteps muffled by fallen leaves.
        The towering trees created a canopy overhead, filtering the sunlight into
        dappled patterns on the forest floor. She paused to listen to the birdsong,
        feeling at peace in this natural sanctuary.
        """
        scores = calculate_readability(text)

        assert scores.word_count > 30
        assert scores.sentence_count >= 3
        assert 0 <= scores.flesch_reading_ease <= 100
        assert scores.flesch_kincaid_grade >= 0


class TestPacingAnalysis:
    """Tests for pacing analysis."""

    def test_dialogue_heavy_text(self):
        """Test detection of dialogue-heavy content."""
        text = """
        "Hello," she said with a smile.

        "How are you today?" he asked cheerfully.

        "I'm doing well, thank you for asking," she replied.
        """
        metrics = analyze_pacing(text)

        assert isinstance(metrics, PacingMetrics)
        assert metrics.dialogue_percentage > 50
        assert metrics.total_word_count > 0

    def test_action_heavy_text(self):
        """Test detection of action-heavy content."""
        text = """
        He jumped over the fence and ran towards the building.
        She grabbed her weapon and charged forward.
        They fought fiercely, dodging and striking with precision.
        The warrior lunged at his opponent and threw a powerful punch.
        """
        metrics = analyze_pacing(text)

        assert metrics.action_percentage > 30  # Should detect action
        assert metrics.total_word_count > 0

    def test_narrative_heavy_text(self):
        """Test detection of narrative content."""
        text = """
        The old mansion stood on the hill, surrounded by overgrown gardens.
        Its windows were dark and empty, like the eyes of a forgotten soul.
        Time had weathered its walls, giving it a melancholic beauty.
        """
        metrics = analyze_pacing(text)

        assert metrics.narrative_percentage > 50
        assert metrics.total_word_count > 0

    def test_mixed_content(self):
        """Test analysis of mixed content."""
        text = """
        The detective examined the crime scene carefully.

        "What do you think happened here?" asked his partner.

        He rushed to the door and kicked it open. Inside, clues were scattered everywhere.

        "We need to document everything," he said, pulling out his camera.
        """
        metrics = analyze_pacing(text)

        # Should have a mix of all three types
        assert metrics.total_word_count > 0
        total_pct = (
            metrics.dialogue_percentage + metrics.action_percentage + metrics.narrative_percentage
        )
        assert 99 <= total_pct <= 101  # Should sum to ~100% (allow for rounding)

    def test_empty_text(self):
        """Test handling of empty text."""
        metrics = analyze_pacing("")
        assert metrics.total_word_count == 0
        assert metrics.dialogue_percentage == 0.0
        assert metrics.action_percentage == 0.0
        assert metrics.narrative_percentage == 0.0

    def test_percentages_sum_to_100(self):
        """Test that percentages sum to approximately 100."""
        text = """
        He walked down the street.
        "Hello there," she said.
        The car suddenly swerved and crashed into the barrier.
        """
        metrics = analyze_pacing(text)

        total = (
            metrics.dialogue_percentage + metrics.action_percentage + metrics.narrative_percentage
        )
        assert 99 <= total <= 101  # Allow for floating point rounding


class TestReadabilityInterpretation:
    """Tests for readability interpretation."""

    def test_very_easy(self):
        """Test very easy interpretation."""
        assert "Very Easy" in get_readability_interpretation(95)

    def test_easy(self):
        """Test easy interpretation."""
        assert "Easy" in get_readability_interpretation(85)

    def test_fairly_easy(self):
        """Test fairly easy interpretation."""
        assert "Fairly Easy" in get_readability_interpretation(75)

    def test_standard(self):
        """Test standard interpretation."""
        assert "Standard" in get_readability_interpretation(65)

    def test_fairly_difficult(self):
        """Test fairly difficult interpretation."""
        assert "Fairly Difficult" in get_readability_interpretation(55)

    def test_difficult(self):
        """Test difficult interpretation."""
        assert "Difficult" in get_readability_interpretation(40)

    def test_very_difficult(self):
        """Test very difficult interpretation."""
        assert "Very Difficult" in get_readability_interpretation(20)

    def test_boundary_values(self):
        """Test boundary values."""
        assert get_readability_interpretation(0)
        assert get_readability_interpretation(50)
        assert get_readability_interpretation(100)


class TestEdgeCases:
    """Tests for edge cases in text analytics."""

    def test_count_syllables_non_alphabetic_only(self):
        """Test syllable counting with only non-alphabetic characters."""
        # Should return 1 for words that reduce to empty string
        assert count_syllables("!@#$%") == 1
        assert count_syllables("12345") == 1
        assert count_syllables("---") == 1

    def test_readability_no_sentences(self):
        """Test readability with text that has no valid sentence endings."""
        # Text with just whitespace and punctuation but no sentence-ending marks
        scores = calculate_readability("   ")
        assert scores.sentence_count == 0
        assert scores.word_count == 0
        assert scores.flesch_reading_ease == 0.0

    def test_pacing_text_with_empty_paragraphs(self):
        """Test pacing analysis with empty paragraphs in text."""
        # Text that splits into some empty paragraphs
        text = "\n\n\n\n"  # Just newlines, no actual content
        metrics = analyze_pacing(text)
        assert metrics.total_word_count == 0

    def test_pacing_paragraph_with_only_whitespace(self):
        """Test pacing with paragraphs that are only whitespace."""
        # Paragraph that is just whitespace (splits to empty words list)
        text = "Hello world\n\n   \n\nGoodbye"
        metrics = analyze_pacing(text)
        # Should handle gracefully (whitespace paragraph skipped)
        assert metrics.total_word_count > 0

    def test_pacing_single_line_text(self):
        """Test pacing with text that doesn't split into paragraphs."""
        # Single line text without paragraph breaks
        text = "Just a single line of text"
        metrics = analyze_pacing(text)
        assert metrics.total_word_count > 0


class TestUncoveredEdgeCases:
    """Tests for previously uncovered edge cases in text analytics."""

    def test_readability_text_with_only_sentence_punctuation(self):
        """Test readability with text that has only sentence-ending punctuation."""
        # Lines 103-104: sentence_count == 0 with non-empty text after stripped() passes
        # Text with only sentence-ending punctuation produces empty segments after split
        text = "...!!!???"
        scores = calculate_readability(text)
        assert scores.sentence_count == 0
        assert scores.word_count == 0
        assert scores.flesch_reading_ease == 0.0

    def test_pacing_with_single_paragraph(self):
        """Test pacing with single paragraph (no double newlines)."""
        text = "This is a simple narrative sentence without action"
        metrics = analyze_pacing(text)
        # Should analyze the text as narrative
        assert metrics.total_word_count > 0
        assert metrics.narrative_percentage > 0

    def test_pacing_with_multiple_paragraphs(self):
        """Test pacing correctly processes multiple paragraphs."""
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird one too."
        metrics = analyze_pacing(text)
        # Should process all paragraphs
        assert metrics.total_word_count > 0

    def test_pacing_empty_text_returns_zeros(self):
        """Test pacing returns zeros for empty/whitespace text."""
        # Empty text hits the early return at line 169-179
        text = "   \n\n   "
        metrics = analyze_pacing(text)
        assert metrics.total_word_count == 0
        assert metrics.dialogue_percentage == 0.0
        assert metrics.action_percentage == 0.0
        assert metrics.narrative_percentage == 0.0

    def test_readability_text_with_sentences_but_no_alphabetic_words(self):
        """Test readability with sentences that have no alphabetic words."""
        # Tests the word_count == 0 branch with sentence_count > 0
        text = "123 456. 789 012."
        scores = calculate_readability(text)
        assert scores.word_count == 0
        assert scores.sentence_count == 2
        assert scores.flesch_reading_ease == 0.0

    def test_pacing_with_mixed_content_paragraphs(self):
        """Test pacing handles paragraphs with mixed content types."""
        # Paragraphs with normal text and punctuation-only content
        text = "Normal text here\n\n!@#$%^&*()\n\nMore normal text here"
        metrics = analyze_pacing(text)
        # Should process all paragraphs
        assert metrics.total_word_count > 0
        # Total word count includes words from all paragraphs
        # (punctuation paragraph has 1 "word" which is the punctuation string)

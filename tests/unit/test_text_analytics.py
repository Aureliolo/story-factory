"""Tests for text analytics utilities."""

from utils.text_analytics import (
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

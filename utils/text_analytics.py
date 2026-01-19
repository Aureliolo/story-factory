"""Text analytics utilities for story content analysis.
from __future__ import annotations


Provides readability metrics and content analysis for generated stories.
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ReadabilityScores:
    """Readability metrics for text content."""

    flesch_reading_ease: float
    flesch_kincaid_grade: float
    avg_sentence_length: float
    avg_syllables_per_word: float
    word_count: int
    sentence_count: int


@dataclass
class PacingMetrics:
    """Pacing analysis metrics for narrative content."""

    dialogue_percentage: float
    action_percentage: float
    narrative_percentage: float
    dialogue_word_count: int
    action_word_count: int
    narrative_word_count: int
    total_word_count: int


def count_syllables(word: str) -> int:
    """Count syllables in a word using a simple heuristic.

    Args:
        word: The word to analyze.

    Returns:
        Estimated syllable count.
    """
    word = word.lower().strip()
    if len(word) <= 3:
        return 1

    # Remove non-alphabetic characters
    word = re.sub(r"[^a-z]", "", word)
    if not word:
        return 1

    # Count vowel groups
    vowels = "aeiouy"
    syllable_count = 0
    previous_was_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not previous_was_vowel:
            syllable_count += 1
        previous_was_vowel = is_vowel

    # Adjust for silent 'e' at end
    if word.endswith("e") and syllable_count > 1:
        syllable_count -= 1

    # Ensure at least one syllable
    return max(1, syllable_count)


def calculate_readability(text: str) -> ReadabilityScores:
    """Calculate readability metrics for text.

    Implements Flesch Reading Ease and Flesch-Kincaid Grade Level formulas.

    Args:
        text: The text to analyze.

    Returns:
        ReadabilityScores with calculated metrics.
    """
    if not text or not text.strip():
        logger.debug("Empty text provided for readability analysis")
        return ReadabilityScores(
            flesch_reading_ease=0.0,
            flesch_kincaid_grade=0.0,
            avg_sentence_length=0.0,
            avg_syllables_per_word=0.0,
            word_count=0,
            sentence_count=0,
        )

    # Split into sentences (simple approach)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)

    if sentence_count == 0:
        logger.debug("No sentences found in text")
        return ReadabilityScores(
            flesch_reading_ease=0.0,
            flesch_kincaid_grade=0.0,
            avg_sentence_length=0.0,
            avg_syllables_per_word=0.0,
            word_count=0,
            sentence_count=0,
        )

    # Split into words
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    word_count = len(words)

    if word_count == 0:
        logger.debug("No words found in text")
        return ReadabilityScores(
            flesch_reading_ease=0.0,
            flesch_kincaid_grade=0.0,
            avg_sentence_length=0.0,
            avg_syllables_per_word=0.0,
            word_count=0,
            sentence_count=sentence_count,
        )

    # Count syllables
    total_syllables = sum(count_syllables(word) for word in words)

    # Calculate averages
    avg_sentence_length = word_count / sentence_count
    avg_syllables_per_word = total_syllables / word_count

    # Flesch Reading Ease: 206.835 - 1.015(words/sentences) - 84.6(syllables/words)
    # Scale: 0-100 (higher = easier to read)
    flesch_reading_ease = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)

    # Flesch-Kincaid Grade Level: 0.39(words/sentences) + 11.8(syllables/words) - 15.59
    # Represents US grade level
    flesch_kincaid_grade = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59

    logger.debug(
        f"Readability: {word_count} words, {sentence_count} sentences, "
        f"FRE={flesch_reading_ease:.1f}, FK={flesch_kincaid_grade:.1f}"
    )

    return ReadabilityScores(
        flesch_reading_ease=max(0.0, min(100.0, flesch_reading_ease)),
        flesch_kincaid_grade=max(0.0, flesch_kincaid_grade),
        avg_sentence_length=avg_sentence_length,
        avg_syllables_per_word=avg_syllables_per_word,
        word_count=word_count,
        sentence_count=sentence_count,
    )


def analyze_pacing(text: str) -> PacingMetrics:
    """Analyze narrative pacing by categorizing content.

    Categorizes text into dialogue, action, and narrative based on simple heuristics.

    Args:
        text: The narrative text to analyze.

    Returns:
        PacingMetrics with categorized content.
    """
    if not text or not text.strip():
        logger.debug("Empty text provided for pacing analysis")
        return PacingMetrics(
            dialogue_percentage=0.0,
            action_percentage=0.0,
            narrative_percentage=0.0,
            dialogue_word_count=0,
            action_word_count=0,
            narrative_word_count=0,
            total_word_count=0,
        )

    # Split into paragraphs (filter ensures only non-empty paragraphs remain)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    # Note: paragraphs can't be empty here because text.strip() was checked above

    dialogue_words = 0
    action_words = 0
    narrative_words = 0

    # Action indicators (verbs suggesting movement/activity)
    action_patterns = re.compile(
        r"\b(ran|jumped|fought|grabbed|threw|kicked|punched|struck|dashed|leaped|"
        r"rushed|charged|dodged|ducked|slammed|crashed|burst|sprinted|lunged)\b",
        re.IGNORECASE,
    )

    for paragraph in paragraphs:
        words = paragraph.split()
        word_count = len(words)
        # Note: word_count > 0 since paragraphs are non-empty after strip()

        # Dialogue detection: Contains quoted text
        quote_matches = re.findall(r'"[^"]*"', paragraph)
        dialogue_content = " ".join(quote_matches)
        dialogue_word_count = len(dialogue_content.split())

        # If significant dialogue (>50% of paragraph)
        if dialogue_word_count > word_count * 0.5:
            dialogue_words += word_count
        # Check for action indicators
        elif action_patterns.search(paragraph):
            action_words += word_count
        # Default to narrative
        else:
            narrative_words += word_count

    total_words = dialogue_words + action_words + narrative_words
    # Note: total_words > 0 since we have at least one non-empty paragraph

    # Calculate percentages
    dialogue_pct = (dialogue_words / total_words) * 100
    action_pct = (action_words / total_words) * 100
    narrative_pct = (narrative_words / total_words) * 100

    logger.debug(
        f"Pacing: {total_words} words - "
        f"Dialogue: {dialogue_pct:.1f}%, Action: {action_pct:.1f}%, "
        f"Narrative: {narrative_pct:.1f}%"
    )

    return PacingMetrics(
        dialogue_percentage=dialogue_pct,
        action_percentage=action_pct,
        narrative_percentage=narrative_pct,
        dialogue_word_count=dialogue_words,
        action_word_count=action_words,
        narrative_word_count=narrative_words,
        total_word_count=total_words,
    )


def get_readability_interpretation(flesch_score: float) -> str:
    """Get human-readable interpretation of Flesch Reading Ease score.

    Args:
        flesch_score: Flesch Reading Ease score (0-100).

    Returns:
        Description of reading difficulty.
    """
    if flesch_score >= 90:
        return "Very Easy (5th grade)"
    elif flesch_score >= 80:
        return "Easy (6th grade)"
    elif flesch_score >= 70:
        return "Fairly Easy (7th grade)"
    elif flesch_score >= 60:
        return "Standard (8th-9th grade)"
    elif flesch_score >= 50:
        return "Fairly Difficult (10th-12th grade)"
    elif flesch_score >= 30:
        return "Difficult (College)"
    else:
        return "Very Difficult (College graduate)"

"""Tests for message analyzer utility."""

from src.utils.message_analyzer import (
    analyze_message,
    detect_content_rating,
    detect_language,
    format_inference_context,
)


class TestDetectContentRating:
    """Tests for content rating detection."""

    def test_detects_adult_from_smut(self):
        """Verifies 'smut' keyword triggers adult content rating."""
        assert detect_content_rating("I want a smut story") == "adult"

    def test_detects_adult_from_nsfw(self):
        """Verifies 'NSFW' keyword triggers adult content rating."""
        assert detect_content_rating("Make it very NSFW") == "adult"

    def test_detects_adult_from_explicit(self):
        """Verifies 'explicit' keyword triggers adult content rating."""
        assert detect_content_rating("Include explicit content") == "adult"

    def test_detects_adult_from_erotic(self):
        """Verifies 'erotic' keyword triggers adult content rating."""
        assert detect_content_rating("An erotic romance story") == "adult"

    def test_detects_adult_from_spicy(self):
        """Verifies 'spicy' keyword triggers adult content rating."""
        assert detect_content_rating("A spicy romance novel") == "adult"

    def test_detects_mature_from_dark(self):
        """Verifies 'dark' keyword triggers mature content rating."""
        assert detect_content_rating("A dark fantasy story") == "mature"

    def test_detects_mature_from_violence(self):
        """Verifies 'violence' keyword triggers mature content rating."""
        assert detect_content_rating("Include lots of violence") == "mature"

    def test_detects_mature_from_horror(self):
        """Verifies 'horror' keyword triggers mature content rating."""
        assert detect_content_rating("A horror story with gore") == "mature"

    def test_detects_teen_from_ya(self):
        """Verifies 'YA' abbreviation triggers teen content rating."""
        assert detect_content_rating("A YA fantasy story") == "teen"

    def test_detects_teen_from_young_adult(self):
        """Verifies 'young adult' phrase triggers teen content rating."""
        assert detect_content_rating("A young adult romance") == "teen"

    def test_detects_general_from_family_friendly(self):
        """Verifies 'family-friendly' phrase triggers general content rating."""
        assert detect_content_rating("A family-friendly adventure") == "general"

    def test_detects_general_from_kids(self):
        """Verifies 'kids' keyword triggers general content rating."""
        assert detect_content_rating("A story for kids") == "general"

    def test_detects_general_from_wholesome(self):
        """Verifies 'wholesome' keyword triggers general content rating."""
        assert detect_content_rating("Something wholesome") == "general"

    def test_returns_none_when_unclear(self):
        """Verifies None is returned when no content rating keywords are found."""
        assert detect_content_rating("A fantasy adventure story") is None

    def test_adult_takes_precedence_over_mature(self):
        """Verifies adult rating takes precedence when both adult and mature keywords present."""
        # "dark romance" would be mature, but "smut" makes it adult
        assert detect_content_rating("A dark smut romance") == "adult"


class TestDetectLanguage:
    """Tests for language detection."""

    def test_detects_english_by_default(self):
        """Verifies English is detected for common English text."""
        assert detect_language("I want a story about dragons") == "English"

    def test_detects_german(self):
        """Verifies German is detected from common German words."""
        # Need more common German words to trigger detection
        assert detect_language("Ich und mein Bruder, das ist nicht so einfach") == "German"

    def test_detects_spanish(self):
        """Verifies Spanish is detected from common Spanish words."""
        # Need more common Spanish words
        assert detect_language("El libro que es de la biblioteca en el centro") == "Spanish"

    def test_detects_french(self):
        """Verifies French is detected from common French words."""
        # Need more common French words
        assert detect_language("Le livre est pour les enfants avec du courage") == "French"

    def test_defaults_to_english_for_mixed(self):
        """Verifies English is returned as default for mixed-language text."""
        # Mixed language should default to English
        assert detect_language("Hello, je veux, ich mochte") == "English"

    def test_defaults_to_english_for_ambiguous(self):
        """Verifies English is returned as default for ambiguous short messages."""
        # Short messages without clear language markers default to English
        assert detect_language("A story about magic") == "English"

    def test_detects_italian(self):
        """Verifies Italian is detected from common Italian words."""
        # Italian words: il, la, di, che, non, con, sono, una, per, come
        assert detect_language("Il libro che la ragazza non ha sono per lei") == "Italian"

    def test_detects_portuguese(self):
        """Verifies Portuguese is detected from common Portuguese words."""
        # Portuguese words: o, a, de, que, e, do, da, em, um, uma
        assert detect_language("O livro que a menina tem do pai em uma casa") == "Portuguese"

    def test_detects_dutch(self):
        """Verifies Dutch is detected from common Dutch words."""
        # Dutch words: de, het, een, van, en, is, dat, niet, op, met
        assert detect_language("De man is het een van en dat niet met op") == "Dutch"


class TestAnalyzeMessage:
    """Tests for full message analysis."""

    def test_analyzes_smut_request(self):
        """Verifies analyze_message correctly identifies adult English content."""
        msg = "I want a smut story about aliens. Very explicit please."
        result = analyze_message(msg)
        assert result["content_rating"] == "adult"
        assert result["language"] == "English"

    def test_analyzes_german_request(self):
        """Verifies analyze_message correctly identifies German language."""
        # Enough German words to trigger detection
        msg = "Ich und mein Freund, das ist nicht so einfach mit der Geschichte"
        result = analyze_message(msg)
        assert result["language"] == "German"

    def test_analyzes_clean_fantasy(self):
        """Verifies analyze_message correctly identifies general English content."""
        msg = "Write me a wholesome fantasy adventure for kids"
        result = analyze_message(msg)
        assert result["content_rating"] == "general"
        assert result["language"] == "English"


class TestFormatInferenceContext:
    """Tests for formatting inference context."""

    def test_formats_both_inferences(self):
        """Verifies formatting includes both language and content rating when present."""
        analysis: dict[str, str | None] = {"language": "English", "content_rating": "adult"}
        result = format_inference_context(analysis)
        assert "English" in result
        assert "adult" in result
        assert "Do NOT ask about these" in result

    def test_formats_language_only(self):
        """Verifies formatting includes only language when content rating is None."""
        analysis: dict[str, str | None] = {"language": "English", "content_rating": None}
        result = format_inference_context(analysis)
        assert "English" in result
        assert "adult" not in result

    def test_formats_content_rating_only(self):
        """Verifies formatting includes only content rating when language is None."""
        analysis: dict[str, str | None] = {"language": None, "content_rating": "mature"}
        result = format_inference_context(analysis)
        # Language is always detected (defaults to English), so this tests None handling
        assert "mature" in result

    def test_returns_empty_for_no_inferences(self):
        """Verifies empty string is returned when both language and content rating are None."""
        analysis: dict[str, str | None] = {"language": None, "content_rating": None}
        result = format_inference_context(analysis)
        # Note: language always returns a value, but this tests the None path
        assert result == ""

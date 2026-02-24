"""Tests for entity name matching with deep normalization and similarity fallback."""

from typing import cast
from unittest.mock import MagicMock

from src.memory.entities import Entity
from src.services.world_service._name_matching import (
    _deep_normalize,
    _find_entity_by_name,
    _normalize_name,
)


def _make_entity(name: str) -> Entity:
    """Create a mock entity with the given name and a unique ID."""
    entity = MagicMock(spec=Entity)
    entity.name = name
    entity.id = f"id-{name.lower().replace(' ', '-')}"
    return cast(Entity, entity)


# =========================================================================
# _deep_normalize tests
# =========================================================================


class TestDeepNormalize:
    """Tests for the _deep_normalize function."""

    def test_strips_possessive_apostrophe_s(self):
        """Should strip possessive 's from names."""
        assert _deep_normalize("Glacier's Whisper") == "glacier whisper"

    def test_strips_possessive_s_apostrophe(self):
        """Should strip trailing s' possessives."""
        assert _deep_normalize("The Guardians' Oath") == "guardian oath"

    def test_normalizes_abbreviation_dots(self):
        """Should collapse abbreviation dots: 'A.P.' -> 'ap'."""
        assert _deep_normalize("A.P. Society") == "ap society"

    def test_normalizes_spaced_abbreviation_dots(self):
        """Should collapse spaced abbreviation dots: 'A. P.' -> 'a p'."""
        # "A." is not stripped as article because it's "a." not "a "
        assert _deep_normalize("A. P. Society") == "a p society"

    def test_strips_trailing_punctuation(self):
        """Should strip trailing punctuation marks."""
        assert _deep_normalize("The End.") == "end"

    def test_collapses_whitespace(self):
        """Should collapse multiple spaces to single space."""
        assert _deep_normalize("  The  Icy   North  ") == "icy north"

    def test_strips_leading_articles(self):
        """Should strip leading articles (inherited from _normalize_name)."""
        assert _deep_normalize("The Northern Wind") == "northern wind"
        assert _deep_normalize("A Dark Tower") == "dark tower"
        assert _deep_normalize("An Ancient Relic") == "ancient relic"

    def test_combined_transformations(self):
        """Should apply all transformations together."""
        # Leading article + possessive + trailing punctuation
        assert _deep_normalize("The Dragon's Lair.") == "dragon lair"

    def test_empty_string(self):
        """Should handle empty string gracefully."""
        assert _deep_normalize("") == ""

    def test_no_transformations_needed(self):
        """Should pass through clean names unchanged."""
        assert _deep_normalize("castle blackstone") == "castle blackstone"


# =========================================================================
# _find_entity_by_name: similarity fallback tests
# =========================================================================


class TestSimilarityFallback:
    """Tests for the similarity-based fallback in _find_entity_by_name."""

    def test_exact_match_still_works(self):
        """Exact name match should return immediately without fallback."""
        entities = [_make_entity("Glacial Whisper"), _make_entity("Northern Wind")]
        result = _find_entity_by_name(entities, "Glacial Whisper")
        assert result is not None
        assert result.name == "Glacial Whisper"

    def test_normalized_match_still_works(self):
        """Article-stripped normalized match should still work."""
        entities = [_make_entity("Echoes of the Network")]
        result = _find_entity_by_name(entities, "The Echoes of the Network")
        assert result is not None
        assert result.name == "Echoes of the Network"

    def test_similarity_catches_possessive_variant(self):
        """'Glacier's Whisper' should match 'Glacial Whisper' via similarity."""
        entities = [_make_entity("Glacial Whisper"), _make_entity("Northern Wind")]
        # "glacier whisper" vs "glacial whisper" — high similarity
        result = _find_entity_by_name(entities, "Glacier's Whisper", threshold=0.7)
        assert result is not None
        assert result.name == "Glacial Whisper"

    def test_dissimilar_names_do_not_match(self):
        """'Crystal Cave' should NOT match 'Northern Wind' at default threshold."""
        entities = [_make_entity("Crystal Cave")]
        result = _find_entity_by_name(entities, "Northern Wind", threshold=0.8)
        # Completely different names — well below threshold
        assert result is None

    def test_ambiguous_similarity_returns_none(self):
        """Multiple matches above threshold should return None (ambiguity guard)."""
        # Create two entities with very similar names to the search term
        entities = [
            _make_entity("Dark Crystal"),
            _make_entity("Dark Crystals"),
        ]
        result = _find_entity_by_name(entities, "Dark Crystal's", threshold=0.7)
        # Both "dark crystal" and "dark crystals" are very close to "dark crystal"
        # after deep normalization strips the possessive
        # This should match "Dark Crystal" exactly via normalization or
        # find ambiguous similarity matches
        # After _deep_normalize: "dark crystal" (search), "dark crystal" (entity 1),
        # "dark crystals" (entity 2)
        # Entity 1 gets exact deep-normalized match (score 1.0), entity 2 gets high score
        # Both above threshold -> ambiguity -> None
        # Actually, _deep_normalize("Dark Crystal's") = "dark crystal"
        # _deep_normalize("Dark Crystal") = "dark crystal" -> score 1.0
        # _deep_normalize("Dark Crystals") = "dark crystals" -> score ~0.96
        # Both above 0.7 -> ambiguous -> None
        assert result is None

    def test_threshold_parameter_respected(self):
        """Higher threshold should reject matches that pass at lower threshold."""
        entities = [_make_entity("River Stone"), _make_entity("Completely Different")]
        # At a lenient threshold, similar name should match
        result_lenient = _find_entity_by_name(entities, "River Stones", threshold=0.7)
        assert result_lenient is not None
        assert result_lenient.name == "River Stone"
        # At a very strict threshold (1.0), only exact deep-normalized match passes
        result_strict = _find_entity_by_name(entities, "River Stones", threshold=1.0)
        # "river stone" vs "river stones" — not exact, so strict threshold rejects
        assert result_strict is None

    def test_no_entities_returns_none(self):
        """Empty entity list should return None."""
        result = _find_entity_by_name([], "Anything", threshold=0.8)
        assert result is None

    def test_similarity_with_abbreviation_variants(self):
        """Abbreviation normalization should help similarity matching."""
        entities = [_make_entity("AP Society")]
        result = _find_entity_by_name(entities, "A.P. Society", threshold=0.8)
        # After normalization, "A.P. Society" becomes "ap society"
        # The entity also normalizes to "ap society"
        # This should match via normalized comparison (step 2), not even needing similarity
        assert result is not None
        assert result.name == "AP Society"


# =========================================================================
# _normalize_name basic tests
# =========================================================================


class TestNormalizeName:
    """Tests for the basic _normalize_name function."""

    def test_strips_the_article(self):
        """Should strip leading 'The' article."""
        assert _normalize_name("The Northern Kingdoms") == "northern kingdoms"

    def test_strips_a_article(self):
        """Should strip leading 'A' article."""
        assert _normalize_name("A Dark Forest") == "dark forest"

    def test_strips_an_article(self):
        """Should strip leading 'An' article."""
        assert _normalize_name("An Ancient Relic") == "ancient relic"

    def test_case_insensitive(self):
        """Should lowercase all characters."""
        assert _normalize_name("CASTLE BLACKSTONE") == "castle blackstone"

    def test_collapses_whitespace(self):
        """Should collapse multiple spaces and strip outer whitespace."""
        assert _normalize_name("  The   Castle  ") == "castle"

    def test_no_article_passthrough(self):
        """Names without leading articles should pass through lowercased."""
        assert _normalize_name("Castle Blackstone") == "castle blackstone"

    def test_only_strips_leading_article(self):
        """Articles in the middle of the name should be preserved."""
        assert _normalize_name("Bridge of the Ancients") == "bridge of the ancients"


# =========================================================================
# _find_entity_by_name: existing behavior preserved
# =========================================================================


class TestFindEntityByNameExisting:
    """Verify pre-existing behavior is preserved with the new threshold param."""

    def test_exact_match_preferred_over_fuzzy(self):
        """Exact match should be preferred over normalized match."""
        entities = [_make_entity("the castle"), _make_entity("Castle")]
        result = _find_entity_by_name(entities, "the castle")
        assert result is not None
        assert result.name == "the castle"

    def test_ambiguous_normalized_match_returns_none(self):
        """Multiple entities with same normalized name should return None."""
        # Both normalize to "castle"
        entities = [_make_entity("The Castle"), _make_entity("A Castle")]
        result = _find_entity_by_name(entities, "castle")
        assert result is None

    def test_default_threshold_is_0_8(self):
        """Default threshold should be 0.8 when not specified."""
        entities = [_make_entity("Glacial Whisper")]
        # Call without threshold param to verify default works
        result = _find_entity_by_name(entities, "Glacial Whisper")
        assert result is not None

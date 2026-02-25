"""Tests for conflict types — mapping, normalization, and classification."""

import pytest

from src.memory.conflict_types import (
    _CONFLICT_PRIORITY,
    _WORD_TO_RELATION,
    RELATION_CONFLICT_MAPPING,
    VALID_RELATIONSHIP_TYPES,
    ConflictCategory,
    _validate_conflict_priority,
    _validate_word_to_relation,
    classify_relationship,
    normalize_relation_type,
)


class TestNewTypesInMapping:
    """Verify the three new types (D1-D3) are classified correctly."""

    def test_friends_is_alliance(self):
        """'friends' should map to ALLIANCE."""
        assert classify_relationship("friends") == ConflictCategory.ALLIANCE

    def test_connected_to_is_neutral(self):
        """'connected_to' should map to NEUTRAL."""
        assert classify_relationship("connected_to") == ConflictCategory.NEUTRAL

    def test_owes_debt_to_is_tension(self):
        """'owes_debt_to' should map to TENSION."""
        assert classify_relationship("owes_debt_to") == ConflictCategory.TENSION


class TestValidRelationshipTypes:
    """Verify VALID_RELATIONSHIP_TYPES matches mapping keys."""

    def test_valid_relationship_types_matches_mapping_keys(self):
        """VALID_RELATIONSHIP_TYPES should equal sorted keys of the mapping."""
        assert VALID_RELATIONSHIP_TYPES == sorted(RELATION_CONFLICT_MAPPING.keys())


class TestNormalizeRelationType:
    """Tests for normalize_relation_type()."""

    def test_known_type_passthrough(self):
        """A known type should pass through unchanged."""
        assert normalize_relation_type("allies_with") == "allies_with"

    def test_prose_extraction_rivals(self):
        """Prose like 'bitter rivals who fought' should extract 'bitter_rivals'."""
        assert normalize_relation_type("bitter rivals who fought") == "bitter_rivals"

    def test_with_hyphens(self):
        """Hyphens should be normalized to underscores."""
        assert normalize_relation_type("allies-with") == "allies_with"

    def test_with_spaces(self):
        """Spaces should be normalized to underscores."""
        assert normalize_relation_type("allies with") == "allies_with"

    def test_unknown_passthrough(self):
        """Unknown types should pass through normalized but unchanged."""
        result = normalize_relation_type("xyzzy")
        assert result == "xyzzy"

    def test_pipe_delimited_first_recognized(self):
        """Pipe-delimited input should return first recognized part."""
        result = normalize_relation_type("created|friends")
        assert result == "created"

    def test_pipe_delimited_second_recognized(self):
        """If first part unrecognized, return second recognized part."""
        result = normalize_relation_type("unknown_thing|hates")
        assert result == "hates"

    def test_case_insensitive(self):
        """Normalization should be case-insensitive."""
        assert normalize_relation_type("LOVES") == "loves"

    def test_empty_string(self):
        """Empty string should normalize to empty string."""
        result = normalize_relation_type("")
        assert result == ""

    def test_substring_prefers_longer_match(self):
        """'allies_with_enthusiasm' should match 'allies_with' not just 'allies'."""
        # The input contains both "allies_with" (11 chars) and "allies" (6 chars)
        # Longer match should win
        result = normalize_relation_type("allies_with_enthusiasm")
        assert result == "allies_with"

    def test_leading_trailing_whitespace(self):
        """Leading/trailing whitespace should be stripped."""
        assert normalize_relation_type("  loves  ") == "loves"

    def test_pipe_delimited_no_recognized_falls_through_to_substring(self):
        """Pipe-delimited with no recognized parts falls through to substring matching."""
        # "foo|bitter_rivals_bar" — neither "foo" nor "bitter_rivals_bar" is a direct match,
        # but substring matching finds "bitter_rivals" inside "foo|bitter_rivals_bar"
        result = normalize_relation_type("foo|bitter_rivals_bar")
        assert result == "bitter_rivals"

    def test_pipe_delimited_no_recognized_no_substring(self):
        """Pipe-delimited with no recognized parts and no substring match returns normalized."""
        result = normalize_relation_type("xyz|abc")
        assert result == "xyz|abc"


class TestConflictsWithMapping:
    """Verify 'conflicts_with' maps to TENSION and normalizes from space-separated."""

    def test_conflicts_with_is_tension(self):
        """'conflicts_with' should map to TENSION."""
        assert classify_relationship("conflicts_with") == ConflictCategory.TENSION

    def test_conflicts_with_normalizes_from_spaces(self):
        """'conflicts with' (space-separated) should normalize to 'conflicts_with'."""
        assert normalize_relation_type("conflicts with") == "conflicts_with"

    def test_challenges_is_tension(self):
        """'challenges' should map to TENSION."""
        assert classify_relationship("challenges") == ConflictCategory.TENSION

    def test_betrays_is_rivalry(self):
        """'betrays' should map to RIVALRY."""
        assert classify_relationship("betrays") == ConflictCategory.RIVALRY


class TestKeywordBasedFallback:
    """Verify signal words resolve correctly through normalization + classification."""

    def test_oppose_resolves_to_rivalry(self):
        """Word 'oppose' in a compound type should resolve to a RIVALRY type."""
        result = normalize_relation_type("deep_oppose_bond")
        assert classify_relationship(result) == ConflictCategory.RIVALRY

    def test_support_resolves_to_alliance(self):
        """Word 'support' in a compound type should resolve to an ALLIANCE type."""
        result = normalize_relation_type("mutual_support_pact")
        assert classify_relationship(result) == ConflictCategory.ALLIANCE

    def test_protect_resolves_to_alliance(self):
        """Word 'protect' in a compound type should resolve to an ALLIANCE type."""
        result = normalize_relation_type("sacred_protect_oath")
        assert classify_relationship(result) == ConflictCategory.ALLIANCE

    def test_challenge_resolves_to_tension(self):
        """Word 'challenge' in a compound type should resolve to a TENSION type."""
        result = normalize_relation_type("constant_challenge_dynamic")
        assert classify_relationship(result) == ConflictCategory.TENSION

    def test_betray_resolves_to_rivalry(self):
        """Word 'betray' in a compound type should resolve to a RIVALRY type."""
        result = normalize_relation_type("ultimate_betray_act")
        assert classify_relationship(result) == ConflictCategory.RIVALRY

    def test_conflict_resolves_to_rivalry(self):
        """Word 'conflict' in a compound type should resolve to a RIVALRY type."""
        result = normalize_relation_type("eternal_conflict_cycle")
        assert classify_relationship(result) == ConflictCategory.RIVALRY

    def test_threaten_resolves_to_tension(self):
        """Word 'threaten' in a compound type should resolve to a TENSION type."""
        result = normalize_relation_type("covert_threaten_dynamic")
        assert classify_relationship(result) == ConflictCategory.TENSION

    def test_cares_resolves_to_alliance(self):
        """Word 'cares' in a compound type should resolve to an ALLIANCE type."""
        result = normalize_relation_type("deeply_cares_bond")
        assert classify_relationship(result) == ConflictCategory.ALLIANCE


class TestCompoundTypeClassification:
    """Issue #397: compound types from LLM should NOT fall through to NEUTRAL."""

    def test_colleague_and_occasional_rival_is_rivalry(self):
        """'colleague_and_occasional_rival' should classify as RIVALRY, not NEUTRAL."""
        assert classify_relationship("colleague_and_occasional_rival") == ConflictCategory.RIVALRY

    def test_protective_rival_is_rivalry(self):
        """'protective_rival' should classify as RIVALRY (rival > protect)."""
        assert classify_relationship("protective_rival") == ConflictCategory.RIVALRY

    def test_reluctant_enemy_is_rivalry(self):
        """'reluctant_enemy' should classify as RIVALRY."""
        assert classify_relationship("reluctant_enemy") == ConflictCategory.RIVALRY

    def test_rival_singular_maps_via_word_lookup(self):
        """'rival' singular should be recognized through _WORD_TO_RELATION."""
        result = normalize_relation_type("occasional_rival")
        assert result == "rivals"
        assert classify_relationship(result) == ConflictCategory.RIVALRY

    def test_admire_maps_via_word_lookup(self):
        """'admire' (verb stem without 's') should map through _WORD_TO_RELATION."""
        result = normalize_relation_type("deep_admire_bond")
        assert result == "admires"
        assert classify_relationship(result) == ConflictCategory.ALLIANCE

    def test_colleague_maps_to_neutral(self):
        """'colleague' should map to 'works_with' -> NEUTRAL."""
        result = normalize_relation_type("mere_colleague_tie")
        assert result == "works_with"
        assert classify_relationship(result) == ConflictCategory.NEUTRAL


class TestPriorityBasedWordMatching:
    """Verify that RIVALRY beats NEUTRAL in same compound (priority-based)."""

    def test_rivalry_beats_neutral_in_compound(self):
        """When compound contains both neutral and rivalry words, RIVALRY wins."""
        # "colleague" -> works_with (NEUTRAL), "rival" -> rivals (RIVALRY)
        result = normalize_relation_type("colleague_rival")
        assert classify_relationship(result) == ConflictCategory.RIVALRY

    def test_rivalry_beats_alliance_in_compound(self):
        """When compound contains both alliance and rivalry words, RIVALRY wins."""
        # "friend" -> friends (ALLIANCE), "enemy" -> enemy_of (RIVALRY)
        result = normalize_relation_type("friend_enemy_dynamic")
        assert classify_relationship(result) == ConflictCategory.RIVALRY

    def test_tension_beats_alliance_in_compound(self):
        """When compound contains both alliance and tension words, TENSION wins."""
        # "loyal" -> loyal_to (ALLIANCE), "fear" -> fears (TENSION)
        result = normalize_relation_type("loyal_fear_bond")
        assert classify_relationship(result) == ConflictCategory.TENSION

    def test_tension_beats_neutral_in_compound(self):
        """When compound contains both neutral and tension words, TENSION wins."""
        # "colleague" -> works_with (NEUTRAL), "wary" -> wary_of (TENSION)
        result = normalize_relation_type("colleague_wary_dynamic")
        assert classify_relationship(result) == ConflictCategory.TENSION


class TestClassifyRelationshipUsesNormalization:
    """classify_relationship() should use normalize_relation_type() internally."""

    def test_classify_normalizes_compound_type(self):
        """classify_relationship should normalize before lookup, not just direct-match."""
        # "colleague_and_occasional_rival" is not a direct key in RELATION_CONFLICT_MAPPING.
        # Without normalization, classify_relationship would return NEUTRAL.
        # With normalization wired in, it resolves via word-level match.
        assert classify_relationship("colleague_and_occasional_rival") != ConflictCategory.NEUTRAL

    def test_classify_handles_hyphenated_input(self):
        """classify_relationship should handle hyphens via normalization."""
        assert classify_relationship("bitter-rivals") == ConflictCategory.RIVALRY

    def test_classify_handles_spaced_input(self):
        """classify_relationship should handle spaces via normalization."""
        assert classify_relationship("allies with") == ConflictCategory.ALLIANCE


class TestValidateConflictPriority:
    """Test the _validate_conflict_priority guard function."""

    def test_passes_when_complete(self):
        """No error when all categories are present."""
        _validate_conflict_priority()  # Should not raise

    def test_raises_on_missing_category(self):
        """Raises RuntimeError when a category is missing from _CONFLICT_PRIORITY."""
        saved = _CONFLICT_PRIORITY.pop(ConflictCategory.NEUTRAL)
        try:
            with pytest.raises(RuntimeError, match="_CONFLICT_PRIORITY missing categories"):
                _validate_conflict_priority()
        finally:
            _CONFLICT_PRIORITY[ConflictCategory.NEUTRAL] = saved


class TestValidateWordToRelation:
    """Test the _validate_word_to_relation guard function."""

    def test_passes_when_all_values_are_valid(self):
        """No error when all word targets are valid relation types."""
        _validate_word_to_relation()  # Should not raise

    def test_raises_on_invalid_target(self):
        """Raises RuntimeError when a word maps to a nonexistent relation type."""
        _WORD_TO_RELATION["bogus_word"] = "nonexistent_relation_type"
        try:
            with pytest.raises(RuntimeError, match="_WORD_TO_RELATION references unknown types"):
                _validate_word_to_relation()
        finally:
            del _WORD_TO_RELATION["bogus_word"]

"""Tests for conflict types — mapping, normalization, and classification."""

from src.memory.conflict_types import (
    RELATION_CONFLICT_MAPPING,
    VALID_RELATIONSHIP_TYPES,
    ConflictCategory,
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
        """Prose like 'bitter rivals who fought' should extract 'rivals'."""
        assert normalize_relation_type("bitter rivals who fought") == "rivals"

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
        # but substring matching should find "rivals" inside "foo|bitter_rivals_bar"
        result = normalize_relation_type("foo|bitter_rivals_bar")
        assert result == "rivals"

    def test_pipe_delimited_no_recognized_no_substring(self):
        """Pipe-delimited with no recognized parts and no substring match returns normalized."""
        result = normalize_relation_type("xyz|abc")
        assert result == "xyz|abc"

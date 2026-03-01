"""Tests for conflict_types prose sentiment keyword normalization (L1)."""

import pytest

from src.memory.conflict_types import normalize_relation_type


class TestProseSentimentKeywordNormalization:
    """Tests for normalize_relation_type with prose sentiment keywords merged into _WORD_TO_RELATION."""

    @pytest.fixture(autouse=True)
    def _clear_lru_cache(self):
        """Clear the lru_cache before and after each test."""
        normalize_relation_type.cache_clear()
        yield
        normalize_relation_type.cache_clear()

    def test_normalize_admiration_to_trusts(self):
        """normalize_relation_type('admiration') should map to 'trusts'."""
        result = normalize_relation_type("admiration")
        assert result == "trusts"

    def test_normalize_hostility_to_enemy_of(self):
        """normalize_relation_type('hostility') should map to 'enemy_of'."""
        result = normalize_relation_type("hostility")
        assert result == "enemy_of"

    def test_normalize_partnership_to_allies_with(self):
        """normalize_relation_type('partnership') should map to 'allies_with'."""
        result = normalize_relation_type("partnership")
        assert result == "allies_with"

    def test_validate_word_to_relation_raises_on_invalid(self, monkeypatch):
        """_validate_word_to_relation raises RuntimeError for unknown relation types."""
        from src.memory import conflict_types
        from src.memory.conflict_types import _validate_word_to_relation

        original = conflict_types._WORD_TO_RELATION.copy()
        monkeypatch.setattr(
            conflict_types,
            "_WORD_TO_RELATION",
            {**original, "bogus_sentiment": "totally_fake_type"},
        )
        with pytest.raises(RuntimeError, match="unknown types"):
            _validate_word_to_relation()

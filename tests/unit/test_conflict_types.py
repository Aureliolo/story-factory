"""Tests for conflict_types prose sentiment keyword normalization (L1)."""

import pytest

from src.memory.conflict_types import normalize_relation_type


class TestProseSentimentKeywordNormalization:
    """Tests for normalize_relation_type prose sentiment keyword fallback."""

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

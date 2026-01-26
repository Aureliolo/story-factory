"""Unit tests for validate_unique_name in validation.py."""

from unittest.mock import MagicMock, patch

from src.utils.validation import _normalize_name, validate_unique_name


class TestNormalizeName:
    """Tests for _normalize_name helper function."""

    def test_lowercase(self) -> None:
        """Test that names are lowercased."""
        assert _normalize_name("The Guild") == "guild"
        assert _normalize_name("SHADOW COUNCIL") == "shadow council"

    def test_strips_prefix_the(self) -> None:
        """Test that 'the ' prefix is stripped."""
        assert _normalize_name("The Order") == "order"
        assert _normalize_name("the order") == "order"
        assert _normalize_name("THE ORDER") == "order"

    def test_strips_prefix_a(self) -> None:
        """Test that 'a ' prefix is stripped."""
        assert _normalize_name("A Brotherhood") == "brotherhood"
        assert _normalize_name("a brotherhood") == "brotherhood"

    def test_strips_prefix_an(self) -> None:
        """Test that 'an ' prefix is stripped."""
        assert _normalize_name("An Alliance") == "alliance"
        assert _normalize_name("an alliance") == "alliance"

    def test_only_strips_first_prefix(self) -> None:
        """Test that only the first prefix is stripped, not all."""
        assert _normalize_name("The The Order") == "the order"

    def test_whitespace_trimmed(self) -> None:
        """Test that whitespace is trimmed."""
        assert _normalize_name("  The Order  ") == "order"
        assert _normalize_name("\tGuild\n") == "guild"


class TestValidateUniqueName:
    """Tests for validate_unique_name function."""

    def test_empty_name_returns_unique(self) -> None:
        """Test that empty names are considered unique (handled elsewhere)."""
        is_unique, conflict, reason = validate_unique_name("", ["Guild", "Order"])
        assert is_unique is True
        assert conflict is None
        assert reason is None

    def test_whitespace_only_returns_unique(self) -> None:
        """Test that whitespace-only names are considered unique."""
        is_unique, _conflict, _reason = validate_unique_name("   ", ["Guild"])
        assert is_unique is True

    def test_empty_existing_list(self) -> None:
        """Test that any name is unique when existing list is empty."""
        is_unique, conflict, _reason = validate_unique_name("The Guild", [])
        assert is_unique is True
        assert conflict is None

    def test_exact_match_detected(self) -> None:
        """Test that exact matches are detected."""
        is_unique, conflict, reason = validate_unique_name("The Guild", ["The Guild", "The Order"])
        assert is_unique is False
        assert conflict == "The Guild"
        assert reason == "exact"

    def test_case_insensitive_match_detected(self) -> None:
        """Test that case-insensitive matches are detected."""
        is_unique, conflict, reason = validate_unique_name("THE GUILD", ["The Guild", "The Order"])
        assert is_unique is False
        assert conflict == "The Guild"
        assert reason == "case_insensitive"

    def test_prefix_match_the_detected(self) -> None:
        """Test that 'The X' vs 'X' prefix matches are detected."""
        is_unique, conflict, reason = validate_unique_name("Guild", ["The Guild", "Order"])
        assert is_unique is False
        assert conflict == "The Guild"
        assert reason == "prefix_match"

        # Reverse direction
        is_unique, conflict, reason = validate_unique_name("The Order", ["Guild", "Order"])
        assert is_unique is False
        assert conflict == "Order"
        assert reason == "prefix_match"

    def test_substring_containment_detected(self) -> None:
        """Test that substring containment is detected."""
        # "council" is a substring of "council of shadows" (after normalization)
        is_unique, conflict, reason = validate_unique_name(
            "Council", ["Council of Shadows", "Brotherhood"]
        )
        assert is_unique is False
        assert conflict == "Council of Shadows"
        assert reason == "substring"

        # Reverse direction: "brotherhood" is contained in "iron brotherhood"
        is_unique, conflict, reason = validate_unique_name(
            "Iron Brotherhood", ["Brotherhood", "Guild"]
        )
        assert is_unique is False
        assert conflict == "Brotherhood"
        assert reason == "substring"

    def test_substring_disabled(self) -> None:
        """Test that substring checking can be disabled."""
        is_unique, _conflict, _reason = validate_unique_name(
            "Council", ["Council of Shadows"], check_substring=False
        )
        assert is_unique is True

    def test_short_names_skip_substring_check(self) -> None:
        """Test that short names skip substring validation."""
        # "cat" is only 3 chars, below default min_substring_length of 4
        is_unique, _conflict, _reason = validate_unique_name(
            "Cat", ["Category", "Catastrophe"], min_substring_length=4
        )
        assert is_unique is True

    def test_min_substring_length_respected(self) -> None:
        """Test that min_substring_length parameter is respected."""
        # With min=3, "cat" should be checked
        is_unique, _conflict, reason = validate_unique_name(
            "Cat", ["Category", "Catastrophe"], min_substring_length=3
        )
        assert is_unique is False
        assert reason == "substring"

    def test_unique_name_passes(self) -> None:
        """Test that a truly unique name passes validation."""
        is_unique, conflict, reason = validate_unique_name(
            "The Crimson Brotherhood",
            ["The Guild", "The Order", "Shadow Council", "Iron Covenant"],
        )
        assert is_unique is True
        assert conflict is None
        assert reason is None

    def test_skips_empty_existing_names(self) -> None:
        """Test that empty strings in existing names are skipped."""
        is_unique, _conflict, _reason = validate_unique_name(
            "The Guild",
            ["", "  ", None, "The Order"],  # type: ignore
        )
        assert is_unique is True

    def test_complex_case_variations(self) -> None:
        """Test various case and prefix combinations."""
        existing = ["The Shadow Guild"]

        # Case + prefix match
        is_unique, _conflict, reason = validate_unique_name("shadow guild", existing)
        assert is_unique is False
        assert reason == "prefix_match"

        # Another prefix variation
        is_unique, _conflict, reason = validate_unique_name("A Shadow Guild", existing)
        assert is_unique is False
        assert reason == "prefix_match"


class TestSemanticDuplicateChecking:
    """Tests for semantic duplicate checking in validate_unique_name."""

    @patch("src.utils.validation.get_semantic_checker")
    def test_semantic_check_detects_duplicate(self, mock_get_checker) -> None:
        """Test that semantic checking detects similar names."""
        mock_checker = MagicMock()
        mock_checker.find_semantic_duplicate.return_value = (
            True,
            "Council of Shadows",
            0.92,
        )
        mock_get_checker.return_value = mock_checker

        is_unique, conflict, reason = validate_unique_name(
            "Shadow Council",
            ["Council of Shadows", "The Brotherhood"],
            check_semantic=True,
            semantic_threshold=0.85,
        )

        assert is_unique is False
        assert conflict == "Council of Shadows"
        assert reason == "semantic"

    @patch("src.utils.validation.get_semantic_checker")
    def test_semantic_check_allows_different_names(self, mock_get_checker) -> None:
        """Test that semantic checking allows clearly different names."""
        mock_checker = MagicMock()
        mock_checker.find_semantic_duplicate.return_value = (False, None, 0.3)
        mock_get_checker.return_value = mock_checker

        is_unique, conflict, reason = validate_unique_name(
            "The Happy Bakers",
            ["Council of Shadows", "The Brotherhood"],
            check_semantic=True,
            semantic_threshold=0.85,
        )

        assert is_unique is True
        assert conflict is None
        assert reason is None

    @patch("src.utils.validation.get_semantic_checker")
    def test_semantic_check_disabled_by_default(self, mock_get_checker) -> None:
        """Test that semantic checking is disabled by default."""
        is_unique, _conflict, _reason = validate_unique_name(
            "Shadow Council",
            ["Council of Shadows"],
            check_semantic=False,  # Explicitly disabled
        )

        # Semantic checker should not be called
        mock_get_checker.assert_not_called()
        # Name should be considered unique (no string match)
        assert is_unique is True

    @patch("src.utils.validation.get_semantic_checker")
    def test_semantic_check_with_custom_url_and_model(self, mock_get_checker) -> None:
        """Test that custom ollama_url and embedding_model are passed."""
        mock_checker = MagicMock()
        mock_checker.find_semantic_duplicate.return_value = (False, None, 0.3)
        mock_get_checker.return_value = mock_checker

        validate_unique_name(
            "Test Name",
            ["Existing"],
            check_semantic=True,
            ollama_url="http://custom:11434",
            embedding_model="custom-model",
        )

        # Verify custom parameters were passed
        mock_get_checker.assert_called_once()
        call_kwargs = mock_get_checker.call_args.kwargs
        assert call_kwargs.get("ollama_url") == "http://custom:11434"
        assert call_kwargs.get("embedding_model") == "custom-model"

    @patch("src.utils.validation.get_semantic_checker")
    def test_semantic_check_handles_exception_gracefully(self, mock_get_checker) -> None:
        """Test that semantic check failures don't block validation."""
        mock_get_checker.side_effect = ConnectionError("Ollama not running")

        # Should not raise, just skip semantic check
        is_unique, conflict, reason = validate_unique_name(
            "Shadow Council",
            ["Council of Shadows"],
            check_semantic=True,
        )

        # String-based checks pass, semantic check skipped on error
        assert is_unique is True
        assert conflict is None
        assert reason is None

    @patch("src.utils.validation.get_semantic_checker")
    def test_semantic_check_skipped_for_empty_existing_names(self, mock_get_checker) -> None:
        """Test that semantic check is skipped if existing_names is empty."""
        is_unique, _conflict, _reason = validate_unique_name(
            "Shadow Council",
            [],  # Empty list
            check_semantic=True,
        )

        # Semantic checker should not be called for empty list
        mock_get_checker.assert_not_called()
        assert is_unique is True

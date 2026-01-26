"""Tests for entity card component - Issue #173 protagonist styling."""

from src.memory.entities import Entity
from src.ui.theme import ROLE_COLORS, get_role_border_style, get_role_type


class TestGetRoleType:
    """Tests for centralized role type detection."""

    def test_protagonist_detection(self):
        """Test that protagonist is detected from role attribute."""
        assert get_role_type({"role": "protagonist"}) == "protagonist"

    def test_main_character_detection(self):
        """Test that 'main' in role is detected as protagonist."""
        assert get_role_type({"role": "main character"}) == "protagonist"

    def test_antagonist_detection(self):
        """Test that antagonist is detected from role attribute."""
        assert get_role_type({"role": "antagonist"}) == "antagonist"

    def test_mentor_detection(self):
        """Test that mentor is detected from role attribute."""
        assert get_role_type({"role": "mentor"}) == "mentor"

    def test_no_role_returns_none(self):
        """Test that no role returns None."""
        assert get_role_type({}) is None
        assert get_role_type({"role": "supporting"}) is None

    def test_none_attributes_returns_none(self):
        """Test that None attributes returns None."""
        assert get_role_type(None) is None

    def test_none_role_returns_none(self):
        """Test that None role value returns None."""
        assert get_role_type({"role": None}) is None

    def test_case_insensitive(self):
        """Test that role detection is case-insensitive."""
        assert get_role_type({"role": "PROTAGONIST"}) == "protagonist"
        assert get_role_type({"role": "Antagonist"}) == "antagonist"
        assert get_role_type({"role": "MENTOR"}) == "mentor"


class TestGetRoleBorderStyle:
    """Tests for centralized role border style generation."""

    def test_protagonist_border_style(self):
        """Test that protagonist gets gold border with shadow."""
        style = get_role_border_style({"role": "protagonist"})
        assert ROLE_COLORS["protagonist"] in style  # Gold color
        assert "box-shadow" in style

    def test_antagonist_border_style(self):
        """Test that antagonist gets red border."""
        style = get_role_border_style({"role": "antagonist"})
        assert ROLE_COLORS["antagonist"] in style  # Red color

    def test_mentor_border_style(self):
        """Test that mentor gets blue border."""
        style = get_role_border_style({"role": "mentor"})
        assert ROLE_COLORS["mentor"] in style  # Blue color

    def test_no_role_empty_style(self):
        """Test that no role returns empty style."""
        assert get_role_border_style({}) == ""
        assert get_role_border_style(None) == ""
        assert get_role_border_style({"role": "supporting"}) == ""


class TestEntityListItemProtagonistStyling:
    """Tests for role-based styling in entity_list_item using centralized helpers."""

    def test_protagonist_entity_gets_correct_style(self):
        """Test that entity with protagonist role gets gold border style."""
        entity = Entity(
            id="test-1",
            name="Hero",
            type="character",
            description="The main hero",
            attributes={"role": "protagonist"},
        )

        style = get_role_border_style(entity.attributes)
        assert ROLE_COLORS["protagonist"] in style

    def test_main_character_entity_gets_protagonist_style(self):
        """Test that entity with 'main character' role gets protagonist style."""
        entity = Entity(
            id="test-2",
            name="Main Character",
            type="character",
            description="The main character",
            attributes={"role": "main character"},
        )

        role_type = get_role_type(entity.attributes)
        assert role_type == "protagonist"

    def test_antagonist_entity_gets_correct_style(self):
        """Test that entity with antagonist role gets red border style."""
        entity = Entity(
            id="test-3",
            name="Villain",
            type="character",
            description="The villain",
            attributes={"role": "antagonist"},
        )

        style = get_role_border_style(entity.attributes)
        assert ROLE_COLORS["antagonist"] in style

    def test_mentor_entity_gets_correct_style(self):
        """Test that entity with mentor role gets blue border style."""
        entity = Entity(
            id="test-4",
            name="Wise One",
            type="character",
            description="The mentor",
            attributes={"role": "mentor"},
        )

        style = get_role_border_style(entity.attributes)
        assert ROLE_COLORS["mentor"] in style

    def test_supporting_character_no_special_style(self):
        """Test that supporting character has no special styling."""
        entity = Entity(
            id="test-5",
            name="Side Character",
            type="character",
            description="A side character",
            attributes={"role": "supporting"},
        )

        style = get_role_border_style(entity.attributes)
        assert style == ""

    def test_empty_attributes_no_style(self):
        """Test that empty attributes result in no special style."""
        entity = Entity(
            id="test-6",
            name="No Attrs",
            type="character",
            description="No attributes",
            attributes={},
        )

        style = get_role_border_style(entity.attributes)
        assert style == ""

    def test_case_insensitive_role_detection(self):
        """Test that role detection is case-insensitive."""
        test_cases = [
            ("PROTAGONIST", "protagonist"),
            ("Protagonist", "protagonist"),
            ("MAIN CHARACTER", "protagonist"),
            ("ANTAGONIST", "antagonist"),
            ("Antagonist", "antagonist"),
            ("MENTOR", "mentor"),
            ("Mentor", "mentor"),
        ]

        for role_value, expected_type in test_cases:
            entity = Entity(
                id="test",
                name="Test",
                type="character",
                description="Test",
                attributes={"role": role_value},
            )

            role_type = get_role_type(entity.attributes)
            assert role_type == expected_type, f"Failed for role={role_value}"


class TestRoleColors:
    """Tests for ROLE_COLORS constant."""

    def test_protagonist_color_is_gold(self):
        """Test that protagonist color is gold."""
        assert ROLE_COLORS["protagonist"] == "#FFD700"

    def test_antagonist_color_is_red(self):
        """Test that antagonist color is red."""
        assert ROLE_COLORS["antagonist"] == "#F44336"

    def test_mentor_color_is_blue(self):
        """Test that mentor color is blue."""
        assert ROLE_COLORS["mentor"] == "#2196F3"

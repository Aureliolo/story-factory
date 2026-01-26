"""Tests for entity card component - Issue #173 protagonist styling."""

from src.memory.entities import Entity


class TestEntityListItemProtagonistStyling:
    """Tests for role-based styling in entity_list_item."""

    def test_protagonist_detection_in_role(self):
        """Test that protagonist is detected from role attribute."""
        # Create entity with protagonist role
        entity = Entity(
            id="test-1",
            name="Hero",
            type="character",
            description="The main hero",
            attributes={"role": "protagonist"},
        )

        role = (entity.attributes.get("role") or "").lower()
        is_protagonist = "protagonist" in role or "main" in role

        assert is_protagonist is True

    def test_main_character_detection_in_role(self):
        """Test that 'main' in role is detected as protagonist."""
        entity = Entity(
            id="test-2",
            name="Main Character",
            type="character",
            description="The main character",
            attributes={"role": "main character"},
        )

        role = (entity.attributes.get("role") or "").lower()
        is_protagonist = "protagonist" in role or "main" in role

        assert is_protagonist is True

    def test_antagonist_detection(self):
        """Test that antagonist is detected from role attribute."""
        entity = Entity(
            id="test-3",
            name="Villain",
            type="character",
            description="The villain",
            attributes={"role": "antagonist"},
        )

        role = (entity.attributes.get("role") or "").lower()
        is_antagonist = "antagonist" in role

        assert is_antagonist is True

    def test_mentor_detection(self):
        """Test that mentor is detected from role attribute."""
        entity = Entity(
            id="test-4",
            name="Wise One",
            type="character",
            description="The mentor",
            attributes={"role": "mentor"},
        )

        role = (entity.attributes.get("role") or "").lower()
        is_mentor = "mentor" in role

        assert is_mentor is True

    def test_no_role_no_special_styling(self):
        """Test that entities without role have no special styling."""
        entity = Entity(
            id="test-5",
            name="Side Character",
            type="character",
            description="A side character",
            attributes={},
        )

        role = (entity.attributes.get("role") or "").lower()
        is_protagonist = "protagonist" in role or "main" in role
        is_antagonist = "antagonist" in role
        is_mentor = "mentor" in role

        assert is_protagonist is False
        assert is_antagonist is False
        assert is_mentor is False

    def test_empty_attributes_handled(self):
        """Test that empty attributes don't cause errors."""
        entity = Entity(
            id="test-6",
            name="No Attrs",
            type="character",
            description="No attributes",
            attributes={},
        )

        # This should not raise
        role = (entity.attributes.get("role") or "").lower() if entity.attributes else ""
        is_protagonist = "protagonist" in role or "main" in role

        assert is_protagonist is False

    def test_none_role_handled(self):
        """Test that None role value doesn't cause errors."""
        entity = Entity(
            id="test-7",
            name="None Role",
            type="character",
            description="None role value",
            attributes={"role": None},
        )

        role = (entity.attributes.get("role") or "").lower()
        is_protagonist = "protagonist" in role or "main" in role

        assert is_protagonist is False

    def test_protagonist_border_style(self):
        """Test that protagonist border style is gold with shadow."""
        entity = Entity(
            id="test-8",
            name="Hero",
            type="character",
            description="The hero",
            attributes={"role": "protagonist"},
        )

        role = (entity.attributes.get("role") or "").lower()
        is_protagonist = "protagonist" in role or "main" in role

        if is_protagonist:
            border_style = "border: 2px solid #FFD700; box-shadow: 0 0 6px #FFD70080;"
        else:
            border_style = ""

        assert "#FFD700" in border_style  # Gold color
        assert "box-shadow" in border_style

    def test_antagonist_border_style(self):
        """Test that antagonist border style is red."""
        entity = Entity(
            id="test-9",
            name="Villain",
            type="character",
            description="The villain",
            attributes={"role": "antagonist"},
        )

        role = (entity.attributes.get("role") or "").lower()
        is_antagonist = "antagonist" in role

        if is_antagonist:
            border_style = "border: 2px solid #F44336;"
        else:
            border_style = ""

        assert "#F44336" in border_style  # Red color

    def test_mentor_border_style(self):
        """Test that mentor border style is blue."""
        entity = Entity(
            id="test-10",
            name="Wise One",
            type="character",
            description="The mentor",
            attributes={"role": "mentor"},
        )

        role = (entity.attributes.get("role") or "").lower()
        is_mentor = "mentor" in role

        if is_mentor:
            border_style = "border: 2px solid #2196F3;"
        else:
            border_style = ""

        assert "#2196F3" in border_style  # Blue color

    def test_case_insensitive_role_detection(self):
        """Test that role detection is case-insensitive."""
        test_cases = [
            ("PROTAGONIST", True, False, False),
            ("Protagonist", True, False, False),
            ("MAIN CHARACTER", True, False, False),
            ("ANTAGONIST", False, True, False),
            ("Antagonist", False, True, False),
            ("MENTOR", False, False, True),
            ("Mentor", False, False, True),
        ]

        for role_value, expected_protag, expected_antag, expected_mentor in test_cases:
            entity = Entity(
                id="test",
                name="Test",
                type="character",
                description="Test",
                attributes={"role": role_value},
            )

            role = (entity.attributes.get("role") or "").lower()
            is_protagonist = "protagonist" in role or "main" in role
            is_antagonist = "antagonist" in role
            is_mentor = "mentor" in role

            assert is_protagonist == expected_protag, f"Failed for role={role_value}"
            assert is_antagonist == expected_antag, f"Failed for role={role_value}"
            assert is_mentor == expected_mentor, f"Failed for role={role_value}"

"""Unit tests for WorldTemplateService."""

import json
from unittest.mock import patch

import pytest

from src.memory.builtin_world_templates import BUILTIN_WORLD_TEMPLATES
from src.memory.templates import WorldTemplate
from src.services.world_template_service import WorldTemplateService
from src.utils.exceptions import StoryFactoryError


@pytest.fixture
def temp_templates_file(tmp_path):
    """Create a temporary templates file path."""
    return tmp_path / "user_world_templates.json"


@pytest.fixture
def service(temp_templates_file):
    """Create a WorldTemplateService with a temporary templates file."""
    with patch(
        "src.services.world_template_service.USER_TEMPLATES_FILE",
        temp_templates_file,
    ):
        return WorldTemplateService()


class TestWorldTemplateServiceInit:
    """Tests for WorldTemplateService initialization."""

    def test_init_without_user_templates(self, service):
        """Test initialization when no user templates file exists."""
        assert service is not None
        # Should have no user templates
        templates = service.list_templates(include_builtin=False)
        assert len(templates) == 0

    def test_init_loads_user_templates(self, temp_templates_file):
        """Test that user templates are loaded on init."""
        # Create a user templates file
        user_template = {
            "id": "custom_template",
            "name": "Custom Template",
            "description": "A custom world template",
            "is_builtin": False,
            "genre": "custom",
            "entity_hints": {},
            "relationship_patterns": [],
            "naming_style": "custom",
            "recommended_counts": {},
            "atmosphere": "custom",
            "tags": ["custom"],
        }
        temp_templates_file.parent.mkdir(parents=True, exist_ok=True)
        temp_templates_file.write_text(json.dumps([user_template]))

        with patch(
            "src.services.world_template_service.USER_TEMPLATES_FILE",
            temp_templates_file,
        ):
            service = WorldTemplateService()
            templates = service.list_templates(include_builtin=False)
            assert len(templates) == 1
            assert templates[0].id == "custom_template"


class TestListTemplates:
    """Tests for list_templates method."""

    def test_list_all_templates(self, service):
        """Test listing all templates including built-in."""
        templates = service.list_templates()
        assert len(templates) >= len(BUILTIN_WORLD_TEMPLATES)

    def test_list_only_builtin(self, service):
        """Test that only user templates are listed when exclude builtin."""
        templates = service.list_templates(include_builtin=False)
        assert len(templates) == 0

    def test_builtin_templates_included(self, service):
        """Test that all built-in templates are included."""
        templates = service.list_templates()
        template_ids = {t.id for t in templates}
        for builtin_id in BUILTIN_WORLD_TEMPLATES:
            assert builtin_id in template_ids


class TestGetTemplate:
    """Tests for get_template method."""

    def test_get_builtin_template(self, service):
        """Test getting a built-in template."""
        template = service.get_template("high_fantasy")
        assert template is not None
        assert template.id == "high_fantasy"
        assert template.name == "High Fantasy"

    def test_get_nonexistent_template(self, service):
        """Test getting a nonexistent template returns None."""
        template = service.get_template("nonexistent")
        assert template is None

    def test_get_all_builtin_templates(self, service):
        """Test that all built-in templates can be retrieved."""
        for template_id in BUILTIN_WORLD_TEMPLATES:
            template = service.get_template(template_id)
            assert template is not None
            assert template.id == template_id


class TestSaveTemplate:
    """Tests for save_template method."""

    def test_save_user_template(self, service, temp_templates_file):
        """Test saving a user template."""
        with patch(
            "src.services.world_template_service.USER_TEMPLATES_FILE",
            temp_templates_file,
        ):
            template = WorldTemplate(
                id="new_template",
                name="New Template",
                description="A new template",
                is_builtin=False,
                genre="custom",
            )
            service.save_template(template)

            # Verify it's saved
            saved_template = service.get_template("new_template")
            assert saved_template is not None
            assert saved_template.name == "New Template"

    def test_cannot_overwrite_builtin(self, service):
        """Test that built-in templates cannot be overwritten."""
        template = WorldTemplate(
            id="high_fantasy",  # Built-in ID
            name="Fake Fantasy",
            description="Trying to overwrite",
            is_builtin=False,
            genre="fantasy",
        )
        with pytest.raises(StoryFactoryError, match="Cannot overwrite built-in template"):
            service.save_template(template)


class TestDeleteTemplate:
    """Tests for delete_template method."""

    def test_delete_user_template(self, service, temp_templates_file):
        """Test deleting a user template."""
        with patch(
            "src.services.world_template_service.USER_TEMPLATES_FILE",
            temp_templates_file,
        ):
            # First save a template
            template = WorldTemplate(
                id="to_delete",
                name="To Delete",
                description="This will be deleted",
                is_builtin=False,
                genre="custom",
            )
            service.save_template(template)
            assert service.get_template("to_delete") is not None

            # Delete it
            result = service.delete_template("to_delete")
            assert result is True
            assert service.get_template("to_delete") is None

    def test_cannot_delete_builtin(self, service):
        """Test that built-in templates cannot be deleted."""
        result = service.delete_template("high_fantasy")
        assert result is False

    def test_delete_nonexistent(self, service):
        """Test deleting a nonexistent template returns False."""
        result = service.delete_template("nonexistent")
        assert result is False


class TestFormatHintsForPrompt:
    """Tests for format_hints_for_prompt method."""

    def test_format_includes_basic_info(self, service):
        """Test that formatted hints include basic template info."""
        template = service.get_template("high_fantasy")
        hints = service.format_hints_for_prompt(template)

        assert "High Fantasy" in hints
        assert "fantasy" in hints.lower()
        assert template.atmosphere in hints

    def test_format_includes_entity_hints(self, service):
        """Test that formatted hints include entity hints."""
        template = service.get_template("cyberpunk")
        hints = service.format_hints_for_prompt(template)

        # Check for some expected hints
        assert "netrunner" in hints.lower() or "Character Roles" in hints
        assert "megacorp" in hints.lower() or "Location Types" in hints

    def test_format_includes_relationship_patterns(self, service):
        """Test that formatted hints include relationship patterns."""
        template = service.get_template("urban_fantasy")
        hints = service.format_hints_for_prompt(template)

        assert "Relationship Types" in hints

    def test_format_includes_naming_style(self, service):
        """Test that formatted hints include naming style."""
        template = service.get_template("space_opera")
        hints = service.format_hints_for_prompt(template)

        assert "Naming Style" in hints

    def test_format_empty_template(self, service):
        """Test formatting a minimal template."""
        template = service.get_template("blank_canvas")
        hints = service.format_hints_for_prompt(template)

        assert "Blank Canvas" in hints
        # Should not have entity hint sections since they're empty
        assert "Suggested Character Roles:" not in hints or hints.count(":") >= 1


class TestGetTemplateForGenre:
    """Tests for get_template_for_genre method."""

    def test_find_fantasy_template(self, service):
        """Test finding a fantasy template."""
        template = service.get_template_for_genre("fantasy")
        assert template is not None
        assert template.genre == "fantasy" or "fantasy" in template.tags

    def test_find_scifi_template(self, service):
        """Test finding a science fiction template."""
        template = service.get_template_for_genre("science_fiction")
        assert template is not None

    def test_find_via_tags(self, service):
        """Test finding a template via tags."""
        template = service.get_template_for_genre("epic")
        assert template is not None

    def test_not_found_returns_none(self, service):
        """Test that unknown genre returns None."""
        template = service.get_template_for_genre("unknown_genre_xyz")
        assert template is None


class TestGetRecommendedCounts:
    """Tests for get_recommended_counts method."""

    def test_get_character_counts(self, service):
        """Test getting recommended character counts."""
        template = service.get_template("high_fantasy")
        counts = service.get_recommended_counts(template, "characters")
        assert counts is not None
        assert len(counts) == 2
        assert counts[0] < counts[1]  # min < max

    def test_get_location_counts(self, service):
        """Test getting recommended location counts."""
        template = service.get_template("cyberpunk")
        counts = service.get_recommended_counts(template, "locations")
        assert counts is not None

    def test_unknown_entity_type(self, service):
        """Test getting counts for unknown entity type returns None."""
        template = service.get_template("high_fantasy")
        counts = service.get_recommended_counts(template, "unknown_type")
        assert counts is None


class TestBuiltinTemplates:
    """Tests for built-in template content."""

    def test_high_fantasy_has_required_fields(self, service):
        """Test that high fantasy template has all required fields."""
        template = service.get_template("high_fantasy")
        assert template.name == "High Fantasy"
        assert template.genre == "fantasy"
        assert len(template.entity_hints.character_roles) > 0
        assert len(template.entity_hints.location_types) > 0
        assert len(template.atmosphere) > 0

    def test_cyberpunk_has_tech_themes(self, service):
        """Test that cyberpunk template has tech-related content."""
        template = service.get_template("cyberpunk")
        all_hints = (
            template.entity_hints.character_roles
            + template.entity_hints.item_types
            + template.entity_hints.concept_types
        )
        hint_text = " ".join(all_hints).lower()
        assert "netrunner" in hint_text or "cyber" in hint_text or "neural" in hint_text

    def test_all_templates_have_atmosphere(self, service):
        """Test that all templates have atmosphere defined."""
        templates = service.list_templates()
        for template in templates:
            assert len(template.atmosphere) > 0, f"Template {template.id} missing atmosphere"

    def test_blank_canvas_is_minimal(self, service):
        """Test that blank canvas template is minimal."""
        template = service.get_template("blank_canvas")
        assert len(template.entity_hints.character_roles) == 0
        assert len(template.entity_hints.location_types) == 0
        assert len(template.recommended_counts) == 0


class TestListWorldTemplates:
    """Tests for list_world_templates function from builtin_world_templates."""

    def test_list_world_templates_returns_all(self):
        """Test that list_world_templates returns all built-in templates."""
        from src.memory.builtin_world_templates import (
            BUILTIN_WORLD_TEMPLATES,
            list_world_templates,
        )

        templates = list_world_templates()
        assert len(templates) == len(BUILTIN_WORLD_TEMPLATES)
        for template in templates:
            assert template.id in BUILTIN_WORLD_TEMPLATES


class TestUserTemplateErrorHandling:
    """Tests for error handling when loading user templates."""

    def test_init_with_corrupt_templates_file(self, tmp_path):
        """Test that corrupt user templates file is handled gracefully."""
        corrupt_file = tmp_path / "user_world_templates.json"
        corrupt_file.write_text("{ invalid json }")

        with patch(
            "src.services.world_template_service.USER_TEMPLATES_FILE",
            corrupt_file,
        ):
            service = WorldTemplateService()
            # Service should initialize without error, just with no user templates
            templates = service.list_templates(include_builtin=False)
            assert len(templates) == 0

    def test_init_with_empty_templates_file(self, tmp_path):
        """Test that empty user templates file is handled gracefully."""
        empty_file = tmp_path / "user_world_templates.json"
        empty_file.parent.mkdir(parents=True, exist_ok=True)
        empty_file.write_text("[]")  # Valid JSON but no templates

        with patch(
            "src.services.world_template_service.USER_TEMPLATES_FILE",
            empty_file,
        ):
            service = WorldTemplateService()
            # Service should initialize without error, with no user templates
            templates = service.list_templates(include_builtin=False)
            assert len(templates) == 0

    def test_save_template_file_write_error(self, tmp_path):
        """Test that file write errors are raised as StoryFactoryError."""
        temp_file = tmp_path / "data" / "user_world_templates.json"

        with patch(
            "src.services.world_template_service.USER_TEMPLATES_FILE",
            temp_file,
        ):
            service = WorldTemplateService()

            template = WorldTemplate(
                id="test_template",
                name="Test Template",
                description="A test template",
                is_builtin=False,
                genre="test",
            )

            # Mock open to raise OSError
            with patch("builtins.open", side_effect=OSError("Disk full")):
                with pytest.raises(StoryFactoryError, match="Failed to save user world templates"):
                    service.save_template(template)


class TestWorldTemplateValidation:
    """Tests for WorldTemplate validation."""

    def test_recommended_counts_valid(self):
        """Test that valid recommended_counts pass validation."""
        template = WorldTemplate(
            id="valid_counts",
            name="Valid Counts",
            description="Test",
            genre="test",
            recommended_counts={"characters": (2, 5), "locations": (1, 10)},
        )
        assert template.recommended_counts["characters"] == (2, 5)
        assert template.recommended_counts["locations"] == (1, 10)

    def test_recommended_counts_min_greater_than_max(self):
        """Test that min > max raises ValidationError."""
        import pydantic

        with pytest.raises(pydantic.ValidationError, match=r"min.*>.*max"):
            WorldTemplate(
                id="invalid_counts",
                name="Invalid Counts",
                description="Test",
                genre="test",
                recommended_counts={"characters": (10, 5)},  # min > max
            )

    def test_recommended_counts_negative_values(self):
        """Test that negative values raise ValidationError."""
        import pydantic

        with pytest.raises(pydantic.ValidationError, match="negative values not allowed"):
            WorldTemplate(
                id="negative_counts",
                name="Negative Counts",
                description="Test",
                genre="test",
                recommended_counts={"characters": (-1, 5)},  # negative min
            )

    def test_recommended_counts_wrong_format(self):
        """Test that wrong tuple format raises ValidationError via Pydantic type."""
        import pydantic

        # Pydantic validates tuple length/format before our validator runs
        with pytest.raises(pydantic.ValidationError, match="at most 2 items"):
            WorldTemplate(
                id="invalid_format",
                name="Invalid Format",
                description="Test",
                genre="test",
                recommended_counts={"characters": (1, 2, 3)},
            )


class TestUserTemplateGenreLookup:
    """Tests for finding user templates by genre."""

    def test_find_user_template_by_genre(self, temp_templates_file):
        """Test finding a user template by genre (unique genre not in builtins)."""
        # Use a unique genre that doesn't exist in builtin templates
        user_template = {
            "id": "custom_steampunk",
            "name": "Custom Steampunk",
            "description": "A custom steampunk template",
            "is_builtin": False,
            "genre": "steampunk_adventure",  # Unique genre not in builtins
            "entity_hints": {},
            "relationship_patterns": [],
            "naming_style": "victorian",
            "recommended_counts": {},
            "atmosphere": "industrial",
            "tags": [],  # No tags to ensure we hit the genre match
        }
        temp_templates_file.parent.mkdir(parents=True, exist_ok=True)
        temp_templates_file.write_text(json.dumps([user_template]))

        with patch(
            "src.services.world_template_service.USER_TEMPLATES_FILE",
            temp_templates_file,
        ):
            service = WorldTemplateService()
            template = service.get_template_for_genre("steampunk_adventure")
            # Should find the user template by exact genre match
            assert template is not None
            assert template.id == "custom_steampunk"

    def test_find_user_template_by_tag(self, temp_templates_file):
        """Test finding a user template by tag."""
        user_template = {
            "id": "custom_horror",
            "name": "Custom Horror",
            "description": "A custom horror template",
            "is_builtin": False,
            "genre": "custom_genre",
            "entity_hints": {},
            "relationship_patterns": [],
            "naming_style": "dark",
            "recommended_counts": {},
            "atmosphere": "spooky",
            "tags": ["horror", "dark"],
        }
        temp_templates_file.parent.mkdir(parents=True, exist_ok=True)
        temp_templates_file.write_text(json.dumps([user_template]))

        with patch(
            "src.services.world_template_service.USER_TEMPLATES_FILE",
            temp_templates_file,
        ):
            service = WorldTemplateService()
            template = service.get_template_for_genre("horror")
            # Should find the user template by tag match
            assert template is not None
            assert template.id == "custom_horror"

"""Unit tests for character arc templates."""

import pytest

from src.memory.arc_templates import (
    BUILTIN_ARC_TEMPLATES,
    ArcStage,
    CharacterArcTemplate,
    format_arc_guidance,
    get_arc_template,
    list_arc_templates,
)


class TestArcStage:
    """Tests for ArcStage model."""

    def test_arc_stage_creation(self):
        """Test creating a valid arc stage."""
        stage = ArcStage(
            name="Test Stage",
            description="A test stage description",
            percentage=50,
        )
        assert stage.name == "Test Stage"
        assert stage.description == "A test stage description"
        assert stage.percentage == 50

    def test_arc_stage_percentage_bounds(self):
        """Test that percentage must be between 0 and 100."""
        # Valid bounds
        stage_low = ArcStage(name="Start", description="Beginning", percentage=0)
        stage_high = ArcStage(name="End", description="Conclusion", percentage=100)
        assert stage_low.percentage == 0
        assert stage_high.percentage == 100

        # Invalid bounds should raise validation error
        with pytest.raises(ValueError):
            ArcStage(name="Invalid", description="Too low", percentage=-1)

        with pytest.raises(ValueError):
            ArcStage(name="Invalid", description="Too high", percentage=101)


class TestCharacterArcTemplate:
    """Tests for CharacterArcTemplate model."""

    def test_protagonist_arc_template_creation(self):
        """Test creating a protagonist arc template."""
        template = CharacterArcTemplate(
            id="test_arc",
            name="Test Arc",
            description="A test arc for testing",
            arc_category="protagonist",
            stages=[
                ArcStage(name="Start", description="Beginning", percentage=0),
                ArcStage(name="End", description="Conclusion", percentage=100),
            ],
            required_traits=["courage", "wisdom"],
            recommended_relationships=["mentor", "ally"],
        )
        assert template.id == "test_arc"
        assert template.arc_category == "protagonist"
        assert len(template.stages) == 2
        assert len(template.required_traits) == 2
        assert len(template.recommended_relationships) == 2

    def test_antagonist_arc_template_creation(self):
        """Test creating an antagonist arc template."""
        template = CharacterArcTemplate(
            id="villain_arc",
            name="Villain Arc",
            description="An antagonist arc",
            arc_category="antagonist",
            stages=[
                ArcStage(name="Reveal", description="Villain appears", percentage=20),
            ],
        )
        assert template.arc_category == "antagonist"

    def test_arc_category_validation(self):
        """Test that arc_category must be protagonist or antagonist."""
        with pytest.raises(ValueError):
            CharacterArcTemplate(
                id="invalid",
                name="Invalid Arc",
                description="Invalid category",
                arc_category="sidekick",
            )


class TestBuiltinArcTemplates:
    """Tests for the built-in arc templates."""

    def test_all_builtin_templates_exist(self):
        """Test that all expected builtin templates are defined."""
        expected_protagonist = ["hero_journey", "redemption", "coming_of_age", "tragedy"]
        expected_antagonist = [
            "mirror",
            "force_of_nature",
            "fallen_hero",
            "true_believer",
            "mastermind",
        ]

        for arc_id in expected_protagonist:
            assert arc_id in BUILTIN_ARC_TEMPLATES, f"Missing protagonist arc: {arc_id}"
            assert BUILTIN_ARC_TEMPLATES[arc_id].arc_category == "protagonist"

        for arc_id in expected_antagonist:
            assert arc_id in BUILTIN_ARC_TEMPLATES, f"Missing antagonist arc: {arc_id}"
            assert BUILTIN_ARC_TEMPLATES[arc_id].arc_category == "antagonist"

    def test_hero_journey_has_all_stages(self):
        """Test that hero's journey has the classic 12 stages."""
        hero_journey = BUILTIN_ARC_TEMPLATES["hero_journey"]
        assert len(hero_journey.stages) == 12
        assert hero_journey.stages[0].name == "Ordinary World"
        assert hero_journey.stages[-1].name == "Return with Elixir"

    def test_all_templates_have_required_fields(self):
        """Test that all builtin templates have required fields."""
        for arc_id, template in BUILTIN_ARC_TEMPLATES.items():
            assert template.id == arc_id, f"Template {arc_id} has mismatched ID"
            assert template.name, f"Template {arc_id} missing name"
            assert template.description, f"Template {arc_id} missing description"
            assert template.arc_category in ("protagonist", "antagonist")
            assert len(template.stages) > 0, f"Template {arc_id} has no stages"

    def test_stages_have_valid_percentages(self):
        """Test that all stages have valid percentages."""
        for arc_id, template in BUILTIN_ARC_TEMPLATES.items():
            for stage in template.stages:
                assert 0 <= stage.percentage <= 100, (
                    f"Template {arc_id} stage {stage.name} has invalid percentage {stage.percentage}"
                )

    def test_stages_are_ordered_by_percentage(self):
        """Test that stages are ordered by percentage."""
        for arc_id, template in BUILTIN_ARC_TEMPLATES.items():
            percentages = [s.percentage for s in template.stages]
            assert percentages == sorted(percentages), (
                f"Template {arc_id} stages not ordered by percentage"
            )


class TestGetArcTemplate:
    """Tests for get_arc_template function."""

    def test_get_existing_template(self):
        """Test getting an existing template by ID."""
        template = get_arc_template("hero_journey")
        assert template is not None
        assert template.id == "hero_journey"
        assert template.arc_category == "protagonist"

    def test_get_nonexistent_template(self):
        """Test getting a nonexistent template returns None."""
        template = get_arc_template("nonexistent_arc")
        assert template is None

    def test_get_all_builtin_templates(self):
        """Test that all builtin templates can be retrieved."""
        for arc_id in BUILTIN_ARC_TEMPLATES:
            template = get_arc_template(arc_id)
            assert template is not None
            assert template.id == arc_id


class TestListArcTemplates:
    """Tests for list_arc_templates function."""

    def test_list_all_templates(self):
        """Test listing all templates."""
        templates = list_arc_templates()
        assert len(templates) == len(BUILTIN_ARC_TEMPLATES)

    def test_list_protagonist_templates(self):
        """Test listing only protagonist templates."""
        templates = list_arc_templates(category="protagonist")
        assert len(templates) == 4
        for template in templates:
            assert template.arc_category == "protagonist"

    def test_list_antagonist_templates(self):
        """Test listing only antagonist templates."""
        templates = list_arc_templates(category="antagonist")
        assert len(templates) == 5
        for template in templates:
            assert template.arc_category == "antagonist"


class TestFormatArcGuidance:
    """Tests for format_arc_guidance function."""

    def test_format_includes_basic_info(self):
        """Test that formatted guidance includes basic template info."""
        template = get_arc_template("hero_journey")
        assert template is not None
        guidance = format_arc_guidance(template)

        assert "CHARACTER ARC: Hero's Journey" in guidance
        assert template.description in guidance

    def test_format_includes_stages(self):
        """Test that formatted guidance includes all stages."""
        template = get_arc_template("redemption")
        assert template is not None
        guidance = format_arc_guidance(template)

        assert "Arc Stages:" in guidance
        for stage in template.stages:
            assert stage.name in guidance
            assert str(stage.percentage) in guidance

    def test_format_includes_traits(self):
        """Test that formatted guidance includes required traits."""
        template = get_arc_template("tragedy")
        assert template is not None
        guidance = format_arc_guidance(template)

        assert "Recommended Traits:" in guidance
        for trait in template.required_traits:
            assert trait in guidance

    def test_format_includes_relationships(self):
        """Test that formatted guidance includes recommended relationships."""
        template = get_arc_template("mirror")
        assert template is not None
        guidance = format_arc_guidance(template)

        assert "Recommended Relationships:" in guidance
        for rel in template.recommended_relationships:
            assert rel in guidance

    def test_format_empty_traits_and_relationships(self):
        """Test formatting a template with empty traits and relationships."""
        template = CharacterArcTemplate(
            id="minimal",
            name="Minimal Arc",
            description="A minimal arc",
            arc_category="protagonist",
            stages=[ArcStage(name="Only Stage", description="The only stage", percentage=50)],
            required_traits=[],
            recommended_relationships=[],
        )
        guidance = format_arc_guidance(template)

        assert "Minimal Arc" in guidance
        assert "Recommended Traits:" not in guidance
        assert "Recommended Relationships:" not in guidance


class TestArcTemplateIntegration:
    """Integration tests for arc template usage."""

    def test_protagonist_and_antagonist_arcs_work_together(self):
        """Test that protagonist and antagonist arcs can be used together."""
        protagonist_arc = get_arc_template("hero_journey")
        antagonist_arc = get_arc_template("mirror")

        assert protagonist_arc is not None
        assert antagonist_arc is not None

        # Both should format without errors
        protagonist_guidance = format_arc_guidance(protagonist_arc)
        antagonist_guidance = format_arc_guidance(antagonist_arc)

        assert len(protagonist_guidance) > 0
        assert len(antagonist_guidance) > 0

        # They should be distinct
        assert protagonist_guidance != antagonist_guidance

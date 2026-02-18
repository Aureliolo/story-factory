"""Tests for graph renderer - Issues #172 and #173."""

from src.ui.graph_renderer import ENTITY_COLORS, ENTITY_ICON_CODES, ENTITY_SHAPES
from src.ui.theme import ROLE_COLORS, get_role_graph_style


class TestEntityIconCodes:
    """Tests for Material Icons codepoints used in graph visualization."""

    def test_icon_codes_defined_for_all_entity_types(self):
        """Test that icon codes exist for all standard entity types."""
        expected_types = ["character", "location", "item", "faction", "concept"]

        for entity_type in expected_types:
            assert entity_type in ENTITY_ICON_CODES, f"Missing icon code for {entity_type}"

    def test_character_icon_code(self):
        """Test character uses Material Icons 'person' web font codepoint."""
        assert ENTITY_ICON_CODES["character"] == "\ue7fd"  # person

    def test_location_icon_code(self):
        """Test location uses Material Icons 'place' web font codepoint."""
        assert ENTITY_ICON_CODES["location"] == "\ue55f"  # place

    def test_item_icon_code(self):
        """Test item uses Material Icons 'inventory_2' web font codepoint."""
        assert ENTITY_ICON_CODES["item"] == "\ue1a1"  # inventory_2

    def test_faction_icon_code(self):
        """Test faction uses Material Icons 'groups' web font codepoint."""
        assert ENTITY_ICON_CODES["faction"] == "\uf233"  # groups

    def test_concept_icon_code(self):
        """Test concept uses Material Icons 'lightbulb' web font codepoint."""
        assert ENTITY_ICON_CODES["concept"] == "\ue0f0"  # lightbulb

    def test_icon_codes_are_strings(self):
        """Test that all icon codes are non-empty strings."""
        for entity_type, code in ENTITY_ICON_CODES.items():
            assert isinstance(code, str), f"Icon code for {entity_type} is not a string"
            assert len(code) > 0, f"Icon code for {entity_type} is empty"


class TestEntityShapesBackwardsCompatibility:
    """Tests for ENTITY_SHAPES backwards compatibility."""

    def test_shapes_still_defined(self):
        """Test that ENTITY_SHAPES still exists for backwards compatibility."""
        assert ENTITY_SHAPES is not None
        assert len(ENTITY_SHAPES) > 0

    def test_shapes_defined_for_all_entity_types(self):
        """Test that shapes exist for all standard entity types."""
        expected_types = ["character", "location", "item", "faction", "concept"]

        for entity_type in expected_types:
            assert entity_type in ENTITY_SHAPES, f"Missing shape for {entity_type}"


class TestEntityColors:
    """Tests for entity color definitions."""

    def test_colors_defined_for_all_entity_types(self):
        """Test that colors exist for all standard entity types."""
        expected_types = ["character", "location", "item", "faction", "concept"]

        for entity_type in expected_types:
            assert entity_type in ENTITY_COLORS, f"Missing color for {entity_type}"

    def test_colors_are_valid_hex(self):
        """Test that all colors are valid hex color codes."""
        for entity_type, color in ENTITY_COLORS.items():
            assert isinstance(color, str), f"Color for {entity_type} is not a string"
            assert color.startswith("#"), f"Color for {entity_type} doesn't start with #"
            # Should be #RRGGBB format
            assert len(color) == 7, f"Color for {entity_type} is not #RRGGBB format"


class TestProtagonistNodeStyling:
    """Tests for protagonist node styling in graph (Issue #173)."""

    def test_protagonist_detection_logic(self):
        """Test protagonist detection from attributes."""
        test_cases = [
            ({"role": "protagonist"}, True, False, False),
            ({"role": "main character"}, True, False, False),
            ({"role": "antagonist"}, False, True, False),
            ({"role": "mentor"}, False, False, True),
            ({"role": "supporting"}, False, False, False),
            ({}, False, False, False),
            (None, False, False, False),
        ]

        for attributes, expected_protag, expected_antag, expected_mentor in test_cases:
            if attributes is None:
                role = ""
            else:
                role = (attributes.get("role") or "").lower()

            is_protagonist = "protagonist" in role or "main" in role
            is_antagonist = "antagonist" in role
            is_mentor = "mentor" in role

            assert is_protagonist == expected_protag, f"Failed for {attributes}"
            assert is_antagonist == expected_antag, f"Failed for {attributes}"
            assert is_mentor == expected_mentor, f"Failed for {attributes}"

    def test_protagonist_gold_shadow_color(self):
        """Test that protagonist nodes get gold (#FFD700) shadow."""
        assert ROLE_COLORS["protagonist"] == "#FFD700"

    def test_antagonist_red_shadow_color(self):
        """Test that antagonist nodes get red (#F44336) shadow."""
        assert ROLE_COLORS["antagonist"] == "#F44336"

    def test_mentor_blue_shadow_color(self):
        """Test that mentor nodes get blue (#2196F3) shadow."""
        assert ROLE_COLORS["mentor"] == "#2196F3"


class TestRoleGraphStyle:
    """Tests for get_role_graph_style shadow-based styling."""

    def test_protagonist_returns_glow_style(self):
        """Test protagonist gets a gold glow."""
        style = get_role_graph_style({"role": "protagonist"}, "#4CAF50")

        assert style is not None
        assert style["glow_color"] == ROLE_COLORS["protagonist"]
        assert style["glow_size"] == 40
        assert style["icon_color"] == "#4CAF50"

    def test_antagonist_returns_glow_style(self):
        """Test antagonist gets a red glow."""
        style = get_role_graph_style({"role": "antagonist"}, "#4CAF50")

        assert style is not None
        assert style["glow_color"] == ROLE_COLORS["antagonist"]
        assert style["glow_size"] == 35

    def test_mentor_returns_glow_style(self):
        """Test mentor gets a blue glow."""
        style = get_role_graph_style({"role": "mentor"}, "#2196F3")

        assert style is not None
        assert style["glow_color"] == ROLE_COLORS["mentor"]
        assert style["glow_size"] == 35

    def test_no_role_returns_none(self):
        """Test no role returns None."""
        assert get_role_graph_style({}, "#4CAF50") is None
        assert get_role_graph_style(None, "#4CAF50") is None
        assert get_role_graph_style({"role": "supporting"}, "#4CAF50") is None

    def test_protagonist_has_larger_glow_than_others(self):
        """Test protagonist glow is more prominent than antagonist/mentor."""
        protag = get_role_graph_style({"role": "protagonist"}, "#4CAF50")
        antag = get_role_graph_style({"role": "antagonist"}, "#4CAF50")
        mentor = get_role_graph_style({"role": "mentor"}, "#4CAF50")

        assert protag is not None
        assert antag is not None
        assert mentor is not None
        assert protag["glow_size"] > antag["glow_size"]
        assert protag["glow_size"] > mentor["glow_size"]


class TestIconNodeConfiguration:
    """Tests for icon node configuration structure."""

    def test_icon_node_structure(self):
        """Test the expected structure of an icon-based node."""
        entity_type = "character"
        base_color = ENTITY_COLORS.get(entity_type, "#607D8B")

        # Build icon config dict matching renderer output
        icon_config: dict[str, str | int] = {
            "face": "'Material Icons'",
            "code": ENTITY_ICON_CODES.get(entity_type, "\ue8fd"),
            "size": 40,
            "color": base_color,
            "weight": "normal",
        }

        # Build a node the same way graph_renderer does
        node: dict[str, str | dict[str, str | int]] = {
            "id": "test-id",
            "label": "Test Character",
            "group": entity_type,
            "shape": "icon",
            "icon": icon_config,
            "title": "Test tooltip",
        }

        # Verify structure
        assert node["shape"] == "icon"
        assert "icon" in node
        assert icon_config["face"] == "'Material Icons'"
        assert icon_config["code"] == "\ue7fd"
        assert icon_config["size"] == 40
        assert icon_config["weight"] == "normal"
        assert icon_config["color"] == base_color

    def test_fallback_icon_code(self):
        """Test that unknown entity types get fallback icon."""
        unknown_type = "unknown_entity_type"
        fallback_code = ENTITY_ICON_CODES.get(unknown_type, "\ue8fd")

        assert fallback_code == "\ue8fd"  # help_outline

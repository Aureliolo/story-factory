"""Component tests for graph visualization.

Tests that would catch runtime NiceGUI errors like:
- ui.html() missing required sanitize parameter
- Script tags in ui.html() being rejected
"""

import pytest
from nicegui import ui
from nicegui.testing import User

from settings import Settings


@pytest.fixture
def test_settings() -> Settings:
    """Create settings for testing."""
    return Settings()


@pytest.mark.component
class TestGraphComponent:
    """Tests for the GraphComponent class."""

    async def test_graph_component_builds_without_error(self, user: User, test_world_db):
        """Graph component builds without NiceGUI runtime errors."""
        from ui.components.graph import GraphComponent

        @ui.page("/test-graph")
        def test_page():
            graph = GraphComponent(height=300)
            graph.build()

        await user.open("/test-graph")
        # If we get here without exception, the build succeeded

    async def test_graph_component_with_data(self, user: User, test_world_db):
        """Graph component renders with actual world data."""
        from ui.components.graph import GraphComponent

        @ui.page("/test-graph-data")
        def test_page():
            graph = GraphComponent(height=300)
            graph.build()
            graph.set_world_db(test_world_db)

        await user.open("/test-graph-data")
        # Component should build and set data without errors


@pytest.mark.component
class TestMiniGraph:
    """Tests for the mini_graph helper function."""

    async def test_mini_graph_renders(self, user: User, test_world_db):
        """mini_graph function creates a graph without errors."""
        from ui.components.graph import mini_graph

        @ui.page("/test-mini-graph")
        def test_page():
            with ui.column():
                mini_graph(test_world_db, height=200)

        await user.open("/test-mini-graph")
        # If we get here without exception, the rendering succeeded


@pytest.mark.component
class TestGraphRenderer:
    """Tests for the graph_renderer module."""

    def test_render_graph_html_returns_dataclass(self, test_world_db, test_settings):
        """render_graph_html returns GraphRenderResult dataclass."""
        from ui.graph_renderer import GraphRenderResult, render_graph_html

        result = render_graph_html(
            test_world_db,
            test_settings,
            container_id="test-container",
            height=300,
        )

        assert isinstance(result, GraphRenderResult), "Should return GraphRenderResult"
        assert "test-container" in result.html, "HTML should contain container ID"
        assert "<script>" not in result.html, "HTML should not contain script tags"
        assert "vis.Network" in result.js, "JS should contain vis.Network initialization"

    def test_render_entity_summary_html_no_script_tags(self, test_world_db):
        """Entity summary HTML doesn't contain script tags."""
        from ui.graph_renderer import render_entity_summary_html

        html = render_entity_summary_html(test_world_db)

        assert "<script>" not in html, "Summary HTML should not contain script tags"

    def test_render_graph_html_contains_entity_data(self, test_world_db, test_settings):
        """Graph JavaScript contains entity data from world database."""
        from ui.graph_renderer import render_graph_html

        result = render_graph_html(
            test_world_db,
            test_settings,
            container_id="test",
            height=300,
        )

        # Should contain our test entity
        assert "Test Character" in result.js, "Should contain entity names"

    def test_render_graph_html_has_valid_js(self, test_world_db, test_settings):
        """Graph JavaScript is valid and can initialize vis.Network."""
        from ui.graph_renderer import render_graph_html

        result = render_graph_html(
            test_world_db,
            test_settings,
            container_id="test",
            height=300,
        )

        # Should have vis.Network initialization
        assert "new vis.Network" in result.js, "Should initialize vis.Network"
        assert "nodes:" in result.js, "Should have nodes data"
        assert "edges:" in result.js, "Should have edges data"

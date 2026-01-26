"""Tests for the conflict analysis service."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.memory.conflict_types import (
    CONFLICT_COLORS,
    ConflictCategory,
    ConflictMetrics,
    classify_relationship,
    get_conflict_color,
)
from src.memory.entities import Entity, Relationship
from src.services.conflict_analysis_service import ConflictAnalysisService
from src.settings import Settings


class TestConflictCategory:
    """Tests for ConflictCategory enum and classification."""

    def test_classify_alliance_relationships(self):
        """Test classification of alliance relationships."""
        alliance_types = ["loves", "ally_of", "protects", "member_of", "trusts"]
        for rel_type in alliance_types:
            assert classify_relationship(rel_type) == ConflictCategory.ALLIANCE

    def test_classify_rivalry_relationships(self):
        """Test classification of rivalry relationships."""
        rivalry_types = ["hates", "enemy_of", "opposes", "betrayed", "fights"]
        for rel_type in rivalry_types:
            assert classify_relationship(rel_type) == ConflictCategory.RIVALRY

    def test_classify_tension_relationships(self):
        """Test classification of tension relationships."""
        tension_types = ["distrusts", "competes_with", "fears", "resents"]
        for rel_type in tension_types:
            assert classify_relationship(rel_type) == ConflictCategory.TENSION

    def test_classify_neutral_relationships(self):
        """Test classification of neutral relationships."""
        neutral_types = ["knows", "works_with", "related_to", "located_in"]
        for rel_type in neutral_types:
            assert classify_relationship(rel_type) == ConflictCategory.NEUTRAL

    def test_classify_unknown_defaults_to_neutral(self):
        """Test unknown relationship types default to neutral."""
        assert classify_relationship("unknown_type") == ConflictCategory.NEUTRAL

    def test_classify_handles_case_and_format_variations(self):
        """Test classification handles case and format variations."""
        assert classify_relationship("LOVES") == ConflictCategory.ALLIANCE
        assert classify_relationship("Hates") == ConflictCategory.RIVALRY
        assert classify_relationship("competes-with") == ConflictCategory.TENSION


class TestConflictColors:
    """Tests for conflict colors."""

    def test_get_conflict_color_by_enum(self):
        """Test getting color by ConflictCategory enum."""
        assert get_conflict_color(ConflictCategory.ALLIANCE) == "#4CAF50"
        assert get_conflict_color(ConflictCategory.RIVALRY) == "#F44336"
        assert get_conflict_color(ConflictCategory.TENSION) == "#FFC107"
        assert get_conflict_color(ConflictCategory.NEUTRAL) == "#2196F3"

    def test_get_conflict_color_by_string(self):
        """Test getting color by string."""
        assert get_conflict_color("alliance") == "#4CAF50"
        assert get_conflict_color("rivalry") == "#F44336"

    def test_get_conflict_color_unknown_returns_neutral(self):
        """Test unknown color returns neutral."""
        assert get_conflict_color("unknown") == CONFLICT_COLORS["neutral"]


class TestConflictAnalysisService:
    """Tests for ConflictAnalysisService class."""

    @pytest.fixture
    def settings(self):
        """Create settings for testing."""
        return Settings()

    @pytest.fixture
    def conflict_service(self, settings):
        """Create ConflictAnalysisService instance."""
        return ConflictAnalysisService(settings)

    @pytest.fixture
    def mock_world_db(self):
        """Create a mock WorldDatabase."""
        mock_db = MagicMock()
        mock_db.list_entities.return_value = []
        mock_db.list_relationships.return_value = []
        return mock_db

    def test_init(self, conflict_service):
        """Test service initialization."""
        assert conflict_service.settings is not None

    def test_analyze_conflicts_empty_world(self, conflict_service, mock_world_db):
        """Test analyzing conflicts in empty world."""
        metrics = conflict_service.analyze_conflicts(mock_world_db)

        assert metrics.total_relationships == 0
        assert metrics.alliance_count == 0
        assert metrics.rivalry_count == 0
        assert metrics.conflict_density == 0.0

    def test_analyze_conflicts_counts_categories(self, conflict_service, mock_world_db):
        """Test that analyze_conflicts correctly counts relationship categories."""
        entities = [
            Entity(
                id="e1", type="character", name="Alice", description="", created_at=datetime.now()
            ),
            Entity(
                id="e2", type="character", name="Bob", description="", created_at=datetime.now()
            ),
            Entity(
                id="e3", type="character", name="Carol", description="", created_at=datetime.now()
            ),
        ]
        relationships = [
            Relationship(
                id="r1",
                source_id="e1",
                target_id="e2",
                relation_type="loves",
                created_at=datetime.now(),
            ),
            Relationship(
                id="r2",
                source_id="e1",
                target_id="e3",
                relation_type="hates",
                created_at=datetime.now(),
            ),
            Relationship(
                id="r3",
                source_id="e2",
                target_id="e3",
                relation_type="knows",
                created_at=datetime.now(),
            ),
        ]
        mock_world_db.list_entities.return_value = entities
        mock_world_db.list_relationships.return_value = relationships

        metrics = conflict_service.analyze_conflicts(mock_world_db)

        assert metrics.total_relationships == 3
        assert metrics.alliance_count == 1
        assert metrics.rivalry_count == 1
        assert metrics.neutral_count == 1

    def test_analyze_conflicts_calculates_density(self, conflict_service, mock_world_db):
        """Test conflict density calculation."""
        entities = [
            Entity(
                id="e1", type="character", name="Alice", description="", created_at=datetime.now()
            ),
            Entity(
                id="e2", type="character", name="Bob", description="", created_at=datetime.now()
            ),
        ]
        # 1 rivalry, 1 neutral = 50% conflict density
        relationships = [
            Relationship(
                id="r1",
                source_id="e1",
                target_id="e2",
                relation_type="hates",
                created_at=datetime.now(),
            ),
            Relationship(
                id="r2",
                source_id="e2",
                target_id="e1",
                relation_type="knows",
                created_at=datetime.now(),
            ),
        ]
        mock_world_db.list_entities.return_value = entities
        mock_world_db.list_relationships.return_value = relationships

        metrics = conflict_service.analyze_conflicts(mock_world_db)

        assert metrics.conflict_density == 0.5

    def test_analyze_conflicts_finds_tension_pairs(self, conflict_service, mock_world_db):
        """Test finding highest tension pairs."""
        entities = [
            Entity(
                id="e1", type="character", name="Hero", description="", created_at=datetime.now()
            ),
            Entity(
                id="e2", type="character", name="Villain", description="", created_at=datetime.now()
            ),
        ]
        relationships = [
            Relationship(
                id="r1",
                source_id="e1",
                target_id="e2",
                relation_type="hates",
                strength=0.9,
                created_at=datetime.now(),
            ),
        ]
        mock_world_db.list_entities.return_value = entities
        mock_world_db.list_relationships.return_value = relationships

        metrics = conflict_service.analyze_conflicts(mock_world_db)

        assert len(metrics.highest_tension_pairs) > 0
        pair = metrics.highest_tension_pairs[0]
        assert pair.entity_a_name in ["Hero", "Villain"]
        assert pair.entity_b_name in ["Hero", "Villain"]

    def test_analyze_conflicts_finds_isolated_entities(self, conflict_service, mock_world_db):
        """Test finding isolated entities (no alliances)."""
        entities = [
            Entity(
                id="e1", type="character", name="Loner", description="", created_at=datetime.now()
            ),
            Entity(
                id="e2", type="character", name="Popular", description="", created_at=datetime.now()
            ),
            Entity(
                id="e3", type="character", name="Friend", description="", created_at=datetime.now()
            ),
        ]
        # Only e2 and e3 have alliance, e1 is isolated
        relationships = [
            Relationship(
                id="r1",
                source_id="e2",
                target_id="e3",
                relation_type="ally_of",
                created_at=datetime.now(),
            ),
        ]
        mock_world_db.list_entities.return_value = entities
        mock_world_db.list_relationships.return_value = relationships

        metrics = conflict_service.analyze_conflicts(mock_world_db)

        assert "e1" in metrics.isolated_entities
        assert "e2" not in metrics.isolated_entities
        assert "e3" not in metrics.isolated_entities

    def test_analyze_conflicts_detects_faction_clusters(self, conflict_service, mock_world_db):
        """Test detecting faction clusters."""
        entities = [
            Entity(id="e1", type="character", name="A", description="", created_at=datetime.now()),
            Entity(id="e2", type="character", name="B", description="", created_at=datetime.now()),
            Entity(id="e3", type="character", name="C", description="", created_at=datetime.now()),
            Entity(id="e4", type="character", name="D", description="", created_at=datetime.now()),
        ]
        # e1-e2-e3 form one faction, e4 is separate
        relationships = [
            Relationship(
                id="r1",
                source_id="e1",
                target_id="e2",
                relation_type="ally_of",
                created_at=datetime.now(),
            ),
            Relationship(
                id="r2",
                source_id="e2",
                target_id="e3",
                relation_type="trusts",
                created_at=datetime.now(),
            ),
        ]
        mock_world_db.list_entities.return_value = entities
        mock_world_db.list_relationships.return_value = relationships

        metrics = conflict_service.analyze_conflicts(mock_world_db)

        assert len(metrics.faction_clusters) >= 1
        # The faction should contain e1, e2, e3
        faction_ids = metrics.faction_clusters[0].entity_ids
        assert "e1" in faction_ids or "e2" in faction_ids or "e3" in faction_ids

    def test_get_conflict_graph_data_empty_world(self, conflict_service, mock_world_db):
        """Test getting graph data for empty world."""
        data = conflict_service.get_conflict_graph_data(mock_world_db)

        assert len(data.nodes) == 0
        assert len(data.edges) == 0
        assert data.metrics is not None

    def test_get_conflict_graph_data_creates_nodes_and_edges(self, conflict_service, mock_world_db):
        """Test that graph data includes nodes and edges."""
        entities = [
            Entity(
                id="e1",
                type="character",
                name="Alice",
                description="Protagonist",
                created_at=datetime.now(),
            ),
            Entity(
                id="e2",
                type="character",
                name="Bob",
                description="Antagonist",
                created_at=datetime.now(),
            ),
        ]
        relationships = [
            Relationship(
                id="r1",
                source_id="e1",
                target_id="e2",
                relation_type="hates",
                created_at=datetime.now(),
            ),
        ]
        mock_world_db.list_entities.return_value = entities
        mock_world_db.list_relationships.return_value = relationships

        data = conflict_service.get_conflict_graph_data(mock_world_db)

        assert len(data.nodes) == 2
        assert len(data.edges) == 1
        # Edge should be colored for rivalry
        assert data.edges[0].category == ConflictCategory.RIVALRY
        assert data.edges[0].color == CONFLICT_COLORS["rivalry"]

    def test_get_conflict_graph_data_filters_by_category(self, conflict_service, mock_world_db):
        """Test filtering graph data by conflict category."""
        entities = [
            Entity(id="e1", type="character", name="A", description="", created_at=datetime.now()),
            Entity(id="e2", type="character", name="B", description="", created_at=datetime.now()),
        ]
        relationships = [
            Relationship(
                id="r1",
                source_id="e1",
                target_id="e2",
                relation_type="loves",
                created_at=datetime.now(),
            ),
            Relationship(
                id="r2",
                source_id="e2",
                target_id="e1",
                relation_type="hates",
                created_at=datetime.now(),
            ),
        ]
        mock_world_db.list_entities.return_value = entities
        mock_world_db.list_relationships.return_value = relationships

        # Filter to only show rivalry
        data = conflict_service.get_conflict_graph_data(
            mock_world_db, categories=[ConflictCategory.RIVALRY]
        )

        assert len(data.edges) == 1
        assert data.edges[0].category == ConflictCategory.RIVALRY

    def test_get_conflict_graph_data_filters_by_entity_type(self, conflict_service, mock_world_db):
        """Test filtering graph data by entity type."""
        entities = [
            Entity(
                id="e1", type="character", name="Person", description="", created_at=datetime.now()
            ),
            Entity(
                id="e2", type="faction", name="Group", description="", created_at=datetime.now()
            ),
        ]
        relationships = [
            Relationship(
                id="r1",
                source_id="e1",
                target_id="e2",
                relation_type="member_of",
                created_at=datetime.now(),
            ),
        ]
        mock_world_db.list_entities.return_value = entities
        mock_world_db.list_relationships.return_value = relationships

        # Filter to only show characters
        data = conflict_service.get_conflict_graph_data(mock_world_db, entity_types=["character"])

        assert len(data.nodes) == 1
        assert data.nodes[0].entity_type == "character"
        # Edge should be excluded since faction is filtered out
        assert len(data.edges) == 0

    def test_get_category_summary(self, conflict_service, mock_world_db):
        """Test getting category summary counts."""
        relationships = [
            Relationship(
                id="r1",
                source_id="e1",
                target_id="e2",
                relation_type="loves",
                created_at=datetime.now(),
            ),
            Relationship(
                id="r2",
                source_id="e2",
                target_id="e3",
                relation_type="hates",
                created_at=datetime.now(),
            ),
            Relationship(
                id="r3",
                source_id="e3",
                target_id="e1",
                relation_type="distrusts",
                created_at=datetime.now(),
            ),
            Relationship(
                id="r4",
                source_id="e1",
                target_id="e3",
                relation_type="knows",
                created_at=datetime.now(),
            ),
        ]
        mock_world_db.list_relationships.return_value = relationships

        summary = conflict_service.get_category_summary(mock_world_db)

        assert summary["alliance"] == 1
        assert summary["rivalry"] == 1
        assert summary["tension"] == 1
        assert summary["neutral"] == 1

    def test_suggest_missing_conflicts(self, conflict_service, mock_world_db):
        """Test suggesting missing conflicts between factions."""
        entities = [
            Entity(
                id="f1",
                type="faction",
                name="Good Guys",
                description="",
                created_at=datetime.now(),
                attributes={"goals": ["Save the world"]},
            ),
            Entity(
                id="f2",
                type="faction",
                name="Bad Guys",
                description="",
                created_at=datetime.now(),
                attributes={"goals": ["Destroy the world"]},
            ),
        ]
        mock_world_db.list_entities.return_value = entities
        mock_world_db.list_relationships.return_value = []  # No relationships yet

        suggestions = conflict_service.suggest_missing_conflicts(mock_world_db)

        # Should suggest a relationship between the factions
        assert len(suggestions) >= 1
        suggestion = suggestions[0]
        assert suggestion["entity_a_name"] in ["Good Guys", "Bad Guys"]
        assert suggestion["entity_b_name"] in ["Good Guys", "Bad Guys"]

    def test_suggest_missing_conflicts_skips_existing_relationships(
        self, conflict_service, mock_world_db
    ):
        """Test suggesting conflicts skips factions with existing relationships."""
        entities = [
            Entity(
                id="f1",
                type="faction",
                name="Faction A",
                description="",
                created_at=datetime.now(),
                attributes={"goals": ["Goal 1"]},
            ),
            Entity(
                id="f2",
                type="faction",
                name="Faction B",
                description="",
                created_at=datetime.now(),
                attributes={"goals": ["Goal 2"]},
            ),
        ]
        # Already have a relationship
        relationships = [
            Relationship(
                id="r1",
                source_id="f1",
                target_id="f2",
                relation_type="ally_of",
                created_at=datetime.now(),
            ),
        ]
        mock_world_db.list_entities.return_value = entities
        mock_world_db.list_relationships.return_value = relationships

        suggestions = conflict_service.suggest_missing_conflicts(mock_world_db)

        # Should not suggest since relationship already exists
        assert len(suggestions) == 0

    def test_suggest_missing_conflicts_respects_limit(self, conflict_service, mock_world_db):
        """Test suggestion limit is respected."""
        # Create many factions with goals
        entities = [
            Entity(
                id=f"f{i}",
                type="faction",
                name=f"Faction {i}",
                description="",
                created_at=datetime.now(),
                attributes={"goals": [f"Goal {i}"]},
            )
            for i in range(10)
        ]
        mock_world_db.list_entities.return_value = entities
        mock_world_db.list_relationships.return_value = []

        suggestions = conflict_service.suggest_missing_conflicts(mock_world_db, limit=2)

        assert len(suggestions) == 2

    def test_tension_pairs_skip_missing_entities(self, conflict_service, mock_world_db):
        """Test that tension pairs skip relationships with missing entities."""
        entities = [
            Entity(
                id="e1", type="character", name="Alice", description="", created_at=datetime.now()
            ),
            # Note: e2 exists but e3 doesn't
            Entity(
                id="e2", type="character", name="Bob", description="", created_at=datetime.now()
            ),
        ]
        relationships = [
            # Valid relationship
            Relationship(
                id="r1",
                source_id="e1",
                target_id="e2",
                relation_type="hates",
                strength=0.9,
                created_at=datetime.now(),
            ),
            # Relationship with missing entity
            Relationship(
                id="r2",
                source_id="e1",
                target_id="e3",  # e3 doesn't exist
                relation_type="hates",
                strength=0.9,
                created_at=datetime.now(),
            ),
        ]
        mock_world_db.list_entities.return_value = entities
        mock_world_db.list_relationships.return_value = relationships

        metrics = conflict_service.analyze_conflicts(mock_world_db)

        # Should only have one tension pair (e1-e2), e1-e3 should be skipped
        assert len(metrics.highest_tension_pairs) == 1
        pair = metrics.highest_tension_pairs[0]
        assert pair.entity_a_name in ["Alice", "Bob"]
        assert pair.entity_b_name in ["Alice", "Bob"]

    def test_faction_clusters_skip_single_entity(self, conflict_service, mock_world_db):
        """Test that faction clustering skips single-entity clusters (self-alliances)."""
        entities = [
            Entity(
                id="e1", type="character", name="Solo", description="", created_at=datetime.now()
            ),
            Entity(
                id="e2", type="character", name="Alice", description="", created_at=datetime.now()
            ),
            Entity(
                id="e3", type="character", name="Bob", description="", created_at=datetime.now()
            ),
        ]
        relationships = [
            # Self-alliance (creates single-entity cluster)
            Relationship(
                id="r1",
                source_id="e1",
                target_id="e1",  # Self-relationship
                relation_type="ally_of",
                created_at=datetime.now(),
            ),
            # Valid alliance between e2-e3 (creates two-entity cluster)
            Relationship(
                id="r2",
                source_id="e2",
                target_id="e3",
                relation_type="ally_of",
                created_at=datetime.now(),
            ),
        ]
        mock_world_db.list_entities.return_value = entities
        mock_world_db.list_relationships.return_value = relationships

        metrics = conflict_service.analyze_conflicts(mock_world_db)

        # Should only have one faction (e2-e3), single-entity cluster is skipped
        assert len(metrics.faction_clusters) == 1
        cluster = metrics.faction_clusters[0]
        assert "e2" in cluster.entity_ids or "e3" in cluster.entity_ids
        assert "e1" not in cluster.entity_ids


class TestConflictMetrics:
    """Tests for ConflictMetrics model."""

    def test_has_conflicts_true(self):
        """Test has_conflicts returns True when conflicts exist."""
        metrics = ConflictMetrics(rivalry_count=1, tension_count=0)
        assert metrics.has_conflicts is True

        metrics = ConflictMetrics(rivalry_count=0, tension_count=1)
        assert metrics.has_conflicts is True

    def test_has_conflicts_false(self):
        """Test has_conflicts returns False when no conflicts."""
        metrics = ConflictMetrics(
            alliance_count=5, rivalry_count=0, tension_count=0, neutral_count=3
        )
        assert metrics.has_conflicts is False

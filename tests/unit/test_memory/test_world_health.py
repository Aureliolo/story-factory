"""Tests for world health models and metrics."""

from src.memory.world_health import (
    Contradiction,
    EntityClaim,
    RelationshipSuggestion,
    WorldHealthMetrics,
)


class TestEntityClaim:
    """Tests for EntityClaim model."""

    def test_create_entity_claim(self):
        """Test creating an entity claim."""
        claim = EntityClaim(
            entity_id="char-001",
            entity_name="Alice",
            entity_type="character",
            claim="Alice is 25 years old",
            source_text="In her description, Alice is mentioned as being 25.",
        )

        assert claim.entity_id == "char-001"
        assert claim.entity_name == "Alice"
        assert claim.entity_type == "character"
        assert claim.claim == "Alice is 25 years old"

    def test_entity_claim_defaults(self):
        """Test entity claim with default values."""
        claim = EntityClaim(
            entity_id="char-001",
            entity_name="Alice",
            claim="Some claim",
        )

        assert claim.entity_type == ""
        assert claim.source_text == ""


class TestContradiction:
    """Tests for Contradiction model."""

    def test_create_contradiction(self):
        """Test creating a contradiction."""
        claim_a = EntityClaim(
            entity_id="char-001",
            entity_name="Alice",
            claim="Alice has blue eyes",
        )
        claim_b = EntityClaim(
            entity_id="char-001",
            entity_name="Alice",
            claim="Alice has green eyes",
        )

        contradiction = Contradiction(
            claim_a=claim_a,
            claim_b=claim_b,
            severity="high",
            explanation="Eye color cannot be both blue and green",
            resolution_suggestion="Choose one eye color and update the description",
            confidence=0.95,
        )

        assert contradiction.severity == "high"
        assert contradiction.confidence == 0.95
        assert "blue" in contradiction.claim_a.claim

    def test_contradiction_defaults(self):
        """Test contradiction with default values."""
        claim_a = EntityClaim(entity_id="1", entity_name="A", claim="X")
        claim_b = EntityClaim(entity_id="2", entity_name="B", claim="Y")

        contradiction = Contradiction(
            claim_a=claim_a,
            claim_b=claim_b,
            explanation="Test explanation",
        )

        assert contradiction.severity == "medium"
        assert contradiction.confidence == 0.5
        assert contradiction.resolution_suggestion == ""


class TestRelationshipSuggestion:
    """Tests for RelationshipSuggestion model."""

    def test_create_relationship_suggestion(self):
        """Test creating a relationship suggestion."""
        suggestion = RelationshipSuggestion(
            source_entity_id="char-001",
            source_entity_name="Alice",
            target_entity_id="char-002",
            target_entity_name="Bob",
            relation_type="ally_of",
            description="They work together to defeat the villain",
            confidence=0.85,
            bidirectional=True,
        )

        assert suggestion.source_entity_name == "Alice"
        assert suggestion.target_entity_name == "Bob"
        assert suggestion.relation_type == "ally_of"
        assert suggestion.bidirectional is True

    def test_relationship_suggestion_defaults(self):
        """Test relationship suggestion with default values."""
        suggestion = RelationshipSuggestion(
            source_entity_id="1",
            source_entity_name="A",
            target_entity_id="2",
            target_entity_name="B",
            relation_type="knows",
        )

        assert suggestion.confidence == 0.5
        assert suggestion.bidirectional is False
        assert suggestion.description == ""


class TestWorldHealthMetrics:
    """Tests for WorldHealthMetrics model."""

    def test_create_metrics(self):
        """Test creating world health metrics."""
        metrics = WorldHealthMetrics(
            total_entities=10,
            entity_counts={"character": 5, "location": 3, "item": 2},
            total_relationships=15,
            orphan_count=2,
            orphan_entities=[
                {"id": "1", "name": "Orphan1", "type": "character"},
                {"id": "2", "name": "Orphan2", "type": "location"},
            ],
            circular_count=1,
            circular_relationships=[{"edges": [], "length": 3}],
            average_quality=7.5,
            quality_distribution={"0-2": 0, "2-4": 1, "4-6": 2, "6-8": 4, "8-10": 3},
            low_quality_entities=[],
            relationship_density=1.5,
        )

        assert metrics.total_entities == 10
        assert metrics.orphan_count == 2
        assert metrics.relationship_density == 1.5

    def test_calculate_health_score_perfect(self):
        """Test health score calculation with no issues and perfect quality."""
        metrics = WorldHealthMetrics(
            total_entities=10,
            total_relationships=15,
            orphan_count=0,
            circular_count=0,
            average_quality=10.0,
            relationship_density=1.5,
        )

        score = metrics.calculate_health_score()

        # structural = min(100 + 10, 100) = 100
        # quality = 10.0 * 10 = 100
        # score = 100 * 0.6 + 100 * 0.4 = 100
        assert score == 100.0

    def test_calculate_health_score_empty_world(self):
        """Test health score calculation for empty world returns 0."""
        metrics = WorldHealthMetrics(
            total_entities=0,
            total_relationships=0,
        )

        score = metrics.calculate_health_score()

        assert score == 0.0

    def test_calculate_health_score_with_orphans(self):
        """Test health score calculation with orphan entities."""
        metrics = WorldHealthMetrics(
            total_entities=10,
            orphan_count=5,  # -2 per orphan = -10
            circular_count=0,
            average_quality=10.0,
            relationship_density=0.5,
        )

        score = metrics.calculate_health_score()

        # structural = 100 - 10 = 90
        # quality = 100
        # score = 90 * 0.6 + 100 * 0.4 = 54 + 40 = 94
        assert score == 94.0

    def test_calculate_health_score_with_circular(self):
        """Test health score calculation with circular relationships."""
        metrics = WorldHealthMetrics(
            total_entities=10,
            orphan_count=0,
            circular_count=3,  # -5 per cycle = -15
            average_quality=10.0,
            relationship_density=0.5,
        )

        score = metrics.calculate_health_score()

        # structural = 100 - 15 = 85
        # quality = 100
        # score = 85 * 0.6 + 100 * 0.4 = 51 + 40 = 91
        assert score == 91.0

    def test_calculate_health_score_with_zero_quality(self):
        """Test health score with zero quality (unscored entities)."""
        metrics = WorldHealthMetrics(
            total_entities=10,
            orphan_count=0,
            circular_count=0,
            average_quality=0.0,
            relationship_density=0.5,
        )

        score = metrics.calculate_health_score()

        # structural = 100, quality = 0
        # score = 100 * 0.6 + 0 * 0.4 = 60
        assert score == 60.0

    def test_calculate_health_score_caps_penalties(self):
        """Test that penalties are capped at their maximum values."""
        metrics = WorldHealthMetrics(
            total_entities=100,
            orphan_count=100,  # Would be -200 but capped at -20
            circular_count=100,  # Would be -500 but capped at -25
            contradiction_count=100,  # Would be -500 but capped at -25
            average_quality=0.0,
            relationship_density=0.0,
        )

        score = metrics.calculate_health_score()

        # structural = 100 - 20 - 25 - 25 = 30
        # quality = 0
        # score = 30 * 0.6 + 0 * 0.4 = 18
        assert score == 18.0

    def test_calculate_health_score_density_bonus(self):
        """Test density bonus in health score calculation."""
        # High density gets +10
        metrics_high = WorldHealthMetrics(
            total_entities=10,
            total_relationships=20,
            average_quality=10.0,
            relationship_density=2.0,
        )
        score_high = metrics_high.calculate_health_score()
        # structural = min(100 + 10, 100) = 100, quality = 100
        # score = 100 * 0.6 + 100 * 0.4 = 100
        assert score_high == 100.0

        # Medium density gets +5
        metrics_med = WorldHealthMetrics(
            total_entities=10,
            total_relationships=12,
            average_quality=10.0,
            relationship_density=1.2,
        )
        score_med = metrics_med.calculate_health_score()
        # structural = min(100 + 5, 100) = 100, quality = 100
        assert score_med == 100.0

    def test_generate_recommendations_orphans(self):
        """Test recommendation generation for orphan entities."""
        metrics = WorldHealthMetrics(
            orphan_count=3,
            orphan_entities=[
                {"id": "1", "name": "Lonely1", "type": "character"},
                {"id": "2", "name": "Lonely2", "type": "location"},
                {"id": "3", "name": "Lonely3", "type": "item"},
            ],
        )

        recs = metrics.generate_recommendations()

        assert len(recs) > 0
        assert any("orphan" in r.lower() or "relationship" in r.lower() for r in recs)

    def test_generate_recommendations_single_orphan_with_name(self):
        """Test recommendation for single orphan entity includes entity name."""
        metrics = WorldHealthMetrics(
            orphan_count=1,
            orphan_entities=[{"id": "1", "name": "LonelyHero", "type": "character"}],
        )

        recs = metrics.generate_recommendations()

        assert len(recs) > 0
        assert any("LonelyHero" in r for r in recs)
        assert any("1 entity" in r for r in recs)

    def test_generate_recommendations_single_orphan_empty_list_fallback(self):
        """Test recommendation fallback when orphan_count=1 but orphan_entities is empty."""
        # Edge case: orphan_count says 1 but the list is empty
        metrics = WorldHealthMetrics(
            orphan_count=1,
            orphan_entities=[],  # Empty but count says 1
        )

        recs = metrics.generate_recommendations()

        assert len(recs) > 0
        # Should use fallback message without crashing
        assert any("1 entity" in r and "relationship" in r.lower() for r in recs)

    def test_generate_recommendations_circular(self):
        """Test recommendation generation for circular relationships."""
        metrics = WorldHealthMetrics(
            circular_count=2,
            circular_relationships=[{"edges": [], "length": 3}, {"edges": [], "length": 4}],
        )

        recs = metrics.generate_recommendations()

        assert len(recs) > 0
        assert any("circular" in r.lower() for r in recs)

    def test_generate_recommendations_low_quality(self):
        """Test recommendation generation for low quality entities."""
        metrics = WorldHealthMetrics(
            low_quality_entities=[
                {"id": "1", "name": "Bad", "type": "character", "quality_score": 2.0},
            ],
        )

        recs = metrics.generate_recommendations()

        assert len(recs) > 0
        assert any("quality" in r.lower() for r in recs)

    def test_generate_recommendations_low_density(self):
        """Test recommendation generation for low relationship density."""
        metrics = WorldHealthMetrics(
            total_entities=10,
            relationship_density=0.5,
        )

        recs = metrics.generate_recommendations()

        assert len(recs) > 0
        assert any("density" in r.lower() for r in recs)

    def test_generate_recommendations_missing_locations(self):
        """Test recommendation generation when no locations exist."""
        metrics = WorldHealthMetrics(
            entity_counts={"character": 5, "location": 0},
        )

        recs = metrics.generate_recommendations()

        assert len(recs) > 0
        assert any("location" in r.lower() for r in recs)

    def test_generate_recommendations_missing_characters(self):
        """Test recommendation generation when no characters exist."""
        metrics = WorldHealthMetrics(
            entity_counts={"character": 0, "location": 5},
        )

        recs = metrics.generate_recommendations()

        assert len(recs) > 0
        assert any("character" in r.lower() for r in recs)

    def test_generate_recommendations_contradictions(self):
        """Test recommendation generation for contradictions."""
        metrics = WorldHealthMetrics(
            contradiction_count=3,
        )

        recs = metrics.generate_recommendations()

        assert len(recs) > 0
        assert any("contradiction" in r.lower() for r in recs)

    def test_generate_recommendations_healthy_world(self):
        """Test that healthy world generates no recommendations."""
        metrics = WorldHealthMetrics(
            total_entities=10,
            entity_counts={"character": 5, "location": 3, "item": 2},
            orphan_count=0,
            circular_count=0,
            contradiction_count=0,
            relationship_density=1.5,
        )

        recs = metrics.generate_recommendations()

        # Healthy world should have empty or minimal recommendations
        assert len(recs) == 0

    def test_generate_recommendations_temporal_errors(self):
        """Test recommendation generation for temporal errors."""
        metrics = WorldHealthMetrics(
            temporal_error_count=3,
            temporal_warning_count=0,
        )

        recs = metrics.generate_recommendations()

        assert len(recs) > 0
        assert any("temporal" in r.lower() for r in recs)
        assert any("3" in r for r in recs)

    def test_generate_recommendations_temporal_warnings_only(self):
        """Test recommendation generation for temporal warnings only."""
        metrics = WorldHealthMetrics(
            temporal_error_count=0,
            temporal_warning_count=5,
        )

        recs = metrics.generate_recommendations()

        assert len(recs) > 0
        assert any("temporal" in r.lower() and "warning" in r.lower() for r in recs)
        assert any("5" in r for r in recs)

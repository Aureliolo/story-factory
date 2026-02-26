"""Tests for relationship generation improvements (#329)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.memory.conflict_types import VALID_RELATIONSHIP_TYPES
from src.memory.story_state import StoryBrief, StoryState
from src.memory.world_database import WorldDatabase
from src.services.world_quality_service._batch import (
    MAX_CONSECUTIVE_BATCH_FAILURES,
    _generate_batch,
)
from src.utils.exceptions import WorldGenerationError


@pytest.fixture
def story_state():
    """Create a story state with brief for relationship tests."""
    state = StoryState(id="test-rel-story")
    state.brief = StoryBrief(
        premise="A space opera with warring factions",
        genre="sci-fi",
        subgenres=["space opera"],
        tone="epic",
        themes=["loyalty", "betrayal"],
        setting_time="Far future",
        setting_place="The Galactic Rim",
        target_length="novella",
        language="English",
        content_rating="none",
    )
    return state


# =========================================================================
# Group 1: Duplicate Prevention
# =========================================================================


class TestExistingPairsNotCapped:
    """A1: Verify all existing pairs are passed to the prompt (no 15-pair cap)."""

    def test_existing_pairs_not_capped(self, story_state):
        """All existing pairs should appear in the prompt, not just the first 15."""
        from src.services.world_quality_service._relationship import _create_relationship

        svc = MagicMock()
        svc._get_creator_model.return_value = "test-model:8b"
        svc.client.generate.return_value = {
            "response": '{"source": "Alpha", "target": "Beta", '
            '"relation_type": "knows", "description": "They met once"}'
        }

        # Create 20 existing pairs — all should appear
        existing_rels = [(f"Entity{i}", f"Entity{i + 1}", "knows") for i in range(20)]
        entity_names = [f"Entity{i}" for i in range(25)]

        _create_relationship(svc, story_state, entity_names, existing_rels, 0.9)

        prompt_arg = svc.client.generate.call_args[1]["prompt"]
        # Every pair must appear in the prompt
        for src, tgt, rel_type in existing_rels:
            expected = f"- {src} -> {tgt} ({rel_type})"
            assert expected in prompt_arg, f"Pair {expected} missing from prompt"


class TestEntityFrequencyHintReturnsTuple:
    """L1: _compute_entity_frequency_hint returns (hint, counter) tuple."""

    def test_returns_tuple_with_hint_and_counter(self):
        """_compute_entity_frequency_hint should return (str, Counter) tuple."""
        from collections import Counter

        from src.services.world_quality_service._relationship import (
            _compute_entity_frequency_hint,
        )

        entity_names = ["Alpha", "Beta", "Gamma", "Delta"]
        existing_rels = [
            ("Alpha", "Beta", "knows"),
            ("Alpha", "Gamma", "rivals"),
            ("Alpha", "Delta", "allies_with"),
            ("Alpha", "Beta", "mentors"),  # Alpha has 4 rels
        ]

        hint, freq = _compute_entity_frequency_hint(entity_names, existing_rels)

        # Should return a Counter
        assert isinstance(freq, Counter)
        # Alpha appears in 4 relationships (as source in all 4)
        assert freq["Alpha"] == 4
        # Beta appears in 2 relationships (as target in "knows" and "mentors")
        assert freq["Beta"] == 2
        # Gamma appears once (as target in "rivals")
        assert freq["Gamma"] == 1
        # Delta appears once (as target in "allies_with")
        assert freq["Delta"] == 1
        # Hint should mention over-connected Alpha
        assert "Alpha" in hint

    def test_returns_empty_hint_with_counter_for_few_entities(self):
        """With fewer than 3 entities, hint is empty but counter is still returned."""
        from collections import Counter

        from src.services.world_quality_service._relationship import (
            _compute_entity_frequency_hint,
        )

        entity_names = ["Alpha", "Beta"]
        existing_rels = [("Alpha", "Beta", "knows")]

        hint, freq = _compute_entity_frequency_hint(entity_names, existing_rels)

        assert hint == ""
        assert isinstance(freq, Counter)
        # Counter should still be populated even when hint is empty
        assert freq["Alpha"] == 1  # source in ("Alpha", "Beta", "knows")
        assert freq["Beta"] == 1  # target in ("Alpha", "Beta", "knows")

    def test_counter_reusable_for_sorting(self):
        """Returned counter can be used for sorting unused pairs."""
        from src.services.world_quality_service._relationship import (
            _compute_entity_frequency_hint,
        )

        entity_names = ["A", "B", "C", "D"]
        existing_rels = [("A", "B", "knows"), ("A", "C", "knows")]

        _hint, freq = _compute_entity_frequency_hint(entity_names, existing_rels)

        # D has 0 connections, should sort first
        sorted_names = sorted(entity_names, key=lambda n: freq[n])
        assert sorted_names[0] == "D"

    def test_balanced_distribution_returns_empty_hint(self):
        """Balanced frequency distribution returns empty hint with populated counter."""
        from collections import Counter

        from src.services.world_quality_service._relationship import (
            _compute_entity_frequency_hint,
        )

        # Each entity appears exactly 2 times (balanced — not under/over-connected)
        entity_names = ["Alice", "Bob", "Carol", "Dave"]
        existing_rels = [
            ("Alice", "Bob", "knows"),
            ("Carol", "Dave", "rivals"),
            ("Alice", "Carol", "mentors"),
            ("Bob", "Dave", "allies_with"),
        ]

        hint, freq = _compute_entity_frequency_hint(entity_names, existing_rels)

        # Balanced: no entity has <= 1 or >= 4 connections → empty hint
        assert hint == ""
        assert isinstance(freq, Counter)
        # Each entity appears exactly twice
        assert freq["Alice"] == 2
        assert freq["Bob"] == 2
        assert freq["Carol"] == 2
        assert freq["Dave"] == 2


class TestConsecutiveFailureEarlyTermination:
    """A2: Batch stops after MAX_CONSECUTIVE_BATCH_FAILURES consecutive failures."""

    def test_consecutive_failure_early_termination(self):
        """Batch should stop early after 3 consecutive failures."""
        svc = MagicMock()
        svc.get_config.return_value.get_threshold.return_value = 7.0
        svc._calculate_eta.return_value = 0.0

        call_count = 0

        def failing_generate(_i):
            """Always raise WorldGenerationError."""
            nonlocal call_count
            call_count += 1
            raise WorldGenerationError("LLM error")

        # When all fail, _generate_batch raises WorldGenerationError
        with pytest.raises(WorldGenerationError, match="Failed to generate any"):
            _generate_batch(
                svc=svc,
                count=10,
                entity_type="relationship",
                generate_fn=failing_generate,
                get_name=lambda r: "test",
            )

        # Batch recovery allows one shuffle attempt, so it takes 3 + 3 = 6 calls
        assert call_count == MAX_CONSECUTIVE_BATCH_FAILURES * 2

    def test_consecutive_failures_reset_on_success(self):
        """Failures counter resets after a success, allowing batch to continue."""
        svc = MagicMock()
        svc.get_config.return_value.get_threshold.return_value = 7.0
        svc._calculate_eta.return_value = 0.0

        call_index = 0
        # Pattern: fail, fail, succeed, fail, fail, succeed, fail, fail, succeed, succeed
        pattern = [False, False, True, False, False, True, False, False, True, True]

        def mixed_generate(_i):
            """Succeed or fail based on the predefined pattern."""
            nonlocal call_index
            idx = call_index
            call_index += 1
            if idx < len(pattern) and pattern[idx]:
                scores = MagicMock()
                scores.average = 8.0
                return {"name": f"rel-{idx}"}, scores, 1
            raise WorldGenerationError("LLM error")

        results = _generate_batch(
            svc=svc,
            count=10,
            entity_type="relationship",
            generate_fn=mixed_generate,
            get_name=lambda r: r.get("name", "test"),
        )

        # All 4 successes should be captured (indices 2, 5, 8, 9 are True)
        assert len(results) == 4
        # All 10 iterations should have been attempted (never hit 3 consecutive)
        assert call_index == 10


class TestDynamicScaling:
    """A3: Dynamic relationship target scaling caps count by entity count."""

    def test_dynamic_scaling_caps_relationship_count(self, story_state, tmp_path):
        """5 entities → max 7 relationships, even if settings allow 25."""
        from src.services.world_service import WorldService

        settings = MagicMock()
        settings.world_gen_relationships_min = 10
        settings.world_gen_relationships_max = 25
        settings.fuzzy_match_threshold = 0.8
        svc = WorldService(settings)

        # Set up world_db with only 5 entities
        world_db = WorldDatabase(tmp_path / "test_scale.db")
        try:
            for i in range(5):
                world_db.add_entity("character", f"Char{i}", f"Character {i}")

            services = MagicMock()
            # Capture the count passed to generate_relationships_with_quality
            captured_count = {}

            def capture_generate(state, names, rels, count, cancel_check=None, **kwargs):
                """Capture the count argument for later assertion."""
                captured_count["value"] = count
                return []

            services.world_quality.generate_relationships_with_quality.side_effect = (
                capture_generate
            )

            from src.services.world_service._build import _generate_relationships

            with patch("src.services.world_service._build.random.randint", return_value=25):
                _generate_relationships(svc, story_state, world_db, services)

            # 5 entities x 1.5 = 7, so count should be capped to 7
            assert captured_count["value"] == 7
        finally:
            world_db.close()

    def test_dynamic_scaling_no_cap_when_entities_sufficient(self, story_state, tmp_path):
        """20 entities → max 30, settings request 15 → stays 15."""
        from src.services.world_service import WorldService

        settings = MagicMock()
        settings.world_gen_relationships_min = 15
        settings.world_gen_relationships_max = 15
        settings.fuzzy_match_threshold = 0.8
        svc = WorldService(settings)

        world_db = WorldDatabase(tmp_path / "test_no_cap.db")
        try:
            for i in range(20):
                world_db.add_entity("character", f"Char{i}", f"Character {i}")

            services = MagicMock()
            captured_count = {}

            def capture_generate(state, names, rels, count, cancel_check=None, **kwargs):
                """Capture the count argument for later assertion."""
                captured_count["value"] = count
                return []

            services.world_quality.generate_relationships_with_quality.side_effect = (
                capture_generate
            )

            from src.services.world_service._build import _generate_relationships

            with patch("src.services.world_service._build.random.randint", return_value=15):
                _generate_relationships(svc, story_state, world_db, services)

            # 20 entities x 1.5 = 30, which is above 15, so no cap applied
            assert captured_count["value"] == 15
        finally:
            world_db.close()


# =========================================================================
# Group 2: Controlled Vocabulary
# =========================================================================


class TestCreationPromptIncludesValidTypes:
    """A4: Verify the creation prompt lists valid relationship types."""

    def test_creation_prompt_includes_valid_types(self, story_state):
        """The prompt should contain multiple types from VALID_RELATIONSHIP_TYPES."""
        from src.services.world_quality_service._relationship import _create_relationship

        svc = MagicMock()
        svc._get_creator_model.return_value = "test-model:8b"
        svc.client.generate.return_value = {
            "response": '{"source": "Alpha", "target": "Beta", '
            '"relation_type": "knows", "description": "They know each other"}'
        }

        entity_names = ["Alpha", "Beta", "Gamma"]
        _create_relationship(svc, story_state, entity_names, [], 0.9)

        prompt_arg = svc.client.generate.call_args[1]["prompt"]

        # Check that several known types from the controlled vocabulary appear
        types_found = [t for t in VALID_RELATIONSHIP_TYPES if t in prompt_arg]
        assert len(types_found) >= 10, (
            f"Expected many valid types in prompt, found {len(types_found)}: {types_found}"
        )


class TestRelationTypeNormalizedBeforeStorage:
    """A4/D4: Verify relation_type is normalized after creation."""

    def test_relation_type_normalized_in_creation(self, story_state):
        """Free-form type from LLM should be normalized after extract_json."""
        from src.services.world_quality_service._relationship import _create_relationship

        svc = MagicMock()
        svc._get_creator_model.return_value = "test-model:8b"
        # LLM returns a prose-style type
        svc.client.generate.return_value = {
            "response": '{"source": "Alpha", "target": "Beta", '
            '"relation_type": "bitter rivals who fought for years", '
            '"description": "A long-standing feud"}'
        }

        entity_names = ["Alpha", "Beta"]
        result = _create_relationship(svc, story_state, entity_names, [], 0.9)

        # "bitter rivals who fought for years" should normalize to "bitter_rivals"
        assert result["relation_type"] == "bitter_rivals"

    def test_relation_type_stored_from_service(self, story_state, tmp_path):
        """Relation type from the service should be stored as-is (service normalizes)."""
        from src.services.world_service import WorldService
        from src.services.world_service._build import _generate_relationships

        settings = MagicMock()
        settings.world_gen_relationships_min = 1
        settings.world_gen_relationships_max = 1
        settings.fuzzy_match_threshold = 0.8
        svc = WorldService(settings)

        world_db = WorldDatabase(tmp_path / "test_normalize.db")
        try:
            world_db.add_entity("character", "Alpha", "Character Alpha")
            world_db.add_entity("character", "Beta", "Character Beta")

            services = MagicMock()

            def fake_generate(state, names, rels, count, cancel_check=None, **kwargs):
                """Return a relationship with a pre-normalized type (as the real service does)."""
                scores = MagicMock()
                scores.average = 8.0
                return [
                    (
                        {
                            "source": "Alpha",
                            "target": "Beta",
                            "relation_type": "rivals",
                            "description": "A feud",
                        },
                        scores,
                    )
                ]

            services.world_quality.generate_relationships_with_quality.side_effect = fake_generate

            with patch("src.services.world_service._build.random.randint", return_value=1):
                _generate_relationships(svc, story_state, world_db, services)

            rels = world_db.list_relationships()
            assert len(rels) == 1
            # Service returns pre-normalized "rivals"; _build.py stores it directly
            assert rels[0].relation_type == "rivals"
        finally:
            world_db.close()


class TestRelationTypeNormalizedInBuild:
    """D4: Verify _build.py stores relation_type from service correctly."""

    def test_known_type_stored_correctly(self, story_state, tmp_path):
        """A pre-normalized type from the service should be stored as-is."""
        from src.services.world_service import WorldService
        from src.services.world_service._build import _generate_relationships

        settings = MagicMock()
        settings.world_gen_relationships_min = 1
        settings.world_gen_relationships_max = 1
        settings.fuzzy_match_threshold = 0.8
        svc = WorldService(settings)

        world_db = WorldDatabase(tmp_path / "test_build_norm.db")
        try:
            world_db.add_entity("character", "Alpha", "Character Alpha")
            world_db.add_entity("character", "Beta", "Character Beta")

            services = MagicMock()

            def fake_generate(state, names, rels, count, cancel_check=None, **kwargs):
                """Return a relationship with a pre-normalized type."""
                scores = MagicMock()
                scores.average = 8.0
                return [
                    (
                        {
                            "source": "Alpha",
                            "target": "Beta",
                            "relation_type": "allies_with",
                            "description": "They work together",
                        },
                        scores,
                    )
                ]

            services.world_quality.generate_relationships_with_quality.side_effect = fake_generate

            with patch("src.services.world_service._build.random.randint", return_value=1):
                _generate_relationships(svc, story_state, world_db, services)

            rels = world_db.list_relationships()
            assert len(rels) == 1
            assert rels[0].relation_type == "allies_with"
        finally:
            world_db.close()

    def test_missing_relation_type_defaults_to_related_to(self, story_state, tmp_path):
        """When LLM omits relation_type, it should default to 'related_to' with a debug log."""
        from src.services.world_service import WorldService
        from src.services.world_service._build import _generate_relationships

        settings = MagicMock()
        settings.world_gen_relationships_min = 1
        settings.world_gen_relationships_max = 1
        settings.fuzzy_match_threshold = 0.8
        svc = WorldService(settings)

        world_db = WorldDatabase(tmp_path / "test_missing_type.db")
        try:
            world_db.add_entity("character", "Alpha", "Character Alpha")
            world_db.add_entity("character", "Beta", "Character Beta")

            services = MagicMock()

            def fake_generate(state, names, rels, count, cancel_check=None, **kwargs):
                """Return a relationship without relation_type key."""
                scores = MagicMock()
                scores.average = 8.0
                return [
                    (
                        {
                            "source": "Alpha",
                            "target": "Beta",
                            "description": "They know each other",
                        },
                        scores,
                    )
                ]

            services.world_quality.generate_relationships_with_quality.side_effect = fake_generate

            with patch("src.services.world_service._build.random.randint", return_value=1):
                _generate_relationships(svc, story_state, world_db, services)

            rels = world_db.list_relationships()
            assert len(rels) == 1
            assert rels[0].relation_type == "related_to"
        finally:
            world_db.close()

    def test_orphaned_relationships_skipped(self, story_state, tmp_path):
        """Relationships referencing deleted entities should be skipped with a warning.

        Note: This test relies on SQLite foreign keys being disabled (the default
        in WorldDatabase.__init__—no ``PRAGMA foreign_keys = ON``). Without FK
        enforcement, ``delete_entity`` removes the entity row but leaves the
        relationship row intact, creating the orphan. If FK cascades are ever
        enabled, the relationship would be auto-deleted and this test would need
        to create the orphan differently.
        """
        from src.services.world_service import WorldService
        from src.services.world_service._build import _generate_relationships

        settings = MagicMock()
        settings.world_gen_relationships_min = 1
        settings.world_gen_relationships_max = 1
        settings.fuzzy_match_threshold = 0.8
        svc = WorldService(settings)

        world_db = WorldDatabase(tmp_path / "test_orphan.db")
        try:
            # Add entities and create a relationship
            world_db.add_entity("character", "Alpha", "Character Alpha")
            world_db.add_entity("character", "Beta", "Character Beta")
            entities = world_db.list_entities()
            world_db.add_relationship(
                source_id=entities[0].id,
                target_id=entities[1].id,
                relation_type="knows",
                description="They know each other",
            )
            # Delete one entity to orphan the relationship
            world_db.delete_entity(entities[1].id)

            services = MagicMock()
            services.world_quality.generate_relationships_with_quality.return_value = []

            with patch("src.services.world_service._build.random.randint", return_value=1):
                _generate_relationships(svc, story_state, world_db, services)

            # The orphaned relationship should have been skipped, not passed to quality service
            call_args = services.world_quality.generate_relationships_with_quality.call_args
            existing_rels_passed = call_args[0][2]
            assert len(existing_rels_passed) == 0
        finally:
            world_db.close()


class TestCreateRelationshipNoBriefLogging:
    """Verify _create_relationship logs warning when brief is None."""

    def test_create_relationship_no_brief_logs_warning(self, caplog):
        """_create_relationship should log a warning and return {} when brief is None."""
        import logging

        from src.services.world_quality_service._relationship import _create_relationship

        svc = MagicMock()
        story_state = MagicMock()
        story_state.brief = None
        story_state.id = "test-story-id"

        with caplog.at_level(logging.WARNING):
            result = _create_relationship(svc, story_state, ["A", "B"], [], 0.9)

        assert result == {}
        assert "no brief" in caplog.text
        assert "test-story-id" in caplog.text


# =========================================================================
# Group 3: Diversity & Refinement
# =========================================================================


class TestDiversityHint:
    """A6: Diversity-weighted prompting based on relationship category distribution."""

    def test_diversity_hint_when_all_alliance(self):
        """All alliance rels → hint should suggest rivalry or tension."""
        from src.services.world_quality_service._relationship import _compute_diversity_hint

        # All alliance types
        existing_types = ["allies_with", "friends_with", "trusts", "loves", "protects"]
        hint = _compute_diversity_hint(existing_types)

        assert hint != ""
        # Should suggest rivalry (the first under-represented category checked)
        assert "rivalry" in hint.lower()

    def test_diversity_hint_when_balanced(self):
        """Balanced distribution → no hint needed."""
        from src.services.world_quality_service._relationship import _compute_diversity_hint

        # Mix of categories — roughly balanced
        existing_types = [
            "allies_with",
            "trusts",
            "friends_with",  # alliance
            "enemy_of",
            "rivals",
            "hates",  # rivalry
            "fears",
            "distrusts",
            "manipulates",  # tension
            "knows",
            "located_in",
            "works_with",  # neutral
        ]
        hint = _compute_diversity_hint(existing_types)

        assert hint == ""

    def test_diversity_hint_with_few_relationships(self):
        """Fewer than 3 existing rels → no hint."""
        from src.services.world_quality_service._relationship import _compute_diversity_hint

        hint = _compute_diversity_hint(["knows", "loves"])
        assert hint == ""

    def test_diversity_hint_empty_list(self):
        """Empty list → no hint."""
        from src.services.world_quality_service._relationship import _compute_diversity_hint

        hint = _compute_diversity_hint([])
        assert hint == ""

    def test_diversity_hint_suggests_alliance_when_missing(self):
        """All rivalry/tension → hint should suggest alliance."""
        from src.services.world_quality_service._relationship import _compute_diversity_hint

        existing_types = ["enemy_of", "hates", "rivals", "fears", "distrusts"]
        hint = _compute_diversity_hint(existing_types)

        assert hint != ""
        # Rivalry is well-represented, tension is also, but alliance is 0%
        # The check order is: RIVALRY, TENSION, ALLIANCE
        # Tension should be checked after rivalry — but "distrusts" and "fears" are tension
        # So tension is 2/5=40%, rivalry is 3/5=60%, alliance is 0/5=0%
        # Alliance is under-represented
        assert "alliance" in hint.lower()


class TestExistingRelsIncludeTypes:
    """A6: Verify 3-tuple tracking works through the batch generation chain."""

    def test_existing_rels_are_3_tuples_in_batch(self, story_state):
        """generate_relationships_with_quality should track 3-tuples in on_success."""
        from src.services.world_quality_service._batch import (
            generate_relationships_with_quality,
        )

        svc = MagicMock()
        svc.get_config.return_value.get_threshold.return_value = 7.0
        svc._calculate_eta.return_value = 0.0
        svc.settings.llm_max_concurrent_requests = 1  # Sequential: test expects ordered rels

        # Capture the rels list passed to generate_fn
        captured_rels = []

        def fake_generate_fn(state, names, rels):
            """Capture rels list and return a successful relationship."""
            captured_rels.append(list(rels))
            scores = MagicMock()
            scores.average = 8.0
            return (
                {"source": "A", "target": "B", "relation_type": "hates"},
                scores,
                1,
            )

        svc.generate_relationship_with_quality.side_effect = fake_generate_fn

        initial_rels: list[tuple[str, str, str]] = [("X", "Y", "knows")]
        generate_relationships_with_quality(
            svc, story_state, ["A", "B", "X", "Y"], initial_rels, count=2
        )

        # First call should see the initial rel
        assert captured_rels[0] == [("X", "Y", "knows")]
        # Second call should see initial + the newly generated one
        assert len(captured_rels[1]) == 2
        assert captured_rels[1][1] == ("A", "B", "hates")

    def test_diversity_hint_injected_into_prompt(self, story_state):
        """When existing rels are all alliance, the prompt should contain a diversity hint."""
        from src.services.world_quality_service._relationship import _create_relationship

        svc = MagicMock()
        svc._get_creator_model.return_value = "test-model:8b"
        svc.client.generate.return_value = {
            "response": '{"source": "Alpha", "target": "Beta", '
            '"relation_type": "knows", "description": "They know each other"}'
        }

        # All alliance types → should trigger diversity hint for rivalry
        existing_rels: list[tuple[str, str, str]] = [
            ("A", "B", "allies_with"),
            ("C", "D", "trusts"),
            ("E", "F", "loves"),
            ("G", "H", "friends_with"),
        ]
        entity_names = ["Alpha", "Beta"]
        _create_relationship(svc, story_state, entity_names, existing_rels, 0.9)

        prompt_arg = svc.client.generate.call_args[1]["prompt"]
        assert "HINT" in prompt_arg
        assert "rivalry" in prompt_arg.lower()

    def test_unused_pairs_sorted_by_entity_frequency(self, story_state):
        """Under-connected entity pairs should appear before over-connected ones."""
        from src.services.world_quality_service._relationship import _create_relationship

        svc = MagicMock()
        svc._get_creator_model.return_value = "test-model:8b"
        svc.client.generate.return_value = {
            "response": '{"source": "Alpha", "target": "Delta", '
            '"relation_type": "knows", "description": "New acquaintance"}'
        }

        # Alpha has 3 connections, Beta has 2, Gamma and Delta have 0
        entity_names = ["Alpha", "Beta", "Gamma", "Delta"]
        existing_rels = [
            ("Alpha", "Beta", "allies_with"),
            ("Alpha", "Beta", "trusts"),
            ("Alpha", "Beta", "mentors"),
        ]
        _create_relationship(svc, story_state, entity_names, existing_rels, 0.9)

        prompt_arg = svc.client.generate.call_args[1]["prompt"]
        # Extract the unused pairs section from the prompt
        pairs_start = prompt_arg.find("AVAILABLE UNUSED PAIRS")
        assert pairs_start != -1, "Unused pairs section should be in the prompt"
        pairs_section = prompt_arg[pairs_start:]
        # Gamma and Delta (0 connections) pairs should appear before Alpha (3) pairs
        gamma_pos = pairs_section.find("Gamma")
        alpha_pos = pairs_section.find("Alpha")
        assert gamma_pos < alpha_pos, "Under-connected entities should appear first in unused pairs"


# =========================================================================
# Group 4: Required Entity Constraint (orphan recovery)
# =========================================================================


class TestRequiredEntityConstraint:
    """Tests for required_entity parameter in relationship generation."""

    def test_generate_with_required_entity_rejects_missing(self, story_state):
        """_is_empty rejects relationships where required_entity is not in source/target."""
        from src.memory.world_quality import RelationshipQualityScores
        from src.services.world_quality_service._relationship import (
            generate_relationship_with_quality,
        )

        svc = MagicMock()
        svc.analytics_db = None
        config = MagicMock()
        config.creator_temperature = 0.9
        config.judge_temperature = 0.1
        config.get_threshold.return_value = 7.0
        config.get_refinement_temperature.return_value = 0.7
        config.max_iterations = 3
        config.early_stop_patience = 2
        svc.get_config.return_value = config

        judge_config = MagicMock()
        judge_config.enabled = False
        judge_config.multi_call_enabled = False
        svc.get_judge_config.return_value = judge_config

        call_count = 0

        def fake_create(
            story_state, entity_names, existing_rels, temperature, required_entity=None
        ):
            """First call ignores required_entity; second call includes it."""
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # LLM ignores the constraint — will be rejected by _is_empty
                return {
                    "source": "Alpha",
                    "target": "Beta",
                    "relation_type": "knows",
                    "description": "They know each other",
                }
            # Second call respects the constraint
            return {
                "source": "Charlie",
                "target": "Alpha",
                "relation_type": "friends",
                "description": "Charlie befriends Alpha",
            }

        svc._create_relationship = MagicMock(side_effect=fake_create)

        # Judge returns passing scores using real Pydantic model
        judge_scores = RelationshipQualityScores(
            tension=8.0,
            dynamics=8.0,
            story_potential=8.0,
            authenticity=8.0,
            feedback="Good",
        )
        svc._judge_relationship_quality = MagicMock(return_value=judge_scores)

        entity_names = ["Alpha", "Beta", "Charlie"]
        rel, _scores, _iterations = generate_relationship_with_quality(
            svc, story_state, entity_names, [], required_entity="Charlie"
        )

        # First attempt rejected (no Charlie), second accepted
        assert call_count >= 2
        assert rel["source"] == "Charlie" or rel["target"] == "Charlie"

    def test_create_relationship_prompt_includes_required_entity(self, story_state):
        """Prompt should contain REQUIRED ENTITY directive when required_entity is set."""
        from src.services.world_quality_service._relationship import _create_relationship

        svc = MagicMock()
        svc._get_creator_model.return_value = "test-model:8b"
        svc.client.generate.return_value = {
            "response": '{"source": "Charlie", "target": "Beta", '
            '"relation_type": "knows", "description": "They met"}'
        }

        entity_names = ["Alpha", "Beta", "Charlie"]
        _create_relationship(svc, story_state, entity_names, [], 0.9, required_entity="Charlie")

        prompt_arg = svc.client.generate.call_args[1]["prompt"]
        assert "REQUIRED ENTITY" in prompt_arg
        assert '"Charlie"' in prompt_arg

    def test_create_relationship_prompt_no_required_entity(self, story_state):
        """Prompt should NOT contain REQUIRED ENTITY when required_entity is None."""
        from src.services.world_quality_service._relationship import _create_relationship

        svc = MagicMock()
        svc._get_creator_model.return_value = "test-model:8b"
        svc.client.generate.return_value = {
            "response": '{"source": "Alpha", "target": "Beta", '
            '"relation_type": "knows", "description": "They met"}'
        }

        entity_names = ["Alpha", "Beta", "Charlie"]
        _create_relationship(svc, story_state, entity_names, [], 0.9)

        prompt_arg = svc.client.generate.call_args[1]["prompt"]
        assert "REQUIRED ENTITY" not in prompt_arg

    def test_create_relationship_filters_unused_pairs_for_required_entity(self, story_state):
        """When required_entity is set, unused pairs should only include that entity."""
        from src.services.world_quality_service._relationship import _create_relationship

        svc = MagicMock()
        svc._get_creator_model.return_value = "test-model:8b"
        svc.client.generate.return_value = {
            "response": '{"source": "Charlie", "target": "Alpha", '
            '"relation_type": "knows", "description": "They met"}'
        }

        entity_names = ["Alpha", "Beta", "Charlie", "Delta"]
        _create_relationship(svc, story_state, entity_names, [], 0.9, required_entity="Charlie")

        prompt_arg = svc.client.generate.call_args[1]["prompt"]
        # Check the unused pairs section exists
        pairs_start = prompt_arg.find("AVAILABLE UNUSED PAIRS")
        assert pairs_start != -1, "Unused pairs section should be in the prompt"
        pairs_section = prompt_arg[pairs_start:]

        # Every pair line should involve Charlie
        pair_lines = [
            line.strip() for line in pairs_section.split("\n") if line.strip().startswith("- ")
        ]
        for line in pair_lines:
            assert "Charlie" in line, f"Pair line should involve Charlie: {line}"

        # Alpha -> Beta (no Charlie) should NOT appear
        assert "Alpha -> Beta" not in pairs_section
        assert "Beta -> Alpha" not in pairs_section

    def test_create_relationship_no_filter_without_required_entity(self, story_state):
        """Without required_entity, unused pairs include all combinations."""
        from src.services.world_quality_service._relationship import _create_relationship

        svc = MagicMock()
        svc._get_creator_model.return_value = "test-model:8b"
        svc.client.generate.return_value = {
            "response": '{"source": "Alpha", "target": "Beta", '
            '"relation_type": "knows", "description": "They met"}'
        }

        entity_names = ["Alpha", "Beta", "Charlie"]
        _create_relationship(svc, story_state, entity_names, [], 0.9)

        prompt_arg = svc.client.generate.call_args[1]["prompt"]
        pairs_start = prompt_arg.find("AVAILABLE UNUSED PAIRS")
        assert pairs_start != -1
        pairs_section = prompt_arg[pairs_start:]

        # Without required_entity, all pairs should appear (including Alpha -> Beta)
        assert "Alpha -> Beta" in pairs_section or "Beta -> Alpha" in pairs_section

    def test_is_empty_accepts_case_insensitive_required_entity(self, story_state):
        """_is_empty accepts relationships where required_entity casing differs from endpoints."""
        from src.memory.world_quality import RelationshipQualityScores
        from src.services.world_quality_service._relationship import (
            generate_relationship_with_quality,
        )

        svc = MagicMock()
        svc.analytics_db = None
        config = MagicMock()
        config.creator_temperature = 0.9
        config.judge_temperature = 0.1
        config.get_threshold.return_value = 7.0
        config.get_refinement_temperature.return_value = 0.7
        config.max_iterations = 3
        config.early_stop_patience = 2
        svc.get_config.return_value = config

        judge_config = MagicMock()
        judge_config.enabled = False
        judge_config.multi_call_enabled = False
        svc.get_judge_config.return_value = judge_config

        # LLM returns the entity with different casing than required_entity
        svc._create_relationship = MagicMock(
            return_value={
                "source": "CHARLIE",
                "target": "Alpha",
                "relation_type": "knows",
                "description": "They know each other",
            }
        )

        judge_scores = RelationshipQualityScores(
            tension=8.0,
            dynamics=8.0,
            story_potential=8.0,
            authenticity=8.0,
            feedback="Good",
        )
        svc._judge_relationship_quality = MagicMock(return_value=judge_scores)

        entity_names = ["Alpha", "Beta", "Charlie"]
        # required_entity is lowercase but LLM returns UPPERCASE — should still match
        rel, _scores, _iterations = generate_relationship_with_quality(
            svc, story_state, entity_names, [], required_entity="charlie"
        )

        # Should accept on first try (no rejection for case mismatch)
        assert svc._create_relationship.call_count == 1
        assert rel["source"] == "CHARLIE"

    def test_is_empty_accepts_required_entity_with_leading_article(self, story_state):
        """_is_empty accepts when LLM adds 'The' prefix to required_entity name."""
        from src.memory.world_quality import RelationshipQualityScores
        from src.services.world_quality_service._relationship import (
            generate_relationship_with_quality,
        )

        svc = MagicMock()
        svc.analytics_db = None
        config = MagicMock()
        config.creator_temperature = 0.9
        config.judge_temperature = 0.1
        config.get_threshold.return_value = 7.0
        config.get_refinement_temperature.return_value = 0.7
        config.max_iterations = 3
        config.early_stop_patience = 2
        svc.get_config.return_value = config

        judge_config = MagicMock()
        judge_config.enabled = False
        judge_config.multi_call_enabled = False
        svc.get_judge_config.return_value = judge_config

        # LLM adds "The" prefix to the entity name
        svc._create_relationship = MagicMock(
            return_value={
                "source": "The Echoes of the Network",
                "target": "Alpha",
                "relation_type": "knows",
                "description": "Connected",
            }
        )

        judge_scores = RelationshipQualityScores(
            tension=8.0,
            dynamics=8.0,
            story_potential=8.0,
            authenticity=8.0,
            feedback="Good",
        )
        svc._judge_relationship_quality = MagicMock(return_value=judge_scores)

        entity_names = ["Alpha", "Beta", "Echoes of the Network"]
        rel, _scores, _iterations = generate_relationship_with_quality(
            svc, story_state, entity_names, [], required_entity="Echoes of the Network"
        )

        # Should accept — "The Echoes of the Network" normalizes to "echoes of the network"
        assert svc._create_relationship.call_count == 1
        assert rel["source"] == "The Echoes of the Network"


# =========================================================================
# Group 5: LLM Relationship Type Classification
# =========================================================================


class TestLLMRelationTypeClassification:
    """Tests for LLM-based fallback classification of unknown relationship types."""

    def setup_method(self):
        """Clear the LLM classification cache before each test."""
        from src.services.world_quality_service._relationship import (
            _llm_classification_cache,
            _llm_classification_cache_lock,
        )

        with _llm_classification_cache_lock:
            _llm_classification_cache.clear()

    def test_cache_hit_skips_llm_call(self):
        """Cached result should be returned without calling the LLM."""
        from src.services.world_quality_service._relationship import (
            _classify_relation_type_with_llm,
            _llm_classification_cache,
            _llm_classification_cache_lock,
        )

        svc = MagicMock()

        # Pre-populate cache
        with _llm_classification_cache_lock:
            _llm_classification_cache["a close friend"] = "friends"

        result = _classify_relation_type_with_llm(svc, "A close friend")

        assert result == "friends"
        # LLM should not have been called
        svc.client.generate.assert_not_called()

    def test_valid_llm_result_cached_and_returned(self):
        """Valid LLM classification should be cached and returned."""
        from src.services.world_quality_service._relationship import (
            _classify_relation_type_with_llm,
            _llm_classification_cache,
            _llm_classification_cache_lock,
        )

        svc = MagicMock()
        svc._get_creator_model.return_value = "test-model:8b"
        svc.client.generate.return_value = {"response": "rivals"}

        result = _classify_relation_type_with_llm(
            svc, "A close friend and fellow scientist who often challenges his views"
        )

        assert result == "rivals"
        # Verify it was cached
        with _llm_classification_cache_lock:
            cache_key = "a close friend and fellow scientist who often challenges his views"
            assert _llm_classification_cache[cache_key] == "rivals"

    def test_invalid_llm_result_returns_none(self):
        """LLM returning an unknown type should return None."""
        from src.services.world_quality_service._relationship import (
            _classify_relation_type_with_llm,
        )

        svc = MagicMock()
        svc._get_creator_model.return_value = "test-model:8b"
        svc.client.generate.return_value = {"response": "some_invalid_type_xyz"}

        result = _classify_relation_type_with_llm(svc, "a cosmic bond of destiny")

        assert result is None

    def test_llm_failure_returns_none(self):
        """LLM error should return None (non-fatal fallback)."""
        from src.services.world_quality_service._relationship import (
            _classify_relation_type_with_llm,
        )

        svc = MagicMock()
        svc._get_creator_model.return_value = "test-model:8b"
        svc.client.generate.side_effect = ConnectionError("Ollama offline")

        result = _classify_relation_type_with_llm(svc, "complicated friendship")

        assert result is None

    def test_normalize_with_llm_fallback_known_type(self):
        """Known types should be returned directly without LLM call."""
        from src.services.world_quality_service._relationship import (
            _normalize_with_llm_fallback,
        )

        svc = MagicMock()

        result = _normalize_with_llm_fallback(svc, "allies_with")
        assert result == "allies_with"
        svc.client.generate.assert_not_called()

    def test_normalize_with_llm_fallback_uses_llm(self):
        """Unknown types should trigger LLM classification."""
        from src.services.world_quality_service._relationship import (
            _normalize_with_llm_fallback,
        )

        svc = MagicMock()
        svc._get_creator_model.return_value = "test-model:8b"
        svc.client.generate.return_value = {"response": "friends"}

        result = _normalize_with_llm_fallback(
            svc, "a dear companion who has been through thick and thin"
        )
        assert result == "friends"

    def test_normalize_with_llm_fallback_returns_normalized_on_failure(self):
        """When LLM also fails, the original normalized result should be returned."""
        from src.services.world_quality_service._relationship import (
            _normalize_with_llm_fallback,
        )

        svc = MagicMock()
        svc._get_creator_model.return_value = "test-model:8b"
        svc.client.generate.side_effect = ConnectionError("Ollama offline")

        result = _normalize_with_llm_fallback(svc, "cosmic destiny bond of the ancients")
        # Should return the original normalized form
        assert result == "cosmic_destiny_bond_of_the_ancients"

    def test_create_relationship_uses_llm_fallback(self, story_state):
        """_create_relationship should use LLM fallback for unrecognized types."""
        from src.services.world_quality_service._relationship import _create_relationship

        svc = MagicMock()
        svc._get_creator_model.return_value = "test-model:8b"

        # First call: create returns a prose-style type
        # Second call (LLM classification): returns a valid type
        svc.client.generate.side_effect = [
            {
                "response": '{"source": "Alpha", "target": "Beta", '
                '"relation_type": "A close friend and fellow scientist", '
                '"description": "They are friends"}'
            },
            {"response": "friends"},
        ]

        entity_names = ["Alpha", "Beta"]
        result = _create_relationship(svc, story_state, entity_names, [], 0.9)

        assert result["relation_type"] == "friends"

    def test_refine_relationship_uses_llm_fallback(self, story_state):
        """_refine_relationship should use LLM fallback for unrecognized types."""
        from src.memory.world_quality import RelationshipQualityScores
        from src.services.world_quality_service._relationship import _refine_relationship

        svc = MagicMock()
        svc.settings.context_size = 32768
        svc._get_creator_model.return_value = "test-model:8b"
        svc.get_config.return_value.get_threshold.return_value = 5.0

        # Refinement returns a prose-style type
        svc.client.generate.side_effect = [
            # First call: the refinement itself
            {
                "response": '{"source": "Alpha", "target": "Beta", '
                '"relation_type": "blood brothers sworn by fire", '
                '"description": "They are inseparable"}'
            },
            # Second call: LLM classification fallback
            {"response": "allies_with"},
        ]

        relationship = {
            "source": "Alpha",
            "target": "Beta",
            "relation_type": "friends",
            "description": "They are friends",
        }
        scores = RelationshipQualityScores(
            tension=3.0,
            dynamics=3.0,
            story_potential=3.0,
            authenticity=3.0,
            feedback="Needs more depth",
        )

        result = _refine_relationship(svc, relationship, scores, story_state, 0.7)
        assert result["relation_type"] == "allies_with"

    def test_negative_cache_hit_returns_none(self):
        """Negative cache entry (empty string) should return None without LLM call."""
        from src.services.world_quality_service._relationship import (
            _classify_relation_type_with_llm,
            _llm_classification_cache,
            _llm_classification_cache_lock,
        )

        svc = MagicMock()

        # Pre-populate cache with negative sentinel
        with _llm_classification_cache_lock:
            _llm_classification_cache["unknown bond"] = ""

        result = _classify_relation_type_with_llm(svc, "Unknown bond")

        assert result is None
        svc.client.generate.assert_not_called()

    def test_cache_overflow_clears_on_valid_result(self):
        """When cache is full, it should be cleared before storing a valid result."""
        from src.services.world_quality_service._relationship import (
            _LLM_CACHE_MAX_SIZE,
            _classify_relation_type_with_llm,
            _llm_classification_cache,
            _llm_classification_cache_lock,
        )

        svc = MagicMock()
        svc._get_creator_model.return_value = "test-model:8b"
        svc.client.generate.return_value = {"response": "rivals"}

        # Fill cache to capacity
        with _llm_classification_cache_lock:
            for i in range(_LLM_CACHE_MAX_SIZE):
                _llm_classification_cache[f"filler_{i}"] = "friends"

        result = _classify_relation_type_with_llm(svc, "nemesis of the realm")
        assert result == "rivals"

        # Cache should have been cleared and now contain only the new entry
        with _llm_classification_cache_lock:
            assert len(_llm_classification_cache) == 1
            assert _llm_classification_cache["nemesis of the realm"] == "rivals"

    def test_cache_overflow_clears_on_invalid_result(self):
        """When cache is full and LLM returns invalid type, cache should be cleared."""
        from src.services.world_quality_service._relationship import (
            _LLM_CACHE_MAX_SIZE,
            _classify_relation_type_with_llm,
            _llm_classification_cache,
            _llm_classification_cache_lock,
        )

        svc = MagicMock()
        svc._get_creator_model.return_value = "test-model:8b"
        svc.client.generate.return_value = {"response": "totally_invalid_xyz"}

        # Fill cache to capacity
        with _llm_classification_cache_lock:
            for i in range(_LLM_CACHE_MAX_SIZE):
                _llm_classification_cache[f"filler_{i}"] = "friends"

        result = _classify_relation_type_with_llm(svc, "mystical cosmic bond")
        assert result is None

        # Cache should have been cleared and now contain the negative entry
        with _llm_classification_cache_lock:
            assert len(_llm_classification_cache) == 1
            assert _llm_classification_cache["mystical cosmic bond"] == ""

    def test_cache_overflow_clears_on_llm_error(self):
        """When cache is full and LLM fails, cache should be cleared."""
        from src.services.world_quality_service._relationship import (
            _LLM_CACHE_MAX_SIZE,
            _classify_relation_type_with_llm,
            _llm_classification_cache,
            _llm_classification_cache_lock,
        )

        svc = MagicMock()
        svc._get_creator_model.return_value = "test-model:8b"
        svc.client.generate.side_effect = ConnectionError("Ollama offline")

        # Fill cache to capacity
        with _llm_classification_cache_lock:
            for i in range(_LLM_CACHE_MAX_SIZE):
                _llm_classification_cache[f"filler_{i}"] = "friends"

        result = _classify_relation_type_with_llm(svc, "deep ancestral link")
        assert result is None

        # Cache should have been cleared and now contain the negative entry
        with _llm_classification_cache_lock:
            assert len(_llm_classification_cache) == 1
            assert _llm_classification_cache["deep ancestral link"] == ""

    def test_key_error_returns_none_with_negative_caching(self):
        """KeyError from response parsing should return None and cache negative result."""
        from src.services.world_quality_service._relationship import (
            _classify_relation_type_with_llm,
            _llm_classification_cache,
            _llm_classification_cache_lock,
        )

        svc = MagicMock()
        svc._get_creator_model.return_value = "test-model:8b"
        # Return dict missing the "response" key
        svc.client.generate.return_value = {"unexpected_key": "value"}

        result = _classify_relation_type_with_llm(svc, "arcane connection")
        assert result is None

        # Should be cached with empty-string sentinel to avoid repeated LLM calls
        with _llm_classification_cache_lock:
            assert "arcane connection" in _llm_classification_cache
            assert _llm_classification_cache["arcane connection"] == ""

    def test_key_error_clears_cache_when_full(self):
        """KeyError path should clear cache before inserting when cache is at capacity."""
        from src.services.world_quality_service._relationship import (
            _LLM_CACHE_MAX_SIZE,
            _classify_relation_type_with_llm,
            _llm_classification_cache,
            _llm_classification_cache_lock,
        )

        svc = MagicMock()
        svc._get_creator_model.return_value = "test-model:8b"
        svc.client.generate.return_value = {"unexpected_key": "value"}

        # Fill cache to capacity
        with _llm_classification_cache_lock:
            _llm_classification_cache.clear()
            for i in range(_LLM_CACHE_MAX_SIZE):
                _llm_classification_cache[f"filler_{i}"] = "alliance"

        result = _classify_relation_type_with_llm(svc, "mystic bond")
        assert result is None

        # Cache should have been cleared and now contain only the negative entry
        with _llm_classification_cache_lock:
            assert len(_llm_classification_cache) == 1
            assert _llm_classification_cache["mystic bond"] == ""


# =========================================================================
# Group: Schema Enum Constraints (#397)
# =========================================================================


class TestCreateRelationshipSchemaEnum:
    """Verify _create_relationship passes format= with enum constraints."""

    def test_create_passes_schema_with_entity_name_enum(self, story_state):
        """Entity names should be constrained via enum in the format schema."""
        from src.services.world_quality_service._relationship import _create_relationship

        svc = MagicMock()
        svc._get_creator_model.return_value = "test-model:8b"
        svc.get_calendar_context.return_value = ""

        entity_names = ["Alice", "Bob", "Castle Noir"]
        response_json = json.dumps(
            {
                "source": "Alice",
                "target": "Bob",
                "relation_type": "rivals",
                "description": "A fierce rivalry",
            }
        )
        svc.client.generate.return_value = {"response": response_json}

        _create_relationship(svc, story_state, entity_names, [], 0.7)

        # Verify format= was passed with enum constraints
        call_kwargs = svc.client.generate.call_args
        schema = call_kwargs.kwargs.get("format") or call_kwargs[1].get("format")
        assert schema is not None
        assert isinstance(schema, dict)
        assert schema["properties"]["source"]["enum"] == entity_names
        assert schema["properties"]["target"]["enum"] == entity_names
        assert schema["properties"]["relation_type"]["enum"] == list(VALID_RELATIONSHIP_TYPES)

    def test_create_uses_json_loads_not_extract_json(self, story_state):
        """Response should be parsed with json.loads (schema guarantees valid JSON)."""
        from src.services.world_quality_service._relationship import _create_relationship

        svc = MagicMock()
        svc._get_creator_model.return_value = "test-model:8b"
        svc.get_calendar_context.return_value = ""

        response_data = {
            "source": "Entity A",
            "target": "Entity B",
            "relation_type": "allies_with",
            "description": "Strong alliance",
        }
        svc.client.generate.return_value = {"response": json.dumps(response_data)}

        result = _create_relationship(svc, story_state, ["Entity A", "Entity B"], [], 0.7)

        assert result["source"] == "Entity A"
        assert result["target"] == "Entity B"


class TestRefineRelationshipPinnedSchema:
    """Verify _refine_relationship pins source/target/type in the schema."""

    def test_refine_pins_source_target_type(self, story_state):
        """Refine schema should lock source, target, and type to single-element enums."""
        from src.services.world_quality_service._relationship import _refine_relationship

        svc = MagicMock()
        svc._get_creator_model.return_value = "test-model:8b"
        svc.get_config.return_value = MagicMock(
            get_threshold=MagicMock(return_value=6.0),
        )

        relationship = {
            "source": "Alice",
            "target": "Bob",
            "relation_type": "rivals",
            "description": "Original description",
        }
        scores = MagicMock()
        scores.tension = 5.0
        scores.dynamics = 5.0
        scores.story_potential = 5.0
        scores.authenticity = 5.0
        scores.feedback = "Needs more depth"

        refined_json = json.dumps(
            {
                "source": "Alice",
                "target": "Bob",
                "relation_type": "rivals",
                "description": "Improved description with much more depth",
            }
        )
        svc.client.generate.return_value = {"response": refined_json}

        _refine_relationship(svc, relationship, scores, story_state, 0.7)

        # Verify format= was passed with pinned single-element enums
        call_kwargs = svc.client.generate.call_args
        schema = call_kwargs.kwargs.get("format") or call_kwargs[1].get("format")
        assert schema is not None
        assert isinstance(schema, dict)
        assert schema["properties"]["source"]["enum"] == ["Alice"]
        assert schema["properties"]["target"]["enum"] == ["Bob"]
        assert schema["properties"]["relation_type"]["enum"] == ["rivals"]


class TestNormalizationConsolidation:
    """Verify _normalize_entity_name was replaced with _normalize_name from _name_matching."""

    def test_normalize_name_imported_from_name_matching(self):
        """_normalize_name should be importable from _relationship module (re-exported)."""
        from src.services.world_quality_service._relationship import _normalize_name
        from src.services.world_service._name_matching import (
            _normalize_name as original_normalize,
        )

        # Should be the same function
        assert _normalize_name is original_normalize

    def test_is_empty_uses_normalize_name_for_required_entity(self, story_state):
        """The _is_empty check should use _normalize_name for required entity matching."""
        from src.memory.world_quality._story_scores import RelationshipQualityScores
        from src.services.world_quality_service._relationship import (
            generate_relationship_with_quality,
        )

        svc = MagicMock()
        svc.analytics_db = None
        svc._get_creator_model.return_value = "test-model:8b"
        svc._get_judge_model.return_value = "test-model:8b"
        svc.get_calendar_context.return_value = ""
        svc.get_config.return_value = MagicMock(
            creator_temperature=0.7,
            judge_temperature=0.3,
            max_iterations=1,
            min_iterations=1,
            get_threshold=MagicMock(return_value=6.0),
            get_refinement_temperature=MagicMock(return_value=0.7),
        )
        svc.get_judge_config.return_value = MagicMock(enabled=False, multi_call_enabled=False)
        svc.settings = MagicMock(story_factory_agent_timeout=120)

        # The relationship contains "The Castle" which normalizes to "castle"
        # Required entity is "Castle" which also normalizes to "castle"
        svc._create_relationship.return_value = {
            "source": "The Castle",
            "target": "Alice",
            "relation_type": "contains",
            "description": "Castle contains Alice",
        }

        # Mock the judge to return scores above threshold so the quality loop completes
        svc._judge_relationship_quality.return_value = RelationshipQualityScores(
            tension=8.0, dynamics=8.0, story_potential=8.0, authenticity=8.0, feedback="Good"
        )

        # This should NOT reject the relationship because _normalize_name
        # strips "The " and matches "castle" == "castle"
        result, scores, _iterations = generate_relationship_with_quality(
            svc, story_state, ["The Castle", "Alice"], [], required_entity="Castle"
        )

        # Ensure the creator was actually invoked, proving the relationship was not
        # rejected by the _is_empty check due to normalization failure.
        svc._create_relationship.assert_called()
        assert result["source"] == "The Castle"
        assert scores.average >= 6.0


class TestRelationshipAutoPass:
    """Tests for relationship auto-pass when historical first-pass rate >= 95%."""

    def test_auto_pass_when_first_pass_rate_high(self, story_state):
        """Relationship auto-passes judge when first-pass rate >= 95%."""
        from src.memory.world_quality._story_scores import RelationshipQualityScores
        from src.services.world_quality_service._relationship import (
            generate_relationship_with_quality,
        )

        svc = MagicMock()
        svc.analytics_db = MagicMock()
        svc.analytics_db.get_first_pass_rate.return_value = 1.0  # 100% first-pass
        svc._get_creator_model.return_value = "test-model:8b"
        svc._get_judge_model.return_value = "test-model:8b"
        svc.get_calendar_context.return_value = ""
        config = MagicMock()
        config.creator_temperature = 0.9
        config.judge_temperature = 0.1
        config.get_threshold.return_value = 7.0
        config.get_refinement_temperature.return_value = 0.7
        config.max_iterations = 3
        config.early_stopping_patience = 2
        config.early_stopping_min_iterations = 1
        config.temperature_decay_rate = 0.0
        svc.get_config.return_value = config

        judge_config = MagicMock()
        judge_config.enabled = False
        judge_config.multi_call_enabled = False
        svc.get_judge_config.return_value = judge_config

        svc._create_relationship.return_value = {
            "source": "Alpha",
            "target": "Beta",
            "relation_type": "allies_with",
            "description": "They are allies",
        }

        entity_names = ["Alpha", "Beta"]
        _rel, scores, iterations = generate_relationship_with_quality(
            svc, story_state, entity_names, []
        )

        # Auto-pass scores are derived from the configured threshold (7.0)
        assert isinstance(scores, RelationshipQualityScores)
        assert scores.tension == 7.0
        assert scores.dynamics == 7.0
        assert scores.story_potential == 7.0
        assert scores.authenticity == 7.0
        assert "Auto-passed" in scores.feedback
        # 0 scoring rounds because judge was skipped entirely
        assert iterations == 0

    def test_no_auto_pass_when_rate_below_threshold(self, story_state):
        """Judge is invoked when first-pass rate is below the 95% threshold."""
        from src.memory.world_quality._story_scores import RelationshipQualityScores
        from src.services.world_quality_service._relationship import (
            generate_relationship_with_quality,
        )

        svc = MagicMock()
        svc.analytics_db = MagicMock()
        svc.analytics_db.get_first_pass_rate.return_value = 0.94  # Below 95%
        svc._get_creator_model.return_value = "test-model:8b"
        svc._get_judge_model.return_value = "test-model:8b"
        svc.get_calendar_context.return_value = ""
        config = MagicMock()
        config.creator_temperature = 0.9
        config.judge_temperature = 0.1
        config.get_threshold.return_value = 7.0
        config.get_refinement_temperature.return_value = 0.7
        config.max_iterations = 1
        config.early_stopping_patience = 2
        config.early_stopping_min_iterations = 1
        config.temperature_decay_rate = 0.0
        svc.get_config.return_value = config

        judge_config = MagicMock()
        judge_config.enabled = False
        judge_config.multi_call_enabled = False
        svc.get_judge_config.return_value = judge_config

        svc._create_relationship.return_value = {
            "source": "Alpha",
            "target": "Beta",
            "relation_type": "allies_with",
            "description": "They are allies",
        }
        judge_scores = RelationshipQualityScores(
            tension=8.0, dynamics=8.0, story_potential=8.0, authenticity=8.0, feedback="Good"
        )
        svc._judge_relationship_quality.return_value = judge_scores

        entity_names = ["Alpha", "Beta"]
        _rel, _scores, iterations = generate_relationship_with_quality(
            svc, story_state, entity_names, []
        )

        # Judge was invoked (not auto-passed), so iterations > 0
        assert iterations >= 1
        svc._judge_relationship_quality.assert_called()

    def test_auto_pass_skipped_when_analytics_db_absent(self, story_state):
        """Auto-pass is skipped when analytics_db is not available."""
        from src.memory.world_quality._story_scores import RelationshipQualityScores
        from src.services.world_quality_service._relationship import (
            generate_relationship_with_quality,
        )

        svc = MagicMock(spec=[])
        svc._get_creator_model = MagicMock(return_value="test-model:8b")
        svc._get_judge_model = MagicMock(return_value="test-model:8b")
        svc.get_calendar_context = MagicMock(return_value="")
        svc._log_refinement_analytics = MagicMock()
        config = MagicMock()
        config.creator_temperature = 0.9
        config.judge_temperature = 0.1
        config.get_threshold.return_value = 7.0
        config.get_refinement_temperature.return_value = 0.7
        config.max_iterations = 1
        config.early_stopping_patience = 2
        config.early_stopping_min_iterations = 1
        config.temperature_decay_rate = 0.0
        svc.get_config = MagicMock(return_value=config)

        judge_config = MagicMock()
        judge_config.enabled = False
        judge_config.multi_call_enabled = False
        svc.get_judge_config = MagicMock(return_value=judge_config)

        svc._create_relationship = MagicMock(
            return_value={
                "source": "Alpha",
                "target": "Beta",
                "relation_type": "allies_with",
                "description": "They are allies",
            }
        )
        judge_scores = RelationshipQualityScores(
            tension=8.0, dynamics=8.0, story_potential=8.0, authenticity=8.0, feedback="Good"
        )
        svc._judge_relationship_quality = MagicMock(return_value=judge_scores)
        svc._refine_relationship = MagicMock()

        entity_names = ["Alpha", "Beta"]
        _rel, _scores, iterations = generate_relationship_with_quality(
            svc, story_state, entity_names, []
        )

        # Without analytics_db, auto-pass cannot activate; judge must be invoked
        assert iterations >= 1
        svc._judge_relationship_quality.assert_called()

    def test_auto_pass_at_exact_boundary(self, story_state):
        """Auto-pass triggers at exactly 95% first-pass rate (boundary)."""
        from src.memory.world_quality._story_scores import RelationshipQualityScores
        from src.services.world_quality_service._relationship import (
            generate_relationship_with_quality,
        )

        svc = MagicMock()
        svc.analytics_db = MagicMock()
        svc.analytics_db.get_first_pass_rate.return_value = 0.95  # Exact boundary
        svc._get_creator_model.return_value = "test-model:8b"
        svc._get_judge_model.return_value = "test-model:8b"
        svc.get_calendar_context.return_value = ""
        config = MagicMock()
        config.creator_temperature = 0.9
        config.judge_temperature = 0.1
        config.get_threshold.return_value = 7.0
        config.get_refinement_temperature.return_value = 0.7
        config.max_iterations = 3
        config.early_stopping_patience = 2
        config.early_stopping_min_iterations = 1
        config.temperature_decay_rate = 0.0
        svc.get_config.return_value = config

        judge_config = MagicMock()
        judge_config.enabled = False
        judge_config.multi_call_enabled = False
        svc.get_judge_config.return_value = judge_config

        svc._create_relationship.return_value = {
            "source": "Alpha",
            "target": "Beta",
            "relation_type": "allies_with",
            "description": "They are allies",
        }

        entity_names = ["Alpha", "Beta"]
        _rel, scores, iterations = generate_relationship_with_quality(
            svc, story_state, entity_names, []
        )

        # At exactly 95%, auto-pass should trigger
        assert isinstance(scores, RelationshipQualityScores)
        assert iterations == 0
        assert "Auto-passed" in scores.feedback

    def test_auto_pass_unexpected_error_falls_through_to_judge(self, story_state, caplog):
        """Auto-pass logs warning and falls through to judge when analytics raises RuntimeError."""
        import logging

        from src.memory.world_quality._story_scores import RelationshipQualityScores
        from src.services.world_quality_service._relationship import (
            generate_relationship_with_quality,
        )

        svc = MagicMock()
        analytics_db = MagicMock()
        analytics_db.get_first_pass_rate.side_effect = RuntimeError("DB connection lost")
        svc.analytics_db = analytics_db
        svc._get_creator_model.return_value = "test-model:8b"
        svc._get_judge_model.return_value = "test-model:8b"
        svc.get_calendar_context.return_value = ""
        svc._log_refinement_analytics = MagicMock()
        config = MagicMock()
        config.creator_temperature = 0.9
        config.judge_temperature = 0.1
        config.get_threshold.return_value = 7.0
        config.get_refinement_temperature.return_value = 0.7
        config.max_iterations = 1
        config.early_stopping_patience = 2
        config.early_stopping_min_iterations = 1
        config.temperature_decay_rate = 0.0
        svc.get_config = MagicMock(return_value=config)

        judge_config = MagicMock()
        judge_config.enabled = False
        judge_config.multi_call_enabled = False
        svc.get_judge_config = MagicMock(return_value=judge_config)

        svc._create_relationship = MagicMock(
            return_value={
                "source": "Alpha",
                "target": "Beta",
                "relation_type": "allies_with",
                "description": "They are allies",
            }
        )
        judge_scores = RelationshipQualityScores(
            tension=8.0, dynamics=8.0, story_potential=8.0, authenticity=8.0, feedback="Good"
        )
        svc._judge_relationship_quality = MagicMock(return_value=judge_scores)
        svc._refine_relationship = MagicMock()

        entity_names = ["Alpha", "Beta"]
        with caplog.at_level(logging.WARNING):
            _rel, _scores, iterations = generate_relationship_with_quality(
                svc, story_state, entity_names, []
            )

        # RuntimeError should be caught; auto-pass skipped, judge invoked
        assert iterations >= 1
        svc._judge_relationship_quality.assert_called()
        assert any("Relationship auto-pass check failed" in msg for msg in caplog.messages)

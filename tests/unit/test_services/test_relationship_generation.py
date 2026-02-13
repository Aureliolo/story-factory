"""Tests for relationship generation improvements (#329)."""

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
        svc.settings.llm_tokens_relationship_create = 800
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
        for src, tgt, _type in existing_rels:
            assert f"{src} <-> {tgt}" in prompt_arg, f"Pair {src} <-> {tgt} missing from prompt"


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

        # Should stop after MAX_CONSECUTIVE_BATCH_FAILURES, not try all 10
        assert call_count == MAX_CONSECUTIVE_BATCH_FAILURES

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
        svc = WorldService(settings)

        # Set up world_db with only 5 entities
        world_db = WorldDatabase(tmp_path / "test_scale.db")
        for i in range(5):
            world_db.add_entity("character", f"Char{i}", f"Character {i}")

        services = MagicMock()
        # Capture the count passed to generate_relationships_with_quality
        captured_count = {}

        def capture_generate(state, names, rels, count, cancel_check=None):
            """Capture the count argument for later assertion."""
            captured_count["value"] = count
            return []

        services.world_quality.generate_relationships_with_quality.side_effect = capture_generate

        from src.services.world_service._build import _generate_relationships

        with patch("src.services.world_service._build.random.randint", return_value=25):
            _generate_relationships(svc, story_state, world_db, services)

        # 5 entities x 1.5 = 7, so count should be capped to 7
        assert captured_count["value"] == 7
        world_db.close()

    def test_dynamic_scaling_no_cap_when_entities_sufficient(self, story_state, tmp_path):
        """20 entities → max 30, settings request 15 → stays 15."""
        from src.services.world_service import WorldService

        settings = MagicMock()
        settings.world_gen_relationships_min = 15
        settings.world_gen_relationships_max = 15
        svc = WorldService(settings)

        world_db = WorldDatabase(tmp_path / "test_no_cap.db")
        for i in range(20):
            world_db.add_entity("character", f"Char{i}", f"Character {i}")

        services = MagicMock()
        captured_count = {}

        def capture_generate(state, names, rels, count, cancel_check=None):
            """Capture the count argument for later assertion."""
            captured_count["value"] = count
            return []

        services.world_quality.generate_relationships_with_quality.side_effect = capture_generate

        from src.services.world_service._build import _generate_relationships

        with patch("src.services.world_service._build.random.randint", return_value=15):
            _generate_relationships(svc, story_state, world_db, services)

        # 20 entities x 1.5 = 30, which is above 15, so no cap applied
        assert captured_count["value"] == 15
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
        svc.settings.llm_tokens_relationship_create = 800
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
        svc.settings.llm_tokens_relationship_create = 800
        svc._get_creator_model.return_value = "test-model:8b"
        # LLM returns a prose-style type
        svc.client.generate.return_value = {
            "response": '{"source": "Alpha", "target": "Beta", '
            '"relation_type": "bitter rivals who fought for years", '
            '"description": "A long-standing feud"}'
        }

        entity_names = ["Alpha", "Beta"]
        result = _create_relationship(svc, story_state, entity_names, [], 0.9)

        # "bitter rivals who fought for years" should normalize to "rivals"
        assert result["relation_type"] == "rivals"

    def test_relation_type_normalized_in_build(self, story_state, tmp_path):
        """Relation type should be normalized when stored via _generate_relationships."""
        from src.services.world_service import WorldService
        from src.services.world_service._build import _generate_relationships

        settings = MagicMock()
        settings.world_gen_relationships_min = 1
        settings.world_gen_relationships_max = 1
        svc = WorldService(settings)

        world_db = WorldDatabase(tmp_path / "test_normalize.db")
        world_db.add_entity("character", "Alpha", "Character Alpha")
        world_db.add_entity("character", "Beta", "Character Beta")

        services = MagicMock()

        def fake_generate(state, names, rels, count, cancel_check=None):
            """Return a relationship with a prose-style type containing 'rivals'."""
            scores = MagicMock()
            scores.average = 8.0
            return [
                (
                    {
                        "source": "Alpha",
                        "target": "Beta",
                        "relation_type": "bitter rivals who fought",
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
        # "bitter rivals who fought" normalizes to "rivals" via substring match
        assert rels[0].relation_type == "rivals"
        world_db.close()


class TestRelationTypeNormalizedInBuild:
    """D4: Verify _build.py normalizes relation_type before storage."""

    def test_known_type_stored_correctly(self, story_state, tmp_path):
        """A known type should be stored as-is after normalization."""
        from src.services.world_service import WorldService
        from src.services.world_service._build import _generate_relationships

        settings = MagicMock()
        settings.world_gen_relationships_min = 1
        settings.world_gen_relationships_max = 1
        svc = WorldService(settings)

        world_db = WorldDatabase(tmp_path / "test_build_norm.db")
        world_db.add_entity("character", "Alpha", "Character Alpha")
        world_db.add_entity("character", "Beta", "Character Beta")

        services = MagicMock()

        def fake_generate(state, names, rels, count, cancel_check=None):
            """Return a relationship with hyphenated-case type."""
            scores = MagicMock()
            scores.average = 8.0
            return [
                (
                    {
                        "source": "Alpha",
                        "target": "Beta",
                        "relation_type": "Allies-With",
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
        world_db.close()


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
        svc.settings.llm_tokens_relationship_create = 800
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

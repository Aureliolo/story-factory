"""Tests for WorldQualityService - cancellation, progress, and logging."""

import logging
import time
from unittest.mock import MagicMock, patch

import pytest

from src.memory.world_quality import BaseQualityScores, CharacterQualityScores, FactionQualityScores
from src.services.model_mode_service import ModelModeService
from src.services.world_quality_service import (
    EntityGenerationProgress,
    WorldQualityService,
)
from src.services.world_quality_service._relationship import (
    _compute_entity_frequency_hint,
    _is_duplicate_relationship,
)
from tests.shared.mock_ollama import TEST_MODEL


@pytest.fixture
def mock_mode_service(tmp_path):
    """Create a mock ModelModeService."""
    mode_service = MagicMock(spec=ModelModeService)
    mode_service.get_model_for_agent.return_value = TEST_MODEL
    return mode_service


@pytest.fixture
def world_quality_service(tmp_settings, mock_mode_service):
    """Create a WorldQualityService with mocked dependencies."""
    return WorldQualityService(tmp_settings, mock_mode_service)


class TestEntityGenerationProgress:
    """Tests for EntityGenerationProgress dataclass."""

    def test_progress_fraction_at_start(self):
        """Progress fraction is 0.0 at the start."""
        progress = EntityGenerationProgress(current=1, total=5, entity_type="character")
        assert progress.progress_fraction == 0.0

    def test_progress_fraction_midway(self):
        """Progress fraction is accurate midway through."""
        progress = EntityGenerationProgress(current=3, total=5, entity_type="character")
        # (3-1)/5 = 0.4
        assert progress.progress_fraction == 0.4

    def test_progress_fraction_near_end(self):
        """Progress fraction is accurate near the end."""
        progress = EntityGenerationProgress(current=5, total=5, entity_type="character")
        # (5-1)/5 = 0.8
        assert progress.progress_fraction == 0.8

    def test_progress_fraction_with_zero_total(self):
        """Progress fraction handles zero total gracefully."""
        progress = EntityGenerationProgress(current=1, total=0, entity_type="character")
        # Division by zero protection: (1-1)/max(0,1) = 0.0
        assert progress.progress_fraction == 0.0

    def test_default_values(self):
        """Default values are set correctly."""
        progress = EntityGenerationProgress(current=1, total=5, entity_type="character")
        assert progress.entity_name is None
        assert progress.phase == "generating"
        assert progress.elapsed_seconds == 0.0
        assert progress.estimated_remaining_seconds is None

    def test_all_values(self):
        """All values can be set."""
        progress = EntityGenerationProgress(
            current=2,
            total=5,
            entity_type="character",
            entity_name="Alice",
            phase="complete",
            elapsed_seconds=10.5,
            estimated_remaining_seconds=30.0,
        )
        assert progress.current == 2
        assert progress.total == 5
        assert progress.entity_type == "character"
        assert progress.entity_name == "Alice"
        assert progress.phase == "complete"
        assert progress.elapsed_seconds == 10.5
        assert progress.estimated_remaining_seconds == 30.0


class TestCalculateEta:
    """Tests for WorldQualityService._calculate_eta method."""

    def test_eta_with_empty_times(self):
        """ETA is None with no completed times."""
        result = WorldQualityService._calculate_eta([], 3)
        assert result is None

    def test_eta_with_zero_remaining(self):
        """ETA is None with zero remaining."""
        result = WorldQualityService._calculate_eta([10.0, 12.0], 0)
        assert result is None

    def test_eta_with_negative_remaining(self):
        """ETA is None with negative remaining."""
        result = WorldQualityService._calculate_eta([10.0, 12.0], -1)
        assert result is None

    def test_eta_with_single_time(self):
        """ETA with single time uses that time."""
        result = WorldQualityService._calculate_eta([10.0], 3)
        # With single value, EMA equals that value: 10 * 3 = 30
        assert result == 30.0

    def test_eta_with_multiple_times(self):
        """ETA with multiple times uses EMA."""
        # Times: [10.0, 20.0]
        # EMA: start=10.0, then alpha*20 + (1-alpha)*10 = 0.3*20 + 0.7*10 = 6 + 7 = 13
        # 13 * 2 = 26
        result = WorldQualityService._calculate_eta([10.0, 20.0], 2)
        assert result == 26.0

    def test_eta_weights_recent_times_more(self):
        """ETA weights recent times more heavily due to EMA."""
        # If recent times are faster, ETA should reflect that
        # Times: [30.0, 10.0] - started slow, got faster
        # EMA: start=30.0, then 0.3*10 + 0.7*30 = 3 + 21 = 24
        # 24 * 2 = 48
        result_faster_recent = WorldQualityService._calculate_eta([30.0, 10.0], 2)

        # Times: [10.0, 30.0] - started fast, got slower
        # EMA: start=10.0, then 0.3*30 + 0.7*10 = 9 + 7 = 16
        # 16 * 2 = 32
        result_slower_recent = WorldQualityService._calculate_eta([10.0, 30.0], 2)

        # Both results should be non-None floats
        assert result_faster_recent is not None
        assert result_slower_recent is not None
        # With EMA alpha=0.3, the first value dominates (70% weight on old average)
        # So starting slow [30, 10] gives higher ETA than starting fast [10, 30]
        assert result_slower_recent < result_faster_recent


class TestMiniDescriptionLogging:
    """Tests for mini description generation logging (#171)."""

    def test_entry_logging_includes_entity_count(self, caplog, world_quality_service):
        """Entry logging includes entity count."""
        with caplog.at_level(logging.INFO):
            # Mock generate_mini_description to avoid actual LLM call
            with patch.object(
                world_quality_service,
                "generate_mini_description",
                return_value="A short description",
            ):
                entities = [
                    {"name": "Alice", "type": "character", "description": "A protagonist"},
                    {"name": "Bob", "type": "character", "description": "A friend"},
                    {"name": "Castle", "type": "location", "description": "A fortress"},
                ]
                world_quality_service.generate_mini_descriptions_batch(entities)

        # Check entry log message
        entry_logs = [
            r for r in caplog.records if "Starting mini description generation" in r.message
        ]
        assert len(entry_logs) == 1
        assert "3 entities" in entry_logs[0].message

    def test_per_entity_progress_logging(self, caplog, world_quality_service):
        """Debug logging shows progress like '1/3: character Alice'."""
        with caplog.at_level(logging.DEBUG):
            with patch.object(
                world_quality_service,
                "generate_mini_description",
                return_value="A short description",
            ):
                entities = [
                    {"name": "Alice", "type": "character", "description": "A protagonist"},
                    {"name": "Bob", "type": "character", "description": "A friend"},
                ]
                world_quality_service.generate_mini_descriptions_batch(entities)

        # Check per-entity progress logs
        progress_logs = [r for r in caplog.records if "Generating mini description" in r.message]
        assert len(progress_logs) == 2
        assert "1/2" in progress_logs[0].message
        assert "character 'Alice'" in progress_logs[0].message
        assert "2/2" in progress_logs[1].message
        assert "character 'Bob'" in progress_logs[1].message

    def test_completion_logging_includes_timing(self, caplog, world_quality_service):
        """Completion logging includes elapsed time."""
        with caplog.at_level(logging.INFO):
            with patch.object(
                world_quality_service,
                "generate_mini_description",
                return_value="A short description",
            ):
                entities = [
                    {"name": "Alice", "type": "character", "description": "A protagonist"},
                ]
                world_quality_service.generate_mini_descriptions_batch(entities)

        # Check completion log message
        completion_logs = [
            r for r in caplog.records if "Completed mini description batch" in r.message
        ]
        assert len(completion_logs) == 1
        assert "1 descriptions" in completion_logs[0].message
        # Should include timing info
        assert "s" in completion_logs[0].message  # seconds

    def test_empty_descriptions_not_processed(self, caplog, world_quality_service):
        """Entities without descriptions are filtered out."""
        with caplog.at_level(logging.INFO):
            with patch.object(
                world_quality_service,
                "generate_mini_description",
                return_value="A short description",
            ) as mock_gen:
                entities = [
                    {"name": "Alice", "type": "character", "description": "A protagonist"},
                    {"name": "Empty", "type": "character", "description": ""},
                    {"name": "None", "type": "character", "description": None},
                ]
                result = world_quality_service.generate_mini_descriptions_batch(entities)

        # Only Alice should be processed
        assert mock_gen.call_count == 1
        assert "Alice" in result
        assert "Empty" not in result
        assert "None" not in result

        # Log should say 1 entity
        entry_logs = [
            r for r in caplog.records if "Starting mini description generation" in r.message
        ]
        assert "1 entities" in entry_logs[0].message


class TestCancellationSupport:
    """Tests for cancellation support in batch generation (#180)."""

    def test_characters_respects_cancel_check(self, world_quality_service, sample_story_state):
        """Generation stops when cancel_check returns True."""
        call_count = 0
        cancel_after = 2

        def cancel_check():
            """Return True after cancel_after calls."""
            nonlocal call_count
            call_count += 1
            return call_count > cancel_after

        # Mock the single character generation
        mock_scores = MagicMock(spec=CharacterQualityScores)
        mock_scores.average = 7.5

        with patch.object(
            world_quality_service,
            "generate_character_with_quality",
            side_effect=lambda *args, **kwargs: (
                MagicMock(name=f"Char{call_count}"),
                mock_scores,
                1,
            ),
        ):
            results = world_quality_service.generate_characters_with_quality(
                sample_story_state,
                existing_names=[],
                count=5,
                cancel_check=cancel_check,
            )

        # Should have generated 2 characters before cancel was triggered
        assert len(results) == 2

    def test_cancellation_preserves_completed_entities(
        self, world_quality_service, sample_story_state
    ):
        """Entities completed before cancel are returned."""
        generated_names: list[str] = []
        cancel_after = 3

        def cancel_check():
            """Return True when cancel_after entities have been generated."""
            return len(generated_names) >= cancel_after

        def mock_generate(*args, **kwargs):
            """Mock character generation that tracks generated names."""
            name = f"Character_{len(generated_names) + 1}"
            mock_char = MagicMock()
            mock_char.name = name
            generated_names.append(name)
            mock_scores = MagicMock()
            mock_scores.average = 7.5
            return mock_char, mock_scores, 1

        with patch.object(
            world_quality_service,
            "generate_character_with_quality",
            side_effect=mock_generate,
        ):
            results = world_quality_service.generate_characters_with_quality(
                sample_story_state,
                existing_names=[],
                count=10,
                cancel_check=cancel_check,
            )

        # Should have exactly 3 characters
        assert len(results) == 3
        # All generated names should be in results
        result_names = [r[0].name for r in results]
        assert result_names == ["Character_1", "Character_2", "Character_3"]

    def test_cancel_check_only_called_between_entities(
        self, world_quality_service, sample_story_state
    ):
        """Cancel is checked before each entity, not during LLM call."""
        cancel_check_times = []
        generation_times = []

        def cancel_check():
            """Record timestamp and return False (never cancel)."""
            cancel_check_times.append(time.time())
            return False

        def mock_generate(*args, **kwargs):
            """Record timestamp and return mock character."""
            generation_times.append(time.time())
            mock_char = MagicMock()
            mock_char.name = f"Char_{len(generation_times)}"
            mock_scores = MagicMock()
            mock_scores.average = 7.5
            return mock_char, mock_scores, 1

        with patch.object(
            world_quality_service,
            "generate_character_with_quality",
            side_effect=mock_generate,
        ):
            world_quality_service.generate_characters_with_quality(
                sample_story_state,
                existing_names=[],
                count=3,
                cancel_check=cancel_check,
            )

        # Cancel check should be called 3 times (once before each entity)
        assert len(cancel_check_times) == 3
        # Each cancel check should happen before the generation
        for i in range(3):
            assert cancel_check_times[i] < generation_times[i]


class TestProgressCallback:
    """Tests for progress callback in batch generation (#185)."""

    def test_progress_callback_called_per_entity(self, world_quality_service, sample_story_state):
        """Callback receives update for each entity."""
        progress_updates = []

        def progress_callback(progress: EntityGenerationProgress):
            """Collect progress updates."""
            progress_updates.append(progress)

        def mock_generate(*args, **kwargs):
            """Return mock character with progress-aware name."""
            mock_char = MagicMock()
            mock_char.name = f"Char_{len(progress_updates) // 2 + 1}"
            mock_scores = MagicMock()
            mock_scores.average = 7.5
            return mock_char, mock_scores, 1

        with patch.object(
            world_quality_service,
            "generate_character_with_quality",
            side_effect=mock_generate,
        ):
            world_quality_service.generate_characters_with_quality(
                sample_story_state,
                existing_names=[],
                count=3,
                progress_callback=progress_callback,
            )

        # Should have 6 updates: 2 per entity (before and after)
        assert len(progress_updates) == 6

        # Check "generating" updates (indices: 0, 2, 4)
        for i in [0, 2, 4]:
            assert progress_updates[i].phase == "generating"
            assert progress_updates[i].entity_name is None

        # Check "complete" updates (indices: 1, 3, 5)
        for i in [1, 3, 5]:
            assert progress_updates[i].phase == "complete"
            assert progress_updates[i].entity_name is not None

    def test_progress_includes_eta_after_first(self, world_quality_service, sample_story_state):
        """ETA calculated after first entity completes."""
        progress_updates = []

        def progress_callback(progress: EntityGenerationProgress):
            """Collect progress updates."""
            progress_updates.append(progress)

        call_count = 0

        def mock_generate(*args, **kwargs):
            """Mock generation with simulated delay."""
            nonlocal call_count
            call_count += 1
            # Simulate some generation time
            time.sleep(0.01)
            mock_char = MagicMock()
            mock_char.name = f"Char_{call_count}"
            mock_scores = MagicMock()
            mock_scores.average = 7.5
            return mock_char, mock_scores, 1

        with patch.object(
            world_quality_service,
            "generate_character_with_quality",
            side_effect=mock_generate,
        ):
            world_quality_service.generate_characters_with_quality(
                sample_story_state,
                existing_names=[],
                count=3,
                progress_callback=progress_callback,
            )

        # First "generating" update should have no ETA (no prior data)
        assert progress_updates[0].estimated_remaining_seconds is None

        # After first completion, subsequent generating updates should have ETA
        # Index 2 is second entity's "generating" phase
        assert progress_updates[2].estimated_remaining_seconds is not None

    def test_progress_fraction_accuracy(self, world_quality_service, sample_story_state):
        """Progress fraction matches current/total."""
        progress_updates = []

        def progress_callback(progress: EntityGenerationProgress):
            """Collect progress updates to verify fraction accuracy."""
            progress_updates.append(progress)

        def mock_generate(*args, **kwargs):
            """Return mock character with fixed scores for fraction testing."""
            mock_char = MagicMock()
            mock_char.name = "TestChar"
            mock_scores = MagicMock()
            mock_scores.average = 7.5
            return mock_char, mock_scores, 1

        with patch.object(
            world_quality_service,
            "generate_character_with_quality",
            side_effect=mock_generate,
        ):
            world_quality_service.generate_characters_with_quality(
                sample_story_state,
                existing_names=[],
                count=4,
                progress_callback=progress_callback,
            )

        # Check generating phase progress fractions
        generating_updates = [p for p in progress_updates if p.phase == "generating"]
        expected_fractions = [0.0, 0.25, 0.5, 0.75]  # (current-1)/total

        for update, expected in zip(generating_updates, expected_fractions, strict=True):
            assert update.progress_fraction == expected

    def test_progress_callback_works_with_other_entity_types(
        self, world_quality_service, sample_story_state
    ):
        """Progress callback works for locations, factions, items, concepts."""
        # Test locations
        location_updates = []

        def mock_loc(*args, **kwargs):
            """Return mock location data for multi-entity-type testing."""
            return {"name": "TestLoc", "description": "A place"}, MagicMock(average=7.5), 1

        with patch.object(
            world_quality_service, "generate_location_with_quality", side_effect=mock_loc
        ):
            world_quality_service.generate_locations_with_quality(
                sample_story_state,
                existing_names=[],
                count=2,
                progress_callback=lambda p: location_updates.append(p),
            )

        assert len(location_updates) == 4  # 2 per entity
        assert all(p.entity_type == "location" for p in location_updates)

        # Test items
        item_updates = []

        def mock_item(*args, **kwargs):
            """Return mock item data for multi-entity-type testing."""
            return {"name": "TestItem", "description": "A thing"}, MagicMock(average=7.5), 1

        with patch.object(
            world_quality_service, "generate_item_with_quality", side_effect=mock_item
        ):
            world_quality_service.generate_items_with_quality(
                sample_story_state,
                existing_names=[],
                count=2,
                progress_callback=lambda p: item_updates.append(p),
            )

        assert len(item_updates) == 4
        assert all(p.entity_type == "item" for p in item_updates)

    def test_faction_cancellation_and_progress(self, world_quality_service, sample_story_state):
        """Faction generation supports cancel_check and progress_callback."""
        cancel_after = 1
        call_count = [0]
        progress_updates = []

        def cancel_check():
            """Cancel after first faction is generated."""
            return call_count[0] >= cancel_after

        def mock_faction(*args, **kwargs):
            """Return mock faction data and track call count."""
            call_count[0] += 1
            return (
                {"name": f"Faction{call_count[0]}", "description": "A group"},
                MagicMock(average=7.5),
                1,
            )

        def progress_callback(progress: EntityGenerationProgress):
            """Collect faction progress updates."""
            progress_updates.append(progress)

        with patch.object(
            world_quality_service, "generate_faction_with_quality", side_effect=mock_faction
        ):
            results = world_quality_service.generate_factions_with_quality(
                sample_story_state,
                existing_names=[],
                count=3,
                cancel_check=cancel_check,
                progress_callback=progress_callback,
            )

        # Should have completed 1 and cancelled before second
        assert len(results) == 1
        assert len(progress_updates) == 2  # 1 generating + 1 complete
        assert all(p.entity_type == "faction" for p in progress_updates)

    def test_item_cancellation(self, world_quality_service, sample_story_state):
        """Item generation respects cancel_check."""
        cancel_after = 1
        call_count = [0]

        def cancel_check():
            """Cancel after first item is generated."""
            return call_count[0] >= cancel_after

        def mock_item(*args, **kwargs):
            """Return mock item data and track call count."""
            call_count[0] += 1
            return (
                {"name": f"Item{call_count[0]}", "description": "A thing"},
                MagicMock(average=7.5),
                1,
            )

        with patch.object(
            world_quality_service, "generate_item_with_quality", side_effect=mock_item
        ):
            results = world_quality_service.generate_items_with_quality(
                sample_story_state,
                existing_names=[],
                count=3,
                cancel_check=cancel_check,
            )

        # Should have completed 1 and cancelled before second
        assert len(results) == 1

    def test_location_cancellation(self, world_quality_service, sample_story_state):
        """Location generation respects cancel_check."""
        cancel_after = 1
        call_count = [0]

        def cancel_check():
            """Cancel after first location is generated."""
            return call_count[0] >= cancel_after

        def mock_location(*args, **kwargs):
            """Return mock location data and track call count."""
            call_count[0] += 1
            return (
                {"name": f"Location{call_count[0]}", "description": "A place"},
                MagicMock(average=7.5),
                1,
            )

        with patch.object(
            world_quality_service, "generate_location_with_quality", side_effect=mock_location
        ):
            results = world_quality_service.generate_locations_with_quality(
                sample_story_state,
                existing_names=[],
                count=3,
                cancel_check=cancel_check,
            )

        # Should have completed 1 and cancelled before second
        assert len(results) == 1

    def test_concept_cancellation_and_progress(self, world_quality_service, sample_story_state):
        """Concept generation supports cancel_check and progress_callback."""
        cancel_after = 1
        call_count = [0]
        progress_updates = []

        def cancel_check():
            """Cancel after first concept is generated."""
            return call_count[0] >= cancel_after

        def mock_concept(*args, **kwargs):
            """Return mock concept data and track call count."""
            call_count[0] += 1
            return (
                {"name": f"Concept{call_count[0]}", "description": "An idea"},
                MagicMock(average=7.5),
                1,
            )

        def progress_callback(progress: EntityGenerationProgress):
            """Collect concept progress updates."""
            progress_updates.append(progress)

        with patch.object(
            world_quality_service, "generate_concept_with_quality", side_effect=mock_concept
        ):
            results = world_quality_service.generate_concepts_with_quality(
                sample_story_state,
                existing_names=[],
                count=3,
                cancel_check=cancel_check,
                progress_callback=progress_callback,
            )

        # Should have completed 1 and cancelled before second
        assert len(results) == 1
        assert len(progress_updates) == 2  # 1 generating + 1 complete
        assert all(p.entity_type == "concept" for p in progress_updates)

    def test_relationship_cancellation_and_progress(
        self, world_quality_service, sample_story_state
    ):
        """Relationship generation supports cancel_check and progress_callback."""
        cancel_after = 1
        call_count = [0]
        progress_updates = []

        def cancel_check():
            """Cancel after first relationship is generated."""
            return call_count[0] >= cancel_after

        def mock_relationship(*args, **kwargs):
            """Return mock relationship data and track call count."""
            call_count[0] += 1
            return (
                {
                    "source": "Entity1",
                    "target": "Entity2",
                    "relation_type": "knows",
                    "description": "A connection",
                },
                MagicMock(average=7.5),
                1,
            )

        def progress_callback(progress: EntityGenerationProgress):
            """Collect relationship progress updates."""
            progress_updates.append(progress)

        # Entity names for relationship generation
        entity_names = ["Entity1", "Entity2", "Entity3", "Entity4"]

        with patch.object(
            world_quality_service,
            "generate_relationship_with_quality",
            side_effect=mock_relationship,
        ):
            results = world_quality_service.generate_relationships_with_quality(
                sample_story_state,
                entity_names=entity_names,
                existing_rels=[],
                count=3,
                cancel_check=cancel_check,
                progress_callback=progress_callback,
            )

        # Should have completed 1 and cancelled before second
        assert len(results) == 1
        assert len(progress_updates) == 2  # 1 generating + 1 complete
        assert all(p.entity_type == "relationship" for p in progress_updates)


class TestFactionCreationTemperatureEscalation:
    """Tests for temperature escalation when faction creation returns duplicate names."""

    def test_temperature_increases_on_creation_retries(self, world_quality_service):
        """When _create_faction returns empty (name conflict), temperature escalates."""
        svc = world_quality_service

        # Build a minimal StoryState with a brief
        brief = MagicMock()
        brief.genre = "fantasy"
        brief.premise = "A dark fantasy world"
        brief.tone = "dark"
        brief.themes = ["power"]
        brief.setting_place = "Realm"
        brief.setting_time = "Medieval"
        brief.language = "English"
        story_state = MagicMock()
        story_state.brief = brief
        story_state.id = "test-story"

        existing_names = ["The Luminous Circuit"]

        # Track temperature arguments passed to _create_faction
        temps_used = []

        def fake_create_faction(_story_state, _names, temperature, _locations=None):
            """Return empty dict twice (name conflict), then valid faction."""
            temps_used.append(temperature)
            # Return empty dict (name conflict) for first 2 calls, valid on 3rd
            if len(temps_used) <= 2:
                return {}
            return {"name": "New Faction", "type": "faction", "description": "A new faction"}

        # Mock all internal methods
        svc._create_faction = fake_create_faction
        svc._judge_faction_quality = MagicMock(
            return_value=FactionQualityScores(
                coherence=9,
                influence=9,
                conflict_potential=9,
                distinctiveness=9,
                temporal_plausibility=9,
                feedback="Good",
            )
        )
        svc._log_refinement_analytics = MagicMock()

        faction, _scores, _iterations = svc.generate_faction_with_quality(
            story_state, existing_names
        )

        assert faction["name"] == "New Faction"
        assert len(temps_used) == 3
        # First call: base temp (0.9), second: 0.9 + 0.15 = 1.05, third: 0.9 + 0.30 = 1.20
        assert temps_used[0] == pytest.approx(0.9, abs=0.01)
        assert temps_used[1] > temps_used[0]
        assert temps_used[2] > temps_used[1]

    def test_temperature_capped_at_1_5(self, world_quality_service):
        """Temperature escalation is capped at 1.5 even after many retries."""
        svc = world_quality_service

        brief = MagicMock()
        brief.genre = "fantasy"
        brief.premise = "A dark fantasy world"
        brief.tone = "dark"
        brief.themes = ["power"]
        brief.setting_place = "Realm"
        brief.setting_time = "Medieval"
        brief.language = "English"
        story_state = MagicMock()
        story_state.brief = brief
        story_state.id = "test-story"

        existing_names = ["Old Faction"]
        temps_used = []

        # Return empty 6 times (enough for temp to exceed 1.5 uncapped), then valid
        def fake_create_faction(_story_state, _names, temperature, _locations=None):
            """Return empty dict 6 times to force temperature past uncapped 1.5."""
            temps_used.append(temperature)
            if len(temps_used) <= 6:
                return {}
            return {"name": "Capped Faction", "type": "faction", "description": "desc"}

        svc._create_faction = fake_create_faction
        svc._judge_faction_quality = MagicMock(
            return_value=FactionQualityScores(
                coherence=9,
                influence=9,
                conflict_potential=9,
                distinctiveness=9,
                temporal_plausibility=9,
                feedback="Good",
            )
        )
        svc._log_refinement_analytics = MagicMock()

        # Override max_iterations so the loop runs long enough to test the cap
        config = svc.get_config()
        config_with_high_iterations = config.model_copy(update={"max_iterations": 8})
        svc.get_config = lambda: config_with_high_iterations

        faction, _scores, _iterations = svc.generate_faction_with_quality(
            story_state, existing_names
        )

        assert faction["name"] == "Capped Faction"
        assert len(temps_used) == 7
        # Retry 5 would be 0.9 + 5*0.15 = 1.65 uncapped, but should be capped at 1.5
        assert temps_used[5] == pytest.approx(1.5)
        # Retry 6 would be 0.9 + 6*0.15 = 1.80 uncapped, but should be capped at 1.5
        assert temps_used[6] == pytest.approx(1.5)
        # Verify all temps are <= 1.5
        assert all(t <= 1.5 for t in temps_used)


class TestDuplicateRelationshipDetection:
    """Tests for _is_duplicate_relationship bidirectionality."""

    def test_detects_same_direction_duplicate(self):
        """Duplicate detected when same source->target pair exists."""
        existing = [("Alice", "Bob", "knows")]
        assert _is_duplicate_relationship("Alice", "Bob", existing) is True

    def test_detects_reverse_direction_duplicate(self):
        """Duplicate detected when target->source pair exists (bidirectional)."""
        existing = [("Alice", "Bob", "knows")]
        assert _is_duplicate_relationship("Bob", "Alice", existing) is True

    def test_allows_new_pair(self):
        """New pair is not flagged as duplicate."""
        existing = [("Alice", "Bob", "knows")]
        assert _is_duplicate_relationship("Alice", "Carol", existing) is False

    def test_empty_existing_rels(self):
        """No duplicates when existing list is empty."""
        assert _is_duplicate_relationship("Alice", "Bob", []) is False

    def test_multiple_existing_rels(self):
        """Correctly checks against multiple existing pairs."""
        existing = [("Alice", "Bob", "knows"), ("Carol", "Dave", "loves")]
        assert _is_duplicate_relationship("Dave", "Carol", existing) is True
        assert _is_duplicate_relationship("Alice", "Carol", existing) is False


class TestJudgeCalibrationPrompts:
    """Tests for Issue 2: judge prompts contain scoring calibration text."""

    @pytest.fixture
    def judge_modules(self):
        """Import all judge function modules."""
        from src.services.world_quality_service import (
            _character as char_mod,
        )
        from src.services.world_quality_service import (
            _concept as concept_mod,
        )
        from src.services.world_quality_service import (
            _faction as faction_mod,
        )
        from src.services.world_quality_service import (
            _item as item_mod,
        )
        from src.services.world_quality_service import (
            _location as location_mod,
        )
        from src.services.world_quality_service import (
            _relationship as rel_mod,
        )

        return {
            "character": char_mod,
            "faction": faction_mod,
            "item": item_mod,
            "location": location_mod,
            "concept": concept_mod,
            "relationship": rel_mod,
        }

    def test_character_judge_has_calibration(self, judge_modules):
        """Test character judge module uses shared calibration block."""
        import inspect

        source = inspect.getsource(judge_modules["character"])
        assert "JUDGE_CALIBRATION_BLOCK" in source
        assert "judge_with_averaging" in source

    def test_faction_judge_has_calibration(self, judge_modules):
        """Test faction judge module uses shared calibration block."""
        import inspect

        source = inspect.getsource(judge_modules["faction"])
        assert "JUDGE_CALIBRATION_BLOCK" in source
        assert "judge_with_averaging" in source

    def test_item_judge_has_calibration(self, judge_modules):
        """Test item judge module uses shared calibration block."""
        import inspect

        source = inspect.getsource(judge_modules["item"])
        assert "JUDGE_CALIBRATION_BLOCK" in source
        assert "judge_with_averaging" in source

    def test_location_judge_has_calibration(self, judge_modules):
        """Test location judge module uses shared calibration block."""
        import inspect

        source = inspect.getsource(judge_modules["location"])
        assert "JUDGE_CALIBRATION_BLOCK" in source
        assert "judge_with_averaging" in source

    def test_concept_judge_has_calibration(self, judge_modules):
        """Test concept judge module uses shared calibration block."""
        import inspect

        source = inspect.getsource(judge_modules["concept"])
        assert "JUDGE_CALIBRATION_BLOCK" in source
        assert "judge_with_averaging" in source

    def test_relationship_judge_has_calibration(self, judge_modules):
        """Test relationship judge module uses shared calibration block."""
        import inspect

        source = inspect.getsource(judge_modules["relationship"])
        assert "JUDGE_CALIBRATION_BLOCK" in source
        assert "judge_with_averaging" in source


class TestItemRetryLogLevel:
    """Tests for Issue 4: item/location retry uses WARNING not ERROR."""

    def test_item_empty_creation_uses_warning_log(self):
        """Test that the quality loop uses logger.warning for empty creation retries."""
        import inspect

        from src.services.world_quality_service._quality_loop import quality_refinement_loop

        source = inspect.getsource(quality_refinement_loop)
        # Should use warning for empty creation retries, not error
        assert "logger.warning(" in source
        assert "creation_retries" in source

    def test_location_empty_creation_uses_warning_log(self):
        """Test that the quality loop uses logger.warning for empty creation retries."""
        import inspect

        from src.services.world_quality_service._quality_loop import quality_refinement_loop

        source = inspect.getsource(quality_refinement_loop)
        assert "logger.warning(" in source
        assert "creation_retries" in source


class TestRetryTemperatureHelper:
    """Tests for retry_temperature helper in shared _common module."""

    def test_zero_retries_returns_base_temperature(self):
        """With no retries, temperature equals the base creator temperature."""
        from src.services.world_quality_service._common import retry_temperature

        config = MagicMock()
        config.creator_temperature = 0.9
        assert retry_temperature(config, 0) == 0.9

    def test_increments_by_015_per_retry(self):
        """Each retry adds 0.15 to the temperature."""
        from src.services.world_quality_service._common import retry_temperature

        config = MagicMock()
        config.creator_temperature = 0.9
        assert retry_temperature(config, 1) == pytest.approx(1.05)
        assert retry_temperature(config, 2) == pytest.approx(1.2)

    def test_caps_at_1_5(self):
        """Temperature should never exceed 1.5."""
        from src.services.world_quality_service._common import retry_temperature

        config = MagicMock()
        config.creator_temperature = 0.9
        assert retry_temperature(config, 10) == 1.5

    def test_entity_modules_use_shared_retry_temperature(self):
        """All entity modules with retry logic should import from _common, not duplicate."""
        from src.services.world_quality_service import (
            _concept as concept_mod,
        )
        from src.services.world_quality_service import (
            _faction as faction_mod,
        )
        from src.services.world_quality_service import (
            _item as item_mod,
        )
        from src.services.world_quality_service import (
            _location as location_mod,
        )
        from src.services.world_quality_service._common import retry_temperature

        for name, mod in [
            ("item", item_mod),
            ("location", location_mod),
            ("faction", faction_mod),
            ("concept", concept_mod),
        ]:
            # Verify retry_temperature is actually imported from _common, not just
            # mentioned in a comment or string literal
            assert hasattr(mod, "retry_temperature"), (
                f"{name} module should import retry_temperature from _common"
            )
            assert mod.retry_temperature is retry_temperature, (
                f"{name} module's retry_temperature should be the same function from _common"
            )


class TestIterationCountAlignment:
    """Tests for iteration count alignment: use history iteration, not loop counter."""

    def test_quality_loop_uses_history_iteration(self):
        """Generic quality loop should use history.iterations[-1].iteration, not iteration + 1."""
        import inspect

        from src.services.world_quality_service._quality_loop import quality_refinement_loop

        source = inspect.getsource(quality_refinement_loop)
        assert "current_iter = history.iterations[-1].iteration" in source, (
            "quality_loop: should use history iteration, not loop counter"
        )
        assert "history.final_iteration = current_iter" in source, (
            "quality_loop: final_iteration should use current_iter"
        )
        assert "return " in source and "current_iter" in source, (
            "quality_loop: return should use current_iter"
        )


class TestModelSelectionLogLevel:
    """Tests for Issue 5: model auto-selection logs at DEBUG not INFO."""

    def test_modes_auto_selection_logs_at_debug(self):
        """Test that _modes.py auto-selection uses logger.debug."""
        import inspect

        from src.services.model_mode_service import _modes as modes_mod

        source = inspect.getsource(modes_mod.get_model_for_agent)
        # The auto-selection log should use debug, not info
        assert 'logger.debug(\n            f"Auto-selected' in source or (
            "logger.debug(" in source and "Auto-selected" in source
        )

    def test_settings_auto_selection_logs_at_debug(self):
        """Test that _settings.py tagged model auto-selection uses logger.debug."""
        import inspect

        from src.settings import _settings as settings_mod

        source = inspect.getsource(settings_mod.Settings.get_model_for_agent)
        # Should have debug for the successful auto-selection
        assert "logger.debug(" in source
        # The warning for no model fitting VRAM should still be warning
        assert "logger.warning(" in source


class TestPlateauWarningFix:
    """Tests for plateau warning: only warn when score actually drops, not on plateaus."""

    def test_worsened_warning_uses_score_comparison(self):
        """Test generic quality loop uses score comparison, not iteration number comparison."""
        import inspect

        from src.services.world_quality_service._quality_loop import quality_refinement_loop

        source = inspect.getsource(quality_refinement_loop)
        # Should use actual score comparison, not iteration number comparison
        assert "average_score < history.peak_score" in source, (
            "quality_loop: should compare scores, not iteration numbers"
        )
        # Should NOT use the old broken condition
        assert "best_iteration != len(history.iterations)" not in source, (
            "quality_loop: still uses old broken iteration number comparison"
        )


class TestRetryTemperature:
    """Tests for the temperature strategy feature in _common.py."""

    def test_retry_temperature_stable_strategy(self):
        """Stable strategy returns base temperature regardless of retries."""
        from src.services.world_quality_service._common import retry_temperature

        config = MagicMock()
        config.creator_temperature = 0.7
        result = retry_temperature(config, creation_retries=5, temperature_strategy="stable")
        assert result == 0.7

    def test_retry_temperature_escalating_strategy(self):
        """Escalating strategy increases temperature by 0.15 per retry."""
        from src.services.world_quality_service._common import retry_temperature

        config = MagicMock()
        config.creator_temperature = 0.7
        result = retry_temperature(config, creation_retries=2, temperature_strategy="escalating")
        assert result == pytest.approx(1.0)

    def test_retry_temperature_escalating_is_default(self):
        """Default strategy is escalating when no strategy is specified."""
        from src.services.world_quality_service._common import retry_temperature

        config = MagicMock()
        config.creator_temperature = 0.7
        result = retry_temperature(config, creation_retries=1)
        assert result == pytest.approx(0.85)


class TestMiniDescriptionCaching:
    """Tests for mini description caching (#13)."""

    def test_mini_description_batch_skips_cached(self, caplog, world_quality_service):
        """Batch generation skips entities with cached mini descriptions."""
        with patch.object(
            world_quality_service,
            "generate_mini_description",
            return_value="Generated description",
        ) as mock_gen:
            entities = [
                {
                    "name": "CachedEntity",
                    "type": "character",
                    "description": "Some description",
                    "attributes": {"mini_description": "Already summarized"},
                },
                {
                    "name": "UncachedEntity",
                    "type": "character",
                    "description": "Another description",
                    "attributes": {},
                },
            ]
            result = world_quality_service.generate_mini_descriptions_batch(entities)

        # Only the uncached entity should trigger a generate call
        assert mock_gen.call_count == 1
        assert result["CachedEntity"] == "Already summarized"
        assert result["UncachedEntity"] == "Generated description"


class TestBatchFailureRecovery:
    """Tests for batch shuffle recovery (#15)."""

    def test_batch_recovery_after_consecutive_failures(self):
        """Batch continues after 3 consecutive failures via recovery mechanism."""
        from src.services.world_quality_service._batch import _generate_batch
        from src.utils.exceptions import WorldGenerationError

        mock_svc = MagicMock()
        mock_svc._calculate_eta.return_value = None
        mock_svc.get_config.return_value.get_threshold.return_value = 7.5

        call_count = [0]

        def mock_generate_fn(i):
            """Fail 3 times, then succeed."""
            call_count[0] += 1
            if call_count[0] <= 3:
                raise WorldGenerationError(f"Generation failed (attempt {call_count[0]})")
            mock_scores = MagicMock()
            mock_scores.average = 8.0
            return {"name": f"Entity_{call_count[0]}"}, mock_scores, 1

        results = _generate_batch(
            svc=mock_svc,
            count=5,
            entity_type="test",
            generate_fn=mock_generate_fn,
            get_name=lambda e: e["name"],
        )

        # After 3 failures the batch recovers and the 4th call succeeds
        assert len(results) >= 1
        entity_names = [e["name"] for e, _ in results]
        assert any("Entity_" in name for name in entity_names)

    def test_batch_no_infinite_retry_after_recovery(self):
        """After one recovery, 3 more consecutive failures terminate the batch."""
        from src.services.world_quality_service._batch import _generate_batch
        from src.utils.exceptions import WorldGenerationError

        mock_svc = MagicMock()
        mock_svc._calculate_eta.return_value = None
        mock_svc.get_config.return_value.get_threshold.return_value = 7.5

        def always_fail(i):
            """Always raise WorldGenerationError."""
            raise WorldGenerationError("Persistent failure")

        # With enough count, the loop should terminate after recovery + second round of failures
        with pytest.raises(WorldGenerationError, match="Failed to generate any"):
            _generate_batch(
                svc=mock_svc,
                count=10,
                entity_type="test",
                generate_fn=always_fail,
                get_name=lambda e: e.get("name", "Unknown"),
            )


class _TestScores(BaseQualityScores):
    """Test quality scores for hail-mary tests."""

    dim_a: float = 0.0
    dim_b: float = 0.0
    feedback: str = ""

    @property
    def average(self) -> float:
        """Calculate average score across dimensions."""
        return (self.dim_a + self.dim_b) / 2

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {"dim_a": self.dim_a, "dim_b": self.dim_b, "feedback": self.feedback}

    def weak_dimensions(self, threshold: float = 7.0) -> list[str]:
        """List dimension names with scores below threshold."""
        weak = []
        if self.dim_a < threshold:
            weak.append("dim_a")
        if self.dim_b < threshold:
            weak.append("dim_b")
        return weak


class TestChapterRefinePromptDimensionInstructions:
    """Tests for dimension-specific instructions in chapter refinement prompts."""

    def test_chapter_refine_prompt_includes_dimension_instructions(
        self, world_quality_service, sample_story_state
    ):
        """Refinement prompt includes dimension-specific text when scores are below threshold."""
        from src.memory.story_state import Chapter
        from src.memory.world_quality import ChapterQualityScores as ChapterScores

        chapter = Chapter(number=1, title="The Beginning", outline="A hero leaves home.")
        scores = ChapterScores(
            purpose=4.0,
            pacing=3.0,
            hook=5.0,
            coherence=4.5,
            feedback="Needs work on all fronts.",
        )
        sample_story_state.plot_summary = "An epic quest to save the world."

        captured_prompts = []

        def capture_generate_structured(settings, model, prompt, response_model, temperature):
            """Capture the prompt sent to generate_structured."""
            captured_prompts.append(prompt)
            return Chapter(number=1, title="The Refined Beginning", outline="Improved outline.")

        with patch(
            "src.services.world_quality_service._chapter_quality.generate_structured",
            side_effect=capture_generate_structured,
        ):
            world_quality_service._refine_chapter_outline(
                chapter, scores, sample_story_state, temperature=0.7
            )

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]

        # All four dimensions are below the default threshold (7.0), so all should appear
        assert "PURPOSE: Clarify what this chapter accomplishes" in prompt
        assert "PACING: Vary the rhythm" in prompt
        assert "HOOK: Start with an immediate question or conflict" in prompt
        assert "COHERENCE: Ensure each scene leads logically" in prompt
        assert "Make SUBSTANTIAL changes to the outline" in prompt
        assert "DIMENSION-SPECIFIC INSTRUCTIONS:" in prompt

    def test_chapter_refine_prompt_only_includes_weak_dimensions(
        self, world_quality_service, sample_story_state
    ):
        """Refinement prompt only includes instructions for dimensions below threshold."""
        from src.memory.story_state import Chapter
        from src.memory.world_quality import ChapterQualityScores as ChapterScores

        chapter = Chapter(number=2, title="Rising Action", outline="Conflict intensifies.")
        # Only purpose and hook are below default threshold (7.0)
        scores = ChapterScores(
            purpose=5.0,
            pacing=8.0,
            hook=4.0,
            coherence=9.0,
            feedback="Purpose and hook need improvement.",
        )
        sample_story_state.plot_summary = "A mystery unfolds."

        captured_prompts = []

        def capture_generate_structured(settings, model, prompt, response_model, temperature):
            """Capture the prompt sent to generate_structured."""
            captured_prompts.append(prompt)
            return Chapter(number=2, title="Rising Action Refined", outline="Better outline.")

        with patch(
            "src.services.world_quality_service._chapter_quality.generate_structured",
            side_effect=capture_generate_structured,
        ):
            world_quality_service._refine_chapter_outline(
                chapter, scores, sample_story_state, temperature=0.7
            )

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]

        # Only purpose and hook should have instructions
        assert "PURPOSE: Clarify what this chapter accomplishes" in prompt
        assert "HOOK: Start with an immediate question or conflict" in prompt
        # Pacing and coherence scored above threshold, should NOT appear
        assert "PACING: Vary the rhythm" not in prompt
        assert "COHERENCE: Ensure each scene leads logically" not in prompt

    def test_chapter_refine_prompt_no_weak_dimensions(
        self, world_quality_service, sample_story_state
    ):
        """Refinement prompt shows minor improvements note when no dimensions are weak."""
        from src.memory.story_state import Chapter
        from src.memory.world_quality import ChapterQualityScores as ChapterScores

        chapter = Chapter(number=3, title="Climax", outline="The final battle.")
        # All dimensions above threshold
        scores = ChapterScores(
            purpose=8.0,
            pacing=8.5,
            hook=9.0,
            coherence=8.0,
            feedback="Good overall.",
        )
        sample_story_state.plot_summary = "The hero prevails."

        captured_prompts = []

        def capture_generate_structured(settings, model, prompt, response_model, temperature):
            """Capture the prompt sent to generate_structured."""
            captured_prompts.append(prompt)
            return Chapter(number=3, title="Climax Refined", outline="Polished outline.")

        with patch(
            "src.services.world_quality_service._chapter_quality.generate_structured",
            side_effect=capture_generate_structured,
        ):
            world_quality_service._refine_chapter_outline(
                chapter, scores, sample_story_state, temperature=0.7
            )

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]

        # No dimension-specific instructions, just the minor note
        assert "No critical weaknesses" in prompt
        assert "minor improvements" in prompt
        assert "PURPOSE: Clarify" not in prompt
        assert "PACING: Vary" not in prompt
        assert "HOOK: Start" not in prompt
        assert "COHERENCE: Ensure" not in prompt
        # No weak dims → no "SUBSTANTIAL changes" directive (contradictory with "minor")
        assert "Make SUBSTANTIAL changes" not in prompt
        assert "Polish the outline with minor improvements" in prompt

    def test_build_dimension_instructions_all_dimensions(self):
        """_build_dimension_instructions returns all four instructions when all are weak."""
        from src.services.world_quality_service._chapter_quality import (
            _build_dimension_instructions,
        )

        result = _build_dimension_instructions(["purpose", "pacing", "hook", "coherence"])
        assert "PURPOSE:" in result
        assert "PACING:" in result
        assert "HOOK:" in result
        assert "COHERENCE:" in result

    def test_build_dimension_instructions_empty_list(self):
        """_build_dimension_instructions returns minor improvements note for empty list."""
        from src.services.world_quality_service._chapter_quality import (
            _build_dimension_instructions,
        )

        result = _build_dimension_instructions([])
        assert "No critical weaknesses" in result
        assert "minor improvements" in result


class TestHailMaryFreshCreation:
    """Tests for hail-mary fresh creation in quality_refinement_loop (#7)."""

    def _make_loop_args(self, create_fn, judge_fn, max_iterations=2, threshold=9.0):
        """Build common keyword arguments for quality_refinement_loop.

        Args:
            create_fn: Entity creation callable.
            judge_fn: Entity judging callable.
            max_iterations: Maximum loop iterations.
            threshold: Quality threshold.

        Returns:
            Dict of keyword arguments for quality_refinement_loop.
        """
        from src.memory.world_quality import RefinementConfig

        config = RefinementConfig(
            max_iterations=max_iterations,
            quality_threshold=threshold,
            quality_thresholds={"test": threshold},
            creator_temperature=0.9,
            judge_temperature=0.1,
            refinement_temperature=0.7,
        )
        mock_svc = MagicMock()
        mock_svc._log_refinement_analytics = MagicMock()

        return {
            "entity_type": "test",
            "create_fn": create_fn,
            "judge_fn": judge_fn,
            "refine_fn": lambda entity, scores, iteration: entity,
            "get_name": lambda e: e.get("name", "Unknown") if isinstance(e, dict) else "Unknown",
            "serialize": lambda e: e if isinstance(e, dict) else {},
            "is_empty": lambda e: e is None or (isinstance(e, dict) and not e),
            "score_cls": _TestScores,
            "config": config,
            "svc": mock_svc,
            "story_id": "test-story",
        }

    def test_hail_mary_beats_best_score(self):
        """Hail-mary fresh creation replaces original when it scores higher."""
        from src.services.world_quality_service._quality_loop import quality_refinement_loop

        create_calls = [0]

        def create_fn(retries):
            """Return 'Original' first, then 'Fresh' for hail-mary."""
            create_calls[0] += 1
            if create_calls[0] == 1:
                return {"name": "Original", "data": "first"}
            return {"name": "Fresh", "data": "second"}

        def judge_fn(entity):
            """Score 'Original' at 6.0, 'Fresh' at 8.0."""
            if entity.get("name") == "Original":
                return _TestScores(dim_a=6.0, dim_b=6.0, feedback="OK")
            return _TestScores(dim_a=8.0, dim_b=8.0, feedback="Great")

        kwargs = self._make_loop_args(create_fn, judge_fn, max_iterations=2, threshold=9.0)
        entity, scores, _iterations = quality_refinement_loop(**kwargs)

        assert entity["name"] == "Fresh"
        assert scores.average == 8.0

    def test_hail_mary_does_not_beat(self):
        """Hail-mary is discarded when it scores lower than the original."""
        from src.services.world_quality_service._quality_loop import quality_refinement_loop

        create_calls = [0]

        def create_fn(retries):
            """Return 'Original' first, then 'Worse' for hail-mary."""
            create_calls[0] += 1
            if create_calls[0] == 1:
                return {"name": "Original", "data": "first"}
            return {"name": "Worse", "data": "second"}

        def judge_fn(entity):
            """Score 'Original' at 6.0, 'Worse' at 5.0."""
            if entity.get("name") == "Original":
                return _TestScores(dim_a=6.0, dim_b=6.0, feedback="OK")
            return _TestScores(dim_a=5.0, dim_b=5.0, feedback="Bad")

        kwargs = self._make_loop_args(create_fn, judge_fn, max_iterations=2, threshold=9.0)
        entity, scores, _iterations = quality_refinement_loop(**kwargs)

        assert entity["name"] == "Original"
        assert scores.average == 6.0

    def test_no_hail_mary_when_threshold_met(self):
        """No hail-mary attempted when entity meets quality threshold immediately."""
        from src.services.world_quality_service._quality_loop import quality_refinement_loop

        create_calls = [0]

        def create_fn(retries):
            """Return entity on first call; should not be called again."""
            create_calls[0] += 1
            return {"name": "GoodEntity", "data": "good"}

        def judge_fn(entity):
            """Score above threshold."""
            return _TestScores(dim_a=6.0, dim_b=6.0, feedback="Good enough")

        kwargs = self._make_loop_args(create_fn, judge_fn, max_iterations=3, threshold=5.0)
        entity, _scores, _iterations = quality_refinement_loop(**kwargs)

        assert entity["name"] == "GoodEntity"
        # Only one create call — no hail-mary needed
        assert create_calls[0] == 1

    def test_hail_mary_creation_fails_gracefully(self):
        """Original entity is returned when hail-mary creation raises an exception."""
        from src.services.world_quality_service._quality_loop import quality_refinement_loop
        from src.utils.exceptions import WorldGenerationError

        create_calls = [0]

        def create_fn(retries):
            """Return 'Original' first, then raise on hail-mary."""
            create_calls[0] += 1
            if create_calls[0] == 1:
                return {"name": "Original", "data": "first"}
            raise WorldGenerationError("Hail-mary creation exploded")

        def judge_fn(entity):
            """Score below threshold."""
            return _TestScores(dim_a=6.0, dim_b=6.0, feedback="OK")

        kwargs = self._make_loop_args(create_fn, judge_fn, max_iterations=2, threshold=9.0)
        entity, scores, _iterations = quality_refinement_loop(**kwargs)

        assert entity["name"] == "Original"
        assert scores.average == 6.0

    def test_hail_mary_returns_empty_entity(self):
        """Hail-mary create_fn returning empty entity keeps original."""
        from src.services.world_quality_service._quality_loop import quality_refinement_loop

        create_calls = [0]

        def create_fn(retries):
            """Return 'Original' first, then empty dict for hail-mary."""
            create_calls[0] += 1
            if create_calls[0] == 1:
                return {"name": "Original", "data": "first"}
            return {}

        def judge_fn(entity):
            """Score below threshold."""
            return _TestScores(dim_a=6.0, dim_b=6.0, feedback="OK")

        kwargs = self._make_loop_args(create_fn, judge_fn, max_iterations=2, threshold=9.0)
        entity, scores, _iterations = quality_refinement_loop(**kwargs)

        assert entity["name"] == "Original"
        assert scores.average == 6.0


class TestEntityFrequencyHint:
    """Tests for _compute_entity_frequency_hint focal-character bias reduction."""

    def test_entity_frequency_hint_empty_for_few_entities(self):
        """Returns empty hint when fewer than 3 entities or no rels."""
        # Fewer than 3 entities
        hint, _freq = _compute_entity_frequency_hint(["Alice", "Bob"], [("Alice", "Bob", "knows")])
        assert hint == ""
        # 3 entities but no relationships
        hint, _freq = _compute_entity_frequency_hint(["Alice", "Bob", "Carol"], [])
        assert hint == ""
        # Single entity
        hint, _freq = _compute_entity_frequency_hint(["Alice"], [("Alice", "Bob", "knows")])
        assert hint == ""

    def test_entity_frequency_hint_highlights_under_connected(self):
        """Entities with 0-1 connections are listed as PRIORITY."""
        entities = ["Alice", "Bob", "Carol", "Dave"]
        # Alice has 2 connections, Bob has 1, Carol has 1, Dave has 0
        existing_rels = [
            ("Alice", "Bob", "knows"),
            ("Alice", "Carol", "rivals"),
        ]
        hint, _freq = _compute_entity_frequency_hint(entities, existing_rels)
        assert "PRIORITY" in hint
        # Bob has 1, Carol has 1, Dave has 0 — all under-connected
        assert "Bob" in hint
        assert "Carol" in hint
        assert "Dave" in hint

    def test_entity_frequency_hint_warns_over_connected(self):
        """Entities with 4+ connections are listed as AVOID."""
        entities = ["Alice", "Bob", "Carol", "Dave", "Eve"]
        # Alice appears in 4 relationships — over-connected
        existing_rels = [
            ("Alice", "Bob", "knows"),
            ("Alice", "Carol", "rivals"),
            ("Alice", "Dave", "mentors"),
            ("Alice", "Eve", "betrays"),
        ]
        hint, _freq = _compute_entity_frequency_hint(entities, existing_rels)
        assert "AVOID" in hint
        assert "Alice" in hint

    def test_entity_frequency_hint_balanced_returns_empty(self):
        """Returns empty hint when all entities have 2-3 connections (balanced)."""
        entities = ["Alice", "Bob", "Carol"]
        # Each entity has exactly 2 connections
        existing_rels = [
            ("Alice", "Bob", "knows"),
            ("Bob", "Carol", "rivals"),
            ("Carol", "Alice", "mentors"),
        ]
        hint, _freq = _compute_entity_frequency_hint(entities, existing_rels)
        assert hint == ""

    def test_entity_frequency_hint_ignores_unknown_entities(self):
        """Entities in rels but not in entity_names are not counted."""
        entities = ["Alice", "Bob", "Carol"]
        # "Zoe" is in rels but not in entity_names
        existing_rels = [
            ("Alice", "Zoe", "knows"),
            ("Zoe", "Bob", "rivals"),
        ]
        hint, _freq = _compute_entity_frequency_hint(entities, existing_rels)
        # Alice and Bob have 1 connection each, Carol has 0 — all under-connected
        assert "PRIORITY" in hint
        assert "Carol" in hint


class TestUnusedPairsSortedByFrequency:
    """Tests for unused pairs sorting by connection frequency in _create_relationship."""

    def test_unused_pairs_sorted_by_frequency(self):
        """Verify unused pairs are sorted so under-connected entities appear first."""
        from collections import Counter
        from itertools import permutations

        entity_names = ["Alice", "Bob", "Carol", "Dave", "Eve"]
        # Alice: 4 connections (over-connected), Bob: 3, Carol: 1, Dave: 0, Eve: 0
        existing_rels = [
            ("Alice", "Bob", "knows"),
            ("Alice", "Carol", "rivals"),
            ("Alice", "Bob", "mentors"),
            ("Alice", "Bob", "betrays"),
        ]

        existing_pair_set = set()
        for s, t, _rt in existing_rels:
            existing_pair_set.add((s, t))
            existing_pair_set.add((t, s))

        entity_freq: Counter[str] = Counter()
        for s, t, _rt in existing_rels:
            if s in entity_names:
                entity_freq[s] += 1
            if t in entity_names:
                entity_freq[t] += 1

        raw_unused = [
            (a, b) for a, b in permutations(entity_names, 2) if (a, b) not in existing_pair_set
        ]
        raw_unused.sort(key=lambda pair: entity_freq[pair[0]] + entity_freq[pair[1]])

        # Dave (0) and Eve (0) have the lowest frequency — pairs between them come first
        first_pair = raw_unused[0]
        assert first_pair[0] in ("Dave", "Eve") and first_pair[1] in ("Dave", "Eve"), (
            f"Expected pair of under-connected entities first, got {first_pair}"
        )

        # Last pairs should involve the most connected entity (Alice=4)
        last_pair = raw_unused[-1]
        assert "Alice" in last_pair, f"Expected Alice (4 connections) in last pair, got {last_pair}"

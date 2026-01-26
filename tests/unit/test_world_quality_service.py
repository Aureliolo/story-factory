"""Tests for WorldQualityService - cancellation, progress, and logging."""

import logging
import time
from unittest.mock import MagicMock, patch

import pytest

from src.memory.world_quality import CharacterQualityScores
from src.services.model_mode_service import ModelModeService
from src.services.world_quality_service import (
    EntityGenerationProgress,
    WorldQualityService,
)


@pytest.fixture
def mock_mode_service(tmp_path):
    """Create a mock ModelModeService."""
    mode_service = MagicMock(spec=ModelModeService)
    mode_service.get_model_for_agent.return_value = "huihui_ai/dolphin3-abliterated:8b"
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
            r for r in caplog.records if "Completed mini description generation" in r.message
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
                    "relationship_type": "knows",
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

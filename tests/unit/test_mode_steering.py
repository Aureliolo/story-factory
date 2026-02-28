"""Tests for mode steering and size preference functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.memory.mode_models import (
    PRESET_MODES,
    GenerationMode,
    ModelSizeTier,
    SizePreference,
    get_size_tier,
)
from src.services.model_mode_service import ModelModeService
from src.settings import Settings


class TestSizePreference:
    """Tests for the SizePreference enum."""

    def test_size_preference_values(self):
        """Verify SizePreference enum has expected values."""
        assert SizePreference.LARGEST.value == "largest"
        assert SizePreference.MEDIUM.value == "medium"
        assert SizePreference.SMALLEST.value == "smallest"

    def test_size_preference_from_string(self):
        """Verify SizePreference can be created from string."""
        assert SizePreference("largest") == SizePreference.LARGEST
        assert SizePreference("medium") == SizePreference.MEDIUM
        assert SizePreference("smallest") == SizePreference.SMALLEST


class TestPresetModes:
    """Tests for PRESET_MODES with size preferences."""

    def test_quality_max_has_largest_preference(self):
        """Quality max mode should prefer largest models."""
        mode = PRESET_MODES["quality_max"]
        assert mode.size_preference == SizePreference.LARGEST.value

    def test_quality_creative_has_largest_preference(self):
        """Quality creative mode should prefer largest models."""
        mode = PRESET_MODES["quality_creative"]
        assert mode.size_preference == SizePreference.LARGEST.value

    def test_balanced_has_medium_preference(self):
        """Balanced mode should prefer medium models."""
        mode = PRESET_MODES["balanced"]
        assert mode.size_preference == SizePreference.MEDIUM.value

    def test_draft_fast_has_smallest_preference(self):
        """Draft fast mode should prefer smallest models."""
        mode = PRESET_MODES["draft_fast"]
        assert mode.size_preference == SizePreference.SMALLEST.value

    def test_experimental_has_medium_preference(self):
        """Experimental mode should prefer medium models."""
        mode = PRESET_MODES["experimental"]
        assert mode.size_preference == SizePreference.MEDIUM.value

    def test_preset_modes_have_empty_temperatures(self):
        """All preset modes should have empty agent_temperatures."""
        for mode_id, mode in PRESET_MODES.items():
            assert mode.agent_temperatures == {}, (
                f"Mode {mode_id} should have empty agent_temperatures, "
                f"got {mode.agent_temperatures}"
            )


class TestGenerationModeModel:
    """Tests for the GenerationMode Pydantic model."""

    def test_default_size_preference_is_medium(self):
        """Default size preference should be medium."""
        mode = GenerationMode(
            id="test",
            name="Test Mode",
        )
        assert mode.size_preference == SizePreference.MEDIUM.value

    def test_can_set_size_preference(self):
        """Size preference can be set explicitly."""
        mode = GenerationMode(
            id="test",
            name="Test Mode",
            size_preference=SizePreference.LARGEST,
        )
        assert mode.size_preference == SizePreference.LARGEST.value

    def test_agent_models_default_empty(self):
        """Agent models should default to empty dict."""
        mode = GenerationMode(
            id="test",
            name="Test Mode",
        )
        assert mode.agent_models == {}

    def test_agent_temperatures_default_empty(self):
        """Agent temperatures should default to empty dict."""
        mode = GenerationMode(
            id="test",
            name="Test Mode",
        )
        assert mode.agent_temperatures == {}


class TestGetSizeTier:
    """Tests for the get_size_tier function."""

    def test_large_model(self):
        """Models >= 20GB should be classified as large."""
        assert get_size_tier(20.0) == ModelSizeTier.LARGE
        assert get_size_tier(40.0) == ModelSizeTier.LARGE

    def test_medium_model(self):
        """Models >= 8GB and < 20GB should be classified as medium."""
        assert get_size_tier(8.0) == ModelSizeTier.MEDIUM
        assert get_size_tier(15.0) == ModelSizeTier.MEDIUM
        assert get_size_tier(19.9) == ModelSizeTier.MEDIUM

    def test_small_model(self):
        """Models >= 3GB and < 8GB should be classified as small."""
        assert get_size_tier(3.0) == ModelSizeTier.SMALL
        assert get_size_tier(5.0) == ModelSizeTier.SMALL
        assert get_size_tier(7.9) == ModelSizeTier.SMALL

    def test_tiny_model(self):
        """Models < 3GB should be classified as tiny."""
        assert get_size_tier(0.5) == ModelSizeTier.TINY
        assert get_size_tier(2.9) == ModelSizeTier.TINY


class TestModelModeServiceTierScore:
    """Tests for tier score calculation in ModelModeService."""

    @pytest.fixture
    def service(self) -> ModelModeService:
        """
        Create a ModelModeService configured with default Settings and a patched ModeDatabase.

        Returns:
            service (ModelModeService): Instance initialized with a default Settings object and ModeDatabase patched to prevent real database access.
        """
        settings = Settings()
        with patch("src.services.model_mode_service.ModeDatabase"):
            return ModelModeService(settings)

    def test_tier_score_largest_prefers_large(self, service: ModelModeService):
        """LARGEST preference should score large models highest."""
        large_score = service._calculate_tier_score(25.0, SizePreference.LARGEST)
        medium_score = service._calculate_tier_score(10.0, SizePreference.LARGEST)
        small_score = service._calculate_tier_score(5.0, SizePreference.LARGEST)
        tiny_score = service._calculate_tier_score(1.0, SizePreference.LARGEST)

        assert large_score > medium_score > small_score > tiny_score

    def test_tier_score_smallest_prefers_tiny(self, service: ModelModeService):
        """SMALLEST preference should score tiny models highest."""
        large_score = service._calculate_tier_score(25.0, SizePreference.SMALLEST)
        medium_score = service._calculate_tier_score(10.0, SizePreference.SMALLEST)
        small_score = service._calculate_tier_score(5.0, SizePreference.SMALLEST)
        tiny_score = service._calculate_tier_score(1.0, SizePreference.SMALLEST)

        assert tiny_score > small_score > medium_score > large_score

    def test_tier_score_medium_prefers_medium(self, service: ModelModeService):
        """MEDIUM preference should score medium models highest."""
        large_score = service._calculate_tier_score(25.0, SizePreference.MEDIUM)
        medium_score = service._calculate_tier_score(10.0, SizePreference.MEDIUM)
        small_score = service._calculate_tier_score(5.0, SizePreference.MEDIUM)
        tiny_score = service._calculate_tier_score(1.0, SizePreference.MEDIUM)

        assert medium_score > small_score > large_score > tiny_score


class TestModelModeServiceModelSelection:
    """Tests for model selection with size preference."""

    @pytest.fixture
    def service(self) -> ModelModeService:
        """
        Create a ModelModeService configured with default Settings and a patched ModeDatabase.

        Returns:
            service (ModelModeService): Instance initialized with a default Settings object and ModeDatabase patched to prevent real database access.
        """
        settings = Settings()
        with patch("src.services.model_mode_service.ModeDatabase"):
            return ModelModeService(settings)

    @patch("src.settings.get_installed_models_with_sizes")
    def test_select_largest_for_quality_max(
        self, mock_get_models: MagicMock, service: ModelModeService
    ):
        """LARGEST preference should select largest available model."""
        mock_get_models.return_value = {
            "fake-small:8b": 3.0,
            "fake-medium:14b": 9.0,
            "fake-large:32b": 20.0,
        }
        service.settings.custom_model_tags = {
            "fake-small:8b": ["writer"],
            "fake-medium:14b": ["writer"],
            "fake-large:32b": ["writer"],
        }

        result = service._select_model_with_size_preference("writer", SizePreference.LARGEST, 48)

        assert result == "fake-large:32b"

    @patch("src.settings.get_installed_models_with_sizes")
    def test_select_smallest_for_draft_fast(
        self, mock_get_models: MagicMock, service: ModelModeService
    ):
        """SMALLEST preference should select smallest available model."""
        mock_get_models.return_value = {
            "qwen2.5:1.5b": 1.0,
            "huihui_ai/dolphin3-abliterated:8b": 3.0,
            "qwen2.5:32b": 20.0,
        }
        service.settings.custom_model_tags = {
            "qwen2.5:1.5b": ["writer"],
            "huihui_ai/dolphin3-abliterated:8b": ["writer"],
            "qwen2.5:32b": ["writer"],
        }

        result = service._select_model_with_size_preference("writer", SizePreference.SMALLEST, 48)

        assert result == "qwen2.5:1.5b"

    @patch("src.settings.get_installed_models_with_sizes")
    def test_respects_vram_constraint(self, mock_get_models: MagicMock, service: ModelModeService):
        """Should prefer models that fit in VRAM even with LARGEST preference."""
        mock_get_models.return_value = {
            "huihui_ai/dolphin3-abliterated:8b": 3.0,  # Fits in 8GB
            "qwen2.5:32b": 20.0,  # Doesn't fit
        }
        service.settings.custom_model_tags = {
            "huihui_ai/dolphin3-abliterated:8b": ["writer"],
            "qwen2.5:32b": ["writer"],
        }

        result = service._select_model_with_size_preference(
            "writer",
            SizePreference.LARGEST,
            8,  # Only 8GB VRAM
        )

        # Should select small model because large doesn't fit
        assert result == "huihui_ai/dolphin3-abliterated:8b"

    @patch("src.settings.get_installed_models_with_sizes")
    def test_excludes_model_below_80_percent_gpu_residency(
        self, mock_get_models: MagicMock, service: ModelModeService
    ):
        """Models that can't achieve 80% GPU residency should be excluded."""
        mock_get_models.return_value = {
            "small-model:8b": 5.0,  # 24 GiB = 25.8 GB → 25.8/5 = 515% — fits easily
            "huge-model:70b": 43.0,  # 24 GiB = 25.8 GB → 25.8/43 = 60% — below 80%
        }
        service.settings.custom_model_tags = {
            "small-model:8b": ["writer"],
            "huge-model:70b": ["writer"],
        }

        result = service._select_model_with_size_preference(
            "writer",
            SizePreference.LARGEST,
            24,  # 24GB GPU
        )

        # 70B model excluded by 80% rule, only small model remains
        assert result == "small-model:8b"

    @patch("src.settings.get_installed_models_with_sizes")
    def test_allows_model_above_80_percent_gpu_residency(
        self, mock_get_models: MagicMock, service: ModelModeService
    ):
        """Models above 80% GPU residency should be allowed."""
        mock_get_models.return_value = {
            "borderline-model:30b": 30.0,  # 24 GiB = 25.8 GB → 25.8/30 = 86% — above threshold
        }
        service.settings.custom_model_tags = {
            "borderline-model:30b": ["writer"],
        }

        result = service._select_model_with_size_preference(
            "writer",
            SizePreference.LARGEST,
            24,  # 24GB GPU
        )

        assert result == "borderline-model:30b"

    @patch("src.settings.get_installed_models_with_sizes")
    def test_gpu_residency_excludes_all_raises_no_tagged(
        self, mock_get_models: MagicMock, service: ModelModeService
    ):
        """When all tagged models fail GPU residency, should raise ValueError."""
        mock_get_models.return_value = {
            "huge-model:70b": 43.0,  # 8 GiB = 8.6 GB → 8.6/43 = 20% — way below threshold
        }
        service.settings.custom_model_tags = {
            "huge-model:70b": ["writer"],
        }

        with pytest.raises(ValueError, match="excluded by GPU residency"):
            service._select_model_with_size_preference(
                "writer",
                SizePreference.LARGEST,
                8,  # 8GB GPU
            )

    @patch("src.settings.get_installed_models_with_sizes")
    def test_raises_when_no_tagged_models(
        self, mock_get_models: MagicMock, service: ModelModeService
    ):
        """Should raise ValueError when no models are tagged for the role."""
        mock_get_models.return_value = {
            "qwen2.5:3b": 3.0,
        }
        service.settings.custom_model_tags = {}

        with pytest.raises(ValueError, match="No model tagged for role"):
            service._select_model_with_size_preference("writer", SizePreference.MEDIUM, 48)

    @patch("src.settings.get_installed_models_with_sizes")
    def test_returns_default_when_no_models_installed(
        self, mock_get_models: MagicMock, service: ModelModeService
    ):
        """Should return default model from RECOMMENDED_MODELS when no models installed."""
        mock_get_models.return_value = {}

        result = service._select_model_with_size_preference("writer", SizePreference.MEDIUM, 48)

        # Should return the first recommended model
        assert result is not None
        assert isinstance(result, str)


class TestModelModeServiceAdditionalCoverage:
    """Additional tests for full coverage."""

    @pytest.fixture
    def temp_db(self, tmp_path: Path) -> Path:
        """
        Create a Path for a temporary test database file named "test_mode.db".

        Returns:
            Path: Path to the temporary database file within the provided `tmp_path`.
        """
        return tmp_path / "test_mode.db"

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create a MagicMock of Settings preconfigured for tests.

        The mock exposes:
        - ollama_url set to "http://localhost:11434".
        - get_model_for_agent() returning "huihui_ai/dolphin3-abliterated:8b".
        - agent_temperatures mapping {"writer": 0.9, "editor": 0.6}.
        - get_temperature_for_agent(role) returning the temperature for known roles.

        Returns:
            MagicMock: A mock object conforming to the Settings spec.
        """
        mock = MagicMock(spec=Settings)
        mock.ollama_url = "http://localhost:11434"
        mock.get_model_for_agent.return_value = "huihui_ai/dolphin3-abliterated:8b"
        mock.agent_temperatures = {"writer": 0.9, "editor": 0.6}

        def _get_temp(role: str) -> float:
            """
            Retrieve the temperature value for a given agent role from the mock configuration.

            Parameters:
                role (str): Agent role name to look up in mock.agent_temperatures.

            Returns:
                float: Temperature value associated with the specified role.

            Raises:
                ValueError: If the provided role is not present in mock.agent_temperatures.
            """
            if role not in mock.agent_temperatures:
                raise ValueError(f"Unknown agent role: {role}")
            return float(mock.agent_temperatures[role])

        mock.get_temperature_for_agent.side_effect = _get_temp
        return mock

    @pytest.fixture
    def service(self, mock_settings: MagicMock, temp_db: Path) -> ModelModeService:
        """
        Create a ModelModeService configured with mocked settings and a temporary database path.

        Parameters:
            mock_settings (MagicMock): Mocked Settings object used to configure the service.
            temp_db (Path): Filesystem path to a temporary database used by the service.

        Returns:
            ModelModeService: An instance of ModelModeService initialized with the provided mocks.
        """
        return ModelModeService(mock_settings, db_path=temp_db)

    def test_set_mode_invalid_size_preference_raises_error(
        self, service: ModelModeService, temp_db: Path
    ):
        """Test that invalid size_preference raises ValueError."""
        import sqlite3

        from src.memory.mode_models import GenerationMode, VramStrategy

        # Save a custom mode first
        custom = GenerationMode(
            id="test_invalid_pref",
            name="Test",
            description="",
            agent_models={},
            agent_temperatures={},
            vram_strategy=VramStrategy.ADAPTIVE,
            is_preset=False,
        )
        service.save_custom_mode(custom)

        # Manually corrupt the size_preference value
        conn = sqlite3.connect(temp_db)
        try:
            conn.execute(
                "UPDATE custom_modes SET size_preference = 'invalid_pref' WHERE id = ?",
                ("test_invalid_pref",),
            )
            conn.commit()
        finally:
            conn.close()

        # Load the mode - should raise error for invalid size_preference
        with pytest.raises(ValueError, match="Invalid size_preference 'invalid_pref'"):
            service.set_mode("test_invalid_pref")

    def test_list_modes_invalid_size_preference_raises_error(
        self, service: ModelModeService, temp_db: Path
    ):
        """Test that list_modes raises error for invalid size_preference."""
        import sqlite3

        from src.memory.mode_models import GenerationMode, VramStrategy

        # Save a custom mode first
        custom = GenerationMode(
            id="test_list_pref",
            name="Test List",
            description="",
            agent_models={},
            agent_temperatures={},
            vram_strategy=VramStrategy.ADAPTIVE,
            is_preset=False,
        )
        service.save_custom_mode(custom)

        # Manually corrupt the size_preference value
        conn = sqlite3.connect(temp_db)
        try:
            conn.execute(
                "UPDATE custom_modes SET size_preference = 'bad_value' WHERE id = ?",
                ("test_list_pref",),
            )
            conn.commit()
        finally:
            conn.close()

        # list_modes should raise error for invalid size_preference
        with pytest.raises(ValueError, match="Invalid size_preference 'bad_value'"):
            service.list_modes()

    def test_set_mode_missing_size_preference_raises_error(
        self, service: ModelModeService, temp_db: Path
    ):
        """Test that missing size_preference raises ValueError."""
        from unittest.mock import patch

        # Mock the database to return a custom mode with missing size_preference
        mock_custom = {
            "id": "test_missing_pref",
            "name": "Test",
            "description": "",
            "agent_models": "{}",
            "agent_temperatures": "{}",
            "vram_strategy": "adaptive",
            "is_experimental": False,
            # size_preference is missing (None)
        }

        with patch.object(service._db, "get_custom_mode", return_value=mock_custom):
            # Load the mode - should raise error for missing size_preference
            with pytest.raises(ValueError, match="Missing size_preference in custom mode"):
                service.set_mode("test_missing_pref")

    def test_list_modes_missing_size_preference_raises_error(
        self, service: ModelModeService, temp_db: Path
    ):
        """Test that list_modes raises error for missing size_preference."""
        from unittest.mock import patch

        # Mock the database to return a custom mode with missing size_preference
        mock_custom = {
            "id": "test_list_missing",
            "name": "Test List",
            "description": "",
            "agent_models": {},
            "agent_temperatures": {},
            "vram_strategy": "adaptive",
            "is_experimental": False,
            # size_preference is missing (None)
        }

        with patch.object(service._db, "list_custom_modes", return_value=[mock_custom]):
            # list_modes should raise error for missing size_preference
            with pytest.raises(ValueError, match="Missing size_preference in custom mode"):
                service.list_modes()

    def test_get_temperature_from_mode(self, service: ModelModeService):
        """Test getting temperature when mode has explicit value."""
        from src.memory.mode_models import GenerationMode, VramStrategy

        custom = GenerationMode(
            id="explicit_temp",
            name="Explicit Temp",
            description="",
            agent_models={},
            agent_temperatures={"writer": 0.95},  # Explicit temperature
            vram_strategy=VramStrategy.ADAPTIVE,
            is_preset=False,
        )
        service.save_custom_mode(custom)
        service.set_mode("explicit_temp")

        temp = service.get_temperature_for_agent("writer")
        assert temp == 0.95

    def test_on_regenerate_marks_score(self, service: ModelModeService):
        """Test on_regenerate marks the score as regenerated."""
        # Record a generation first
        score_id = service.record_generation(
            project_id="test-proj",
            agent_role="writer",
            model_id="huihui_ai/dolphin3-abliterated:8b",
            chapter_id="ch-1",
        )
        assert score_id > 0

        # Now simulate regeneration
        service.on_regenerate("test-proj", "ch-1")

        # Verify the score was marked as regenerated
        score = service._db.get_latest_score_for_chapter("test-proj", "ch-1")
        assert score is not None
        assert score["was_regenerated"] == 1

    def test_on_regenerate_no_matching_score(self, service: ModelModeService):
        """Test on_regenerate handles no matching score gracefully."""
        # Try to regenerate for a non-existent project/chapter
        service.on_regenerate("nonexistent-proj", "ch-99")
        # Should not raise, just log

    def test_on_regenerate_handles_exception(self, service: ModelModeService):
        """Test on_regenerate handles exceptions gracefully."""
        with patch.object(
            service._db, "get_latest_score_for_chapter", side_effect=Exception("DB error")
        ):
            # Should not raise, just log warning
            service.on_regenerate("test-proj", "ch-1")


class TestPendingRecommendations:
    """Tests for pending recommendations and dismissal."""

    @pytest.fixture
    def temp_db(self, tmp_path: Path) -> Path:
        """Create a temporary database file using pytest's tmp_path fixture."""
        return tmp_path / "test_rec.db"

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create a MagicMock of Settings preconfigured for tests.

        The mock exposes:
        - ollama_url set to "http://localhost:11434".
        - get_model_for_agent() returning "huihui_ai/dolphin3-abliterated:8b".
        - agent_temperatures mapping {"writer": 0.9, "editor": 0.6}.
        - get_temperature_for_agent(role) returning the temperature for known roles.

        Returns:
            MagicMock: A mock object conforming to the Settings spec.
        """
        mock = MagicMock(spec=Settings)
        mock.ollama_url = "http://localhost:11434"
        mock.get_model_for_agent.return_value = "huihui_ai/dolphin3-abliterated:8b"
        mock.agent_temperatures = {"writer": 0.9, "editor": 0.6}

        def _get_temp(role: str) -> float:
            """
            Retrieve the temperature value for a given agent role from the mock configuration.

            Parameters:
                role (str): Agent role name to look up in mock.agent_temperatures.

            Returns:
                float: Temperature value associated with the specified role.

            Raises:
                ValueError: If the provided role is not present in mock.agent_temperatures.
            """
            if role not in mock.agent_temperatures:
                raise ValueError(f"Unknown agent role: {role}")
            return float(mock.agent_temperatures[role])

        mock.get_temperature_for_agent.side_effect = _get_temp
        return mock

    @pytest.fixture
    def service(self, mock_settings: MagicMock, temp_db: Path) -> ModelModeService:
        """
        Create a ModelModeService configured with mocked settings and a temporary database path.

        Parameters:
            mock_settings (MagicMock): Mocked Settings object used to configure the service.
            temp_db (Path): Filesystem path to a temporary database used by the service.

        Returns:
            ModelModeService: An instance of ModelModeService initialized with the provided mocks.
        """
        return ModelModeService(mock_settings, db_path=temp_db)

    def test_get_pending_recommendations_empty(self, service: ModelModeService):
        """Should return empty list when no recommendations exist."""
        recs = service.get_pending_recommendations()
        assert recs == []

    def test_get_pending_recommendations_converts_to_objects(self, service: ModelModeService):
        """Should convert database rows to TuningRecommendation objects."""
        from src.memory.mode_models import RecommendationType

        # Insert a recommendation directly into the database
        service._db.record_recommendation(
            recommendation_type="model_swap",
            current_value="fake-current:8b",
            suggested_value="fake-writer:14b",
            affected_role="writer",
            reason="Better quality observed",
            confidence=0.85,
            evidence={"avg_score": 4.2},
            expected_improvement="15% quality increase",
        )

        recs = service.get_pending_recommendations()
        assert len(recs) == 1
        rec = recs[0]
        assert rec.recommendation_type == RecommendationType.MODEL_SWAP
        assert rec.current_value == "fake-current:8b"
        assert rec.suggested_value == "fake-writer:14b"
        assert rec.affected_role == "writer"
        assert rec.confidence == 0.85

    def test_get_pending_recommendations_handles_parse_error(self, service: ModelModeService):
        """Should handle parsing errors gracefully and log warning."""
        # Row with valid required fields but invalid recommendation_type that causes exception
        row = {
            "id": 99,
            "timestamp": "2026-01-01T12:00:00",
            "recommendation_type": "INVALID_TYPE_THAT_DOESNT_EXIST",
            "current_value": "old",
            "suggested_value": "new",
            "reason": "Test",
            "confidence": 0.8,
        }
        with patch.object(service._db, "get_pending_recommendations", return_value=[row]):
            recs = service.get_pending_recommendations()
            assert recs == []  # Parsing fails, row is skipped

    def test_get_pending_recommendations_handles_invalid_evidence_json(
        self, service: ModelModeService
    ):
        """
        Verifies that get_pending_recommendations converts rows with invalid JSON in `evidence_json` into recommendations with `evidence` set to None.

        Patches the database to return a single row whose `evidence_json` is not valid JSON and asserts the service still returns one recommendation object with `evidence is None`.
        """
        from datetime import datetime

        # Mock a row with invalid JSON in evidence_json
        invalid_row = {
            "id": 99,
            "timestamp": datetime.now().isoformat(),
            "recommendation_type": "model_swap",
            "current_value": "old-model",
            "suggested_value": "new-model",
            "affected_role": "writer",
            "reason": "Test",
            "confidence": 0.8,
            "evidence_json": "not-valid-json{",  # Invalid JSON
            "expected_improvement": None,
        }
        with patch.object(service._db, "get_pending_recommendations", return_value=[invalid_row]):
            recs = service.get_pending_recommendations()
            # Should still parse the recommendation, just with None evidence
            assert len(recs) == 1
            assert recs[0].evidence is None

    def test_dismiss_recommendation(self, service: ModelModeService):
        """Should persist dismissal to database."""
        # Insert a recommendation
        service._db.record_recommendation(
            recommendation_type="temp_adjust",
            current_value="0.7",
            suggested_value="0.9",
            affected_role="writer",
            reason="Test reason",
            confidence=0.8,
        )

        # Get the recommendation
        recs = service.get_pending_recommendations()
        assert len(recs) == 1

        # Dismiss it
        service.dismiss_recommendation(recs[0])

        # Should no longer appear in pending
        remaining = service.get_pending_recommendations()
        assert len(remaining) == 0

    def test_dismiss_recommendation_without_id(self, service: ModelModeService):
        """Should handle recommendation without ID gracefully."""
        from src.memory.mode_models import RecommendationType, TuningRecommendation

        rec = TuningRecommendation(
            id=None,  # No ID
            recommendation_type=RecommendationType.MODEL_SWAP,
            current_value="a",
            suggested_value="b",
            reason="test",
            confidence=0.5,
        )

        # Should not raise, just log warning
        service.dismiss_recommendation(rec)

    def test_get_pending_recommendations_missing_timestamp(self, service: ModelModeService):
        """Should use datetime.now() when timestamp is missing."""
        from datetime import datetime

        row = {
            "id": 1,
            "timestamp": None,  # Missing timestamp
            "recommendation_type": "model_swap",
            "current_value": "old",
            "suggested_value": "new",
            "affected_role": "writer",
            "reason": "Test",
            "confidence": 0.8,
        }
        with patch.object(service._db, "get_pending_recommendations", return_value=[row]):
            before = datetime.now()
            recs = service.get_pending_recommendations()
            after = datetime.now()
            assert len(recs) == 1
            assert before <= recs[0].timestamp <= after

    def test_get_pending_recommendations_missing_required_fields(self, service: ModelModeService):
        """Should skip recommendations with missing required fields."""
        row = {
            "id": 1,
            "timestamp": "2026-01-01T12:00:00",
            "recommendation_type": "model_swap",
            "current_value": None,  # Missing required field
            "suggested_value": "new",
            "reason": "Test",
            "confidence": 0.8,
        }
        with patch.object(service._db, "get_pending_recommendations", return_value=[row]):
            recs = service.get_pending_recommendations()
            assert recs == []  # Skipped due to missing field


class TestGetModelForAgentValidation:
    """Tests for get_model_for_agent with invalid mode configuration."""

    @pytest.fixture
    def service(self, tmp_path: Path) -> ModelModeService:
        """Create a ModelModeService with real database."""
        db_path = tmp_path / "test.db"
        settings = Settings()
        with patch("src.services.model_mode_service.ModeDatabase") as mock_db_class:
            mock_db = MagicMock()
            mock_db_class.return_value = mock_db
            return ModelModeService(settings, db_path=db_path)

    def test_invalid_size_preference_raises_error(self, service: ModelModeService):
        """Should raise ValueError for invalid size_preference in mode."""
        # Create a mock mode object with invalid size_preference
        invalid_mode = MagicMock()
        invalid_mode.id = "invalid_mode"
        invalid_mode.agent_models = {}  # No explicit model assignment
        invalid_mode.size_preference = "not_a_valid_preference"

        with patch.object(service, "get_current_mode", return_value=invalid_mode):
            with pytest.raises(ValueError, match="Invalid size_preference"):
                service.get_model_for_agent("writer")


class TestVramStrategySetting:
    """Tests for VRAM strategy in Settings."""

    def test_default_vram_strategy(self):
        """Default VRAM strategy should be adaptive."""
        settings = Settings()
        assert settings.vram_strategy == "adaptive"

    def test_vram_strategy_can_be_set(self):
        """VRAM strategy can be changed."""
        settings = Settings()
        settings.vram_strategy = "sequential"
        assert settings.vram_strategy == "sequential"
        settings.vram_strategy = "parallel"
        assert settings.vram_strategy == "parallel"

    def test_invalid_vram_strategy_raises(self):
        """Invalid VRAM strategy should raise ValueError."""
        settings = Settings()
        settings.vram_strategy = "invalid_strategy"
        with pytest.raises(ValueError, match="vram_strategy must be one of"):
            settings.validate()


class TestVramStrategyIntegration:
    """Tests for VRAM strategy integration between modes and settings."""

    @pytest.fixture
    def temp_db(self, tmp_path: Path) -> Path:
        """
        Create a temporary database path for tests.

        Parameters:
                tmp_path (Path): Base temporary directory provided by pytest.

        Returns:
                db_path (Path): Path to a temporary database file named "test_vram.db" inside `tmp_path`.
        """
        return tmp_path / "test_vram.db"

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create a Settings-like MagicMock preconfigured for tests.

        The mock uses Settings as its spec and includes:
        - ollama_url set to "http://localhost:11434"
        - vram_strategy set to "adaptive"
        - get_model_for_agent returning "huihui_ai/dolphin3-abliterated:8b"
        - agent_temperatures initialized to {"writer": 0.9}
        - get_temperature_for_agent that returns the temperature for known roles

        Returns:
            MagicMock: A configured MagicMock instance that conforms to the Settings spec.
        """
        mock = MagicMock(spec=Settings)
        mock.ollama_url = "http://localhost:11434"
        mock.vram_strategy = "adaptive"
        mock.get_model_for_agent.return_value = "huihui_ai/dolphin3-abliterated:8b"
        mock.agent_temperatures = {"writer": 0.9}

        def _get_temp(role: str) -> float:
            """
            Retrieve the temperature value for a given agent role from the mock configuration.

            Parameters:
                role (str): Agent role name to look up in mock.agent_temperatures.

            Returns:
                float: Temperature value associated with the specified role.

            Raises:
                ValueError: If the provided role is not present in mock.agent_temperatures.
            """
            if role not in mock.agent_temperatures:
                raise ValueError(f"Unknown agent role: {role}")
            return float(mock.agent_temperatures[role])

        mock.get_temperature_for_agent.side_effect = _get_temp
        return mock

    @pytest.fixture
    def service(self, mock_settings: MagicMock, temp_db: Path) -> ModelModeService:
        """
        Create a ModelModeService configured with the provided settings and database path.

        Returns:
            ModelModeService: Instance configured with `mock_settings` and `db_path=temp_db`.
        """
        return ModelModeService(mock_settings, db_path=temp_db)

    def test_set_mode_syncs_vram_strategy_to_settings(self, service: ModelModeService):
        """Setting a mode should sync its VRAM strategy to settings."""
        # quality_max uses SEQUENTIAL
        service.set_mode("quality_max")
        assert service.settings.vram_strategy == "sequential"

        # draft_fast uses PARALLEL
        service.set_mode("draft_fast")
        assert service.settings.vram_strategy == "parallel"

        # balanced uses ADAPTIVE
        service.set_mode("balanced")
        assert service.settings.vram_strategy == "adaptive"

    def test_prepare_model_uses_settings_vram_strategy(self, service: ModelModeService):
        """prepare_model should use settings.vram_strategy, not mode's strategy."""
        from unittest.mock import patch

        # Set mode to quality_max (SEQUENTIAL)
        service.set_mode("quality_max")

        # Override settings to use parallel
        service.settings.vram_strategy = "parallel"

        # prepare_model should use "parallel" from settings
        with patch("src.services.model_mode_service._vram.unload_all_except") as mock_unload:
            service.prepare_model("huihui_ai/dolphin3-abliterated:8b")
            # PARALLEL strategy should NOT call unload
            mock_unload.assert_not_called()

    def test_prepare_model_with_sequential_strategy(self, service: ModelModeService):
        """SEQUENTIAL strategy should unload other models."""
        from unittest.mock import patch

        from src.services.model_mode_service import _vram

        service.settings.vram_strategy = "sequential"

        # Reset the module-level short-circuit cache so prepare_model runs full preparation
        with _vram._last_prepared_model_lock:
            _vram._last_prepared_model_key = None

        with patch("src.services.model_mode_service._vram.unload_all_except") as mock_unload:
            service.prepare_model("huihui_ai/dolphin3-abliterated:8b")
            mock_unload.assert_called_once()

    def test_prepare_model_with_invalid_strategy_raises_error(self, service: ModelModeService):
        """Invalid vram_strategy should raise ValueError."""
        service.settings.vram_strategy = "invalid_strategy"

        # Should raise error for invalid strategy
        with pytest.raises(ValueError, match="Invalid vram_strategy 'invalid_strategy'"):
            service.prepare_model("huihui_ai/dolphin3-abliterated:8b")

"""Tests for the settings module."""


from settings import Settings, get_model_info, AVAILABLE_MODELS, AGENT_ROLES


class TestSettings:
    """Tests for Settings class."""

    def test_default_values(self):
        """Should have sensible default values."""
        settings = Settings()
        assert settings.ollama_url == "http://localhost:11434"
        assert settings.context_size == 32768
        assert settings.max_tokens == 8192
        assert settings.use_per_agent_models is True
        assert settings.interaction_mode == "checkpoint"

    def test_get_temperature_for_agent(self):
        """Should return correct temperature for each agent role."""
        settings = Settings()
        assert settings.get_temperature_for_agent("writer") == 0.9
        assert settings.get_temperature_for_agent("editor") == 0.6
        assert settings.get_temperature_for_agent("continuity") == 0.3
        # Unknown role should return default
        assert settings.get_temperature_for_agent("unknown") == 0.8

    def test_get_model_for_agent_with_override(self):
        """Should return overridden model when set."""
        settings = Settings()
        settings.agent_models["writer"] = "custom-model:7b"
        result = settings.get_model_for_agent("writer")
        assert result == "custom-model:7b"

    def test_get_model_for_agent_uses_default_when_not_per_agent(self):
        """Should return default model when per-agent models disabled."""
        settings = Settings()
        settings.use_per_agent_models = False
        settings.default_model = "test-model:8b"
        result = settings.get_model_for_agent("writer")
        assert result == "test-model:8b"


class TestGetModelInfo:
    """Tests for get_model_info function."""

    def test_returns_known_model_info(self):
        """Should return info for known models."""
        # Pick a model from the registry
        model_id = list(AVAILABLE_MODELS.keys())[0]
        info = get_model_info(model_id)
        assert "name" in info
        assert "quality" in info
        assert "speed" in info

    def test_returns_default_for_unknown_model(self):
        """Should return default info for unknown models."""
        info = get_model_info("completely-unknown-model:99b")
        assert info["name"] == "completely-unknown-model:99b"
        assert info["quality"] == 5
        assert info["speed"] == 5


class TestAgentRoles:
    """Tests for agent role definitions."""

    def test_all_roles_defined(self):
        """Should have all required agent roles defined."""
        required_roles = ["interviewer", "architect", "writer", "editor", "continuity"]
        for role in required_roles:
            assert role in AGENT_ROLES
            assert "name" in AGENT_ROLES[role]
            assert "recommended_quality" in AGENT_ROLES[role]

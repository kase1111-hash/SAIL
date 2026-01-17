"""
Tests for SAIL Configuration System
"""

from pathlib import Path

import pytest
from src.config import (
    AccessLevel,
    Config,
    ConfigurationError,
    Environment,
    InterventionMode,
    UserConfig,
    UserRole,
    load_config,
    save_config,
)
from src.config.loader import _convert_env_value, deep_merge, get_env_overrides


class TestConfigSchema:
    """Tests for configuration schema validation."""

    def test_default_config_creation(self):
        """Test that default config can be created."""
        config = Config()
        assert config.sail.version == "0.1.0"
        assert config.sail.environment == Environment.DEVELOPMENT
        assert config.llm.provider == "ollama"

    def test_user_role_enum(self):
        """Test user role enum values."""
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.USER.value == "user"
        assert UserRole.MINOR.value == "minor"

    def test_access_level_enum(self):
        """Test access level enum values."""
        assert AccessLevel.FULL.value == "full"
        assert AccessLevel.STANDARD.value == "standard"
        assert AccessLevel.LIMITED.value == "limited"

    def test_intervention_mode_enum(self):
        """Test intervention mode enum values."""
        assert InterventionMode.AMBIENT.value == "ambient"
        assert InterventionMode.ADVISORY.value == "advisory"
        assert InterventionMode.GUARDIAN.value == "guardian"
        assert InterventionMode.CRISIS.value == "crisis"

    def test_user_config_validation(self):
        """Test user configuration validation."""
        user = UserConfig(name="test_user", role=UserRole.ADMIN, access=AccessLevel.FULL)
        assert user.name == "test_user"
        assert user.role == UserRole.ADMIN
        assert user.restrictions == []

    def test_user_name_required(self):
        """Test that user name is required."""
        with pytest.raises(ValueError):
            UserConfig(name="", role=UserRole.USER, access=AccessLevel.STANDARD)

    def test_unique_user_names(self):
        """Test that user names must be unique."""
        with pytest.raises(ValueError, match="unique"):
            Config(
                users=[
                    UserConfig(name="user1", role=UserRole.USER, access=AccessLevel.STANDARD),
                    UserConfig(name="user1", role=UserRole.ADMIN, access=AccessLevel.FULL),
                ]
            )

    def test_get_user_by_name(self):
        """Test retrieving user by name."""
        config = Config(
            users=[
                UserConfig(name="alice", role=UserRole.ADMIN, access=AccessLevel.FULL),
                UserConfig(name="bob", role=UserRole.USER, access=AccessLevel.STANDARD),
            ]
        )
        alice = config.get_user("alice")
        assert alice is not None
        assert alice.role == UserRole.ADMIN

        unknown = config.get_user("unknown")
        assert unknown is None

    def test_get_admins(self):
        """Test retrieving admin users."""
        config = Config(
            users=[
                UserConfig(name="admin1", role=UserRole.ADMIN, access=AccessLevel.FULL),
                UserConfig(name="user1", role=UserRole.USER, access=AccessLevel.STANDARD),
                UserConfig(name="admin2", role=UserRole.ADMIN, access=AccessLevel.FULL),
            ]
        )
        admins = config.get_admins()
        assert len(admins) == 2
        assert all(admin.role == UserRole.ADMIN for admin in admins)

    def test_llm_config_validation(self):
        """Test LLM configuration validation."""
        config = Config()
        assert config.llm.timeout_seconds == 60
        assert 0.0 <= config.llm.temperature <= 2.0

    def test_intervention_thresholds(self):
        """Test that intervention thresholds are valid."""
        config = Config()
        assert config.intervention.advisory_threshold < config.intervention.guardian_threshold
        assert config.intervention.guardian_threshold < config.intervention.crisis_threshold


class TestConfigLoader:
    """Tests for configuration loading functionality."""

    def test_load_from_yaml(self, temp_config_file: Path):
        """Test loading configuration from YAML file."""
        config = load_config(temp_config_file)
        assert config.sail.environment == Environment.TESTING
        assert config.llm.model == "llama3"

    def test_load_nonexistent_file_strict(self):
        """Test that loading nonexistent file raises error in strict mode."""
        with pytest.raises(ConfigurationError):
            load_config("/nonexistent/path/config.yaml", strict=True)

    def test_load_nonexistent_file_not_strict(self):
        """Test that loading nonexistent file returns defaults when not strict."""
        config = load_config("/nonexistent/path/config.yaml", strict=False)
        assert config.sail.version == "0.1.0"

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        """Test that saving and loading config preserves values."""
        original = Config(
            sail={"version": "1.0.0", "environment": "production"},
            llm={"model": "mistral", "temperature": 0.5},
        )

        config_path = tmp_path / "test_config.yaml"
        save_config(original, config_path)

        loaded = load_config(config_path)
        assert loaded.sail.version == "1.0.0"
        assert loaded.sail.environment == Environment.PRODUCTION
        assert loaded.llm.model == "mistral"
        assert loaded.llm.temperature == 0.5

    def test_invalid_yaml_raises_error(self, tmp_path: Path):
        """Test that invalid YAML raises ConfigurationError."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("invalid: yaml: content: [")

        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            load_config(bad_yaml)


class TestDeepMerge:
    """Tests for deep dictionary merging."""

    def test_simple_merge(self):
        """Test simple key merging."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        """Test nested dictionary merging."""
        base = {"outer": {"inner1": 1, "inner2": 2}}
        override = {"outer": {"inner2": 3, "inner3": 4}}
        result = deep_merge(base, override)
        assert result == {"outer": {"inner1": 1, "inner2": 3, "inner3": 4}}

    def test_base_unchanged(self):
        """Test that base dictionary is not modified."""
        base = {"a": 1}
        override = {"b": 2}
        deep_merge(base, override)
        assert base == {"a": 1}


class TestEnvOverrides:
    """Tests for environment variable overrides."""

    def test_convert_env_value_bool(self):
        """Test boolean conversion."""
        assert _convert_env_value("true") is True
        assert _convert_env_value("True") is True
        assert _convert_env_value("yes") is True
        assert _convert_env_value("1") is True
        assert _convert_env_value("false") is False
        assert _convert_env_value("no") is False
        assert _convert_env_value("0") is False

    def test_convert_env_value_int(self):
        """Test integer conversion."""
        assert _convert_env_value("42") == 42
        assert _convert_env_value("-10") == -10

    def test_convert_env_value_float(self):
        """Test float conversion."""
        assert _convert_env_value("3.14") == 3.14
        assert _convert_env_value("-0.5") == -0.5

    def test_convert_env_value_string(self):
        """Test string passthrough."""
        assert _convert_env_value("hello") == "hello"
        assert _convert_env_value("some value") == "some value"

    def test_get_env_overrides(self, monkeypatch):
        """Test extracting config from environment."""
        monkeypatch.setenv("SAIL_LLM_MODEL", "mistral")
        monkeypatch.setenv("SAIL_LLM_TEMPERATURE", "0.5")
        monkeypatch.setenv("SAIL_VOICE_STT_ENGINE", "faster-whisper")

        overrides = get_env_overrides()

        assert overrides["llm"]["model"] == "mistral"
        assert overrides["llm"]["temperature"] == 0.5
        assert overrides["voice"]["stt"]["engine"] == "faster-whisper"

    def test_env_overrides_applied(self, temp_config_file: Path, monkeypatch):
        """Test that env overrides are applied to loaded config."""
        monkeypatch.setenv("SAIL_LLM_MODEL", "custom-model")

        config = load_config(temp_config_file, use_env_overrides=True)
        assert config.llm.model == "custom-model"

    def test_env_overrides_disabled(self, temp_config_file: Path, monkeypatch):
        """Test that env overrides can be disabled."""
        monkeypatch.setenv("SAIL_LLM_MODEL", "custom-model")

        config = load_config(temp_config_file, use_env_overrides=False)
        assert config.llm.model == "llama3"  # Original value from fixture

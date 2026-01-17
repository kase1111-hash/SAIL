"""
SAIL Configuration Loader

Loads configuration from YAML files with environment variable overrides.
Supports hierarchical configuration merging and validation.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .schema import Config


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""

    pass


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge two dictionaries.

    Args:
        base: Base dictionary
        override: Dictionary with values to override

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def get_env_overrides() -> dict[str, Any]:
    """
    Extract configuration overrides from environment variables.

    Environment variables follow the pattern: SAIL_SECTION_KEY=value
    For nested keys: SAIL_SECTION_SUBSECTION_KEY=value

    Examples:
        SAIL_LLM_MODEL=mistral -> {"llm": {"model": "mistral"}}
        SAIL_VOICE_STT_ENGINE=faster-whisper -> {"voice": {"stt": {"engine": "faster-whisper"}}}

    Returns:
        Dictionary of configuration overrides
    """
    overrides: dict[str, Any] = {}
    prefix = "SAIL_"

    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue

        # Remove prefix and split into parts
        parts = key[len(prefix) :].lower().split("_")

        if len(parts) < 2:
            continue

        # Build nested dictionary
        current = overrides
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the value, attempting type conversion
        final_key = parts[-1]
        current[final_key] = _convert_env_value(value)

    return overrides


def _convert_env_value(value: str) -> Any:
    """
    Convert environment variable string to appropriate Python type.

    Args:
        value: String value from environment

    Returns:
        Converted value (bool, int, float, or str)
    """
    # Boolean
    if value.lower() in ("true", "yes", "1", "on"):
        return True
    if value.lower() in ("false", "no", "0", "off"):
        return False

    # Integer
    try:
        return int(value)
    except ValueError:
        pass

    # Float
    try:
        return float(value)
    except ValueError:
        pass

    # String (default)
    return value


def load_yaml_file(path: Path) -> dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        path: Path to YAML file

    Returns:
        Parsed YAML as dictionary

    Raises:
        ConfigurationError: If file cannot be read or parsed
    """
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
            return data if data else {}
    except FileNotFoundError:
        raise ConfigurationError(f"Configuration file not found: {path}")
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in {path}: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error reading {path}: {e}")


def find_config_file() -> Path | None:
    """
    Search for configuration file in standard locations.

    Search order:
        1. SAIL_CONFIG_PATH environment variable
        2. ./config/config.yaml
        3. ./config.yaml
        4. ~/.config/sail/config.yaml
        5. /etc/sail/config.yaml

    Returns:
        Path to configuration file, or None if not found
    """
    # Check environment variable first
    env_path = os.environ.get("SAIL_CONFIG_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    # Standard search locations
    search_paths = [
        Path("config/config.yaml"),
        Path("config.yaml"),
        Path.home() / ".config" / "sail" / "config.yaml",
        Path("/etc/sail/config.yaml"),
    ]

    for path in search_paths:
        if path.exists():
            return path

    return None


def load_config(
    config_path: Path | str | None = None,
    *,
    use_env_overrides: bool = True,
    strict: bool = True,
) -> Config:
    """
    Load and validate SAIL configuration.

    Args:
        config_path: Path to configuration file (optional, will search if not provided)
        use_env_overrides: Whether to apply environment variable overrides
        strict: Whether to raise on validation errors (False returns defaults)

    Returns:
        Validated Config object

    Raises:
        ConfigurationError: If configuration is invalid (when strict=True)
    """
    config_data: dict[str, Any] = {}

    # Load from file if available
    if config_path:
        path = Path(config_path) if isinstance(config_path, str) else config_path
        try:
            config_data = load_yaml_file(path)
        except ConfigurationError:
            if strict:
                raise
            # Return defaults on file not found if not strict
            return Config()
    else:
        found_path = find_config_file()
        if found_path:
            config_data = load_yaml_file(found_path)

    # Apply environment variable overrides
    if use_env_overrides:
        env_overrides = get_env_overrides()
        if env_overrides:
            config_data = deep_merge(config_data, env_overrides)

    # Validate and create Config object
    try:
        return Config.model_validate(config_data)
    except ValidationError as e:
        if strict:
            raise ConfigurationError(f"Configuration validation failed:\n{e}")
        # Return defaults on validation failure if not strict
        return Config()


def save_config(config: Config, path: Path | str) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: Config object to save
        path: Destination path

    Raises:
        ConfigurationError: If file cannot be written
    """
    path = Path(path) if isinstance(path, str) else path

    try:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and save
        data = config.model_dump(mode="json")

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    except Exception as e:
        raise ConfigurationError(f"Error saving configuration to {path}: {e}")


def create_default_config(path: Path | str) -> Config:
    """
    Create a default configuration file.

    Args:
        path: Destination path

    Returns:
        The default Config object that was saved
    """
    config = Config()
    save_config(config, path)
    return config


# Module-level configuration instance (lazy loaded)
_config: Config | None = None


def get_config() -> Config:
    """
    Get the global configuration instance.

    Loads configuration on first access.

    Returns:
        Global Config instance
    """
    global _config
    if _config is None:
        _config = load_config(strict=False)
    return _config


def reload_config(config_path: Path | str | None = None) -> Config:
    """
    Reload the global configuration.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Reloaded Config instance
    """
    global _config
    _config = load_config(config_path, strict=False)
    return _config

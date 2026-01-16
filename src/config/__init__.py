"""
SAIL Configuration Module

Provides configuration schema, loading, and validation.
"""

from .loader import (
    ConfigurationError,
    create_default_config,
    get_config,
    load_config,
    reload_config,
    save_config,
)
from .schema import (
    AccessLevel,
    Config,
    ContextConfig,
    Environment,
    InterventionConfig,
    InterventionMode,
    LLMConfig,
    SAILConfig,
    SensorConfig,
    STTConfig,
    TTSConfig,
    UserConfig,
    UserRestriction,
    UserRole,
    VoiceConfig,
    WakeWordConfig,
)

__all__ = [
    # Schema classes
    "Config",
    "SAILConfig",
    "LLMConfig",
    "VoiceConfig",
    "STTConfig",
    "TTSConfig",
    "WakeWordConfig",
    "ContextConfig",
    "InterventionConfig",
    "SensorConfig",
    "UserConfig",
    # Enums
    "Environment",
    "UserRole",
    "AccessLevel",
    "UserRestriction",
    "InterventionMode",
    # Loader functions
    "load_config",
    "save_config",
    "get_config",
    "reload_config",
    "create_default_config",
    "ConfigurationError",
]

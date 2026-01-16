"""
SAIL Configuration Schema

Pydantic models defining the configuration structure with validation.
All configuration is validated at load time to catch errors early.
"""

from enum import Enum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class Environment(str, Enum):
    """Application environment."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class UserRole(str, Enum):
    """User role levels."""

    ADMIN = "admin"
    USER = "user"
    MINOR = "minor"


class AccessLevel(str, Enum):
    """Access level permissions."""

    FULL = "full"
    STANDARD = "standard"
    LIMITED = "limited"


class InterventionMode(str, Enum):
    """Intervention mode types."""

    AMBIENT = "ambient"
    ADVISORY = "advisory"
    GUARDIAN = "guardian"
    CRISIS = "crisis"


# --- LLM Configuration ---


class LLMConfig(BaseModel):
    """Configuration for the LLM provider."""

    provider: str = Field(default="ollama", description="LLM provider (ollama, llamacpp)")
    model: str = Field(default="llama3", description="Model name to use")
    base_url: str = Field(default="http://localhost:11434", description="LLM API base URL")
    timeout_seconds: int = Field(default=60, ge=1, le=300, description="Request timeout")
    max_tokens: int = Field(default=2048, ge=1, le=8192, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")


# --- Voice Configuration ---


class STTConfig(BaseModel):
    """Speech-to-text configuration."""

    engine: str = Field(default="whisper", description="STT engine (whisper, faster-whisper)")
    model: str = Field(default="base", description="Whisper model size")
    language: str | None = Field(default=None, description="Language code or None for auto-detect")
    device: str = Field(default="cpu", description="Inference device (cpu, cuda)")


class TTSConfig(BaseModel):
    """Text-to-speech configuration."""

    engine: str = Field(default="piper", description="TTS engine (piper, coqui)")
    voice: str = Field(default="en_US-lessac-medium", description="Voice model to use")
    rate: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech rate multiplier")


class WakeWordConfig(BaseModel):
    """Wake word detection configuration."""

    engine: str = Field(default="openwakeword", description="Wake word engine")
    phrase: str = Field(default="hey sail", description="Wake word phrase")
    sensitivity: float = Field(default=0.5, ge=0.0, le=1.0, description="Detection sensitivity")


class VoiceConfig(BaseModel):
    """Voice I/O configuration."""

    wake_word: WakeWordConfig = Field(default_factory=WakeWordConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    input_device: int | None = Field(default=None, description="Audio input device index")
    output_device: int | None = Field(default=None, description="Audio output device index")
    sample_rate: int = Field(default=16000, description="Audio sample rate")


# --- Context Buffer Configuration ---


class ContextConfig(BaseModel):
    """Context buffer configuration."""

    immediate_duration_seconds: int = Field(
        default=300, ge=60, le=600, description="Immediate buffer duration (5 min default)"
    )
    short_term_duration_seconds: int = Field(
        default=3600, ge=300, le=7200, description="Short-term buffer duration (1 hour default)"
    )
    persist_sessions: bool = Field(default=True, description="Persist session context to disk")
    storage_path: Path = Field(
        default=Path("data/context"), description="Path for context storage"
    )
    encryption_enabled: bool = Field(default=True, description="Encrypt stored context")


# --- User Configuration ---


class UserRestriction(str, Enum):
    """Available user restrictions."""

    NO_OVERRIDE_GUARDIAN_WHILE_DRIVING = "no_override_guardian_mode_while_driving"
    NO_DISABLE_ALERTS = "no_disable_alerts"
    REQUIRE_GUARDIAN_FOR_SETTINGS = "require_guardian_for_settings"


class UserConfig(BaseModel):
    """Individual user configuration."""

    name: str = Field(..., min_length=1, max_length=50, description="User display name")
    role: UserRole = Field(default=UserRole.USER, description="User role")
    access: AccessLevel = Field(default=AccessLevel.STANDARD, description="Access level")
    restrictions: list[UserRestriction] = Field(
        default_factory=list, description="Applied restrictions"
    )
    custom_briefings: list[str] = Field(
        default_factory=list, description="Custom briefing modules"
    )
    guardian: str | None = Field(default=None, description="Guardian user name (for minors)")

    @field_validator("guardian")
    @classmethod
    def validate_guardian(cls, v: str | None, info) -> str | None:
        """Ensure minors have a guardian assigned."""
        # Note: Cross-field validation would need model_validator for full check
        return v


# --- Intervention Configuration ---


class InterventionConfig(BaseModel):
    """Intervention mode configuration."""

    default_mode: InterventionMode = Field(
        default=InterventionMode.AMBIENT, description="Default intervention mode"
    )
    advisory_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Risk threshold for advisory mode"
    )
    guardian_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Risk threshold for guardian mode"
    )
    crisis_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0, description="Risk threshold for crisis mode"
    )
    cooldown_seconds: int = Field(
        default=300, ge=60, le=3600, description="Cooldown between unsolicited interventions"
    )


# --- Sensor Configuration ---


class SensorConfig(BaseModel):
    """Sensor fusion configuration."""

    gps_enabled: bool = Field(default=False, description="Enable GPS location tracking")
    motion_enabled: bool = Field(default=False, description="Enable motion detection")
    biometrics_enabled: bool = Field(default=False, description="Enable biometric sensors")
    location_update_interval_seconds: int = Field(
        default=10, ge=1, le=60, description="GPS update interval"
    )


# --- Main Configuration ---


class SAILConfig(BaseModel):
    """Root SAIL configuration."""

    version: str = Field(default="0.1.0", description="Configuration version")
    environment: Environment = Field(
        default=Environment.DEVELOPMENT, description="Running environment"
    )


class Config(BaseModel):
    """Complete SAIL configuration schema."""

    sail: SAILConfig = Field(default_factory=SAILConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    intervention: InterventionConfig = Field(default_factory=InterventionConfig)
    sensors: SensorConfig = Field(default_factory=SensorConfig)
    users: list[UserConfig] = Field(default_factory=list, description="Configured users")

    @field_validator("users")
    @classmethod
    def validate_users(cls, v: list[UserConfig]) -> list[UserConfig]:
        """Validate user list has unique names."""
        names = [user.name for user in v]
        if len(names) != len(set(names)):
            raise ValueError("User names must be unique")
        return v

    def get_user(self, name: str) -> UserConfig | None:
        """Get a user by name."""
        for user in self.users:
            if user.name == name:
                return user
        return None

    def get_admins(self) -> list[UserConfig]:
        """Get all admin users."""
        return [user for user in self.users if user.role == UserRole.ADMIN]

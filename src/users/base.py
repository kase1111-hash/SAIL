"""
SAIL User Management Base Module

Provides base types and data models for the multi-user and family system
including user sessions, voice profiles, and permission models.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


# ============================================================================
# User Identity Types
# ============================================================================


class UserRole(str, Enum):
    """User role levels in the system."""

    ADMIN = "admin"         # Full system access and oversight
    USER = "user"           # Standard access
    MINOR = "minor"         # Limited access with guardian oversight


class AccessLevel(str, Enum):
    """Access level permissions."""

    FULL = "full"           # Complete unrestricted access
    STANDARD = "standard"   # Normal operation with possible restrictions
    LIMITED = "limited"     # Restricted access (typically for minors)


class UserRestriction(str, Enum):
    """Available user restrictions."""

    NO_OVERRIDE_GUARDIAN_WHILE_DRIVING = "no_override_guardian_mode_while_driving"
    NO_DISABLE_ALERTS = "no_disable_alerts"
    REQUIRE_GUARDIAN_FOR_SETTINGS = "require_guardian_for_settings"
    NO_CRISIS_MODE_DISABLE = "no_crisis_mode_disable"
    REQUIRE_GUARDIAN_FOR_LOCATION_SHARE = "require_guardian_for_location_share"


class IdentificationMethod(str, Enum):
    """Methods for identifying users."""

    VOICE = "voice"                 # Voice print matching
    EXPLICIT = "explicit"           # User explicitly identified themselves
    PIN = "pin"                     # PIN code entry
    DEVICE = "device"               # Device-based identification
    DEFAULT = "default"             # Default/unknown user
    GUARDIAN_OVERRIDE = "guardian_override"  # Guardian took control


class SessionState(str, Enum):
    """States of a user session."""

    ACTIVE = "active"               # User is actively interacting
    IDLE = "idle"                   # User identified but not active
    EXPIRED = "expired"             # Session has timed out
    LOCKED = "locked"               # Session locked (requires re-auth)
    GUARDIAN_CONTROLLED = "guardian_controlled"  # Guardian has taken control


# ============================================================================
# Voice Profile
# ============================================================================


@dataclass
class VoiceProfile:
    """
    Voice profile for speaker identification.

    Stores voice embeddings and metadata for identifying
    a user by their voice characteristics.
    """

    user_id: str
    profile_id: str = field(default_factory=lambda: str(uuid4()))

    # Voice embeddings (list of embedding vectors from enrollment samples)
    embeddings: list[list[float]] = field(default_factory=list)

    # Profile metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    sample_count: int = 0

    # Quality metrics
    average_confidence: float = 0.0
    min_confidence_threshold: float = 0.7

    # Enrollment status
    is_enrolled: bool = False
    enrollment_complete: bool = False
    required_samples: int = 5

    def add_embedding(self, embedding: list[float], confidence: float = 1.0) -> None:
        """Add a voice embedding sample."""
        self.embeddings.append(embedding)
        self.sample_count = len(self.embeddings)
        self.updated_at = datetime.now()

        # Update average confidence
        if self.average_confidence == 0:
            self.average_confidence = confidence
        else:
            self.average_confidence = (
                self.average_confidence * (self.sample_count - 1) + confidence
            ) / self.sample_count

        # Check if enrollment is complete
        if self.sample_count >= self.required_samples:
            self.enrollment_complete = True
            self.is_enrolled = True

    def get_centroid(self) -> Optional[list[float]]:
        """Get the centroid (average) of all embeddings."""
        if not self.embeddings:
            return None

        # Average all embeddings
        dim = len(self.embeddings[0])
        centroid = [0.0] * dim

        for embedding in self.embeddings:
            for i, val in enumerate(embedding):
                centroid[i] += val

        for i in range(dim):
            centroid[i] /= len(self.embeddings)

        return centroid

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "user_id": self.user_id,
            "profile_id": self.profile_id,
            "embeddings": self.embeddings,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "sample_count": self.sample_count,
            "average_confidence": self.average_confidence,
            "min_confidence_threshold": self.min_confidence_threshold,
            "is_enrolled": self.is_enrolled,
            "enrollment_complete": self.enrollment_complete,
            "required_samples": self.required_samples,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VoiceProfile:
        """Create from dictionary."""
        profile = cls(
            user_id=data["user_id"],
            profile_id=data.get("profile_id", str(uuid4())),
        )
        profile.embeddings = data.get("embeddings", [])
        profile.created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now()
        profile.updated_at = datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now()
        profile.sample_count = data.get("sample_count", len(profile.embeddings))
        profile.average_confidence = data.get("average_confidence", 0.0)
        profile.min_confidence_threshold = data.get("min_confidence_threshold", 0.7)
        profile.is_enrolled = data.get("is_enrolled", False)
        profile.enrollment_complete = data.get("enrollment_complete", False)
        profile.required_samples = data.get("required_samples", 5)
        return profile


# ============================================================================
# User Model
# ============================================================================


@dataclass
class User:
    """
    Represents a user in the SAIL system.

    Contains identity, permissions, and profile information.
    """

    user_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""

    # Role and access
    role: UserRole = UserRole.USER
    access_level: AccessLevel = AccessLevel.STANDARD
    restrictions: list[UserRestriction] = field(default_factory=list)

    # Guardian relationship
    guardian_id: Optional[str] = None
    dependents: list[str] = field(default_factory=list)

    # Voice profile
    voice_profile: Optional[VoiceProfile] = None

    # Custom briefings and preferences
    custom_briefings: list[str] = field(default_factory=list)
    preferences: dict[str, Any] = field(default_factory=dict)

    # Account metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_active: Optional[datetime] = None
    is_active: bool = True

    def __post_init__(self) -> None:
        """Initialize voice profile if not set."""
        if self.voice_profile is None:
            self.voice_profile = VoiceProfile(user_id=self.user_id)

    @property
    def is_minor(self) -> bool:
        """Check if user is a minor."""
        return self.role == UserRole.MINOR

    @property
    def is_admin(self) -> bool:
        """Check if user is an admin."""
        return self.role == UserRole.ADMIN

    @property
    def has_guardian(self) -> bool:
        """Check if user has a guardian assigned."""
        return self.guardian_id is not None

    @property
    def is_guardian(self) -> bool:
        """Check if user is a guardian (has dependents)."""
        return len(self.dependents) > 0

    def has_restriction(self, restriction: UserRestriction) -> bool:
        """Check if user has a specific restriction."""
        return restriction in self.restrictions

    def can_override_guardian_mode(self, is_driving: bool = False) -> bool:
        """Check if user can override guardian mode."""
        if self.is_admin:
            return True
        if is_driving and self.has_restriction(UserRestriction.NO_OVERRIDE_GUARDIAN_WHILE_DRIVING):
            return False
        return True

    def can_disable_alerts(self) -> bool:
        """Check if user can disable alerts."""
        if self.is_admin:
            return True
        return not self.has_restriction(UserRestriction.NO_DISABLE_ALERTS)

    def can_change_settings(self) -> bool:
        """Check if user can change settings independently."""
        if self.is_admin:
            return True
        if self.has_restriction(UserRestriction.REQUIRE_GUARDIAN_FOR_SETTINGS):
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "user_id": self.user_id,
            "name": self.name,
            "role": self.role.value,
            "access_level": self.access_level.value,
            "restrictions": [r.value for r in self.restrictions],
            "guardian_id": self.guardian_id,
            "dependents": self.dependents,
            "voice_profile": self.voice_profile.to_dict() if self.voice_profile else None,
            "custom_briefings": self.custom_briefings,
            "preferences": self.preferences,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat() if self.last_active else None,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> User:
        """Create from dictionary."""
        user = cls(
            user_id=data.get("user_id", str(uuid4())),
            name=data.get("name", ""),
            role=UserRole(data.get("role", "user")),
            access_level=AccessLevel(data.get("access_level", "standard")),
            restrictions=[UserRestriction(r) for r in data.get("restrictions", [])],
            guardian_id=data.get("guardian_id"),
            dependents=data.get("dependents", []),
            custom_briefings=data.get("custom_briefings", []),
            preferences=data.get("preferences", {}),
            is_active=data.get("is_active", True),
        )

        if data.get("voice_profile"):
            user.voice_profile = VoiceProfile.from_dict(data["voice_profile"])

        if data.get("created_at"):
            user.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("last_active"):
            user.last_active = datetime.fromisoformat(data["last_active"])

        return user


# ============================================================================
# User Session
# ============================================================================


@dataclass
class UserSession:
    """
    Represents an active user session.

    Tracks the currently identified user, session state,
    and interaction history.
    """

    session_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    user: Optional[User] = None

    # Session state
    state: SessionState = SessionState.ACTIVE
    identification_method: IdentificationMethod = IdentificationMethod.DEFAULT
    identification_confidence: float = 0.0

    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    # Session configuration
    session_timeout_seconds: int = 1800  # 30 minutes default

    # Guardian control
    guardian_controlled_by: Optional[str] = None
    guardian_control_reason: Optional[str] = None

    # Interaction tracking
    interaction_count: int = 0
    alerts_shown: int = 0
    alerts_dismissed: int = 0

    def __post_init__(self) -> None:
        """Set expiration time."""
        if self.expires_at is None:
            self.expires_at = self.started_at + timedelta(seconds=self.session_timeout_seconds)

    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return self.state == SessionState.ACTIVE

    @property
    def is_expired(self) -> bool:
        """Check if session has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    @property
    def is_guardian_controlled(self) -> bool:
        """Check if session is under guardian control."""
        return self.state == SessionState.GUARDIAN_CONTROLLED

    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds."""
        return (datetime.now() - self.started_at).total_seconds()

    @property
    def idle_seconds(self) -> float:
        """Get idle time in seconds."""
        return (datetime.now() - self.last_activity).total_seconds()

    def touch(self) -> None:
        """Update last activity time."""
        self.last_activity = datetime.now()
        self.interaction_count += 1

        # Extend expiration
        self.expires_at = self.last_activity + timedelta(seconds=self.session_timeout_seconds)

        # Reactivate if idle
        if self.state == SessionState.IDLE:
            self.state = SessionState.ACTIVE

    def set_idle(self) -> None:
        """Set session to idle state."""
        if self.state == SessionState.ACTIVE:
            self.state = SessionState.IDLE

    def expire(self) -> None:
        """Expire the session."""
        self.state = SessionState.EXPIRED

    def lock(self) -> None:
        """Lock the session."""
        self.state = SessionState.LOCKED

    def guardian_takeover(self, guardian_id: str, reason: str = "") -> None:
        """Guardian takes control of the session."""
        self.state = SessionState.GUARDIAN_CONTROLLED
        self.guardian_controlled_by = guardian_id
        self.guardian_control_reason = reason
        self.touch()

    def release_guardian_control(self) -> None:
        """Release guardian control of the session."""
        if self.state == SessionState.GUARDIAN_CONTROLLED:
            self.state = SessionState.ACTIVE
            self.guardian_controlled_by = None
            self.guardian_control_reason = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "state": self.state.value,
            "identification_method": self.identification_method.value,
            "identification_confidence": self.identification_confidence,
            "started_at": self.started_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "session_timeout_seconds": self.session_timeout_seconds,
            "guardian_controlled_by": self.guardian_controlled_by,
            "guardian_control_reason": self.guardian_control_reason,
            "interaction_count": self.interaction_count,
            "alerts_shown": self.alerts_shown,
            "alerts_dismissed": self.alerts_dismissed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserSession:
        """Create from dictionary."""
        session = cls(
            session_id=data.get("session_id", str(uuid4())),
            user_id=data.get("user_id", ""),
            state=SessionState(data.get("state", "active")),
            identification_method=IdentificationMethod(data.get("identification_method", "default")),
            identification_confidence=data.get("identification_confidence", 0.0),
            session_timeout_seconds=data.get("session_timeout_seconds", 1800),
            guardian_controlled_by=data.get("guardian_controlled_by"),
            guardian_control_reason=data.get("guardian_control_reason"),
            interaction_count=data.get("interaction_count", 0),
            alerts_shown=data.get("alerts_shown", 0),
            alerts_dismissed=data.get("alerts_dismissed", 0),
        )

        if data.get("started_at"):
            session.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("last_activity"):
            session.last_activity = datetime.fromisoformat(data["last_activity"])
        if data.get("expires_at"):
            session.expires_at = datetime.fromisoformat(data["expires_at"])

        return session


# ============================================================================
# User Events
# ============================================================================


class UserEventType(str, Enum):
    """Types of user-related events."""

    USER_IDENTIFIED = "user_identified"
    USER_SWITCHED = "user_switched"
    SESSION_STARTED = "session_started"
    SESSION_EXPIRED = "session_expired"
    SESSION_LOCKED = "session_locked"
    GUARDIAN_TAKEOVER = "guardian_takeover"
    GUARDIAN_RELEASED = "guardian_released"
    VOICE_ENROLLED = "voice_enrolled"
    PERMISSION_DENIED = "permission_denied"
    RESTRICTION_TRIGGERED = "restriction_triggered"


@dataclass
class UserEvent:
    """Event from the user management system."""

    event_type: UserEventType
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class UserManagerConfig:
    """Configuration for the user management system."""

    # Voice identification
    voice_id_enabled: bool = True
    voice_confidence_threshold: float = 0.7
    voice_enrollment_samples: int = 5

    # Session management
    session_timeout_seconds: int = 1800
    idle_timeout_seconds: int = 300
    require_identification: bool = False

    # Guardian system
    guardian_alerts_enabled: bool = True
    guardian_alert_cooldown_seconds: int = 300

    # Data storage
    data_path: Path = field(default_factory=lambda: Path("data/users"))
    encryption_enabled: bool = True

    # Default user
    default_user_name: str = "default"
    allow_anonymous: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "voice_id_enabled": self.voice_id_enabled,
            "voice_confidence_threshold": self.voice_confidence_threshold,
            "voice_enrollment_samples": self.voice_enrollment_samples,
            "session_timeout_seconds": self.session_timeout_seconds,
            "idle_timeout_seconds": self.idle_timeout_seconds,
            "require_identification": self.require_identification,
            "guardian_alerts_enabled": self.guardian_alerts_enabled,
            "guardian_alert_cooldown_seconds": self.guardian_alert_cooldown_seconds,
            "data_path": str(self.data_path),
            "encryption_enabled": self.encryption_enabled,
            "default_user_name": self.default_user_name,
            "allow_anonymous": self.allow_anonymous,
        }

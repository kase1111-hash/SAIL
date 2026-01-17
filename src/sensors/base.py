"""
Base types and abstractions for the sensor fusion layer.

Provides sensor abstractions, reading types, and state management
for unified situational awareness.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================
# Sensor Types and Enumerations
# ============================================================================


class SensorType(str, Enum):
    """Types of sensors in the system."""

    LOCATION = "location"
    MOTION = "motion"
    TEMPORAL = "temporal"
    AUDIO = "audio"
    BIOMETRIC = "biometric"


class SensorStatus(str, Enum):
    """Status of a sensor."""

    UNKNOWN = "unknown"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


class MotionState(str, Enum):
    """Motion state of the user."""

    UNKNOWN = "unknown"
    STATIONARY = "stationary"
    WALKING = "walking"
    RUNNING = "running"
    DRIVING = "driving"
    IN_VEHICLE = "in_vehicle"  # Passenger, not driving


class AudioEnvironment(str, Enum):
    """Classification of audio environment."""

    UNKNOWN = "unknown"
    QUIET = "quiet"
    INDOOR_NORMAL = "indoor_normal"
    OUTDOOR = "outdoor"
    TRAFFIC = "traffic"
    CROWD = "crowd"
    CONVERSATION = "conversation"
    LOUD = "loud"


class TimeContext(str, Enum):
    """Time-based context categories."""

    EARLY_MORNING = "early_morning"     # 5-8 AM
    MORNING = "morning"                  # 8-12 PM
    AFTERNOON = "afternoon"              # 12-5 PM
    EVENING = "evening"                  # 5-9 PM
    NIGHT = "night"                      # 9 PM - 12 AM
    LATE_NIGHT = "late_night"           # 12-5 AM


class LocationContext(str, Enum):
    """Location-based context categories."""

    UNKNOWN = "unknown"
    HOME = "home"
    WORK = "work"
    FAMILIAR = "familiar"
    UNFAMILIAR = "unfamiliar"
    TRANSIT = "transit"


class StressLevel(str, Enum):
    """Detected stress level."""

    UNKNOWN = "unknown"
    LOW = "low"
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"


class Confidence(str, Enum):
    """Confidence level for sensor readings."""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


# ============================================================================
# Data Classes for Sensor Readings
# ============================================================================


@dataclass
class GeoLocation:
    """Geographic location data."""

    latitude: float
    longitude: float
    altitude: float | None = None
    accuracy_meters: float | None = None
    heading: float | None = None  # Degrees from true north
    speed_mps: float | None = None  # Meters per second
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
            "accuracy_meters": self.accuracy_meters,
            "heading": self.heading,
            "speed_mps": self.speed_mps,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeoLocation":
        """Create from dictionary."""
        return cls(
            latitude=data["latitude"],
            longitude=data["longitude"],
            altitude=data.get("altitude"),
            accuracy_meters=data.get("accuracy_meters"),
            heading=data.get("heading"),
            speed_mps=data.get("speed_mps"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
        )


@dataclass
class Jurisdiction:
    """Jurisdiction information for location."""

    country: str
    country_code: str  # ISO 3166-1 alpha-2
    state: str | None = None
    state_code: str | None = None
    county: str | None = None
    city: str | None = None
    postal_code: str | None = None
    timezone: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "country": self.country,
            "country_code": self.country_code,
            "state": self.state,
            "state_code": self.state_code,
            "county": self.county,
            "city": self.city,
            "postal_code": self.postal_code,
            "timezone": self.timezone,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Jurisdiction":
        """Create from dictionary."""
        return cls(
            country=data["country"],
            country_code=data["country_code"],
            state=data.get("state"),
            state_code=data.get("state_code"),
            county=data.get("county"),
            city=data.get("city"),
            postal_code=data.get("postal_code"),
            timezone=data.get("timezone"),
        )


@dataclass
class Geofence:
    """A geographic boundary for location detection."""

    id: str
    name: str
    latitude: float
    longitude: float
    radius_meters: float
    context: LocationContext = LocationContext.FAMILIAR
    metadata: dict[str, Any] = field(default_factory=dict)

    def contains(self, location: GeoLocation) -> bool:
        """Check if a location is within this geofence."""
        from math import atan2, cos, radians, sin, sqrt

        # Haversine formula
        R = 6371000  # Earth's radius in meters

        lat1 = radians(self.latitude)
        lat2 = radians(location.latitude)
        dlat = radians(location.latitude - self.latitude)
        dlon = radians(location.longitude - self.longitude)

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))

        distance = R * c
        return distance <= self.radius_meters

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "radius_meters": self.radius_meters,
            "context": self.context.value,
            "metadata": self.metadata,
        }


@dataclass
class MotionReading:
    """Motion sensor reading."""

    state: MotionState
    confidence: Confidence
    speed_mps: float | None = None
    acceleration: tuple[float, float, float] | None = None  # x, y, z
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state": self.state.value,
            "confidence": self.confidence.value,
            "speed_mps": self.speed_mps,
            "acceleration": self.acceleration,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TemporalReading:
    """Temporal context reading."""

    time_context: TimeContext
    is_weekend: bool
    is_holiday: bool
    local_time: datetime
    timezone: str
    calendar_event: str | None = None
    event_context: str | None = None  # "work_hours", "commute", etc.

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "time_context": self.time_context.value,
            "is_weekend": self.is_weekend,
            "is_holiday": self.is_holiday,
            "local_time": self.local_time.isoformat(),
            "timezone": self.timezone,
            "calendar_event": self.calendar_event,
            "event_context": self.event_context,
        }


@dataclass
class AudioReading:
    """Audio environment reading."""

    environment: AudioEnvironment
    confidence: Confidence
    noise_level_db: float | None = None
    speech_detected: bool = False
    stress_indicators: StressLevel = StressLevel.UNKNOWN
    threat_cues: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "environment": self.environment.value,
            "confidence": self.confidence.value,
            "noise_level_db": self.noise_level_db,
            "speech_detected": self.speech_detected,
            "stress_indicators": self.stress_indicators.value,
            "threat_cues": self.threat_cues,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BiometricReading:
    """Biometric sensor reading (optional wearable data)."""

    heart_rate_bpm: int | None = None
    heart_rate_variability: float | None = None
    stress_level: StressLevel = StressLevel.UNKNOWN
    confidence: Confidence = Confidence.LOW
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "heart_rate_bpm": self.heart_rate_bpm,
            "heart_rate_variability": self.heart_rate_variability,
            "stress_level": self.stress_level.value,
            "confidence": self.confidence.value,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SensorReading:
    """Generic sensor reading wrapper."""

    sensor_type: SensorType
    sensor_id: str
    data: Any
    confidence: Confidence
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data_dict = self.data.to_dict() if hasattr(self.data, "to_dict") else self.data
        return {
            "sensor_type": self.sensor_type.value,
            "sensor_id": self.sensor_id,
            "data": data_dict,
            "confidence": self.confidence.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class SensorState:
    """Current state of a sensor."""

    sensor_type: SensorType
    sensor_id: str
    status: SensorStatus
    last_reading: SensorReading | None = None
    last_update: datetime | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sensor_type": self.sensor_type.value,
            "sensor_id": self.sensor_id,
            "status": self.status.value,
            "last_reading": self.last_reading.to_dict() if self.last_reading else None,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "error_message": self.error_message,
        }


# ============================================================================
# Situational State (Combined Awareness)
# ============================================================================


@dataclass
class SituationalState:
    """Combined situational awareness from all sensors."""

    # Location awareness
    location: GeoLocation | None = None
    jurisdiction: Jurisdiction | None = None
    location_context: LocationContext = LocationContext.UNKNOWN
    in_geofence: Geofence | None = None

    # Motion awareness
    motion_state: MotionState = MotionState.UNKNOWN
    speed_mps: float | None = None

    # Temporal awareness
    time_context: TimeContext = TimeContext.MORNING
    is_weekend: bool = False
    is_late_night: bool = False
    calendar_context: str | None = None

    # Audio awareness
    audio_environment: AudioEnvironment = AudioEnvironment.UNKNOWN
    stress_level: StressLevel = StressLevel.UNKNOWN
    threat_cues: list[str] = field(default_factory=list)

    # Overall assessment
    overall_confidence: Confidence = Confidence.LOW
    risk_factors: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "location": self.location.to_dict() if self.location else None,
            "jurisdiction": self.jurisdiction.to_dict() if self.jurisdiction else None,
            "location_context": self.location_context.value,
            "in_geofence": self.in_geofence.to_dict() if self.in_geofence else None,
            "motion_state": self.motion_state.value,
            "speed_mps": self.speed_mps,
            "time_context": self.time_context.value,
            "is_weekend": self.is_weekend,
            "is_late_night": self.is_late_night,
            "calendar_context": self.calendar_context,
            "audio_environment": self.audio_environment.value,
            "stress_level": self.stress_level.value,
            "threat_cues": self.threat_cues,
            "overall_confidence": self.overall_confidence.value,
            "risk_factors": self.risk_factors,
            "timestamp": self.timestamp.isoformat(),
        }

    def is_high_risk_context(self) -> bool:
        """Check if current context indicates elevated risk."""
        risk_indicators = [
            self.is_late_night and self.location_context == LocationContext.UNFAMILIAR,
            self.stress_level in (StressLevel.ELEVATED, StressLevel.HIGH),
            len(self.threat_cues) > 0,
            self.motion_state == MotionState.DRIVING and self.is_late_night,
        ]
        return any(risk_indicators)


# ============================================================================
# Sensor Configuration
# ============================================================================


@dataclass
class SensorConfig:
    """Configuration for sensors."""

    # Update intervals (seconds)
    location_update_interval: float = 10.0
    motion_update_interval: float = 1.0
    temporal_update_interval: float = 60.0
    audio_update_interval: float = 5.0

    # Geofencing
    geofences: list[Geofence] = field(default_factory=list)
    unfamiliar_area_threshold_km: float = 50.0

    # Motion detection
    walking_speed_threshold_mps: float = 0.5
    running_speed_threshold_mps: float = 2.5
    driving_speed_threshold_mps: float = 5.0

    # Audio analysis
    audio_sample_duration_seconds: float = 2.0
    stress_detection_enabled: bool = True

    # General
    timezone: str = "UTC"
    holidays: list[str] = field(default_factory=list)  # ISO date strings


# ============================================================================
# Abstract Sensor Base Class
# ============================================================================


class Sensor(ABC):
    """Abstract base class for all sensors."""

    def __init__(
        self,
        sensor_id: str,
        sensor_type: SensorType,
        config: SensorConfig | None = None,
    ):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.config = config or SensorConfig()
        self._status = SensorStatus.UNKNOWN
        self._last_reading: SensorReading | None = None
        self._last_update: datetime | None = None
        self._error_message: str | None = None
        self._callbacks: list[Callable[[SensorReading], None]] = []
        self._running = False

    @property
    def status(self) -> SensorStatus:
        """Get current sensor status."""
        return self._status

    @property
    def last_reading(self) -> SensorReading | None:
        """Get last sensor reading."""
        return self._last_reading

    def get_state(self) -> SensorState:
        """Get current sensor state."""
        return SensorState(
            sensor_type=self.sensor_type,
            sensor_id=self.sensor_id,
            status=self._status,
            last_reading=self._last_reading,
            last_update=self._last_update,
            error_message=self._error_message,
        )

    def register_callback(self, callback: Callable[[SensorReading], None]) -> None:
        """Register a callback for new readings."""
        self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[SensorReading], None]) -> None:
        """Unregister a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _notify_callbacks(self, reading: SensorReading) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(reading)
            except Exception as e:
                logger.error(f"Callback error for {self.sensor_id}: {e}")

    def _set_reading(self, reading: SensorReading) -> None:
        """Set the latest reading and notify callbacks."""
        self._last_reading = reading
        self._last_update = datetime.now()
        self._notify_callbacks(reading)

    def _set_error(self, message: str) -> None:
        """Set sensor to error state."""
        self._status = SensorStatus.ERROR
        self._error_message = message
        logger.error(f"Sensor {self.sensor_id} error: {message}")

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the sensor. Returns True on success."""
        pass

    @abstractmethod
    async def read(self) -> SensorReading:
        """Take a reading from the sensor."""
        pass

    @abstractmethod
    async def start(self) -> None:
        """Start continuous sensor monitoring."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop sensor monitoring."""
        pass

    async def __aenter__(self) -> "Sensor":
        """Async context manager entry."""
        await self.initialize()
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()


# ============================================================================
# Sensor Events
# ============================================================================


class SensorEventType(str, Enum):
    """Types of sensor events."""

    READING = "reading"
    STATE_CHANGE = "state_change"
    GEOFENCE_ENTER = "geofence_enter"
    GEOFENCE_EXIT = "geofence_exit"
    MOTION_CHANGE = "motion_change"
    STRESS_DETECTED = "stress_detected"
    THREAT_DETECTED = "threat_detected"
    JURISDICTION_CHANGE = "jurisdiction_change"


@dataclass
class SensorEvent:
    """An event from the sensor system."""

    event_type: SensorEventType
    sensor_type: SensorType
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "sensor_type": self.sensor_type.value,
            "data": self.data.to_dict() if hasattr(self.data, "to_dict") else self.data,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

"""
Sensor Fusion Layer for SAIL.

Provides unified situational awareness through multiple sensor inputs:
- Location (GPS, geofencing, jurisdiction detection)
- Motion (state classification, accident detection)
- Temporal (time context, calendar awareness)
- Audio (environment classification, stress detection)
"""

from .audio import (
    AudioAnalysis,
    AudioEnvironmentClassifier,
    AudioFeatureExtractor,
    AudioFeatures,
    AudioProvider,
    AudioSample,
    AudioSensor,
    SimulatedAudioProvider,
    SystemAudioProvider,
    ThreatCueDetector,
    VoiceStressDetector,
)
from .base import (
    AudioEnvironment,
    AudioReading,
    BiometricReading,
    Confidence,
    Geofence,
    # Data classes
    GeoLocation,
    Jurisdiction,
    LocationContext,
    MotionReading,
    MotionState,
    # Base sensor class
    Sensor,
    SensorConfig,
    SensorEvent,
    # Events
    SensorEventType,
    SensorReading,
    SensorState,
    SensorStatus,
    # Sensor types and enums
    SensorType,
    SituationalState,
    StressLevel,
    TemporalReading,
    TimeContext,
)
from .fusion import (
    FusionEvent,
    FusionEventType,
    RiskAssessor,
    SensorFusionManager,
    create_default_fusion_manager,
)
from .location import (
    GeofenceManager,
    GPSProvider,
    JurisdictionDetector,
    LocationReading,
    LocationSensor,
    SimulatedGPSProvider,
    SystemGPSProvider,
    haversine_distance,
)
from .motion import (
    AccelerometerProvider,
    AccelerometerReading,
    AccidentDetector,
    MotionAnalysis,
    MotionSensor,
    MotionStateClassifier,
    SimulatedAccelerometerProvider,
    SystemAccelerometerProvider,
)
from .temporal import (
    CalendarEvent,
    CalendarProvider,
    ICalendarProvider,
    SimulatedCalendarProvider,
    TemporalAnalysis,
    TemporalSensor,
    format_time_until,
    get_day_period,
    get_time_context,
    is_us_holiday,
    is_weekend,
)

__all__ = [
    # Base types
    "SensorType",
    "SensorStatus",
    "MotionState",
    "AudioEnvironment",
    "TimeContext",
    "LocationContext",
    "StressLevel",
    "Confidence",
    "GeoLocation",
    "Jurisdiction",
    "Geofence",
    "MotionReading",
    "TemporalReading",
    "AudioReading",
    "BiometricReading",
    "SensorReading",
    "SensorState",
    "SituationalState",
    "SensorConfig",
    "Sensor",
    "SensorEventType",
    "SensorEvent",
    # Location
    "LocationSensor",
    "LocationReading",
    "GPSProvider",
    "SimulatedGPSProvider",
    "SystemGPSProvider",
    "JurisdictionDetector",
    "GeofenceManager",
    "haversine_distance",
    # Motion
    "MotionSensor",
    "MotionAnalysis",
    "AccelerometerReading",
    "AccelerometerProvider",
    "SimulatedAccelerometerProvider",
    "SystemAccelerometerProvider",
    "MotionStateClassifier",
    "AccidentDetector",
    # Temporal
    "TemporalSensor",
    "TemporalAnalysis",
    "CalendarEvent",
    "CalendarProvider",
    "SimulatedCalendarProvider",
    "ICalendarProvider",
    "get_time_context",
    "is_weekend",
    "is_us_holiday",
    "format_time_until",
    "get_day_period",
    # Audio
    "AudioSensor",
    "AudioAnalysis",
    "AudioSample",
    "AudioFeatures",
    "AudioProvider",
    "SimulatedAudioProvider",
    "SystemAudioProvider",
    "AudioFeatureExtractor",
    "AudioEnvironmentClassifier",
    "VoiceStressDetector",
    "ThreatCueDetector",
    # Fusion
    "SensorFusionManager",
    "FusionEvent",
    "FusionEventType",
    "RiskAssessor",
    "create_default_fusion_manager",
]

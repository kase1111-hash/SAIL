"""
Sensor Fusion Layer for SAIL.

Provides unified situational awareness through multiple sensor inputs:
- Location (GPS, geofencing, jurisdiction detection)
- Motion (state classification, accident detection)
- Temporal (time context, calendar awareness)
- Audio (environment classification, stress detection)
"""

from .base import (
    # Sensor types and enums
    SensorType,
    SensorStatus,
    MotionState,
    AudioEnvironment,
    TimeContext,
    LocationContext,
    StressLevel,
    Confidence,
    # Data classes
    GeoLocation,
    Jurisdiction,
    Geofence,
    MotionReading,
    TemporalReading,
    AudioReading,
    BiometricReading,
    SensorReading,
    SensorState,
    SituationalState,
    SensorConfig,
    # Base sensor class
    Sensor,
    # Events
    SensorEventType,
    SensorEvent,
)

from .location import (
    LocationSensor,
    LocationReading,
    GPSProvider,
    SimulatedGPSProvider,
    SystemGPSProvider,
    JurisdictionDetector,
    GeofenceManager,
    haversine_distance,
)

from .motion import (
    MotionSensor,
    MotionAnalysis,
    AccelerometerReading,
    AccelerometerProvider,
    SimulatedAccelerometerProvider,
    SystemAccelerometerProvider,
    MotionStateClassifier,
    AccidentDetector,
)

from .temporal import (
    TemporalSensor,
    TemporalAnalysis,
    CalendarEvent,
    CalendarProvider,
    SimulatedCalendarProvider,
    ICalendarProvider,
    get_time_context,
    is_weekend,
    is_us_holiday,
    format_time_until,
    get_day_period,
)

from .audio import (
    AudioSensor,
    AudioAnalysis,
    AudioSample,
    AudioFeatures,
    AudioProvider,
    SimulatedAudioProvider,
    SystemAudioProvider,
    AudioFeatureExtractor,
    AudioEnvironmentClassifier,
    VoiceStressDetector,
    ThreatCueDetector,
)

from .fusion import (
    SensorFusionManager,
    FusionEvent,
    FusionEventType,
    RiskAssessor,
    create_default_fusion_manager,
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

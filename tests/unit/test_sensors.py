"""
Unit tests for the Sensor Fusion Layer (Phase 6).

Tests cover:
- Base sensor types and abstractions
- Location sensor with GPS and geofencing
- Motion sensor with state classification
- Temporal sensor with time context
- Audio sensor with environment classification
- Sensor fusion manager
"""

import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from src.sensors.base import (
    SensorType,
    SensorStatus,
    MotionState,
    AudioEnvironment,
    TimeContext,
    LocationContext,
    StressLevel,
    Confidence,
    GeoLocation,
    Jurisdiction,
    Geofence,
    MotionReading,
    TemporalReading,
    AudioReading,
    SensorReading,
    SensorState,
    SituationalState,
    SensorConfig,
    SensorEventType,
    SensorEvent,
)


# ============================================================================
# Base Types Tests
# ============================================================================


class TestGeoLocation:
    """Tests for GeoLocation data class."""

    def test_geolocation_creation(self):
        """Test creating a geo location."""
        loc = GeoLocation(
            latitude=37.7749,
            longitude=-122.4194,
            altitude=10.0,
            accuracy_meters=5.0,
        )

        assert loc.latitude == 37.7749
        assert loc.longitude == -122.4194
        assert loc.altitude == 10.0
        assert loc.accuracy_meters == 5.0

    def test_geolocation_to_dict(self):
        """Test converting geo location to dict."""
        loc = GeoLocation(latitude=37.7749, longitude=-122.4194)
        data = loc.to_dict()

        assert data["latitude"] == 37.7749
        assert data["longitude"] == -122.4194
        assert "timestamp" in data

    def test_geolocation_from_dict(self):
        """Test creating geo location from dict."""
        data = {
            "latitude": 37.7749,
            "longitude": -122.4194,
            "altitude": 10.0,
            "timestamp": datetime.now().isoformat(),
        }
        loc = GeoLocation.from_dict(data)

        assert loc.latitude == 37.7749
        assert loc.longitude == -122.4194


class TestJurisdiction:
    """Tests for Jurisdiction data class."""

    def test_jurisdiction_creation(self):
        """Test creating a jurisdiction."""
        jur = Jurisdiction(
            country="United States",
            country_code="US",
            state="California",
            state_code="CA",
        )

        assert jur.country == "United States"
        assert jur.country_code == "US"
        assert jur.state == "California"
        assert jur.state_code == "CA"

    def test_jurisdiction_to_dict(self):
        """Test converting jurisdiction to dict."""
        jur = Jurisdiction(country="United States", country_code="US")
        data = jur.to_dict()

        assert data["country"] == "United States"
        assert data["country_code"] == "US"


class TestGeofence:
    """Tests for Geofence data class."""

    def test_geofence_creation(self):
        """Test creating a geofence."""
        gf = Geofence(
            id="home",
            name="Home",
            latitude=37.7749,
            longitude=-122.4194,
            radius_meters=100.0,
            context=LocationContext.HOME,
        )

        assert gf.id == "home"
        assert gf.name == "Home"
        assert gf.radius_meters == 100.0
        assert gf.context == LocationContext.HOME

    def test_geofence_contains_inside(self):
        """Test geofence contains for location inside."""
        gf = Geofence(
            id="test",
            name="Test",
            latitude=37.7749,
            longitude=-122.4194,
            radius_meters=1000.0,
        )

        loc = GeoLocation(latitude=37.7749, longitude=-122.4194)
        assert gf.contains(loc)

    def test_geofence_contains_outside(self):
        """Test geofence contains for location outside."""
        gf = Geofence(
            id="test",
            name="Test",
            latitude=37.7749,
            longitude=-122.4194,
            radius_meters=100.0,
        )

        # Location about 10km away
        loc = GeoLocation(latitude=37.8749, longitude=-122.4194)
        assert not gf.contains(loc)


class TestSituationalState:
    """Tests for SituationalState data class."""

    def test_situational_state_creation(self):
        """Test creating situational state."""
        state = SituationalState()

        assert state.motion_state == MotionState.UNKNOWN
        assert state.location_context == LocationContext.UNKNOWN
        assert state.stress_level == StressLevel.UNKNOWN

    def test_high_risk_context_late_night_unfamiliar(self):
        """Test high risk detection for late night in unfamiliar area."""
        state = SituationalState(
            is_late_night=True,
            location_context=LocationContext.UNFAMILIAR,
        )

        assert state.is_high_risk_context()

    def test_high_risk_context_high_stress(self):
        """Test high risk detection for high stress."""
        state = SituationalState(stress_level=StressLevel.HIGH)
        assert state.is_high_risk_context()

    def test_high_risk_context_threat_cues(self):
        """Test high risk detection for threat cues."""
        state = SituationalState(threat_cues=["raised_voices"])
        assert state.is_high_risk_context()

    def test_normal_context(self):
        """Test normal (not high risk) context."""
        state = SituationalState(
            motion_state=MotionState.STATIONARY,
            location_context=LocationContext.HOME,
            stress_level=StressLevel.LOW,
        )
        assert not state.is_high_risk_context()


class TestSensorConfig:
    """Tests for SensorConfig."""

    def test_default_config(self):
        """Test default sensor configuration."""
        config = SensorConfig()

        assert config.location_update_interval == 10.0
        assert config.motion_update_interval == 1.0
        assert config.temporal_update_interval == 60.0
        assert config.audio_update_interval == 5.0

    def test_custom_config(self):
        """Test custom sensor configuration."""
        config = SensorConfig(
            location_update_interval=5.0,
            walking_speed_threshold_mps=0.8,
        )

        assert config.location_update_interval == 5.0
        assert config.walking_speed_threshold_mps == 0.8


# ============================================================================
# Location Sensor Tests
# ============================================================================


class TestJurisdictionDetector:
    """Tests for JurisdictionDetector."""

    def test_detect_us_california(self):
        """Test detecting California jurisdiction."""
        from src.sensors.location import JurisdictionDetector

        detector = JurisdictionDetector()
        loc = GeoLocation(latitude=37.7749, longitude=-122.4194)  # San Francisco

        jur = detector.detect(loc)

        assert jur is not None
        assert jur.country_code == "US"
        assert jur.state_code == "CA"

    def test_detect_us_texas(self):
        """Test detecting Texas jurisdiction."""
        from src.sensors.location import JurisdictionDetector

        detector = JurisdictionDetector()
        loc = GeoLocation(latitude=30.2672, longitude=-97.7431)  # Austin

        jur = detector.detect(loc)

        assert jur is not None
        assert jur.country_code == "US"
        assert jur.state_code == "TX"

    def test_detect_outside_bounds(self):
        """Test detection for location outside known bounds."""
        from src.sensors.location import JurisdictionDetector

        detector = JurisdictionDetector()
        loc = GeoLocation(latitude=0.0, longitude=0.0)  # Atlantic Ocean

        jur = detector.detect(loc)

        assert jur is None


class TestGeofenceManager:
    """Tests for GeofenceManager."""

    def test_add_geofence(self):
        """Test adding a geofence."""
        from src.sensors.location import GeofenceManager

        manager = GeofenceManager()
        gf = Geofence(
            id="home",
            name="Home",
            latitude=37.7749,
            longitude=-122.4194,
            radius_meters=100.0,
        )

        manager.add_geofence(gf)

        assert "home" in manager.geofences

    def test_check_location_inside(self):
        """Test checking location inside geofence."""
        from src.sensors.location import GeofenceManager

        manager = GeofenceManager()
        gf = Geofence(
            id="home",
            name="Home",
            latitude=37.7749,
            longitude=-122.4194,
            radius_meters=1000.0,
        )
        manager.add_geofence(gf)

        loc = GeoLocation(latitude=37.7749, longitude=-122.4194)
        active_gf, events = manager.check_location(loc)

        assert active_gf is not None
        assert active_gf.id == "home"
        assert len(events) == 1  # Enter event
        assert events[0].event_type == SensorEventType.GEOFENCE_ENTER

    def test_check_location_exit(self):
        """Test geofence exit detection."""
        from src.sensors.location import GeofenceManager

        manager = GeofenceManager()
        gf = Geofence(
            id="home",
            name="Home",
            latitude=37.7749,
            longitude=-122.4194,
            radius_meters=100.0,
        )
        manager.add_geofence(gf)

        # First, enter the geofence
        loc_inside = GeoLocation(latitude=37.7749, longitude=-122.4194)
        manager.check_location(loc_inside)

        # Then exit
        loc_outside = GeoLocation(latitude=38.0, longitude=-122.0)
        active_gf, events = manager.check_location(loc_outside)

        assert active_gf is None
        assert len(events) == 1  # Exit event
        assert events[0].event_type == SensorEventType.GEOFENCE_EXIT


class TestLocationSensor:
    """Tests for LocationSensor."""

    @pytest.mark.asyncio
    async def test_sensor_initialization(self):
        """Test location sensor initialization."""
        from src.sensors.location import LocationSensor, SimulatedGPSProvider

        sensor = LocationSensor(
            gps_provider=SimulatedGPSProvider(latitude=37.7749, longitude=-122.4194)
        )

        success = await sensor.initialize()
        assert success
        assert sensor.status == SensorStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_sensor_read(self):
        """Test taking a location reading."""
        from src.sensors.location import LocationSensor, SimulatedGPSProvider

        sensor = LocationSensor(
            gps_provider=SimulatedGPSProvider(latitude=37.7749, longitude=-122.4194)
        )
        await sensor.initialize()

        reading = await sensor.read()

        assert reading.sensor_type == SensorType.LOCATION
        assert reading.data is not None
        assert reading.data.location.latitude == pytest.approx(37.7749, rel=0.01)


class TestHaversineDistance:
    """Tests for distance calculation."""

    def test_same_location(self):
        """Test distance between same location is zero."""
        from src.sensors.location import haversine_distance

        loc = GeoLocation(latitude=37.7749, longitude=-122.4194)
        distance = haversine_distance(loc, loc)

        assert distance == pytest.approx(0.0, abs=0.001)

    def test_known_distance(self):
        """Test distance calculation for known locations."""
        from src.sensors.location import haversine_distance

        # San Francisco to Los Angeles is approximately 559 km
        sf = GeoLocation(latitude=37.7749, longitude=-122.4194)
        la = GeoLocation(latitude=34.0522, longitude=-118.2437)

        distance = haversine_distance(sf, la)

        assert 550 < distance < 570  # Allow some tolerance


# ============================================================================
# Motion Sensor Tests
# ============================================================================


class TestMotionStateClassifier:
    """Tests for MotionStateClassifier."""

    def test_stationary_by_speed(self):
        """Test classifying stationary state by speed."""
        from src.sensors.motion import MotionStateClassifier

        classifier = MotionStateClassifier()
        state, confidence = classifier.classify(speed_mps=0.0)

        assert state == MotionState.STATIONARY
        assert confidence == Confidence.HIGH

    def test_walking_by_speed(self):
        """Test classifying walking state by speed."""
        from src.sensors.motion import MotionStateClassifier

        classifier = MotionStateClassifier()
        state, confidence = classifier.classify(speed_mps=1.5)

        assert state == MotionState.WALKING

    def test_driving_by_speed(self):
        """Test classifying driving state by speed."""
        from src.sensors.motion import MotionStateClassifier

        classifier = MotionStateClassifier()
        state, confidence = classifier.classify(speed_mps=20.0)

        assert state == MotionState.DRIVING
        assert confidence == Confidence.HIGH


class TestAccidentDetector:
    """Tests for AccidentDetector."""

    def test_no_accident_normal_conditions(self):
        """Test no accident detected under normal conditions."""
        from src.sensors.motion import AccidentDetector, AccelerometerReading

        detector = AccidentDetector()

        # Add some normal readings
        for _ in range(10):
            accel = AccelerometerReading(x=0.1, y=0.1, z=9.81)
            detector.add_reading(accel=accel, speed_mps=10.0)

        is_decel, is_impact = detector.check()

        assert not is_decel
        assert not is_impact


class TestMotionSensor:
    """Tests for MotionSensor."""

    @pytest.mark.asyncio
    async def test_sensor_initialization(self):
        """Test motion sensor initialization."""
        from src.sensors.motion import MotionSensor, SimulatedAccelerometerProvider

        sensor = MotionSensor(
            accelerometer_provider=SimulatedAccelerometerProvider()
        )

        success = await sensor.initialize()
        assert success
        assert sensor.status == SensorStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_sensor_read(self):
        """Test taking a motion reading."""
        from src.sensors.motion import MotionSensor, SimulatedAccelerometerProvider

        sensor = MotionSensor(
            accelerometer_provider=SimulatedAccelerometerProvider(
                base_state=MotionState.STATIONARY
            )
        )
        await sensor.initialize()

        reading = await sensor.read()

        assert reading.sensor_type == SensorType.MOTION
        assert reading.data is not None

    @pytest.mark.asyncio
    async def test_speed_integration(self):
        """Test integrating speed from external source."""
        from src.sensors.motion import MotionSensor, SimulatedAccelerometerProvider

        sensor = MotionSensor(
            accelerometer_provider=SimulatedAccelerometerProvider()
        )
        await sensor.initialize()

        sensor.set_speed(15.0)  # Set speed from GPS

        reading = await sensor.read()
        assert reading.data.speed_mps == 15.0


# ============================================================================
# Temporal Sensor Tests
# ============================================================================


class TestTimeContext:
    """Tests for time context functions."""

    def test_get_time_context_morning(self):
        """Test getting morning time context."""
        from src.sensors.temporal import get_time_context

        context = get_time_context(9)
        assert context == TimeContext.MORNING

    def test_get_time_context_late_night(self):
        """Test getting late night time context."""
        from src.sensors.temporal import get_time_context

        context = get_time_context(2)
        assert context == TimeContext.LATE_NIGHT

    def test_get_time_context_evening(self):
        """Test getting evening time context."""
        from src.sensors.temporal import get_time_context

        context = get_time_context(19)
        assert context == TimeContext.EVENING


class TestIsWeekend:
    """Tests for weekend detection."""

    def test_saturday_is_weekend(self):
        """Test that Saturday is detected as weekend."""
        from src.sensors.temporal import is_weekend

        # Create a datetime for a Saturday
        sat = datetime(2024, 1, 6)  # January 6, 2024 was a Saturday
        assert is_weekend(sat)

    def test_monday_not_weekend(self):
        """Test that Monday is not weekend."""
        from src.sensors.temporal import is_weekend

        mon = datetime(2024, 1, 8)  # January 8, 2024 was a Monday
        assert not is_weekend(mon)


class TestIsHoliday:
    """Tests for US holiday detection."""

    def test_christmas_is_holiday(self):
        """Test that Christmas is detected as holiday."""
        from src.sensors.temporal import is_us_holiday
        from datetime import date

        is_holiday, name = is_us_holiday(date(2024, 12, 25))
        assert is_holiday
        assert name == "Christmas Day"

    def test_regular_day_not_holiday(self):
        """Test that regular day is not a holiday."""
        from src.sensors.temporal import is_us_holiday
        from datetime import date

        is_holiday, name = is_us_holiday(date(2024, 3, 15))
        assert not is_holiday
        assert name is None


class TestTemporalSensor:
    """Tests for TemporalSensor."""

    @pytest.mark.asyncio
    async def test_sensor_initialization(self):
        """Test temporal sensor initialization."""
        from src.sensors.temporal import TemporalSensor, SimulatedCalendarProvider

        sensor = TemporalSensor(
            calendar_provider=SimulatedCalendarProvider()
        )

        success = await sensor.initialize()
        assert success
        assert sensor.status == SensorStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_sensor_read(self):
        """Test taking a temporal reading."""
        from src.sensors.temporal import TemporalSensor, SimulatedCalendarProvider

        sensor = TemporalSensor(
            calendar_provider=SimulatedCalendarProvider(),
            timezone="America/Los_Angeles",
        )
        await sensor.initialize()

        reading = await sensor.read()

        assert reading.sensor_type == SensorType.TEMPORAL
        assert reading.data is not None
        assert reading.confidence == Confidence.VERY_HIGH


class TestCalendarEvent:
    """Tests for CalendarEvent."""

    def test_event_is_active(self):
        """Test checking if event is active."""
        from src.sensors.temporal import CalendarEvent

        now = datetime.now()
        event = CalendarEvent(
            id="test",
            title="Test Event",
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(hours=1),
        )

        assert event.is_active(now)

    def test_event_is_upcoming(self):
        """Test checking if event is upcoming."""
        from src.sensors.temporal import CalendarEvent

        now = datetime.now()
        event = CalendarEvent(
            id="test",
            title="Test Event",
            start_time=now + timedelta(minutes=15),
            end_time=now + timedelta(hours=1),
        )

        assert event.is_upcoming(30, now)
        assert not event.is_upcoming(10, now)


# ============================================================================
# Audio Sensor Tests
# ============================================================================


class TestAudioFeatureExtractor:
    """Tests for AudioFeatureExtractor."""

    def test_extract_features(self):
        """Test extracting features from audio sample."""
        from src.sensors.audio import AudioFeatureExtractor, AudioSample

        extractor = AudioFeatureExtractor()

        # Create a simple test sample
        sample = AudioSample(
            data=np.random.randn(16000).astype(np.float32) * 0.1,
            sample_rate=16000,
        )

        features = extractor.extract(sample)

        assert features.rms_energy > 0
        assert features.peak_amplitude > 0
        assert features.zero_crossing_rate >= 0


class TestAudioEnvironmentClassifier:
    """Tests for AudioEnvironmentClassifier."""

    def test_classify_quiet(self):
        """Test classifying quiet environment."""
        from src.sensors.audio import AudioEnvironmentClassifier, AudioFeatures

        classifier = AudioEnvironmentClassifier()
        features = AudioFeatures(
            rms_energy=0.0001,
            peak_amplitude=0.001,
            zero_crossing_rate=0.01,
            noise_level_db=-60,
        )

        env, confidence = classifier.classify(features)
        assert env == AudioEnvironment.QUIET


class TestVoiceStressDetector:
    """Tests for VoiceStressDetector."""

    def test_no_stress_without_speech(self):
        """Test no stress detection when no speech."""
        from src.sensors.audio import VoiceStressDetector, AudioFeatures

        detector = VoiceStressDetector()
        features = AudioFeatures(
            rms_energy=0.1,
            peak_amplitude=0.3,
            zero_crossing_rate=0.05,
            has_speech=False,
        )

        stress = detector.detect(features)
        assert stress == StressLevel.UNKNOWN


class TestAudioSensor:
    """Tests for AudioSensor."""

    @pytest.mark.asyncio
    async def test_sensor_initialization(self):
        """Test audio sensor initialization."""
        from src.sensors.audio import AudioSensor, SimulatedAudioProvider

        sensor = AudioSensor(
            audio_provider=SimulatedAudioProvider()
        )

        success = await sensor.initialize()
        assert success
        assert sensor.status == SensorStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_sensor_read(self):
        """Test taking an audio reading."""
        from src.sensors.audio import AudioSensor, SimulatedAudioProvider

        sensor = AudioSensor(
            audio_provider=SimulatedAudioProvider(
                environment=AudioEnvironment.QUIET
            )
        )
        await sensor.initialize()

        reading = await sensor.read()

        assert reading.sensor_type == SensorType.AUDIO
        assert reading.data is not None


# ============================================================================
# Sensor Fusion Tests
# ============================================================================


class TestRiskAssessor:
    """Tests for RiskAssessor."""

    def test_assess_low_risk(self):
        """Test risk assessment for low risk scenario."""
        from src.sensors.fusion import RiskAssessor

        assessor = RiskAssessor()
        state = SituationalState(
            motion_state=MotionState.STATIONARY,
            location_context=LocationContext.HOME,
            stress_level=StressLevel.LOW,
        )

        score, factors = assessor.assess(state)
        assert score == 0
        assert len(factors) == 0

    def test_assess_high_risk(self):
        """Test risk assessment for high risk scenario."""
        from src.sensors.fusion import RiskAssessor

        assessor = RiskAssessor()
        state = SituationalState(
            is_late_night=True,
            location_context=LocationContext.UNFAMILIAR,
            stress_level=StressLevel.HIGH,
            threat_cues=["raised_voices"],
        )

        score, factors = assessor.assess(state)
        assert score >= 6  # Multiple risk factors
        assert len(factors) >= 3


class TestSensorFusionManager:
    """Tests for SensorFusionManager."""

    @pytest.mark.asyncio
    async def test_manager_creation(self):
        """Test creating sensor fusion manager."""
        from src.sensors.fusion import SensorFusionManager

        manager = SensorFusionManager()
        assert manager.situational_state is not None
        assert len(manager.sensors) == 0

    @pytest.mark.asyncio
    async def test_add_sensor(self):
        """Test adding sensor to manager."""
        from src.sensors.fusion import SensorFusionManager
        from src.sensors.location import LocationSensor, SimulatedGPSProvider

        manager = SensorFusionManager()
        sensor = LocationSensor(gps_provider=SimulatedGPSProvider())

        manager.add_sensor(sensor)

        assert SensorType.LOCATION in manager.sensors

    @pytest.mark.asyncio
    async def test_manager_initialization(self):
        """Test initializing fusion manager."""
        from src.sensors.fusion import SensorFusionManager
        from src.sensors.location import LocationSensor, SimulatedGPSProvider
        from src.sensors.temporal import TemporalSensor, SimulatedCalendarProvider

        manager = SensorFusionManager()
        manager.add_sensor(LocationSensor(gps_provider=SimulatedGPSProvider()))
        manager.add_sensor(TemporalSensor(calendar_provider=SimulatedCalendarProvider()))

        success = await manager.initialize()
        assert success

    @pytest.mark.asyncio
    async def test_read_all_sensors(self):
        """Test reading from all sensors."""
        from src.sensors.fusion import SensorFusionManager
        from src.sensors.location import LocationSensor, SimulatedGPSProvider
        from src.sensors.motion import MotionSensor, SimulatedAccelerometerProvider

        manager = SensorFusionManager()
        manager.add_sensor(LocationSensor(gps_provider=SimulatedGPSProvider()))
        manager.add_sensor(MotionSensor(accelerometer_provider=SimulatedAccelerometerProvider()))

        await manager.initialize()
        readings = await manager.read_all()

        assert SensorType.LOCATION in readings
        assert SensorType.MOTION in readings

    @pytest.mark.asyncio
    async def test_state_callback(self):
        """Test state update callback."""
        from src.sensors.fusion import SensorFusionManager
        from src.sensors.temporal import TemporalSensor, SimulatedCalendarProvider

        manager = SensorFusionManager()
        manager.add_sensor(TemporalSensor(calendar_provider=SimulatedCalendarProvider()))

        state_updates = []

        def on_state_update(state: SituationalState):
            state_updates.append(state)

        manager.register_state_callback(on_state_update)
        await manager.initialize()

        # Manually trigger state update
        await manager._update_situational_state()

        assert len(state_updates) >= 1

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting fusion manager stats."""
        from src.sensors.fusion import SensorFusionManager
        from src.sensors.temporal import TemporalSensor, SimulatedCalendarProvider

        manager = SensorFusionManager()
        manager.add_sensor(TemporalSensor(calendar_provider=SimulatedCalendarProvider()))

        await manager.initialize()
        stats = manager.get_stats()

        assert "running" in stats
        assert "sensor_count" in stats
        assert stats["sensor_count"] == 1


class TestCreateDefaultFusionManager:
    """Tests for create_default_fusion_manager factory."""

    @pytest.mark.asyncio
    async def test_create_with_simulated(self):
        """Test creating fusion manager with simulated providers."""
        from src.sensors.fusion import create_default_fusion_manager

        manager = create_default_fusion_manager(use_simulated=True)

        assert SensorType.LOCATION in manager.sensors
        assert SensorType.MOTION in manager.sensors
        assert SensorType.TEMPORAL in manager.sensors
        assert SensorType.AUDIO in manager.sensors

    @pytest.mark.asyncio
    async def test_initialize_default_manager(self):
        """Test initializing default fusion manager."""
        from src.sensors.fusion import create_default_fusion_manager

        manager = create_default_fusion_manager(use_simulated=True)
        success = await manager.initialize()

        assert success


# ============================================================================
# Integration Tests
# ============================================================================


class TestSensorIntegration:
    """Integration tests for sensor fusion system."""

    @pytest.mark.asyncio
    async def test_full_sensor_fusion_cycle(self):
        """Test full sensor fusion cycle."""
        from src.sensors.fusion import create_default_fusion_manager

        manager = create_default_fusion_manager(use_simulated=True)

        # Initialize
        await manager.initialize()

        # Read all sensors
        readings = await manager.read_all()
        assert len(readings) == 4  # All 4 sensor types

        # Update state
        await manager._update_situational_state()

        # Check state is populated
        state = manager.situational_state
        assert state.timestamp is not None

    @pytest.mark.asyncio
    async def test_context_manager_usage(self):
        """Test using fusion manager as async context manager."""
        from src.sensors.fusion import SensorFusionManager
        from src.sensors.temporal import TemporalSensor, SimulatedCalendarProvider

        manager = SensorFusionManager()
        manager.add_sensor(TemporalSensor(calendar_provider=SimulatedCalendarProvider()))

        # Just test initialization (not full start to avoid long-running tasks)
        await manager.initialize()
        assert manager.situational_state is not None

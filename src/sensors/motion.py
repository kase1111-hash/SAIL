"""
Motion sensor implementation.

Provides motion state detection (stationary/walking/driving),
speed calculation, and accident detection.
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional
from math import sqrt

from .base import (
    Sensor,
    SensorType,
    SensorStatus,
    SensorReading,
    SensorConfig,
    SensorEvent,
    SensorEventType,
    MotionState,
    MotionReading,
    Confidence,
    GeoLocation,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Motion Data Types
# ============================================================================


@dataclass
class AccelerometerReading:
    """Raw accelerometer reading."""

    x: float  # m/s²
    y: float  # m/s²
    z: float  # m/s²
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def magnitude(self) -> float:
        """Calculate acceleration magnitude."""
        return sqrt(self.x**2 + self.y**2 + self.z**2)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "magnitude": self.magnitude,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MotionAnalysis:
    """Analyzed motion data."""

    state: MotionState
    confidence: Confidence
    speed_mps: Optional[float] = None
    acceleration_magnitude: Optional[float] = None
    is_sudden_deceleration: bool = False
    is_impact_detected: bool = False
    state_duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state": self.state.value,
            "confidence": self.confidence.value,
            "speed_mps": self.speed_mps,
            "acceleration_magnitude": self.acceleration_magnitude,
            "is_sudden_deceleration": self.is_sudden_deceleration,
            "is_impact_detected": self.is_impact_detected,
            "state_duration_seconds": self.state_duration_seconds,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# Accelerometer Provider Abstraction
# ============================================================================


class AccelerometerProvider:
    """Abstract accelerometer data provider."""

    async def get_reading(self) -> Optional[AccelerometerReading]:
        """Get current accelerometer reading."""
        raise NotImplementedError


class SimulatedAccelerometerProvider(AccelerometerProvider):
    """Simulated accelerometer for testing."""

    def __init__(self, base_state: MotionState = MotionState.STATIONARY):
        self.base_state = base_state
        self._noise_level = 0.1

    async def get_reading(self) -> Optional[AccelerometerReading]:
        """Get simulated accelerometer reading."""
        import random

        # Base gravity reading (pointing down)
        x = random.gauss(0, self._noise_level)
        y = random.gauss(0, self._noise_level)
        z = 9.81 + random.gauss(0, self._noise_level)

        # Add motion-based variations
        if self.base_state == MotionState.WALKING:
            # Walking produces rhythmic vertical acceleration
            y += random.gauss(0, 0.5)
            z += random.gauss(0, 1.0)
        elif self.base_state == MotionState.RUNNING:
            y += random.gauss(0, 1.0)
            z += random.gauss(0, 2.0)
        elif self.base_state == MotionState.DRIVING:
            # Driving produces smoother, horizontal accelerations
            x += random.gauss(0, 0.3)
            y += random.gauss(0, 0.3)

        return AccelerometerReading(x=x, y=y, z=z)


class SystemAccelerometerProvider(AccelerometerProvider):
    """System accelerometer provider using platform APIs."""

    def __init__(self):
        self._available = False

    async def initialize(self) -> bool:
        """Initialize accelerometer access."""
        # Platform-specific initialization would go here
        # For example, using Android's SensorManager or iOS CoreMotion
        logger.info("System accelerometer initialization (placeholder)")
        return False

    async def get_reading(self) -> Optional[AccelerometerReading]:
        """Get accelerometer reading from system."""
        if not self._available:
            return None
        # Platform-specific reading would go here
        return None


# ============================================================================
# Motion State Classifier
# ============================================================================


class MotionStateClassifier:
    """Classifies motion state from sensor data."""

    def __init__(self, config: Optional[SensorConfig] = None):
        self.config = config or SensorConfig()

        # Thresholds
        self.walking_speed_threshold = self.config.walking_speed_threshold_mps
        self.running_speed_threshold = self.config.running_speed_threshold_mps
        self.driving_speed_threshold = self.config.driving_speed_threshold_mps

        # Accelerometer thresholds
        self.stationary_accel_threshold = 0.3  # m/s² variance
        self.walking_accel_range = (0.5, 3.0)  # m/s² deviation from gravity
        self.running_accel_range = (2.0, 6.0)
        self.driving_accel_range = (0.2, 1.5)

        # History for smoothing
        self._accel_history: deque[AccelerometerReading] = deque(maxlen=50)
        self._speed_history: deque[float] = deque(maxlen=20)
        self._state_history: deque[MotionState] = deque(maxlen=10)

    def add_accelerometer_reading(self, reading: AccelerometerReading) -> None:
        """Add an accelerometer reading to history."""
        self._accel_history.append(reading)

    def add_speed_reading(self, speed_mps: float) -> None:
        """Add a speed reading to history."""
        self._speed_history.append(speed_mps)

    def classify(
        self,
        speed_mps: Optional[float] = None,
        accel: Optional[AccelerometerReading] = None,
    ) -> tuple[MotionState, Confidence]:
        """
        Classify current motion state.
        Returns (state, confidence).
        """
        # Add readings to history
        if speed_mps is not None:
            self.add_speed_reading(speed_mps)
        if accel is not None:
            self.add_accelerometer_reading(accel)

        # Classify based on available data
        state = MotionState.UNKNOWN
        confidence = Confidence.LOW

        # Speed-based classification (highest priority if available)
        if speed_mps is not None:
            state, confidence = self._classify_by_speed(speed_mps)
        elif self._speed_history:
            avg_speed = sum(self._speed_history) / len(self._speed_history)
            state, confidence = self._classify_by_speed(avg_speed)

        # Accelerometer-based classification (use if no speed or to boost confidence)
        if accel is not None or self._accel_history:
            accel_state, accel_conf = self._classify_by_acceleration()

            if state == MotionState.UNKNOWN:
                state = accel_state
                confidence = accel_conf
            elif accel_state == state:
                # Both agree, boost confidence
                if confidence == Confidence.LOW:
                    confidence = Confidence.MEDIUM
                elif confidence == Confidence.MEDIUM:
                    confidence = Confidence.HIGH

        # Apply temporal smoothing
        self._state_history.append(state)
        smoothed_state = self._smooth_state()

        return smoothed_state, confidence

    def _classify_by_speed(self, speed_mps: float) -> tuple[MotionState, Confidence]:
        """Classify motion state by speed."""
        if speed_mps < self.walking_speed_threshold:
            return MotionState.STATIONARY, Confidence.HIGH
        elif speed_mps < self.running_speed_threshold:
            return MotionState.WALKING, Confidence.MEDIUM
        elif speed_mps < self.driving_speed_threshold:
            return MotionState.RUNNING, Confidence.MEDIUM
        else:
            return MotionState.DRIVING, Confidence.HIGH

    def _classify_by_acceleration(self) -> tuple[MotionState, Confidence]:
        """Classify motion state by accelerometer data."""
        if not self._accel_history:
            return MotionState.UNKNOWN, Confidence.VERY_LOW

        # Calculate acceleration statistics
        magnitudes = [r.magnitude for r in self._accel_history]
        avg_mag = sum(magnitudes) / len(magnitudes)
        variance = sum((m - avg_mag) ** 2 for m in magnitudes) / len(magnitudes)
        deviation_from_gravity = abs(avg_mag - 9.81)

        # Classify based on patterns
        if variance < self.stationary_accel_threshold:
            return MotionState.STATIONARY, Confidence.MEDIUM
        elif self.walking_accel_range[0] <= variance <= self.walking_accel_range[1]:
            return MotionState.WALKING, Confidence.LOW
        elif self.running_accel_range[0] <= variance <= self.running_accel_range[1]:
            return MotionState.RUNNING, Confidence.LOW
        elif self.driving_accel_range[0] <= variance <= self.driving_accel_range[1]:
            # Driving typically has low variance but occasional horizontal acceleration
            return MotionState.DRIVING, Confidence.LOW

        return MotionState.UNKNOWN, Confidence.VERY_LOW

    def _smooth_state(self) -> MotionState:
        """Apply temporal smoothing to reduce noise."""
        if not self._state_history:
            return MotionState.UNKNOWN

        # Count occurrences of each state
        state_counts: dict[MotionState, int] = {}
        for state in self._state_history:
            state_counts[state] = state_counts.get(state, 0) + 1

        # Return most common state
        return max(state_counts, key=lambda s: state_counts[s])


# ============================================================================
# Accident Detector
# ============================================================================


class AccidentDetector:
    """Detects potential accidents from motion data."""

    def __init__(self):
        # Detection thresholds
        self.sudden_decel_threshold = 15.0  # m/s² (roughly 1.5g)
        self.impact_threshold = 30.0  # m/s² (roughly 3g)
        self.speed_drop_threshold = 10.0  # m/s speed drop in short time

        # History
        self._accel_history: deque[AccelerometerReading] = deque(maxlen=100)
        self._speed_history: deque[tuple[datetime, float]] = deque(maxlen=50)

        # Detection state
        self._potential_accident = False
        self._accident_timestamp: Optional[datetime] = None

    def add_reading(
        self,
        accel: Optional[AccelerometerReading] = None,
        speed_mps: Optional[float] = None,
    ) -> None:
        """Add sensor readings."""
        if accel:
            self._accel_history.append(accel)
        if speed_mps is not None:
            self._speed_history.append((datetime.now(), speed_mps))

    def check(self) -> tuple[bool, bool]:
        """
        Check for accident indicators.
        Returns (is_sudden_deceleration, is_impact_detected).
        """
        is_sudden_decel = self._check_sudden_deceleration()
        is_impact = self._check_impact()

        return is_sudden_decel, is_impact

    def _check_sudden_deceleration(self) -> bool:
        """Check for sudden deceleration patterns."""
        if len(self._speed_history) < 2:
            return False

        # Check recent speed changes
        recent = list(self._speed_history)[-10:]
        if len(recent) < 2:
            return False

        for i in range(1, len(recent)):
            time_diff = (recent[i][0] - recent[i-1][0]).total_seconds()
            if time_diff > 0:
                speed_diff = recent[i-1][1] - recent[i][1]  # Positive if decelerating
                decel = speed_diff / time_diff

                if decel > self.sudden_decel_threshold:
                    return True

        return False

    def _check_impact(self) -> bool:
        """Check for impact patterns in accelerometer data."""
        if not self._accel_history:
            return False

        # Check recent acceleration magnitudes
        recent = list(self._accel_history)[-20:]

        for reading in recent:
            # Subtract gravity to get dynamic acceleration
            dynamic_accel = abs(reading.magnitude - 9.81)
            if dynamic_accel > self.impact_threshold:
                return True

        return False


# ============================================================================
# Motion Sensor
# ============================================================================


class MotionSensor(Sensor):
    """Motion sensor with state classification and accident detection."""

    def __init__(
        self,
        sensor_id: str = "motion_primary",
        config: Optional[SensorConfig] = None,
        accelerometer_provider: Optional[AccelerometerProvider] = None,
    ):
        super().__init__(sensor_id, SensorType.MOTION, config)

        self.accelerometer = accelerometer_provider or SimulatedAccelerometerProvider()
        self.classifier = MotionStateClassifier(config)
        self.accident_detector = AccidentDetector()

        self._current_state = MotionState.UNKNOWN
        self._state_start_time: Optional[datetime] = None
        self._last_speed: Optional[float] = None
        self._event_callbacks: list[Callable[[SensorEvent], None]] = []

    def set_speed(self, speed_mps: float) -> None:
        """Set current speed from external source (e.g., GPS)."""
        self._last_speed = speed_mps
        self.classifier.add_speed_reading(speed_mps)
        self.accident_detector.add_reading(speed_mps=speed_mps)

    def register_event_callback(self, callback: Callable[[SensorEvent], None]) -> None:
        """Register callback for motion events."""
        self._event_callbacks.append(callback)

    async def initialize(self) -> bool:
        """Initialize the motion sensor."""
        self._status = SensorStatus.INITIALIZING

        try:
            # Initialize accelerometer if it has an initialize method
            if hasattr(self.accelerometer, "initialize"):
                await self.accelerometer.initialize()

            self._status = SensorStatus.ACTIVE
            self._state_start_time = datetime.now()
            logger.info(f"Motion sensor {self.sensor_id} initialized")
            return True

        except Exception as e:
            self._set_error(f"Initialization failed: {e}")
            return False

    async def read(self) -> SensorReading:
        """Take a motion reading."""
        # Get accelerometer data
        accel = await self.accelerometer.get_reading()

        if accel:
            self.accident_detector.add_reading(accel=accel)

        # Classify motion state
        state, confidence = self.classifier.classify(
            speed_mps=self._last_speed,
            accel=accel,
        )

        # Check for state change
        if state != self._current_state:
            old_state = self._current_state
            self._current_state = state
            self._state_start_time = datetime.now()

            # Emit state change event
            event = SensorEvent(
                event_type=SensorEventType.MOTION_CHANGE,
                sensor_type=SensorType.MOTION,
                data={
                    "previous_state": old_state,
                    "new_state": state,
                },
            )
            self._notify_event(event)

        # Calculate state duration
        state_duration = 0.0
        if self._state_start_time:
            state_duration = (datetime.now() - self._state_start_time).total_seconds()

        # Check for accidents
        is_sudden_decel, is_impact = self.accident_detector.check()

        # Create motion analysis
        analysis = MotionAnalysis(
            state=state,
            confidence=confidence,
            speed_mps=self._last_speed,
            acceleration_magnitude=accel.magnitude if accel else None,
            is_sudden_deceleration=is_sudden_decel,
            is_impact_detected=is_impact,
            state_duration_seconds=state_duration,
        )

        # Create motion reading for the sensor reading
        motion_reading = MotionReading(
            state=state,
            confidence=confidence,
            speed_mps=self._last_speed,
            acceleration=(accel.x, accel.y, accel.z) if accel else None,
        )

        reading = SensorReading(
            sensor_type=self.sensor_type,
            sensor_id=self.sensor_id,
            data=analysis,
            confidence=confidence,
            metadata={"motion_reading": motion_reading.to_dict()},
        )

        self._set_reading(reading)
        return reading

    def _notify_event(self, event: SensorEvent) -> None:
        """Notify event callbacks."""
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    async def start(self) -> None:
        """Start continuous motion monitoring."""
        self._running = True
        self._status = SensorStatus.ACTIVE

        while self._running:
            try:
                await self.read()
                await asyncio.sleep(self.config.motion_update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Motion sensor error: {e}")
                await asyncio.sleep(1.0)

    async def stop(self) -> None:
        """Stop motion monitoring."""
        self._running = False
        self._status = SensorStatus.INACTIVE
        logger.info(f"Motion sensor {self.sensor_id} stopped")

"""
Sensor fusion manager implementation.

Combines data from all sensors into unified situational awareness,
correlating location, motion, temporal, and audio data.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.core.events import EventEmitter

from .audio import AudioAnalysis, AudioSensor
from .base import (
    Confidence,
    LocationContext,
    MotionState,
    Sensor,
    SensorConfig,
    SensorEvent,
    SensorReading,
    SensorState,
    SensorType,
    SituationalState,
    StressLevel,
)
from .location import LocationReading, LocationSensor
from .motion import MotionAnalysis, MotionSensor
from .temporal import TemporalAnalysis, TemporalSensor

logger = logging.getLogger(__name__)

# Speed above which driving is treated as a risk factor. Mirrored by
# HIGH_SPEED_KMH in src.intervention.risk, which scores the same signal for the
# intervention engine; a test asserts the two stay equal.
HIGH_SPEED_KMH = 120.0


# ============================================================================
# Fusion Event Types
# ============================================================================


class FusionEventType(str):
    """Types of fusion events."""

    STATE_UPDATE = "state_update"
    RISK_ELEVATED = "risk_elevated"
    CONTEXT_CHANGE = "context_change"
    ANOMALY_DETECTED = "anomaly_detected"


@dataclass
class FusionEvent:
    """Event from sensor fusion system."""

    event_type: str
    situational_state: SituationalState
    trigger: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type,
            "situational_state": self.situational_state.to_dict(),
            "trigger": self.trigger,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# Risk Assessment
# ============================================================================


class RiskAssessor:
    """Assesses risk level from situational state."""

    def __init__(self):
        self.risk_weights = {
            "late_night_unfamiliar": 3,
            "high_stress": 3,
            "threat_cues": 4,
            "driving_late_night": 2,
            "driving_high_speed": 2,
            "unfamiliar_area": 1,
            "elevated_stress": 1,
        }

    def assess(self, state: SituationalState) -> tuple[int, list[str]]:
        """
        Assess risk from situational state.
        Returns (risk_score, list_of_factors).
        """
        risk_score = 0
        factors: list[str] = []

        # Late night in unfamiliar area
        if state.is_late_night and state.location_context == LocationContext.UNFAMILIAR:
            risk_score += self.risk_weights["late_night_unfamiliar"]
            factors.append("Late night in unfamiliar area")

        # High stress detected
        if state.stress_level == StressLevel.HIGH:
            risk_score += self.risk_weights["high_stress"]
            factors.append("High stress level detected")
        elif state.stress_level == StressLevel.ELEVATED:
            risk_score += self.risk_weights["elevated_stress"]
            factors.append("Elevated stress level")

        # Threat cues present
        if state.threat_cues:
            risk_score += self.risk_weights["threat_cues"]
            factors.append(f"Threat cues: {', '.join(state.threat_cues)}")

        # Driving late at night
        if state.motion_state == MotionState.DRIVING and state.is_late_night:
            risk_score += self.risk_weights["driving_late_night"]
            factors.append("Driving late at night")

        # High speed driving
        if state.motion_state == MotionState.DRIVING and state.speed_mps:
            speed_kmh = state.speed_mps * 3.6
            if speed_kmh > HIGH_SPEED_KMH:
                risk_score += self.risk_weights["driving_high_speed"]
                factors.append(f"High speed driving: {speed_kmh:.0f} km/h")

        # Unfamiliar area
        if state.location_context == LocationContext.UNFAMILIAR:
            risk_score += self.risk_weights["unfamiliar_area"]
            factors.append("In unfamiliar area")

        return risk_score, factors


# ============================================================================
# Sensor Fusion Manager
# ============================================================================


class SensorFusionManager(EventEmitter[FusionEvent]):
    """
    Manages all sensors and fuses their data into unified situational awareness.
    """

    def __init__(self, config: SensorConfig | None = None):
        self.config = config or SensorConfig()

        # Sensors
        self._sensors: dict[SensorType, Sensor] = {}
        self._sensor_tasks: dict[SensorType, asyncio.Task] = {}

        # State
        self._situational_state = SituationalState()
        self._running = False
        self._update_lock = asyncio.Lock()

        # Risk assessment
        self._risk_assessor = RiskAssessor()

        self.__init_emitter__()
        self._state_callbacks: list[Callable[[SituationalState], None]] = []
        self._sensor_event_callbacks: list[Callable[[SensorEvent], None]] = []

    @property
    def situational_state(self) -> SituationalState:
        """Get current situational state."""
        return self._situational_state

    @property
    def sensors(self) -> dict[SensorType, Sensor]:
        """Get all sensors."""
        return self._sensors

    def add_sensor(self, sensor: Sensor) -> None:
        """Add a sensor to the fusion manager."""
        self._sensors[sensor.sensor_type] = sensor

        # Register for sensor readings
        sensor.register_callback(self._on_sensor_reading)

        # Register for sensor events if supported
        if hasattr(sensor, "register_event_callback"):
            sensor.register_event_callback(self._on_sensor_event)

        logger.info(f"Added sensor: {sensor.sensor_type.value} ({sensor.sensor_id})")

    def remove_sensor(self, sensor_type: SensorType) -> None:
        """Remove a sensor from the fusion manager."""
        if sensor_type in self._sensors:
            del self._sensors[sensor_type]
            logger.info(f"Removed sensor: {sensor_type.value}")

    def register_state_callback(self, callback: Callable[[SituationalState], None]) -> None:
        """Register callback for state updates."""
        self._state_callbacks.append(callback)

    def register_sensor_event_callback(self, callback: Callable[[SensorEvent], None]) -> None:
        """Register callback for sensor events."""
        self._sensor_event_callbacks.append(callback)

    async def initialize(self) -> bool:
        """Initialize all sensors."""
        logger.info("Initializing sensor fusion manager...")

        success = True
        for sensor_type, sensor in self._sensors.items():
            try:
                if not await sensor.initialize():
                    logger.warning(f"Sensor {sensor_type.value} initialization returned False")
            except Exception as e:
                logger.error(f"Failed to initialize {sensor_type.value}: {e}")
                success = False

        return success

    async def start(self) -> None:
        """Start all sensors and fusion processing."""
        if self._running:
            return

        self._running = True
        logger.info("Starting sensor fusion manager...")

        # Start all sensors
        for sensor_type, sensor in self._sensors.items():
            task = asyncio.create_task(self._run_sensor(sensor))
            self._sensor_tasks[sensor_type] = task

        # Start fusion update loop
        asyncio.create_task(self._fusion_loop())

        logger.info("Sensor fusion manager started")

    async def stop(self) -> None:
        """Stop all sensors and fusion processing."""
        self._running = False
        logger.info("Stopping sensor fusion manager...")

        # Cancel sensor tasks
        for task in self._sensor_tasks.values():
            task.cancel()

        # Stop all sensors
        for sensor in self._sensors.values():
            try:
                await sensor.stop()
            except Exception as e:
                logger.error(f"Error stopping sensor: {e}")

        self._sensor_tasks.clear()
        logger.info("Sensor fusion manager stopped")

    async def _run_sensor(self, sensor: Sensor) -> None:
        """Run a sensor's monitoring loop."""
        try:
            await sensor.start()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Sensor {sensor.sensor_id} error: {e}")

    async def _fusion_loop(self) -> None:
        """Main fusion processing loop."""
        while self._running:
            try:
                await self._update_situational_state()
                await asyncio.sleep(1.0)  # Update state every second
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fusion loop error: {e}")
                await asyncio.sleep(1.0)

    def _on_sensor_reading(self, reading: SensorReading) -> None:
        """Handle incoming sensor reading."""
        # Queue the reading for processing
        asyncio.create_task(self._process_reading(reading))

    def _on_sensor_event(self, event: SensorEvent) -> None:
        """Handle sensor event."""
        # Notify sensor event callbacks
        for callback in self._sensor_event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Sensor event callback error: {e}")

    async def _process_reading(self, reading: SensorReading) -> None:
        """Process a sensor reading and update state."""
        async with self._update_lock:
            if reading.sensor_type == SensorType.LOCATION:
                self._update_location_state(reading)
            elif reading.sensor_type == SensorType.MOTION:
                self._update_motion_state(reading)
            elif reading.sensor_type == SensorType.TEMPORAL:
                self._update_temporal_state(reading)
            elif reading.sensor_type == SensorType.AUDIO:
                self._update_audio_state(reading)

    def _update_location_state(self, reading: SensorReading) -> None:
        """Update situational state from location reading."""
        if reading.data is None:
            return

        data: LocationReading = reading.data
        self._situational_state.location = data.location
        self._situational_state.jurisdiction = data.jurisdiction
        self._situational_state.location_context = data.context
        self._situational_state.in_geofence = data.active_geofence

        # Update speed in motion state
        if data.location.speed_mps is not None:
            self._situational_state.speed_mps = data.location.speed_mps

            # Also update motion sensor if present
            motion_sensor = self._sensors.get(SensorType.MOTION)
            if motion_sensor and isinstance(motion_sensor, MotionSensor):
                motion_sensor.set_speed(data.location.speed_mps)

    def _update_motion_state(self, reading: SensorReading) -> None:
        """Update situational state from motion reading."""
        if reading.data is None:
            return

        data: MotionAnalysis = reading.data
        self._situational_state.motion_state = data.state
        if data.speed_mps is not None:
            self._situational_state.speed_mps = data.speed_mps

    def _update_temporal_state(self, reading: SensorReading) -> None:
        """Update situational state from temporal reading."""
        if reading.data is None:
            return

        data: TemporalAnalysis = reading.data
        self._situational_state.time_context = data.time_context
        self._situational_state.is_weekend = data.is_weekend
        self._situational_state.is_late_night = data.is_late_night
        self._situational_state.calendar_context = data.event_context

    def _update_audio_state(self, reading: SensorReading) -> None:
        """Update situational state from audio reading."""
        if reading.data is None:
            return

        data: AudioAnalysis = reading.data
        self._situational_state.audio_environment = data.environment
        self._situational_state.stress_level = data.stress_indicators
        self._situational_state.threat_cues = data.threat_cues

    async def _update_situational_state(self) -> None:
        """Update overall situational state and assess risk."""
        async with self._update_lock:
            # Update timestamp
            self._situational_state.timestamp = datetime.now()

            # Calculate overall confidence
            self._situational_state.overall_confidence = self._calculate_confidence()

            # Assess risk
            risk_score, risk_factors = self._risk_assessor.assess(self._situational_state)
            self._situational_state.risk_factors = risk_factors

            # Notify callbacks
            self._notify_state_update()

            # Check for elevated risk
            if risk_score >= 3:
                self._emit_fusion_event(
                    FusionEventType.RISK_ELEVATED,
                    trigger="risk_threshold",
                    details={"risk_score": risk_score, "factors": risk_factors},
                )

    def _calculate_confidence(self) -> Confidence:
        """Calculate overall confidence from sensor readings."""
        confidences: list[Confidence] = []

        for sensor in self._sensors.values():
            if sensor.last_reading:
                confidences.append(sensor.last_reading.confidence)

        if not confidences:
            return Confidence.VERY_LOW

        # Map to numeric values
        conf_values = {
            Confidence.VERY_LOW: 1,
            Confidence.LOW: 2,
            Confidence.MEDIUM: 3,
            Confidence.HIGH: 4,
            Confidence.VERY_HIGH: 5,
        }

        avg = sum(conf_values[c] for c in confidences) / len(confidences)

        # Map back to confidence level
        if avg >= 4.5:
            return Confidence.VERY_HIGH
        elif avg >= 3.5:
            return Confidence.HIGH
        elif avg >= 2.5:
            return Confidence.MEDIUM
        elif avg >= 1.5:
            return Confidence.LOW
        else:
            return Confidence.VERY_LOW

    def _notify_state_update(self) -> None:
        """Notify state callbacks."""
        for callback in self._state_callbacks:
            try:
                callback(self._situational_state)
            except Exception as e:
                logger.error(f"State callback error: {e}")

    def _emit_fusion_event(
        self,
        event_type: str,
        trigger: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Emit a fusion event."""
        self._emit(FusionEvent(
            event_type=event_type,
            situational_state=self._situational_state,
            trigger=trigger,
            details=details or {},
        ))

    def get_sensor_states(self) -> dict[SensorType, SensorState]:
        """Get states of all sensors."""
        return {
            sensor_type: sensor.get_state()
            for sensor_type, sensor in self._sensors.items()
        }

    async def read_all(self) -> dict[SensorType, SensorReading]:
        """Take a reading from all sensors."""
        readings: dict[SensorType, SensorReading] = {}

        for sensor_type, sensor in self._sensors.items():
            try:
                reading = await sensor.read()
                readings[sensor_type] = reading
            except Exception as e:
                logger.error(f"Error reading {sensor_type.value}: {e}")

        return readings

    def get_stats(self) -> dict[str, Any]:
        """Get fusion manager statistics."""
        return {
            "running": self._running,
            "sensor_count": len(self._sensors),
            "sensors": {
                st.value: {
                    "status": s.status.value,
                    "last_update": s._last_update.isoformat() if s._last_update else None,
                }
                for st, s in self._sensors.items()
            },
            "situational_state": self._situational_state.to_dict(),
            "risk_factors": self._situational_state.risk_factors,
        }

    async def __aenter__(self) -> "SensorFusionManager":
        """Async context manager entry."""
        await self.initialize()
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()


# ============================================================================
# Factory Function
# ============================================================================


def create_default_fusion_manager(
    config: SensorConfig | None = None,
    use_simulated: bool = True,
) -> SensorFusionManager:
    """
    Create a sensor fusion manager with default sensors.

    Args:
        config: Sensor configuration
        use_simulated: Whether to use simulated providers (True) or system providers (False)

    Returns:
        Configured SensorFusionManager
    """
    from .audio import SimulatedAudioProvider, SystemAudioProvider
    from .location import SimulatedGPSProvider, SystemGPSProvider
    from .motion import SimulatedAccelerometerProvider, SystemAccelerometerProvider
    from .temporal import SimulatedCalendarProvider

    config = config or SensorConfig()
    manager = SensorFusionManager(config)

    # Location sensor
    if use_simulated:
        gps = SimulatedGPSProvider()
    else:
        gps = SystemGPSProvider()
    manager.add_sensor(LocationSensor(config=config, gps_provider=gps))

    # Motion sensor
    if use_simulated:
        accel = SimulatedAccelerometerProvider()
    else:
        accel = SystemAccelerometerProvider()
    manager.add_sensor(MotionSensor(config=config, accelerometer_provider=accel))

    # Temporal sensor
    calendar = SimulatedCalendarProvider()
    manager.add_sensor(TemporalSensor(config=config, calendar_provider=calendar))

    # Audio sensor
    if use_simulated:
        audio = SimulatedAudioProvider()
    else:
        audio = SystemAudioProvider()
    manager.add_sensor(AudioSensor(config=config, audio_provider=audio))

    return manager

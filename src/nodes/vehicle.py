"""
SAIL Vehicle Node

Vehicle-based node with OBD-II integration for automotive context.
Features speed monitoring, driving mode detection, accident detection,
and Bluetooth audio integration.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from pydantic import Field

from src.hub.base import NodeCapability, NodeType
from src.nodes.base import BaseNode, NodeConfig, NodeMessage, MessageType

logger = logging.getLogger(__name__)


class DrivingState(str, Enum):
    """Vehicle driving state."""

    PARKED = "parked"
    IDLING = "idling"
    MOVING = "moving"
    HIGHWAY = "highway"


class EngineState(str, Enum):
    """Engine state."""

    OFF = "off"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


class AlertLevel(str, Enum):
    """Alert severity level."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class VehicleData:
    """Real-time vehicle data from OBD-II."""

    speed_kmh: float = 0.0
    rpm: int = 0
    engine_load: float = 0.0
    coolant_temp: float = 0.0
    fuel_level: float = 0.0
    throttle_position: float = 0.0
    intake_temp: float | None = None
    runtime_seconds: int = 0
    distance_km: float = 0.0
    dtc_codes: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "speed_kmh": self.speed_kmh,
            "rpm": self.rpm,
            "engine_load": self.engine_load,
            "coolant_temp": self.coolant_temp,
            "fuel_level": self.fuel_level,
            "throttle_position": self.throttle_position,
            "intake_temp": self.intake_temp,
            "runtime_seconds": self.runtime_seconds,
            "distance_km": self.distance_km,
            "dtc_codes": self.dtc_codes,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AccelerationData:
    """Acceleration data for motion analysis."""

    x: float = 0.0  # Forward/backward
    y: float = 0.0  # Left/right
    z: float = 0.0  # Up/down
    magnitude: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AccidentEvent:
    """Detected accident event."""

    event_id: str
    severity: AlertLevel
    acceleration_g: float
    speed_before: float
    location: dict[str, float] | None = None
    timestamp: datetime = field(default_factory=datetime.now)


class VehicleNodeConfig(NodeConfig):
    """Vehicle node specific configuration."""

    # OBD-II settings
    obd_enabled: bool = Field(default=True, description="Enable OBD-II connection")
    obd_protocol: str = Field(default="auto", description="OBD protocol (auto, can, etc)")
    obd_port: str | None = Field(default=None, description="OBD adapter port")
    obd_baudrate: int = Field(default=38400, description="OBD adapter baudrate")

    # Polling intervals
    speed_poll_interval_ms: int = Field(default=500, description="Speed polling interval")
    data_poll_interval_ms: int = Field(default=2000, description="Full data polling interval")

    # Driving detection
    driving_speed_threshold_kmh: float = Field(default=5.0, description="Speed to detect driving")
    highway_speed_threshold_kmh: float = Field(default=80.0, description="Highway speed threshold")

    # Accident detection
    accident_detection_enabled: bool = Field(default=True, description="Enable accident detection")
    accident_g_threshold: float = Field(default=3.0, description="G-force threshold for accident")

    # Audio settings
    bluetooth_audio_enabled: bool = Field(default=True, description="Enable Bluetooth audio")
    audio_device_name: str | None = Field(default=None, description="Bluetooth audio device")

    # Auto mode switching
    auto_driving_mode: bool = Field(default=True, description="Auto-switch to driving mode")


class OBDConnection:
    """
    OBD-II connection handler.

    Manages connection to vehicle's OBD-II port
    and retrieves real-time vehicle data.
    """

    # Standard OBD-II PIDs
    PID_SPEED = "010D"  # Vehicle speed
    PID_RPM = "010C"  # Engine RPM
    PID_LOAD = "0104"  # Engine load
    PID_COOLANT = "0105"  # Coolant temperature
    PID_FUEL = "012F"  # Fuel level
    PID_THROTTLE = "0111"  # Throttle position

    def __init__(
        self,
        port: str | None = None,
        protocol: str = "auto",
        baudrate: int = 38400,
    ) -> None:
        """Initialize OBD connection."""
        self._port = port
        self._protocol = protocol
        self._baudrate = baudrate

        self._connected = False
        self._connection = None  # Would be obd.OBD() in production

    async def connect(self) -> bool:
        """
        Connect to OBD-II adapter.

        Returns:
            True if connection successful
        """
        try:
            logger.info(f"Connecting to OBD-II adapter on {self._port or 'auto'}")
            # In production: self._connection = obd.OBD(self._port)

            # Simulate connection
            self._connected = True
            logger.info("OBD-II connection established")
            return True

        except Exception as e:
            logger.error(f"OBD-II connection failed: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from OBD-II adapter."""
        if self._connection:
            # In production: self._connection.close()
            pass

        self._connected = False
        logger.info("OBD-II disconnected")

    async def get_speed(self) -> float:
        """Get current speed in km/h."""
        if not self._connected:
            return 0.0

        # In production: response = self._connection.query(obd.commands.SPEED)
        # return response.value.to("km/h").magnitude

        return 0.0  # Simulated

    async def get_rpm(self) -> int:
        """Get current RPM."""
        if not self._connected:
            return 0

        # In production: response = self._connection.query(obd.commands.RPM)
        return 0

    async def get_full_data(self) -> VehicleData:
        """Get full vehicle data snapshot."""
        if not self._connected:
            return VehicleData()

        # In production, would query multiple PIDs
        return VehicleData(
            speed_kmh=await self.get_speed(),
            rpm=await self.get_rpm(),
        )

    async def get_dtc_codes(self) -> list[str]:
        """Get diagnostic trouble codes."""
        if not self._connected:
            return []

        # In production: response = self._connection.query(obd.commands.GET_DTC)
        return []

    async def clear_dtc_codes(self) -> bool:
        """Clear diagnostic trouble codes."""
        if not self._connected:
            return False

        # In production: response = self._connection.query(obd.commands.CLEAR_DTC)
        return True

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected


class AccidentDetector:
    """
    Accident detection using accelerometer data.

    Monitors sudden deceleration events that may indicate
    a collision or accident.
    """

    def __init__(
        self,
        g_threshold: float = 3.0,
        sample_window_ms: int = 100,
    ) -> None:
        """Initialize accident detector."""
        self._g_threshold = g_threshold
        self._sample_window_ms = sample_window_ms

        self._samples: list[AccelerationData] = []
        self._max_samples = 50  # 5 seconds at 10Hz

        self._callbacks: list[Callable[[AccidentEvent], None]] = []
        self._last_event: AccidentEvent | None = None

    def process_acceleration(
        self,
        x: float,
        y: float,
        z: float,
        speed: float,
    ) -> AccidentEvent | None:
        """
        Process acceleration sample.

        Args:
            x, y, z: Acceleration in g
            speed: Current speed in km/h

        Returns:
            AccidentEvent if accident detected
        """
        import math
        import uuid

        magnitude = math.sqrt(x**2 + y**2 + z**2)

        sample = AccelerationData(
            x=x, y=y, z=z, magnitude=magnitude
        )
        self._samples.append(sample)

        # Keep window limited
        if len(self._samples) > self._max_samples:
            self._samples.pop(0)

        # Check for accident
        if magnitude >= self._g_threshold:
            severity = AlertLevel.WARNING
            if magnitude >= self._g_threshold * 1.5:
                severity = AlertLevel.CRITICAL

            event = AccidentEvent(
                event_id=str(uuid.uuid4()),
                severity=severity,
                acceleration_g=magnitude,
                speed_before=speed,
            )

            self._last_event = event
            self._notify_callbacks(event)
            return event

        return None

    def on_accident(self, callback: Callable[[AccidentEvent], None]) -> None:
        """Register accident callback."""
        self._callbacks.append(callback)

    def _notify_callbacks(self, event: AccidentEvent) -> None:
        """Notify callbacks of accident event."""
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Accident callback error: {e}")

    @property
    def last_event(self) -> AccidentEvent | None:
        """Get last accident event."""
        return self._last_event


class DrivingModeManager:
    """
    Manages driving mode detection and state transitions.

    Determines driving state based on speed, RPM, and motion data.
    """

    def __init__(
        self,
        driving_threshold: float = 5.0,
        highway_threshold: float = 80.0,
    ) -> None:
        """Initialize driving mode manager."""
        self._driving_threshold = driving_threshold
        self._highway_threshold = highway_threshold

        self._current_state = DrivingState.PARKED
        self._engine_state = EngineState.OFF
        self._state_since = datetime.now()

        self._callbacks: list[Callable[[DrivingState, DrivingState], None]] = []

    def update(self, speed: float, rpm: int) -> DrivingState:
        """
        Update driving state.

        Args:
            speed: Current speed in km/h
            rpm: Current RPM

        Returns:
            Current driving state
        """
        old_state = self._current_state

        # Determine engine state
        if rpm > 0:
            self._engine_state = EngineState.RUNNING
        else:
            self._engine_state = EngineState.OFF

        # Determine driving state
        if speed >= self._highway_threshold:
            new_state = DrivingState.HIGHWAY
        elif speed >= self._driving_threshold:
            new_state = DrivingState.MOVING
        elif rpm > 0:
            new_state = DrivingState.IDLING
        else:
            new_state = DrivingState.PARKED

        if new_state != old_state:
            self._current_state = new_state
            self._state_since = datetime.now()
            self._notify_transition(old_state, new_state)

        return self._current_state

    def on_state_change(
        self,
        callback: Callable[[DrivingState, DrivingState], None],
    ) -> None:
        """Register state change callback."""
        self._callbacks.append(callback)

    def _notify_transition(
        self,
        old_state: DrivingState,
        new_state: DrivingState,
    ) -> None:
        """Notify callbacks of state transition."""
        for callback in self._callbacks:
            try:
                callback(old_state, new_state)
            except Exception as e:
                logger.error(f"State change callback error: {e}")

    @property
    def state(self) -> DrivingState:
        """Get current driving state."""
        return self._current_state

    @property
    def engine_state(self) -> EngineState:
        """Get current engine state."""
        return self._engine_state

    @property
    def is_driving(self) -> bool:
        """Check if vehicle is driving."""
        return self._current_state in (DrivingState.MOVING, DrivingState.HIGHWAY)


class VehicleNode(BaseNode):
    """
    Vehicle node with OBD-II integration.

    Provides:
    - Real-time vehicle data monitoring
    - Driving mode detection
    - Accident detection and alerts
    - Bluetooth audio integration
    - Auto driving mode activation
    """

    def __init__(self, config: VehicleNodeConfig) -> None:
        """Initialize vehicle node."""
        capabilities = [
            NodeCapability.VOICE_INPUT,
            NodeCapability.VOICE_OUTPUT,
            NodeCapability.OBD,
            NodeCapability.MOTION,
            NodeCapability.GPS,
        ]

        super().__init__(
            node_type=NodeType.VEHICLE,
            config=config,
            capabilities=capabilities,
        )

        self._vehicle_config = config

        # Components
        self._obd: OBDConnection | None = None
        self._driving_manager = DrivingModeManager(
            driving_threshold=config.driving_speed_threshold_kmh,
            highway_threshold=config.highway_speed_threshold_kmh,
        )
        self._accident_detector: AccidentDetector | None = None

        # State
        self._vehicle_data: VehicleData = VehicleData()
        self._trip_start_km: float = 0.0
        self._trip_start_time: datetime | None = None

        # Tasks
        self._speed_task: asyncio.Task | None = None
        self._data_task: asyncio.Task | None = None

    @property
    def firmware_version(self) -> str:
        """Get firmware version."""
        return "1.0.0"

    @property
    def hardware_info(self) -> dict[str, Any]:
        """Get hardware information."""
        return {
            "platform": "vehicle_obd",
            "obd_protocol": self._vehicle_config.obd_protocol,
            "has_accelerometer": self._vehicle_config.accident_detection_enabled,
            "bluetooth_audio": self._vehicle_config.bluetooth_audio_enabled,
        }

    async def _on_start(self) -> None:
        """Initialize vehicle node resources."""
        # Connect to OBD-II
        if self._vehicle_config.obd_enabled:
            self._obd = OBDConnection(
                port=self._vehicle_config.obd_port,
                protocol=self._vehicle_config.obd_protocol,
                baudrate=self._vehicle_config.obd_baudrate,
            )
            await self._obd.connect()

        # Initialize accident detector
        if self._vehicle_config.accident_detection_enabled:
            self._accident_detector = AccidentDetector(
                g_threshold=self._vehicle_config.accident_g_threshold,
            )
            self._accident_detector.on_accident(self._on_accident_detected)

        # Set up driving mode callbacks
        self._driving_manager.on_state_change(self._on_driving_state_change)

        # Start polling tasks
        self._speed_task = asyncio.create_task(self._speed_polling_loop())
        self._tasks.append(self._speed_task)

        self._data_task = asyncio.create_task(self._data_polling_loop())
        self._tasks.append(self._data_task)

        logger.info("Vehicle node initialized")

    async def _on_stop(self) -> None:
        """Cleanup vehicle node resources."""
        if self._obd:
            await self._obd.disconnect()

    async def _speed_polling_loop(self) -> None:
        """Fast speed polling loop."""
        interval = self._vehicle_config.speed_poll_interval_ms / 1000

        while True:
            try:
                if self._obd and self._obd.is_connected:
                    speed = await self._obd.get_speed()
                    rpm = await self._obd.get_rpm()

                    self._vehicle_data.speed_kmh = speed
                    self._vehicle_data.rpm = rpm

                    # Update driving state
                    self._driving_manager.update(speed, rpm)

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Speed polling error: {e}")
                await asyncio.sleep(1)

    async def _data_polling_loop(self) -> None:
        """Full data polling loop."""
        interval = self._vehicle_config.data_poll_interval_ms / 1000

        while True:
            try:
                if self._obd and self._obd.is_connected:
                    self._vehicle_data = await self._obd.get_full_data()

                    # Send to hub
                    await self.send_sensor_data("vehicle", self._vehicle_data.to_dict())

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Data polling error: {e}")
                await asyncio.sleep(5)

    def _on_driving_state_change(
        self,
        old_state: DrivingState,
        new_state: DrivingState,
    ) -> None:
        """Handle driving state change."""
        logger.info(f"Driving state changed: {old_state.value} → {new_state.value}")

        # Start/end trip tracking
        if not self._driving_manager.is_driving and old_state in (
            DrivingState.MOVING,
            DrivingState.HIGHWAY,
        ):
            self._end_trip()

        elif self._driving_manager.is_driving and old_state in (
            DrivingState.PARKED,
            DrivingState.IDLING,
        ):
            self._start_trip()

        # Notify hub
        asyncio.create_task(
            self.send_sensor_data(
                "driving_state",
                {
                    "old_state": old_state.value,
                    "new_state": new_state.value,
                    "is_driving": self._driving_manager.is_driving,
                },
            )
        )

        self._emit_event(
            "driving_state_changed",
            {"old": old_state.value, "new": new_state.value},
        )

    def _on_accident_detected(self, event: AccidentEvent) -> None:
        """Handle detected accident."""
        logger.critical(f"ACCIDENT DETECTED: {event.severity.value} - {event.acceleration_g}g")

        # Trigger crisis intervention
        asyncio.create_task(
            self.trigger_intervention(
                "accident",
                {
                    "event_id": event.event_id,
                    "severity": event.severity.value,
                    "acceleration_g": event.acceleration_g,
                    "speed_before_kmh": event.speed_before,
                    "timestamp": event.timestamp.isoformat(),
                },
            )
        )

        self._emit_event("accident_detected", {
            "event_id": event.event_id,
            "severity": event.severity.value,
        })

    def _start_trip(self) -> None:
        """Start trip tracking."""
        self._trip_start_km = self._vehicle_data.distance_km
        self._trip_start_time = datetime.now()
        logger.info("Trip started")

    def _end_trip(self) -> None:
        """End trip tracking."""
        if not self._trip_start_time:
            return

        duration = datetime.now() - self._trip_start_time
        distance = self._vehicle_data.distance_km - self._trip_start_km

        logger.info(f"Trip ended: {distance:.1f}km in {duration}")

        self._emit_event("trip_ended", {
            "distance_km": distance,
            "duration_seconds": duration.total_seconds(),
        })

        self._trip_start_time = None

    # Public API

    async def get_vehicle_data(self) -> VehicleData:
        """Get current vehicle data."""
        return self._vehicle_data

    async def get_dtc_codes(self) -> list[str]:
        """Get diagnostic trouble codes."""
        if self._obd:
            return await self._obd.get_dtc_codes()
        return []

    async def clear_dtc_codes(self) -> bool:
        """Clear diagnostic trouble codes."""
        if self._obd:
            return await self._obd.clear_dtc_codes()
        return False

    def process_acceleration(
        self,
        x: float,
        y: float,
        z: float,
    ) -> AccidentEvent | None:
        """
        Process acceleration data from external sensor.

        Args:
            x, y, z: Acceleration in g

        Returns:
            AccidentEvent if accident detected
        """
        if self._accident_detector:
            return self._accident_detector.process_acceleration(
                x, y, z, self._vehicle_data.speed_kmh
            )
        return None

    def get_status(self) -> dict[str, Any]:
        """Get vehicle node status."""
        status = super().get_status()
        status.update({
            "obd_connected": self._obd.is_connected if self._obd else False,
            "driving_state": self._driving_manager.state.value,
            "engine_state": self._driving_manager.engine_state.value,
            "is_driving": self._driving_manager.is_driving,
            "vehicle_data": self._vehicle_data.to_dict(),
            "trip_active": self._trip_start_time is not None,
        })
        return status

"""
SAIL Mobile Node

Mobile app node for portable SAIL access via smartphone.
Features GPS integration, push notifications, offline capability,
and seamless handoff with other nodes.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import Field

from src.hub.base import NodeCapability, NodeType
from src.nodes.base import BaseNode, MessageType, NodeConfig, NodeMessage

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class LocationAccuracy(str, Enum):
    """GPS location accuracy levels."""

    HIGH = "high"  # GPS + WiFi + Cell
    BALANCED = "balanced"  # WiFi + Cell
    LOW = "low"  # Cell only
    PASSIVE = "passive"  # Use last known


class NotificationPriority(str, Enum):
    """Push notification priority."""

    MAX = "max"  # Crisis/emergency
    HIGH = "high"  # Important intervention
    DEFAULT = "default"  # Regular alerts
    LOW = "low"  # Background sync
    MIN = "min"  # Silent


@dataclass
class GPSLocation:
    """GPS location data."""

    latitude: float
    longitude: float
    altitude: float | None = None
    accuracy: float = 0.0  # Meters
    speed: float | None = None  # m/s
    bearing: float | None = None  # Degrees
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
            "accuracy": self.accuracy,
            "speed": self.speed,
            "bearing": self.bearing,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Notification:
    """Push notification."""

    title: str
    body: str
    priority: NotificationPriority = NotificationPriority.DEFAULT
    action: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    expires_at: datetime | None = None


class MobileNodeConfig(NodeConfig):
    """Mobile node specific configuration."""

    # GPS settings
    gps_enabled: bool = Field(default=True, description="Enable GPS tracking")
    gps_accuracy: LocationAccuracy = Field(
        default=LocationAccuracy.BALANCED, description="GPS accuracy level"
    )
    gps_update_interval_seconds: int = Field(default=30, description="GPS update interval")
    gps_min_distance_meters: float = Field(
        default=50.0, description="Min distance for location update"
    )

    # Push notifications
    push_enabled: bool = Field(default=True, description="Enable push notifications")
    push_token: str | None = Field(default=None, description="Push notification token")

    # Background operation
    background_enabled: bool = Field(default=True, description="Enable background operation")
    background_sync_interval_seconds: int = Field(
        default=300, description="Background sync interval"
    )

    # Battery management
    battery_saver_enabled: bool = Field(default=True, description="Enable battery saver mode")
    low_battery_threshold: int = Field(default=20, description="Low battery threshold %")

    # Local processing
    local_stt_enabled: bool = Field(default=False, description="Enable local STT")
    local_tts_enabled: bool = Field(default=False, description="Enable local TTS")


class GPSTracker:
    """
    GPS location tracking.

    Manages location updates with configurable accuracy
    and power management.
    """

    def __init__(
        self,
        accuracy: LocationAccuracy = LocationAccuracy.BALANCED,
        update_interval: int = 30,
        min_distance: float = 50.0,
    ) -> None:
        """Initialize GPS tracker."""
        self._accuracy = accuracy
        self._update_interval = update_interval
        self._min_distance = min_distance

        self._is_tracking = False
        self._last_location: GPSLocation | None = None
        self._location_callbacks: list[Callable[[GPSLocation], None]] = []

        # Tracking task
        self._tracking_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start GPS tracking."""
        if self._is_tracking:
            return

        logger.info(f"Starting GPS tracking (accuracy={self._accuracy.value})")
        self._is_tracking = True
        self._tracking_task = asyncio.create_task(self._tracking_loop())

    async def stop(self) -> None:
        """Stop GPS tracking."""
        if not self._is_tracking:
            return

        self._is_tracking = False
        if self._tracking_task:
            self._tracking_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._tracking_task

        logger.info("GPS tracking stopped")

    async def _tracking_loop(self) -> None:
        """Main tracking loop."""
        while self._is_tracking:
            try:
                location = await self._get_location()
                if location and self._should_update(location):
                    self._last_location = location
                    self._notify_location(location)

                await asyncio.sleep(self._update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"GPS tracking error: {e}")
                await asyncio.sleep(5)

    async def _get_location(self) -> GPSLocation | None:
        """Get current GPS location."""
        # In production, would use platform-specific location API
        # For now, return simulated location
        return GPSLocation(
            latitude=37.7749,
            longitude=-122.4194,
            accuracy=10.0,
        )

    def _should_update(self, location: GPSLocation) -> bool:
        """Check if location should trigger update."""
        if not self._last_location:
            return True

        # Calculate distance
        distance = self._calculate_distance(
            self._last_location.latitude,
            self._last_location.longitude,
            location.latitude,
            location.longitude,
        )

        return distance >= self._min_distance

    @staticmethod
    def _calculate_distance(
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """Calculate distance between two points in meters."""
        import math

        R = 6371000  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = (
            math.sin(delta_phi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _notify_location(self, location: GPSLocation) -> None:
        """Notify callbacks of location update."""
        for callback in self._location_callbacks:
            try:
                callback(location)
            except Exception as e:
                logger.error(f"Location callback error: {e}")

    def on_location(self, callback: Callable[[GPSLocation], None]) -> None:
        """Register location callback."""
        self._location_callbacks.append(callback)

    def get_last_location(self) -> GPSLocation | None:
        """Get last known location."""
        return self._last_location

    @property
    def is_tracking(self) -> bool:
        """Check if tracking is active."""
        return self._is_tracking


class PushNotificationManager:
    """
    Push notification management.

    Handles sending and receiving push notifications
    for mobile alerts and interventions.
    """

    def __init__(self, token: str | None = None) -> None:
        """Initialize push notification manager."""
        self._token = token
        self._notification_callbacks: list[Callable[[Notification], None]] = []
        self._pending_notifications: list[Notification] = []

    def set_token(self, token: str) -> None:
        """Set push notification token."""
        self._token = token
        logger.info("Push notification token set")

    async def show_notification(self, notification: Notification) -> None:
        """Show a push notification."""
        if not self._token:
            logger.warning("No push token set, queueing notification")
            self._pending_notifications.append(notification)
            return

        logger.info(f"Showing notification: {notification.title}")
        # In production, would use platform notification API
        self._notify_handlers(notification)

    async def show_intervention_alert(
        self,
        title: str,
        message: str,
        intervention_type: str,
    ) -> None:
        """Show intervention alert notification."""
        priority = NotificationPriority.HIGH
        if intervention_type == "crisis":
            priority = NotificationPriority.MAX

        notification = Notification(
            title=title,
            body=message,
            priority=priority,
            action="open_intervention",
            data={"intervention_type": intervention_type},
        )

        await self.show_notification(notification)

    def on_notification(self, callback: Callable[[Notification], None]) -> None:
        """Register notification callback."""
        self._notification_callbacks.append(callback)

    def _notify_handlers(self, notification: Notification) -> None:
        """Notify handlers of notification."""
        for callback in self._notification_callbacks:
            try:
                callback(notification)
            except Exception as e:
                logger.error(f"Notification callback error: {e}")

    async def flush_pending(self) -> int:
        """Flush pending notifications."""
        if not self._token:
            return 0

        count = 0
        while self._pending_notifications:
            notification = self._pending_notifications.pop(0)
            await self.show_notification(notification)
            count += 1

        return count


class OfflineStore:
    """
    Offline data storage.

    Stores context and queries when offline for sync
    when connection is restored.
    """

    def __init__(self, max_entries: int = 500) -> None:
        """Initialize offline store."""
        self._max_entries = max_entries
        self._entries: list[dict[str, Any]] = []
        self._queries: list[dict[str, Any]] = []

    def store_entry(self, entry: dict[str, Any]) -> None:
        """Store a context entry."""
        if len(self._entries) >= self._max_entries:
            self._entries.pop(0)
        self._entries.append(entry)

    def store_query(self, query: str, metadata: dict[str, Any] | None = None) -> None:
        """Store an offline query."""
        self._queries.append({
            "query": query,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        })

    def get_entries(self) -> list[dict[str, Any]]:
        """Get all stored entries."""
        return self._entries.copy()

    def get_queries(self) -> list[dict[str, Any]]:
        """Get all stored queries."""
        return self._queries.copy()

    def clear(self) -> None:
        """Clear all stored data."""
        self._entries.clear()
        self._queries.clear()

    @property
    def entry_count(self) -> int:
        """Get stored entry count."""
        return len(self._entries)

    @property
    def query_count(self) -> int:
        """Get stored query count."""
        return len(self._queries)


class MobileNode(BaseNode):
    """
    Mobile node for portable SAIL access.

    Provides:
    - GPS location tracking
    - Push notifications
    - Offline operation
    - Background sync
    - Battery-aware operation
    - Handoff support
    """

    def __init__(self, config: MobileNodeConfig) -> None:
        """Initialize mobile node."""
        capabilities = [
            NodeCapability.VOICE_INPUT,
            NodeCapability.VOICE_OUTPUT,
            NodeCapability.GPS,
            NodeCapability.PUSH_NOTIFICATION,
            NodeCapability.OFFLINE_PROCESSING,
            NodeCapability.DISPLAY,
        ]

        if config.local_stt_enabled:
            capabilities.append(NodeCapability.LOCAL_STT)
        if config.local_tts_enabled:
            capabilities.append(NodeCapability.LOCAL_TTS)

        super().__init__(
            node_type=NodeType.MOBILE,
            config=config,
            capabilities=capabilities,
        )

        self._mobile_config = config

        # Components
        self._gps: GPSTracker | None = None
        self._notifications = PushNotificationManager(config.push_token)
        self._offline_store = OfflineStore(config.offline_buffer_size)

        # State
        self._is_foreground = True
        self._battery_level = 100
        self._is_charging = False
        self._current_user: str | None = None

        # Background task
        self._background_task: asyncio.Task | None = None

    @property
    def firmware_version(self) -> str:
        """Get firmware version."""
        return "1.0.0"

    @property
    def hardware_info(self) -> dict[str, Any]:
        """Get hardware information."""
        return {
            "platform": "mobile",
            "os": "unknown",  # Would be iOS/Android
            "device_model": "unknown",
            "has_gps": self._mobile_config.gps_enabled,
            "battery_level": self._battery_level,
        }

    async def _on_start(self) -> None:
        """Initialize mobile node resources."""
        # Initialize GPS tracker
        if self._mobile_config.gps_enabled:
            self._gps = GPSTracker(
                accuracy=self._mobile_config.gps_accuracy,
                update_interval=self._mobile_config.gps_update_interval_seconds,
                min_distance=self._mobile_config.gps_min_distance_meters,
            )
            self._gps.on_location(self._on_location_update)
            await self._gps.start()

        # Set up notification token
        if self._mobile_config.push_token:
            self._notifications.set_token(self._mobile_config.push_token)

        # Register message handlers
        self._message_handlers[MessageType.BROADCAST] = self._handle_intervention_broadcast

        # Start background sync if enabled
        if self._mobile_config.background_enabled:
            self._background_task = asyncio.create_task(self._background_sync_loop())
            self._tasks.append(self._background_task)

        logger.info("Mobile node initialized")

    async def _on_stop(self) -> None:
        """Cleanup mobile node resources."""
        if self._gps:
            await self._gps.stop()

    def _on_location_update(self, location: GPSLocation) -> None:
        """Handle GPS location update."""
        logger.debug(f"Location update: {location.latitude}, {location.longitude}")

        # Send to hub
        asyncio.create_task(
            self.send_sensor_data("gps", location.to_dict())
        )

        self._emit_event("location_update", location.to_dict())

    async def _handle_intervention_broadcast(self, message: NodeMessage) -> None:
        """Handle intervention broadcast from hub."""
        payload = message.payload
        intervention_type = payload.get("intervention_type", "advisory")
        title = payload.get("title", "SAIL Alert")
        body = payload.get("message", "")

        await self._notifications.show_intervention_alert(
            title=title,
            message=body,
            intervention_type=intervention_type,
        )

    async def _background_sync_loop(self) -> None:
        """Background synchronization loop."""
        while True:
            try:
                await asyncio.sleep(self._mobile_config.background_sync_interval_seconds)

                # Check battery saver
                if self._mobile_config.battery_saver_enabled:
                    if self._battery_level < self._mobile_config.low_battery_threshold:
                        if not self._is_charging:
                            continue  # Skip sync in low battery

                # Sync offline data
                if self._offline_store.entry_count > 0:
                    await self._sync_offline_data()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background sync error: {e}")

    async def _sync_offline_data(self) -> None:
        """Sync offline stored data."""
        if not self.is_connected:
            return

        entries = self._offline_store.get_entries()
        self._offline_store.get_queries()

        if entries:
            try:
                await self.request_sync({"entries": entries})
                self._offline_store.clear()
                logger.info(f"Synced {len(entries)} offline entries")
            except Exception as e:
                logger.error(f"Failed to sync offline data: {e}")

    # Public API

    async def send_query_with_offline(
        self,
        text: str,
        user_id: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Send query with offline fallback.

        If offline, stores query for later processing.
        """
        if self.is_connected:
            return await self.send_query(text, user_id)
        else:
            self._offline_store.store_query(text, {"user_id": user_id})
            return None

    async def request_handoff(self, target_node_id: str) -> dict[str, Any]:
        """
        Request handoff to another node.

        Used when transitioning between nodes (e.g., phone → car).
        """
        return await self._send_request(
            MessageType.SYNC_REQUEST,
            {
                "type": "handoff",
                "target_node": target_node_id,
                "current_location": self._gps.get_last_location().to_dict()
                if self._gps and self._gps.get_last_location()
                else None,
            },
        )

    def set_foreground(self, is_foreground: bool) -> None:
        """Set foreground/background state."""
        self._is_foreground = is_foreground
        self._emit_event("foreground_changed", {"is_foreground": is_foreground})

    def update_battery(self, level: int, is_charging: bool) -> None:
        """Update battery status."""
        self._battery_level = level
        self._is_charging = is_charging

    def get_location(self) -> GPSLocation | None:
        """Get current location."""
        return self._gps.get_last_location() if self._gps else None

    def get_status(self) -> dict[str, Any]:
        """Get mobile node status."""
        status = super().get_status()
        status.update({
            "is_foreground": self._is_foreground,
            "battery_level": self._battery_level,
            "is_charging": self._is_charging,
            "gps_tracking": self._gps.is_tracking if self._gps else False,
            "location": self._gps.get_last_location().to_dict()
            if self._gps and self._gps.get_last_location()
            else None,
            "offline_entries": self._offline_store.entry_count,
            "offline_queries": self._offline_store.query_count,
        })
        return status

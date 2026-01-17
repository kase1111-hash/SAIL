"""
Location sensor implementation.

Provides GPS integration, geofencing, and jurisdiction detection
for location-based situational awareness.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional
from math import radians, sin, cos, sqrt, atan2

from .base import (
    Sensor,
    SensorType,
    SensorStatus,
    SensorReading,
    SensorConfig,
    SensorEvent,
    SensorEventType,
    GeoLocation,
    Jurisdiction,
    Geofence,
    LocationContext,
    Confidence,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Location Data Types
# ============================================================================


@dataclass
class LocationReading:
    """Combined location reading with context."""

    location: GeoLocation
    jurisdiction: Optional[Jurisdiction] = None
    context: LocationContext = LocationContext.UNKNOWN
    active_geofence: Optional[Geofence] = None
    is_unfamiliar: bool = False
    distance_from_home_km: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "location": self.location.to_dict(),
            "jurisdiction": self.jurisdiction.to_dict() if self.jurisdiction else None,
            "context": self.context.value,
            "active_geofence": self.active_geofence.to_dict() if self.active_geofence else None,
            "is_unfamiliar": self.is_unfamiliar,
            "distance_from_home_km": self.distance_from_home_km,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# GPS Provider Abstraction
# ============================================================================


class GPSProvider:
    """Abstract GPS data provider."""

    async def get_location(self) -> Optional[GeoLocation]:
        """Get current GPS location."""
        raise NotImplementedError


class SimulatedGPSProvider(GPSProvider):
    """Simulated GPS provider for testing."""

    def __init__(
        self,
        latitude: float = 37.7749,
        longitude: float = -122.4194,
        simulate_movement: bool = False,
    ):
        self.latitude = latitude
        self.longitude = longitude
        self.simulate_movement = simulate_movement
        self._movement_step = 0

    async def get_location(self) -> Optional[GeoLocation]:
        """Get simulated GPS location."""
        if self.simulate_movement:
            # Simulate movement
            self._movement_step += 1
            lat_offset = (self._movement_step % 100) * 0.0001
            lon_offset = (self._movement_step % 50) * 0.0001
        else:
            lat_offset = 0
            lon_offset = 0

        return GeoLocation(
            latitude=self.latitude + lat_offset,
            longitude=self.longitude + lon_offset,
            altitude=10.0,
            accuracy_meters=5.0,
            heading=0.0,
            speed_mps=0.0 if not self.simulate_movement else 1.5,
            timestamp=datetime.now(),
        )


class SystemGPSProvider(GPSProvider):
    """System GPS provider using gpsd or platform APIs."""

    def __init__(self):
        self._gpsd_available = False
        self._last_location: Optional[GeoLocation] = None

    async def initialize(self) -> bool:
        """Initialize connection to GPS daemon."""
        try:
            # Try to import and connect to gpsd
            import gpsd
            gpsd.connect()
            self._gpsd_available = True
            logger.info("Connected to gpsd")
            return True
        except ImportError:
            logger.warning("gpsd library not available")
            return False
        except Exception as e:
            logger.warning(f"Could not connect to gpsd: {e}")
            return False

    async def get_location(self) -> Optional[GeoLocation]:
        """Get location from system GPS."""
        if not self._gpsd_available:
            return None

        try:
            import gpsd
            packet = gpsd.get_current()

            if packet.mode >= 2:  # 2D or 3D fix
                self._last_location = GeoLocation(
                    latitude=packet.lat,
                    longitude=packet.lon,
                    altitude=packet.alt if packet.mode >= 3 else None,
                    accuracy_meters=packet.error.get("x", None),
                    heading=packet.track if hasattr(packet, "track") else None,
                    speed_mps=packet.speed if hasattr(packet, "speed") else None,
                    timestamp=datetime.now(),
                )
                return self._last_location
        except Exception as e:
            logger.error(f"GPS read error: {e}")

        return self._last_location


# ============================================================================
# Jurisdiction Detection
# ============================================================================


class JurisdictionDetector:
    """Detects jurisdiction from coordinates."""

    # US State bounding boxes (simplified)
    US_STATE_BOUNDS: dict[str, tuple[float, float, float, float]] = {
        # (min_lat, max_lat, min_lon, max_lon)
        "CA": (32.5, 42.0, -124.5, -114.0),
        "TX": (25.8, 36.5, -106.7, -93.5),
        "FL": (24.5, 31.0, -87.6, -80.0),
        "NY": (40.5, 45.0, -79.8, -71.8),
        "WA": (45.5, 49.0, -124.8, -116.9),
        "OR": (41.9, 46.3, -124.6, -116.5),
        "NV": (35.0, 42.0, -120.0, -114.0),
        "AZ": (31.3, 37.0, -114.8, -109.0),
        "CO": (37.0, 41.0, -109.1, -102.0),
        "IL": (36.9, 42.5, -91.5, -87.0),
        "PA": (39.7, 42.3, -80.5, -74.7),
        "OH": (38.4, 42.0, -84.8, -80.5),
        "GA": (30.4, 35.0, -85.6, -80.8),
        "NC": (33.8, 36.6, -84.3, -75.5),
        "MI": (41.7, 48.3, -90.4, -82.4),
    }

    # Country codes by rough coordinate ranges
    COUNTRY_BOUNDS: dict[str, tuple[float, float, float, float, str]] = {
        # (min_lat, max_lat, min_lon, max_lon, country_name)
        "US": (24.0, 50.0, -125.0, -66.0, "United States"),
        "CA": (41.0, 84.0, -141.0, -52.0, "Canada"),
        "MX": (14.0, 33.0, -118.0, -86.0, "Mexico"),
        "GB": (49.0, 61.0, -8.0, 2.0, "United Kingdom"),
        "DE": (47.0, 55.0, 6.0, 15.0, "Germany"),
        "FR": (41.0, 51.5, -5.0, 10.0, "France"),
        "AU": (-44.0, -10.0, 112.0, 154.0, "Australia"),
        "JP": (24.0, 46.0, 122.0, 154.0, "Japan"),
    }

    # State names for US
    US_STATE_NAMES: dict[str, str] = {
        "CA": "California",
        "TX": "Texas",
        "FL": "Florida",
        "NY": "New York",
        "WA": "Washington",
        "OR": "Oregon",
        "NV": "Nevada",
        "AZ": "Arizona",
        "CO": "Colorado",
        "IL": "Illinois",
        "PA": "Pennsylvania",
        "OH": "Ohio",
        "GA": "Georgia",
        "NC": "North Carolina",
        "MI": "Michigan",
    }

    def __init__(self):
        self._cache: dict[tuple[float, float], Jurisdiction] = {}
        self._cache_precision = 2  # Decimal places for cache key

    def detect(self, location: GeoLocation) -> Optional[Jurisdiction]:
        """Detect jurisdiction from location."""
        # Check cache
        cache_key = (
            round(location.latitude, self._cache_precision),
            round(location.longitude, self._cache_precision),
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Detect country
        country_code = None
        country_name = None
        for code, (min_lat, max_lat, min_lon, max_lon, name) in self.COUNTRY_BOUNDS.items():
            if min_lat <= location.latitude <= max_lat and min_lon <= location.longitude <= max_lon:
                country_code = code
                country_name = name
                break

        if not country_code:
            return None

        # Detect state (US only for now)
        state_code = None
        state_name = None
        if country_code == "US":
            for code, (min_lat, max_lat, min_lon, max_lon) in self.US_STATE_BOUNDS.items():
                if min_lat <= location.latitude <= max_lat and min_lon <= location.longitude <= max_lon:
                    state_code = code
                    state_name = self.US_STATE_NAMES.get(code)
                    break

        jurisdiction = Jurisdiction(
            country=country_name,
            country_code=country_code,
            state=state_name,
            state_code=state_code,
        )

        # Cache result
        self._cache[cache_key] = jurisdiction
        return jurisdiction

    async def detect_async(self, location: GeoLocation) -> Optional[Jurisdiction]:
        """Async version of detect (for future geocoding API support)."""
        return self.detect(location)


# ============================================================================
# Geofence Manager
# ============================================================================


class GeofenceManager:
    """Manages geofences and detects enter/exit events."""

    def __init__(self, geofences: Optional[list[Geofence]] = None):
        self.geofences: dict[str, Geofence] = {}
        self._active_geofences: set[str] = set()
        self._callbacks: list[Callable[[SensorEvent], None]] = []

        if geofences:
            for gf in geofences:
                self.add_geofence(gf)

    def add_geofence(self, geofence: Geofence) -> None:
        """Add a geofence."""
        self.geofences[geofence.id] = geofence

    def remove_geofence(self, geofence_id: str) -> None:
        """Remove a geofence."""
        if geofence_id in self.geofences:
            del self.geofences[geofence_id]
            self._active_geofences.discard(geofence_id)

    def get_home_geofence(self) -> Optional[Geofence]:
        """Get the home geofence if defined."""
        for gf in self.geofences.values():
            if gf.context == LocationContext.HOME:
                return gf
        return None

    def register_callback(self, callback: Callable[[SensorEvent], None]) -> None:
        """Register callback for geofence events."""
        self._callbacks.append(callback)

    def check_location(self, location: GeoLocation) -> tuple[Optional[Geofence], list[SensorEvent]]:
        """
        Check location against geofences.
        Returns (active_geofence, list_of_events).
        """
        events: list[SensorEvent] = []
        currently_in: set[str] = set()
        active_geofence: Optional[Geofence] = None

        for gf_id, geofence in self.geofences.items():
            if geofence.contains(location):
                currently_in.add(gf_id)
                active_geofence = geofence  # Take the most recent one

                # Check for enter event
                if gf_id not in self._active_geofences:
                    event = SensorEvent(
                        event_type=SensorEventType.GEOFENCE_ENTER,
                        sensor_type=SensorType.LOCATION,
                        data={"geofence": geofence, "location": location},
                        metadata={"geofence_id": gf_id, "geofence_name": geofence.name},
                    )
                    events.append(event)

        # Check for exit events
        for gf_id in self._active_geofences - currently_in:
            geofence = self.geofences.get(gf_id)
            if geofence:
                event = SensorEvent(
                    event_type=SensorEventType.GEOFENCE_EXIT,
                    sensor_type=SensorType.LOCATION,
                    data={"geofence": geofence, "location": location},
                    metadata={"geofence_id": gf_id, "geofence_name": geofence.name},
                )
                events.append(event)

        # Update active set
        self._active_geofences = currently_in

        # Notify callbacks
        for event in events:
            for callback in self._callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Geofence callback error: {e}")

        return active_geofence, events


# ============================================================================
# Distance Calculator
# ============================================================================


def haversine_distance(loc1: GeoLocation, loc2: GeoLocation) -> float:
    """Calculate distance between two locations in kilometers."""
    R = 6371.0  # Earth's radius in kilometers

    lat1 = radians(loc1.latitude)
    lat2 = radians(loc2.latitude)
    dlat = radians(loc2.latitude - loc1.latitude)
    dlon = radians(loc2.longitude - loc1.longitude)

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c


def calculate_distance_from_home(
    location: GeoLocation,
    home_geofence: Optional[Geofence],
) -> Optional[float]:
    """Calculate distance from home in kilometers."""
    if not home_geofence:
        return None

    home_location = GeoLocation(
        latitude=home_geofence.latitude,
        longitude=home_geofence.longitude,
    )
    return haversine_distance(location, home_location)


# ============================================================================
# Location Sensor
# ============================================================================


class LocationSensor(Sensor):
    """Location sensor with GPS, geofencing, and jurisdiction detection."""

    def __init__(
        self,
        sensor_id: str = "location_primary",
        config: Optional[SensorConfig] = None,
        gps_provider: Optional[GPSProvider] = None,
    ):
        super().__init__(sensor_id, SensorType.LOCATION, config)

        self.gps_provider = gps_provider or SimulatedGPSProvider()
        self.jurisdiction_detector = JurisdictionDetector()
        self.geofence_manager = GeofenceManager(config.geofences if config else None)

        self._last_jurisdiction: Optional[Jurisdiction] = None
        self._event_callbacks: list[Callable[[SensorEvent], None]] = []

    def add_geofence(self, geofence: Geofence) -> None:
        """Add a geofence."""
        self.geofence_manager.add_geofence(geofence)

    def remove_geofence(self, geofence_id: str) -> None:
        """Remove a geofence."""
        self.geofence_manager.remove_geofence(geofence_id)

    def register_event_callback(self, callback: Callable[[SensorEvent], None]) -> None:
        """Register callback for sensor events."""
        self._event_callbacks.append(callback)
        self.geofence_manager.register_callback(callback)

    async def initialize(self) -> bool:
        """Initialize the location sensor."""
        self._status = SensorStatus.INITIALIZING

        try:
            # Initialize GPS provider if it has an initialize method
            if hasattr(self.gps_provider, "initialize"):
                success = await self.gps_provider.initialize()
                if not success:
                    logger.warning("GPS provider initialization failed, using simulated data")

            self._status = SensorStatus.ACTIVE
            logger.info(f"Location sensor {self.sensor_id} initialized")
            return True

        except Exception as e:
            self._set_error(f"Initialization failed: {e}")
            return False

    async def read(self) -> SensorReading:
        """Take a location reading."""
        location = await self.gps_provider.get_location()

        if not location:
            return SensorReading(
                sensor_type=self.sensor_type,
                sensor_id=self.sensor_id,
                data=None,
                confidence=Confidence.VERY_LOW,
                metadata={"error": "No GPS fix"},
            )

        # Detect jurisdiction
        jurisdiction = self.jurisdiction_detector.detect(location)

        # Check for jurisdiction change
        if jurisdiction and self._last_jurisdiction:
            if jurisdiction.state_code != self._last_jurisdiction.state_code:
                event = SensorEvent(
                    event_type=SensorEventType.JURISDICTION_CHANGE,
                    sensor_type=SensorType.LOCATION,
                    data={
                        "previous": self._last_jurisdiction,
                        "current": jurisdiction,
                    },
                )
                self._notify_event(event)

        self._last_jurisdiction = jurisdiction

        # Check geofences
        active_geofence, _ = self.geofence_manager.check_location(location)

        # Determine location context
        if active_geofence:
            context = active_geofence.context
        else:
            context = LocationContext.UNFAMILIAR

        # Calculate distance from home
        home_geofence = self.geofence_manager.get_home_geofence()
        distance_from_home = calculate_distance_from_home(location, home_geofence)

        # Determine if unfamiliar
        is_unfamiliar = (
            context == LocationContext.UNFAMILIAR
            and distance_from_home is not None
            and distance_from_home > self.config.unfamiliar_area_threshold_km
        )

        # Determine confidence based on GPS accuracy
        if location.accuracy_meters is not None:
            if location.accuracy_meters < 10:
                confidence = Confidence.VERY_HIGH
            elif location.accuracy_meters < 25:
                confidence = Confidence.HIGH
            elif location.accuracy_meters < 50:
                confidence = Confidence.MEDIUM
            elif location.accuracy_meters < 100:
                confidence = Confidence.LOW
            else:
                confidence = Confidence.VERY_LOW
        else:
            confidence = Confidence.MEDIUM

        # Create location reading
        location_reading = LocationReading(
            location=location,
            jurisdiction=jurisdiction,
            context=context,
            active_geofence=active_geofence,
            is_unfamiliar=is_unfamiliar,
            distance_from_home_km=distance_from_home,
        )

        reading = SensorReading(
            sensor_type=self.sensor_type,
            sensor_id=self.sensor_id,
            data=location_reading,
            confidence=confidence,
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
        """Start continuous location monitoring."""
        self._running = True
        self._status = SensorStatus.ACTIVE

        while self._running:
            try:
                await self.read()
                await asyncio.sleep(self.config.location_update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Location sensor error: {e}")
                await asyncio.sleep(1.0)

    async def stop(self) -> None:
        """Stop location monitoring."""
        self._running = False
        self._status = SensorStatus.INACTIVE
        logger.info(f"Location sensor {self.sensor_id} stopped")

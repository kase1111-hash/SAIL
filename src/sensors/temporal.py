"""
Temporal sensor implementation.

Provides time-based context awareness including time of day,
calendar integration, and event awareness.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

from .base import (
    Sensor,
    SensorType,
    SensorStatus,
    SensorReading,
    SensorConfig,
    TimeContext,
    TemporalReading,
    Confidence,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Time Context Definitions
# ============================================================================


# Time ranges for each context (hour ranges, 24-hour format)
TIME_CONTEXT_RANGES: dict[TimeContext, tuple[int, int]] = {
    TimeContext.EARLY_MORNING: (5, 8),    # 5:00 AM - 7:59 AM
    TimeContext.MORNING: (8, 12),          # 8:00 AM - 11:59 AM
    TimeContext.AFTERNOON: (12, 17),       # 12:00 PM - 4:59 PM
    TimeContext.EVENING: (17, 21),         # 5:00 PM - 8:59 PM
    TimeContext.NIGHT: (21, 24),           # 9:00 PM - 11:59 PM
    TimeContext.LATE_NIGHT: (0, 5),        # 12:00 AM - 4:59 AM
}


def get_time_context(hour: int) -> TimeContext:
    """Get time context from hour (0-23)."""
    for context, (start, end) in TIME_CONTEXT_RANGES.items():
        if start <= hour < end:
            return context
    return TimeContext.LATE_NIGHT  # Fallback for edge cases


# ============================================================================
# Calendar Event Types
# ============================================================================


@dataclass
class CalendarEvent:
    """A calendar event."""

    id: str
    title: str
    start_time: datetime
    end_time: datetime
    all_day: bool = False
    location: Optional[str] = None
    description: Optional[str] = None
    event_type: Optional[str] = None  # "work", "personal", "commute", etc.
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_active(self, at_time: Optional[datetime] = None) -> bool:
        """Check if event is currently active."""
        check_time = at_time or datetime.now(self.start_time.tzinfo)
        return self.start_time <= check_time <= self.end_time

    def is_upcoming(self, within_minutes: int = 30, at_time: Optional[datetime] = None) -> bool:
        """Check if event is upcoming within specified minutes."""
        check_time = at_time or datetime.now(self.start_time.tzinfo)
        time_until = self.start_time - check_time
        return timedelta(0) < time_until <= timedelta(minutes=within_minutes)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "all_day": self.all_day,
            "location": self.location,
            "description": self.description,
            "event_type": self.event_type,
            "metadata": self.metadata,
        }


# ============================================================================
# Holiday Definitions
# ============================================================================


# US Federal Holidays (fixed dates)
US_FIXED_HOLIDAYS: dict[tuple[int, int], str] = {
    (1, 1): "New Year's Day",
    (7, 4): "Independence Day",
    (11, 11): "Veterans Day",
    (12, 25): "Christmas Day",
}


def is_weekend(dt: datetime) -> bool:
    """Check if date is a weekend (Saturday or Sunday)."""
    return dt.weekday() >= 5


def is_us_holiday(dt: date) -> tuple[bool, Optional[str]]:
    """Check if date is a US holiday. Returns (is_holiday, holiday_name)."""
    # Check fixed holidays
    key = (dt.month, dt.day)
    if key in US_FIXED_HOLIDAYS:
        return True, US_FIXED_HOLIDAYS[key]

    # Check floating holidays (simplified)
    # MLK Day - Third Monday in January
    if dt.month == 1 and dt.weekday() == 0 and 15 <= dt.day <= 21:
        return True, "Martin Luther King Jr. Day"

    # Presidents Day - Third Monday in February
    if dt.month == 2 and dt.weekday() == 0 and 15 <= dt.day <= 21:
        return True, "Presidents Day"

    # Memorial Day - Last Monday in May
    if dt.month == 5 and dt.weekday() == 0 and dt.day >= 25:
        return True, "Memorial Day"

    # Labor Day - First Monday in September
    if dt.month == 9 and dt.weekday() == 0 and dt.day <= 7:
        return True, "Labor Day"

    # Thanksgiving - Fourth Thursday in November
    if dt.month == 11 and dt.weekday() == 3 and 22 <= dt.day <= 28:
        return True, "Thanksgiving Day"

    return False, None


# ============================================================================
# Calendar Provider Abstraction
# ============================================================================


class CalendarProvider:
    """Abstract calendar data provider."""

    async def get_events(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> list[CalendarEvent]:
        """Get events in time range."""
        raise NotImplementedError

    async def get_current_event(self) -> Optional[CalendarEvent]:
        """Get currently active event."""
        raise NotImplementedError

    async def get_next_event(self) -> Optional[CalendarEvent]:
        """Get next upcoming event."""
        raise NotImplementedError


class SimulatedCalendarProvider(CalendarProvider):
    """Simulated calendar for testing."""

    def __init__(self, events: Optional[list[CalendarEvent]] = None):
        self.events = events or []

    def add_event(self, event: CalendarEvent) -> None:
        """Add an event to the calendar."""
        self.events.append(event)

    async def get_events(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> list[CalendarEvent]:
        """Get events in time range."""
        return [
            e for e in self.events
            if not (e.end_time < start_time or e.start_time > end_time)
        ]

    async def get_current_event(self) -> Optional[CalendarEvent]:
        """Get currently active event."""
        now = datetime.now()
        for event in self.events:
            if event.is_active(now):
                return event
        return None

    async def get_next_event(self) -> Optional[CalendarEvent]:
        """Get next upcoming event."""
        now = datetime.now()
        upcoming = [e for e in self.events if e.start_time > now]
        if upcoming:
            return min(upcoming, key=lambda e: e.start_time)
        return None


class ICalendarProvider(CalendarProvider):
    """Calendar provider using iCalendar files."""

    def __init__(self, calendar_path: Optional[str] = None):
        self.calendar_path = calendar_path
        self._events: list[CalendarEvent] = []

    async def load(self) -> bool:
        """Load calendar from file."""
        if not self.calendar_path:
            return False

        try:
            # Would use icalendar library here
            # from icalendar import Calendar
            logger.info(f"iCalendar loading from {self.calendar_path} (placeholder)")
            return True
        except Exception as e:
            logger.error(f"Failed to load calendar: {e}")
            return False

    async def get_events(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> list[CalendarEvent]:
        """Get events in time range."""
        return [
            e for e in self._events
            if not (e.end_time < start_time or e.start_time > end_time)
        ]

    async def get_current_event(self) -> Optional[CalendarEvent]:
        """Get currently active event."""
        now = datetime.now()
        for event in self._events:
            if event.is_active(now):
                return event
        return None

    async def get_next_event(self) -> Optional[CalendarEvent]:
        """Get next upcoming event."""
        now = datetime.now()
        upcoming = [e for e in self._events if e.start_time > now]
        if upcoming:
            return min(upcoming, key=lambda e: e.start_time)
        return None


# ============================================================================
# Temporal Analysis
# ============================================================================


@dataclass
class TemporalAnalysis:
    """Analyzed temporal data."""

    time_context: TimeContext
    local_time: datetime
    timezone: str
    is_weekend: bool
    is_holiday: bool
    holiday_name: Optional[str] = None

    # Calendar awareness
    current_event: Optional[CalendarEvent] = None
    next_event: Optional[CalendarEvent] = None
    event_context: Optional[str] = None

    # Work/personal context inference
    is_likely_work_hours: bool = False
    is_likely_commute_time: bool = False

    # Risk-relevant timing
    is_late_night: bool = False
    is_early_morning: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "time_context": self.time_context.value,
            "local_time": self.local_time.isoformat(),
            "timezone": self.timezone,
            "is_weekend": self.is_weekend,
            "is_holiday": self.is_holiday,
            "holiday_name": self.holiday_name,
            "current_event": self.current_event.to_dict() if self.current_event else None,
            "next_event": self.next_event.to_dict() if self.next_event else None,
            "event_context": self.event_context,
            "is_likely_work_hours": self.is_likely_work_hours,
            "is_likely_commute_time": self.is_likely_commute_time,
            "is_late_night": self.is_late_night,
            "is_early_morning": self.is_early_morning,
        }


# ============================================================================
# Temporal Sensor
# ============================================================================


class TemporalSensor(Sensor):
    """Temporal sensor for time-based context awareness."""

    def __init__(
        self,
        sensor_id: str = "temporal_primary",
        config: Optional[SensorConfig] = None,
        calendar_provider: Optional[CalendarProvider] = None,
        timezone: Optional[str] = None,
    ):
        super().__init__(sensor_id, SensorType.TEMPORAL, config)

        self.calendar = calendar_provider or SimulatedCalendarProvider()
        self._timezone = timezone or config.timezone if config else "UTC"

        # Work hours configuration (can be customized)
        self.work_start_hour = 9
        self.work_end_hour = 17
        self.commute_morning_hours = (7, 9)
        self.commute_evening_hours = (17, 19)

        # Custom holidays (ISO date strings)
        self._custom_holidays: set[str] = set()
        if config and config.holidays:
            self._custom_holidays = set(config.holidays)

    def set_timezone(self, timezone: str) -> None:
        """Set the timezone for temporal calculations."""
        self._timezone = timezone

    def add_custom_holiday(self, date_str: str) -> None:
        """Add a custom holiday (ISO date format: YYYY-MM-DD)."""
        self._custom_holidays.add(date_str)

    async def initialize(self) -> bool:
        """Initialize the temporal sensor."""
        self._status = SensorStatus.INITIALIZING

        try:
            # Initialize calendar if it has a load method
            if hasattr(self.calendar, "load"):
                await self.calendar.load()

            self._status = SensorStatus.ACTIVE
            logger.info(f"Temporal sensor {self.sensor_id} initialized")
            return True

        except Exception as e:
            self._set_error(f"Initialization failed: {e}")
            return False

    async def read(self) -> SensorReading:
        """Take a temporal reading."""
        try:
            tz = ZoneInfo(self._timezone)
        except Exception:
            tz = ZoneInfo("UTC")
            logger.warning(f"Invalid timezone {self._timezone}, using UTC")

        now = datetime.now(tz)

        # Get time context
        time_context = get_time_context(now.hour)

        # Check weekend
        weekend = is_weekend(now)

        # Check holiday
        holiday, holiday_name = is_us_holiday(now.date())
        if not holiday and now.date().isoformat() in self._custom_holidays:
            holiday = True
            holiday_name = "Custom Holiday"

        # Get calendar events
        current_event = await self.calendar.get_current_event()
        next_event = await self.calendar.get_next_event()

        # Determine event context
        event_context = None
        if current_event:
            event_context = current_event.event_type or "event"
        elif next_event and next_event.is_upcoming(30):
            event_context = f"upcoming_{next_event.event_type or 'event'}"

        # Infer work/commute context
        is_work_hours = (
            not weekend
            and not holiday
            and self.work_start_hour <= now.hour < self.work_end_hour
        )

        is_commute = (
            not weekend
            and not holiday
            and (
                self.commute_morning_hours[0] <= now.hour < self.commute_morning_hours[1]
                or self.commute_evening_hours[0] <= now.hour < self.commute_evening_hours[1]
            )
        )

        # Create analysis
        analysis = TemporalAnalysis(
            time_context=time_context,
            local_time=now,
            timezone=self._timezone,
            is_weekend=weekend,
            is_holiday=holiday,
            holiday_name=holiday_name,
            current_event=current_event,
            next_event=next_event,
            event_context=event_context,
            is_likely_work_hours=is_work_hours,
            is_likely_commute_time=is_commute,
            is_late_night=time_context == TimeContext.LATE_NIGHT,
            is_early_morning=time_context == TimeContext.EARLY_MORNING,
        )

        # Create temporal reading
        temporal_reading = TemporalReading(
            time_context=time_context,
            is_weekend=weekend,
            is_holiday=holiday,
            local_time=now,
            timezone=self._timezone,
            calendar_event=current_event.title if current_event else None,
            event_context=event_context,
        )

        reading = SensorReading(
            sensor_type=self.sensor_type,
            sensor_id=self.sensor_id,
            data=analysis,
            confidence=Confidence.VERY_HIGH,  # Time is always accurate
            metadata={"temporal_reading": temporal_reading.to_dict()},
        )

        self._set_reading(reading)
        return reading

    async def start(self) -> None:
        """Start continuous temporal monitoring."""
        self._running = True
        self._status = SensorStatus.ACTIVE

        while self._running:
            try:
                await self.read()
                await asyncio.sleep(self.config.temporal_update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Temporal sensor error: {e}")
                await asyncio.sleep(1.0)

    async def stop(self) -> None:
        """Stop temporal monitoring."""
        self._running = False
        self._status = SensorStatus.INACTIVE
        logger.info(f"Temporal sensor {self.sensor_id} stopped")


# ============================================================================
# Utility Functions
# ============================================================================


def format_time_until(target: datetime, from_time: Optional[datetime] = None) -> str:
    """Format time until target as human-readable string."""
    now = from_time or datetime.now(target.tzinfo)
    delta = target - now

    if delta.total_seconds() < 0:
        return "in the past"

    total_minutes = int(delta.total_seconds() / 60)
    hours, minutes = divmod(total_minutes, 60)

    if hours == 0:
        return f"in {minutes} minute{'s' if minutes != 1 else ''}"
    elif minutes == 0:
        return f"in {hours} hour{'s' if hours != 1 else ''}"
    else:
        return f"in {hours} hour{'s' if hours != 1 else ''} and {minutes} minute{'s' if minutes != 1 else ''}"


def get_day_period(hour: int) -> str:
    """Get simple day period description."""
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"

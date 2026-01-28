"""
SAIL Guardian Alert System

Provides guardian notification and alert management for the
multi-user family system. Handles parent/guardian alerts when
dependents encounter concerning situations.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from src.users.base import (
    User,
    UserEvent,
    UserEventType,
    UserManagerConfig,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Guardian Alert Types
# ============================================================================


class GuardianAlertLevel(str, Enum):
    """Severity levels for guardian alerts."""

    INFO = "info"               # Informational, no action needed
    NOTICE = "notice"           # Worth knowing, low priority
    WARNING = "warning"         # Concerning, should review
    URGENT = "urgent"           # Requires prompt attention
    EMERGENCY = "emergency"     # Immediate action needed


class GuardianAlertType(str, Enum):
    """Types of guardian alerts."""

    # Safety alerts
    CRISIS_MODE_ACTIVATED = "crisis_mode_activated"
    HIGH_RISK_DETECTED = "high_risk_detected"
    THREAT_DETECTED = "threat_detected"
    ACCIDENT_DETECTED = "accident_detected"

    # Behavior alerts
    REPEATED_ALERTS_DISMISSED = "repeated_alerts_dismissed"
    RESTRICTION_VIOLATION_ATTEMPT = "restriction_violation_attempt"
    LATE_NIGHT_ACTIVITY = "late_night_activity"
    UNFAMILIAR_LOCATION = "unfamiliar_location"

    # System alerts
    LOCATION_SHARING_REQUESTED = "location_sharing_requested"
    SETTINGS_CHANGE_REQUESTED = "settings_change_requested"
    VOICE_NOT_RECOGNIZED = "voice_not_recognized"

    # Check-in alerts
    CHECK_IN_MISSED = "check_in_missed"
    CHECK_IN_RECEIVED = "check_in_received"

    # Emergency
    EMERGENCY_KEYWORD = "emergency_keyword"
    SILENT_ALERT = "silent_alert"


class NotificationChannel(str, Enum):
    """Channels for delivering notifications."""

    VOICE = "voice"             # Voice announcement in system
    PUSH = "push"               # Push notification (mobile app)
    SMS = "sms"                 # Text message
    EMAIL = "email"             # Email notification
    IN_APP = "in_app"           # In-app notification queue


# ============================================================================
# Guardian Alert Model
# ============================================================================


@dataclass
class GuardianAlert:
    """
    A guardian alert to be sent to a parent/guardian.
    """

    alert_id: str = field(default_factory=lambda: f"alert_{datetime.now().strftime('%Y%m%d%H%M%S%f')}")
    alert_type: GuardianAlertType = GuardianAlertType.HIGH_RISK_DETECTED
    level: GuardianAlertLevel = GuardianAlertLevel.WARNING

    # Who the alert is about and for
    dependent_id: str = ""
    dependent_name: str = ""
    guardian_id: str = ""
    guardian_name: str = ""

    # Alert content
    title: str = ""
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    # Location context
    location: dict[str, Any] | None = None

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    acknowledged_at: datetime | None = None

    # Delivery
    channels: list[NotificationChannel] = field(default_factory=lambda: [NotificationChannel.IN_APP])
    delivered_via: list[NotificationChannel] = field(default_factory=list)
    delivery_attempts: int = 0

    # State
    is_acknowledged: bool = False
    is_resolved: bool = False
    resolution_note: str = ""

    # Actions
    available_actions: list[str] = field(default_factory=list)
    action_taken: str | None = None

    def acknowledge(self, note: str = "") -> None:
        """Acknowledge the alert."""
        self.is_acknowledged = True
        self.acknowledged_at = datetime.now()
        if note:
            self.resolution_note = note

    def resolve(self, action: str, note: str = "") -> None:
        """Mark alert as resolved."""
        self.is_resolved = True
        self.action_taken = action
        self.resolution_note = note
        if not self.is_acknowledged:
            self.acknowledge()

    @property
    def is_expired(self) -> bool:
        """Check if alert has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    @property
    def age_seconds(self) -> float:
        """Get age of alert in seconds."""
        return (datetime.now() - self.created_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "level": self.level.value,
            "dependent_id": self.dependent_id,
            "dependent_name": self.dependent_name,
            "guardian_id": self.guardian_id,
            "guardian_name": self.guardian_name,
            "title": self.title,
            "message": self.message,
            "details": self.details,
            "location": self.location,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "channels": [c.value for c in self.channels],
            "delivered_via": [c.value for c in self.delivered_via],
            "delivery_attempts": self.delivery_attempts,
            "is_acknowledged": self.is_acknowledged,
            "is_resolved": self.is_resolved,
            "resolution_note": self.resolution_note,
            "available_actions": self.available_actions,
            "action_taken": self.action_taken,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GuardianAlert:
        """Create from dictionary."""
        alert = cls(
            alert_id=data.get("alert_id", ""),
            alert_type=GuardianAlertType(data.get("alert_type", "high_risk_detected")),
            level=GuardianAlertLevel(data.get("level", "warning")),
            dependent_id=data.get("dependent_id", ""),
            dependent_name=data.get("dependent_name", ""),
            guardian_id=data.get("guardian_id", ""),
            guardian_name=data.get("guardian_name", ""),
            title=data.get("title", ""),
            message=data.get("message", ""),
            details=data.get("details", {}),
            location=data.get("location"),
            channels=[NotificationChannel(c) for c in data.get("channels", ["in_app"])],
            delivered_via=[NotificationChannel(c) for c in data.get("delivered_via", [])],
            delivery_attempts=data.get("delivery_attempts", 0),
            is_acknowledged=data.get("is_acknowledged", False),
            is_resolved=data.get("is_resolved", False),
            resolution_note=data.get("resolution_note", ""),
            available_actions=data.get("available_actions", []),
            action_taken=data.get("action_taken"),
        )

        if data.get("created_at"):
            alert.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("expires_at"):
            alert.expires_at = datetime.fromisoformat(data["expires_at"])
        if data.get("acknowledged_at"):
            alert.acknowledged_at = datetime.fromisoformat(data["acknowledged_at"])

        return alert


# ============================================================================
# Alert Templates
# ============================================================================


ALERT_TEMPLATES: dict[GuardianAlertType, dict[str, Any]] = {
    GuardianAlertType.CRISIS_MODE_ACTIVATED: {
        "level": GuardianAlertLevel.EMERGENCY,
        "title": "Crisis Mode Activated",
        "message": "{dependent_name} triggered crisis mode. SAIL is providing emergency guidance.",
        "channels": [NotificationChannel.PUSH, NotificationChannel.SMS, NotificationChannel.IN_APP],
        "actions": ["call_dependent", "view_location", "call_911", "take_control"],
    },
    GuardianAlertType.HIGH_RISK_DETECTED: {
        "level": GuardianAlertLevel.URGENT,
        "title": "High Risk Situation Detected",
        "message": "SAIL detected a potentially concerning situation for {dependent_name}.",
        "channels": [NotificationChannel.PUSH, NotificationChannel.IN_APP],
        "actions": ["view_details", "contact_dependent", "take_control"],
    },
    GuardianAlertType.THREAT_DETECTED: {
        "level": GuardianAlertLevel.URGENT,
        "title": "Potential Threat Detected",
        "message": "SAIL detected potential threat indicators in {dependent_name}'s environment.",
        "channels": [NotificationChannel.PUSH, NotificationChannel.IN_APP],
        "actions": ["view_details", "contact_dependent", "view_location"],
    },
    GuardianAlertType.ACCIDENT_DETECTED: {
        "level": GuardianAlertLevel.EMERGENCY,
        "title": "Possible Accident Detected",
        "message": "SAIL detected a possible accident or sudden impact for {dependent_name}.",
        "channels": [NotificationChannel.PUSH, NotificationChannel.SMS, NotificationChannel.IN_APP],
        "actions": ["call_dependent", "view_location", "call_911"],
    },
    GuardianAlertType.REPEATED_ALERTS_DISMISSED: {
        "level": GuardianAlertLevel.WARNING,
        "title": "Safety Alerts Being Dismissed",
        "message": "{dependent_name} has dismissed multiple safety alerts.",
        "channels": [NotificationChannel.IN_APP],
        "actions": ["view_details", "contact_dependent"],
    },
    GuardianAlertType.RESTRICTION_VIOLATION_ATTEMPT: {
        "level": GuardianAlertLevel.NOTICE,
        "title": "Restriction Triggered",
        "message": "{dependent_name} attempted an action blocked by their restrictions.",
        "channels": [NotificationChannel.IN_APP],
        "actions": ["view_details", "modify_restrictions"],
    },
    GuardianAlertType.LATE_NIGHT_ACTIVITY: {
        "level": GuardianAlertLevel.INFO,
        "title": "Late Night Activity",
        "message": "{dependent_name} is active late at night.",
        "channels": [NotificationChannel.IN_APP],
        "actions": ["view_details"],
    },
    GuardianAlertType.UNFAMILIAR_LOCATION: {
        "level": GuardianAlertLevel.NOTICE,
        "title": "Unfamiliar Location",
        "message": "{dependent_name} is in an unfamiliar area.",
        "channels": [NotificationChannel.IN_APP],
        "actions": ["view_location", "contact_dependent"],
    },
    GuardianAlertType.LOCATION_SHARING_REQUESTED: {
        "level": GuardianAlertLevel.INFO,
        "title": "Location Sharing Request",
        "message": "{dependent_name} wants to share their location with someone.",
        "channels": [NotificationChannel.IN_APP],
        "actions": ["approve", "deny", "view_details"],
    },
    GuardianAlertType.SETTINGS_CHANGE_REQUESTED: {
        "level": GuardianAlertLevel.INFO,
        "title": "Settings Change Request",
        "message": "{dependent_name} requested a settings change that requires your approval.",
        "channels": [NotificationChannel.IN_APP],
        "actions": ["approve", "deny", "view_details"],
    },
    GuardianAlertType.CHECK_IN_MISSED: {
        "level": GuardianAlertLevel.WARNING,
        "title": "Missed Check-In",
        "message": "{dependent_name} missed their scheduled check-in.",
        "channels": [NotificationChannel.PUSH, NotificationChannel.IN_APP],
        "actions": ["contact_dependent", "view_location"],
    },
    GuardianAlertType.EMERGENCY_KEYWORD: {
        "level": GuardianAlertLevel.EMERGENCY,
        "title": "Emergency Keyword Detected",
        "message": "{dependent_name} used an emergency keyword.",
        "channels": [NotificationChannel.PUSH, NotificationChannel.SMS, NotificationChannel.IN_APP],
        "actions": ["call_dependent", "view_location", "call_911", "take_control"],
    },
    GuardianAlertType.SILENT_ALERT: {
        "level": GuardianAlertLevel.EMERGENCY,
        "title": "Silent Alert Triggered",
        "message": "{dependent_name} triggered a silent alert (may be in danger).",
        "channels": [NotificationChannel.PUSH, NotificationChannel.SMS, NotificationChannel.IN_APP],
        "actions": ["view_location", "call_911", "listen_in"],
    },
}


# ============================================================================
# Guardian Alert System
# ============================================================================


class GuardianAlertSystem:
    """
    System for managing guardian alerts and notifications.
    """

    def __init__(self, config: UserManagerConfig | None = None):
        self.config = config or UserManagerConfig()

        # Alert queue and history
        # Use unlimited deque to prevent silent data loss
        # Cleanup of old alerts is done in deliver_pending_alerts
        self._pending_alerts: deque[GuardianAlert] = deque()
        self._alert_history: dict[str, GuardianAlert] = {}
        # Maximum alerts to retain in queue before warning
        self._max_pending_alerts = 500

        # Cooldown tracking (guardian_id -> alert_type -> last_sent)
        self._cooldowns: dict[str, dict[str, datetime]] = {}

        # Notification handlers
        self._notification_handlers: dict[NotificationChannel, Callable] = {}

        # Event callbacks
        self._event_callbacks: list[Callable[[UserEvent], None]] = []

        # Statistics
        self._alerts_created = 0
        self._alerts_delivered = 0
        self._alerts_acknowledged = 0

    def create_alert(
        self,
        alert_type: GuardianAlertType,
        dependent: User,
        guardian: User,
        details: dict[str, Any] | None = None,
        location: dict[str, Any] | None = None,
        custom_message: str | None = None,
    ) -> GuardianAlert | None:
        """
        Create a guardian alert.

        Args:
            alert_type: Type of alert
            dependent: The dependent user the alert is about
            guardian: The guardian to notify
            details: Additional details
            location: Location information
            custom_message: Custom message override

        Returns:
            GuardianAlert if created (None if suppressed by cooldown)
        """
        # Check cooldown
        if self._is_in_cooldown(guardian.user_id, alert_type):
            logger.debug(f"Alert {alert_type.value} suppressed by cooldown for {guardian.name}")
            return None

        # Get template
        template = ALERT_TEMPLATES.get(alert_type, {
            "level": GuardianAlertLevel.INFO,
            "title": alert_type.value.replace("_", " ").title(),
            "message": f"Alert for {dependent.name}",
            "channels": [NotificationChannel.IN_APP],
            "actions": [],
        })

        # Build message
        message = custom_message or template["message"].format(
            dependent_name=dependent.name,
            guardian_name=guardian.name,
        )

        # Create alert
        alert = GuardianAlert(
            alert_type=alert_type,
            level=template["level"],
            dependent_id=dependent.user_id,
            dependent_name=dependent.name,
            guardian_id=guardian.user_id,
            guardian_name=guardian.name,
            title=template["title"],
            message=message,
            details=details or {},
            location=location,
            channels=template.get("channels", [NotificationChannel.IN_APP]),
            available_actions=template.get("actions", []),
        )

        # Set expiration for lower priority alerts
        if alert.level in (GuardianAlertLevel.INFO, GuardianAlertLevel.NOTICE):
            alert.expires_at = datetime.now() + timedelta(hours=24)
        elif alert.level == GuardianAlertLevel.WARNING:
            alert.expires_at = datetime.now() + timedelta(hours=48)

        # Store and queue
        self._alert_history[alert.alert_id] = alert
        self._pending_alerts.append(alert)
        self._alerts_created += 1

        # Check queue size and log warning if growing too large
        if len(self._pending_alerts) > self._max_pending_alerts:
            logger.warning(
                f"Guardian alert queue is large ({len(self._pending_alerts)} alerts). "
                "Consider calling deliver_pending_alerts() more frequently."
            )
            # Clean up expired alerts to reclaim memory
            self._cleanup_expired_pending_alerts()

        # Update cooldown
        self._set_cooldown(guardian.user_id, alert_type)

        # Emit event
        self._emit_event(UserEvent(
            event_type=UserEventType.RESTRICTION_TRIGGERED,  # Using as guardian alert event
            user_id=dependent.user_id,
            data={
                "alert_id": alert.alert_id,
                "alert_type": alert_type.value,
                "guardian_id": guardian.user_id,
            },
        ))

        logger.info(f"Created {alert_type.value} alert for {guardian.name} about {dependent.name}")
        return alert

    def _is_in_cooldown(self, guardian_id: str, alert_type: GuardianAlertType) -> bool:
        """Check if an alert type is in cooldown for a guardian."""
        if not self.config.guardian_alerts_enabled:
            return True

        guardian_cooldowns = self._cooldowns.get(guardian_id, {})
        last_sent = guardian_cooldowns.get(alert_type.value)

        if last_sent is None:
            return False

        elapsed = (datetime.now() - last_sent).total_seconds()
        return elapsed < self.config.guardian_alert_cooldown_seconds

    def _set_cooldown(self, guardian_id: str, alert_type: GuardianAlertType) -> None:
        """Set cooldown for an alert type."""
        if guardian_id not in self._cooldowns:
            self._cooldowns[guardian_id] = {}
        self._cooldowns[guardian_id][alert_type.value] = datetime.now()

    def _cleanup_expired_pending_alerts(self) -> None:
        """Remove expired alerts from the pending queue."""
        non_expired = deque()
        while self._pending_alerts:
            alert = self._pending_alerts.popleft()
            if not alert.is_expired:
                non_expired.append(alert)
        self._pending_alerts = non_expired

    async def deliver_pending_alerts(self) -> int:
        """
        Deliver all pending alerts.

        Returns:
            Number of alerts delivered
        """
        delivered = 0

        while self._pending_alerts:
            alert = self._pending_alerts.popleft()

            if alert.is_expired:
                continue

            success = await self._deliver_alert(alert)
            if success:
                delivered += 1
                self._alerts_delivered += 1

        return delivered

    async def _deliver_alert(self, alert: GuardianAlert) -> bool:
        """Deliver a single alert through its channels."""
        alert.delivery_attempts += 1
        delivered_any = False

        for channel in alert.channels:
            if channel in alert.delivered_via:
                continue

            handler = self._notification_handlers.get(channel)
            if handler:
                try:
                    await handler(alert)
                    alert.delivered_via.append(channel)
                    delivered_any = True
                except Exception as e:
                    logger.error(f"Failed to deliver alert via {channel.value}: {e}")
            # Default handling for in-app notifications
            elif channel == NotificationChannel.IN_APP:
                alert.delivered_via.append(channel)
                delivered_any = True
                logger.info(f"Alert {alert.alert_id} queued for in-app delivery")

        return delivered_any

    def register_notification_handler(
        self,
        channel: NotificationChannel,
        handler: Callable,
    ) -> None:
        """Register a handler for a notification channel."""
        self._notification_handlers[channel] = handler

    def acknowledge_alert(self, alert_id: str, note: str = "") -> bool:
        """Acknowledge an alert."""
        alert = self._alert_history.get(alert_id)
        if alert is None:
            return False

        alert.acknowledge(note)
        self._alerts_acknowledged += 1
        return True

    def resolve_alert(self, alert_id: str, action: str, note: str = "") -> bool:
        """Resolve an alert with an action."""
        alert = self._alert_history.get(alert_id)
        if alert is None:
            return False

        alert.resolve(action, note)
        return True

    def get_pending_alerts(self, guardian_id: str) -> list[GuardianAlert]:
        """Get pending alerts for a guardian."""
        return [
            alert for alert in self._alert_history.values()
            if alert.guardian_id == guardian_id
            and not alert.is_resolved
            and not alert.is_expired
        ]

    def get_alert(self, alert_id: str) -> GuardianAlert | None:
        """Get a specific alert."""
        return self._alert_history.get(alert_id)

    def get_alert_history(
        self,
        guardian_id: str | None = None,
        dependent_id: str | None = None,
        limit: int = 50,
    ) -> list[GuardianAlert]:
        """Get alert history with optional filters."""
        alerts = list(self._alert_history.values())

        if guardian_id:
            alerts = [a for a in alerts if a.guardian_id == guardian_id]

        if dependent_id:
            alerts = [a for a in alerts if a.dependent_id == dependent_id]

        # Sort by creation time, newest first
        alerts.sort(key=lambda a: a.created_at, reverse=True)

        return alerts[:limit]

    def register_event_callback(self, callback: Callable[[UserEvent], None]) -> None:
        """Register callback for guardian alert events."""
        self._event_callbacks.append(callback)

    def _emit_event(self, event: UserEvent) -> None:
        """Emit a user event."""
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get guardian alert statistics."""
        pending_count = len([
            a for a in self._alert_history.values()
            if not a.is_resolved and not a.is_expired
        ])

        return {
            "alerts_created": self._alerts_created,
            "alerts_delivered": self._alerts_delivered,
            "alerts_acknowledged": self._alerts_acknowledged,
            "pending_alerts": pending_count,
            "total_in_history": len(self._alert_history),
        }


# ============================================================================
# Trigger Detection
# ============================================================================


class GuardianAlertTrigger:
    """
    Detects conditions that should trigger guardian alerts.
    """

    def __init__(self, alert_system: GuardianAlertSystem):
        self._alert_system = alert_system

        # Thresholds
        self._dismissed_alert_threshold = 3
        self._high_risk_score_threshold = 0.7

        # Tracking
        self._dismissed_counts: dict[str, int] = {}

    def check_for_triggers(
        self,
        dependent: User,
        guardian: User,
        context: dict[str, Any],
    ) -> list[GuardianAlert]:
        """
        Check context for alert triggers.

        Args:
            dependent: The dependent user
            guardian: Their guardian
            context: Current situational context

        Returns:
            List of alerts created
        """
        alerts = []

        # Check intervention mode
        if context.get("intervention_mode") == "crisis":
            alert = self._alert_system.create_alert(
                GuardianAlertType.CRISIS_MODE_ACTIVATED,
                dependent,
                guardian,
                details=context,
                location=context.get("location"),
            )
            if alert:
                alerts.append(alert)

        # Check risk score
        risk_score = context.get("risk_score", 0)
        if risk_score >= self._high_risk_score_threshold:
            alert = self._alert_system.create_alert(
                GuardianAlertType.HIGH_RISK_DETECTED,
                dependent,
                guardian,
                details={"risk_score": risk_score, "risk_factors": context.get("risk_factors", [])},
                location=context.get("location"),
            )
            if alert:
                alerts.append(alert)

        # Check for accident detection
        if context.get("accident_detected"):
            alert = self._alert_system.create_alert(
                GuardianAlertType.ACCIDENT_DETECTED,
                dependent,
                guardian,
                details=context,
                location=context.get("location"),
            )
            if alert:
                alerts.append(alert)

        # Check for late night activity
        if context.get("is_late_night") and dependent.is_minor:
            alert = self._alert_system.create_alert(
                GuardianAlertType.LATE_NIGHT_ACTIVITY,
                dependent,
                guardian,
                details={"time": datetime.now().isoformat()},
            )
            if alert:
                alerts.append(alert)

        # Check for unfamiliar location
        if context.get("location_context") == "unfamiliar" and dependent.is_minor:
            alert = self._alert_system.create_alert(
                GuardianAlertType.UNFAMILIAR_LOCATION,
                dependent,
                guardian,
                location=context.get("location"),
            )
            if alert:
                alerts.append(alert)

        return alerts

    def record_alert_dismissal(self, dependent_id: str) -> bool:
        """
        Record that a dependent dismissed an alert.

        Returns:
            True if threshold reached and guardian should be notified
        """
        count = self._dismissed_counts.get(dependent_id, 0) + 1
        self._dismissed_counts[dependent_id] = count

        return count >= self._dismissed_alert_threshold

    def reset_dismissal_count(self, dependent_id: str) -> None:
        """Reset dismissal count for a dependent."""
        self._dismissed_counts[dependent_id] = 0


# ============================================================================
# Factory Functions
# ============================================================================


def create_guardian_alert_system(
    config: UserManagerConfig | None = None,
) -> GuardianAlertSystem:
    """Create a guardian alert system."""
    return GuardianAlertSystem(config)

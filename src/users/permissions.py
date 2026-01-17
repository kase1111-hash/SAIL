"""
SAIL Role-Based Access Control System

Provides permission checking and access control enforcement
for the multi-user family system.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.users.base import (
    AccessLevel,
    User,
    UserEvent,
    UserEventType,
    UserRestriction,
    UserRole,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Permission Types
# ============================================================================


class Permission(str, Enum):
    """Available permissions in the system."""

    # System settings
    CHANGE_SETTINGS = "change_settings"
    VIEW_SETTINGS = "view_settings"
    MANAGE_USERS = "manage_users"

    # Intervention modes
    OVERRIDE_GUARDIAN_MODE = "override_guardian_mode"
    OVERRIDE_CRISIS_MODE = "override_crisis_mode"
    CHANGE_INTERVENTION_MODE = "change_intervention_mode"

    # Alerts and notifications
    DISMISS_ALERTS = "dismiss_alerts"
    CONFIGURE_ALERTS = "configure_alerts"

    # Privacy and data
    VIEW_OWN_DATA = "view_own_data"
    DELETE_OWN_DATA = "delete_own_data"
    VIEW_DEPENDENT_DATA = "view_dependent_data"
    SHARE_LOCATION = "share_location"

    # Knowledge access
    ACCESS_ALL_KNOWLEDGE = "access_all_knowledge"
    ACCESS_AGE_APPROPRIATE = "access_age_appropriate"

    # Guardian functions
    GUARDIAN_TAKEOVER = "guardian_takeover"
    VIEW_GUARDIAN_ALERTS = "view_guardian_alerts"
    CONFIGURE_DEPENDENT = "configure_dependent"


class PermissionContext(str, Enum):
    """Contextual factors that affect permissions."""

    DRIVING = "driving"
    CRISIS_MODE = "crisis_mode"
    GUARDIAN_MODE = "guardian_mode"
    LATE_NIGHT = "late_night"
    UNFAMILIAR_LOCATION = "unfamiliar_location"


# ============================================================================
# Permission Result
# ============================================================================


@dataclass
class PermissionResult:
    """Result of a permission check."""

    allowed: bool
    permission: Permission
    reason: str = ""
    requires_guardian: bool = False
    guardian_notified: bool = False

    # Context that affected the decision
    context_factors: list[PermissionContext] = field(default_factory=list)
    restrictions_triggered: list[UserRestriction] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "permission": self.permission.value,
            "reason": self.reason,
            "requires_guardian": self.requires_guardian,
            "guardian_notified": self.guardian_notified,
            "context_factors": [c.value for c in self.context_factors],
            "restrictions_triggered": [r.value for r in self.restrictions_triggered],
        }


# ============================================================================
# Permission Rules
# ============================================================================


# Default permission matrix by role
ROLE_PERMISSIONS: dict[UserRole, set[Permission]] = {
    UserRole.ADMIN: {
        Permission.CHANGE_SETTINGS,
        Permission.VIEW_SETTINGS,
        Permission.MANAGE_USERS,
        Permission.OVERRIDE_GUARDIAN_MODE,
        Permission.OVERRIDE_CRISIS_MODE,
        Permission.CHANGE_INTERVENTION_MODE,
        Permission.DISMISS_ALERTS,
        Permission.CONFIGURE_ALERTS,
        Permission.VIEW_OWN_DATA,
        Permission.DELETE_OWN_DATA,
        Permission.VIEW_DEPENDENT_DATA,
        Permission.SHARE_LOCATION,
        Permission.ACCESS_ALL_KNOWLEDGE,
        Permission.GUARDIAN_TAKEOVER,
        Permission.VIEW_GUARDIAN_ALERTS,
        Permission.CONFIGURE_DEPENDENT,
    },
    UserRole.USER: {
        Permission.VIEW_SETTINGS,
        Permission.OVERRIDE_GUARDIAN_MODE,
        Permission.CHANGE_INTERVENTION_MODE,
        Permission.DISMISS_ALERTS,
        Permission.VIEW_OWN_DATA,
        Permission.DELETE_OWN_DATA,
        Permission.SHARE_LOCATION,
        Permission.ACCESS_ALL_KNOWLEDGE,
        Permission.VIEW_GUARDIAN_ALERTS,
        Permission.CONFIGURE_DEPENDENT,
    },
    UserRole.MINOR: {
        Permission.VIEW_SETTINGS,
        Permission.VIEW_OWN_DATA,
        Permission.ACCESS_AGE_APPROPRIATE,
    },
}

# Access level modifiers
ACCESS_LEVEL_GRANTS: dict[AccessLevel, set[Permission]] = {
    AccessLevel.FULL: {
        Permission.CHANGE_SETTINGS,
        Permission.OVERRIDE_GUARDIAN_MODE,
        Permission.OVERRIDE_CRISIS_MODE,
    },
    AccessLevel.STANDARD: {
        Permission.CHANGE_INTERVENTION_MODE,
        Permission.DISMISS_ALERTS,
    },
    AccessLevel.LIMITED: set(),
}


# ============================================================================
# Permission Checker
# ============================================================================


class PermissionChecker:
    """
    Checks permissions for users based on role, restrictions, and context.
    """

    def __init__(self):
        self._event_callbacks: list[Callable[[UserEvent], None]] = []

    def check_permission(
        self,
        user: User,
        permission: Permission,
        context: dict[str, Any] | None = None,
    ) -> PermissionResult:
        """
        Check if a user has a specific permission.

        Args:
            user: User to check
            permission: Permission to check
            context: Optional situational context

        Returns:
            PermissionResult with decision and details
        """
        context = context or {}
        context_factors = self._get_context_factors(context)

        # Start with role-based permissions
        role_perms = ROLE_PERMISSIONS.get(user.role, set())
        has_base_permission = permission in role_perms

        # Check access level grants
        access_grants = ACCESS_LEVEL_GRANTS.get(user.access_level, set())
        has_access_grant = permission in access_grants

        # Check restrictions
        restrictions_triggered = self._check_restrictions(user, permission, context_factors)

        # Build result
        result = PermissionResult(
            permission=permission,
            context_factors=context_factors,
            restrictions_triggered=restrictions_triggered,
        )

        # Determine if allowed
        if restrictions_triggered:
            result.allowed = False
            result.reason = self._get_restriction_reason(restrictions_triggered[0])
            result.requires_guardian = user.has_guardian
        elif has_base_permission or has_access_grant:
            result.allowed = True
            result.reason = "Permission granted"
        else:
            result.allowed = False
            result.reason = f"Permission '{permission.value}' not available for role '{user.role.value}'"

        # Log if permission denied
        if not result.allowed:
            self._emit_event(UserEvent(
                event_type=UserEventType.PERMISSION_DENIED,
                user_id=user.user_id,
                data={
                    "permission": permission.value,
                    "reason": result.reason,
                    "restrictions": [r.value for r in restrictions_triggered],
                },
            ))

        return result

    def _get_context_factors(self, context: dict[str, Any]) -> list[PermissionContext]:
        """Extract permission context factors from situational context."""
        factors = []

        if context.get("is_driving") or context.get("motion_state") == "driving":
            factors.append(PermissionContext.DRIVING)

        if context.get("intervention_mode") == "crisis":
            factors.append(PermissionContext.CRISIS_MODE)

        if context.get("intervention_mode") == "guardian":
            factors.append(PermissionContext.GUARDIAN_MODE)

        if context.get("is_late_night"):
            factors.append(PermissionContext.LATE_NIGHT)

        if context.get("location_context") == "unfamiliar":
            factors.append(PermissionContext.UNFAMILIAR_LOCATION)

        return factors

    def _check_restrictions(
        self,
        user: User,
        permission: Permission,
        context_factors: list[PermissionContext],
    ) -> list[UserRestriction]:
        """Check which restrictions are triggered."""
        triggered = []

        # Check driving-related restrictions
        if PermissionContext.DRIVING in context_factors:
            if (
                permission == Permission.OVERRIDE_GUARDIAN_MODE
                and user.has_restriction(UserRestriction.NO_OVERRIDE_GUARDIAN_WHILE_DRIVING)
            ):
                triggered.append(UserRestriction.NO_OVERRIDE_GUARDIAN_WHILE_DRIVING)

        # Check alert restrictions
        if permission == Permission.DISMISS_ALERTS:
            if user.has_restriction(UserRestriction.NO_DISABLE_ALERTS):
                triggered.append(UserRestriction.NO_DISABLE_ALERTS)

        # Check settings restrictions
        if permission == Permission.CHANGE_SETTINGS:
            if user.has_restriction(UserRestriction.REQUIRE_GUARDIAN_FOR_SETTINGS):
                triggered.append(UserRestriction.REQUIRE_GUARDIAN_FOR_SETTINGS)

        # Check crisis mode restrictions
        if permission == Permission.OVERRIDE_CRISIS_MODE:
            if user.has_restriction(UserRestriction.NO_CRISIS_MODE_DISABLE):
                triggered.append(UserRestriction.NO_CRISIS_MODE_DISABLE)

        # Check location sharing restrictions
        if permission == Permission.SHARE_LOCATION:
            if user.has_restriction(UserRestriction.REQUIRE_GUARDIAN_FOR_LOCATION_SHARE):
                triggered.append(UserRestriction.REQUIRE_GUARDIAN_FOR_LOCATION_SHARE)

        return triggered

    def _get_restriction_reason(self, restriction: UserRestriction) -> str:
        """Get human-readable reason for a restriction."""
        reasons = {
            UserRestriction.NO_OVERRIDE_GUARDIAN_WHILE_DRIVING:
                "Cannot override guardian mode while driving",
            UserRestriction.NO_DISABLE_ALERTS:
                "Not permitted to disable safety alerts",
            UserRestriction.REQUIRE_GUARDIAN_FOR_SETTINGS:
                "Guardian approval required to change settings",
            UserRestriction.NO_CRISIS_MODE_DISABLE:
                "Cannot disable crisis mode",
            UserRestriction.REQUIRE_GUARDIAN_FOR_LOCATION_SHARE:
                "Guardian approval required for location sharing",
        }
        return reasons.get(restriction, "Restriction applies")

    def can_access_knowledge_domain(
        self,
        user: User,
        domain: str,
        category: str | None = None,
    ) -> bool:
        """
        Check if user can access a knowledge domain/category.

        Args:
            user: User to check
            domain: Knowledge domain
            category: Optional category within domain

        Returns:
            True if access allowed
        """
        # Admins and users with full access can access everything
        if user.is_admin or user.access_level == AccessLevel.FULL:
            return True

        # Check for all knowledge access
        if Permission.ACCESS_ALL_KNOWLEDGE in ROLE_PERMISSIONS.get(user.role, set()):
            return True

        # Minors have age-appropriate access
        if user.role == UserRole.MINOR:
            return self._is_age_appropriate(domain, category)

        return True

    def _is_age_appropriate(self, domain: str, category: str | None) -> bool:
        """
        Check if content is age-appropriate.

        This is a simplified check - real implementation would be more nuanced.
        """
        # Most safety and emergency content is always appropriate
        if domain in ("emergency", "safety"):
            return True

        # Legal content may have restrictions
        if domain == "legal":
            # Driving-related legal content for teen drivers
            if category in ("traffic_stop", "police_interaction"):
                return True
            # Other legal content might need filtering
            return False

        # Financial content needs care
        if domain == "financial":
            # Basic scam awareness is appropriate
            if category in ("scam_detection", "phishing"):
                return True
            return False

        return True

    def get_user_permissions(self, user: User) -> set[Permission]:
        """Get all permissions available to a user."""
        perms = ROLE_PERMISSIONS.get(user.role, set()).copy()
        perms.update(ACCESS_LEVEL_GRANTS.get(user.access_level, set()))
        return perms

    def register_event_callback(self, callback: Callable[[UserEvent], None]) -> None:
        """Register callback for permission events."""
        self._event_callbacks.append(callback)

    def _emit_event(self, event: UserEvent) -> None:
        """Emit a user event."""
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")


# ============================================================================
# Permission Enforcement Decorator
# ============================================================================


def require_permission(permission: Permission):
    """
    Decorator to enforce permission on a method.

    Usage:
        @require_permission(Permission.CHANGE_SETTINGS)
        async def change_setting(self, user: User, setting: str, value: Any):
            ...
    """
    def decorator(func):
        async def wrapper(self, user: User, *args, **kwargs):
            checker = PermissionChecker()
            context = kwargs.get("context", {})

            result = checker.check_permission(user, permission, context)

            if not result.allowed:
                raise PermissionError(
                    f"Permission denied: {result.reason}"
                )

            return await func(self, user, *args, **kwargs)

        return wrapper
    return decorator


# ============================================================================
# Guardian Permission Helpers
# ============================================================================


class GuardianPermissionHelper:
    """
    Helper for guardian-related permission operations.
    """

    @staticmethod
    def can_guardian_access_dependent(
        guardian: User,
        dependent: User,
    ) -> bool:
        """Check if guardian can access dependent's data."""
        if not guardian.is_admin and guardian.role != UserRole.USER:
            return False

        if dependent.guardian_id != guardian.user_id:
            return False

        return True

    @staticmethod
    def can_guardian_takeover(
        guardian: User,
        dependent: User,
        context: dict[str, Any] | None = None,
    ) -> PermissionResult:
        """
        Check if guardian can take over dependent's session.

        Args:
            guardian: Guardian user
            dependent: Dependent user
            context: Situational context

        Returns:
            PermissionResult
        """
        result = PermissionResult(permission=Permission.GUARDIAN_TAKEOVER)

        # Must be the assigned guardian
        if dependent.guardian_id != guardian.user_id:
            result.allowed = False
            result.reason = "Not the assigned guardian for this user"
            return result

        # Guardian must have takeover permission
        if Permission.GUARDIAN_TAKEOVER not in ROLE_PERMISSIONS.get(guardian.role, set()):
            result.allowed = False
            result.reason = "Guardian does not have takeover permission"
            return result

        result.allowed = True
        result.reason = "Guardian takeover permitted"
        return result

    @staticmethod
    def get_guardian_visible_restrictions(
        guardian: User,
        dependent: User,
    ) -> list[UserRestriction]:
        """Get restrictions that guardian can see/modify for a dependent."""
        if not GuardianPermissionHelper.can_guardian_access_dependent(guardian, dependent):
            return []

        return list(dependent.restrictions)

    @staticmethod
    def can_modify_restriction(
        guardian: User,
        dependent: User,
        restriction: UserRestriction,
    ) -> bool:
        """Check if guardian can modify a specific restriction."""
        if not GuardianPermissionHelper.can_guardian_access_dependent(guardian, dependent):
            return False

        # Admins can modify all restrictions
        if guardian.is_admin:
            return True

        # Regular guardians can modify most restrictions
        # Some might be locked by admins
        return True


# ============================================================================
# Factory Functions
# ============================================================================


def create_permission_checker() -> PermissionChecker:
    """Create a permission checker instance."""
    return PermissionChecker()

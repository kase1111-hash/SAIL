"""
SAIL User Manager

Central manager for the multi-user family system coordinating:
- User identification and sessions
- Voice identification
- Permission checking
- Guardian alerts
- Data isolation
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np

from src.users.base import (
    AccessLevel,
    IdentificationMethod,
    User,
    UserEvent,
    UserEventType,
    UserManagerConfig,
    UserRestriction,
    UserRole,
    UserSession,
)
from src.users.guardian import (
    GuardianAlert,
    GuardianAlertSystem,
    GuardianAlertTrigger,
)
from src.users.permissions import (
    GuardianPermissionHelper,
    Permission,
    PermissionChecker,
    PermissionResult,
)
from src.users.voice_id import (
    VoiceIdentificationSystem,
)

logger = logging.getLogger(__name__)


# ============================================================================
# User Manager
# ============================================================================


class UserManager:
    """
    Central manager for the multi-user family system.

    Coordinates user identification, sessions, permissions,
    and guardian alerts.
    """

    def __init__(self, config: UserManagerConfig | None = None):
        self.config = config or UserManagerConfig()

        # Users indexed by user_id
        self._users: dict[str, User] = {}

        # User name to ID mapping
        self._name_to_id: dict[str, str] = {}

        # Current session
        self._current_session: UserSession | None = None

        # Session history
        self._session_history: deque[UserSession] = deque(maxlen=50)

        # Subsystems
        self._voice_id: VoiceIdentificationSystem | None = None
        self._permission_checker = PermissionChecker()
        self._alert_system: GuardianAlertSystem | None = None
        self._alert_trigger: GuardianAlertTrigger | None = None

        # State
        self._initialized = False
        self._running = False

        # Event callbacks
        self._event_callbacks: list[Callable[[UserEvent], None]] = []

        # Default user for anonymous/unidentified access
        self._default_user: User | None = None

    def _safe_create_task(self, coro: Any, name: str | None = None) -> asyncio.Task[Any]:
        """Create a task with error handling to prevent silent failures."""
        task = asyncio.create_task(coro, name=name)
        task.add_done_callback(self._on_task_done)
        return task

    def _on_task_done(self, task: asyncio.Task[Any]) -> None:
        """Handle task completion, logging any exceptions."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(f"Background task {task.get_name()} failed: {exc}")

    async def initialize(self) -> bool:
        """Initialize the user manager."""
        logger.info("Initializing user manager...")

        try:
            # Ensure data directory exists
            self.config.data_path.mkdir(parents=True, exist_ok=True)

            # Initialize voice identification
            if self.config.voice_id_enabled:
                self._voice_id = VoiceIdentificationSystem(self.config)
                await self._voice_id.initialize()
                self._voice_id.register_event_callback(self._on_user_event)

            # Initialize guardian alert system
            if self.config.guardian_alerts_enabled:
                self._alert_system = GuardianAlertSystem(self.config)
                self._alert_trigger = GuardianAlertTrigger(self._alert_system)
                self._alert_system.register_event_callback(self._on_user_event)

            # Load users from storage
            await self._load_users()

            # Create default user if allowed
            if self.config.allow_anonymous:
                self._create_default_user()

            self._initialized = True
            logger.info(f"User manager initialized with {len(self._users)} users")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize user manager: {e}")
            return False

    async def start(self) -> None:
        """Start the user manager."""
        if not self._initialized:
            await self.initialize()
        self._running = True
        logger.info("User manager started")

    async def stop(self) -> None:
        """Stop the user manager."""
        self._running = False

        # End current session
        if self._current_session:
            self._end_session(self._current_session)

        # Save users
        await self._save_users()

        logger.info("User manager stopped")

    async def _load_users(self) -> None:
        """Load users from storage."""
        users_file = self.config.data_path / "users.json"
        if not users_file.exists():
            return

        try:
            with open(users_file) as f:
                data = json.load(f)

            for user_data in data.get("users", []):
                user = User.from_dict(user_data)
                self._users[user.user_id] = user
                self._name_to_id[user.name.lower()] = user.user_id

                # Register for voice ID
                if self._voice_id and user.voice_profile:
                    self._voice_id.register_user(user.user_id, user.name)

        except Exception as e:
            logger.error(f"Failed to load users: {e}")

    async def _save_users(self) -> None:
        """Save users to storage."""
        users_file = self.config.data_path / "users.json"

        data = {
            "users": [user.to_dict() for user in self._users.values()],
            "saved_at": datetime.now().isoformat(),
        }

        with open(users_file, "w") as f:
            json.dump(data, f, indent=2)

    def _create_default_user(self) -> None:
        """Create the default user for anonymous access."""
        self._default_user = User(
            user_id="default",
            name=self.config.default_user_name,
            role=UserRole.USER,
            access_level=AccessLevel.LIMITED,
        )
        self._users["default"] = self._default_user
        self._name_to_id[self.config.default_user_name.lower()] = "default"

    # ========================================================================
    # User Management
    # ========================================================================

    def register_user(
        self,
        name: str,
        role: UserRole = UserRole.USER,
        access_level: AccessLevel = AccessLevel.STANDARD,
        restrictions: list[UserRestriction] | None = None,
        guardian_name: str | None = None,
        custom_briefings: list[str] | None = None,
    ) -> User:
        """
        Register a new user in the system.

        Args:
            name: User's display name
            role: User role
            access_level: Access level
            restrictions: List of restrictions
            guardian_name: Name of guardian (for minors)
            custom_briefings: Custom briefing modules

        Returns:
            The created User
        """
        # Check for duplicate name
        if name.lower() in self._name_to_id:
            raise ValueError(f"User '{name}' already exists")

        # Create user
        user = User(
            name=name,
            role=role,
            access_level=access_level,
            restrictions=restrictions or [],
            custom_briefings=custom_briefings or [],
        )

        # Set guardian if specified
        if guardian_name:
            guardian_id = self._name_to_id.get(guardian_name.lower())
            if guardian_id:
                user.guardian_id = guardian_id
                guardian = self._users.get(guardian_id)
                if guardian:
                    guardian.dependents.append(user.user_id)

        # Store user
        self._users[user.user_id] = user
        self._name_to_id[name.lower()] = user.user_id

        # Register for voice ID
        if self._voice_id:
            self._voice_id.register_user(user.user_id, user.name)

        logger.info(f"Registered user: {name} (role={role.value})")

        # Save
        self._safe_create_task(self._save_users(), name="save_users")

        return user

    def get_user(self, user_id: str) -> User | None:
        """Get a user by ID."""
        return self._users.get(user_id)

    def get_user_by_name(self, name: str) -> User | None:
        """Get a user by name."""
        user_id = self._name_to_id.get(name.lower())
        if user_id:
            return self._users.get(user_id)
        return None

    def get_all_users(self) -> list[User]:
        """Get all registered users."""
        return list(self._users.values())

    def get_dependents(self, guardian_id: str) -> list[User]:
        """Get all dependents for a guardian."""
        guardian = self._users.get(guardian_id)
        if not guardian:
            return []

        return [
            self._users[dep_id]
            for dep_id in guardian.dependents
            if dep_id in self._users
        ]

    def get_guardian(self, user_id: str) -> User | None:
        """Get the guardian for a user."""
        user = self._users.get(user_id)
        if user and user.guardian_id:
            return self._users.get(user.guardian_id)
        return None

    # ========================================================================
    # Session Management
    # ========================================================================

    async def identify_user(
        self,
        audio: np.ndarray | None = None,
        sample_rate: int = 16000,
        explicit_name: str | None = None,
    ) -> tuple[User | None, float]:
        """
        Identify a user from voice or explicit name.

        Args:
            audio: Audio sample for voice identification
            sample_rate: Audio sample rate
            explicit_name: Explicitly provided user name

        Returns:
            Tuple of (User or None, confidence score)
        """
        # Explicit identification
        if explicit_name:
            user = self.get_user_by_name(explicit_name)
            if user:
                await self._start_session(user, IdentificationMethod.EXPLICIT, 1.0)
                return user, 1.0

        # Voice identification
        if audio is not None and self._voice_id:
            result = await self._voice_id.identify(audio, sample_rate)
            if result.is_identified and result.user_id:
                user = self._users.get(result.user_id)
                if user:
                    await self._start_session(user, IdentificationMethod.VOICE, result.confidence)
                    return user, result.confidence

        # Fall back to default user
        if self._default_user and self.config.allow_anonymous:
            await self._start_session(self._default_user, IdentificationMethod.DEFAULT, 0.0)
            return self._default_user, 0.0

        return None, 0.0

    async def _start_session(
        self,
        user: User,
        method: IdentificationMethod,
        confidence: float,
    ) -> UserSession:
        """Start a new session for a user."""
        # End current session if exists
        if self._current_session:
            self._end_session(self._current_session)

        # Create new session
        session = UserSession(
            user_id=user.user_id,
            user=user,
            identification_method=method,
            identification_confidence=confidence,
            session_timeout_seconds=self.config.session_timeout_seconds,
        )

        self._current_session = session
        user.last_active = datetime.now()

        # Emit event
        self._emit_event(UserEvent(
            event_type=UserEventType.SESSION_STARTED,
            user_id=user.user_id,
            session_id=session.session_id,
            data={
                "method": method.value,
                "confidence": confidence,
            },
        ))

        logger.info(f"Started session for {user.name} via {method.value}")
        return session

    def _end_session(self, session: UserSession) -> None:
        """End a session."""
        session.expire()
        self._session_history.append(session)

        self._emit_event(UserEvent(
            event_type=UserEventType.SESSION_EXPIRED,
            user_id=session.user_id,
            session_id=session.session_id,
            data={
                "duration_seconds": session.duration_seconds,
                "interaction_count": session.interaction_count,
            },
        ))

    def get_current_session(self) -> UserSession | None:
        """Get the current active session."""
        if self._current_session and not self._current_session.is_expired:
            return self._current_session
        return None

    def get_current_user(self) -> User | None:
        """Get the currently active user."""
        session = self.get_current_session()
        if session:
            return session.user or self._users.get(session.user_id)
        return None

    def touch_session(self) -> None:
        """Update activity on current session."""
        if self._current_session:
            self._current_session.touch()

    # ========================================================================
    # Voice Enrollment
    # ========================================================================

    async def enroll_voice_sample(
        self,
        user_id: str,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> tuple[bool, str]:
        """
        Add a voice enrollment sample for a user.

        Args:
            user_id: User to enroll
            audio: Audio sample
            sample_rate: Sample rate

        Returns:
            Tuple of (success, message)
        """
        if not self._voice_id:
            return False, "Voice identification not enabled"

        user = self._users.get(user_id)
        if not user:
            return False, "User not found"

        return await self._voice_id.enroll_sample(user_id, audio, sample_rate)

    def get_enrollment_status(self, user_id: str) -> dict[str, Any]:
        """Get voice enrollment status for a user."""
        if not self._voice_id:
            return {"enabled": False}

        status = self._voice_id.get_enrollment_status(user_id)
        status["enabled"] = True
        return status

    # ========================================================================
    # Permission Checking
    # ========================================================================

    def check_permission(
        self,
        permission: Permission,
        user: User | None = None,
        context: dict[str, Any] | None = None,
    ) -> PermissionResult:
        """
        Check if a user has a permission.

        Args:
            permission: Permission to check
            user: User to check (uses current user if not specified)
            context: Situational context

        Returns:
            PermissionResult
        """
        if user is None:
            user = self.get_current_user()

        if user is None:
            return PermissionResult(
                allowed=False,
                permission=permission,
                reason="No user identified",
            )

        result = self._permission_checker.check_permission(user, permission, context)

        # Emit event if denied
        if not result.allowed:
            self._emit_event(UserEvent(
                event_type=UserEventType.PERMISSION_DENIED,
                user_id=user.user_id,
                data={
                    "permission": permission.value,
                    "reason": result.reason,
                },
            ))

        return result

    def get_user_permissions(self, user: User | None = None) -> set[Permission]:
        """Get all permissions for a user."""
        if user is None:
            user = self.get_current_user()

        if user is None:
            return set()

        return self._permission_checker.get_user_permissions(user)

    # ========================================================================
    # Guardian Functions
    # ========================================================================

    async def guardian_takeover(
        self,
        guardian: User,
        dependent: User,
        reason: str = "",
    ) -> bool:
        """
        Guardian takes control of dependent's session.

        Args:
            guardian: The guardian
            dependent: The dependent
            reason: Reason for takeover

        Returns:
            True if takeover successful
        """
        # Check permission
        result = GuardianPermissionHelper.can_guardian_takeover(guardian, dependent)
        if not result.allowed:
            logger.warning(f"Guardian takeover denied: {result.reason}")
            return False

        # Take over current session if it belongs to dependent
        if self._current_session and self._current_session.user_id == dependent.user_id:
            self._current_session.guardian_takeover(guardian.user_id, reason)

            self._emit_event(UserEvent(
                event_type=UserEventType.GUARDIAN_TAKEOVER,
                user_id=dependent.user_id,
                session_id=self._current_session.session_id,
                data={
                    "guardian_id": guardian.user_id,
                    "reason": reason,
                },
            ))

            logger.info(f"Guardian {guardian.name} took control of {dependent.name}'s session")
            return True

        return False

    def release_guardian_control(self, guardian: User) -> bool:
        """Release guardian control of current session."""
        if not self._current_session:
            return False

        if self._current_session.guardian_controlled_by != guardian.user_id:
            return False

        self._current_session.release_guardian_control()

        self._emit_event(UserEvent(
            event_type=UserEventType.GUARDIAN_RELEASED,
            user_id=self._current_session.user_id,
            session_id=self._current_session.session_id,
            data={"guardian_id": guardian.user_id},
        ))

        return True

    def check_guardian_triggers(
        self,
        context: dict[str, Any],
    ) -> list[GuardianAlert]:
        """
        Check for guardian alert triggers based on current context.

        Args:
            context: Current situational context

        Returns:
            List of alerts created
        """
        if not self._alert_trigger or not self._alert_system:
            return []

        user = self.get_current_user()
        if not user or not user.has_guardian:
            return []

        guardian = self.get_guardian(user.user_id)
        if not guardian:
            return []

        return self._alert_trigger.check_for_triggers(user, guardian, context)

    def get_pending_guardian_alerts(
        self,
        guardian_id: str | None = None,
    ) -> list[GuardianAlert]:
        """Get pending guardian alerts."""
        if not self._alert_system:
            return []

        if guardian_id is None:
            user = self.get_current_user()
            if user:
                guardian_id = user.user_id

        if guardian_id is None:
            return []

        return self._alert_system.get_pending_alerts(guardian_id)

    def acknowledge_alert(self, alert_id: str, note: str = "") -> bool:
        """Acknowledge a guardian alert."""
        if not self._alert_system:
            return False
        return self._alert_system.acknowledge_alert(alert_id, note)

    # ========================================================================
    # Events
    # ========================================================================

    def register_event_callback(self, callback: Callable[[UserEvent], None]) -> None:
        """Register callback for user events."""
        self._event_callbacks.append(callback)

    def _on_user_event(self, event: UserEvent) -> None:
        """Handle events from subsystems."""
        self._emit_event(event)

    def _emit_event(self, event: UserEvent) -> None:
        """Emit a user event."""
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    # ========================================================================
    # Statistics
    # ========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get user manager statistics."""
        stats = {
            "initialized": self._initialized,
            "running": self._running,
            "total_users": len(self._users),
            "admins": sum(1 for u in self._users.values() if u.is_admin),
            "minors": sum(1 for u in self._users.values() if u.is_minor),
            "current_user": self.get_current_user().name if self.get_current_user() else None,
            "sessions_in_history": len(self._session_history),
        }

        if self._voice_id:
            stats["voice_id"] = self._voice_id.get_stats()

        if self._alert_system:
            stats["guardian_alerts"] = self._alert_system.get_stats()

        return stats

    # ========================================================================
    # Context Manager
    # ========================================================================

    async def __aenter__(self) -> UserManager:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()


# ============================================================================
# Factory Functions
# ============================================================================


def create_user_manager(
    config: UserManagerConfig | None = None,
) -> UserManager:
    """Create a user manager instance."""
    return UserManager(config)


async def create_initialized_user_manager(
    config: UserManagerConfig | None = None,
) -> UserManager:
    """Create and initialize a user manager."""
    manager = UserManager(config)
    await manager.initialize()
    return manager

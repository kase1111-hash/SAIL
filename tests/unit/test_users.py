"""
Unit tests for the Multi-User & Family System (Phase 9).

Tests cover:
- User and session management
- Voice identification
- Permission checking and RBAC
- Guardian alerts
- Data isolation
- Custom briefings
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

import pytest
import numpy as np

from src.users.base import (
    UserRole,
    AccessLevel,
    UserRestriction,
    IdentificationMethod,
    SessionState,
    VoiceProfile,
    User,
    UserSession,
    UserEvent,
    UserEventType,
    UserManagerConfig,
)


# ============================================================================
# User Model Tests
# ============================================================================


class TestUser:
    """Tests for User data model."""

    def test_user_creation(self):
        """Test creating a user."""
        user = User(
            name="John",
            role=UserRole.USER,
            access_level=AccessLevel.STANDARD,
        )

        assert user.name == "John"
        assert user.role == UserRole.USER
        assert user.access_level == AccessLevel.STANDARD
        assert user.user_id is not None

    def test_user_is_minor(self):
        """Test minor role detection."""
        minor = User(name="Child", role=UserRole.MINOR)
        adult = User(name="Adult", role=UserRole.USER)

        assert minor.is_minor
        assert not adult.is_minor

    def test_user_is_admin(self):
        """Test admin role detection."""
        admin = User(name="Admin", role=UserRole.ADMIN)
        user = User(name="User", role=UserRole.USER)

        assert admin.is_admin
        assert not user.is_admin

    def test_user_has_restriction(self):
        """Test restriction checking."""
        user = User(
            name="Test",
            restrictions=[UserRestriction.NO_DISABLE_ALERTS],
        )

        assert user.has_restriction(UserRestriction.NO_DISABLE_ALERTS)
        assert not user.has_restriction(UserRestriction.NO_OVERRIDE_GUARDIAN_WHILE_DRIVING)

    def test_user_guardian_relationship(self):
        """Test guardian relationship."""
        guardian = User(name="Parent", role=UserRole.ADMIN)
        child = User(
            name="Child",
            role=UserRole.MINOR,
            guardian_id=guardian.user_id,
        )
        guardian.dependents.append(child.user_id)

        assert child.has_guardian
        assert child.guardian_id == guardian.user_id
        assert guardian.is_guardian
        assert child.user_id in guardian.dependents

    def test_user_to_dict(self):
        """Test converting user to dictionary."""
        user = User(
            name="Test",
            role=UserRole.USER,
            custom_briefings=["new_driver"],
        )

        data = user.to_dict()

        assert data["name"] == "Test"
        assert data["role"] == "user"
        assert "new_driver" in data["custom_briefings"]

    def test_user_from_dict(self):
        """Test creating user from dictionary."""
        data = {
            "user_id": "test-123",
            "name": "Test User",
            "role": "minor",
            "access_level": "limited",
            "restrictions": ["no_disable_alerts"],
        }

        user = User.from_dict(data)

        assert user.user_id == "test-123"
        assert user.name == "Test User"
        assert user.role == UserRole.MINOR
        assert UserRestriction.NO_DISABLE_ALERTS in user.restrictions


# ============================================================================
# Voice Profile Tests
# ============================================================================


class TestVoiceProfile:
    """Tests for VoiceProfile."""

    def test_profile_creation(self):
        """Test creating a voice profile."""
        profile = VoiceProfile(
            user_id="user-123",
            required_samples=5,
        )

        assert profile.user_id == "user-123"
        assert profile.required_samples == 5
        assert not profile.is_enrolled

    def test_add_embedding(self):
        """Test adding embeddings to profile."""
        profile = VoiceProfile(user_id="user-123", required_samples=3)

        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        profile.add_embedding(embedding, confidence=0.9)

        assert profile.sample_count == 1
        assert len(profile.embeddings) == 1
        assert not profile.enrollment_complete

    def test_enrollment_completion(self):
        """Test enrollment completes after enough samples."""
        profile = VoiceProfile(user_id="user-123", required_samples=3)

        for i in range(3):
            embedding = [float(i)] * 10
            profile.add_embedding(embedding)

        assert profile.sample_count == 3
        assert profile.enrollment_complete
        assert profile.is_enrolled

    def test_get_centroid(self):
        """Test computing centroid of embeddings."""
        profile = VoiceProfile(user_id="user-123")

        profile.add_embedding([1.0, 2.0, 3.0])
        profile.add_embedding([3.0, 2.0, 1.0])

        centroid = profile.get_centroid()

        assert centroid is not None
        assert len(centroid) == 3
        assert centroid[0] == 2.0
        assert centroid[1] == 2.0
        assert centroid[2] == 2.0


# ============================================================================
# User Session Tests
# ============================================================================


class TestUserSession:
    """Tests for UserSession."""

    def test_session_creation(self):
        """Test creating a session."""
        session = UserSession(
            user_id="user-123",
            identification_method=IdentificationMethod.VOICE,
            identification_confidence=0.95,
        )

        assert session.user_id == "user-123"
        assert session.identification_method == IdentificationMethod.VOICE
        assert session.is_active

    def test_session_touch(self):
        """Test updating session activity."""
        session = UserSession(user_id="user-123")
        initial_count = session.interaction_count

        session.touch()

        assert session.interaction_count == initial_count + 1
        assert session.state == SessionState.ACTIVE

    def test_session_expiration(self):
        """Test session expiration."""
        session = UserSession(
            user_id="user-123",
            session_timeout_seconds=1,
        )
        session.expires_at = datetime.now() - timedelta(seconds=10)

        assert session.is_expired

    def test_session_guardian_takeover(self):
        """Test guardian taking over session."""
        session = UserSession(user_id="user-123")

        session.guardian_takeover("guardian-456", "Safety check")

        assert session.is_guardian_controlled
        assert session.guardian_controlled_by == "guardian-456"
        assert session.guardian_control_reason == "Safety check"

    def test_session_release_guardian_control(self):
        """Test releasing guardian control."""
        session = UserSession(user_id="user-123")
        session.guardian_takeover("guardian-456", "Test")

        session.release_guardian_control()

        assert not session.is_guardian_controlled
        assert session.guardian_controlled_by is None


# ============================================================================
# Permission Tests
# ============================================================================


class TestPermissions:
    """Tests for Permission system."""

    def test_admin_has_all_permissions(self):
        """Test that admins have all permissions."""
        from src.users.permissions import Permission, PermissionChecker

        admin = User(name="Admin", role=UserRole.ADMIN)
        checker = PermissionChecker()

        # Admins should have change settings permission
        result = checker.check_permission(admin, Permission.CHANGE_SETTINGS)
        assert result.allowed

        # And manage users
        result = checker.check_permission(admin, Permission.MANAGE_USERS)
        assert result.allowed

    def test_minor_has_limited_permissions(self):
        """Test that minors have limited permissions."""
        from src.users.permissions import Permission, PermissionChecker

        minor = User(name="Child", role=UserRole.MINOR)
        checker = PermissionChecker()

        # Minors can view settings
        result = checker.check_permission(minor, Permission.VIEW_SETTINGS)
        assert result.allowed

        # But cannot change them
        result = checker.check_permission(minor, Permission.CHANGE_SETTINGS)
        assert not result.allowed

    def test_restriction_blocks_permission(self):
        """Test that restrictions block permissions."""
        from src.users.permissions import Permission, PermissionChecker

        user = User(
            name="User",
            role=UserRole.USER,
            restrictions=[UserRestriction.NO_DISABLE_ALERTS],
        )
        checker = PermissionChecker()

        result = checker.check_permission(user, Permission.DISMISS_ALERTS)

        assert not result.allowed
        assert UserRestriction.NO_DISABLE_ALERTS in result.restrictions_triggered

    def test_context_affects_permission(self):
        """Test that context affects permissions."""
        from src.users.permissions import Permission, PermissionChecker, PermissionContext

        user = User(
            name="User",
            role=UserRole.USER,
            restrictions=[UserRestriction.NO_OVERRIDE_GUARDIAN_WHILE_DRIVING],
        )
        checker = PermissionChecker()

        # Without driving context, can override
        result = checker.check_permission(
            user,
            Permission.OVERRIDE_GUARDIAN_MODE,
            context={},
        )
        assert result.allowed

        # While driving, cannot override
        result = checker.check_permission(
            user,
            Permission.OVERRIDE_GUARDIAN_MODE,
            context={"is_driving": True},
        )
        assert not result.allowed
        assert PermissionContext.DRIVING in result.context_factors


# ============================================================================
# Guardian Alert Tests
# ============================================================================


class TestGuardianAlerts:
    """Tests for Guardian Alert system."""

    def test_create_alert(self):
        """Test creating a guardian alert."""
        from src.users.guardian import (
            GuardianAlertSystem,
            GuardianAlertType,
            GuardianAlertLevel,
        )

        system = GuardianAlertSystem()

        guardian = User(name="Parent", role=UserRole.ADMIN)
        child = User(name="Child", role=UserRole.MINOR, guardian_id=guardian.user_id)

        alert = system.create_alert(
            GuardianAlertType.HIGH_RISK_DETECTED,
            child,
            guardian,
            details={"risk_score": 0.8},
        )

        assert alert is not None
        assert alert.alert_type == GuardianAlertType.HIGH_RISK_DETECTED
        assert alert.dependent_id == child.user_id
        assert alert.guardian_id == guardian.user_id

    def test_alert_cooldown(self):
        """Test alert cooldown prevents spam."""
        from src.users.guardian import GuardianAlertSystem, GuardianAlertType

        config = UserManagerConfig(guardian_alert_cooldown_seconds=300)
        system = GuardianAlertSystem(config)

        guardian = User(name="Parent", role=UserRole.ADMIN)
        child = User(name="Child", role=UserRole.MINOR)

        # First alert should succeed
        alert1 = system.create_alert(
            GuardianAlertType.UNFAMILIAR_LOCATION,
            child,
            guardian,
        )
        assert alert1 is not None

        # Second same-type alert should be suppressed
        alert2 = system.create_alert(
            GuardianAlertType.UNFAMILIAR_LOCATION,
            child,
            guardian,
        )
        assert alert2 is None

    def test_alert_acknowledgment(self):
        """Test acknowledging an alert."""
        from src.users.guardian import GuardianAlertSystem, GuardianAlertType

        system = GuardianAlertSystem()

        guardian = User(name="Parent", role=UserRole.ADMIN)
        child = User(name="Child", role=UserRole.MINOR)

        alert = system.create_alert(
            GuardianAlertType.LATE_NIGHT_ACTIVITY,
            child,
            guardian,
        )

        system.acknowledge_alert(alert.alert_id, "Noted")

        updated = system.get_alert(alert.alert_id)
        assert updated.is_acknowledged

    def test_get_pending_alerts(self):
        """Test getting pending alerts for guardian."""
        from src.users.guardian import GuardianAlertSystem, GuardianAlertType

        # Disable cooldown for test
        config = UserManagerConfig(guardian_alert_cooldown_seconds=0)
        system = GuardianAlertSystem(config)

        guardian = User(name="Parent", role=UserRole.ADMIN)
        child = User(name="Child", role=UserRole.MINOR)

        system.create_alert(GuardianAlertType.LATE_NIGHT_ACTIVITY, child, guardian)
        system.create_alert(GuardianAlertType.UNFAMILIAR_LOCATION, child, guardian)

        pending = system.get_pending_alerts(guardian.user_id)
        assert len(pending) == 2


# ============================================================================
# Data Isolation Tests
# ============================================================================


class TestDataIsolation:
    """Tests for data isolation."""

    def test_store_creation(self):
        """Test creating a user data store."""
        from src.users.isolation import UserDataStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = UserDataStore(
                user_id="user-123",
                data_path=Path(tmpdir),
                encryption_enabled=False,
            )

            assert store.user_id == "user-123"

    def test_save_and_load_context(self):
        """Test saving and loading context."""
        from src.users.isolation import UserDataStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = UserDataStore(
                user_id="user-123",
                data_path=Path(tmpdir),
                encryption_enabled=False,
            )

            context_data = {"key": "value", "count": 42}
            store.save_context("test_context", context_data)

            loaded = store.load_context("test_context")

            assert loaded is not None
            assert loaded["key"] == "value"
            assert loaded["count"] == 42

    def test_save_and_load_preferences(self):
        """Test saving and loading preferences."""
        from src.users.isolation import UserDataStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = UserDataStore(
                user_id="user-123",
                data_path=Path(tmpdir),
                encryption_enabled=False,
            )

            prefs = {"theme": "dark", "volume": 80}
            store.save_preferences(prefs)

            loaded = store.load_preferences()

            assert loaded["theme"] == "dark"
            assert loaded["volume"] == 80

    def test_isolation_manager(self):
        """Test isolation manager creates separate stores."""
        from src.users.isolation import DataIsolationManager

        with tempfile.TemporaryDirectory() as tmpdir:
            config = UserManagerConfig(
                data_path=Path(tmpdir),
                encryption_enabled=False,
            )
            manager = DataIsolationManager(config)

            store1 = manager.get_store("user-1")
            store2 = manager.get_store("user-2")

            # Stores should be different
            assert store1.user_id != store2.user_id

            # Data should be isolated
            store1.save_context("test", {"user": 1})
            store2.save_context("test", {"user": 2})

            assert store1.load_context("test")["user"] == 1
            assert store2.load_context("test")["user"] == 2


# ============================================================================
# Custom Briefings Tests
# ============================================================================


class TestCustomBriefings:
    """Tests for custom briefings system."""

    def test_builtin_briefings_exist(self):
        """Test that built-in briefings are loaded."""
        from src.users.briefings import CustomBriefingSystem

        system = CustomBriefingSystem()

        # Check for some built-in briefings
        assert system.get_briefing("new_driver_protocol") is not None
        assert system.get_briefing("first_job_rights") is not None

    def test_briefing_accessibility(self):
        """Test briefing access control."""
        from src.users.briefings import CustomBriefingSystem

        system = CustomBriefingSystem()

        minor = User(name="Child", role=UserRole.MINOR)
        adult = User(name="Adult", role=UserRole.USER)

        briefing = system.get_briefing("new_driver_protocol")

        # Both should be able to access teen-appropriate content
        assert briefing.is_accessible_by(minor)
        assert briefing.is_accessible_by(adult)

    def test_get_user_briefings(self):
        """Test getting briefings assigned to a user."""
        from src.users.briefings import CustomBriefingSystem

        system = CustomBriefingSystem()

        user = User(
            name="Teen",
            role=UserRole.MINOR,
            custom_briefings=["new_driver_protocol", "first_job_rights"],
        )

        briefings = system.get_user_briefings(user)

        assert len(briefings) == 2
        assert any(b.briefing_id == "new_driver_protocol" for b in briefings)

    def test_search_briefings(self):
        """Test searching briefings."""
        from src.users.briefings import CustomBriefingSystem

        system = CustomBriefingSystem()

        results = system.search_briefings("driving")

        assert len(results) > 0
        assert any("driver" in r.title.lower() for r in results)

    def test_get_briefings_by_category(self):
        """Test getting briefings by category."""
        from src.users.briefings import CustomBriefingSystem, BriefingCategory

        system = CustomBriefingSystem()

        driving_briefings = system.get_briefings_by_category(BriefingCategory.DRIVING)

        assert len(driving_briefings) > 0
        assert all(b.category == BriefingCategory.DRIVING for b in driving_briefings)


# ============================================================================
# Voice Identification Tests
# ============================================================================


class TestVoiceIdentification:
    """Tests for voice identification system."""

    @pytest.mark.asyncio
    async def test_system_initialization(self):
        """Test voice ID system initialization."""
        from src.users.voice_id import VoiceIdentificationSystem

        with tempfile.TemporaryDirectory() as tmpdir:
            config = UserManagerConfig(data_path=Path(tmpdir))
            system = VoiceIdentificationSystem(config)

            success = await system.initialize()
            assert success

    @pytest.mark.asyncio
    async def test_user_registration(self):
        """Test registering a user for voice ID."""
        from src.users.voice_id import VoiceIdentificationSystem

        with tempfile.TemporaryDirectory() as tmpdir:
            config = UserManagerConfig(data_path=Path(tmpdir))
            system = VoiceIdentificationSystem(config)
            await system.initialize()

            profile = system.register_user("user-123", "Test User")

            assert profile is not None
            assert profile.user_id == "user-123"
            assert not profile.is_enrolled

    @pytest.mark.asyncio
    async def test_enrollment(self):
        """Test voice enrollment."""
        from src.users.voice_id import VoiceIdentificationSystem

        with tempfile.TemporaryDirectory() as tmpdir:
            config = UserManagerConfig(
                data_path=Path(tmpdir),
                voice_enrollment_samples=2,
            )
            system = VoiceIdentificationSystem(config)
            await system.initialize()

            system.register_user("user-123", "Test User")

            # Enroll with fake audio
            audio = np.random.randn(16000).astype(np.float32)

            success1, msg1 = await system.enroll_sample("user-123", audio)
            assert success1
            assert "more sample" in msg1.lower() or "complete" in msg1.lower()

            success2, msg2 = await system.enroll_sample("user-123", audio)
            assert success2

    @pytest.mark.asyncio
    async def test_identification(self):
        """Test voice identification."""
        from src.users.voice_id import VoiceIdentificationSystem

        with tempfile.TemporaryDirectory() as tmpdir:
            config = UserManagerConfig(
                data_path=Path(tmpdir),
                voice_enrollment_samples=1,
                voice_confidence_threshold=0.0,  # Low threshold for test
            )
            system = VoiceIdentificationSystem(config)
            await system.initialize()

            system.register_user("user-123", "Test User")

            # Enroll
            audio = np.random.randn(16000).astype(np.float32)
            await system.enroll_sample("user-123", audio)

            # Identify with same-ish audio
            result = await system.identify(audio)

            # Should get a result (may or may not match depending on random audio)
            assert result is not None


# ============================================================================
# User Manager Integration Tests
# ============================================================================


class TestUserManager:
    """Integration tests for UserManager."""

    @pytest.mark.asyncio
    async def test_manager_initialization(self):
        """Test user manager initialization."""
        from src.users.manager import UserManager

        with tempfile.TemporaryDirectory() as tmpdir:
            config = UserManagerConfig(
                data_path=Path(tmpdir),
                voice_id_enabled=False,  # Disable for faster test
            )
            manager = UserManager(config)

            success = await manager.initialize()
            assert success
            assert manager._initialized

    @pytest.mark.asyncio
    async def test_register_user(self):
        """Test registering a user."""
        from src.users.manager import UserManager

        with tempfile.TemporaryDirectory() as tmpdir:
            config = UserManagerConfig(
                data_path=Path(tmpdir),
                voice_id_enabled=False,
            )
            manager = UserManager(config)
            await manager.initialize()

            user = manager.register_user(
                name="Test User",
                role=UserRole.USER,
            )

            assert user is not None
            assert user.name == "Test User"
            assert manager.get_user(user.user_id) is not None

    @pytest.mark.asyncio
    async def test_identify_user_explicit(self):
        """Test explicit user identification."""
        from src.users.manager import UserManager

        with tempfile.TemporaryDirectory() as tmpdir:
            config = UserManagerConfig(
                data_path=Path(tmpdir),
                voice_id_enabled=False,
            )
            manager = UserManager(config)
            await manager.initialize()

            manager.register_user("Test User", UserRole.USER)

            user, confidence = await manager.identify_user(explicit_name="Test User")

            assert user is not None
            assert user.name == "Test User"
            assert confidence == 1.0

    @pytest.mark.asyncio
    async def test_session_management(self):
        """Test session creation and management."""
        from src.users.manager import UserManager

        with tempfile.TemporaryDirectory() as tmpdir:
            config = UserManagerConfig(
                data_path=Path(tmpdir),
                voice_id_enabled=False,
            )
            manager = UserManager(config)
            await manager.initialize()

            manager.register_user("Test User", UserRole.USER)
            await manager.identify_user(explicit_name="Test User")

            session = manager.get_current_session()
            user = manager.get_current_user()

            assert session is not None
            assert user is not None
            assert user.name == "Test User"

    @pytest.mark.asyncio
    async def test_permission_checking(self):
        """Test permission checking via manager."""
        from src.users.manager import UserManager
        from src.users.permissions import Permission

        with tempfile.TemporaryDirectory() as tmpdir:
            config = UserManagerConfig(
                data_path=Path(tmpdir),
                voice_id_enabled=False,
            )
            manager = UserManager(config)
            await manager.initialize()

            manager.register_user("Admin", UserRole.ADMIN)
            await manager.identify_user(explicit_name="Admin")

            result = manager.check_permission(Permission.MANAGE_USERS)

            assert result.allowed

    @pytest.mark.asyncio
    async def test_guardian_setup(self):
        """Test setting up guardian relationship."""
        from src.users.manager import UserManager

        with tempfile.TemporaryDirectory() as tmpdir:
            config = UserManagerConfig(
                data_path=Path(tmpdir),
                voice_id_enabled=False,
            )
            manager = UserManager(config)
            await manager.initialize()

            guardian = manager.register_user("Parent", UserRole.ADMIN)
            child = manager.register_user(
                "Child",
                UserRole.MINOR,
                guardian_name="Parent",
            )

            assert child.guardian_id == guardian.user_id
            assert child.user_id in guardian.dependents

            dependents = manager.get_dependents(guardian.user_id)
            assert len(dependents) == 1
            assert dependents[0].name == "Child"

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        from src.users.manager import UserManager

        with tempfile.TemporaryDirectory() as tmpdir:
            config = UserManagerConfig(
                data_path=Path(tmpdir),
                voice_id_enabled=False,
            )

            async with UserManager(config) as manager:
                assert manager._running
                manager.register_user("Test", UserRole.USER)
                assert manager.get_user_by_name("Test") is not None


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    @pytest.mark.asyncio
    async def test_create_user_manager(self):
        """Test create_user_manager factory."""
        from src.users.manager import create_user_manager

        manager = create_user_manager()
        assert manager is not None
        assert not manager._initialized

    @pytest.mark.asyncio
    async def test_create_initialized_user_manager(self):
        """Test create_initialized_user_manager factory."""
        from src.users.manager import create_initialized_user_manager

        with tempfile.TemporaryDirectory() as tmpdir:
            config = UserManagerConfig(
                data_path=Path(tmpdir),
                voice_id_enabled=False,
            )
            manager = await create_initialized_user_manager(config)

            assert manager is not None
            assert manager._initialized

    def test_create_permission_checker(self):
        """Test create_permission_checker factory."""
        from src.users.permissions import create_permission_checker

        checker = create_permission_checker()
        assert checker is not None

    def test_create_guardian_alert_system(self):
        """Test create_guardian_alert_system factory."""
        from src.users.guardian import create_guardian_alert_system

        system = create_guardian_alert_system()
        assert system is not None

    def test_create_briefing_system(self):
        """Test create_briefing_system factory."""
        from src.users.briefings import create_briefing_system

        system = create_briefing_system()
        assert system is not None
        assert system.get_briefing("new_driver_protocol") is not None

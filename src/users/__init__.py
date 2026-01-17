"""
SAIL Multi-User & Family System (Phase 9)

Provides comprehensive multi-user support including:
- User identification and session management
- Voice-based speaker recognition
- Role-based access control
- Guardian alerts and notifications
- Per-user data isolation
- Custom briefings with age-appropriate content
"""

from src.users.base import (
    # Enums
    UserRole,
    AccessLevel,
    UserRestriction,
    IdentificationMethod,
    SessionState,
    UserEventType,
    # Data models
    VoiceProfile,
    User,
    UserSession,
    UserEvent,
    # Configuration
    UserManagerConfig,
)

from src.users.manager import (
    UserManager,
    create_user_manager,
    create_initialized_user_manager,
)

from src.users.voice_id import (
    VoiceIdentificationSystem,
    VoiceIdentificationResult,
    VoiceEmbeddingProvider,
    LocalVoiceEmbeddingProvider,
    create_voice_id_system,
)

from src.users.permissions import (
    Permission,
    PermissionContext,
    PermissionResult,
    PermissionChecker,
    GuardianPermissionHelper,
    require_permission,
    create_permission_checker,
)

from src.users.guardian import (
    GuardianAlertLevel,
    GuardianAlertType,
    NotificationChannel,
    GuardianAlert,
    GuardianAlertSystem,
    GuardianAlertTrigger,
    create_guardian_alert_system,
)

from src.users.isolation import (
    UserDataStore,
    UserDataEncryptor,
    DataIsolationManager,
    create_isolation_manager,
)

from src.users.briefings import (
    BriefingCategory,
    AgeAppropriateness,
    BriefingContent,
    CustomBriefingSystem,
    create_briefing_system,
)

__all__ = [
    # Base types
    "UserRole",
    "AccessLevel",
    "UserRestriction",
    "IdentificationMethod",
    "SessionState",
    "UserEventType",
    # Data models
    "VoiceProfile",
    "User",
    "UserSession",
    "UserEvent",
    # Configuration
    "UserManagerConfig",
    # Manager
    "UserManager",
    "create_user_manager",
    "create_initialized_user_manager",
    # Voice ID
    "VoiceIdentificationSystem",
    "VoiceIdentificationResult",
    "VoiceEmbeddingProvider",
    "LocalVoiceEmbeddingProvider",
    "create_voice_id_system",
    # Permissions
    "Permission",
    "PermissionContext",
    "PermissionResult",
    "PermissionChecker",
    "GuardianPermissionHelper",
    "require_permission",
    "create_permission_checker",
    # Guardian alerts
    "GuardianAlertLevel",
    "GuardianAlertType",
    "NotificationChannel",
    "GuardianAlert",
    "GuardianAlertSystem",
    "GuardianAlertTrigger",
    "create_guardian_alert_system",
    # Data isolation
    "UserDataStore",
    "UserDataEncryptor",
    "DataIsolationManager",
    "create_isolation_manager",
    # Briefings
    "BriefingCategory",
    "AgeAppropriateness",
    "BriefingContent",
    "CustomBriefingSystem",
    "create_briefing_system",
]

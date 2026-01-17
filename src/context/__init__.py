"""
SAIL Context Buffer System

Provides multi-tier rolling memory for conversation history and
situational awareness.

Phase 5 Implementation:
- Multi-tier memory: immediate (5 min), short-term (1 hour), session
- Adaptive depth modes: minimal, driving, conversation, crisis
- SQLite persistence with optional encryption
- Crisis mode context locking
- Session recovery support
"""

from src.context.base import (
    ContextConfig,
    ContextDepthMode,
    ContextEntry,
    ContextSummary,
    ContextTier,
    DepthModeConfig,
    EntryPriority,
    EntryType,
    get_depth_mode_config,
)
from src.context.buffer import (
    MultiTierBuffer,
    TierBuffer,
    TierStats,
)
from src.context.manager import (
    ContextBufferManager,
    ContextManagerEvent,
    SessionInfo,
    create_context_manager,
)
from src.context.persistence import (
    ContextEncryption,
    ContextStore,
    EncryptionError,
)

__all__ = [
    # Base types
    "ContextConfig",
    "ContextDepthMode",
    "ContextEntry",
    "ContextSummary",
    "ContextTier",
    "DepthModeConfig",
    "EntryPriority",
    "EntryType",
    "get_depth_mode_config",
    # Buffer
    "MultiTierBuffer",
    "TierBuffer",
    "TierStats",
    # Persistence
    "ContextEncryption",
    "ContextStore",
    "EncryptionError",
    # Manager
    "ContextBufferManager",
    "ContextManagerEvent",
    "SessionInfo",
    "create_context_manager",
]

"""
SAIL Context Buffer Base Module

Provides base types and configuration for the context buffer system.
The context buffer maintains conversation history and situational awareness
across multiple time scales.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4


class ContextTier(str, Enum):
    """Tiers of context memory with different retention periods."""

    IMMEDIATE = "immediate"     # Last ~5 minutes - active conversation
    SHORT_TERM = "short_term"   # Last ~1 hour - recent interactions
    SESSION = "session"         # Current session - since startup/login
    BACKGROUND = "background"   # Persistent knowledge - user preferences, history


class ContextDepthMode(str, Enum):
    """
    Adaptive depth modes for different situations.

    Controls how much context is maintained based on current activity.
    """

    MINIMAL = "minimal"         # Very low context - background mode
    DRIVING = "driving"         # Reduced context - safety focus
    CONVERSATION = "conversation"  # Normal context - active interaction
    CRISIS = "crisis"           # Maximum context - emergency situation


class EntryType(str, Enum):
    """Types of context entries."""

    USER_INPUT = "user_input"           # User's spoken/typed input
    ASSISTANT_RESPONSE = "assistant_response"  # SAIL's responses
    SYSTEM_EVENT = "system_event"       # System events (mode changes, alerts)
    SENSOR_DATA = "sensor_data"         # Sensor readings (location, motion)
    INTERVENTION = "intervention"       # Intervention events
    SUMMARY = "summary"                 # Compressed summaries of older context


class EntryPriority(int, Enum):
    """Priority levels for context entries."""

    CRITICAL = 100      # Must not be evicted (crisis info, emergency contacts)
    HIGH = 75           # Important (recent user requests, active tasks)
    NORMAL = 50         # Standard context
    LOW = 25            # Background info (can be compressed/evicted)


@dataclass
class ContextEntry:
    """A single entry in the context buffer."""

    entry_id: str = field(default_factory=lambda: str(uuid4()))
    entry_type: EntryType = EntryType.USER_INPUT
    content: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    tier: ContextTier = ContextTier.IMMEDIATE
    priority: EntryPriority = EntryPriority.NORMAL

    # Optional metadata
    user_id: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Linking for conversation threads
    parent_id: str | None = None
    thread_id: str | None = None

    # Flags
    is_pinned: bool = False         # Prevent automatic eviction
    is_compressed: bool = False     # Has been summarized
    is_encrypted: bool = False      # Stored with encryption

    def to_dict(self) -> dict[str, Any]:
        """Convert entry to dictionary for storage."""
        return {
            "entry_id": self.entry_id,
            "entry_type": self.entry_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "tier": self.tier.value,
            "priority": self.priority.value,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
            "thread_id": self.thread_id,
            "is_pinned": self.is_pinned,
            "is_compressed": self.is_compressed,
            "is_encrypted": self.is_encrypted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextEntry:
        """Create entry from dictionary."""
        return cls(
            entry_id=data.get("entry_id", str(uuid4())),
            entry_type=EntryType(data.get("entry_type", "user_input")),
            content=data.get("content", ""),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if "timestamp" in data else datetime.now(),
            tier=ContextTier(data.get("tier", "immediate")),
            priority=EntryPriority(data.get("priority", 50)),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            metadata=data.get("metadata", {}),
            parent_id=data.get("parent_id"),
            thread_id=data.get("thread_id"),
            is_pinned=data.get("is_pinned", False),
            is_compressed=data.get("is_compressed", False),
            is_encrypted=data.get("is_encrypted", False),
        )

    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return (datetime.now() - self.timestamp).total_seconds()


@dataclass
class ContextSummary:
    """A compressed summary of multiple context entries."""

    summary_id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    source_entry_ids: list[str] = field(default_factory=list)
    time_range_start: datetime | None = None
    time_range_end: datetime | None = None
    entry_count: int = 0
    topics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary for storage."""
        return {
            "summary_id": self.summary_id,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "source_entry_ids": self.source_entry_ids,
            "time_range_start": self.time_range_start.isoformat()
            if self.time_range_start else None,
            "time_range_end": self.time_range_end.isoformat()
            if self.time_range_end else None,
            "entry_count": self.entry_count,
            "topics": self.topics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextSummary:
        """Create summary from dictionary."""
        return cls(
            summary_id=data.get("summary_id", str(uuid4())),
            content=data.get("content", ""),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if "timestamp" in data else datetime.now(),
            source_entry_ids=data.get("source_entry_ids", []),
            time_range_start=datetime.fromisoformat(data["time_range_start"])
            if data.get("time_range_start") else None,
            time_range_end=datetime.fromisoformat(data["time_range_end"])
            if data.get("time_range_end") else None,
            entry_count=data.get("entry_count", 0),
            topics=data.get("topics", []),
        )


@dataclass
class ContextConfig:
    """Configuration for the context buffer system."""

    # Tier durations
    immediate_duration_seconds: int = 300       # 5 minutes
    short_term_duration_seconds: int = 3600     # 1 hour
    session_timeout_seconds: int = 86400        # 24 hours

    # Capacity limits
    immediate_max_entries: int = 100
    short_term_max_entries: int = 500
    session_max_entries: int = 2000

    # Persistence
    persist_sessions: bool = True
    storage_path: Path = Path("data/context")
    encryption_enabled: bool = True
    encryption_key: bytes | None = None

    # Depth mode settings
    default_depth_mode: ContextDepthMode = ContextDepthMode.CONVERSATION

    # Compression settings
    compress_after_entries: int = 50    # Compress when tier exceeds this
    summary_target_length: int = 200    # Target summary length in words

    # Crisis mode settings
    crisis_lock_enabled: bool = True    # Lock context during crisis
    crisis_retention_hours: int = 24    # Keep crisis context for 24 hours


@dataclass
class DepthModeConfig:
    """Configuration for a specific depth mode."""

    mode: ContextDepthMode
    immediate_retention_factor: float = 1.0     # Multiplier for immediate retention
    short_term_retention_factor: float = 1.0    # Multiplier for short-term retention
    max_context_tokens: int = 4000              # Max tokens for LLM context
    include_summaries: bool = True              # Include compressed summaries
    include_sensor_data: bool = True            # Include sensor readings
    priority_threshold: EntryPriority = EntryPriority.LOW  # Min priority to include


# Default depth mode configurations
DEPTH_MODE_CONFIGS = {
    ContextDepthMode.MINIMAL: DepthModeConfig(
        mode=ContextDepthMode.MINIMAL,
        immediate_retention_factor=0.5,
        short_term_retention_factor=0.25,
        max_context_tokens=1000,
        include_summaries=False,
        include_sensor_data=False,
        priority_threshold=EntryPriority.HIGH,
    ),
    ContextDepthMode.DRIVING: DepthModeConfig(
        mode=ContextDepthMode.DRIVING,
        immediate_retention_factor=0.75,
        short_term_retention_factor=0.5,
        max_context_tokens=2000,
        include_summaries=True,
        include_sensor_data=True,  # Important for driving
        priority_threshold=EntryPriority.NORMAL,
    ),
    ContextDepthMode.CONVERSATION: DepthModeConfig(
        mode=ContextDepthMode.CONVERSATION,
        immediate_retention_factor=1.0,
        short_term_retention_factor=1.0,
        max_context_tokens=4000,
        include_summaries=True,
        include_sensor_data=True,
        priority_threshold=EntryPriority.LOW,
    ),
    ContextDepthMode.CRISIS: DepthModeConfig(
        mode=ContextDepthMode.CRISIS,
        immediate_retention_factor=1.5,  # Keep more immediate context
        short_term_retention_factor=1.0,
        max_context_tokens=6000,         # More context for crisis handling
        include_summaries=True,
        include_sensor_data=True,
        priority_threshold=EntryPriority.LOW,
    ),
}


def get_depth_mode_config(mode: ContextDepthMode) -> DepthModeConfig:
    """Get configuration for a depth mode."""
    return DEPTH_MODE_CONFIGS.get(mode, DEPTH_MODE_CONFIGS[ContextDepthMode.CONVERSATION])

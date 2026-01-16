"""
SAIL Multi-Tier Context Buffer

Implements the multi-tier rolling memory system with automatic promotion,
eviction, and compression of context entries.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Iterator

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

logger = logging.getLogger(__name__)


@dataclass
class TierStats:
    """Statistics for a context tier."""

    tier: ContextTier
    entry_count: int = 0
    oldest_entry_age: float = 0.0
    newest_entry_age: float = 0.0
    pinned_count: int = 0
    compressed_count: int = 0


class TierBuffer:
    """
    A single tier of the context buffer.

    Manages entries within a specific time window with automatic eviction
    based on age and capacity limits.
    """

    def __init__(
        self,
        tier: ContextTier,
        max_age_seconds: float,
        max_entries: int,
    ) -> None:
        """
        Initialize tier buffer.

        Args:
            tier: The context tier this buffer represents
            max_age_seconds: Maximum age for entries in this tier
            max_entries: Maximum number of entries to hold
        """
        self.tier = tier
        self.max_age_seconds = max_age_seconds
        self.max_entries = max_entries

        self._entries: deque[ContextEntry] = deque(maxlen=max_entries)
        self._entry_index: dict[str, ContextEntry] = {}
        self._lock = threading.RLock()

    def add(self, entry: ContextEntry) -> None:
        """
        Add an entry to the buffer.

        Args:
            entry: Entry to add
        """
        with self._lock:
            entry.tier = self.tier

            # Check if entry already exists
            if entry.entry_id in self._entry_index:
                self._update_entry(entry)
                return

            # Add to buffer
            self._entries.append(entry)
            self._entry_index[entry.entry_id] = entry

            # Evict old entries if needed
            self._evict_old_entries()

    def _update_entry(self, entry: ContextEntry) -> None:
        """Update an existing entry."""
        old_entry = self._entry_index.get(entry.entry_id)
        if old_entry:
            # Remove old entry from deque
            try:
                self._entries.remove(old_entry)
            except ValueError:
                pass
            # Add updated entry
            self._entries.append(entry)
            self._entry_index[entry.entry_id] = entry

    def get(self, entry_id: str) -> ContextEntry | None:
        """
        Get an entry by ID.

        Args:
            entry_id: Entry ID to find

        Returns:
            Entry or None if not found
        """
        with self._lock:
            return self._entry_index.get(entry_id)

    def remove(self, entry_id: str) -> ContextEntry | None:
        """
        Remove an entry by ID.

        Args:
            entry_id: Entry ID to remove

        Returns:
            Removed entry or None
        """
        with self._lock:
            entry = self._entry_index.pop(entry_id, None)
            if entry:
                try:
                    self._entries.remove(entry)
                except ValueError:
                    pass
            return entry

    def get_entries(
        self,
        limit: int | None = None,
        min_priority: EntryPriority = EntryPriority.LOW,
        entry_types: list[EntryType] | None = None,
        since: datetime | None = None,
    ) -> list[ContextEntry]:
        """
        Get entries matching criteria.

        Args:
            limit: Maximum entries to return
            min_priority: Minimum priority to include
            entry_types: Filter by entry types
            since: Only entries after this time

        Returns:
            List of matching entries (newest first)
        """
        with self._lock:
            entries = []

            for entry in reversed(self._entries):
                # Check priority
                if entry.priority.value < min_priority.value:
                    continue

                # Check entry type
                if entry_types and entry.entry_type not in entry_types:
                    continue

                # Check timestamp
                if since and entry.timestamp < since:
                    continue

                entries.append(entry)

                if limit and len(entries) >= limit:
                    break

            return entries

    def get_expired_entries(self) -> list[ContextEntry]:
        """
        Get entries that have exceeded max age and are not pinned.

        Returns:
            List of expired entries
        """
        with self._lock:
            expired = []
            for entry in self._entries:
                if entry.age_seconds > self.max_age_seconds and not entry.is_pinned:
                    expired.append(entry)
            return expired

    def _evict_old_entries(self) -> list[ContextEntry]:
        """
        Evict entries that exceed max age or capacity.

        Returns:
            List of evicted entries
        """
        evicted = []

        # Evict by age
        while self._entries:
            oldest = self._entries[0]
            if oldest.age_seconds > self.max_age_seconds and not oldest.is_pinned:
                entry = self._entries.popleft()
                self._entry_index.pop(entry.entry_id, None)
                evicted.append(entry)
            else:
                break

        # Evict by capacity (remove lowest priority first)
        if len(self._entries) >= self.max_entries:
            # Sort by priority (ascending) and age (descending)
            sorted_entries = sorted(
                [e for e in self._entries if not e.is_pinned],
                key=lambda e: (e.priority.value, -e.age_seconds),
            )

            # Remove lowest priority entries
            to_remove = len(self._entries) - self.max_entries + 10  # Buffer
            for entry in sorted_entries[:to_remove]:
                self.remove(entry.entry_id)
                evicted.append(entry)

        return evicted

    def pin_entry(self, entry_id: str) -> bool:
        """
        Pin an entry to prevent eviction.

        Args:
            entry_id: Entry to pin

        Returns:
            True if entry was pinned
        """
        with self._lock:
            entry = self._entry_index.get(entry_id)
            if entry:
                entry.is_pinned = True
                return True
            return False

    def unpin_entry(self, entry_id: str) -> bool:
        """
        Unpin an entry.

        Args:
            entry_id: Entry to unpin

        Returns:
            True if entry was unpinned
        """
        with self._lock:
            entry = self._entry_index.get(entry_id)
            if entry:
                entry.is_pinned = False
                return True
            return False

    def clear(self) -> list[ContextEntry]:
        """
        Clear all entries.

        Returns:
            List of cleared entries
        """
        with self._lock:
            entries = list(self._entries)
            self._entries.clear()
            self._entry_index.clear()
            return entries

    def get_stats(self) -> TierStats:
        """Get statistics for this tier."""
        with self._lock:
            if not self._entries:
                return TierStats(tier=self.tier)

            return TierStats(
                tier=self.tier,
                entry_count=len(self._entries),
                oldest_entry_age=self._entries[0].age_seconds if self._entries else 0,
                newest_entry_age=self._entries[-1].age_seconds if self._entries else 0,
                pinned_count=sum(1 for e in self._entries if e.is_pinned),
                compressed_count=sum(1 for e in self._entries if e.is_compressed),
            )

    def __len__(self) -> int:
        """Get number of entries."""
        return len(self._entries)

    def __iter__(self) -> Iterator[ContextEntry]:
        """Iterate over entries (oldest first)."""
        with self._lock:
            return iter(list(self._entries))


class MultiTierBuffer:
    """
    Multi-tier context buffer with automatic promotion and eviction.

    Manages immediate, short-term, and session tiers with entries
    automatically moving between tiers based on age.
    """

    def __init__(self, config: ContextConfig | None = None) -> None:
        """
        Initialize multi-tier buffer.

        Args:
            config: Context configuration
        """
        self.config = config or ContextConfig()

        # Create tier buffers
        self._tiers: dict[ContextTier, TierBuffer] = {
            ContextTier.IMMEDIATE: TierBuffer(
                tier=ContextTier.IMMEDIATE,
                max_age_seconds=self.config.immediate_duration_seconds,
                max_entries=self.config.immediate_max_entries,
            ),
            ContextTier.SHORT_TERM: TierBuffer(
                tier=ContextTier.SHORT_TERM,
                max_age_seconds=self.config.short_term_duration_seconds,
                max_entries=self.config.short_term_max_entries,
            ),
            ContextTier.SESSION: TierBuffer(
                tier=ContextTier.SESSION,
                max_age_seconds=self.config.session_timeout_seconds,
                max_entries=self.config.session_max_entries,
            ),
        }

        # Background tier (no automatic eviction)
        self._background: dict[str, ContextEntry] = {}
        self._summaries: list[ContextSummary] = []

        # Current depth mode
        self._depth_mode = self.config.default_depth_mode
        self._depth_config = get_depth_mode_config(self._depth_mode)

        # Crisis mode lock
        self._crisis_locked = False
        self._crisis_start_time: datetime | None = None

        # Callbacks
        self._on_evict_callbacks: list[Callable[[ContextEntry], None]] = []
        self._on_promote_callbacks: list[Callable[[ContextEntry, ContextTier, ContextTier], None]] = []

        self._lock = threading.RLock()

    def add_entry(
        self,
        content: str,
        entry_type: EntryType = EntryType.USER_INPUT,
        priority: EntryPriority = EntryPriority.NORMAL,
        metadata: dict | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        parent_id: str | None = None,
        thread_id: str | None = None,
    ) -> ContextEntry:
        """
        Add a new entry to the buffer.

        Args:
            content: Entry content
            entry_type: Type of entry
            priority: Entry priority
            metadata: Optional metadata
            user_id: Optional user ID
            session_id: Optional session ID
            parent_id: Parent entry ID for threading
            thread_id: Thread ID for grouping

        Returns:
            Created entry
        """
        entry = ContextEntry(
            entry_type=entry_type,
            content=content,
            priority=priority,
            metadata=metadata or {},
            user_id=user_id,
            session_id=session_id,
            parent_id=parent_id,
            thread_id=thread_id,
        )

        with self._lock:
            # In crisis mode, elevate priority
            if self._crisis_locked:
                entry.priority = max(entry.priority, EntryPriority.HIGH)
                entry.is_pinned = True

            # Add to immediate tier
            self._tiers[ContextTier.IMMEDIATE].add(entry)

        logger.debug(f"Added context entry: {entry.entry_id} ({entry_type.value})")
        return entry

    def add_user_input(self, content: str, **kwargs) -> ContextEntry:
        """Add user input entry."""
        return self.add_entry(content, EntryType.USER_INPUT, **kwargs)

    def add_assistant_response(self, content: str, **kwargs) -> ContextEntry:
        """Add assistant response entry."""
        return self.add_entry(content, EntryType.ASSISTANT_RESPONSE, **kwargs)

    def add_system_event(
        self,
        content: str,
        priority: EntryPriority = EntryPriority.NORMAL,
        **kwargs,
    ) -> ContextEntry:
        """Add system event entry."""
        return self.add_entry(content, EntryType.SYSTEM_EVENT, priority=priority, **kwargs)

    def add_sensor_data(self, content: str, metadata: dict | None = None, **kwargs) -> ContextEntry:
        """Add sensor data entry."""
        return self.add_entry(
            content,
            EntryType.SENSOR_DATA,
            priority=EntryPriority.LOW,
            metadata=metadata,
            **kwargs,
        )

    def get_entry(self, entry_id: str) -> ContextEntry | None:
        """
        Get an entry by ID from any tier.

        Args:
            entry_id: Entry ID to find

        Returns:
            Entry or None
        """
        with self._lock:
            # Check all tiers
            for tier in self._tiers.values():
                entry = tier.get(entry_id)
                if entry:
                    return entry

            # Check background
            return self._background.get(entry_id)

    def get_context(
        self,
        max_entries: int | None = None,
        max_tokens: int | None = None,
        include_tiers: list[ContextTier] | None = None,
        min_priority: EntryPriority | None = None,
        entry_types: list[EntryType] | None = None,
        since: datetime | None = None,
    ) -> list[ContextEntry]:
        """
        Get context entries for LLM context window.

        Args:
            max_entries: Maximum entries to return
            max_tokens: Maximum estimated tokens (rough estimate)
            include_tiers: Which tiers to include
            min_priority: Minimum priority to include
            entry_types: Filter by entry types
            since: Only entries after this time

        Returns:
            List of context entries (newest first)
        """
        with self._lock:
            # Apply depth mode settings
            depth_config = self._depth_config

            max_tokens = max_tokens or depth_config.max_context_tokens
            min_priority = min_priority or depth_config.priority_threshold
            include_tiers = include_tiers or [
                ContextTier.IMMEDIATE,
                ContextTier.SHORT_TERM,
                ContextTier.SESSION,
            ]

            entries = []
            estimated_tokens = 0

            # Collect from each tier
            for tier_type in include_tiers:
                tier = self._tiers.get(tier_type)
                if tier is None:
                    continue

                tier_entries = tier.get_entries(
                    min_priority=min_priority,
                    entry_types=entry_types,
                    since=since,
                )

                for entry in tier_entries:
                    # Estimate tokens (rough: ~4 chars per token)
                    entry_tokens = len(entry.content) // 4

                    if max_tokens and estimated_tokens + entry_tokens > max_tokens:
                        break

                    entries.append(entry)
                    estimated_tokens += entry_tokens

                    if max_entries and len(entries) >= max_entries:
                        break

            # Sort by timestamp (newest first)
            entries.sort(key=lambda e: e.timestamp, reverse=True)

            return entries

    def get_conversation_context(
        self,
        max_turns: int = 10,
        include_system: bool = False,
    ) -> list[ContextEntry]:
        """
        Get recent conversation turns.

        Args:
            max_turns: Maximum conversation turns
            include_system: Include system events

        Returns:
            List of conversation entries (oldest first for chat format)
        """
        entry_types = [EntryType.USER_INPUT, EntryType.ASSISTANT_RESPONSE]
        if include_system:
            entry_types.append(EntryType.SYSTEM_EVENT)

        entries = self.get_context(
            max_entries=max_turns * 2,  # Account for user + assistant
            entry_types=entry_types,
        )

        # Reverse to get oldest first (for chat format)
        return list(reversed(entries))

    def promote_entries(self) -> int:
        """
        Promote expired entries from immediate to short-term,
        and from short-term to session.

        Returns:
            Number of entries promoted
        """
        with self._lock:
            promoted = 0

            # Immediate -> Short-term
            expired_immediate = self._tiers[ContextTier.IMMEDIATE].get_expired_entries()
            for entry in expired_immediate:
                self._tiers[ContextTier.IMMEDIATE].remove(entry.entry_id)
                self._tiers[ContextTier.SHORT_TERM].add(entry)
                promoted += 1
                self._trigger_promote_callback(
                    entry, ContextTier.IMMEDIATE, ContextTier.SHORT_TERM
                )

            # Short-term -> Session
            expired_short = self._tiers[ContextTier.SHORT_TERM].get_expired_entries()
            for entry in expired_short:
                self._tiers[ContextTier.SHORT_TERM].remove(entry.entry_id)
                self._tiers[ContextTier.SESSION].add(entry)
                promoted += 1
                self._trigger_promote_callback(
                    entry, ContextTier.SHORT_TERM, ContextTier.SESSION
                )

            return promoted

    def set_depth_mode(self, mode: ContextDepthMode) -> None:
        """
        Set the current depth mode.

        Args:
            mode: Depth mode to use
        """
        with self._lock:
            self._depth_mode = mode
            self._depth_config = get_depth_mode_config(mode)
            logger.info(f"Context depth mode set to: {mode.value}")

    def enter_crisis_mode(self) -> None:
        """
        Enter crisis mode - lock context and prevent eviction.

        All entries become pinned and high priority.
        """
        with self._lock:
            if self._crisis_locked:
                return

            self._crisis_locked = True
            self._crisis_start_time = datetime.now()
            self.set_depth_mode(ContextDepthMode.CRISIS)

            # Pin all current entries
            for tier in self._tiers.values():
                for entry in tier:
                    entry.is_pinned = True

            logger.warning("Context entered CRISIS mode - entries locked")

    def exit_crisis_mode(self) -> None:
        """Exit crisis mode and restore normal operation."""
        with self._lock:
            if not self._crisis_locked:
                return

            self._crisis_locked = False
            self._crisis_start_time = None
            self.set_depth_mode(self.config.default_depth_mode)

            # Unpin entries (except those explicitly pinned)
            # We leave crisis entries pinned for retention period

            logger.info("Context exited CRISIS mode")

    def add_summary(self, summary: ContextSummary) -> None:
        """Add a context summary."""
        with self._lock:
            self._summaries.append(summary)

    def get_summaries(self, limit: int | None = None) -> list[ContextSummary]:
        """Get stored summaries."""
        with self._lock:
            if limit:
                return self._summaries[-limit:]
            return list(self._summaries)

    def add_to_background(self, entry: ContextEntry) -> None:
        """Add entry to background (persistent) tier."""
        with self._lock:
            entry.tier = ContextTier.BACKGROUND
            self._background[entry.entry_id] = entry

    def get_background_entries(self) -> list[ContextEntry]:
        """Get all background entries."""
        with self._lock:
            return list(self._background.values())

    def clear_tier(self, tier: ContextTier) -> list[ContextEntry]:
        """Clear all entries from a tier."""
        with self._lock:
            if tier == ContextTier.BACKGROUND:
                entries = list(self._background.values())
                self._background.clear()
                return entries
            return self._tiers[tier].clear()

    def clear_all(self) -> None:
        """Clear all entries from all tiers."""
        with self._lock:
            for tier in self._tiers.values():
                tier.clear()
            self._background.clear()
            self._summaries.clear()

    def get_stats(self) -> dict[str, TierStats]:
        """Get statistics for all tiers."""
        with self._lock:
            stats = {}
            for tier_type, tier in self._tiers.items():
                stats[tier_type.value] = tier.get_stats()

            # Background stats
            stats["background"] = TierStats(
                tier=ContextTier.BACKGROUND,
                entry_count=len(self._background),
            )

            return stats

    def on_evict(self, callback: Callable[[ContextEntry], None]) -> None:
        """Register callback for entry eviction."""
        self._on_evict_callbacks.append(callback)

    def on_promote(
        self,
        callback: Callable[[ContextEntry, ContextTier, ContextTier], None],
    ) -> None:
        """Register callback for entry promotion."""
        self._on_promote_callbacks.append(callback)

    def _trigger_promote_callback(
        self,
        entry: ContextEntry,
        from_tier: ContextTier,
        to_tier: ContextTier,
    ) -> None:
        """Trigger promotion callbacks."""
        for callback in self._on_promote_callbacks:
            try:
                callback(entry, from_tier, to_tier)
            except Exception as e:
                logger.error(f"Promote callback error: {e}")

    @property
    def depth_mode(self) -> ContextDepthMode:
        """Get current depth mode."""
        return self._depth_mode

    @property
    def is_crisis_mode(self) -> bool:
        """Check if in crisis mode."""
        return self._crisis_locked

    @property
    def total_entries(self) -> int:
        """Get total number of entries across all tiers."""
        with self._lock:
            return (
                sum(len(tier) for tier in self._tiers.values())
                + len(self._background)
            )

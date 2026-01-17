"""
SAIL Context Buffer Manager

Orchestrates the complete context buffer system including multi-tier
memory, persistence, and session management.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Callable

from src.context.base import (
    ContextConfig,
    ContextDepthMode,
    ContextEntry,
    ContextSummary,
    ContextTier,
    EntryPriority,
    EntryType,
)
from src.context.buffer import MultiTierBuffer, TierStats
from src.context.persistence import ContextEncryption, ContextStore

if TYPE_CHECKING:
    from src.config.schema import ContextConfig as SchemaContextConfig

logger = logging.getLogger(__name__)


@dataclass
class ContextManagerEvent:
    """Event emitted by the context manager."""

    event_type: str
    data: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SessionInfo:
    """Information about the current session."""

    session_id: str
    user_id: str | None = None
    started_at: datetime = field(default_factory=datetime.now)
    entry_count: int = 0
    depth_mode: ContextDepthMode = ContextDepthMode.CONVERSATION
    is_crisis: bool = False


class ContextBufferManager:
    """
    Manages the complete context buffer system.

    Coordinates multi-tier memory, persistence, compression, and
    session management to provide a unified context interface.
    """

    def __init__(
        self,
        context_config: SchemaContextConfig | None = None,
        encryption_key: bytes | None = None,
        encryption_password: str | None = None,
    ) -> None:
        """
        Initialize context buffer manager.

        Args:
            context_config: Context configuration from SAIL config
            encryption_key: Direct encryption key
            encryption_password: Password to derive encryption key
        """
        self._schema_config = context_config

        # Build internal config
        self._config = self._build_config()

        # Initialize encryption
        self._encryption: ContextEncryption | None = None
        if self._config.encryption_enabled:
            self._encryption = ContextEncryption(
                key=encryption_key,
                password=encryption_password,
            )

        # Initialize components
        self._buffer = MultiTierBuffer(self._config)
        self._store: ContextStore | None = None

        # Session management
        self._session: SessionInfo | None = None
        self._is_running = False

        # Background tasks
        self._maintenance_task: asyncio.Task | None = None
        self._maintenance_interval = 60  # seconds

        # Event callbacks
        self._event_callbacks: list[Callable[[ContextManagerEvent], None]] = []

        self._lock = threading.RLock()

    def _build_config(self) -> ContextConfig:
        """Build internal config from schema config."""
        config = ContextConfig()

        if self._schema_config:
            config.immediate_duration_seconds = self._schema_config.immediate_duration_seconds
            config.short_term_duration_seconds = self._schema_config.short_term_duration_seconds
            config.persist_sessions = self._schema_config.persist_sessions
            config.storage_path = self._schema_config.storage_path
            config.encryption_enabled = self._schema_config.encryption_enabled

        return config

    async def start(self, user_id: str | None = None) -> str:
        """
        Start the context manager and begin a new session.

        Args:
            user_id: Optional user identifier

        Returns:
            Session ID
        """
        if self._is_running:
            return self._session.session_id if self._session else ""

        logger.info("Starting context buffer manager")
        self._is_running = True

        # Initialize persistence if enabled
        if self._config.persist_sessions:
            self._store = ContextStore(
                storage_path=self._config.storage_path,
                encryption=self._encryption,
            )

        # Create new session
        session_id = str(uuid.uuid4())
        self._session = SessionInfo(
            session_id=session_id,
            user_id=user_id,
            depth_mode=self._config.default_depth_mode,
        )

        # Record session start
        if self._store:
            self._store.start_session(session_id, user_id)

        # Start maintenance task
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())

        self._emit_event("started", {"session_id": session_id})
        logger.info(f"Context session started: {session_id}")

        return session_id

    async def stop(self) -> None:
        """Stop the context manager and end the session."""
        if not self._is_running:
            return

        logger.info("Stopping context buffer manager")
        self._is_running = False

        # Cancel maintenance task
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass

        # Save current state
        if self._store and self._session:
            await self._save_session()
            self._store.end_session(self._session.session_id)

        self._emit_event("stopped", {
            "session_id": self._session.session_id if self._session else None,
        })

        self._session = None

    async def _maintenance_loop(self) -> None:
        """Background maintenance loop for promotion and cleanup."""
        while self._is_running:
            try:
                await asyncio.sleep(self._maintenance_interval)

                # Promote entries between tiers
                promoted = self._buffer.promote_entries()
                if promoted > 0:
                    logger.debug(f"Promoted {promoted} entries between tiers")

                # Save to persistence
                if self._store and self._session:
                    await self._save_recent_entries()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Maintenance error: {e}")

    async def _save_session(self) -> None:
        """Save current session state to persistence."""
        if not self._store:
            return

        # Get all entries from buffer
        for tier_type in [ContextTier.IMMEDIATE, ContextTier.SHORT_TERM, ContextTier.SESSION]:
            entries = self._buffer.get_context(
                include_tiers=[tier_type],
                max_entries=None,
            )
            for entry in entries:
                self._store.save_entry(entry)

        # Save summaries
        for summary in self._buffer.get_summaries():
            self._store.save_summary(summary)

    async def _save_recent_entries(self) -> None:
        """Save recent entries to persistence."""
        if not self._store:
            return

        # Save immediate tier entries
        entries = self._buffer.get_context(
            include_tiers=[ContextTier.IMMEDIATE],
            max_entries=50,
        )
        for entry in entries:
            self._store.save_entry(entry)

    def add_entry(
        self,
        content: str,
        entry_type: EntryType = EntryType.USER_INPUT,
        priority: EntryPriority = EntryPriority.NORMAL,
        metadata: dict | None = None,
        parent_id: str | None = None,
        thread_id: str | None = None,
    ) -> ContextEntry:
        """
        Add a new entry to the context buffer.

        Args:
            content: Entry content
            entry_type: Type of entry
            priority: Entry priority
            metadata: Optional metadata
            parent_id: Parent entry for threading
            thread_id: Thread identifier

        Returns:
            Created entry
        """
        with self._lock:
            entry = self._buffer.add_entry(
                content=content,
                entry_type=entry_type,
                priority=priority,
                metadata=metadata,
                user_id=self._session.user_id if self._session else None,
                session_id=self._session.session_id if self._session else None,
                parent_id=parent_id,
                thread_id=thread_id,
            )

            if self._session:
                self._session.entry_count += 1

            self._emit_event("entry_added", {
                "entry_id": entry.entry_id,
                "entry_type": entry_type.value,
            })

            return entry

    def add_user_input(self, content: str, **kwargs) -> ContextEntry:
        """Add user input to context."""
        return self.add_entry(content, EntryType.USER_INPUT, **kwargs)

    def add_assistant_response(self, content: str, **kwargs) -> ContextEntry:
        """Add assistant response to context."""
        return self.add_entry(content, EntryType.ASSISTANT_RESPONSE, **kwargs)

    def add_system_event(
        self,
        content: str,
        priority: EntryPriority = EntryPriority.NORMAL,
        **kwargs,
    ) -> ContextEntry:
        """Add system event to context."""
        return self.add_entry(content, EntryType.SYSTEM_EVENT, priority=priority, **kwargs)

    def add_sensor_data(
        self,
        content: str,
        metadata: dict | None = None,
        **kwargs,
    ) -> ContextEntry:
        """Add sensor data to context."""
        return self.add_entry(
            content,
            EntryType.SENSOR_DATA,
            priority=EntryPriority.LOW,
            metadata=metadata,
            **kwargs,
        )

    def add_intervention(
        self,
        content: str,
        priority: EntryPriority = EntryPriority.HIGH,
        **kwargs,
    ) -> ContextEntry:
        """Add intervention event to context."""
        return self.add_entry(
            content,
            EntryType.INTERVENTION,
            priority=priority,
            **kwargs,
        )

    def get_context(
        self,
        max_entries: int | None = None,
        max_tokens: int | None = None,
        include_system: bool = True,
        include_sensor: bool = True,
    ) -> list[ContextEntry]:
        """
        Get context entries for LLM context window.

        Args:
            max_entries: Maximum entries
            max_tokens: Maximum estimated tokens
            include_system: Include system events
            include_sensor: Include sensor data

        Returns:
            List of context entries
        """
        entry_types = [EntryType.USER_INPUT, EntryType.ASSISTANT_RESPONSE]
        if include_system:
            entry_types.extend([EntryType.SYSTEM_EVENT, EntryType.INTERVENTION])
        if include_sensor:
            entry_types.append(EntryType.SENSOR_DATA)

        return self._buffer.get_context(
            max_entries=max_entries,
            max_tokens=max_tokens,
            entry_types=entry_types,
        )

    def get_conversation(self, max_turns: int = 10) -> list[ContextEntry]:
        """
        Get recent conversation turns.

        Args:
            max_turns: Maximum conversation turns

        Returns:
            List of entries (oldest first for chat format)
        """
        return self._buffer.get_conversation_context(max_turns=max_turns)

    def format_for_llm(self, max_tokens: int | None = None) -> str:
        """
        Format context as a string for LLM input.

        Args:
            max_tokens: Maximum tokens

        Returns:
            Formatted context string
        """
        entries = self.get_conversation()
        lines = []

        for entry in entries:
            if entry.entry_type == EntryType.USER_INPUT:
                lines.append(f"User: {entry.content}")
            elif entry.entry_type == EntryType.ASSISTANT_RESPONSE:
                lines.append(f"Assistant: {entry.content}")
            elif entry.entry_type == EntryType.SYSTEM_EVENT:
                lines.append(f"[System: {entry.content}]")

        return "\n\n".join(lines)

    def set_depth_mode(self, mode: ContextDepthMode) -> None:
        """
        Set the context depth mode.

        Args:
            mode: Depth mode to use
        """
        with self._lock:
            self._buffer.set_depth_mode(mode)
            if self._session:
                self._session.depth_mode = mode

            self._emit_event("depth_mode_changed", {"mode": mode.value})

    def enter_crisis_mode(self) -> None:
        """
        Enter crisis mode.

        Locks context to prevent loss and increases retention.
        """
        with self._lock:
            self._buffer.enter_crisis_mode()
            if self._session:
                self._session.is_crisis = True

            # Add system event
            self.add_system_event(
                "CRISIS MODE ACTIVATED - Context locked",
                priority=EntryPriority.CRITICAL,
            )

            self._emit_event("crisis_mode_entered")
            logger.warning("Context entered CRISIS mode")

    def exit_crisis_mode(self) -> None:
        """Exit crisis mode and restore normal operation."""
        with self._lock:
            self._buffer.exit_crisis_mode()
            if self._session:
                self._session.is_crisis = False

            self.add_system_event(
                "Crisis mode ended - Normal operation resumed",
                priority=EntryPriority.HIGH,
            )

            self._emit_event("crisis_mode_exited")
            logger.info("Context exited CRISIS mode")

    def pin_entry(self, entry_id: str) -> bool:
        """
        Pin an entry to prevent eviction.

        Args:
            entry_id: Entry to pin

        Returns:
            True if pinned
        """
        entry = self._buffer.get_entry(entry_id)
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
            True if unpinned
        """
        entry = self._buffer.get_entry(entry_id)
        if entry:
            entry.is_pinned = False
            return True
        return False

    async def recover_session(self, session_id: str) -> bool:
        """
        Recover a previous session from persistence.

        Args:
            session_id: Session to recover

        Returns:
            True if recovery successful
        """
        if not self._store:
            logger.warning("Cannot recover session - persistence not enabled")
            return False

        try:
            entries = self._store.load_entries(session_id=session_id)
            if not entries:
                return False

            # Load entries into buffer
            for entry in entries:
                if entry.tier == ContextTier.BACKGROUND:
                    self._buffer.add_to_background(entry)
                else:
                    # Add to appropriate tier based on age
                    if entry.age_seconds < self._config.immediate_duration_seconds:
                        self._buffer._tiers[ContextTier.IMMEDIATE].add(entry)
                    elif entry.age_seconds < self._config.short_term_duration_seconds:
                        self._buffer._tiers[ContextTier.SHORT_TERM].add(entry)
                    else:
                        self._buffer._tiers[ContextTier.SESSION].add(entry)

            # Load summaries
            summaries = self._store.load_summaries()
            for summary in summaries:
                self._buffer.add_summary(summary)

            logger.info(f"Recovered session {session_id} with {len(entries)} entries")
            self._emit_event("session_recovered", {
                "session_id": session_id,
                "entry_count": len(entries),
            })

            return True

        except Exception as e:
            logger.error(f"Session recovery failed: {e}")
            return False

    async def cleanup_old_data(self, days: int = 30) -> int:
        """
        Clean up old persisted data.

        Args:
            days: Delete data older than this many days

        Returns:
            Number of entries deleted
        """
        if not self._store:
            return 0

        before = datetime.now() - timedelta(days=days)
        deleted = self._store.delete_old_entries(before)

        if deleted > 0:
            self._store.vacuum()
            logger.info(f"Cleaned up {deleted} old context entries")

        return deleted

    def get_stats(self) -> dict:
        """Get context buffer statistics."""
        stats = {
            "buffer": self._buffer.get_stats(),
            "session": None,
            "persistence": None,
        }

        if self._session:
            stats["session"] = {
                "session_id": self._session.session_id,
                "user_id": self._session.user_id,
                "started_at": self._session.started_at.isoformat(),
                "entry_count": self._session.entry_count,
                "depth_mode": self._session.depth_mode.value,
                "is_crisis": self._session.is_crisis,
            }

        if self._store:
            stats["persistence"] = self._store.get_stats()

        return stats

    def on_event(self, callback: Callable[[ContextManagerEvent], None]) -> None:
        """
        Register callback for context events.

        Args:
            callback: Function to call on events
        """
        self._event_callbacks.append(callback)

    def _emit_event(self, event_type: str, data: dict | None = None) -> None:
        """Emit an event to all registered callbacks."""
        event = ContextManagerEvent(
            event_type=event_type,
            data=data or {},
        )

        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    @property
    def session_id(self) -> str | None:
        """Get current session ID."""
        return self._session.session_id if self._session else None

    @property
    def depth_mode(self) -> ContextDepthMode:
        """Get current depth mode."""
        return self._buffer.depth_mode

    @property
    def is_crisis_mode(self) -> bool:
        """Check if in crisis mode."""
        return self._buffer.is_crisis_mode

    @property
    def is_running(self) -> bool:
        """Check if manager is running."""
        return self._is_running

    @property
    def total_entries(self) -> int:
        """Get total entries in buffer."""
        return self._buffer.total_entries

    async def __aenter__(self) -> ContextBufferManager:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Async context manager exit."""
        await self.stop()


async def create_context_manager(
    context_config: SchemaContextConfig | None = None,
    encryption_password: str | None = None,
) -> ContextBufferManager:
    """
    Factory function to create a context buffer manager.

    Args:
        context_config: Context configuration
        encryption_password: Optional encryption password

    Returns:
        Configured ContextBufferManager
    """
    return ContextBufferManager(
        context_config=context_config,
        encryption_password=encryption_password,
    )

"""
SAIL Context Synchronization

Handles synchronization of context between the central hub and satellite nodes,
including conflict resolution, offline sync recovery, and context handoff.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.context.manager import ContextBufferManager

logger = logging.getLogger(__name__)


class SyncStrategy(str, Enum):
    """Strategy for resolving sync conflicts."""

    LAST_WRITE_WINS = "last_write_wins"
    HUB_PRIORITY = "hub_priority"
    NODE_PRIORITY = "node_priority"
    MERGE_ALL = "merge_all"


class SyncState(str, Enum):
    """State of a sync operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CONFLICT = "conflict"


@dataclass
class SyncConflict:
    """Represents a synchronization conflict."""

    conflict_id: str
    entry_id: str
    hub_version: dict[str, Any]
    node_version: dict[str, Any]
    node_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolution: str | None = None
    resolved_at: datetime | None = None


@dataclass
class SyncEntry:
    """A single entry to be synchronized."""

    entry_id: str
    content: str
    entry_type: str
    timestamp: datetime
    version: int
    checksum: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SyncEntry:
        """Create from dictionary."""
        return cls(
            entry_id=data["entry_id"],
            content=data["content"],
            entry_type=data["entry_type"],
            timestamp=datetime.fromisoformat(data["timestamp"])
            if isinstance(data["timestamp"], str)
            else data["timestamp"],
            version=data.get("version", 1),
            checksum=data.get("checksum", cls._compute_checksum(data["content"])),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_id": self.entry_id,
            "content": self.content,
            "entry_type": self.entry_type,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "checksum": self.checksum,
            "metadata": self.metadata,
        }

    @staticmethod
    def _compute_checksum(content: str) -> str:
        """Compute checksum for content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class SyncBatch:
    """A batch of entries to synchronize."""

    batch_id: str
    node_id: str
    entries: list[SyncEntry]
    created_at: datetime = field(default_factory=datetime.now)
    state: SyncState = SyncState.PENDING
    conflicts: list[SyncConflict] = field(default_factory=list)


@dataclass
class NodeSyncState:
    """Synchronization state for a specific node."""

    node_id: str
    last_sync_time: datetime | None = None
    last_sync_version: int = 0
    pending_entries: list[SyncEntry] = field(default_factory=list)
    offline_buffer: list[SyncEntry] = field(default_factory=list)
    sync_in_progress: bool = False
    consecutive_failures: int = 0


class ContextSync:
    """
    Manages context synchronization between hub and nodes.

    Handles real-time sync, conflict resolution, offline buffering,
    and context handoff between nodes.
    """

    def __init__(
        self,
        context_manager: ContextBufferManager | None = None,
        sync_interval: int = 5,
        batch_size: int = 100,
        strategy: SyncStrategy = SyncStrategy.LAST_WRITE_WINS,
    ) -> None:
        """
        Initialize context sync.

        Args:
            context_manager: Context buffer manager
            sync_interval: Seconds between sync attempts
            batch_size: Maximum entries per sync batch
            strategy: Conflict resolution strategy
        """
        self._context_manager = context_manager
        self._sync_interval = sync_interval
        self._batch_size = batch_size
        self._strategy = strategy

        # Node sync states
        self._node_states: dict[str, NodeSyncState] = {}

        # Version tracking
        self._hub_version = 0
        self._entry_versions: dict[str, int] = {}

        # Conflict tracking
        self._unresolved_conflicts: list[SyncConflict] = []

        # Offline buffer management
        self._max_offline_entries = 1000

        # Statistics
        self._stats = {
            "total_syncs": 0,
            "successful_syncs": 0,
            "failed_syncs": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
            "entries_synced": 0,
        }

    def register_node(self, node_id: str) -> None:
        """Register a node for synchronization."""
        if node_id not in self._node_states:
            self._node_states[node_id] = NodeSyncState(node_id=node_id)
            logger.info(f"Registered node for sync: {node_id}")

    def unregister_node(self, node_id: str) -> None:
        """Unregister a node from synchronization."""
        if node_id in self._node_states:
            # Save any pending data to offline buffer
            state = self._node_states[node_id]
            if state.pending_entries:
                state.offline_buffer.extend(state.pending_entries)
                state.pending_entries.clear()
            logger.info(f"Unregistered node from sync: {node_id}")

    async def sync_from_node(
        self,
        node_id: str,
        context_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Receive context data from a node and sync.

        Args:
            node_id: Source node identifier
            context_data: Context data from node

        Returns:
            Sync result with any hub updates
        """
        self._stats["total_syncs"] += 1

        # Ensure node is registered
        if node_id not in self._node_states:
            self.register_node(node_id)

        state = self._node_states[node_id]
        state.sync_in_progress = True

        try:
            # Parse incoming entries
            incoming_entries = [
                SyncEntry.from_dict(e) for e in context_data.get("entries", [])
            ]
            node_version = context_data.get("version", 0)

            # Process entries and detect conflicts
            conflicts = []
            merged_entries = []

            for entry in incoming_entries:
                existing_version = self._entry_versions.get(entry.entry_id, 0)

                if existing_version > 0 and entry.version <= existing_version:
                    # Potential conflict
                    conflict = await self._handle_conflict(node_id, entry)
                    if conflict:
                        conflicts.append(conflict)
                else:
                    # Accept entry
                    merged_entries.append(entry)
                    self._entry_versions[entry.entry_id] = entry.version

            # Merge accepted entries to context manager
            if self._context_manager and merged_entries:
                await self.merge_entries(node_id, [e.to_dict() for e in merged_entries])

            # Update node state
            state.last_sync_time = datetime.now()
            state.last_sync_version = node_version
            state.consecutive_failures = 0

            # Prepare hub updates for node
            hub_updates = await self._get_hub_updates(node_id, node_version)

            self._stats["successful_syncs"] += 1
            self._stats["entries_synced"] += len(merged_entries)

            logger.debug(f"Sync from node {node_id}: {len(merged_entries)} entries merged")

            return {
                "success": True,
                "merged_count": len(merged_entries),
                "conflicts": [
                    {
                        "entry_id": c.entry_id,
                        "resolution": c.resolution,
                    }
                    for c in conflicts
                ],
                "hub_updates": hub_updates,
                "hub_version": self._hub_version,
            }

        except Exception as e:
            state.consecutive_failures += 1
            self._stats["failed_syncs"] += 1
            logger.error(f"Sync from node {node_id} failed: {e}")
            raise

        finally:
            state.sync_in_progress = False

    async def push_to_node(self, node_id: str) -> dict[str, Any]:
        """
        Push pending context updates to a node.

        Args:
            node_id: Target node identifier

        Returns:
            Push result
        """
        if node_id not in self._node_states:
            raise ValueError(f"Unknown node: {node_id}")

        state = self._node_states[node_id]

        # Get entries to push (pending + any offline buffer)
        entries_to_push = state.pending_entries.copy()

        if not entries_to_push:
            return {"pushed_count": 0}

        # Create batch
        batch = SyncBatch(
            batch_id=f"push_{node_id}_{datetime.now().timestamp()}",
            node_id=node_id,
            entries=entries_to_push[:self._batch_size],
        )

        # Clear pushed entries from pending
        state.pending_entries = entries_to_push[self._batch_size:]

        return {
            "pushed_count": len(batch.entries),
            "entries": [e.to_dict() for e in batch.entries],
            "hub_version": self._hub_version,
        }

    async def merge_entries(
        self,
        source_node_id: str,
        entries: list[dict[str, Any]],
    ) -> int:
        """
        Merge entries into the hub context.

        Args:
            source_node_id: Source node identifier
            entries: Entries to merge

        Returns:
            Number of entries merged
        """
        if not self._context_manager:
            return 0

        merged = 0
        for entry_data in entries:
            try:
                content = entry_data.get("content", "")
                entry_data.get("entry_type", "user_input")
                metadata = entry_data.get("metadata", {})
                metadata["source_node"] = source_node_id

                self._context_manager.add_entry(
                    content=content,
                    metadata=metadata,
                )
                merged += 1
                self._hub_version += 1

            except Exception as e:
                logger.error(f"Failed to merge entry: {e}")

        return merged

    async def _handle_conflict(
        self,
        node_id: str,
        entry: SyncEntry,
    ) -> SyncConflict | None:
        """Handle a sync conflict."""
        self._stats["conflicts_detected"] += 1

        conflict = SyncConflict(
            conflict_id=f"conflict_{entry.entry_id}_{datetime.now().timestamp()}",
            entry_id=entry.entry_id,
            hub_version={"version": self._entry_versions.get(entry.entry_id, 0)},
            node_version=entry.to_dict(),
            node_id=node_id,
        )

        # Apply resolution strategy
        if self._strategy == SyncStrategy.LAST_WRITE_WINS:
            # Compare timestamps
            hub_time = datetime.now()  # Would need actual hub entry timestamp
            if entry.timestamp > hub_time:
                conflict.resolution = "node_accepted"
                self._entry_versions[entry.entry_id] = entry.version
            else:
                conflict.resolution = "hub_kept"

        elif self._strategy == SyncStrategy.HUB_PRIORITY:
            conflict.resolution = "hub_kept"

        elif self._strategy == SyncStrategy.NODE_PRIORITY:
            conflict.resolution = "node_accepted"
            self._entry_versions[entry.entry_id] = entry.version

        elif self._strategy == SyncStrategy.MERGE_ALL:
            # Accept both versions
            conflict.resolution = "merged"
            self._entry_versions[entry.entry_id] = max(
                self._entry_versions.get(entry.entry_id, 0),
                entry.version,
            )

        conflict.resolved_at = datetime.now()
        self._stats["conflicts_resolved"] += 1

        return conflict

    async def _get_hub_updates(
        self,
        node_id: str,
        node_version: int,
    ) -> list[dict[str, Any]]:
        """Get hub updates newer than node's version."""
        state = self._node_states.get(node_id)
        if not state:
            return []

        # Return pending entries for this node
        updates = [e.to_dict() for e in state.pending_entries]
        return updates[:self._batch_size]

    async def initiate_handoff(
        self,
        source_node_id: str,
        target_node_id: str,
    ) -> dict[str, Any]:
        """
        Initiate context handoff between nodes.

        Used when user moves from one node to another (e.g., home → car).

        Args:
            source_node_id: Node user is leaving
            target_node_id: Node user is moving to

        Returns:
            Handoff result
        """
        logger.info(f"Initiating handoff: {source_node_id} → {target_node_id}")

        # Ensure both nodes are registered
        if source_node_id not in self._node_states:
            raise ValueError(f"Unknown source node: {source_node_id}")

        if target_node_id not in self._node_states:
            self.register_node(target_node_id)

        source_state = self._node_states[source_node_id]
        target_state = self._node_states[target_node_id]

        # Sync from source first
        if source_state.pending_entries:
            await self.sync_from_node(source_node_id, {
                "entries": [e.to_dict() for e in source_state.pending_entries],
                "version": source_state.last_sync_version,
            })

        # Get recent context for target
        handoff_context = []
        if self._context_manager:
            recent = self._context_manager.get_conversation(max_turns=20)
            handoff_context = [
                {
                    "entry_id": e.entry_id,
                    "content": e.content,
                    "entry_type": e.entry_type.value,
                    "timestamp": e.timestamp.isoformat(),
                    "version": 1,
                }
                for e in recent
            ]

        # Queue for target node
        for entry_data in handoff_context:
            entry = SyncEntry.from_dict(entry_data)
            target_state.pending_entries.append(entry)

        return {
            "success": True,
            "source_node": source_node_id,
            "target_node": target_node_id,
            "entries_transferred": len(handoff_context),
            "hub_version": self._hub_version,
        }

    async def recover_offline_data(self, node_id: str) -> int:
        """
        Recover data buffered while node was offline.

        Args:
            node_id: Node identifier

        Returns:
            Number of entries recovered
        """
        if node_id not in self._node_states:
            return 0

        state = self._node_states[node_id]
        if not state.offline_buffer:
            return 0

        # Move offline buffer to pending
        recovered = len(state.offline_buffer)
        state.pending_entries.extend(state.offline_buffer)
        state.offline_buffer.clear()

        logger.info(f"Recovered {recovered} offline entries for node {node_id}")
        return recovered

    def buffer_for_offline_node(
        self,
        node_id: str,
        entry: SyncEntry,
    ) -> bool:
        """
        Buffer an entry for an offline node.

        Args:
            node_id: Target node identifier
            entry: Entry to buffer

        Returns:
            True if buffered successfully
        """
        if node_id not in self._node_states:
            self.register_node(node_id)

        state = self._node_states[node_id]

        # Check buffer limit
        if len(state.offline_buffer) >= self._max_offline_entries:
            # Remove oldest entry
            state.offline_buffer.pop(0)

        state.offline_buffer.append(entry)
        return True

    def get_stats(self) -> dict[str, Any]:
        """Get synchronization statistics."""
        return {
            **self._stats,
            "hub_version": self._hub_version,
            "registered_nodes": len(self._node_states),
            "unresolved_conflicts": len(self._unresolved_conflicts),
            "node_states": {
                node_id: {
                    "last_sync": state.last_sync_time.isoformat()
                    if state.last_sync_time
                    else None,
                    "pending_entries": len(state.pending_entries),
                    "offline_buffer": len(state.offline_buffer),
                    "consecutive_failures": state.consecutive_failures,
                }
                for node_id, state in self._node_states.items()
            },
        }

    def get_node_sync_state(self, node_id: str) -> NodeSyncState | None:
        """Get sync state for a specific node."""
        return self._node_states.get(node_id)

    def set_strategy(self, strategy: SyncStrategy) -> None:
        """Set the conflict resolution strategy."""
        self._strategy = strategy

    @property
    def hub_version(self) -> int:
        """Get current hub version."""
        return self._hub_version

"""
SAIL OTA Update Manager

Over-the-air update system for satellite nodes.
Manages firmware distribution, update scheduling, and rollback.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from src.hub.base import NodeType

logger = logging.getLogger(__name__)


class UpdateChannel(str, Enum):
    """Update release channel."""

    STABLE = "stable"
    BETA = "beta"
    NIGHTLY = "nightly"


class UpdateState(str, Enum):
    """State of an update operation."""

    PENDING = "pending"
    DOWNLOADING = "downloading"
    VERIFYING = "verifying"
    STAGING = "staging"
    INSTALLING = "installing"
    REBOOTING = "rebooting"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class UpdatePriority(str, Enum):
    """Priority of an update."""

    CRITICAL = "critical"  # Security fixes
    HIGH = "high"  # Important bug fixes
    NORMAL = "normal"  # Feature updates
    LOW = "low"  # Optional improvements


@dataclass
class FirmwareVersion:
    """Firmware version information."""

    major: int
    minor: int
    patch: int
    build: str | None = None

    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.build:
            version += f"+{self.build}"
        return version

    @classmethod
    def from_string(cls, version_str: str) -> FirmwareVersion:
        """Parse version string."""
        parts = version_str.split("+")
        build = parts[1] if len(parts) > 1 else None
        version_parts = parts[0].split(".")
        return cls(
            major=int(version_parts[0]),
            minor=int(version_parts[1]) if len(version_parts) > 1 else 0,
            patch=int(version_parts[2]) if len(version_parts) > 2 else 0,
            build=build,
        )

    def __gt__(self, other: FirmwareVersion) -> bool:
        return (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch)


@dataclass
class FirmwareUpdate:
    """Firmware update package."""

    update_id: str
    node_type: NodeType
    version: FirmwareVersion
    channel: UpdateChannel
    priority: UpdatePriority
    release_notes: str
    file_path: Path | None = None
    file_size: int = 0
    checksum_sha256: str = ""
    min_version: FirmwareVersion | None = None
    released_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "update_id": self.update_id,
            "node_type": self.node_type.value,
            "version": str(self.version),
            "channel": self.channel.value,
            "priority": self.priority.value,
            "release_notes": self.release_notes,
            "file_size": self.file_size,
            "checksum_sha256": self.checksum_sha256,
            "min_version": str(self.min_version) if self.min_version else None,
            "released_at": self.released_at.isoformat(),
        }


@dataclass
class UpdateOperation:
    """An in-progress update operation."""

    operation_id: str
    node_id: str
    update: FirmwareUpdate
    state: UpdateState = UpdateState.PENDING
    progress: float = 0.0
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    error: str | None = None
    previous_version: FirmwareVersion | None = None


class OTAConfig(BaseModel):
    """OTA update configuration."""

    # Storage
    firmware_path: Path = Field(default=Path("firmware"), description="Firmware storage path")
    max_stored_versions: int = Field(default=3, description="Max firmware versions to keep")

    # Download
    download_timeout_seconds: int = Field(default=600, description="Download timeout")
    chunk_size: int = Field(default=65536, description="Download chunk size")

    # Update behavior
    auto_update_enabled: bool = Field(default=False, description="Enable automatic updates")
    update_check_interval_hours: int = Field(default=6, description="Update check interval")
    preferred_channel: UpdateChannel = Field(
        default=UpdateChannel.STABLE, description="Preferred update channel"
    )

    # Safety
    require_idle: bool = Field(default=True, description="Require node idle before update")
    backup_before_update: bool = Field(default=True, description="Backup before update")
    auto_rollback_on_failure: bool = Field(default=True, description="Auto rollback on failure")
    rollback_timeout_seconds: int = Field(default=300, description="Rollback verification timeout")

    # Scheduling
    maintenance_window_start: int = Field(default=2, description="Maintenance window start (hour)")
    maintenance_window_end: int = Field(default=5, description="Maintenance window end (hour)")


class OTAManager:
    """
    Over-the-air update manager.

    Manages firmware updates for satellite nodes including
    distribution, scheduling, verification, and rollback.
    """

    def __init__(self, config: OTAConfig | None = None) -> None:
        """Initialize OTA manager."""
        self._config = config or OTAConfig()

        # Available updates by node type
        self._available_updates: dict[NodeType, list[FirmwareUpdate]] = {
            node_type: [] for node_type in NodeType
        }

        # Active operations
        self._operations: dict[str, UpdateOperation] = {}

        # Callbacks
        self._update_callbacks: list[Callable[[UpdateOperation], None]] = []

        # Statistics
        self._stats = {
            "total_updates": 0,
            "successful_updates": 0,
            "failed_updates": 0,
            "rollbacks": 0,
        }

        # Background task
        self._check_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the OTA manager."""
        # Ensure firmware directory exists
        self._config.firmware_path.mkdir(parents=True, exist_ok=True)

        # Start update check task
        if self._config.auto_update_enabled:
            self._check_task = asyncio.create_task(self._update_check_loop())

        logger.info("OTA manager started")

    async def stop(self) -> None:
        """Stop the OTA manager."""
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass

        logger.info("OTA manager stopped")

    async def check_for_updates(
        self,
        node_type: NodeType,
        current_version: FirmwareVersion,
        channel: UpdateChannel | None = None,
    ) -> list[FirmwareUpdate]:
        """
        Check for available updates.

        Args:
            node_type: Type of node
            current_version: Current firmware version
            channel: Update channel (uses config default if not specified)

        Returns:
            List of available updates
        """
        channel = channel or self._config.preferred_channel

        available = []
        for update in self._available_updates.get(node_type, []):
            if update.channel != channel:
                continue
            if update.version > current_version:
                if update.min_version and current_version < update.min_version:
                    continue
                available.append(update)

        return sorted(available, key=lambda u: u.version, reverse=True)

    async def download_update(self, update: FirmwareUpdate) -> Path:
        """
        Download firmware update.

        Args:
            update: Update to download

        Returns:
            Path to downloaded firmware
        """
        logger.info(f"Downloading update {update.update_id}: v{update.version}")

        firmware_dir = self._config.firmware_path / update.node_type.value
        firmware_dir.mkdir(parents=True, exist_ok=True)

        firmware_path = firmware_dir / f"firmware_{update.version}.bin"

        # In production, would download from update server
        # For now, simulate download
        await asyncio.sleep(2)

        # Verify checksum
        if update.checksum_sha256:
            # Would verify downloaded file
            pass

        update.file_path = firmware_path
        logger.info(f"Update downloaded to {firmware_path}")

        return firmware_path

    async def install_update(
        self,
        node_id: str,
        update: FirmwareUpdate,
        current_version: FirmwareVersion,
    ) -> UpdateOperation:
        """
        Install update on a node.

        Args:
            node_id: Target node identifier
            update: Update to install
            current_version: Current node version

        Returns:
            Update operation
        """
        import uuid

        operation = UpdateOperation(
            operation_id=str(uuid.uuid4()),
            node_id=node_id,
            update=update,
            previous_version=current_version,
        )

        self._operations[operation.operation_id] = operation
        self._stats["total_updates"] += 1

        try:
            # Download if not already downloaded
            if not update.file_path or not update.file_path.exists():
                operation.state = UpdateState.DOWNLOADING
                self._notify_update(operation)
                await self.download_update(update)

            # Verify firmware
            operation.state = UpdateState.VERIFYING
            operation.progress = 0.2
            self._notify_update(operation)
            await self._verify_firmware(update)

            # Create backup if enabled
            if self._config.backup_before_update:
                logger.info(f"Creating backup for node {node_id}")
                await self._create_backup(node_id)

            # Stage update
            operation.state = UpdateState.STAGING
            operation.progress = 0.4
            self._notify_update(operation)
            await self._stage_update(node_id, update)

            # Install
            operation.state = UpdateState.INSTALLING
            operation.progress = 0.6
            self._notify_update(operation)
            await self._install_firmware(node_id, update)

            # Reboot node
            operation.state = UpdateState.REBOOTING
            operation.progress = 0.8
            self._notify_update(operation)
            await self._reboot_node(node_id)

            # Verify installation
            verified = await self._verify_installation(node_id, update.version)
            if not verified:
                raise RuntimeError("Installation verification failed")

            # Success
            operation.state = UpdateState.COMPLETED
            operation.progress = 1.0
            operation.completed_at = datetime.now()
            self._stats["successful_updates"] += 1

            logger.info(f"Update completed for node {node_id}: v{update.version}")

        except Exception as e:
            operation.state = UpdateState.FAILED
            operation.error = str(e)
            operation.completed_at = datetime.now()
            self._stats["failed_updates"] += 1

            logger.error(f"Update failed for node {node_id}: {e}")

            # Attempt rollback
            if self._config.auto_rollback_on_failure and current_version:
                await self._rollback(node_id, current_version)
                operation.state = UpdateState.ROLLED_BACK

        finally:
            self._notify_update(operation)

        return operation

    async def _verify_firmware(self, update: FirmwareUpdate) -> None:
        """Verify firmware integrity."""
        if not update.file_path or not update.file_path.exists():
            raise FileNotFoundError("Firmware file not found")

        if update.checksum_sha256:
            with open(update.file_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            if file_hash != update.checksum_sha256:
                raise ValueError("Firmware checksum mismatch")

    async def _create_backup(self, node_id: str) -> None:
        """Create backup of current node state."""
        # Would backup current firmware and config
        await asyncio.sleep(1)

    async def _stage_update(self, node_id: str, update: FirmwareUpdate) -> None:
        """Stage update on node."""
        # Would transfer firmware to node staging area
        await asyncio.sleep(2)

    async def _install_firmware(self, node_id: str, update: FirmwareUpdate) -> None:
        """Install firmware on node."""
        # Would trigger firmware installation on node
        await asyncio.sleep(3)

    async def _reboot_node(self, node_id: str) -> None:
        """Reboot node after update."""
        # Would send reboot command to node
        await asyncio.sleep(2)

    async def _verify_installation(
        self,
        node_id: str,
        expected_version: FirmwareVersion,
    ) -> bool:
        """Verify installation succeeded."""
        # Would check node reports correct version
        await asyncio.sleep(1)
        return True

    async def _rollback(
        self,
        node_id: str,
        target_version: FirmwareVersion,
    ) -> bool:
        """Rollback to previous version."""
        logger.warning(f"Rolling back node {node_id} to v{target_version}")
        self._stats["rollbacks"] += 1

        try:
            # Would restore from backup or install previous version
            await asyncio.sleep(3)
            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    async def _update_check_loop(self) -> None:
        """Background loop for checking updates."""
        interval = self._config.update_check_interval_hours * 3600

        while True:
            try:
                await asyncio.sleep(interval)

                # Check for updates for each node type
                # In production, would fetch from update server

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Update check error: {e}")

    def register_update(self, update: FirmwareUpdate) -> None:
        """Register an available update."""
        updates = self._available_updates.setdefault(update.node_type, [])
        updates.append(update)
        updates.sort(key=lambda u: u.version, reverse=True)

        # Clean up old versions
        while len(updates) > self._config.max_stored_versions:
            old = updates.pop()
            if old.file_path and old.file_path.exists():
                old.file_path.unlink()

    def get_operation(self, operation_id: str) -> UpdateOperation | None:
        """Get update operation by ID."""
        return self._operations.get(operation_id)

    def get_active_operations(self) -> list[UpdateOperation]:
        """Get all active operations."""
        return [
            op for op in self._operations.values()
            if op.state not in (UpdateState.COMPLETED, UpdateState.FAILED, UpdateState.ROLLED_BACK)
        ]

    def on_update(self, callback: Callable[[UpdateOperation], None]) -> None:
        """Register update callback."""
        self._update_callbacks.append(callback)

    def _notify_update(self, operation: UpdateOperation) -> None:
        """Notify callbacks of update progress."""
        for callback in self._update_callbacks:
            try:
                callback(operation)
            except Exception as e:
                logger.error(f"Update callback error: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get OTA statistics."""
        return {
            **self._stats,
            "active_operations": len(self.get_active_operations()),
            "available_updates": {
                node_type.value: len(updates)
                for node_type, updates in self._available_updates.items()
            },
        }

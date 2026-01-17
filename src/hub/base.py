"""
SAIL Hub Base Types

Defines the foundational types for hub-node communication and coordination.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Types of satellite nodes."""

    ROOM = "room"  # Raspberry Pi room node
    MOBILE = "mobile"  # Mobile app node
    VEHICLE = "vehicle"  # Vehicle OBD-II node
    HUB = "hub"  # The central hub itself


class NodeState(str, Enum):
    """State of a node connection."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    SYNCING = "syncing"
    READY = "ready"
    ERROR = "error"
    OFFLINE = "offline"


class HubState(str, Enum):
    """State of the central hub."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"  # Running with some issues
    STOPPING = "stopping"
    ERROR = "error"


class NodeCapability(str, Enum):
    """Capabilities a node may have."""

    VOICE_INPUT = "voice_input"
    VOICE_OUTPUT = "voice_output"
    DISPLAY = "display"
    GPS = "gps"
    MOTION = "motion"
    OBD = "obd"
    BIOMETRICS = "biometrics"
    LED_INDICATOR = "led_indicator"
    PUSH_NOTIFICATION = "push_notification"
    OFFLINE_PROCESSING = "offline_processing"
    WAKE_WORD = "wake_word"
    LOCAL_STT = "local_stt"
    LOCAL_TTS = "local_tts"


@dataclass
class NodeInfo:
    """Information about a registered node."""

    node_id: str
    node_type: NodeType
    name: str
    capabilities: list[NodeCapability] = field(default_factory=list)
    firmware_version: str = "0.0.0"
    hardware_info: dict[str, Any] = field(default_factory=dict)
    location: str | None = None  # Physical location description
    registered_at: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Node:
    """
    Represents a connected satellite node.

    Tracks connection state, capabilities, and health metrics.
    """

    info: NodeInfo
    state: NodeState = NodeState.DISCONNECTED
    ip_address: str | None = None
    port: int | None = None
    auth_token: str | None = None
    session_id: str | None = None

    # Health metrics
    latency_ms: float = 0.0
    last_heartbeat: datetime | None = None
    missed_heartbeats: int = 0
    error_count: int = 0
    last_error: str | None = None

    # Context tracking
    last_sync_time: datetime | None = None
    pending_sync: bool = False
    context_version: int = 0

    def is_online(self) -> bool:
        """Check if node is online and responsive."""
        return self.state in (
            NodeState.CONNECTED,
            NodeState.AUTHENTICATED,
            NodeState.SYNCING,
            NodeState.READY,
        )

    def is_ready(self) -> bool:
        """Check if node is fully ready for operations."""
        return self.state == NodeState.READY

    def has_capability(self, capability: NodeCapability) -> bool:
        """Check if node has a specific capability."""
        return capability in self.info.capabilities

    def update_heartbeat(self, latency_ms: float = 0.0) -> None:
        """Update heartbeat tracking."""
        self.last_heartbeat = datetime.now()
        self.latency_ms = latency_ms
        self.missed_heartbeats = 0
        self.info.last_seen = datetime.now()

    def record_error(self, error: str) -> None:
        """Record an error occurrence."""
        self.error_count += 1
        self.last_error = error

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "node_id": self.info.node_id,
            "node_type": self.info.node_type.value,
            "name": self.info.name,
            "state": self.state.value,
            "capabilities": [c.value for c in self.info.capabilities],
            "ip_address": self.ip_address,
            "port": self.port,
            "latency_ms": self.latency_ms,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "last_seen": self.info.last_seen.isoformat(),
            "firmware_version": self.info.firmware_version,
            "context_version": self.context_version,
            "error_count": self.error_count,
        }


class HubConfig(BaseModel):
    """Configuration for the central hub."""

    # Network settings
    bind_address: str = Field(default="0.0.0.0", description="Address to bind hub server")
    port: int = Field(default=8420, ge=1024, le=65535, description="Hub server port")
    discovery_port: int = Field(
        default=8421, ge=1024, le=65535, description="Node discovery broadcast port"
    )

    # Authentication
    require_auth: bool = Field(default=True, description="Require node authentication")
    auth_token_ttl_hours: int = Field(default=24, description="Auth token TTL in hours")
    max_auth_attempts: int = Field(default=5, description="Max failed auth attempts before lockout")

    # Heartbeat and health
    heartbeat_interval_seconds: int = Field(default=30, ge=5, le=300, description="Heartbeat interval")
    heartbeat_timeout_seconds: int = Field(default=90, ge=15, le=600, description="Heartbeat timeout")
    max_missed_heartbeats: int = Field(default=3, ge=1, le=10, description="Max missed heartbeats")

    # Context sync
    sync_interval_seconds: int = Field(default=5, ge=1, le=60, description="Context sync interval")
    sync_batch_size: int = Field(default=100, ge=10, le=1000, description="Max entries per sync")
    offline_buffer_size: int = Field(default=1000, ge=100, le=10000, description="Offline buffer size")

    # Deployment
    enable_ota_updates: bool = Field(default=True, description="Enable OTA updates")
    update_check_interval_hours: int = Field(default=6, description="Update check interval")
    firmware_path: Path = Field(default=Path("firmware"), description="Firmware storage path")

    # Security
    tls_enabled: bool = Field(default=True, description="Enable TLS encryption")
    tls_cert_path: Path | None = Field(default=None, description="TLS certificate path")
    tls_key_path: Path | None = Field(default=None, description="TLS private key path")

    # Limits
    max_nodes: int = Field(default=20, ge=1, le=100, description="Maximum connected nodes")
    max_requests_per_second: int = Field(default=100, ge=10, le=1000, description="Rate limit")


def generate_node_id(node_type: NodeType, hardware_id: str | None = None) -> str:
    """
    Generate a unique node ID.

    Args:
        node_type: Type of node
        hardware_id: Optional hardware identifier for deterministic ID

    Returns:
        Unique node identifier
    """
    if hardware_id:
        # Deterministic ID based on hardware
        hash_input = f"{node_type.value}:{hardware_id}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    else:
        # Random ID
        return f"{node_type.value[:3]}_{secrets.token_hex(8)}"


def generate_auth_token() -> str:
    """Generate a secure authentication token."""
    return secrets.token_urlsafe(32)

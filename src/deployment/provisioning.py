"""
SAIL Node Provisioning

Automated provisioning of satellite nodes with configuration
deployment and initial setup.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.hub.base import NodeType

logger = logging.getLogger(__name__)


class ProvisioningState(str, Enum):
    """State of provisioning process."""

    PENDING = "pending"
    DISCOVERING = "discovering"
    CONNECTING = "connecting"
    CONFIGURING = "configuring"
    INSTALLING = "installing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProvisioningResult:
    """Result of a provisioning operation."""

    node_id: str
    node_type: NodeType
    success: bool
    state: ProvisioningState
    message: str = ""
    config_hash: str | None = None
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "success": self.success,
            "state": self.state.value,
            "message": self.message,
            "config_hash": self.config_hash,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "errors": self.errors,
        }


class ProvisioningConfig(BaseModel):
    """Configuration for node provisioning."""

    # Discovery
    discovery_timeout_seconds: int = Field(default=30, description="Discovery timeout")
    discovery_port: int = Field(default=8421, description="Discovery broadcast port")

    # Connection
    connection_timeout_seconds: int = Field(default=60, description="Connection timeout")
    max_retries: int = Field(default=3, description="Max connection retries")

    # Configuration
    config_template_path: Path = Field(
        default=Path("config/templates"), description="Config template path"
    )
    config_encryption_enabled: bool = Field(default=True, description="Encrypt node configs")

    # Installation
    firmware_path: Path = Field(default=Path("firmware"), description="Firmware storage path")
    install_timeout_seconds: int = Field(default=300, description="Installation timeout")

    # Verification
    verify_connection: bool = Field(default=True, description="Verify after provisioning")
    verify_timeout_seconds: int = Field(default=30, description="Verification timeout")


@dataclass
class NodeTemplate:
    """Template for node configuration."""

    node_type: NodeType
    template_name: str
    base_config: dict[str, Any]
    required_params: list[str]
    optional_params: list[str] = field(default_factory=list)


class NodeProvisioner:
    """
    Automated node provisioning system.

    Handles discovery, configuration, and initial setup
    of satellite nodes.
    """

    def __init__(self, config: ProvisioningConfig | None = None) -> None:
        """Initialize node provisioner."""
        self._config = config or ProvisioningConfig()

        # Templates for each node type
        self._templates: dict[NodeType, NodeTemplate] = {}
        self._load_default_templates()

        # Provisioning state
        self._active_provisions: dict[str, ProvisioningResult] = {}
        self._completed_provisions: list[ProvisioningResult] = []

        # Statistics
        self._stats = {
            "total_provisioned": 0,
            "successful": 0,
            "failed": 0,
        }

    def _load_default_templates(self) -> None:
        """Load default configuration templates."""
        self._templates[NodeType.ROOM] = NodeTemplate(
            node_type=NodeType.ROOM,
            template_name="room_node",
            base_config={
                "audio_device": None,
                "sample_rate": 16000,
                "wake_word_enabled": True,
                "led_enabled": True,
            },
            required_params=["name", "location"],
            optional_params=["audio_device", "led_gpio_pin"],
        )

        self._templates[NodeType.MOBILE] = NodeTemplate(
            node_type=NodeType.MOBILE,
            template_name="mobile_node",
            base_config={
                "gps_enabled": True,
                "push_enabled": True,
                "background_enabled": True,
            },
            required_params=["name"],
            optional_params=["push_token", "gps_accuracy"],
        )

        self._templates[NodeType.VEHICLE] = NodeTemplate(
            node_type=NodeType.VEHICLE,
            template_name="vehicle_node",
            base_config={
                "obd_enabled": True,
                "accident_detection_enabled": True,
                "bluetooth_audio_enabled": True,
            },
            required_params=["name"],
            optional_params=["obd_port", "obd_protocol"],
        )

    async def provision_node(
        self,
        node_type: NodeType,
        name: str,
        params: dict[str, Any] | None = None,
        ip_address: str | None = None,
    ) -> ProvisioningResult:
        """
        Provision a new node.

        Args:
            node_type: Type of node to provision
            name: Node name
            params: Configuration parameters
            ip_address: Optional IP address (if known)

        Returns:
            Provisioning result
        """
        provision_id = secrets.token_hex(8)
        params = params or {}

        result = ProvisioningResult(
            node_id=provision_id,
            node_type=node_type,
            success=False,
            state=ProvisioningState.PENDING,
        )

        self._active_provisions[provision_id] = result

        try:
            # Get template
            template = self._templates.get(node_type)
            if not template:
                raise ValueError(f"No template for node type: {node_type.value}")

            # Validate required parameters
            params["name"] = name
            for required in template.required_params:
                if required not in params:
                    raise ValueError(f"Missing required parameter: {required}")

            # Discover node if IP not provided
            if not ip_address:
                result.state = ProvisioningState.DISCOVERING
                logger.info(f"Discovering {node_type.value} node...")

                ip_address = await self._discover_node(node_type)
                if not ip_address:
                    raise TimeoutError("Node discovery timed out")

            # Connect to node
            result.state = ProvisioningState.CONNECTING
            logger.info(f"Connecting to node at {ip_address}...")

            connected = await self._connect_to_node(ip_address)
            if not connected:
                raise ConnectionError("Failed to connect to node")

            # Generate configuration
            result.state = ProvisioningState.CONFIGURING
            logger.info("Generating node configuration...")

            node_config = self._generate_config(template, params)
            config_hash = self._hash_config(node_config)

            # Deploy configuration
            result.state = ProvisioningState.INSTALLING
            logger.info("Deploying configuration...")

            await self._deploy_config(ip_address, node_config)

            # Verify installation
            if self._config.verify_connection:
                result.state = ProvisioningState.VERIFYING
                logger.info("Verifying installation...")

                verified = await self._verify_node(ip_address, config_hash)
                if not verified:
                    raise RuntimeError("Verification failed")

            # Success
            result.state = ProvisioningState.COMPLETED
            result.success = True
            result.message = f"Successfully provisioned {node_type.value} node: {name}"
            result.config_hash = config_hash
            result.completed_at = datetime.now()

            self._stats["total_provisioned"] += 1
            self._stats["successful"] += 1

            logger.info(result.message)

        except Exception as e:
            result.state = ProvisioningState.FAILED
            result.success = False
            result.message = str(e)
            result.errors.append(str(e))
            result.completed_at = datetime.now()

            self._stats["failed"] += 1
            logger.error(f"Provisioning failed: {e}")

        finally:
            del self._active_provisions[provision_id]
            self._completed_provisions.append(result)

        return result

    async def _discover_node(self, node_type: NodeType) -> str | None:
        """Discover a node on the network."""
        try:
            # Send discovery broadcast
            loop = asyncio.get_event_loop()

            transport, _ = await loop.create_datagram_endpoint(
                lambda: _DiscoveryProtocol(node_type),
                local_addr=("0.0.0.0", 0),
                allow_broadcast=True,
            )

            try:
                # Send discovery message
                discovery_msg = f"SAIL_PROVISION:{node_type.value}".encode()
                transport.sendto(
                    discovery_msg,
                    ("255.255.255.255", self._config.discovery_port),
                )

                # Wait for response (simulated)
                await asyncio.sleep(2)
                return None  # Would return discovered IP

            finally:
                transport.close()

        except Exception as e:
            logger.error(f"Discovery error: {e}")
            return None

    async def _connect_to_node(self, ip_address: str) -> bool:
        """Connect to node for provisioning."""
        for attempt in range(self._config.max_retries):
            try:
                # Would establish provisioning connection
                # For now, simulate
                await asyncio.sleep(1)
                return True

            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(2)

        return False

    def _generate_config(
        self,
        template: NodeTemplate,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate node configuration from template."""
        config = template.base_config.copy()

        # Add hub connection info
        config["hub_address"] = "localhost"  # Would be actual hub address
        config["hub_port"] = 8420

        # Generate pre-shared key
        config["pre_shared_key"] = secrets.token_urlsafe(32)

        # Apply parameters
        for key, value in params.items():
            config[key] = value

        return config

    def _hash_config(self, config: dict[str, Any]) -> str:
        """Generate hash of configuration."""
        import json

        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    async def _deploy_config(
        self,
        ip_address: str,
        config: dict[str, Any],
    ) -> None:
        """Deploy configuration to node."""
        # Would transfer config over secure connection
        # For now, simulate
        logger.info(f"Deploying config to {ip_address}")
        await asyncio.sleep(2)

    async def _verify_node(self, ip_address: str, config_hash: str) -> bool:
        """Verify node installation."""
        try:
            # Would check node status and config hash
            # For now, simulate
            await asyncio.sleep(1)
            return True

        except Exception as e:
            logger.error(f"Verification error: {e}")
            return False

    def get_template(self, node_type: NodeType) -> NodeTemplate | None:
        """Get template for node type."""
        return self._templates.get(node_type)

    def add_template(self, template: NodeTemplate) -> None:
        """Add or update a configuration template."""
        self._templates[template.node_type] = template

    def get_stats(self) -> dict[str, Any]:
        """Get provisioning statistics."""
        return {
            **self._stats,
            "active_provisions": len(self._active_provisions),
            "completed_provisions": len(self._completed_provisions),
        }

    def get_provision_history(
        self,
        limit: int = 10,
    ) -> list[ProvisioningResult]:
        """Get recent provisioning history."""
        return self._completed_provisions[-limit:]


class _DiscoveryProtocol(asyncio.DatagramProtocol):
    """UDP discovery protocol for provisioning."""

    def __init__(self, node_type: NodeType) -> None:
        self._node_type = node_type
        self._transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:  # type: ignore
        self._transport = transport

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        logger.debug(f"Discovery response from {addr}: {data}")

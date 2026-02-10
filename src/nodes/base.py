"""
SAIL Base Node

Abstract base class for all satellite nodes in the distributed SAIL system.
Provides common functionality for hub communication, context sync, and events.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from src.core.events import EventEmitter
from src.hub.base import NodeCapability, NodeState, NodeType

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.nodes.protocol import SecureChannel

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """Types of messages between nodes and hub."""

    # Control messages
    REGISTER = "register"
    AUTHENTICATE = "authenticate"
    HEARTBEAT = "heartbeat"
    DISCONNECT = "disconnect"

    # Data messages
    QUERY = "query"
    RESPONSE = "response"
    CONTEXT_UPDATE = "context_update"
    SENSOR_DATA = "sensor_data"

    # Events
    INTERVENTION_TRIGGER = "intervention_trigger"
    USER_IDENTIFY = "user_identify"
    STATUS = "status"
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"

    # System
    ERROR = "error"
    ACK = "ack"
    BROADCAST = "broadcast"


@dataclass
class NodeMessage:
    """Message between node and hub."""

    message_id: str
    message_type: MessageType
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source_node_id: str | None = None
    target_node_id: str | None = None
    requires_ack: bool = False
    correlation_id: str | None = None  # For request-response correlation

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "requires_ack": self.requires_ack,
            "correlation_id": self.correlation_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeMessage:
        """Create from dictionary."""
        return cls(
            message_id=data["message_id"],
            message_type=MessageType(data["message_type"]),
            payload=data.get("payload", {}),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if isinstance(data["timestamp"], str)
            else data["timestamp"],
            source_node_id=data.get("source_node_id"),
            target_node_id=data.get("target_node_id"),
            requires_ack=data.get("requires_ack", False),
            correlation_id=data.get("correlation_id"),
        )


@dataclass
class NodeEvent:
    """Event emitted by a node."""

    event_type: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class NodeConfig(BaseModel):
    """Base configuration for all nodes."""

    # Identity
    name: str = Field(..., description="Node display name")
    location: str | None = Field(default=None, description="Physical location description")

    # Hub connection
    hub_address: str = Field(default="localhost", description="Hub address")
    hub_port: int = Field(default=8420, description="Hub port")
    hub_discovery_port: int = Field(default=8421, description="Discovery broadcast port")

    # Authentication
    pre_shared_key: str | None = Field(default=None, description="Pre-shared key for auth")

    # Connection settings
    auto_reconnect: bool = Field(default=True, description="Auto-reconnect on disconnect")
    reconnect_interval_seconds: int = Field(default=5, description="Reconnect interval")
    max_reconnect_attempts: int = Field(default=10, description="Max reconnect attempts")
    heartbeat_interval_seconds: int = Field(default=30, description="Heartbeat interval")

    # Offline mode
    enable_offline_mode: bool = Field(default=True, description="Enable offline operation")
    offline_buffer_size: int = Field(default=500, description="Offline buffer size")

    # TLS
    tls_enabled: bool = Field(default=True, description="Enable TLS")
    tls_verify: bool = Field(default=True, description="Verify TLS certificates")
    tls_ca_cert_path: str | None = Field(default=None, description="CA certificate path")


class BaseNode(ABC, EventEmitter[NodeEvent]):
    """
    Abstract base class for satellite nodes.

    Provides common functionality for hub communication, heartbeat,
    context synchronization, and event handling.
    """

    def __init__(
        self,
        node_type: NodeType,
        config: NodeConfig,
        capabilities: list[NodeCapability] | None = None,
    ) -> None:
        """
        Initialize base node.

        Args:
            node_type: Type of this node
            config: Node configuration
            capabilities: Node capabilities
        """
        self._node_type = node_type
        self._config = config
        self._capabilities = capabilities or []

        # Identity
        self._node_id: str | None = None
        self._auth_token: str | None = None

        # State
        self._state = NodeState.DISCONNECTED
        self._started_at: datetime | None = None

        # Connection
        self._channel: SecureChannel | None = None
        self._reconnect_attempts = 0

        # Background tasks
        self._tasks: list[asyncio.Task] = []
        self._heartbeat_task: asyncio.Task | None = None
        self._receive_task: asyncio.Task | None = None

        # Message handling
        self._message_handlers: dict[MessageType, Callable] = {}
        self._pending_responses: dict[str, asyncio.Future] = {}
        self._register_default_handlers()

        self.__init_emitter__()

        # Offline buffer
        self._offline_buffer: list[NodeMessage] = []

        # Statistics
        self._stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "errors": 0,
            "reconnections": 0,
        }

    def _register_default_handlers(self) -> None:
        """Register default message handlers."""
        self._message_handlers = {
            MessageType.ACK: self._handle_ack,
            MessageType.RESPONSE: self._handle_response,
            MessageType.ERROR: self._handle_error,
            MessageType.BROADCAST: self._handle_broadcast,
            MessageType.SYNC_RESPONSE: self._handle_sync_response,
        }

    @property
    @abstractmethod
    def firmware_version(self) -> str:
        """Get node firmware version."""
        pass

    @property
    @abstractmethod
    def hardware_info(self) -> dict[str, Any]:
        """Get hardware information."""
        pass

    async def start(self) -> None:
        """Start the node and connect to hub."""
        if self._state != NodeState.DISCONNECTED:
            return

        logger.info(f"Starting {self._node_type.value} node: {self._config.name}")
        self._started_at = datetime.now()
        self._state = NodeState.CONNECTING

        try:
            # Connect to hub
            await self._connect_to_hub()

            # Register with hub
            await self._register()

            # Start background tasks
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._tasks.append(self._heartbeat_task)

            self._receive_task = asyncio.create_task(self._receive_loop())
            self._tasks.append(self._receive_task)

            # Start node-specific initialization
            await self._on_start()

            self._state = NodeState.READY
            self._emit_event("started")
            logger.info(f"Node started: {self._node_id}")

        except Exception as e:
            self._state = NodeState.ERROR
            logger.error(f"Failed to start node: {e}")
            raise

    async def stop(self) -> None:
        """Stop the node and disconnect from hub."""
        if self._state == NodeState.DISCONNECTED:
            return

        logger.info(f"Stopping node: {self._node_id}")

        # Stop node-specific resources
        await self._on_stop()

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        self._tasks.clear()

        # Disconnect from hub
        await self._disconnect()

        self._state = NodeState.DISCONNECTED
        self._emit_event("stopped")
        logger.info("Node stopped")

    async def _connect_to_hub(self) -> None:
        """Establish connection to hub."""
        from src.nodes.protocol import ProtocolConfig, SecureChannel

        protocol_config = ProtocolConfig(
            host=self._config.hub_address,
            port=self._config.hub_port,
            tls_enabled=self._config.tls_enabled,
            tls_verify=self._config.tls_verify,
        )

        self._channel = SecureChannel(protocol_config)
        await self._channel.connect()

        self._state = NodeState.CONNECTED
        logger.info(f"Connected to hub at {self._config.hub_address}:{self._config.hub_port}")

    async def _disconnect(self) -> None:
        """Disconnect from hub."""
        if self._channel:
            # Send disconnect message
            with contextlib.suppress(Exception):
                await self._send_message(NodeMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.DISCONNECT,
                    source_node_id=self._node_id,
                ))

            await self._channel.disconnect()
            self._channel = None

        self._auth_token = None

    async def _register(self) -> None:
        """Register with the hub."""
        registration_payload = {
            "node_type": self._node_type.value,
            "name": self._config.name,
            "capabilities": [c.value for c in self._capabilities],
            "firmware_version": self.firmware_version,
            "hardware_info": self.hardware_info,
            "location": self._config.location,
            "pre_shared_key": self._config.pre_shared_key,
        }

        response = await self._send_request(
            MessageType.REGISTER,
            registration_payload,
            timeout=10.0,
        )

        if response.get("success"):
            self._node_id = response.get("node_id")
            self._auth_token = response.get("auth_token")
            self._state = NodeState.AUTHENTICATED
            logger.info(f"Registered with hub as {self._node_id}")
        else:
            raise ConnectionError(f"Registration failed: {response.get('error')}")

    async def _send_message(self, message: NodeMessage) -> None:
        """Send a message to the hub."""
        if not self._channel or not self._channel.is_connected:
            if self._config.enable_offline_mode:
                self._buffer_offline(message)
                return
            raise ConnectionError("Not connected to hub")

        message.source_node_id = self._node_id
        await self._channel.send(message.to_dict())
        self._stats["messages_sent"] += 1

    async def _send_request(
        self,
        message_type: MessageType,
        payload: dict[str, Any],
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """
        Send a request and wait for response.

        Args:
            message_type: Type of request
            payload: Request payload
            timeout: Response timeout

        Returns:
            Response payload
        """
        message_id = str(uuid.uuid4())
        message = NodeMessage(
            message_id=message_id,
            message_type=message_type,
            payload=payload,
            source_node_id=self._node_id,
            requires_ack=True,
        )

        # Create future for response
        future: asyncio.Future = asyncio.Future()
        self._pending_responses[message_id] = future

        try:
            await self._send_message(message)
            return await asyncio.wait_for(future, timeout=timeout)

        except TimeoutError:
            raise TimeoutError(f"Request {message_id} timed out")

        finally:
            self._pending_responses.pop(message_id, None)

    def _buffer_offline(self, message: NodeMessage) -> None:
        """Buffer message for offline mode."""
        if len(self._offline_buffer) >= self._config.offline_buffer_size:
            self._offline_buffer.pop(0)

        self._offline_buffer.append(message)
        logger.debug(f"Buffered message for offline: {message.message_type.value}")

    async def _flush_offline_buffer(self) -> int:
        """Flush offline buffer to hub."""
        if not self._offline_buffer or not self._channel:
            return 0

        flushed = 0
        while self._offline_buffer:
            message = self._offline_buffer.pop(0)
            try:
                await self._send_message(message)
                flushed += 1
            except Exception as e:
                # Put back and stop
                self._offline_buffer.insert(0, message)
                logger.error(f"Failed to flush offline buffer: {e}")
                break

        if flushed:
            logger.info(f"Flushed {flushed} messages from offline buffer")

        return flushed

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to hub."""
        while self._state in (NodeState.AUTHENTICATED, NodeState.READY):
            try:
                await asyncio.sleep(self._config.heartbeat_interval_seconds)

                if self._state not in (NodeState.AUTHENTICATED, NodeState.READY):
                    break

                message = NodeMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.HEARTBEAT,
                    source_node_id=self._node_id,
                    payload={"timestamp": datetime.now().isoformat()},
                )

                await self._send_message(message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                self._stats["errors"] += 1

                if self._config.auto_reconnect:
                    await self._attempt_reconnect()

    async def _receive_loop(self) -> None:
        """Receive messages from hub."""
        while self._state != NodeState.DISCONNECTED:
            try:
                if not self._channel or not self._channel.is_connected:
                    await asyncio.sleep(1)
                    continue

                data = await self._channel.receive()
                if data:
                    message = NodeMessage.from_dict(data)
                    self._stats["messages_received"] += 1
                    await self._handle_message(message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Receive error: {e}")
                self._stats["errors"] += 1

    async def _handle_message(self, message: NodeMessage) -> None:
        """Handle incoming message."""
        handler = self._message_handlers.get(message.message_type)
        if handler:
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Message handler error: {e}")
        else:
            logger.warning(f"No handler for message type: {message.message_type.value}")

    async def _handle_ack(self, message: NodeMessage) -> None:
        """Handle acknowledgment message."""
        correlation_id = message.correlation_id
        if correlation_id and correlation_id in self._pending_responses:
            future = self._pending_responses[correlation_id]
            if not future.done():
                future.set_result(message.payload)

    async def _handle_response(self, message: NodeMessage) -> None:
        """Handle response message."""
        correlation_id = message.correlation_id
        if correlation_id and correlation_id in self._pending_responses:
            future = self._pending_responses[correlation_id]
            if not future.done():
                future.set_result(message.payload)

    async def _handle_error(self, message: NodeMessage) -> None:
        """Handle error message."""
        error = message.payload.get("error", "Unknown error")
        logger.error(f"Error from hub: {error}")

        correlation_id = message.correlation_id
        if correlation_id and correlation_id in self._pending_responses:
            future = self._pending_responses[correlation_id]
            if not future.done():
                future.set_exception(Exception(error))

    async def _handle_broadcast(self, message: NodeMessage) -> None:
        """Handle broadcast message from hub."""
        self._emit_event("broadcast", message.payload)

    async def _handle_sync_response(self, message: NodeMessage) -> None:
        """Handle context sync response."""
        pass  # Override in subclasses

    async def _attempt_reconnect(self) -> None:
        """Attempt to reconnect to hub."""
        if not self._config.auto_reconnect:
            return

        if self._reconnect_attempts >= self._config.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            self._state = NodeState.OFFLINE
            return

        self._reconnect_attempts += 1
        self._stats["reconnections"] += 1

        logger.info(f"Reconnection attempt {self._reconnect_attempts}")

        try:
            await self._connect_to_hub()
            await self._register()
            await self._flush_offline_buffer()
            self._reconnect_attempts = 0
            self._state = NodeState.READY

        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            await asyncio.sleep(self._config.reconnect_interval_seconds)

    # Abstract methods for subclasses

    @abstractmethod
    async def _on_start(self) -> None:
        """Called when node starts. Initialize node-specific resources."""
        pass

    @abstractmethod
    async def _on_stop(self) -> None:
        """Called when node stops. Cleanup node-specific resources."""
        pass

    # Public API

    async def send_query(self, text: str, user_id: str | None = None) -> dict[str, Any]:
        """Send a query to the hub."""
        return await self._send_request(
            MessageType.QUERY,
            {"text": text, "user_id": user_id},
        )

    async def send_sensor_data(
        self,
        sensor_type: str,
        data: dict[str, Any],
    ) -> None:
        """Send sensor data to hub."""
        message = NodeMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.SENSOR_DATA,
            payload={"sensor_type": sensor_type, "data": data},
            source_node_id=self._node_id,
        )
        await self._send_message(message)

    async def trigger_intervention(
        self,
        intervention_type: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Trigger an intervention on the hub."""
        return await self._send_request(
            MessageType.INTERVENTION_TRIGGER,
            {"type": intervention_type, "context": context},
        )

    async def request_sync(self, context_data: dict[str, Any]) -> dict[str, Any]:
        """Request context synchronization."""
        return await self._send_request(
            MessageType.SYNC_REQUEST,
            {"context": context_data},
        )

    def register_message_handler(
        self,
        message_type: MessageType,
        handler: Callable,
    ) -> None:
        """Register a custom message handler."""
        self._message_handlers[message_type] = handler

    def _emit_event(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """Emit event to callbacks."""
        self._emit(NodeEvent(event_type=event_type, data=data or {}))

    def get_status(self) -> dict[str, Any]:
        """Get node status."""
        return {
            "node_id": self._node_id,
            "node_type": self._node_type.value,
            "name": self._config.name,
            "state": self._state.value,
            "capabilities": [c.value for c in self._capabilities],
            "firmware_version": self.firmware_version,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "uptime_seconds": (datetime.now() - self._started_at).total_seconds()
            if self._started_at
            else 0,
            "connected": self._channel.is_connected if self._channel else False,
            "offline_buffer_size": len(self._offline_buffer),
            "stats": self._stats.copy(),
        }

    @property
    def node_id(self) -> str | None:
        """Get node ID."""
        return self._node_id

    @property
    def state(self) -> NodeState:
        """Get node state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected to hub."""
        return self._channel is not None and self._channel.is_connected

    @property
    def is_ready(self) -> bool:
        """Check if node is ready."""
        return self._state == NodeState.READY

    async def __aenter__(self) -> BaseNode:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()

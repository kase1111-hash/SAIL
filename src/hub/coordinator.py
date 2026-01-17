"""
SAIL Hub Coordinator

The central coordinator for managing satellite nodes, handling requests,
and orchestrating the distributed SAIL system.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable

from src.hub.base import (
    HubConfig,
    HubState,
    Node,
    NodeCapability,
    NodeInfo,
    NodeState,
    NodeType,
    generate_auth_token,
    generate_node_id,
)

if TYPE_CHECKING:
    from src.context.manager import ContextBufferManager
    from src.hub.context_sync import ContextSync
    from src.hub.router import RequestRouter

logger = logging.getLogger(__name__)


@dataclass
class HubEvent:
    """Event emitted by the hub coordinator."""

    event_type: str
    node_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class NodeRegistration:
    """Registration request from a node."""

    node_type: NodeType
    name: str
    capabilities: list[NodeCapability]
    firmware_version: str = "0.0.0"
    hardware_id: str | None = None
    hardware_info: dict[str, Any] = field(default_factory=dict)
    location: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Request:
    """A request from a node to the hub."""

    request_id: str
    source_node_id: str
    request_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 0
    user_id: str | None = None


@dataclass
class Response:
    """A response from the hub to a node."""

    request_id: str
    success: bool
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)


class HubCoordinator:
    """
    Central hub coordinator for the distributed SAIL system.

    Manages node registration, authentication, request routing,
    context synchronization, and health monitoring.
    """

    def __init__(
        self,
        config: HubConfig | None = None,
        context_manager: ContextBufferManager | None = None,
    ) -> None:
        """
        Initialize the hub coordinator.

        Args:
            config: Hub configuration
            context_manager: Optional context buffer manager
        """
        self._config = config or HubConfig()
        self._context_manager = context_manager

        # Node management
        self._nodes: dict[str, Node] = {}
        self._pending_registrations: dict[str, NodeRegistration] = {}
        self._auth_attempts: dict[str, int] = {}
        self._lockouts: dict[str, datetime] = {}

        # State
        self._state = HubState.STOPPED
        self._started_at: datetime | None = None

        # Components
        self._context_sync: ContextSync | None = None
        self._router: RequestRouter | None = None

        # Background tasks
        self._tasks: list[asyncio.Task] = []
        self._heartbeat_task: asyncio.Task | None = None
        self._sync_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None

        # Event callbacks
        self._event_callbacks: list[Callable[[HubEvent], None]] = []

        # Request handlers
        self._request_handlers: dict[str, Callable] = {}
        self._register_default_handlers()

        # Statistics
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_registrations": 0,
            "total_disconnections": 0,
        }

    def _register_default_handlers(self) -> None:
        """Register default request handlers."""
        self._request_handlers = {
            "query": self._handle_query,
            "context_update": self._handle_context_update,
            "sensor_data": self._handle_sensor_data,
            "intervention_trigger": self._handle_intervention_trigger,
            "user_identify": self._handle_user_identify,
            "status": self._handle_status_request,
            "sync_request": self._handle_sync_request,
        }

    async def start(self) -> None:
        """Start the hub coordinator."""
        if self._state == HubState.RUNNING:
            return

        logger.info("Starting hub coordinator")
        self._state = HubState.STARTING
        self._started_at = datetime.now()

        try:
            # Initialize context sync if context manager available
            if self._context_manager:
                from src.hub.context_sync import ContextSync

                self._context_sync = ContextSync(
                    context_manager=self._context_manager,
                    sync_interval=self._config.sync_interval_seconds,
                    batch_size=self._config.sync_batch_size,
                )

            # Initialize router
            from src.hub.router import RequestRouter

            self._router = RequestRouter()

            # Start background tasks
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._tasks.append(self._heartbeat_task)

            if self._context_sync:
                self._sync_task = asyncio.create_task(self._sync_loop())
                self._tasks.append(self._sync_task)

            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._tasks.append(self._cleanup_task)

            self._state = HubState.RUNNING
            self._emit_event("hub_started")
            logger.info(f"Hub coordinator started on port {self._config.port}")

        except Exception as e:
            self._state = HubState.ERROR
            logger.error(f"Failed to start hub coordinator: {e}")
            raise

    async def stop(self) -> None:
        """Stop the hub coordinator."""
        if self._state == HubState.STOPPED:
            return

        logger.info("Stopping hub coordinator")
        self._state = HubState.STOPPING

        # Cancel all background tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()

        # Disconnect all nodes
        for node_id in list(self._nodes.keys()):
            await self._disconnect_node(node_id, reason="Hub shutting down")

        self._state = HubState.STOPPED
        self._emit_event("hub_stopped")
        logger.info("Hub coordinator stopped")

    async def register_node(
        self,
        registration: NodeRegistration,
        ip_address: str,
        port: int,
    ) -> tuple[str, str]:
        """
        Register a new node with the hub.

        Args:
            registration: Node registration details
            ip_address: Node IP address
            port: Node port

        Returns:
            Tuple of (node_id, auth_token)

        Raises:
            ValueError: If registration fails
        """
        if len(self._nodes) >= self._config.max_nodes:
            raise ValueError("Maximum node limit reached")

        # Check for lockout
        if ip_address in self._lockouts:
            if datetime.now() < self._lockouts[ip_address]:
                raise ValueError("Registration temporarily locked due to auth failures")
            del self._lockouts[ip_address]

        # Generate node ID
        node_id = generate_node_id(registration.node_type, registration.hardware_id)

        # Check if already registered
        if node_id in self._nodes:
            existing = self._nodes[node_id]
            if existing.is_online():
                raise ValueError(f"Node {node_id} already registered and online")
            # Re-registration of offline node
            logger.info(f"Re-registering previously offline node: {node_id}")

        # Create node info
        node_info = NodeInfo(
            node_id=node_id,
            node_type=registration.node_type,
            name=registration.name,
            capabilities=registration.capabilities,
            firmware_version=registration.firmware_version,
            hardware_info=registration.hardware_info,
            location=registration.location,
            metadata=registration.metadata,
        )

        # Generate auth token
        auth_token = generate_auth_token()

        # Create node
        node = Node(
            info=node_info,
            state=NodeState.AUTHENTICATED,
            ip_address=ip_address,
            port=port,
            auth_token=auth_token,
        )

        self._nodes[node_id] = node
        self._stats["total_registrations"] += 1

        self._emit_event("node_registered", node_id, {
            "node_type": registration.node_type.value,
            "name": registration.name,
            "capabilities": [c.value for c in registration.capabilities],
        })

        logger.info(f"Node registered: {node_id} ({registration.name})")
        return node_id, auth_token

    async def authenticate_node(
        self,
        node_id: str,
        auth_token: str,
    ) -> bool:
        """
        Authenticate a node connection.

        Args:
            node_id: Node identifier
            auth_token: Authentication token

        Returns:
            True if authentication successful
        """
        node = self._nodes.get(node_id)
        if not node:
            logger.warning(f"Auth attempt for unknown node: {node_id}")
            return False

        ip_key = node.ip_address or node_id
        if ip_key in self._lockouts:
            if datetime.now() < self._lockouts[ip_key]:
                return False
            del self._lockouts[ip_key]

        if node.auth_token != auth_token:
            # Track failed attempts
            self._auth_attempts[ip_key] = self._auth_attempts.get(ip_key, 0) + 1

            if self._auth_attempts[ip_key] >= self._config.max_auth_attempts:
                self._lockouts[ip_key] = datetime.now() + timedelta(hours=1)
                logger.warning(f"Node locked out due to auth failures: {node_id}")

            return False

        # Successful auth
        self._auth_attempts.pop(ip_key, None)
        node.state = NodeState.READY
        node.update_heartbeat()

        self._emit_event("node_authenticated", node_id)
        logger.info(f"Node authenticated: {node_id}")
        return True

    async def _disconnect_node(self, node_id: str, reason: str = "Unknown") -> None:
        """Disconnect a node from the hub."""
        node = self._nodes.get(node_id)
        if not node:
            return

        node.state = NodeState.DISCONNECTED
        node.auth_token = None
        self._stats["total_disconnections"] += 1

        self._emit_event("node_disconnected", node_id, {"reason": reason})
        logger.info(f"Node disconnected: {node_id} - {reason}")

    async def handle_heartbeat(self, node_id: str, latency_ms: float = 0.0) -> bool:
        """
        Handle heartbeat from a node.

        Args:
            node_id: Node identifier
            latency_ms: Round-trip latency

        Returns:
            True if heartbeat accepted
        """
        node = self._nodes.get(node_id)
        if not node or not node.is_online():
            return False

        node.update_heartbeat(latency_ms)
        return True

    async def route_request(self, request: Request) -> Response:
        """
        Route a request from a node to the appropriate handler.

        Args:
            request: The incoming request

        Returns:
            Response to send back to node
        """
        self._stats["total_requests"] += 1

        # Validate source node
        node = self._nodes.get(request.source_node_id)
        if not node or not node.is_ready():
            self._stats["failed_requests"] += 1
            return Response(
                request_id=request.request_id,
                success=False,
                error="Node not authenticated or not ready",
            )

        # Find handler
        handler = self._request_handlers.get(request.request_type)
        if not handler:
            self._stats["failed_requests"] += 1
            return Response(
                request_id=request.request_id,
                success=False,
                error=f"Unknown request type: {request.request_type}",
            )

        try:
            # Execute handler
            result = await handler(request, node)
            self._stats["successful_requests"] += 1
            return Response(
                request_id=request.request_id,
                success=True,
                payload=result,
            )

        except Exception as e:
            logger.error(f"Request handler error: {e}")
            self._stats["failed_requests"] += 1
            node.record_error(str(e))
            return Response(
                request_id=request.request_id,
                success=False,
                error=str(e),
            )

    async def sync_context(self, node_id: str, context_data: dict[str, Any]) -> dict[str, Any]:
        """
        Synchronize context with a node.

        Args:
            node_id: Node identifier
            context_data: Context data from node

        Returns:
            Updated context to send back
        """
        node = self._nodes.get(node_id)
        if not node or not node.is_ready():
            raise ValueError("Node not ready for sync")

        if not self._context_sync:
            raise ValueError("Context sync not available")

        node.state = NodeState.SYNCING
        try:
            result = await self._context_sync.sync_from_node(node_id, context_data)
            node.last_sync_time = datetime.now()
            node.context_version += 1
            return result
        finally:
            node.state = NodeState.READY

    async def broadcast(
        self,
        message_type: str,
        payload: dict[str, Any],
        target_types: list[NodeType] | None = None,
        required_capabilities: list[NodeCapability] | None = None,
    ) -> int:
        """
        Broadcast a message to multiple nodes.

        Args:
            message_type: Type of message
            payload: Message payload
            target_types: Limit to specific node types
            required_capabilities: Limit to nodes with these capabilities

        Returns:
            Number of nodes that received the message
        """
        sent_count = 0

        for node in self._nodes.values():
            if not node.is_ready():
                continue

            # Filter by type
            if target_types and node.info.node_type not in target_types:
                continue

            # Filter by capabilities
            if required_capabilities:
                if not all(node.has_capability(cap) for cap in required_capabilities):
                    continue

            # Send message (actual implementation would use network)
            try:
                await self._send_to_node(node, message_type, payload)
                sent_count += 1
            except Exception as e:
                logger.error(f"Failed to send to node {node.info.node_id}: {e}")

        return sent_count

    async def _send_to_node(
        self,
        node: Node,
        message_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Send a message to a specific node."""
        # Placeholder for actual network implementation
        # In production, this would use secure websockets or similar
        logger.debug(f"Sending {message_type} to node {node.info.node_id}")

    # Request handlers

    async def _handle_query(self, request: Request, node: Node) -> dict[str, Any]:
        """Handle a query request from a node."""
        query_text = request.payload.get("text", "")
        user_id = request.user_id

        # Add to context if available
        if self._context_manager:
            self._context_manager.add_user_input(query_text)

        # Return acknowledgment (actual LLM processing would be done elsewhere)
        return {
            "acknowledged": True,
            "query_id": request.request_id,
        }

    async def _handle_context_update(self, request: Request, node: Node) -> dict[str, Any]:
        """Handle context update from a node."""
        context_entries = request.payload.get("entries", [])

        if self._context_sync:
            await self._context_sync.merge_entries(node.info.node_id, context_entries)

        return {"merged": len(context_entries)}

    async def _handle_sensor_data(self, request: Request, node: Node) -> dict[str, Any]:
        """Handle sensor data from a node."""
        sensor_type = request.payload.get("sensor_type", "unknown")
        data = request.payload.get("data", {})

        if self._context_manager:
            self._context_manager.add_sensor_data(
                f"{sensor_type}: {data}",
                metadata={"node_id": node.info.node_id, "sensor_type": sensor_type},
            )

        return {"received": True}

    async def _handle_intervention_trigger(self, request: Request, node: Node) -> dict[str, Any]:
        """Handle intervention trigger from a node."""
        intervention_type = request.payload.get("type", "")
        context = request.payload.get("context", {})

        # Forward to intervention engine (would be integrated)
        return {
            "acknowledged": True,
            "intervention_id": request.request_id,
        }

    async def _handle_user_identify(self, request: Request, node: Node) -> dict[str, Any]:
        """Handle user identification request."""
        voice_sample = request.payload.get("voice_sample")
        confidence = request.payload.get("confidence", 0.0)

        # Would integrate with user identification system
        return {
            "user_id": None,  # Would be determined by voice ID
            "confidence": confidence,
        }

    async def _handle_status_request(self, request: Request, node: Node) -> dict[str, Any]:
        """Handle status request."""
        return self.get_status()

    async def _handle_sync_request(self, request: Request, node: Node) -> dict[str, Any]:
        """Handle context sync request."""
        context_data = request.payload.get("context", {})
        return await self.sync_context(node.info.node_id, context_data)

    # Background loops

    async def _heartbeat_loop(self) -> None:
        """Monitor node heartbeats."""
        while self._state == HubState.RUNNING:
            try:
                await asyncio.sleep(self._config.heartbeat_interval_seconds)

                now = datetime.now()
                timeout = timedelta(seconds=self._config.heartbeat_timeout_seconds)

                for node_id, node in list(self._nodes.items()):
                    if not node.is_online():
                        continue

                    if node.last_heartbeat and (now - node.last_heartbeat) > timeout:
                        node.missed_heartbeats += 1

                        if node.missed_heartbeats >= self._config.max_missed_heartbeats:
                            await self._disconnect_node(
                                node_id,
                                reason=f"Missed {node.missed_heartbeats} heartbeats",
                            )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")

    async def _sync_loop(self) -> None:
        """Periodic context synchronization."""
        while self._state == HubState.RUNNING:
            try:
                await asyncio.sleep(self._config.sync_interval_seconds)

                if self._context_sync:
                    for node_id, node in self._nodes.items():
                        if node.is_ready() and node.pending_sync:
                            try:
                                await self._context_sync.push_to_node(node_id)
                                node.pending_sync = False
                            except Exception as e:
                                logger.error(f"Sync to node {node_id} failed: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync loop error: {e}")

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of stale data."""
        while self._state == HubState.RUNNING:
            try:
                await asyncio.sleep(3600)  # Every hour

                # Clean up old lockouts
                now = datetime.now()
                self._lockouts = {
                    k: v for k, v in self._lockouts.items()
                    if v > now
                }

                # Clean up auth attempts
                self._auth_attempts = {
                    k: v for k, v in self._auth_attempts.items()
                    if v > 0
                }

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    # Utility methods

    def get_node(self, node_id: str) -> Node | None:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_nodes(
        self,
        node_type: NodeType | None = None,
        state: NodeState | None = None,
    ) -> list[Node]:
        """
        Get nodes matching criteria.

        Args:
            node_type: Filter by node type
            state: Filter by state

        Returns:
            List of matching nodes
        """
        result = []
        for node in self._nodes.values():
            if node_type and node.info.node_type != node_type:
                continue
            if state and node.state != state:
                continue
            result.append(node)
        return result

    def get_online_nodes(self) -> list[Node]:
        """Get all online nodes."""
        return [n for n in self._nodes.values() if n.is_online()]

    def get_ready_nodes(self) -> list[Node]:
        """Get all ready nodes."""
        return [n for n in self._nodes.values() if n.is_ready()]

    def get_status(self) -> dict[str, Any]:
        """Get hub status summary."""
        return {
            "state": self._state.value,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "uptime_seconds": (datetime.now() - self._started_at).total_seconds()
            if self._started_at
            else 0,
            "config": {
                "port": self._config.port,
                "max_nodes": self._config.max_nodes,
            },
            "nodes": {
                "total": len(self._nodes),
                "online": len(self.get_online_nodes()),
                "ready": len(self.get_ready_nodes()),
                "by_type": {
                    t.value: len(self.get_nodes(node_type=t))
                    for t in NodeType
                    if t != NodeType.HUB
                },
            },
            "stats": self._stats.copy(),
        }

    def register_request_handler(self, request_type: str, handler: Callable) -> None:
        """Register a custom request handler."""
        self._request_handlers[request_type] = handler

    def on_event(self, callback: Callable[[HubEvent], None]) -> None:
        """Register event callback."""
        self._event_callbacks.append(callback)

    def _emit_event(
        self,
        event_type: str,
        node_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Emit an event to callbacks."""
        event = HubEvent(
            event_type=event_type,
            node_id=node_id,
            data=data or {},
        )

        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    @property
    def state(self) -> HubState:
        """Get hub state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if hub is running."""
        return self._state == HubState.RUNNING

    @property
    def node_count(self) -> int:
        """Get total node count."""
        return len(self._nodes)

    async def __aenter__(self) -> HubCoordinator:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Async context manager exit."""
        await self.stop()

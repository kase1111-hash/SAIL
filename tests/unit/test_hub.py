"""
SAIL Hub Module Tests

Comprehensive tests for the hub coordinator, context sync, and request routing.
"""

from datetime import datetime

import pytest
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
from src.hub.context_sync import (
    ContextSync,
    SyncEntry,
    SyncStrategy,
)
from src.hub.coordinator import (
    HubCoordinator,
    HubEvent,
    NodeRegistration,
    Request,
)
from src.hub.router import (
    RateLimitError,
    RequestPriority,
    RequestRouter,
    RoutedRequest,
    RoutingStrategy,
)

# ============================================================================
# Hub Base Types Tests
# ============================================================================


class TestHubBaseTypes:
    """Tests for hub base types."""

    def test_node_type_enum(self):
        """Test NodeType enum values."""
        assert NodeType.ROOM.value == "room"
        assert NodeType.MOBILE.value == "mobile"
        assert NodeType.VEHICLE.value == "vehicle"
        assert NodeType.HUB.value == "hub"

    def test_node_state_enum(self):
        """Test NodeState enum values."""
        assert NodeState.DISCONNECTED.value == "disconnected"
        assert NodeState.CONNECTED.value == "connected"
        assert NodeState.READY.value == "ready"
        assert NodeState.ERROR.value == "error"

    def test_hub_state_enum(self):
        """Test HubState enum values."""
        assert HubState.STOPPED.value == "stopped"
        assert HubState.RUNNING.value == "running"
        assert HubState.DEGRADED.value == "degraded"

    def test_node_capability_enum(self):
        """Test NodeCapability enum values."""
        assert NodeCapability.VOICE_INPUT.value == "voice_input"
        assert NodeCapability.GPS.value == "gps"
        assert NodeCapability.OBD.value == "obd"


class TestNodeInfo:
    """Tests for NodeInfo dataclass."""

    def test_node_info_creation(self):
        """Test creating NodeInfo."""
        info = NodeInfo(
            node_id="test_node",
            node_type=NodeType.ROOM,
            name="Living Room",
            capabilities=[NodeCapability.VOICE_INPUT, NodeCapability.VOICE_OUTPUT],
        )

        assert info.node_id == "test_node"
        assert info.node_type == NodeType.ROOM
        assert info.name == "Living Room"
        assert len(info.capabilities) == 2
        assert info.firmware_version == "0.0.0"

    def test_node_info_defaults(self):
        """Test NodeInfo default values."""
        info = NodeInfo(
            node_id="test",
            node_type=NodeType.MOBILE,
            name="Test",
        )

        assert info.capabilities == []
        assert info.hardware_info == {}
        assert info.location is None


class TestNode:
    """Tests for Node dataclass."""

    @pytest.fixture
    def node(self):
        """Create a test node."""
        info = NodeInfo(
            node_id="test_node",
            node_type=NodeType.ROOM,
            name="Test Room",
            capabilities=[NodeCapability.VOICE_INPUT],
        )
        return Node(info=info)

    def test_node_creation(self, node):
        """Test creating a Node."""
        assert node.info.node_id == "test_node"
        assert node.state == NodeState.DISCONNECTED
        assert node.latency_ms == 0.0
        assert node.error_count == 0

    def test_node_is_online(self, node):
        """Test is_online method."""
        assert not node.is_online()

        node.state = NodeState.CONNECTED
        assert node.is_online()

        node.state = NodeState.READY
        assert node.is_online()

        node.state = NodeState.ERROR
        assert not node.is_online()

    def test_node_is_ready(self, node):
        """Test is_ready method."""
        assert not node.is_ready()

        node.state = NodeState.READY
        assert node.is_ready()

    def test_node_has_capability(self, node):
        """Test has_capability method."""
        assert node.has_capability(NodeCapability.VOICE_INPUT)
        assert not node.has_capability(NodeCapability.GPS)

    def test_node_update_heartbeat(self, node):
        """Test update_heartbeat method."""
        node.missed_heartbeats = 5

        node.update_heartbeat(latency_ms=50.0)

        assert node.last_heartbeat is not None
        assert node.latency_ms == 50.0
        assert node.missed_heartbeats == 0

    def test_node_record_error(self, node):
        """Test record_error method."""
        assert node.error_count == 0

        node.record_error("Test error")

        assert node.error_count == 1
        assert node.last_error == "Test error"

    def test_node_to_dict(self, node):
        """Test to_dict method."""
        node.state = NodeState.READY
        node.ip_address = "192.168.1.100"

        result = node.to_dict()

        assert result["node_id"] == "test_node"
        assert result["state"] == "ready"
        assert result["ip_address"] == "192.168.1.100"
        assert "voice_input" in result["capabilities"]


class TestHubConfig:
    """Tests for HubConfig."""

    def test_hub_config_defaults(self):
        """Test HubConfig default values."""
        config = HubConfig()

        assert config.bind_address == "0.0.0.0"
        assert config.port == 8420
        assert config.discovery_port == 8421
        assert config.require_auth is True
        assert config.heartbeat_interval_seconds == 30
        assert config.tls_enabled is True

    def test_hub_config_custom(self):
        """Test HubConfig with custom values."""
        config = HubConfig(
            port=9000,
            max_nodes=50,
            heartbeat_interval_seconds=60,
        )

        assert config.port == 9000
        assert config.max_nodes == 50
        assert config.heartbeat_interval_seconds == 60


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_generate_node_id_random(self):
        """Test generating random node ID."""
        node_id = generate_node_id(NodeType.ROOM)

        assert node_id.startswith("roo_")
        assert len(node_id) == 20  # "roo_" + 16 hex chars

    def test_generate_node_id_deterministic(self):
        """Test generating deterministic node ID."""
        node_id1 = generate_node_id(NodeType.ROOM, "hardware123")
        node_id2 = generate_node_id(NodeType.ROOM, "hardware123")

        assert node_id1 == node_id2
        assert len(node_id1) == 16

    def test_generate_auth_token(self):
        """Test generating auth token."""
        token = generate_auth_token()

        assert len(token) == 43  # Base64 URL-safe encoding of 32 bytes
        assert token.isalnum() or "-" in token or "_" in token


# ============================================================================
# Hub Coordinator Tests
# ============================================================================


class TestHubCoordinator:
    """Tests for HubCoordinator."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return HubConfig(
            require_auth=True,
            max_nodes=5,
            heartbeat_interval_seconds=5,
        )

    @pytest.fixture
    def coordinator(self, config):
        """Create test coordinator."""
        return HubCoordinator(config=config)

    @pytest.mark.asyncio
    async def test_start_stop(self, coordinator):
        """Test starting and stopping coordinator."""
        assert coordinator.state == HubState.STOPPED

        await coordinator.start()
        assert coordinator.state == HubState.RUNNING
        assert coordinator.is_running

        await coordinator.stop()
        assert coordinator.state == HubState.STOPPED

    @pytest.mark.asyncio
    async def test_register_node(self, coordinator):
        """Test registering a node."""
        await coordinator.start()

        registration = NodeRegistration(
            node_type=NodeType.ROOM,
            name="Living Room",
            capabilities=[NodeCapability.VOICE_INPUT],
        )

        node_id, auth_token = await coordinator.register_node(
            registration=registration,
            ip_address="192.168.1.100",
            port=8000,
        )

        assert node_id is not None
        assert auth_token is not None

        node = coordinator.get_node(node_id)
        assert node is not None
        assert node.info.name == "Living Room"

        await coordinator.stop()

    @pytest.mark.asyncio
    async def test_register_node_max_limit(self, coordinator):
        """Test node registration respects max limit."""
        await coordinator.start()

        # Register up to max
        for i in range(5):
            reg = NodeRegistration(
                node_type=NodeType.ROOM,
                name=f"Room {i}",
                capabilities=[],
                hardware_id=f"hw_{i}",
            )
            await coordinator.register_node(reg, f"192.168.1.{i}", 8000)

        # Should fail on 6th
        reg = NodeRegistration(
            node_type=NodeType.ROOM,
            name="One Too Many",
            capabilities=[],
        )

        with pytest.raises(ValueError, match="Maximum node limit"):
            await coordinator.register_node(reg, "192.168.1.100", 8000)

        await coordinator.stop()

    @pytest.mark.asyncio
    async def test_authenticate_node(self, coordinator):
        """Test node authentication."""
        await coordinator.start()

        registration = NodeRegistration(
            node_type=NodeType.ROOM,
            name="Test Room",
            capabilities=[],
        )
        node_id, auth_token = await coordinator.register_node(
            registration, "192.168.1.100", 8000
        )

        # Correct token should work
        result = await coordinator.authenticate_node(node_id, auth_token)
        assert result is True

        node = coordinator.get_node(node_id)
        assert node.state == NodeState.READY

        # Wrong token should fail
        result = await coordinator.authenticate_node(node_id, "wrong_token")
        assert result is False

        await coordinator.stop()

    @pytest.mark.asyncio
    async def test_handle_heartbeat(self, coordinator):
        """Test heartbeat handling."""
        await coordinator.start()

        registration = NodeRegistration(
            node_type=NodeType.ROOM,
            name="Test",
            capabilities=[],
        )
        node_id, token = await coordinator.register_node(registration, "192.168.1.1", 8000)
        await coordinator.authenticate_node(node_id, token)

        result = await coordinator.handle_heartbeat(node_id, latency_ms=25.0)
        assert result is True

        node = coordinator.get_node(node_id)
        assert node.latency_ms == 25.0
        assert node.last_heartbeat is not None

        await coordinator.stop()

    @pytest.mark.asyncio
    async def test_route_request(self, coordinator):
        """Test request routing."""
        await coordinator.start()

        # Register and authenticate a node
        registration = NodeRegistration(
            node_type=NodeType.ROOM,
            name="Test",
            capabilities=[],
        )
        node_id, token = await coordinator.register_node(registration, "192.168.1.1", 8000)
        await coordinator.authenticate_node(node_id, token)

        # Create and route a request
        request = Request(
            request_id="test_req",
            source_node_id=node_id,
            request_type="status",
            payload={},
        )

        response = await coordinator.route_request(request)
        assert response.success is True
        assert response.request_id == "test_req"

        await coordinator.stop()

    @pytest.mark.asyncio
    async def test_route_request_unknown_type(self, coordinator):
        """Test routing request with unknown type."""
        await coordinator.start()

        registration = NodeRegistration(
            node_type=NodeType.ROOM,
            name="Test",
            capabilities=[],
        )
        node_id, token = await coordinator.register_node(registration, "192.168.1.1", 8000)
        await coordinator.authenticate_node(node_id, token)

        request = Request(
            request_id="test_req",
            source_node_id=node_id,
            request_type="unknown_type",
            payload={},
        )

        response = await coordinator.route_request(request)
        assert response.success is False
        assert "Unknown request type" in response.error

        await coordinator.stop()

    @pytest.mark.asyncio
    async def test_get_nodes(self, coordinator):
        """Test getting nodes with filters."""
        await coordinator.start()

        # Register different node types
        for i, node_type in enumerate([NodeType.ROOM, NodeType.MOBILE, NodeType.ROOM]):
            reg = NodeRegistration(
                node_type=node_type,
                name=f"Node {i}",
                capabilities=[],
                hardware_id=f"hw_{i}",
            )
            await coordinator.register_node(reg, f"192.168.1.{i}", 8000)

        all_nodes = coordinator.get_nodes()
        assert len(all_nodes) == 3

        room_nodes = coordinator.get_nodes(node_type=NodeType.ROOM)
        assert len(room_nodes) == 2

        mobile_nodes = coordinator.get_nodes(node_type=NodeType.MOBILE)
        assert len(mobile_nodes) == 1

        await coordinator.stop()

    @pytest.mark.asyncio
    async def test_get_status(self, coordinator):
        """Test getting hub status."""
        await coordinator.start()

        status = coordinator.get_status()

        assert status["state"] == "running"
        assert "nodes" in status
        assert "stats" in status
        assert status["nodes"]["total"] == 0

        await coordinator.stop()

    @pytest.mark.asyncio
    async def test_event_callbacks(self, coordinator):
        """Test event callbacks."""
        events = []

        def callback(event: HubEvent):
            events.append(event)

        coordinator.on_event(callback)

        await coordinator.start()
        assert len(events) == 1
        assert events[0].event_type == "hub_started"

        await coordinator.stop()
        assert len(events) == 2
        assert events[1].event_type == "hub_stopped"


# ============================================================================
# Context Sync Tests
# ============================================================================


class TestSyncEntry:
    """Tests for SyncEntry."""

    def test_sync_entry_from_dict(self):
        """Test creating SyncEntry from dict."""
        data = {
            "entry_id": "entry_1",
            "content": "Test content",
            "entry_type": "user_input",
            "timestamp": "2024-01-01T12:00:00",
            "version": 1,
        }

        entry = SyncEntry.from_dict(data)

        assert entry.entry_id == "entry_1"
        assert entry.content == "Test content"
        assert entry.version == 1

    def test_sync_entry_to_dict(self):
        """Test converting SyncEntry to dict."""
        entry = SyncEntry(
            entry_id="entry_1",
            content="Test",
            entry_type="user_input",
            timestamp=datetime.now(),
            version=1,
            checksum="abc123",
        )

        result = entry.to_dict()

        assert result["entry_id"] == "entry_1"
        assert result["content"] == "Test"
        assert result["version"] == 1


class TestContextSync:
    """Tests for ContextSync."""

    @pytest.fixture
    def context_sync(self):
        """Create test context sync."""
        return ContextSync(
            sync_interval=5,
            batch_size=10,
            strategy=SyncStrategy.LAST_WRITE_WINS,
        )

    def test_register_node(self, context_sync):
        """Test registering a node for sync."""
        context_sync.register_node("node_1")

        state = context_sync.get_node_sync_state("node_1")
        assert state is not None
        assert state.node_id == "node_1"
        assert state.last_sync_time is None

    def test_unregister_node(self, context_sync):
        """Test unregistering a node."""
        context_sync.register_node("node_1")
        context_sync.unregister_node("node_1")

        state = context_sync.get_node_sync_state("node_1")
        assert state is not None  # Still exists but disconnected

    @pytest.mark.asyncio
    async def test_sync_from_node(self, context_sync):
        """Test syncing context from a node."""
        context_sync.register_node("node_1")

        context_data = {
            "entries": [
                {
                    "entry_id": "e1",
                    "content": "Hello",
                    "entry_type": "user_input",
                    "timestamp": datetime.now().isoformat(),
                    "version": 1,
                }
            ],
            "version": 1,
        }

        result = await context_sync.sync_from_node("node_1", context_data)

        assert result["success"] is True
        assert result["merged_count"] == 1

        state = context_sync.get_node_sync_state("node_1")
        assert state.last_sync_time is not None

    @pytest.mark.asyncio
    async def test_initiate_handoff(self, context_sync):
        """Test context handoff between nodes."""
        context_sync.register_node("source_node")
        context_sync.register_node("target_node")

        result = await context_sync.initiate_handoff("source_node", "target_node")

        assert result["success"] is True
        assert result["source_node"] == "source_node"
        assert result["target_node"] == "target_node"

    def test_buffer_for_offline_node(self, context_sync):
        """Test buffering data for offline node."""
        entry = SyncEntry(
            entry_id="e1",
            content="Test",
            entry_type="user_input",
            timestamp=datetime.now(),
            version=1,
            checksum="abc",
        )

        result = context_sync.buffer_for_offline_node("offline_node", entry)
        assert result is True

        state = context_sync.get_node_sync_state("offline_node")
        assert len(state.offline_buffer) == 1

    @pytest.mark.asyncio
    async def test_recover_offline_data(self, context_sync):
        """Test recovering offline data."""
        context_sync.register_node("node_1")

        # Buffer some data
        for i in range(3):
            entry = SyncEntry(
                entry_id=f"e{i}",
                content=f"Test {i}",
                entry_type="user_input",
                timestamp=datetime.now(),
                version=1,
                checksum=f"abc{i}",
            )
            context_sync.buffer_for_offline_node("node_1", entry)

        recovered = await context_sync.recover_offline_data("node_1")
        assert recovered == 3

        state = context_sync.get_node_sync_state("node_1")
        assert len(state.offline_buffer) == 0
        assert len(state.pending_entries) == 3

    def test_get_stats(self, context_sync):
        """Test getting sync statistics."""
        context_sync.register_node("node_1")
        context_sync.register_node("node_2")

        stats = context_sync.get_stats()

        assert stats["registered_nodes"] == 2
        assert stats["hub_version"] == 0
        assert "node_states" in stats


# ============================================================================
# Request Router Tests
# ============================================================================


class TestRoutedRequest:
    """Tests for RoutedRequest."""

    def test_request_creation(self):
        """Test creating a routed request."""
        request = RoutedRequest(
            request_id="req_1",
            request_type="query",
            source_node_id="node_1",
            payload={"text": "Hello"},
        )

        assert request.request_id == "req_1"
        assert request.priority == RequestPriority.NORMAL
        assert request.retries == 0

    def test_request_is_expired(self):
        """Test checking if request is expired."""
        request = RoutedRequest(
            request_id="req_1",
            request_type="query",
            source_node_id="node_1",
            payload={},
            timeout_seconds=0.001,  # Very short timeout
        )

        import time
        time.sleep(0.01)

        assert request.is_expired() is True


class TestRequestRouter:
    """Tests for RequestRouter."""

    @pytest.fixture
    def router(self):
        """Create test router."""
        return RequestRouter(
            strategy=RoutingStrategy.DIRECT,
            max_queue_size=10,
            worker_count=2,
        )

    @pytest.mark.asyncio
    async def test_start_stop(self, router):
        """Test starting and stopping router."""
        assert not router.is_running

        await router.start()
        assert router.is_running

        await router.stop()
        assert not router.is_running

    @pytest.mark.asyncio
    async def test_register_route(self, router):
        """Test registering a route."""
        async def handler(request: RoutedRequest) -> dict:
            return {"result": "success"}

        router.register_route("test_type", handler)

        stats = router.get_stats()
        assert "test_type" in stats["registered_routes"]

    @pytest.mark.asyncio
    async def test_route_direct(self, router):
        """Test direct routing."""
        async def handler(request: RoutedRequest) -> dict:
            return {"received": request.payload}

        router.register_route("test_type", handler)
        await router.start()

        result = await router.route(
            request_id="req_1",
            request_type="test_type",
            source_node_id="node_1",
            payload={"data": "test"},
        )

        assert result["received"]["data"] == "test"
        await router.stop()

    @pytest.mark.asyncio
    async def test_route_unknown_type(self, router):
        """Test routing unknown request type."""
        await router.start()

        with pytest.raises(ValueError, match="No route"):
            await router.route(
                request_id="req_1",
                request_type="unknown",
                source_node_id="node_1",
                payload={},
            )

        await router.stop()

    @pytest.mark.asyncio
    async def test_route_with_rate_limit(self, router):
        """Test routing with rate limiting."""
        async def handler(request: RoutedRequest) -> dict:
            return {"ok": True}

        router.register_route("limited", handler, rate_limit=1)
        await router.start()

        # First request should succeed
        result = await router.route(
            request_id="req_1",
            request_type="limited",
            source_node_id="node_1",
            payload={},
        )
        assert result["ok"] is True

        # Second immediate request should be rate limited
        with pytest.raises(RateLimitError):
            await router.route(
                request_id="req_2",
                request_type="limited",
                source_node_id="node_1",
                payload={},
            )

        await router.stop()

    @pytest.mark.asyncio
    async def test_route_priority_queue(self):
        """Test priority queue routing."""
        router = RequestRouter(strategy=RoutingStrategy.PRIORITY)

        async def handler(request: RoutedRequest) -> dict:
            return {"priority": request.priority.value}

        router.register_route("test", handler)
        await router.start()

        result = await router.route(
            request_id="req_1",
            request_type="test",
            source_node_id="node_1",
            payload={},
            priority=RequestPriority.HIGH,
        )

        assert result["priority"] == RequestPriority.HIGH.value
        await router.stop()

    def test_get_queue_depth(self, router):
        """Test getting queue depth."""
        depth = router.get_queue_depth()
        assert depth == 0

        depth = router.get_queue_depth(RequestPriority.CRITICAL)
        assert depth == 0

    def test_get_stats(self, router):
        """Test getting router statistics."""
        stats = router.get_stats()

        assert stats["total_routed"] == 0
        assert stats["strategy"] == "direct"
        assert "queue_sizes" in stats

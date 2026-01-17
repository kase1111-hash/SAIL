"""
SAIL Node Integration Tests

Integration tests for cross-node communication, hub coordination,
and context synchronization.
"""

import asyncio
from datetime import datetime

import pytest
from src.deployment.health import (
    HealthMonitor,
)
from src.hub.base import (
    HubConfig,
    NodeCapability,
    NodeState,
    NodeType,
)
from src.hub.context_sync import (
    ContextSync,
    SyncEntry,
    SyncStrategy,
)
from src.hub.coordinator import (
    HubCoordinator,
    NodeRegistration,
    Request,
)
from src.hub.router import (
    RequestPriority,
    RequestRouter,
    RoutingStrategy,
)

# ============================================================================
# Hub and Node Integration Tests
# ============================================================================


class TestHubNodeIntegration:
    """Integration tests for hub and node communication."""

    @pytest.fixture
    def hub_config(self):
        """Create hub config for testing."""
        return HubConfig(
            port=8420,
            max_nodes=10,
            heartbeat_interval_seconds=1,
            heartbeat_timeout_seconds=5,
        )

    @pytest.fixture
    async def coordinator(self, hub_config):
        """Create and start a coordinator."""
        coord = HubCoordinator(config=hub_config)
        await coord.start()
        yield coord
        await coord.stop()

    @pytest.mark.asyncio
    async def test_multiple_node_registration(self, coordinator):
        """Test registering multiple nodes of different types."""
        # Register room node
        room_reg = NodeRegistration(
            node_type=NodeType.ROOM,
            name="Living Room",
            capabilities=[NodeCapability.VOICE_INPUT, NodeCapability.VOICE_OUTPUT],
        )
        _room_id, _room_token = await coordinator.register_node(
            room_reg, "192.168.1.10", 8000
        )

        # Register mobile node
        mobile_reg = NodeRegistration(
            node_type=NodeType.MOBILE,
            name="John's Phone",
            capabilities=[NodeCapability.GPS, NodeCapability.PUSH_NOTIFICATION],
        )
        _mobile_id, _mobile_token = await coordinator.register_node(
            mobile_reg, "192.168.1.20", 8000
        )

        # Register vehicle node
        vehicle_reg = NodeRegistration(
            node_type=NodeType.VEHICLE,
            name="Family Car",
            capabilities=[NodeCapability.OBD, NodeCapability.VOICE_INPUT],
        )
        _vehicle_id, _vehicle_token = await coordinator.register_node(
            vehicle_reg, "192.168.1.30", 8000
        )

        # Verify all registered
        assert coordinator.node_count == 3

        room_nodes = coordinator.get_nodes(node_type=NodeType.ROOM)
        assert len(room_nodes) == 1

        mobile_nodes = coordinator.get_nodes(node_type=NodeType.MOBILE)
        assert len(mobile_nodes) == 1

        vehicle_nodes = coordinator.get_nodes(node_type=NodeType.VEHICLE)
        assert len(vehicle_nodes) == 1

    @pytest.mark.asyncio
    async def test_node_authentication_flow(self, coordinator):
        """Test full node authentication flow."""
        # Register
        registration = NodeRegistration(
            node_type=NodeType.ROOM,
            name="Test Node",
            capabilities=[],
        )
        node_id, auth_token = await coordinator.register_node(
            registration, "192.168.1.100", 8000
        )

        # Verify initial state
        node = coordinator.get_node(node_id)
        assert node.state == NodeState.AUTHENTICATED

        # Authenticate
        result = await coordinator.authenticate_node(node_id, auth_token)
        assert result is True

        # Verify ready state
        node = coordinator.get_node(node_id)
        assert node.state == NodeState.READY

    @pytest.mark.asyncio
    async def test_request_routing_between_nodes(self, coordinator):
        """Test routing requests between nodes."""
        # Register two nodes
        reg1 = NodeRegistration(
            node_type=NodeType.ROOM,
            name="Node 1",
            capabilities=[],
            hardware_id="hw1",
        )
        node1_id, token1 = await coordinator.register_node(reg1, "192.168.1.10", 8000)
        await coordinator.authenticate_node(node1_id, token1)

        reg2 = NodeRegistration(
            node_type=NodeType.MOBILE,
            name="Node 2",
            capabilities=[],
            hardware_id="hw2",
        )
        node2_id, token2 = await coordinator.register_node(reg2, "192.168.1.20", 8000)
        await coordinator.authenticate_node(node2_id, token2)

        # Send request from node 1
        request = Request(
            request_id="req_1",
            source_node_id=node1_id,
            request_type="status",
            payload={},
        )

        response = await coordinator.route_request(request)
        assert response.success is True

    @pytest.mark.asyncio
    async def test_heartbeat_monitoring(self, coordinator):
        """Test heartbeat monitoring across nodes."""
        registration = NodeRegistration(
            node_type=NodeType.ROOM,
            name="Test",
            capabilities=[],
        )
        node_id, token = await coordinator.register_node(
            registration, "192.168.1.100", 8000
        )
        await coordinator.authenticate_node(node_id, token)

        # Send heartbeats
        for i in range(3):
            result = await coordinator.handle_heartbeat(node_id, latency_ms=10.0 + i)
            assert result is True

        node = coordinator.get_node(node_id)
        assert node.latency_ms == 12.0  # Last latency
        assert node.missed_heartbeats == 0


# ============================================================================
# Context Synchronization Integration Tests
# ============================================================================


class TestContextSyncIntegration:
    """Integration tests for context synchronization."""

    @pytest.fixture
    def context_sync(self):
        """Create context sync instance."""
        return ContextSync(
            sync_interval=1,
            batch_size=50,
            strategy=SyncStrategy.LAST_WRITE_WINS,
        )

    @pytest.mark.asyncio
    async def test_multi_node_sync(self, context_sync):
        """Test synchronizing context across multiple nodes."""
        # Register multiple nodes
        context_sync.register_node("room_node")
        context_sync.register_node("mobile_node")
        context_sync.register_node("vehicle_node")

        # Sync from room node
        room_data = {
            "entries": [
                {
                    "entry_id": "e1",
                    "content": "User said hello in living room",
                    "entry_type": "user_input",
                    "timestamp": datetime.now().isoformat(),
                    "version": 1,
                }
            ],
            "version": 1,
        }

        result = await context_sync.sync_from_node("room_node", room_data)
        assert result["success"] is True
        assert result["merged_count"] == 1

        # Sync from mobile node
        mobile_data = {
            "entries": [
                {
                    "entry_id": "e2",
                    "content": "User is at the store",
                    "entry_type": "sensor_data",
                    "timestamp": datetime.now().isoformat(),
                    "version": 1,
                }
            ],
            "version": 1,
        }

        result = await context_sync.sync_from_node("mobile_node", mobile_data)
        assert result["success"] is True

        # Verify hub version increased
        assert context_sync.hub_version >= 2

    @pytest.mark.asyncio
    async def test_context_handoff_flow(self, context_sync):
        """Test context handoff between nodes."""
        context_sync.register_node("home_node")
        context_sync.register_node("car_node")

        # Simulate user at home with some context
        home_data = {
            "entries": [
                {
                    "entry_id": "home_1",
                    "content": "Reminder to get groceries",
                    "entry_type": "assistant_response",
                    "timestamp": datetime.now().isoformat(),
                    "version": 1,
                },
                {
                    "entry_id": "home_2",
                    "content": "What should I buy at the store?",
                    "entry_type": "user_input",
                    "timestamp": datetime.now().isoformat(),
                    "version": 1,
                },
            ],
            "version": 1,
        }

        await context_sync.sync_from_node("home_node", home_data)

        # Initiate handoff to car
        handoff_result = await context_sync.initiate_handoff("home_node", "car_node")

        assert handoff_result["success"] is True
        assert handoff_result["source_node"] == "home_node"
        assert handoff_result["target_node"] == "car_node"

    @pytest.mark.asyncio
    async def test_offline_buffer_recovery(self, context_sync):
        """Test recovery of offline buffered data."""
        context_sync.register_node("mobile_node")

        # Buffer data for offline node
        for i in range(5):
            entry = SyncEntry(
                entry_id=f"offline_{i}",
                content=f"Offline message {i}",
                entry_type="user_input",
                timestamp=datetime.now(),
                version=1,
                checksum=f"chk_{i}",
            )
            context_sync.buffer_for_offline_node("mobile_node", entry)

        # Verify buffer
        state = context_sync.get_node_sync_state("mobile_node")
        assert len(state.offline_buffer) == 5

        # Recover
        recovered = await context_sync.recover_offline_data("mobile_node")
        assert recovered == 5

        # Verify moved to pending
        state = context_sync.get_node_sync_state("mobile_node")
        assert len(state.offline_buffer) == 0
        assert len(state.pending_entries) == 5


# ============================================================================
# Request Router Integration Tests
# ============================================================================


class TestRouterIntegration:
    """Integration tests for request routing."""

    @pytest.fixture
    async def router(self):
        """Create and start a router."""
        router = RequestRouter(
            strategy=RoutingStrategy.PRIORITY,
            max_queue_size=100,
            worker_count=4,
        )
        await router.start()
        yield router
        await router.stop()

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, router):
        """Test handling concurrent requests."""
        results = []

        async def handler(request):
            await asyncio.sleep(0.01)
            return {"request_id": request.request_id}

        router.register_route("test", handler)

        # Send multiple concurrent requests
        tasks = []
        for i in range(10):
            tasks.append(
                router.route(
                    request_id=f"req_{i}",
                    request_type="test",
                    source_node_id="node_1",
                    payload={},
                )
            )

        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert all(r["request_id"].startswith("req_") for r in results)

    @pytest.mark.asyncio
    async def test_priority_ordering(self, router):
        """Test that priority affects processing order."""
        processed = []

        async def handler(request):
            processed.append((request.request_id, request.priority))
            return {}

        router.register_route("test", handler)

        # Queue requests with different priorities
        await router.route(
            request_id="low",
            request_type="test",
            source_node_id="node_1",
            payload={},
            priority=RequestPriority.LOW,
        )

        await router.route(
            request_id="critical",
            request_type="test",
            source_node_id="node_1",
            payload={},
            priority=RequestPriority.CRITICAL,
        )

        await router.route(
            request_id="normal",
            request_type="test",
            source_node_id="node_1",
            payload={},
            priority=RequestPriority.NORMAL,
        )

        # Wait for processing
        await asyncio.sleep(0.1)

        assert len(processed) == 3


# ============================================================================
# Health Monitoring Integration Tests
# ============================================================================


class TestHealthMonitorIntegration:
    """Integration tests for health monitoring."""

    @pytest.fixture
    async def monitor(self):
        """Create and start a health monitor."""
        monitor = HealthMonitor()
        await monitor.start()
        yield monitor
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_multi_node_health_tracking(self, monitor):
        """Test tracking health of multiple nodes."""
        # Register multiple nodes
        monitor.register_node("room_1", NodeType.ROOM, "Living Room")
        monitor.register_node("room_2", NodeType.ROOM, "Bedroom")
        monitor.register_node("mobile_1", NodeType.MOBILE, "Phone")

        # Update health for each
        monitor.update_node_health(
            "room_1",
            connection_state=NodeState.READY,
            latency_ms=30.0,
            heartbeat=True,
        )

        monitor.update_node_health(
            "room_2",
            connection_state=NodeState.READY,
            latency_ms=250.0,  # High latency
            heartbeat=True,
        )

        monitor.update_node_health(
            "mobile_1",
            connection_state=NodeState.DISCONNECTED,
        )

        # Check system health
        system = monitor.get_system_health()

        assert system.total_nodes == 3
        assert system.healthy_nodes >= 1  # room_1 should be healthy
        assert system.unhealthy_nodes >= 1  # mobile_1 is disconnected

    @pytest.mark.asyncio
    async def test_alert_generation_and_resolution(self, monitor):
        """Test alert generation and resolution flow."""
        monitor.register_node("node_1", NodeType.ROOM, "Test")

        # Record unhealthy metric
        monitor.record_metric(
            node_id="node_1",
            name="latency",
            value=600.0,  # Very high
            unit="ms",
            max_healthy=200.0,
        )

        # Should generate alert
        alerts = monitor.get_active_alerts()
        assert len(alerts) >= 1

        # Acknowledge alert
        alert_id = alerts[0].alert_id
        monitor.acknowledge_alert(alert_id)

        # Resolve alert
        monitor.resolve_alert(alert_id)

        # Should be no active alerts
        active = monitor.get_active_alerts()
        assert len(active) == 0

    @pytest.mark.asyncio
    async def test_dashboard_data_consistency(self, monitor):
        """Test dashboard data is consistent."""
        # Set up some nodes
        for i in range(3):
            monitor.register_node(f"node_{i}", NodeType.ROOM, f"Room {i}")
            monitor.update_node_health(
                f"node_{i}",
                connection_state=NodeState.READY,
                latency_ms=50.0,
                heartbeat=True,
            )

        # Get dashboard data
        data = monitor.get_dashboard_data()

        # Verify structure
        assert "system" in data
        assert "nodes" in data
        assert "alerts" in data
        assert "stats" in data

        # Verify counts match
        assert len(data["nodes"]) == 3
        assert data["system"]["total_nodes"] == 3


# ============================================================================
# Full System Integration Tests
# ============================================================================


class TestFullSystemIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_complete_node_lifecycle(self):
        """Test complete node lifecycle from registration to disconnection."""
        # Create hub
        config = HubConfig(max_nodes=5)
        coordinator = HubCoordinator(config=config)
        await coordinator.start()

        try:
            # 1. Register node
            registration = NodeRegistration(
                node_type=NodeType.ROOM,
                name="Test Room",
                capabilities=[NodeCapability.VOICE_INPUT],
            )
            node_id, token = await coordinator.register_node(
                registration, "192.168.1.100", 8000
            )
            assert node_id is not None

            # 2. Authenticate
            result = await coordinator.authenticate_node(node_id, token)
            assert result is True

            node = coordinator.get_node(node_id)
            assert node.state == NodeState.READY

            # 3. Send heartbeats
            for _ in range(3):
                await coordinator.handle_heartbeat(node_id, latency_ms=25.0)
                await asyncio.sleep(0.1)

            # 4. Route requests
            request = Request(
                request_id="req_1",
                source_node_id=node_id,
                request_type="status",
                payload={},
            )
            response = await coordinator.route_request(request)
            assert response.success is True

            # 5. Verify status
            status = coordinator.get_status()
            assert status["nodes"]["total"] == 1
            assert status["nodes"]["ready"] == 1

        finally:
            await coordinator.stop()

    @pytest.mark.asyncio
    async def test_multi_node_coordination(self):
        """Test coordination between multiple node types."""
        coordinator = HubCoordinator()
        context_sync = ContextSync()
        health_monitor = HealthMonitor()

        await coordinator.start()
        await health_monitor.start()

        try:
            # Register different node types
            nodes = []
            for node_type, name in [
                (NodeType.ROOM, "Living Room"),
                (NodeType.MOBILE, "Phone"),
                (NodeType.VEHICLE, "Car"),
            ]:
                reg = NodeRegistration(
                    node_type=node_type,
                    name=name,
                    capabilities=[],
                )
                node_id, token = await coordinator.register_node(
                    reg, f"192.168.1.{len(nodes) + 10}", 8000
                )
                await coordinator.authenticate_node(node_id, token)
                nodes.append(node_id)

                # Also register with health monitor
                health_monitor.register_node(node_id, node_type, name)
                context_sync.register_node(node_id)

            # Verify all nodes registered
            assert coordinator.node_count == 3
            assert health_monitor.get_system_health().total_nodes == 3

            # Simulate activity
            for node_id in nodes:
                await coordinator.handle_heartbeat(node_id, latency_ms=30.0)
                health_monitor.update_node_health(
                    node_id,
                    connection_state=NodeState.READY,
                    heartbeat=True,
                )

            # Verify health
            system_health = health_monitor.get_system_health()
            assert system_health.healthy_nodes >= 2

        finally:
            await health_monitor.stop()
            await coordinator.stop()

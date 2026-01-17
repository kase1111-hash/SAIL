"""
SAIL Deployment Module Tests

Comprehensive tests for node provisioning, OTA updates, and health monitoring.
"""


import pytest
from src.deployment.health import (
    AlertSeverity,
    HealthAlert,
    HealthConfig,
    HealthMetric,
    HealthMonitor,
    HealthStatus,
    NodeHealth,
)
from src.deployment.ota import (
    FirmwareUpdate,
    FirmwareVersion,
    OTAConfig,
    OTAManager,
    UpdateChannel,
    UpdatePriority,
)
from src.deployment.provisioning import (
    NodeProvisioner,
    NodeTemplate,
    ProvisioningConfig,
    ProvisioningResult,
    ProvisioningState,
)
from src.hub.base import NodeState, NodeType

# ============================================================================
# Provisioning Tests
# ============================================================================


class TestProvisioningConfig:
    """Tests for ProvisioningConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ProvisioningConfig()

        assert config.discovery_timeout_seconds == 30
        assert config.max_retries == 3
        assert config.config_encryption_enabled is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ProvisioningConfig(
            discovery_timeout_seconds=60,
            max_retries=5,
        )

        assert config.discovery_timeout_seconds == 60
        assert config.max_retries == 5


class TestProvisioningResult:
    """Tests for ProvisioningResult."""

    def test_result_creation(self):
        """Test creating a provisioning result."""
        result = ProvisioningResult(
            node_id="node_1",
            node_type=NodeType.ROOM,
            success=True,
            state=ProvisioningState.COMPLETED,
            message="Success",
        )

        assert result.node_id == "node_1"
        assert result.success is True
        assert result.state == ProvisioningState.COMPLETED

    def test_result_to_dict(self):
        """Test converting result to dict."""
        result = ProvisioningResult(
            node_id="node_1",
            node_type=NodeType.ROOM,
            success=True,
            state=ProvisioningState.COMPLETED,
        )

        data = result.to_dict()

        assert data["node_id"] == "node_1"
        assert data["node_type"] == "room"
        assert data["success"] is True


class TestNodeProvisioner:
    """Tests for NodeProvisioner."""

    @pytest.fixture
    def provisioner(self):
        """Create test provisioner."""
        return NodeProvisioner()

    def test_default_templates(self, provisioner):
        """Test default templates are loaded."""
        room_template = provisioner.get_template(NodeType.ROOM)
        mobile_template = provisioner.get_template(NodeType.MOBILE)
        vehicle_template = provisioner.get_template(NodeType.VEHICLE)

        assert room_template is not None
        assert mobile_template is not None
        assert vehicle_template is not None

    def test_get_template(self, provisioner):
        """Test getting a template."""
        template = provisioner.get_template(NodeType.ROOM)

        assert template.node_type == NodeType.ROOM
        assert "name" in template.required_params
        assert "location" in template.required_params

    def test_add_template(self, provisioner):
        """Test adding a custom template."""
        template = NodeTemplate(
            node_type=NodeType.ROOM,
            template_name="custom_room",
            base_config={"custom": True},
            required_params=["name"],
        )

        provisioner.add_template(template)

        result = provisioner.get_template(NodeType.ROOM)
        assert result.template_name == "custom_room"

    @pytest.mark.asyncio
    async def test_provision_node_missing_params(self, provisioner):
        """Test provisioning fails with missing params."""
        # Room node requires 'location' parameter
        result = await provisioner.provision_node(
            node_type=NodeType.ROOM,
            name="Test Room",
            params={},  # Missing location
            ip_address="192.168.1.100",
        )

        assert result.success is False
        assert "Missing required parameter" in result.message

    def test_get_stats(self, provisioner):
        """Test getting provisioner statistics."""
        stats = provisioner.get_stats()

        assert stats["total_provisioned"] == 0
        assert stats["successful"] == 0
        assert stats["failed"] == 0

    def test_get_provision_history(self, provisioner):
        """Test getting provision history."""
        history = provisioner.get_provision_history(limit=5)

        assert isinstance(history, list)


# ============================================================================
# OTA Update Tests
# ============================================================================


class TestFirmwareVersion:
    """Tests for FirmwareVersion."""

    def test_version_creation(self):
        """Test creating a firmware version."""
        version = FirmwareVersion(major=1, minor=2, patch=3)

        assert str(version) == "1.2.3"

    def test_version_with_build(self):
        """Test version with build number."""
        version = FirmwareVersion(major=1, minor=0, patch=0, build="beta1")

        assert str(version) == "1.0.0+beta1"

    def test_version_from_string(self):
        """Test parsing version from string."""
        version = FirmwareVersion.from_string("2.1.0+rc1")

        assert version.major == 2
        assert version.minor == 1
        assert version.patch == 0
        assert version.build == "rc1"

    def test_version_comparison(self):
        """Test version comparison."""
        v1 = FirmwareVersion(major=1, minor=0, patch=0)
        v2 = FirmwareVersion(major=1, minor=0, patch=1)
        v3 = FirmwareVersion(major=1, minor=1, patch=0)
        v4 = FirmwareVersion(major=2, minor=0, patch=0)

        assert v2 > v1
        assert v3 > v2
        assert v4 > v3


class TestFirmwareUpdate:
    """Tests for FirmwareUpdate."""

    def test_update_creation(self):
        """Test creating a firmware update."""
        update = FirmwareUpdate(
            update_id="update_1",
            node_type=NodeType.ROOM,
            version=FirmwareVersion(1, 1, 0),
            channel=UpdateChannel.STABLE,
            priority=UpdatePriority.NORMAL,
            release_notes="Bug fixes",
        )

        assert update.update_id == "update_1"
        assert update.channel == UpdateChannel.STABLE

    def test_update_to_dict(self):
        """Test converting update to dict."""
        update = FirmwareUpdate(
            update_id="update_1",
            node_type=NodeType.ROOM,
            version=FirmwareVersion(1, 0, 0),
            channel=UpdateChannel.STABLE,
            priority=UpdatePriority.HIGH,
            release_notes="Important update",
        )

        data = update.to_dict()

        assert data["update_id"] == "update_1"
        assert data["version"] == "1.0.0"
        assert data["channel"] == "stable"


class TestOTAConfig:
    """Tests for OTAConfig."""

    def test_default_config(self):
        """Test default OTA configuration."""
        config = OTAConfig()

        assert config.auto_update_enabled is False
        assert config.preferred_channel == UpdateChannel.STABLE
        assert config.backup_before_update is True

    def test_custom_config(self):
        """Test custom OTA configuration."""
        config = OTAConfig(
            auto_update_enabled=True,
            preferred_channel=UpdateChannel.BETA,
        )

        assert config.auto_update_enabled is True
        assert config.preferred_channel == UpdateChannel.BETA


class TestOTAManager:
    """Tests for OTAManager."""

    @pytest.fixture
    def ota_manager(self, tmp_path):
        """Create test OTA manager."""
        config = OTAConfig(firmware_path=tmp_path / "firmware")
        return OTAManager(config)

    @pytest.mark.asyncio
    async def test_start_stop(self, ota_manager):
        """Test starting and stopping OTA manager."""
        await ota_manager.start()
        await ota_manager.stop()
        # Should not raise

    def test_register_update(self, ota_manager):
        """Test registering an update."""
        update = FirmwareUpdate(
            update_id="update_1",
            node_type=NodeType.ROOM,
            version=FirmwareVersion(1, 1, 0),
            channel=UpdateChannel.STABLE,
            priority=UpdatePriority.NORMAL,
            release_notes="Test update",
        )

        ota_manager.register_update(update)

        stats = ota_manager.get_stats()
        assert stats["available_updates"]["room"] == 1

    @pytest.mark.asyncio
    async def test_check_for_updates(self, ota_manager):
        """Test checking for updates."""
        # Register an update
        update = FirmwareUpdate(
            update_id="update_1",
            node_type=NodeType.ROOM,
            version=FirmwareVersion(1, 1, 0),
            channel=UpdateChannel.STABLE,
            priority=UpdatePriority.NORMAL,
            release_notes="Test",
        )
        ota_manager.register_update(update)

        # Check for updates with older version
        current = FirmwareVersion(1, 0, 0)
        available = await ota_manager.check_for_updates(
            NodeType.ROOM,
            current,
            UpdateChannel.STABLE,
        )

        assert len(available) == 1
        assert available[0].version > current

    @pytest.mark.asyncio
    async def test_check_for_updates_none_available(self, ota_manager):
        """Test checking when no updates available."""
        current = FirmwareVersion(2, 0, 0)  # Newer than any update

        available = await ota_manager.check_for_updates(
            NodeType.ROOM,
            current,
            UpdateChannel.STABLE,
        )

        assert len(available) == 0

    def test_get_active_operations(self, ota_manager):
        """Test getting active operations."""
        ops = ota_manager.get_active_operations()

        assert isinstance(ops, list)

    def test_get_stats(self, ota_manager):
        """Test getting OTA statistics."""
        stats = ota_manager.get_stats()

        assert stats["total_updates"] == 0
        assert stats["successful_updates"] == 0
        assert "available_updates" in stats


# ============================================================================
# Health Monitoring Tests
# ============================================================================


class TestHealthMetric:
    """Tests for HealthMetric."""

    def test_metric_creation(self):
        """Test creating a health metric."""
        metric = HealthMetric(
            name="latency",
            value=50.0,
            unit="ms",
            max_healthy=100.0,
        )

        assert metric.name == "latency"
        assert metric.value == 50.0
        assert metric.is_healthy is True

    def test_metric_unhealthy_max(self):
        """Test metric exceeding max threshold."""
        metric = HealthMetric(
            name="latency",
            value=150.0,
            unit="ms",
            max_healthy=100.0,
        )

        assert metric.is_healthy is False

    def test_metric_unhealthy_min(self):
        """Test metric below min threshold."""
        metric = HealthMetric(
            name="signal",
            value=20.0,
            unit="dB",
            min_healthy=30.0,
        )

        assert metric.is_healthy is False


class TestHealthAlert:
    """Tests for HealthAlert."""

    def test_alert_creation(self):
        """Test creating a health alert."""
        alert = HealthAlert(
            alert_id="alert_1",
            severity=AlertSeverity.WARNING,
            source="node_1",
            message="High latency detected",
        )

        assert alert.alert_id == "alert_1"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.acknowledged is False
        assert alert.resolved is False

    def test_alert_to_dict(self):
        """Test converting alert to dict."""
        alert = HealthAlert(
            alert_id="alert_1",
            severity=AlertSeverity.ERROR,
            source="node_1",
            message="Test alert",
            metric_name="latency",
            metric_value=500.0,
        )

        data = alert.to_dict()

        assert data["alert_id"] == "alert_1"
        assert data["severity"] == "error"
        assert data["metric_name"] == "latency"


class TestNodeHealth:
    """Tests for NodeHealth."""

    def test_node_health_creation(self):
        """Test creating node health."""
        health = NodeHealth(
            node_id="node_1",
            node_type=NodeType.ROOM,
            name="Living Room",
        )

        assert health.node_id == "node_1"
        assert health.status == HealthStatus.UNKNOWN
        assert health.error_count == 0

    def test_node_health_to_dict(self):
        """Test converting node health to dict."""
        health = NodeHealth(
            node_id="node_1",
            node_type=NodeType.ROOM,
            name="Test",
            status=HealthStatus.HEALTHY,
        )

        data = health.to_dict()

        assert data["node_id"] == "node_1"
        assert data["status"] == "healthy"


class TestHealthConfig:
    """Tests for HealthConfig."""

    def test_default_config(self):
        """Test default health configuration."""
        config = HealthConfig()

        assert config.health_check_interval_seconds == 30
        assert config.heartbeat_timeout_seconds == 90
        assert config.latency_warning_ms == 200.0

    def test_custom_config(self):
        """Test custom health configuration."""
        config = HealthConfig(
            health_check_interval_seconds=60,
            latency_warning_ms=100.0,
        )

        assert config.health_check_interval_seconds == 60
        assert config.latency_warning_ms == 100.0


class TestHealthMonitor:
    """Tests for HealthMonitor."""

    @pytest.fixture
    def monitor(self):
        """Create test health monitor."""
        return HealthMonitor()

    @pytest.mark.asyncio
    async def test_start_stop(self, monitor):
        """Test starting and stopping monitor."""
        await monitor.start()
        await monitor.stop()
        # Should not raise

    def test_register_node(self, monitor):
        """Test registering a node."""
        health = monitor.register_node(
            node_id="node_1",
            node_type=NodeType.ROOM,
            name="Test Room",
        )

        assert health.node_id == "node_1"
        assert health.status == HealthStatus.UNKNOWN

    def test_unregister_node(self, monitor):
        """Test unregistering a node."""
        monitor.register_node("node_1", NodeType.ROOM, "Test")
        monitor.unregister_node("node_1")

        health = monitor.get_node_health("node_1")
        assert health is None

    def test_update_node_health(self, monitor):
        """Test updating node health."""
        monitor.register_node("node_1", NodeType.ROOM, "Test")

        health = monitor.update_node_health(
            node_id="node_1",
            connection_state=NodeState.READY,
            latency_ms=50.0,
            heartbeat=True,
        )

        assert health.connection_state == NodeState.READY
        assert health.latency_ms == 50.0
        assert health.last_heartbeat is not None

    def test_record_metric(self, monitor):
        """Test recording a metric."""
        monitor.register_node("node_1", NodeType.ROOM, "Test")

        monitor.record_metric(
            node_id="node_1",
            name="cpu_usage",
            value=45.0,
            unit="%",
            max_healthy=80.0,
        )

        health = monitor.get_node_health("node_1")
        assert "cpu_usage" in health.metrics

    def test_acknowledge_alert(self, monitor):
        """Test acknowledging an alert."""
        monitor.register_node("node_1", NodeType.ROOM, "Test")

        # Generate an alert
        monitor._generate_alert(
            source="node_1",
            severity=AlertSeverity.WARNING,
            message="Test alert",
        )

        alerts = monitor.get_active_alerts()
        assert len(alerts) == 1

        # Acknowledge it
        result = monitor.acknowledge_alert(alerts[0].alert_id)
        assert result is True

    def test_resolve_alert(self, monitor):
        """Test resolving an alert."""
        monitor.register_node("node_1", NodeType.ROOM, "Test")

        monitor._generate_alert(
            source="node_1",
            severity=AlertSeverity.WARNING,
            message="Test alert",
        )

        alerts = monitor.get_active_alerts()
        result = monitor.resolve_alert(alerts[0].alert_id)
        assert result is True

        active = monitor.get_active_alerts()
        assert len(active) == 0

    def test_get_system_health(self, monitor):
        """Test getting system health."""
        monitor.register_node("node_1", NodeType.ROOM, "Test 1")
        monitor.register_node("node_2", NodeType.MOBILE, "Test 2")

        system_health = monitor.get_system_health()

        assert system_health.total_nodes == 2

    def test_get_dashboard_data(self, monitor):
        """Test getting dashboard data."""
        monitor.register_node("node_1", NodeType.ROOM, "Test")

        data = monitor.get_dashboard_data()

        assert "system" in data
        assert "nodes" in data
        assert "alerts" in data
        assert "stats" in data

    def test_get_stats(self, monitor):
        """Test getting monitor statistics."""
        stats = monitor.get_stats()

        assert stats["total_checks"] == 0
        assert stats["alerts_generated"] == 0
        assert stats["monitored_nodes"] == 0

    def test_alert_callback(self, monitor):
        """Test alert callback registration."""
        alerts_received = []
        monitor.on_alert(lambda a: alerts_received.append(a))

        monitor.register_node("node_1", NodeType.ROOM, "Test")
        monitor._generate_alert(
            source="node_1",
            severity=AlertSeverity.WARNING,
            message="Test",
        )

        assert len(alerts_received) == 1

    def test_status_callback(self, monitor):
        """Test status change callback."""
        status_updates = []
        monitor.on_status_change(lambda s: status_updates.append(s))

        monitor.register_node("node_1", NodeType.ROOM, "Test")

        # Status callback should be triggered on registration
        assert len(status_updates) >= 1

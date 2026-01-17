"""
SAIL Nodes Module Tests

Comprehensive tests for satellite nodes: room, mobile, and vehicle.
"""


import pytest
from src.hub.base import NodeCapability, NodeType
from src.nodes.base import (
    MessageType,
    NodeConfig,
    NodeMessage,
)
from src.nodes.mobile import (
    GPSLocation,
    GPSTracker,
    LocationAccuracy,
    MobileNode,
    MobileNodeConfig,
    OfflineStore,
    PushNotificationManager,
)
from src.nodes.protocol import (
    FrameType,
    MessageFrame,
    Protocol,
    ProtocolConfig,
    SecureChannel,
)
from src.nodes.room import (
    LEDController,
    LEDState,
    RoomNode,
    RoomNodeConfig,
    WakeWordDetector,
)
from src.nodes.vehicle import (
    AccidentDetector,
    AccidentEvent,
    AlertLevel,
    DrivingModeManager,
    DrivingState,
    OBDConnection,
    VehicleData,
    VehicleNode,
    VehicleNodeConfig,
)

# ============================================================================
# Base Node Tests
# ============================================================================


class TestNodeMessage:
    """Tests for NodeMessage."""

    def test_message_creation(self):
        """Test creating a node message."""
        msg = NodeMessage(
            message_id="msg_1",
            message_type=MessageType.QUERY,
            payload={"text": "Hello"},
        )

        assert msg.message_id == "msg_1"
        assert msg.message_type == MessageType.QUERY
        assert msg.payload["text"] == "Hello"

    def test_message_to_dict(self):
        """Test converting message to dict."""
        msg = NodeMessage(
            message_id="msg_1",
            message_type=MessageType.QUERY,
            payload={"data": "test"},
            source_node_id="node_1",
        )

        result = msg.to_dict()

        assert result["message_id"] == "msg_1"
        assert result["message_type"] == "query"
        assert result["source_node_id"] == "node_1"

    def test_message_from_dict(self):
        """Test creating message from dict."""
        data = {
            "message_id": "msg_1",
            "message_type": "heartbeat",
            "payload": {},
            "timestamp": "2024-01-01T12:00:00",
        }

        msg = NodeMessage.from_dict(data)

        assert msg.message_id == "msg_1"
        assert msg.message_type == MessageType.HEARTBEAT


class TestNodeConfig:
    """Tests for NodeConfig."""

    def test_config_defaults(self):
        """Test NodeConfig default values."""
        config = NodeConfig(name="Test Node")

        assert config.name == "Test Node"
        assert config.hub_address == "localhost"
        assert config.hub_port == 8420
        assert config.auto_reconnect is True
        assert config.tls_enabled is True

    def test_config_custom(self):
        """Test NodeConfig with custom values."""
        config = NodeConfig(
            name="Custom Node",
            hub_address="192.168.1.1",
            hub_port=9000,
            reconnect_interval_seconds=10,
        )

        assert config.hub_address == "192.168.1.1"
        assert config.hub_port == 9000
        assert config.reconnect_interval_seconds == 10


# ============================================================================
# Protocol Tests
# ============================================================================


class TestMessageFrame:
    """Tests for MessageFrame."""

    def test_frame_pack(self):
        """Test packing a frame."""
        frame = MessageFrame(
            frame_type=FrameType.DATA,
            sequence=1,
            payload=b"Hello",
        )

        packed = frame.pack()

        assert len(packed) > MessageFrame.HEADER_SIZE
        assert packed[:1] == bytes([FrameType.DATA.value])

    def test_frame_unpack(self):
        """Test unpacking a frame."""
        frame = MessageFrame(
            frame_type=FrameType.DATA,
            sequence=42,
            payload=b"Test data",
        )

        packed = frame.pack()
        unpacked = MessageFrame.unpack(packed)

        assert unpacked.frame_type == FrameType.DATA
        assert unpacked.sequence == 42
        assert unpacked.payload == b"Test data"

    def test_frame_with_hmac(self):
        """Test frame with HMAC signature."""
        hmac_key = b"secret_key_12345"

        frame = MessageFrame(
            frame_type=FrameType.DATA,
            sequence=1,
            payload=b"Signed data",
        )

        packed = frame.pack(hmac_key=hmac_key)
        unpacked = MessageFrame.unpack(packed, hmac_key=hmac_key)

        assert unpacked.payload == b"Signed data"
        assert unpacked.signature is not None


class TestProtocol:
    """Tests for Protocol."""

    @pytest.fixture
    def protocol(self):
        """Create test protocol."""
        return Protocol()

    def test_create_frame(self, protocol):
        """Test creating a frame."""
        frame = protocol.create_frame(
            FrameType.DATA,
            {"message": "Hello"},
        )

        assert frame.frame_type == FrameType.DATA
        assert frame.sequence == 1
        assert b"Hello" in frame.payload

    def test_decode_payload(self, protocol):
        """Test decoding frame payload."""
        frame = protocol.create_frame(
            FrameType.DATA,
            {"key": "value"},
        )

        result = protocol.decode_payload(frame)

        assert result["key"] == "value"

    def test_sequence_increment(self, protocol):
        """Test sequence number incrementing."""
        frame1 = protocol.create_frame(FrameType.DATA, {})
        frame2 = protocol.create_frame(FrameType.DATA, {})

        assert frame2.sequence == frame1.sequence + 1


class TestSecureChannel:
    """Tests for SecureChannel."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return ProtocolConfig(
            host="localhost",
            port=8420,
            tls_enabled=False,
        )

    def test_channel_creation(self, config):
        """Test creating a secure channel."""
        channel = SecureChannel(config)

        assert not channel.is_connected
        assert channel.local_address is None


# ============================================================================
# Room Node Tests
# ============================================================================


class TestRoomNodeConfig:
    """Tests for RoomNodeConfig."""

    def test_config_defaults(self):
        """Test RoomNodeConfig defaults."""
        config = RoomNodeConfig(name="Living Room")

        assert config.name == "Living Room"
        assert config.sample_rate == 16000
        assert config.wake_word_enabled is True
        assert config.led_enabled is True

    def test_config_custom(self):
        """Test RoomNodeConfig custom values."""
        config = RoomNodeConfig(
            name="Kitchen",
            audio_device=2,
            led_brightness=0.8,
            wake_word_sensitivity=0.7,
        )

        assert config.audio_device == 2
        assert config.led_brightness == 0.8
        assert config.wake_word_sensitivity == 0.7


class TestLEDController:
    """Tests for LEDController."""

    @pytest.fixture
    def led(self):
        """Create test LED controller."""
        return LEDController(enabled=True, gpio_pin=18)

    @pytest.mark.asyncio
    async def test_set_state(self, led):
        """Test setting LED state."""
        await led.set_state(LEDState.LISTENING)

        assert led._current_state == LEDState.LISTENING

    @pytest.mark.asyncio
    async def test_set_state_off(self, led):
        """Test turning LED off."""
        await led.set_state(LEDState.LISTENING)
        await led.set_state(LEDState.OFF)

        assert led._current_state == LEDState.OFF

    @pytest.mark.asyncio
    async def test_cleanup(self, led):
        """Test LED cleanup."""
        await led.set_state(LEDState.PROCESSING)
        await led.cleanup()

        # Should not raise


class TestWakeWordDetector:
    """Tests for WakeWordDetector."""

    def test_detector_creation(self):
        """Test creating wake word detector."""
        detector = WakeWordDetector(
            wake_word="hey sail",
            sensitivity=0.6,
        )

        assert detector._wake_word == "hey sail"
        assert detector._sensitivity == 0.6

    @pytest.mark.asyncio
    async def test_load_model(self):
        """Test loading wake word model."""
        detector = WakeWordDetector()
        await detector.load_model()

        assert detector._model_loaded is True

    def test_on_wake_callback(self):
        """Test wake word callback registration."""
        detector = WakeWordDetector()
        calls = []

        detector.on_wake(lambda: calls.append(1))
        detector._trigger_wake()

        assert len(calls) == 1


class TestRoomNode:
    """Tests for RoomNode."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return RoomNodeConfig(
            name="Test Room",
            location="Living Room",
            hub_address="localhost",
        )

    def test_node_creation(self, config):
        """Test creating a room node."""
        node = RoomNode(config)

        assert node._node_type == NodeType.ROOM
        assert node.firmware_version == "1.0.0"
        assert NodeCapability.VOICE_INPUT in node._capabilities
        assert NodeCapability.LED_INDICATOR in node._capabilities

    def test_hardware_info(self, config):
        """Test getting hardware info."""
        node = RoomNode(config)
        info = node.hardware_info

        assert info["platform"] == "raspberry_pi"
        assert "gpio_pins" in info

    def test_get_status(self, config):
        """Test getting node status."""
        node = RoomNode(config)
        status = node.get_status()

        assert status["node_type"] == "room"
        assert "is_listening" in status
        assert "led_state" in status


# ============================================================================
# Mobile Node Tests
# ============================================================================


class TestGPSLocation:
    """Tests for GPSLocation."""

    def test_location_creation(self):
        """Test creating GPS location."""
        loc = GPSLocation(
            latitude=37.7749,
            longitude=-122.4194,
            accuracy=10.0,
        )

        assert loc.latitude == 37.7749
        assert loc.longitude == -122.4194
        assert loc.accuracy == 10.0

    def test_location_to_dict(self):
        """Test converting location to dict."""
        loc = GPSLocation(
            latitude=37.7749,
            longitude=-122.4194,
            speed=15.0,
        )

        result = loc.to_dict()

        assert result["latitude"] == 37.7749
        assert result["speed"] == 15.0


class TestGPSTracker:
    """Tests for GPSTracker."""

    @pytest.fixture
    def tracker(self):
        """Create test GPS tracker."""
        return GPSTracker(
            accuracy=LocationAccuracy.BALANCED,
            update_interval=5,
            min_distance=10.0,
        )

    @pytest.mark.asyncio
    async def test_start_stop(self, tracker):
        """Test starting and stopping tracker."""
        await tracker.start()
        assert tracker.is_tracking

        await tracker.stop()
        assert not tracker.is_tracking

    def test_calculate_distance(self, tracker):
        """Test distance calculation."""
        # San Francisco to Oakland (~13km)
        distance = tracker._calculate_distance(
            37.7749, -122.4194,  # SF
            37.8044, -122.2712,  # Oakland
        )

        assert 12000 < distance < 15000  # ~13km

    def test_on_location_callback(self, tracker):
        """Test location callback registration."""
        locations = []
        tracker.on_location(lambda loc: locations.append(loc))

        loc = GPSLocation(latitude=0, longitude=0)
        tracker._notify_location(loc)

        assert len(locations) == 1


class TestPushNotificationManager:
    """Tests for PushNotificationManager."""

    @pytest.fixture
    def notifier(self):
        """Create test notification manager."""
        return PushNotificationManager(token="test_token")

    def test_set_token(self, notifier):
        """Test setting push token."""
        notifier.set_token("new_token")
        assert notifier._token == "new_token"

    @pytest.mark.asyncio
    async def test_show_notification(self, notifier):
        """Test showing notification."""
        from src.nodes.mobile import Notification, NotificationPriority

        notification = Notification(
            title="Test",
            body="Test notification",
            priority=NotificationPriority.HIGH,
        )

        await notifier.show_notification(notification)
        # Should not raise


class TestOfflineStore:
    """Tests for OfflineStore."""

    @pytest.fixture
    def store(self):
        """Create test offline store."""
        return OfflineStore(max_entries=5)

    def test_store_entry(self, store):
        """Test storing an entry."""
        store.store_entry({"content": "test"})
        assert store.entry_count == 1

    def test_store_query(self, store):
        """Test storing a query."""
        store.store_query("What time is it?", {"user": "test"})
        assert store.query_count == 1

    def test_max_entries_limit(self, store):
        """Test max entries limit."""
        for i in range(10):
            store.store_entry({"index": i})

        assert store.entry_count == 5
        entries = store.get_entries()
        assert entries[0]["index"] == 5  # Oldest removed

    def test_clear(self, store):
        """Test clearing store."""
        store.store_entry({"test": "data"})
        store.store_query("query")
        store.clear()

        assert store.entry_count == 0
        assert store.query_count == 0


class TestMobileNode:
    """Tests for MobileNode."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return MobileNodeConfig(
            name="Test Phone",
            gps_enabled=True,
            push_enabled=True,
        )

    def test_node_creation(self, config):
        """Test creating a mobile node."""
        node = MobileNode(config)

        assert node._node_type == NodeType.MOBILE
        assert NodeCapability.GPS in node._capabilities
        assert NodeCapability.PUSH_NOTIFICATION in node._capabilities

    def test_hardware_info(self, config):
        """Test getting hardware info."""
        node = MobileNode(config)
        info = node.hardware_info

        assert info["platform"] == "mobile"
        assert info["has_gps"] is True

    def test_update_battery(self, config):
        """Test updating battery status."""
        node = MobileNode(config)

        node.update_battery(50, is_charging=True)

        assert node._battery_level == 50
        assert node._is_charging is True

    def test_get_status(self, config):
        """Test getting node status."""
        node = MobileNode(config)
        status = node.get_status()

        assert status["node_type"] == "mobile"
        assert "battery_level" in status
        assert "is_foreground" in status


# ============================================================================
# Vehicle Node Tests
# ============================================================================


class TestVehicleData:
    """Tests for VehicleData."""

    def test_vehicle_data_creation(self):
        """Test creating vehicle data."""
        data = VehicleData(
            speed_kmh=60.0,
            rpm=2500,
            fuel_level=75.0,
        )

        assert data.speed_kmh == 60.0
        assert data.rpm == 2500
        assert data.fuel_level == 75.0

    def test_vehicle_data_to_dict(self):
        """Test converting to dict."""
        data = VehicleData(speed_kmh=100.0, rpm=3000)
        result = data.to_dict()

        assert result["speed_kmh"] == 100.0
        assert result["rpm"] == 3000


class TestOBDConnection:
    """Tests for OBDConnection."""

    @pytest.fixture
    def obd(self):
        """Create test OBD connection."""
        return OBDConnection(protocol="auto")

    @pytest.mark.asyncio
    async def test_connect(self, obd):
        """Test connecting to OBD."""
        result = await obd.connect()
        assert result is True
        assert obd.is_connected

    @pytest.mark.asyncio
    async def test_disconnect(self, obd):
        """Test disconnecting from OBD."""
        await obd.connect()
        await obd.disconnect()

        assert not obd.is_connected

    @pytest.mark.asyncio
    async def test_get_speed(self, obd):
        """Test getting speed."""
        await obd.connect()
        speed = await obd.get_speed()

        assert isinstance(speed, float)

    @pytest.mark.asyncio
    async def test_get_full_data(self, obd):
        """Test getting full vehicle data."""
        await obd.connect()
        data = await obd.get_full_data()

        assert isinstance(data, VehicleData)


class TestAccidentDetector:
    """Tests for AccidentDetector."""

    @pytest.fixture
    def detector(self):
        """Create test accident detector."""
        return AccidentDetector(g_threshold=3.0)

    def test_normal_acceleration(self, detector):
        """Test normal acceleration doesn't trigger."""
        result = detector.process_acceleration(0.5, 0.3, 1.0, 50.0)
        assert result is None

    def test_accident_detection(self, detector):
        """Test accident detection."""
        result = detector.process_acceleration(5.0, 2.0, 3.0, 80.0)

        assert result is not None
        assert isinstance(result, AccidentEvent)
        assert result.severity in (AlertLevel.WARNING, AlertLevel.CRITICAL)

    def test_on_accident_callback(self, detector):
        """Test accident callback."""
        events = []
        detector.on_accident(lambda e: events.append(e))

        detector.process_acceleration(10.0, 0.0, 0.0, 100.0)

        assert len(events) == 1


class TestDrivingModeManager:
    """Tests for DrivingModeManager."""

    @pytest.fixture
    def manager(self):
        """Create test driving mode manager."""
        return DrivingModeManager(
            driving_threshold=5.0,
            highway_threshold=80.0,
        )

    def test_initial_state(self, manager):
        """Test initial state is parked."""
        assert manager.state == DrivingState.PARKED
        assert not manager.is_driving

    def test_update_to_moving(self, manager):
        """Test updating to moving state."""
        state = manager.update(speed=30.0, rpm=2000)

        assert state == DrivingState.MOVING
        assert manager.is_driving

    def test_update_to_highway(self, manager):
        """Test updating to highway state."""
        state = manager.update(speed=100.0, rpm=3000)

        assert state == DrivingState.HIGHWAY
        assert manager.is_driving

    def test_update_to_idling(self, manager):
        """Test updating to idling state."""
        state = manager.update(speed=0.0, rpm=800)

        assert state == DrivingState.IDLING
        assert not manager.is_driving

    def test_state_change_callback(self, manager):
        """Test state change callback."""
        transitions = []
        manager.on_state_change(lambda old, new: transitions.append((old, new)))

        manager.update(speed=50.0, rpm=2500)

        assert len(transitions) == 1
        assert transitions[0] == (DrivingState.PARKED, DrivingState.MOVING)


class TestVehicleNode:
    """Tests for VehicleNode."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return VehicleNodeConfig(
            name="Test Vehicle",
            obd_enabled=True,
            accident_detection_enabled=True,
        )

    def test_node_creation(self, config):
        """Test creating a vehicle node."""
        node = VehicleNode(config)

        assert node._node_type == NodeType.VEHICLE
        assert NodeCapability.OBD in node._capabilities
        assert NodeCapability.MOTION in node._capabilities

    def test_hardware_info(self, config):
        """Test getting hardware info."""
        node = VehicleNode(config)
        info = node.hardware_info

        assert info["platform"] == "vehicle_obd"
        assert info["has_accelerometer"] is True

    def test_process_acceleration(self, config):
        """Test processing acceleration data."""
        node = VehicleNode(config)
        # Initialize accident detector manually for test
        node._accident_detector = AccidentDetector(g_threshold=3.0)

        # Normal acceleration
        result = node.process_acceleration(0.5, 0.3, 1.0)
        assert result is None

        # Accident-level acceleration
        result = node.process_acceleration(10.0, 5.0, 3.0)
        assert result is not None

    def test_get_status(self, config):
        """Test getting node status."""
        node = VehicleNode(config)
        status = node.get_status()

        assert status["node_type"] == "vehicle"
        assert "driving_state" in status
        assert "vehicle_data" in status

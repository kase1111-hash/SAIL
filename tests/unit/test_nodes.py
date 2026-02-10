"""
SAIL Nodes Module Tests

Tests for satellite nodes: base, protocol, and room.
"""


import pytest
from src.hub.base import NodeCapability, NodeType
from src.nodes.base import (
    MessageType,
    NodeConfig,
    NodeMessage,
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



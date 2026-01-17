"""
SAIL Node Communication Protocol

Secure communication protocol for node-hub communication.
Implements TLS encryption, message framing, and authentication.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import logging
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FrameType(int, Enum):
    """Types of protocol frames."""

    DATA = 0x01
    HEARTBEAT = 0x02
    ACK = 0x03
    ERROR = 0x04
    CLOSE = 0x05


class ProtocolConfig(BaseModel):
    """Protocol configuration."""

    host: str = Field(default="localhost", description="Hub host")
    port: int = Field(default=8420, description="Hub port")
    tls_enabled: bool = Field(default=True, description="Enable TLS")
    tls_verify: bool = Field(default=True, description="Verify TLS certs")
    tls_ca_cert: str | None = Field(default=None, description="CA certificate path")

    # Framing
    max_frame_size: int = Field(default=1048576, description="Max frame size (1MB)")
    frame_header_size: int = Field(default=16, description="Frame header size")

    # Timeouts
    connect_timeout: float = Field(default=10.0, description="Connection timeout")
    read_timeout: float = Field(default=30.0, description="Read timeout")
    write_timeout: float = Field(default=10.0, description="Write timeout")

    # Authentication
    auth_timeout: float = Field(default=10.0, description="Auth timeout")
    hmac_key: bytes | None = Field(default=None, description="HMAC key for signing")


@dataclass
class MessageFrame:
    """A framed message for transmission."""

    frame_type: FrameType
    sequence: int
    payload: bytes
    timestamp: float = field(default_factory=time.time)
    signature: bytes | None = None

    # Frame header format: type(1) + flags(1) + seq(4) + length(4) + timestamp(4) + reserved(2)
    HEADER_FORMAT = "!BBIIIH"
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

    def pack(self, hmac_key: bytes | None = None) -> bytes:
        """Pack frame for transmission."""
        payload_length = len(self.payload)
        flags = 0x01 if hmac_key else 0x00

        header = struct.pack(
            self.HEADER_FORMAT,
            self.frame_type.value,
            flags,
            self.sequence & 0xFFFFFFFF,
            payload_length,
            int(self.timestamp) & 0xFFFFFFFF,
            0,  # Reserved
        )

        frame_data = header + self.payload

        if hmac_key:
            signature = hmac.new(hmac_key, frame_data, hashlib.sha256).digest()
            return frame_data + signature

        return frame_data

    @classmethod
    def unpack(
        cls,
        data: bytes,
        hmac_key: bytes | None = None,
    ) -> MessageFrame:
        """Unpack frame from received data."""
        if len(data) < cls.HEADER_SIZE:
            raise ValueError("Data too short for frame header")

        header = data[: cls.HEADER_SIZE]
        frame_type, flags, sequence, payload_length, timestamp, _ = struct.unpack(
            cls.HEADER_FORMAT, header
        )

        expected_size = cls.HEADER_SIZE + payload_length
        has_signature = flags & 0x01

        if has_signature:
            expected_size += 32  # SHA256 digest size

        if len(data) < expected_size:
            raise ValueError(f"Data too short: expected {expected_size}, got {len(data)}")

        # Extract payload
        payload = data[cls.HEADER_SIZE : cls.HEADER_SIZE + payload_length]

        # Verify signature if present
        signature = None
        if has_signature:
            signature = data[cls.HEADER_SIZE + payload_length :]
            if hmac_key:
                frame_data = data[: cls.HEADER_SIZE + payload_length]
                expected_sig = hmac.new(hmac_key, frame_data, hashlib.sha256).digest()
                if not hmac.compare_digest(signature, expected_sig):
                    raise SecurityError("Invalid frame signature")

        return cls(
            frame_type=FrameType(frame_type),
            sequence=sequence,
            payload=payload,
            timestamp=float(timestamp),
            signature=signature,
        )


class Protocol:
    """
    SAIL communication protocol.

    Handles message framing, serialization, and integrity verification.
    """

    def __init__(self, config: ProtocolConfig | None = None) -> None:
        """Initialize protocol."""
        self._config = config or ProtocolConfig()
        self._sequence = 0
        self._hmac_key = self._config.hmac_key

    def create_frame(
        self,
        frame_type: FrameType,
        data: dict[str, Any] | bytes,
    ) -> MessageFrame:
        """
        Create a message frame.

        Args:
            frame_type: Type of frame
            data: Frame payload (dict will be JSON encoded)

        Returns:
            Message frame
        """
        payload = json.dumps(data).encode("utf-8") if isinstance(data, dict) else data

        if len(payload) > self._config.max_frame_size:
            raise ValueError(f"Payload too large: {len(payload)} > {self._config.max_frame_size}")

        self._sequence += 1

        return MessageFrame(
            frame_type=frame_type,
            sequence=self._sequence,
            payload=payload,
        )

    def pack_frame(self, frame: MessageFrame) -> bytes:
        """Pack frame for transmission."""
        return frame.pack(self._hmac_key)

    def unpack_frame(self, data: bytes) -> MessageFrame:
        """Unpack frame from received data."""
        return MessageFrame.unpack(data, self._hmac_key)

    def decode_payload(self, frame: MessageFrame) -> dict[str, Any]:
        """Decode frame payload as JSON."""
        try:
            return json.loads(frame.payload.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON payload: {e}")

    def set_hmac_key(self, key: bytes) -> None:
        """Set HMAC key for message signing."""
        self._hmac_key = key


class SecureChannel:
    """
    Secure communication channel to hub.

    Provides encrypted, authenticated communication using TLS
    with additional message-level integrity verification.
    """

    def __init__(self, config: ProtocolConfig | None = None) -> None:
        """Initialize secure channel."""
        self._config = config or ProtocolConfig()
        self._protocol = Protocol(config)

        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._connected = False

        self._read_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()

        # Connection info
        self._local_addr: tuple[str, int] | None = None
        self._remote_addr: tuple[str, int] | None = None

    async def connect(self) -> None:
        """Establish connection to hub."""
        if self._connected:
            return

        logger.info(f"Connecting to {self._config.host}:{self._config.port}")

        try:
            # Configure SSL if enabled
            ssl_context = None
            if self._config.tls_enabled:
                import ssl

                ssl_context = ssl.create_default_context()
                if not self._config.tls_verify:
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE
                elif self._config.tls_ca_cert:
                    ssl_context.load_verify_locations(self._config.tls_ca_cert)

            # Open connection
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(
                    self._config.host,
                    self._config.port,
                    ssl=ssl_context,
                ),
                timeout=self._config.connect_timeout,
            )

            # Get addresses
            self._local_addr = self._writer.get_extra_info("sockname")
            self._remote_addr = self._writer.get_extra_info("peername")

            self._connected = True
            logger.info(f"Connected to hub: {self._remote_addr}")

        except TimeoutError:
            raise ConnectionError("Connection timeout")
        except Exception as e:
            raise ConnectionError(f"Connection failed: {e}")

    async def disconnect(self) -> None:
        """Close connection."""
        if not self._connected:
            return

        self._connected = False

        if self._writer:
            try:
                # Send close frame
                frame = self._protocol.create_frame(FrameType.CLOSE, {})
                await self._write_frame(frame)
            except Exception:
                pass

            self._writer.close()
            with contextlib.suppress(Exception):
                await self._writer.wait_closed()

        self._reader = None
        self._writer = None
        logger.info("Disconnected from hub")

    async def send(self, data: dict[str, Any]) -> None:
        """
        Send data to hub.

        Args:
            data: Data to send (will be JSON encoded)
        """
        if not self._connected:
            raise ConnectionError("Not connected")

        frame = self._protocol.create_frame(FrameType.DATA, data)
        await self._write_frame(frame)

    async def receive(self, timeout: float | None = None) -> dict[str, Any] | None:
        """
        Receive data from hub.

        Args:
            timeout: Optional timeout override

        Returns:
            Received data or None on timeout
        """
        if not self._connected:
            raise ConnectionError("Not connected")

        try:
            frame = await self._read_frame(timeout or self._config.read_timeout)
            if frame is None:
                return None

            if frame.frame_type == FrameType.CLOSE:
                await self.disconnect()
                return None

            if frame.frame_type == FrameType.DATA:
                return self._protocol.decode_payload(frame)

            return None

        except TimeoutError:
            return None

    async def _write_frame(self, frame: MessageFrame) -> None:
        """Write frame to connection."""
        if not self._writer:
            raise ConnectionError("No connection")

        async with self._write_lock:
            packed = self._protocol.pack_frame(frame)

            # Write length prefix
            length_prefix = struct.pack("!I", len(packed))
            self._writer.write(length_prefix + packed)

            await asyncio.wait_for(
                self._writer.drain(),
                timeout=self._config.write_timeout,
            )

    async def _read_frame(self, timeout: float) -> MessageFrame | None:
        """Read frame from connection."""
        if not self._reader:
            raise ConnectionError("No connection")

        async with self._read_lock:
            # Read length prefix
            length_data = await asyncio.wait_for(
                self._reader.readexactly(4),
                timeout=timeout,
            )
            length = struct.unpack("!I", length_data)[0]

            if length > self._config.max_frame_size + MessageFrame.HEADER_SIZE + 32:
                raise ValueError(f"Frame too large: {length}")

            # Read frame data
            frame_data = await asyncio.wait_for(
                self._reader.readexactly(length),
                timeout=timeout,
            )

            return self._protocol.unpack_frame(frame_data)

    async def send_heartbeat(self) -> None:
        """Send heartbeat frame."""
        if not self._connected:
            return

        frame = self._protocol.create_frame(
            FrameType.HEARTBEAT,
            {"timestamp": time.time()},
        )
        await self._write_frame(frame)

    def set_hmac_key(self, key: bytes) -> None:
        """Set HMAC key for message integrity."""
        self._protocol.set_hmac_key(key)

    @property
    def is_connected(self) -> bool:
        """Check if channel is connected."""
        return self._connected

    @property
    def local_address(self) -> tuple[str, int] | None:
        """Get local address."""
        return self._local_addr

    @property
    def remote_address(self) -> tuple[str, int] | None:
        """Get remote address."""
        return self._remote_addr


class SecurityError(Exception):
    """Security-related error."""

    pass


# Discovery protocol for automatic hub finding

class DiscoveryProtocol(asyncio.DatagramProtocol):
    """UDP discovery protocol for finding hub on local network."""

    DISCOVERY_MESSAGE = b"SAIL_DISCOVER"
    RESPONSE_PREFIX = b"SAIL_HUB:"

    def __init__(self, future: asyncio.Future) -> None:
        """Initialize discovery protocol."""
        self._future = future
        self._transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:  # type: ignore
        """Called when connection is established."""
        self._transport = transport

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """Called when datagram is received."""
        if data.startswith(self.RESPONSE_PREFIX):
            try:
                hub_info = json.loads(data[len(self.RESPONSE_PREFIX) :].decode())
                if not self._future.done():
                    self._future.set_result({
                        "address": addr[0],
                        "port": hub_info.get("port", 8420),
                        "name": hub_info.get("name", "SAIL Hub"),
                        "version": hub_info.get("version", "unknown"),
                    })
            except Exception as e:
                logger.error(f"Failed to parse discovery response: {e}")

    def error_received(self, exc: Exception) -> None:
        """Called on error."""
        logger.error(f"Discovery error: {exc}")


async def discover_hub(
    broadcast_port: int = 8421,
    timeout: float = 5.0,
) -> dict[str, Any] | None:
    """
    Discover hub on local network.

    Args:
        broadcast_port: Port to broadcast discovery on
        timeout: Discovery timeout

    Returns:
        Hub info dict or None if not found
    """
    loop = asyncio.get_event_loop()
    future: asyncio.Future = loop.create_future()

    try:
        transport, _protocol = await loop.create_datagram_endpoint(
            lambda: DiscoveryProtocol(future),
            local_addr=("0.0.0.0", 0),
            allow_broadcast=True,
        )

        # Send discovery broadcast
        transport.sendto(
            DiscoveryProtocol.DISCOVERY_MESSAGE,
            ("255.255.255.255", broadcast_port),
        )

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except TimeoutError:
            logger.info("Hub discovery timed out")
            return None

    finally:
        if transport:
            transport.close()

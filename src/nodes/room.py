"""
SAIL Room Node

Raspberry Pi-based room node for in-home voice interaction.
Features wake word detection, streaming audio to hub, LED indicators,
and auto-discovery of the central hub.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import Field

from src.hub.base import NodeCapability, NodeType
from src.nodes.base import BaseNode, MessageType, NodeConfig

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class LEDState(str, Enum):
    """LED indicator states."""

    OFF = "off"
    LISTENING = "listening"  # Blue pulse
    PROCESSING = "processing"  # Yellow spin
    SPEAKING = "speaking"  # Green solid
    ERROR = "error"  # Red blink
    CONNECTING = "connecting"  # Blue blink


class LEDColor(str, Enum):
    """LED colors."""

    OFF = "off"
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"
    WHITE = "white"
    CYAN = "cyan"
    MAGENTA = "magenta"


@dataclass
class LEDPattern:
    """LED pattern configuration."""

    color: LEDColor
    brightness: float = 1.0  # 0.0 to 1.0
    pattern: str = "solid"  # solid, blink, pulse, spin
    duration_ms: int = 0  # 0 = indefinite


class RoomNodeConfig(NodeConfig):
    """Room node specific configuration."""

    # Audio settings
    audio_device: int | None = Field(default=None, description="Audio input device index")
    sample_rate: int = Field(default=16000, description="Audio sample rate")
    chunk_size: int = Field(default=480, description="Audio chunk size (30ms at 16kHz)")

    # Wake word
    wake_word_enabled: bool = Field(default=True, description="Enable wake word detection")
    wake_word_sensitivity: float = Field(default=0.5, description="Wake word sensitivity")

    # Local processing
    local_vad_enabled: bool = Field(default=True, description="Enable local VAD")
    stream_to_hub: bool = Field(default=True, description="Stream audio to hub")

    # LED settings
    led_enabled: bool = Field(default=True, description="Enable LED indicators")
    led_brightness: float = Field(default=0.5, description="LED brightness")
    led_gpio_pin: int = Field(default=18, description="GPIO pin for LED (Pi)")

    # Hardware
    button_enabled: bool = Field(default=True, description="Enable physical button")
    button_gpio_pin: int = Field(default=17, description="GPIO pin for button")


class LEDController:
    """
    Controls LED indicators for room node.

    On Raspberry Pi, uses GPIO/PWM for RGB LED control.
    Provides visual feedback for node state.
    """

    def __init__(
        self,
        enabled: bool = True,
        gpio_pin: int = 18,
        brightness: float = 0.5,
    ) -> None:
        """Initialize LED controller."""
        self._enabled = enabled
        self._gpio_pin = gpio_pin
        self._brightness = brightness
        self._current_state = LEDState.OFF
        self._animation_task: asyncio.Task | None = None
        self._gpio_available = False

        # Try to initialize GPIO
        if enabled:
            self._init_gpio()

    def _init_gpio(self) -> None:
        """Initialize GPIO for LED control."""
        try:
            # Would use RPi.GPIO or gpiozero in production
            # For now, just log
            logger.info(f"LED controller initialized on GPIO {self._gpio_pin}")
            self._gpio_available = True
        except Exception as e:
            logger.warning(f"GPIO not available: {e}")
            self._gpio_available = False

    async def set_state(self, state: LEDState) -> None:
        """Set LED state."""
        if not self._enabled:
            return

        if state == self._current_state:
            return

        # Cancel existing animation
        if self._animation_task:
            self._animation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._animation_task

        self._current_state = state

        # Start new animation
        if state == LEDState.OFF:
            await self._set_color(LEDColor.OFF)

        elif state == LEDState.LISTENING:
            self._animation_task = asyncio.create_task(
                self._pulse_animation(LEDColor.BLUE, 1.0)
            )

        elif state == LEDState.PROCESSING:
            self._animation_task = asyncio.create_task(
                self._spin_animation(LEDColor.YELLOW, 0.5)
            )

        elif state == LEDState.SPEAKING:
            await self._set_color(LEDColor.GREEN)

        elif state == LEDState.ERROR:
            self._animation_task = asyncio.create_task(
                self._blink_animation(LEDColor.RED, 0.25)
            )

        elif state == LEDState.CONNECTING:
            self._animation_task = asyncio.create_task(
                self._blink_animation(LEDColor.BLUE, 0.5)
            )

    async def _set_color(self, color: LEDColor, brightness: float | None = None) -> None:
        """Set LED color."""
        if not self._gpio_available:
            logger.debug(f"LED: {color.value} (GPIO not available)")
            return

        b = brightness if brightness is not None else self._brightness
        logger.debug(f"LED: {color.value} at {b * 100:.0f}% brightness")
        # Would set actual GPIO PWM values here

    async def _pulse_animation(self, color: LEDColor, period: float) -> None:
        """Pulse animation."""
        import math

        t = 0.0
        try:
            while True:
                brightness = (math.sin(t * 2 * math.pi / period) + 1) / 2
                await self._set_color(color, brightness * self._brightness)
                await asyncio.sleep(0.05)
                t += 0.05
        except asyncio.CancelledError:
            pass

    async def _blink_animation(self, color: LEDColor, period: float) -> None:
        """Blink animation."""
        try:
            while True:
                await self._set_color(color)
                await asyncio.sleep(period / 2)
                await self._set_color(LEDColor.OFF)
                await asyncio.sleep(period / 2)
        except asyncio.CancelledError:
            pass

    async def _spin_animation(self, color: LEDColor, period: float) -> None:
        """Spin animation (for ring LEDs)."""
        # Simplified as just pulsing for single LED
        await self._pulse_animation(color, period)

    async def cleanup(self) -> None:
        """Cleanup LED resources."""
        if self._animation_task:
            self._animation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._animation_task

        await self._set_color(LEDColor.OFF)


class AudioCapture:
    """
    Audio capture for room node.

    Captures audio from microphone with VAD support.
    """

    def __init__(
        self,
        device: int | None = None,
        sample_rate: int = 16000,
        chunk_size: int = 480,
        vad_enabled: bool = True,
    ) -> None:
        """Initialize audio capture."""
        self._device = device
        self._sample_rate = sample_rate
        self._chunk_size = chunk_size
        self._vad_enabled = vad_enabled

        self._stream = None
        self._is_capturing = False
        self._audio_queue: asyncio.Queue | None = None
        self._capture_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start audio capture."""
        if self._is_capturing:
            return

        logger.info(f"Starting audio capture (device={self._device}, rate={self._sample_rate})")
        self._audio_queue = asyncio.Queue(maxsize=100)
        self._is_capturing = True

        # Would initialize sounddevice/pyaudio here
        # For now, simulate
        self._capture_task = asyncio.create_task(self._capture_loop())

    async def stop(self) -> None:
        """Stop audio capture."""
        self._is_capturing = False
        if self._stream:
            # Would close stream
            pass

    async def _capture_loop(self) -> None:
        """Main capture loop (simulated)."""
        while self._is_capturing:
            # In production, would read from actual audio device
            await asyncio.sleep(self._chunk_size / self._sample_rate)
            # Would put audio chunks in queue

    async def get_audio(self) -> bytes | None:
        """Get next audio chunk."""
        if not self._audio_queue:
            return None
        try:
            return await asyncio.wait_for(self._audio_queue.get(), timeout=0.1)
        except TimeoutError:
            return None

    @property
    def is_capturing(self) -> bool:
        """Check if capturing."""
        return self._is_capturing


class WakeWordDetector:
    """
    Local wake word detection.

    Uses OpenWakeWord for efficient local detection.
    """

    def __init__(
        self,
        wake_word: str = "hey sail",
        sensitivity: float = 0.5,
    ) -> None:
        """Initialize wake word detector."""
        self._wake_word = wake_word
        self._sensitivity = sensitivity
        self._model_loaded = False
        self._callbacks: list[Callable[[], None]] = []

    async def load_model(self) -> None:
        """Load wake word model."""
        logger.info(f"Loading wake word model for '{self._wake_word}'")
        # Would load OpenWakeWord model
        self._model_loaded = True

    def process_audio(self, audio_data: bytes) -> bool:
        """
        Process audio chunk for wake word.

        Args:
            audio_data: Audio data chunk

        Returns:
            True if wake word detected
        """
        if not self._model_loaded:
            return False

        # Would run through OpenWakeWord model
        # For now, simulate
        return False

    def on_wake(self, callback: Callable[[], None]) -> None:
        """Register wake callback."""
        self._callbacks.append(callback)

    def _trigger_wake(self) -> None:
        """Trigger wake callbacks."""
        for callback in self._callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Wake callback error: {e}")


class RoomNode(BaseNode):
    """
    Room node for in-home voice interaction.

    Provides:
    - Wake word detection
    - Audio streaming to hub
    - LED status indicators
    - Physical button support
    - Auto hub discovery
    """

    def __init__(self, config: RoomNodeConfig) -> None:
        """Initialize room node."""
        capabilities = [
            NodeCapability.VOICE_INPUT,
            NodeCapability.VOICE_OUTPUT,
            NodeCapability.WAKE_WORD,
            NodeCapability.LED_INDICATOR,
        ]

        super().__init__(
            node_type=NodeType.ROOM,
            config=config,
            capabilities=capabilities,
        )

        self._room_config = config

        # Components
        self._led = LEDController(
            enabled=config.led_enabled,
            gpio_pin=config.led_gpio_pin,
            brightness=config.led_brightness,
        )
        self._audio = AudioCapture(
            device=config.audio_device,
            sample_rate=config.sample_rate,
            chunk_size=config.chunk_size,
            vad_enabled=config.local_vad_enabled,
        )
        self._wake_detector: WakeWordDetector | None = None

        # State
        self._is_listening = False
        self._is_processing = False
        self._current_user: str | None = None

        # Tasks
        self._audio_task: asyncio.Task | None = None

    @property
    def firmware_version(self) -> str:
        """Get firmware version."""
        return "1.0.0"

    @property
    def hardware_info(self) -> dict[str, Any]:
        """Get hardware information."""
        return {
            "platform": "raspberry_pi",
            "model": "4B",
            "audio_device": self._room_config.audio_device,
            "gpio_pins": {
                "led": self._room_config.led_gpio_pin,
                "button": self._room_config.button_gpio_pin,
            },
        }

    async def _on_start(self) -> None:
        """Initialize room node resources."""
        await self._led.set_state(LEDState.CONNECTING)

        # Initialize wake word detector
        if self._room_config.wake_word_enabled:
            self._wake_detector = WakeWordDetector(
                sensitivity=self._room_config.wake_word_sensitivity,
            )
            await self._wake_detector.load_model()
            self._wake_detector.on_wake(self._on_wake_word)

        # Start audio capture
        await self._audio.start()

        # Start audio processing task
        self._audio_task = asyncio.create_task(self._audio_loop())
        self._tasks.append(self._audio_task)

        await self._led.set_state(LEDState.OFF)
        logger.info("Room node initialized")

    async def _on_stop(self) -> None:
        """Cleanup room node resources."""
        await self._audio.stop()
        await self._led.cleanup()

    async def _audio_loop(self) -> None:
        """Main audio processing loop."""
        while self._state.value != "disconnected":
            try:
                audio_chunk = await self._audio.get_audio()
                if audio_chunk:
                    await self._process_audio(audio_chunk)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Audio loop error: {e}")
                await asyncio.sleep(0.1)

    async def _process_audio(self, audio_data: bytes) -> None:
        """Process audio chunk."""
        # Check for wake word
        if self._wake_detector and not self._is_listening:
            if self._wake_detector.process_audio(audio_data):
                self._on_wake_word()
                return

        # Stream audio if listening
        if self._is_listening and self._room_config.stream_to_hub:
            await self.send_sensor_data("audio", {
                "data": audio_data.hex(),
                "sample_rate": self._room_config.sample_rate,
            })

    def _on_wake_word(self) -> None:
        """Handle wake word detection."""
        logger.info("Wake word detected")
        self._is_listening = True
        _ = asyncio.create_task(self._led.set_state(LEDState.LISTENING))
        self._emit_event("wake_word_detected")

    async def start_listening(self) -> None:
        """Start listening for user input."""
        if self._is_listening:
            return

        self._is_listening = True
        await self._led.set_state(LEDState.LISTENING)
        self._emit_event("listening_started")

    async def stop_listening(self) -> None:
        """Stop listening."""
        if not self._is_listening:
            return

        self._is_listening = False
        await self._led.set_state(LEDState.OFF)
        self._emit_event("listening_stopped")

    async def speak(self, text: str) -> None:
        """Speak text through room audio output."""
        await self._led.set_state(LEDState.SPEAKING)

        # Would send to TTS and play audio
        # For now, send to hub for processing
        await self._send_request(
            MessageType.QUERY,
            {"type": "speak", "text": text},
        )

        await self._led.set_state(LEDState.OFF)

    async def show_error(self, message: str) -> None:
        """Show error state."""
        await self._led.set_state(LEDState.ERROR)
        logger.error(f"Room node error: {message}")
        await asyncio.sleep(2)
        await self._led.set_state(LEDState.OFF)

    def get_status(self) -> dict[str, Any]:
        """Get room node status."""
        status = super().get_status()
        status.update({
            "is_listening": self._is_listening,
            "is_processing": self._is_processing,
            "current_user": self._current_user,
            "audio_capturing": self._audio.is_capturing,
            "led_state": self._led._current_state.value,
        })
        return status

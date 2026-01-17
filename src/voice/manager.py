"""
SAIL Voice Input Manager

Orchestrates the complete voice input pipeline including audio capture,
wake word detection, speech-to-text, and command detection.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, AsyncIterator, Callable

import numpy as np

from src.voice.audio.capture import AudioCapture, AudioChunk
from src.voice.audio.vad import AudioAccumulator, VADConfig, VADProcessor, VADState
from src.voice.commands import CommandDetector, CommandResult, CommandType, QuickStopDetector
from src.voice.stt.base import STTConfig, STTProvider, TranscriptionResult
from src.voice.stt.whisper import create_stt_provider
from src.voice.wake_word.base import WakeWordConfig, WakeWordDetector, WakeWordResult
from src.voice.wake_word.openwakeword import OpenWakeWordConfig, OpenWakeWordDetector

if TYPE_CHECKING:
    from src.config.schema import VoiceConfig


logger = logging.getLogger(__name__)


class VoiceInputState(str, Enum):
    """State of the voice input pipeline."""

    IDLE = "idle"
    LISTENING_FOR_WAKE_WORD = "listening_for_wake_word"
    LISTENING_FOR_SPEECH = "listening_for_speech"
    PROCESSING = "processing"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class VoiceInputEvent:
    """Event emitted by the voice input manager."""

    event_type: str
    data: dict = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class VoiceInputResult:
    """Result of voice input processing."""

    text: str
    transcription: TranscriptionResult | None = None
    command: CommandResult | None = None
    was_stopped: bool = False
    audio_duration_ms: float = 0.0

    @property
    def has_text(self) -> bool:
        """Check if result contains text."""
        return bool(self.text.strip())

    @property
    def is_command(self) -> bool:
        """Check if result is a command."""
        return self.command is not None and self.command.detected


class VoiceInputManager:
    """
    Manages the complete voice input pipeline.

    Coordinates audio capture, wake word detection, VAD, STT, and
    command detection to provide a unified voice input interface.
    """

    def __init__(
        self,
        voice_config: VoiceConfig | None = None,
        wake_word_detector: WakeWordDetector | None = None,
        stt_provider: STTProvider | None = None,
        command_detector: CommandDetector | None = None,
    ) -> None:
        """
        Initialize voice input manager.

        Args:
            voice_config: Voice configuration from SAIL config
            wake_word_detector: Custom wake word detector (creates default if None)
            stt_provider: Custom STT provider (creates default if None)
            command_detector: Custom command detector (creates default if None)
        """
        self._voice_config = voice_config
        self._state = VoiceInputState.IDLE

        # Components
        self._audio_capture: AudioCapture | None = None
        self._vad_processor: VADProcessor | None = None
        self._audio_accumulator: AudioAccumulator | None = None
        self._wake_word_detector = wake_word_detector
        self._stt_provider = stt_provider
        self._command_detector = command_detector or CommandDetector()
        self._quick_stop_detector = QuickStopDetector()

        # Event callbacks
        self._event_callbacks: list[Callable[[VoiceInputEvent], None]] = []
        self._on_utterance_callbacks: list[Callable[[VoiceInputResult], None]] = []
        self._on_stop_callbacks: list[Callable[[], None]] = []

        # State tracking
        self._is_running = False
        self._stop_requested = False
        self._wake_word_active = True
        self._current_audio_buffer: list[np.ndarray] = []

        # Initialize components
        self._init_components()

    def _init_components(self) -> None:
        """Initialize pipeline components from config."""
        # Get config values or use defaults
        sample_rate = 16000
        input_device = None
        wake_phrase = "hey sail"
        wake_sensitivity = 0.5
        stt_model = "base"
        stt_device = "cpu"

        if self._voice_config:
            sample_rate = self._voice_config.sample_rate
            input_device = self._voice_config.input_device
            wake_phrase = self._voice_config.wake_word.phrase
            wake_sensitivity = self._voice_config.wake_word.sensitivity
            stt_model = self._voice_config.stt.model
            stt_device = self._voice_config.stt.device

        # Audio capture
        self._audio_capture = AudioCapture(
            sample_rate=sample_rate,
            channels=1,
            chunk_duration_ms=30,
            device=input_device,
        )

        # VAD
        vad_config = VADConfig(
            sample_rate=sample_rate,
            frame_duration_ms=30,
            min_speech_duration_ms=250,
            silence_threshold_ms=700,
        )
        self._vad_processor = VADProcessor(vad_config)
        self._audio_accumulator = AudioAccumulator(
            vad=self._vad_processor,
            max_duration_ms=30000,
            pre_speech_buffer_ms=500,
        )

        # Wake word detector (lazy initialization)
        if self._wake_word_detector is None:
            try:
                wake_config = OpenWakeWordConfig(
                    phrase=wake_phrase,
                    sensitivity=wake_sensitivity,
                    sample_rate=sample_rate,
                )
                self._wake_word_detector = OpenWakeWordDetector(wake_config)
            except Exception as e:
                logger.warning(f"Failed to initialize OpenWakeWord: {e}")
                # Will fall back to always-on listening

        # STT provider (lazy initialization)
        if self._stt_provider is None:
            stt_config = STTConfig(
                model=stt_model,
                device=stt_device,
                sample_rate=sample_rate,
            )
            self._stt_provider = create_stt_provider(stt_config)

    async def start(self) -> None:
        """Start the voice input pipeline."""
        if self._is_running:
            return

        logger.info("Starting voice input pipeline")
        self._is_running = True
        self._stop_requested = False
        self._state = VoiceInputState.LISTENING_FOR_WAKE_WORD

        # Start audio capture
        self._audio_capture.start()

        # Load STT model
        await self._stt_provider.load_model()

        self._emit_event("started")

    async def stop(self) -> None:
        """Stop the voice input pipeline."""
        if not self._is_running:
            return

        logger.info("Stopping voice input pipeline")
        self._stop_requested = True
        self._is_running = False
        self._state = VoiceInputState.STOPPED

        # Stop audio capture
        if self._audio_capture:
            self._audio_capture.stop()

        # Unload STT model
        if self._stt_provider:
            await self._stt_provider.unload_model()

        self._emit_event("stopped")

    async def run(self) -> AsyncIterator[VoiceInputResult]:
        """
        Run the voice input pipeline continuously.

        Yields:
            VoiceInputResult for each detected utterance
        """
        await self.start()

        try:
            async for chunk in self._audio_capture.stream():
                if self._stop_requested:
                    break

                result = await self._process_chunk(chunk)
                if result is not None:
                    yield result

        finally:
            await self.stop()

    async def _process_chunk(self, chunk: AudioChunk) -> VoiceInputResult | None:
        """
        Process a single audio chunk through the pipeline.

        Args:
            chunk: Audio chunk to process

        Returns:
            VoiceInputResult if an utterance was detected, None otherwise
        """
        # Check for wake word if enabled
        if self._wake_word_active and self._wake_word_detector:
            if self._state == VoiceInputState.LISTENING_FOR_WAKE_WORD:
                wake_result = self._wake_word_detector.process(chunk)
                if wake_result.detected:
                    logger.debug(f"Wake word detected: {wake_result.wake_word}")
                    self._state = VoiceInputState.LISTENING_FOR_SPEECH
                    self._emit_event("wake_word_detected", {
                        "wake_word": wake_result.wake_word,
                        "confidence": wake_result.confidence,
                    })
                    # Reset accumulator for new utterance
                    self._audio_accumulator.reset()
                return None

        # Accumulate speech
        accumulated_audio, is_complete = self._audio_accumulator.process(chunk)

        if not is_complete:
            # Update state if speech started
            if self._audio_accumulator.is_accumulating:
                if self._state != VoiceInputState.LISTENING_FOR_SPEECH:
                    self._state = VoiceInputState.LISTENING_FOR_SPEECH
                    self._emit_event("speech_started")
            return None

        # Speech complete - transcribe
        if accumulated_audio is not None and len(accumulated_audio) > 0:
            self._state = VoiceInputState.PROCESSING
            self._emit_event("speech_ended")

            # Transcribe audio
            result = await self._transcribe_audio(accumulated_audio)

            # Return to wake word listening
            if self._wake_word_active:
                self._state = VoiceInputState.LISTENING_FOR_WAKE_WORD
            else:
                self._state = VoiceInputState.LISTENING_FOR_SPEECH

            return result

        return None

    async def _transcribe_audio(self, audio: np.ndarray) -> VoiceInputResult:
        """
        Transcribe audio to text and detect commands.

        Args:
            audio: Audio data to transcribe

        Returns:
            VoiceInputResult with transcription and command detection
        """
        try:
            transcription = await self._stt_provider.transcribe(audio)
            text = transcription.text.strip()

            # Detect commands
            command = self._command_detector.detect(text)

            # Handle stop commands
            was_stopped = False
            if command.is_stop_command:
                was_stopped = True
                self._handle_stop_command()

            result = VoiceInputResult(
                text=text,
                transcription=transcription,
                command=command if command.detected else None,
                was_stopped=was_stopped,
                audio_duration_ms=transcription.duration * 1000,
            )

            self._emit_event("utterance_complete", {
                "text": text,
                "is_command": command.detected,
                "command_type": command.command_type.value if command.detected else None,
            })

            # Trigger callbacks
            for callback in self._on_utterance_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Utterance callback error: {e}")

            return result

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return VoiceInputResult(
                text="",
                was_stopped=False,
                audio_duration_ms=len(audio) / 16000 * 1000,
            )

    def _handle_stop_command(self) -> None:
        """Handle a stop command."""
        logger.info("Stop command received - honoring respect protocol")
        self._emit_event("stop_command")

        # Trigger stop callbacks
        for callback in self._on_stop_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Stop callback error: {e}")

    async def listen_once(self, timeout_seconds: float = 30.0) -> VoiceInputResult | None:
        """
        Listen for a single utterance.

        Args:
            timeout_seconds: Maximum time to wait for speech

        Returns:
            VoiceInputResult or None if timeout
        """
        if not self._is_running:
            await self.start()

        start_time = asyncio.get_event_loop().time()
        accumulated_audio = None

        try:
            while True:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout_seconds:
                    return None

                chunk = await self._audio_capture.read_async(timeout=0.1)
                if chunk is None:
                    continue

                # Process through VAD
                audio, is_complete = self._audio_accumulator.process(chunk)

                if is_complete and audio is not None and len(audio) > 0:
                    accumulated_audio = audio
                    break

            if accumulated_audio is not None:
                return await self._transcribe_audio(accumulated_audio)

        except asyncio.TimeoutError:
            pass

        return None

    def enable_wake_word(self) -> None:
        """Enable wake word detection."""
        self._wake_word_active = True
        if self._is_running:
            self._state = VoiceInputState.LISTENING_FOR_WAKE_WORD

    def disable_wake_word(self) -> None:
        """Disable wake word detection (always listening mode)."""
        self._wake_word_active = False
        if self._is_running:
            self._state = VoiceInputState.LISTENING_FOR_SPEECH

    def on_event(self, callback: Callable[[VoiceInputEvent], None]) -> None:
        """
        Register a callback for all events.

        Args:
            callback: Function to call on events
        """
        self._event_callbacks.append(callback)

    def on_utterance(self, callback: Callable[[VoiceInputResult], None]) -> None:
        """
        Register a callback for completed utterances.

        Args:
            callback: Function to call when utterance is complete
        """
        self._on_utterance_callbacks.append(callback)

    def on_stop(self, callback: Callable[[], None]) -> None:
        """
        Register a callback for stop commands.

        Args:
            callback: Function to call when stop command is detected
        """
        self._on_stop_callbacks.append(callback)

    def _emit_event(self, event_type: str, data: dict | None = None) -> None:
        """Emit an event to all registered callbacks."""
        event = VoiceInputEvent(
            event_type=event_type,
            data=data or {},
            timestamp=asyncio.get_event_loop().time(),
        )

        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    @property
    def state(self) -> VoiceInputState:
        """Get current pipeline state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._is_running

    @property
    def is_listening(self) -> bool:
        """Check if actively listening for speech."""
        return self._state in (
            VoiceInputState.LISTENING_FOR_WAKE_WORD,
            VoiceInputState.LISTENING_FOR_SPEECH,
        )

    async def __aenter__(self) -> VoiceInputManager:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Async context manager exit."""
        await self.stop()


async def create_voice_input_manager(
    voice_config: VoiceConfig | None = None,
) -> VoiceInputManager:
    """
    Factory function to create a configured voice input manager.

    Args:
        voice_config: Voice configuration from SAIL config

    Returns:
        Configured VoiceInputManager
    """
    return VoiceInputManager(voice_config=voice_config)

"""
SAIL Voice Output Manager

Orchestrates the complete voice output pipeline including text-to-speech
synthesis, audio output management, and the respect protocol.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable

from src.voice.tts.base import (
    SpeechMode,
    SpeechPriority,
    SpeechRequest,
    SpeechResult,
    SpeechState,
    TTSConfig,
    TTSProvider,
    VoiceProfile,
)
from src.voice.tts.output import AudioOutputManager, OutputState, QueuedAudio
from src.voice.tts.piper import PiperConfig, PiperProvider, create_tts_provider

if TYPE_CHECKING:
    from src.config.schema import VoiceConfig

logger = logging.getLogger(__name__)


class VoiceOutputState(str, Enum):
    """State of the voice output pipeline."""

    IDLE = "idle"
    SYNTHESIZING = "synthesizing"
    SPEAKING = "speaking"
    PAUSED = "paused"
    INTERRUPTED = "interrupted"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class VoiceOutputEvent:
    """Event emitted by the voice output manager."""

    event_type: str
    data: dict = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class SpeakResult:
    """Result of a speak operation."""

    success: bool
    queue_id: str = ""
    was_interrupted: bool = False
    error: str | None = None
    duration_seconds: float = 0.0


class VoiceOutputManager:
    """
    Manages the complete voice output pipeline.

    Coordinates TTS synthesis and audio playback to provide a unified
    voice output interface with support for priority queuing,
    interruption, and the respect protocol.
    """

    def __init__(
        self,
        voice_config: VoiceConfig | None = None,
        tts_provider: TTSProvider | None = None,
    ) -> None:
        """
        Initialize voice output manager.

        Args:
            voice_config: Voice configuration from SAIL config
            tts_provider: Custom TTS provider (creates default if None)
        """
        self._voice_config = voice_config
        self._state = VoiceOutputState.IDLE

        # Components
        self._tts_provider = tts_provider
        self._audio_output: AudioOutputManager | None = None

        # State tracking
        self._is_running = False
        self._stop_requested = False
        self._current_mode = SpeechMode.NORMAL

        # Event callbacks
        self._event_callbacks: list[Callable[[VoiceOutputEvent], None]] = []
        self._on_speak_start_callbacks: list[Callable[[SpeechRequest], None]] = []
        self._on_speak_complete_callbacks: list[Callable[[SpeechRequest], None]] = []
        self._on_interrupt_callbacks: list[Callable[[], None]] = []

        # Initialize components
        self._init_components()

    def _init_components(self) -> None:
        """Initialize pipeline components from config."""
        # Get config values or use defaults
        sample_rate = 22050
        output_device = None
        voice = "en_US-lessac-medium"
        rate = 1.0

        if self._voice_config:
            output_device = self._voice_config.output_device
            voice = self._voice_config.tts.voice
            rate = self._voice_config.tts.rate

        # TTS provider
        if self._tts_provider is None:
            tts_config = PiperConfig(
                voice=voice,
                rate=rate,
                sample_rate=sample_rate,
                output_device=output_device,
            )
            self._tts_provider = create_tts_provider(tts_config)

        # Audio output manager
        self._audio_output = AudioOutputManager(
            sample_rate=sample_rate,
            output_device=output_device,
        )

        # Register audio callbacks
        self._audio_output.on_start(self._on_audio_start)
        self._audio_output.on_complete(self._on_audio_complete)
        self._audio_output.on_interrupt(self._on_audio_interrupt)

    async def start(self) -> None:
        """Start the voice output pipeline."""
        if self._is_running:
            return

        logger.info("Starting voice output pipeline")
        self._is_running = True
        self._stop_requested = False
        self._state = VoiceOutputState.IDLE

        # Load TTS model
        await self._tts_provider.load_model()

        # Start audio output
        self._audio_output.start()

        self._emit_event("started")

    async def stop(self) -> None:
        """Stop the voice output pipeline."""
        if not self._is_running:
            return

        logger.info("Stopping voice output pipeline")
        self._stop_requested = True
        self._is_running = False
        self._state = VoiceOutputState.STOPPED

        # Stop audio output
        if self._audio_output:
            self._audio_output.stop()

        # Unload TTS model
        if self._tts_provider:
            await self._tts_provider.unload_model()

        self._emit_event("stopped")

    async def speak(
        self,
        text: str,
        priority: SpeechPriority = SpeechPriority.NORMAL,
        mode: SpeechMode | None = None,
        rate: float | None = None,
        interruptible: bool = True,
        wait: bool = False,
    ) -> SpeakResult:
        """
        Synthesize and speak text.

        Args:
            text: Text to speak
            priority: Speech priority
            mode: Speech mode (calm, normal, urgent, crisis)
            rate: Speaking rate override
            interruptible: Whether speech can be interrupted
            wait: Whether to wait for speech to complete

        Returns:
            SpeakResult with operation status
        """
        if not self._is_running:
            return SpeakResult(
                success=False,
                error="Voice output not running",
            )

        if not text.strip():
            return SpeakResult(success=True)

        # Use current mode if not specified
        mode = mode or self._current_mode

        # Create speech request
        request = SpeechRequest(
            text=text,
            priority=priority,
            mode=mode,
            rate=rate,
            interruptible=interruptible,
            wait_for_completion=wait,
        )

        self._state = VoiceOutputState.SYNTHESIZING
        self._emit_event("synthesizing", {"text": text})

        try:
            # Get rate for mode
            if rate is None:
                rate = self._tts_provider.get_rate_for_mode(mode)

            # Synthesize text
            result = await self._tts_provider.synthesize(text)

            if not result.is_success:
                self._state = VoiceOutputState.ERROR
                return SpeakResult(
                    success=False,
                    error=result.error or "Synthesis failed",
                )

            # Queue for playback
            queue_id = self._audio_output.queue_audio(
                audio_data=result.audio_data,
                sample_rate=result.sample_rate,
                request=request,
            )

            self._state = VoiceOutputState.SPEAKING

            # Trigger callbacks
            for callback in self._on_speak_start_callbacks:
                try:
                    callback(request)
                except Exception as e:
                    logger.error(f"Speak start callback error: {e}")

            # Wait for completion if requested
            if wait:
                await self._wait_for_completion(queue_id)

            return SpeakResult(
                success=True,
                queue_id=queue_id,
                duration_seconds=result.duration_seconds,
            )

        except Exception as e:
            self._state = VoiceOutputState.ERROR
            logger.error(f"Speak failed: {e}")
            return SpeakResult(
                success=False,
                error=str(e),
            )

    async def speak_calm(self, text: str, **kwargs) -> SpeakResult:
        """Speak with calm, reassuring tone."""
        return await self.speak(text, mode=SpeechMode.CALM, **kwargs)

    async def speak_urgent(self, text: str, **kwargs) -> SpeakResult:
        """Speak with urgent, emphatic tone."""
        return await self.speak(text, mode=SpeechMode.URGENT, **kwargs)

    async def speak_crisis(
        self,
        text: str,
        priority: SpeechPriority = SpeechPriority.HIGH,
        **kwargs,
    ) -> SpeakResult:
        """Speak with clear, steady crisis tone (higher priority)."""
        return await self.speak(text, priority=priority, mode=SpeechMode.CRISIS, **kwargs)

    async def speak_alert(
        self,
        text: str,
        priority: SpeechPriority = SpeechPriority.CRITICAL,
        **kwargs,
    ) -> SpeakResult:
        """Speak critical alert (cannot be interrupted)."""
        return await self.speak(
            text,
            priority=priority,
            mode=SpeechMode.URGENT,
            interruptible=False,
            **kwargs,
        )

    def interrupt(self, clear_queue: bool = True) -> None:
        """
        Interrupt current speech immediately.

        Implements the respect protocol - immediately stops speech
        when user requests it.

        Args:
            clear_queue: Whether to clear pending speech
        """
        logger.info("Speech interrupted - honoring respect protocol")

        if self._audio_output:
            self._audio_output.interrupt(clear_queue=clear_queue)

        self._state = VoiceOutputState.INTERRUPTED
        self._emit_event("interrupted")

        # Trigger callbacks
        for callback in self._on_interrupt_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Interrupt callback error: {e}")

    def pause(self) -> None:
        """Pause current speech."""
        if self._audio_output:
            self._audio_output.pause()
        self._state = VoiceOutputState.PAUSED
        self._emit_event("paused")

    def resume(self) -> None:
        """Resume paused speech."""
        if self._audio_output:
            self._audio_output.resume()
        self._state = VoiceOutputState.SPEAKING
        self._emit_event("resumed")

    def set_mode(self, mode: SpeechMode) -> None:
        """
        Set the default speech mode.

        Args:
            mode: Speech mode to use by default
        """
        self._current_mode = mode
        logger.debug(f"Speech mode set to: {mode.value}")

    async def change_voice(self, voice: str) -> bool:
        """
        Change the TTS voice.

        Args:
            voice: Voice identifier to switch to

        Returns:
            True if voice change was successful
        """
        try:
            await self._tts_provider.load_model(voice)
            self._emit_event("voice_changed", {"voice": voice})
            return True
        except Exception as e:
            logger.error(f"Failed to change voice: {e}")
            return False

    async def list_voices(self) -> list[VoiceProfile]:
        """
        List available voices.

        Returns:
            List of available voice profiles
        """
        return await self._tts_provider.list_voices()

    async def _wait_for_completion(self, queue_id: str) -> None:
        """Wait for specific audio to complete playback."""
        while self._is_running:
            # Check if this item has completed
            current = self._audio_output.current_item
            if current is None or current.queue_id != queue_id:
                # Our item is not current, check if queue is empty
                if self._audio_output.queue_size == 0 and current is None:
                    break
            await asyncio.sleep(0.05)

    def _on_audio_start(self, item: QueuedAudio) -> None:
        """Callback when audio playback starts."""
        self._state = VoiceOutputState.SPEAKING
        self._emit_event("speaking", {"queue_id": item.queue_id})

    def _on_audio_complete(self, item: QueuedAudio) -> None:
        """Callback when audio playback completes."""
        # Trigger speak complete callbacks
        for callback in self._on_speak_complete_callbacks:
            try:
                callback(item.request)
            except Exception as e:
                logger.error(f"Speak complete callback error: {e}")

        # Check if queue is empty
        if self._audio_output.queue_size == 0:
            self._state = VoiceOutputState.IDLE
            self._emit_event("idle")

    def _on_audio_interrupt(self, item: QueuedAudio) -> None:
        """Callback when audio playback is interrupted."""
        self._state = VoiceOutputState.INTERRUPTED

    def on_event(self, callback: Callable[[VoiceOutputEvent], None]) -> None:
        """
        Register callback for all events.

        Args:
            callback: Function to call on events
        """
        self._event_callbacks.append(callback)

    def on_speak_start(self, callback: Callable[[SpeechRequest], None]) -> None:
        """
        Register callback for when speech starts.

        Args:
            callback: Function to call when speech starts
        """
        self._on_speak_start_callbacks.append(callback)

    def on_speak_complete(self, callback: Callable[[SpeechRequest], None]) -> None:
        """
        Register callback for when speech completes.

        Args:
            callback: Function to call when speech completes
        """
        self._on_speak_complete_callbacks.append(callback)

    def on_interrupt(self, callback: Callable[[], None]) -> None:
        """
        Register callback for speech interruption.

        Args:
            callback: Function to call when speech is interrupted
        """
        self._on_interrupt_callbacks.append(callback)

    def _emit_event(self, event_type: str, data: dict | None = None) -> None:
        """Emit an event to all registered callbacks."""
        import time

        event = VoiceOutputEvent(
            event_type=event_type,
            data=data or {},
            timestamp=time.time(),
        )

        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    @property
    def state(self) -> VoiceOutputState:
        """Get current pipeline state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._is_running

    @property
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self._state == VoiceOutputState.SPEAKING

    @property
    def is_idle(self) -> bool:
        """Check if idle (not speaking)."""
        return self._state == VoiceOutputState.IDLE

    @property
    def current_voice(self) -> VoiceProfile | None:
        """Get current voice profile."""
        return self._tts_provider.current_voice if self._tts_provider else None

    @property
    def current_mode(self) -> SpeechMode:
        """Get current speech mode."""
        return self._current_mode

    async def __aenter__(self) -> VoiceOutputManager:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Async context manager exit."""
        await self.stop()


async def create_voice_output_manager(
    voice_config: VoiceConfig | None = None,
) -> VoiceOutputManager:
    """
    Factory function to create a configured voice output manager.

    Args:
        voice_config: Voice configuration from SAIL config

    Returns:
        Configured VoiceOutputManager
    """
    return VoiceOutputManager(voice_config=voice_config)

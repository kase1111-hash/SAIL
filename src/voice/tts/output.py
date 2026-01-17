"""
SAIL Audio Output Manager

Manages audio output with priority queuing, interruptible speech,
and the respect protocol for immediate stop command handling.
"""

from __future__ import annotations

import logging
import queue
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from src.voice.tts.base import SpeechPriority, SpeechRequest

logger = logging.getLogger(__name__)


class OutputState(str, Enum):
    """State of audio output."""

    IDLE = "idle"
    PLAYING = "playing"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class QueuedAudio:
    """Audio item in the output queue."""

    audio_data: np.ndarray
    sample_rate: int
    request: SpeechRequest
    position: int = 0               # Current playback position
    queue_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def remaining_samples(self) -> int:
        """Get remaining samples to play."""
        return len(self.audio_data) - self.position

    @property
    def is_complete(self) -> bool:
        """Check if playback is complete."""
        return self.position >= len(self.audio_data)


class PriorityAudioQueue:
    """
    Thread-safe priority queue for audio items.

    Higher priority items are played first. Items with the same
    priority are played in FIFO order.
    """

    def __init__(self, maxsize: int = 100) -> None:
        """
        Initialize priority queue.

        Args:
            maxsize: Maximum queue size
        """
        self._queue: queue.PriorityQueue[tuple[int, int, QueuedAudio]] = queue.PriorityQueue(
            maxsize=maxsize
        )
        self._counter = 0
        self._lock = threading.Lock()

    def put(self, item: QueuedAudio) -> None:
        """
        Add item to queue.

        Args:
            item: QueuedAudio to add
        """
        with self._lock:
            # Priority is negative so higher values come first
            priority = -item.request.priority.value
            self._queue.put((priority, self._counter, item))
            self._counter += 1

    def get(self, timeout: float | None = None) -> QueuedAudio | None:
        """
        Get highest priority item from queue.

        Args:
            timeout: Maximum time to wait

        Returns:
            QueuedAudio or None if timeout
        """
        try:
            _, _, item = self._queue.get(timeout=timeout)
            return item
        except queue.Empty:
            return None

    def peek(self) -> QueuedAudio | None:
        """
        Peek at highest priority item without removing.

        Returns:
            QueuedAudio or None if empty
        """
        with self._lock:
            if self._queue.empty():
                return None
            # Get and put back
            priority, counter, item = self._queue.get_nowait()
            self._queue.put((priority, counter, item))
            return item

    def clear(self) -> list[QueuedAudio]:
        """
        Clear all items from queue.

        Returns:
            List of cleared items
        """
        items = []
        with self._lock:
            while not self._queue.empty():
                try:
                    _, _, item = self._queue.get_nowait()
                    items.append(item)
                except queue.Empty:
                    break
        return items

    def clear_below_priority(self, priority: SpeechPriority) -> list[QueuedAudio]:
        """
        Clear items below a certain priority.

        Args:
            priority: Minimum priority to keep

        Returns:
            List of cleared items
        """
        cleared = []
        kept = []

        with self._lock:
            while not self._queue.empty():
                try:
                    p, c, item = self._queue.get_nowait()
                    if item.request.priority.value >= priority.value:
                        kept.append((p, c, item))
                    else:
                        cleared.append(item)
                except queue.Empty:
                    break

            # Put back kept items
            for entry in kept:
                self._queue.put(entry)

        return cleared

    @property
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()

    @property
    def size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()


class AudioOutputManager:
    """
    Manages audio playback with priority queuing and interruption support.

    Implements SAIL's respect protocol by immediately stopping speech
    when a stop command is received.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        output_device: int | None = None,
        chunk_size: int = 1024,
    ) -> None:
        """
        Initialize audio output manager.

        Args:
            sample_rate: Default sample rate for audio
            output_device: Audio output device index
            chunk_size: Audio chunk size for playback
        """
        self.sample_rate = sample_rate
        self.output_device = output_device
        self.chunk_size = chunk_size

        # State
        self._state = OutputState.IDLE
        self._current_item: QueuedAudio | None = None
        self._is_running = False
        self._stop_requested = False
        self._interrupt_requested = False

        # Queue
        self._queue = PriorityAudioQueue()

        # Threading
        self._playback_thread: threading.Thread | None = None
        self._playback_lock = threading.Lock()

        # Stream
        self._stream = None

        # Callbacks
        self._on_start_callbacks: list[Callable[[QueuedAudio], None]] = []
        self._on_complete_callbacks: list[Callable[[QueuedAudio], None]] = []
        self._on_interrupt_callbacks: list[Callable[[QueuedAudio], None]] = []

    def start(self) -> None:
        """Start the audio output manager."""
        if self._is_running:
            return

        self._is_running = True
        self._stop_requested = False
        self._state = OutputState.IDLE

        # Start playback thread
        self._playback_thread = threading.Thread(
            target=self._playback_loop,
            daemon=True,
        )
        self._playback_thread.start()

        logger.info("Audio output manager started")

    def stop(self) -> None:
        """Stop the audio output manager."""
        if not self._is_running:
            return

        self._stop_requested = True
        self._is_running = False

        # Wait for thread to finish
        if self._playback_thread is not None:
            self._playback_thread.join(timeout=2.0)

        # Clear queue
        self._queue.clear()

        # Close stream
        self._close_stream()

        self._state = OutputState.STOPPED
        logger.info("Audio output manager stopped")

    def queue_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        request: SpeechRequest,
    ) -> str:
        """
        Queue audio for playback.

        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            request: Speech request with priority and settings

        Returns:
            Queue ID for the audio item
        """
        item = QueuedAudio(
            audio_data=audio_data,
            sample_rate=sample_rate,
            request=request,
        )
        self._queue.put(item)
        logger.debug(f"Queued audio: {item.queue_id} (priority: {request.priority})")
        return item.queue_id

    def interrupt(self, clear_queue: bool = True) -> None:
        """
        Interrupt current playback.

        Implements the respect protocol - immediately stops speech.

        Args:
            clear_queue: Whether to clear pending queue items
        """
        logger.info("Interrupt requested - honoring respect protocol")

        with self._playback_lock:
            self._interrupt_requested = True

            # Trigger callback for current item
            if self._current_item is not None:
                for callback in self._on_interrupt_callbacks:
                    try:
                        callback(self._current_item)
                    except Exception as e:
                        logger.error(f"Interrupt callback error: {e}")

            if clear_queue:
                # Keep only critical priority items
                self._queue.clear_below_priority(SpeechPriority.CRITICAL)

    def pause(self) -> None:
        """Pause current playback."""
        if self._state == OutputState.PLAYING:
            self._state = OutputState.PAUSED
            logger.debug("Playback paused")

    def resume(self) -> None:
        """Resume paused playback."""
        if self._state == OutputState.PAUSED:
            self._state = OutputState.PLAYING
            logger.debug("Playback resumed")

    def _playback_loop(self) -> None:
        """Main playback loop running in separate thread."""
        while self._is_running and not self._stop_requested:
            # Check for interrupt
            if self._interrupt_requested:
                with self._playback_lock:
                    self._interrupt_requested = False
                    self._current_item = None
                    self._close_stream()
                    self._state = OutputState.IDLE
                continue

            # Get next item from queue
            if self._current_item is None or self._current_item.is_complete:
                item = self._queue.get(timeout=0.1)
                if item is None:
                    self._state = OutputState.IDLE
                    continue

                self._current_item = item
                self._state = OutputState.PLAYING

                # Trigger start callback
                for callback in self._on_start_callbacks:
                    try:
                        callback(item)
                    except Exception as e:
                        logger.error(f"Start callback error: {e}")

            # Play current item
            if self._state == OutputState.PLAYING and self._current_item is not None:
                self._play_chunk()

            # Check for completion
            if self._current_item is not None and self._current_item.is_complete:
                # Trigger complete callback
                for callback in self._on_complete_callbacks:
                    try:
                        callback(self._current_item)
                    except Exception as e:
                        logger.error(f"Complete callback error: {e}")

                self._current_item = None

    def _play_chunk(self) -> None:
        """Play a chunk of the current audio item."""
        if self._current_item is None:
            return

        item = self._current_item

        # Check for interrupt or stop
        if self._interrupt_requested or self._stop_requested:
            return

        # Skip if paused
        if self._state == OutputState.PAUSED:
            import time
            time.sleep(0.05)
            return

        # Check if interruptible and interrupt requested
        if self._interrupt_requested and item.request.interruptible:
            return

        # Ensure stream is open
        if self._stream is None:
            self._open_stream(item.sample_rate)

        # Get chunk to play
        start = item.position
        end = min(start + self.chunk_size, len(item.audio_data))
        chunk = item.audio_data[start:end]

        # Play chunk
        try:
            if self._stream is not None:
                # Convert to float32 for sounddevice
                chunk_float = chunk.astype(np.float32) / 32768.0
                self._stream.write(chunk_float)
        except Exception as e:
            logger.error(f"Playback error: {e}")

        # Update position
        item.position = end

    def _open_stream(self, sample_rate: int) -> None:
        """Open audio output stream."""
        try:
            import sounddevice as sd

            self._stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                device=self.output_device,
                blocksize=self.chunk_size,
            )
            self._stream.start()
        except Exception as e:
            logger.error(f"Failed to open audio stream: {e}")
            self._stream = None

    def _close_stream(self) -> None:
        """Close audio output stream."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def on_start(self, callback: Callable[[QueuedAudio], None]) -> None:
        """
        Register callback for playback start.

        Args:
            callback: Function to call when playback starts
        """
        self._on_start_callbacks.append(callback)

    def on_complete(self, callback: Callable[[QueuedAudio], None]) -> None:
        """
        Register callback for playback completion.

        Args:
            callback: Function to call when playback completes
        """
        self._on_complete_callbacks.append(callback)

    def on_interrupt(self, callback: Callable[[QueuedAudio], None]) -> None:
        """
        Register callback for playback interruption.

        Args:
            callback: Function to call when playback is interrupted
        """
        self._on_interrupt_callbacks.append(callback)

    @property
    def state(self) -> OutputState:
        """Get current output state."""
        return self._state

    @property
    def is_playing(self) -> bool:
        """Check if currently playing audio."""
        return self._state == OutputState.PLAYING

    @property
    def queue_size(self) -> int:
        """Get number of items in queue."""
        return self._queue.size

    @property
    def current_item(self) -> QueuedAudio | None:
        """Get currently playing item."""
        return self._current_item

    def __enter__(self) -> AudioOutputManager:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()

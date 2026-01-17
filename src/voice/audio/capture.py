"""
SAIL Audio Capture Module

Provides audio capture capabilities using sounddevice for microphone input.
Supports device selection, sample rate configuration, and streaming audio chunks.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import sounddevice as sd


class AudioCaptureError(Exception):
    """Base exception for audio capture errors."""


class DeviceNotFoundError(AudioCaptureError):
    """Raised when the specified audio device is not found."""


class CaptureNotStartedError(AudioCaptureError):
    """Raised when trying to read from a capture that hasn't been started."""


class AudioFormat(str, Enum):
    """Supported audio formats."""

    INT16 = "int16"
    FLOAT32 = "float32"


@dataclass
class AudioDeviceInfo:
    """Information about an audio input device."""

    index: int
    name: str
    channels: int
    default_sample_rate: float
    is_default: bool = False

    def __str__(self) -> str:
        default_marker = " (default)" if self.is_default else ""
        return f"[{self.index}] {self.name} - {self.channels}ch @ {self.default_sample_rate}Hz{default_marker}"


@dataclass
class AudioChunk:
    """A chunk of audio data from the capture stream."""

    data: np.ndarray
    sample_rate: int
    channels: int
    timestamp: float
    duration_ms: float = field(init=False)

    def __post_init__(self) -> None:
        """Calculate duration from data length."""
        samples = len(self.data) if self.channels == 1 else len(self.data) // self.channels
        self.duration_ms = (samples / self.sample_rate) * 1000

    def to_int16(self) -> np.ndarray:
        """Convert audio data to int16 format."""
        if self.data.dtype == np.int16:
            return self.data
        # Convert from float32 to int16
        return (self.data * 32767).astype(np.int16)

    def to_float32(self) -> np.ndarray:
        """Convert audio data to float32 format."""
        if self.data.dtype == np.float32:
            return self.data
        # Convert from int16 to float32
        return self.data.astype(np.float32) / 32767.0

    def to_bytes(self) -> bytes:
        """Convert audio data to bytes (int16 format)."""
        return self.to_int16().tobytes()


class AudioCapture:
    """
    Audio capture manager using sounddevice.

    Provides both synchronous and asynchronous interfaces for capturing
    audio from microphone input devices.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration_ms: int = 30,
        device: int | str | None = None,
        audio_format: AudioFormat = AudioFormat.INT16,
    ) -> None:
        """
        Initialize audio capture.

        Args:
            sample_rate: Audio sample rate in Hz (default: 16000 for speech)
            channels: Number of audio channels (default: 1 for mono)
            chunk_duration_ms: Duration of each audio chunk in milliseconds
            device: Device index, name, or None for default device
            audio_format: Audio format for capture (int16 or float32)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration_ms = chunk_duration_ms
        self.audio_format = audio_format

        # Calculate frames per chunk
        self.frames_per_chunk = int(sample_rate * chunk_duration_ms / 1000)

        # Device selection
        self._device_id = device
        self._resolved_device: int | None = None

        # Capture state
        self._stream: sd.InputStream | None = None
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._is_capturing = False
        self._capture_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._start_time: float = 0.0

        # Callbacks
        self._on_audio_callback: Callable[[AudioChunk], None] | None = None

    @staticmethod
    def list_devices() -> list[AudioDeviceInfo]:
        """
        List available audio input devices.

        Returns:
            List of AudioDeviceInfo objects for all input devices
        """
        import sounddevice as sd

        devices = []
        default_device = sd.query_devices(kind="input")
        default_name = default_device["name"] if default_device else None

        for i, device in enumerate(sd.query_devices()):
            if device["max_input_channels"] > 0:
                devices.append(
                    AudioDeviceInfo(
                        index=i,
                        name=device["name"],
                        channels=device["max_input_channels"],
                        default_sample_rate=device["default_samplerate"],
                        is_default=(device["name"] == default_name),
                    )
                )
        return devices

    @staticmethod
    def get_default_device() -> AudioDeviceInfo | None:
        """
        Get the default audio input device.

        Returns:
            AudioDeviceInfo for the default device, or None if no devices available
        """
        import sounddevice as sd

        try:
            device = sd.query_devices(kind="input")
            if device:
                return AudioDeviceInfo(
                    index=int(sd.default.device[0]),  # type: ignore
                    name=device["name"],
                    channels=device["max_input_channels"],
                    default_sample_rate=device["default_samplerate"],
                    is_default=True,
                )
        except sd.PortAudioError:
            pass
        return None

    def _resolve_device(self) -> int | None:
        """Resolve device ID from index, name, or default."""
        import sounddevice as sd

        if self._device_id is None:
            return None  # Use default

        if isinstance(self._device_id, int):
            # Validate device exists and has input channels
            devices = sd.query_devices()
            if self._device_id >= len(devices):
                raise DeviceNotFoundError(f"Device index {self._device_id} not found")
            if devices[self._device_id]["max_input_channels"] == 0:
                raise DeviceNotFoundError(
                    f"Device {self._device_id} has no input channels"
                )
            return self._device_id

        # Search by name
        for i, device in enumerate(sd.query_devices()):
            if self._device_id.lower() in device["name"].lower():
                if device["max_input_channels"] > 0:
                    return i

        raise DeviceNotFoundError(f"Device '{self._device_id}' not found")

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,  # noqa: ARG002
        status: sd.CallbackFlags,  # noqa: ARG002
    ) -> None:
        """Callback for sounddevice to receive audio data."""
        # Make a copy since indata is only valid during this callback
        self._audio_queue.put(indata.copy())

    def start(self) -> None:
        """
        Start audio capture.

        Raises:
            AudioCaptureError: If capture fails to start
            DeviceNotFoundError: If the specified device is not found
        """
        import sounddevice as sd

        if self._is_capturing:
            return

        try:
            self._resolved_device = self._resolve_device()

            # Determine dtype based on format
            dtype = "int16" if self.audio_format == AudioFormat.INT16 else "float32"

            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=dtype,
                blocksize=self.frames_per_chunk,
                device=self._resolved_device,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._is_capturing = True
            self._stop_event.clear()
            self._start_time = asyncio.get_event_loop().time()

        except sd.PortAudioError as e:
            raise AudioCaptureError(f"Failed to start audio capture: {e}") from e

    def stop(self) -> None:
        """Stop audio capture and release resources."""
        if not self._is_capturing:
            return

        self._stop_event.set()
        self._is_capturing = False

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # Clear the queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def read(self, timeout: float | None = None) -> AudioChunk | None:
        """
        Read the next audio chunk synchronously.

        Args:
            timeout: Maximum time to wait for audio data (None for blocking)

        Returns:
            AudioChunk or None if timeout occurred

        Raises:
            CaptureNotStartedError: If capture hasn't been started
        """
        if not self._is_capturing:
            raise CaptureNotStartedError("Audio capture not started")

        try:
            data = self._audio_queue.get(timeout=timeout)
            timestamp = asyncio.get_event_loop().time() - self._start_time
            return AudioChunk(
                data=data.flatten() if self.channels == 1 else data,
                sample_rate=self.sample_rate,
                channels=self.channels,
                timestamp=timestamp,
            )
        except queue.Empty:
            return None

    async def read_async(self, timeout: float | None = None) -> AudioChunk | None:
        """
        Read the next audio chunk asynchronously.

        Args:
            timeout: Maximum time to wait for audio data

        Returns:
            AudioChunk or None if timeout occurred

        Raises:
            CaptureNotStartedError: If capture hasn't been started
        """
        if not self._is_capturing:
            raise CaptureNotStartedError("Audio capture not started")

        loop = asyncio.get_event_loop()
        try:
            data = await asyncio.wait_for(
                loop.run_in_executor(None, self._audio_queue.get),
                timeout=timeout,
            )
            timestamp = loop.time() - self._start_time
            return AudioChunk(
                data=data.flatten() if self.channels == 1 else data,
                sample_rate=self.sample_rate,
                channels=self.channels,
                timestamp=timestamp,
            )
        except TimeoutError:
            return None

    async def stream(self) -> AsyncIterator[AudioChunk]:
        """
        Async generator that yields audio chunks continuously.

        Yields:
            AudioChunk objects as they become available

        Raises:
            CaptureNotStartedError: If capture hasn't been started
        """
        if not self._is_capturing:
            raise CaptureNotStartedError("Audio capture not started")

        while self._is_capturing:
            chunk = await self.read_async(timeout=0.1)
            if chunk:
                yield chunk

    @property
    def is_capturing(self) -> bool:
        """Check if capture is currently active."""
        return self._is_capturing

    @property
    def device_info(self) -> AudioDeviceInfo | None:
        """Get information about the current capture device."""
        import sounddevice as sd

        if self._resolved_device is None:
            return self.get_default_device()

        device = sd.query_devices(self._resolved_device)
        return AudioDeviceInfo(
            index=self._resolved_device,
            name=device["name"],
            channels=device["max_input_channels"],
            default_sample_rate=device["default_samplerate"],
        )

    def __enter__(self) -> AudioCapture:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()

    async def __aenter__(self) -> AudioCapture:
        """Async context manager entry."""
        self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        self.stop()

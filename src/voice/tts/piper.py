"""
SAIL Piper TTS Integration

Provides text-to-speech using the Piper TTS engine.
Piper is a fast, local neural text-to-speech system.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.voice.tts.base import (
    SpeechResult,
    SpeechState,
    TTSConfig,
    TTSEngine,
    TTSProvider,
    VoiceProfile,
    preprocess_text_for_speech,
    split_text_into_sentences,
)

logger = logging.getLogger(__name__)


class PiperError(Exception):
    """Exception raised for Piper-related errors."""


@dataclass
class PiperConfig(TTSConfig):
    """Configuration specific to Piper TTS."""

    # Piper-specific settings
    model_path: Path | None = None      # Path to voice model file
    config_path: Path | None = None     # Path to model config JSON
    use_cuda: bool = False              # Use CUDA acceleration
    speaker_id: int | None = None       # Speaker ID for multi-speaker models
    length_scale: float = 1.0           # Controls speaking speed
    noise_scale: float = 0.667          # Variability in speech
    noise_w: float = 0.8                # Phoneme width noise

    def __post_init__(self) -> None:
        """Set engine type."""
        self.engine = TTSEngine.PIPER


class PiperProvider(TTSProvider):
    """
    Text-to-speech provider using Piper TTS.

    Piper is a fast, local neural text-to-speech system that
    supports many languages and voices.
    """

    # Default voice models directory
    DEFAULT_MODELS_DIR = Path.home() / ".local" / "share" / "piper" / "voices"

    # Known voice models with their characteristics
    KNOWN_VOICES = {
        "en_US-lessac-medium": VoiceProfile(
            name="Lessac",
            model_id="en_US-lessac-medium",
            language="en-US",
            gender="female",
            description="Clear American English female voice",
        ),
        "en_US-amy-medium": VoiceProfile(
            name="Amy",
            model_id="en_US-amy-medium",
            language="en-US",
            gender="female",
            description="Warm American English female voice",
        ),
        "en_US-ryan-medium": VoiceProfile(
            name="Ryan",
            model_id="en_US-ryan-medium",
            language="en-US",
            gender="male",
            description="Clear American English male voice",
        ),
        "en_GB-alba-medium": VoiceProfile(
            name="Alba",
            model_id="en_GB-alba-medium",
            language="en-GB",
            gender="female",
            description="British English female voice",
        ),
    }

    def __init__(self, config: PiperConfig | None = None) -> None:
        """
        Initialize Piper provider.

        Args:
            config: Piper configuration, uses defaults if None
        """
        super().__init__(config or PiperConfig())
        self.config: PiperConfig = self.config  # Type narrowing

        self._piper = None
        self._synthesize_fn = None

    async def load_model(self, voice: str | None = None) -> None:
        """
        Load the Piper model.

        Args:
            voice: Voice model to load, uses config default if None
        """
        if self._is_loaded:
            return

        self._state = SpeechState.SYNTHESIZING
        voice = voice or self.config.voice

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model_sync, voice)
            self._is_loaded = True
            self._state = SpeechState.IDLE

            # Set current voice profile
            self._current_voice = self.KNOWN_VOICES.get(
                voice,
                VoiceProfile(name=voice, model_id=voice),
            )

            logger.info(f"Loaded Piper voice: {voice}")

        except Exception as e:
            self._state = SpeechState.ERROR
            raise PiperError(f"Failed to load Piper model: {e}") from e

    def _load_model_sync(self, voice: str) -> None:
        """Synchronous model loading."""
        try:
            from piper import PiperVoice

            # Try to find the model file
            model_path = self._resolve_model_path(voice)

            if model_path is None:
                # Try to download the model
                model_path = self._download_model(voice)

            if model_path is None:
                raise PiperError(f"Could not find or download voice model: {voice}")

            # Load the model
            self._piper = PiperVoice.load(str(model_path), use_cuda=self.config.use_cuda)

        except ImportError as e:
            raise PiperError(
                "Piper is not installed. Install with: pip install piper-tts"
            ) from e

    def _resolve_model_path(self, voice: str) -> Path | None:
        """
        Resolve the path to a voice model.

        Args:
            voice: Voice identifier

        Returns:
            Path to model file or None
        """
        # Check if config provides explicit path
        if self.config.model_path and self.config.model_path.exists():
            return self.config.model_path

        # Check default models directory
        model_dir = self.DEFAULT_MODELS_DIR / voice
        if model_dir.exists():
            # Look for .onnx file
            for f in model_dir.glob("*.onnx"):
                return f

        # Check if voice is a direct path
        voice_path = Path(voice)
        if voice_path.exists():
            return voice_path

        return None

    def _download_model(self, voice: str) -> Path | None:
        """
        Download a voice model if not available locally.

        Args:
            voice: Voice identifier

        Returns:
            Path to downloaded model or None
        """
        try:
            # Create models directory
            model_dir = self.DEFAULT_MODELS_DIR / voice
            model_dir.mkdir(parents=True, exist_ok=True)

            # Use piper's download utility if available
            try:
                from piper.download import ensure_voice_exists, get_voices

                # This will download the voice if needed
                model_path = ensure_voice_exists(
                    voice,
                    data_dir=str(self.DEFAULT_MODELS_DIR),
                )
                return Path(model_path)
            except ImportError:
                logger.warning("Piper download utilities not available")
                return None

        except Exception as e:
            logger.error(f"Failed to download voice model: {e}")
            return None

    async def unload_model(self) -> None:
        """Unload the model to free resources."""
        self._piper = None
        self._synthesize_fn = None
        self._is_loaded = False
        self._current_voice = None
        self._state = SpeechState.IDLE

    async def synthesize(self, text: str) -> SpeechResult:
        """
        Synthesize text to audio.

        Args:
            text: Text to synthesize

        Returns:
            SpeechResult with audio data
        """
        if not self._is_loaded:
            await self.load_model()

        self._state = SpeechState.SYNTHESIZING

        try:
            # Preprocess text
            text = preprocess_text_for_speech(text)

            # Run synthesis in executor
            loop = asyncio.get_event_loop()
            audio_data, sample_rate = await loop.run_in_executor(
                None,
                self._synthesize_sync,
                text,
            )

            duration = len(audio_data) / sample_rate

            self._state = SpeechState.COMPLETE
            return SpeechResult(
                audio_data=audio_data,
                sample_rate=sample_rate,
                duration_seconds=duration,
            )

        except Exception as e:
            self._state = SpeechState.ERROR
            logger.error(f"Synthesis failed: {e}")
            return SpeechResult(error=str(e))

    def _synthesize_sync(self, text: str) -> tuple[np.ndarray, int]:
        """
        Synchronous synthesis.

        Args:
            text: Text to synthesize

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if self._piper is None:
            raise PiperError("Model not loaded")

        # Create in-memory WAV file
        wav_buffer = io.BytesIO()

        # Synthesize with Piper
        with wave.open(wav_buffer, "wb") as wav_file:
            self._piper.synthesize(
                text,
                wav_file,
                speaker_id=self.config.speaker_id,
                length_scale=self.config.length_scale / self.config.rate,
                noise_scale=self.config.noise_scale,
                noise_w=self.config.noise_w,
            )

        # Read the WAV data
        wav_buffer.seek(0)
        with wave.open(wav_buffer, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            audio_bytes = wav_file.readframes(n_frames)

            # Convert to numpy array
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)

        return audio_data, sample_rate

    async def synthesize_stream(
        self,
        text: str,
    ) -> AsyncIterator[np.ndarray]:
        """
        Synthesize text to audio with streaming output.

        Splits text into sentences and synthesizes each one.

        Args:
            text: Text to synthesize

        Yields:
            Audio chunks as numpy arrays
        """
        if not self._is_loaded:
            await self.load_model()

        self._state = SpeechState.SYNTHESIZING

        try:
            # Split text into sentences
            sentences = split_text_into_sentences(text)

            for sentence in sentences:
                # Synthesize each sentence
                result = await self.synthesize(sentence)

                if result.is_success and result.audio_data is not None:
                    yield result.audio_data

                    # Add pause between sentences
                    pause_samples = int(
                        result.sample_rate * self.config.sentence_pause_ms / 1000
                    )
                    yield np.zeros(pause_samples, dtype=np.int16)

            self._state = SpeechState.COMPLETE

        except Exception as e:
            self._state = SpeechState.ERROR
            raise PiperError(f"Streaming synthesis failed: {e}") from e

    async def list_voices(self) -> list[VoiceProfile]:
        """
        List available voices.

        Returns:
            List of VoiceProfile objects
        """
        voices = list(self.KNOWN_VOICES.values())

        # Also check local models directory
        if self.DEFAULT_MODELS_DIR.exists():
            for model_dir in self.DEFAULT_MODELS_DIR.iterdir():
                if model_dir.is_dir():
                    voice_id = model_dir.name
                    if voice_id not in self.KNOWN_VOICES:
                        # Try to load voice info from config
                        config_path = model_dir / f"{voice_id}.onnx.json"
                        if config_path.exists():
                            try:
                                with open(config_path) as f:
                                    config_data = json.load(f)
                                    voices.append(VoiceProfile(
                                        name=config_data.get("name", voice_id),
                                        model_id=voice_id,
                                        language=config_data.get("language", {}).get("code", "en"),
                                    ))
                            except Exception:
                                voices.append(VoiceProfile(name=voice_id, model_id=voice_id))
                        else:
                            voices.append(VoiceProfile(name=voice_id, model_id=voice_id))

        return voices

    async def get_voice_info(self, voice_id: str) -> VoiceProfile | None:
        """
        Get information about a specific voice.

        Args:
            voice_id: Voice identifier

        Returns:
            VoiceProfile or None if not found
        """
        # Check known voices
        if voice_id in self.KNOWN_VOICES:
            return self.KNOWN_VOICES[voice_id]

        # Check local models
        model_dir = self.DEFAULT_MODELS_DIR / voice_id
        if model_dir.exists():
            config_path = model_dir / f"{voice_id}.onnx.json"
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config_data = json.load(f)
                        return VoiceProfile(
                            name=config_data.get("name", voice_id),
                            model_id=voice_id,
                            language=config_data.get("language", {}).get("code", "en"),
                        )
                except Exception:
                    pass
            return VoiceProfile(name=voice_id, model_id=voice_id)

        return None


def create_tts_provider(config: TTSConfig | None = None) -> TTSProvider:
    """
    Factory function to create the appropriate TTS provider.

    Args:
        config: TTS configuration

    Returns:
        TTSProvider instance
    """
    config = config or TTSConfig()

    if config.engine == TTSEngine.PIPER:
        return PiperProvider(
            PiperConfig(**{k: v for k, v in config.__dict__.items()})
        )
    else:
        # Default to Piper
        return PiperProvider(
            PiperConfig(**{k: v for k, v in config.__dict__.items()})
        )

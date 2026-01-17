"""
Audio environment sensor implementation.

Provides ambient sound classification, stress detection in voice,
and threat cue identification for audio-based situational awareness.
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional
from math import log10
import numpy as np

from .base import (
    Sensor,
    SensorType,
    SensorStatus,
    SensorReading,
    SensorConfig,
    SensorEvent,
    SensorEventType,
    AudioEnvironment,
    AudioReading,
    StressLevel,
    Confidence,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Audio Data Types
# ============================================================================


@dataclass
class AudioSample:
    """Raw audio sample data."""

    data: np.ndarray  # Audio waveform
    sample_rate: int
    channels: int = 1
    duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if self.duration_seconds == 0.0 and len(self.data) > 0:
            self.duration_seconds = len(self.data) / self.sample_rate


@dataclass
class AudioFeatures:
    """Extracted audio features."""

    rms_energy: float  # Root mean square energy
    peak_amplitude: float
    zero_crossing_rate: float
    spectral_centroid: Optional[float] = None
    spectral_rolloff: Optional[float] = None
    noise_level_db: float = 0.0
    has_speech: bool = False
    has_music: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rms_energy": self.rms_energy,
            "peak_amplitude": self.peak_amplitude,
            "zero_crossing_rate": self.zero_crossing_rate,
            "spectral_centroid": self.spectral_centroid,
            "spectral_rolloff": self.spectral_rolloff,
            "noise_level_db": self.noise_level_db,
            "has_speech": self.has_speech,
            "has_music": self.has_music,
        }


@dataclass
class AudioAnalysis:
    """Complete audio environment analysis."""

    environment: AudioEnvironment
    confidence: Confidence
    features: AudioFeatures
    stress_indicators: StressLevel = StressLevel.UNKNOWN
    threat_cues: list[str] = field(default_factory=list)
    speech_detected: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "environment": self.environment.value,
            "confidence": self.confidence.value,
            "features": self.features.to_dict(),
            "stress_indicators": self.stress_indicators.value,
            "threat_cues": self.threat_cues,
            "speech_detected": self.speech_detected,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# Threat Cue Definitions
# ============================================================================


# Audio patterns that might indicate threats
THREAT_CUE_PATTERNS = {
    "raised_voices": {
        "description": "Elevated voice volume detected",
        "energy_threshold": 0.5,  # Normalized RMS
    },
    "screaming": {
        "description": "High-pitched loud vocalizations",
        "energy_threshold": 0.7,
        "frequency_range": (1000, 4000),  # Hz
    },
    "glass_breaking": {
        "description": "Sharp transient sounds",
        "transient_threshold": 0.8,
    },
    "alarm": {
        "description": "Alarm-like repetitive tones",
        "frequency_range": (500, 3000),
    },
    "siren": {
        "description": "Emergency vehicle siren",
        "frequency_pattern": "sweeping",
    },
    "crash": {
        "description": "Loud impact sound",
        "energy_threshold": 0.9,
        "duration": "brief",
    },
}


# ============================================================================
# Audio Feature Extractor
# ============================================================================


class AudioFeatureExtractor:
    """Extracts features from audio samples."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def extract(self, sample: AudioSample) -> AudioFeatures:
        """Extract features from audio sample."""
        data = sample.data.astype(np.float32)

        # Normalize to [-1, 1]
        if data.max() > 1.0 or data.min() < -1.0:
            max_val = max(abs(data.max()), abs(data.min()))
            if max_val > 0:
                data = data / max_val

        # RMS energy
        rms_energy = float(np.sqrt(np.mean(data ** 2)))

        # Peak amplitude
        peak_amplitude = float(np.max(np.abs(data)))

        # Zero crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(data))))
        zcr = zero_crossings / len(data) if len(data) > 0 else 0.0

        # Convert RMS to dB
        noise_level_db = 20 * log10(rms_energy + 1e-10)

        # Simple speech detection based on energy and ZCR
        # Speech typically has moderate energy and moderate ZCR
        has_speech = (
            0.01 < rms_energy < 0.5
            and 0.02 < zcr < 0.15
        )

        # Simple music detection (more regular patterns, sustained energy)
        has_music = (
            0.02 < rms_energy < 0.4
            and zcr < 0.08
        )

        return AudioFeatures(
            rms_energy=rms_energy,
            peak_amplitude=peak_amplitude,
            zero_crossing_rate=zcr,
            noise_level_db=noise_level_db,
            has_speech=has_speech,
            has_music=has_music,
        )


# ============================================================================
# Environment Classifier
# ============================================================================


class AudioEnvironmentClassifier:
    """Classifies audio environment from features."""

    # Thresholds for environment classification
    QUIET_THRESHOLD = -50  # dB
    INDOOR_NORMAL_RANGE = (-50, -30)  # dB
    LOUD_THRESHOLD = -10  # dB

    def __init__(self):
        self._feature_history: deque[AudioFeatures] = deque(maxlen=20)

    def add_features(self, features: AudioFeatures) -> None:
        """Add features to history."""
        self._feature_history.append(features)

    def classify(self, features: AudioFeatures) -> tuple[AudioEnvironment, Confidence]:
        """Classify audio environment."""
        self.add_features(features)

        db = features.noise_level_db
        zcr = features.zero_crossing_rate

        # Quiet environment
        if db < self.QUIET_THRESHOLD:
            return AudioEnvironment.QUIET, Confidence.HIGH

        # Very loud environment
        if db > self.LOUD_THRESHOLD:
            return AudioEnvironment.LOUD, Confidence.HIGH

        # Conversation detected
        if features.has_speech and 0.03 < zcr < 0.12:
            return AudioEnvironment.CONVERSATION, Confidence.MEDIUM

        # Indoor normal
        if self.INDOOR_NORMAL_RANGE[0] <= db <= self.INDOOR_NORMAL_RANGE[1]:
            if zcr < 0.05:
                return AudioEnvironment.INDOOR_NORMAL, Confidence.MEDIUM

        # Traffic (variable energy, moderate-high noise)
        if -30 < db < -10 and self._is_variable_energy():
            return AudioEnvironment.TRAFFIC, Confidence.LOW

        # Outdoor (higher noise floor, variable patterns)
        if -40 < db < -20:
            return AudioEnvironment.OUTDOOR, Confidence.LOW

        # Crowd (sustained moderate-high noise with speech-like patterns)
        if -25 < db < -5 and features.has_speech:
            return AudioEnvironment.CROWD, Confidence.LOW

        # Default to unknown
        return AudioEnvironment.UNKNOWN, Confidence.VERY_LOW

    def _is_variable_energy(self) -> bool:
        """Check if energy is variable (indicating outdoor/traffic)."""
        if len(self._feature_history) < 5:
            return False

        energies = [f.rms_energy for f in self._feature_history]
        variance = np.var(energies)
        return variance > 0.01


# ============================================================================
# Stress Detector
# ============================================================================


class VoiceStressDetector:
    """Detects stress indicators in voice."""

    def __init__(self):
        self._baseline_energy: Optional[float] = None
        self._baseline_zcr: Optional[float] = None
        self._readings: deque[AudioFeatures] = deque(maxlen=50)

    def update_baseline(self, features: AudioFeatures) -> None:
        """Update baseline from normal speech."""
        self._readings.append(features)

        if len(self._readings) >= 10:
            speech_readings = [f for f in self._readings if f.has_speech]
            if speech_readings:
                self._baseline_energy = np.mean([f.rms_energy for f in speech_readings])
                self._baseline_zcr = np.mean([f.zero_crossing_rate for f in speech_readings])

    def detect(self, features: AudioFeatures) -> StressLevel:
        """Detect stress level from audio features."""
        if not features.has_speech:
            return StressLevel.UNKNOWN

        self._readings.append(features)

        # If no baseline, can't determine stress
        if self._baseline_energy is None:
            return StressLevel.UNKNOWN

        # Calculate deviation from baseline
        energy_ratio = features.rms_energy / (self._baseline_energy + 1e-10)
        zcr_ratio = features.zero_crossing_rate / (self._baseline_zcr + 1e-10)

        # Stress indicators:
        # - Elevated vocal energy
        # - Higher pitch (approximated by ZCR)
        # - Faster speech rate (harder to detect without more features)

        if energy_ratio > 2.0 and zcr_ratio > 1.5:
            return StressLevel.HIGH
        elif energy_ratio > 1.5 and zcr_ratio > 1.3:
            return StressLevel.ELEVATED
        elif energy_ratio > 1.2 or zcr_ratio > 1.2:
            return StressLevel.NORMAL
        else:
            return StressLevel.LOW


# ============================================================================
# Threat Cue Detector
# ============================================================================


class ThreatCueDetector:
    """Detects potential threat cues in audio."""

    def __init__(self):
        self._recent_features: deque[AudioFeatures] = deque(maxlen=10)

    def detect(self, features: AudioFeatures) -> list[str]:
        """Detect threat cues from audio features."""
        self._recent_features.append(features)
        threats: list[str] = []

        # Raised voices detection
        if features.has_speech and features.rms_energy > 0.3:
            threats.append("raised_voices")

        # Potential screaming (high energy, high ZCR)
        if features.rms_energy > 0.5 and features.zero_crossing_rate > 0.15:
            threats.append("screaming")

        # Loud impact/crash (sudden high energy spike)
        if self._detect_sudden_spike(features):
            threats.append("crash")

        return threats

    def _detect_sudden_spike(self, features: AudioFeatures) -> bool:
        """Detect sudden energy spike."""
        if len(self._recent_features) < 3:
            return False

        recent_avg = np.mean([f.rms_energy for f in list(self._recent_features)[:-1]])
        return features.rms_energy > recent_avg * 5


# ============================================================================
# Audio Provider Abstraction
# ============================================================================


class AudioProvider:
    """Abstract audio data provider."""

    async def get_sample(self, duration_seconds: float) -> Optional[AudioSample]:
        """Get audio sample of specified duration."""
        raise NotImplementedError


class SimulatedAudioProvider(AudioProvider):
    """Simulated audio provider for testing."""

    def __init__(
        self,
        environment: AudioEnvironment = AudioEnvironment.QUIET,
        sample_rate: int = 16000,
    ):
        self.environment = environment
        self.sample_rate = sample_rate

    async def get_sample(self, duration_seconds: float) -> Optional[AudioSample]:
        """Generate simulated audio sample."""
        num_samples = int(duration_seconds * self.sample_rate)

        # Generate noise based on environment
        if self.environment == AudioEnvironment.QUIET:
            noise_level = 0.001
        elif self.environment == AudioEnvironment.INDOOR_NORMAL:
            noise_level = 0.01
        elif self.environment == AudioEnvironment.OUTDOOR:
            noise_level = 0.05
        elif self.environment == AudioEnvironment.TRAFFIC:
            noise_level = 0.1
        elif self.environment == AudioEnvironment.CROWD:
            noise_level = 0.08
        elif self.environment == AudioEnvironment.LOUD:
            noise_level = 0.3
        else:
            noise_level = 0.02

        data = np.random.randn(num_samples) * noise_level

        return AudioSample(
            data=data.astype(np.float32),
            sample_rate=self.sample_rate,
            duration_seconds=duration_seconds,
        )


class SystemAudioProvider(AudioProvider):
    """System audio provider using microphone."""

    def __init__(self, sample_rate: int = 16000, device: Optional[int] = None):
        self.sample_rate = sample_rate
        self.device = device
        self._available = False

    async def initialize(self) -> bool:
        """Initialize audio capture."""
        try:
            import sounddevice as sd
            # Check if audio device is available
            devices = sd.query_devices()
            self._available = True
            logger.info(f"Audio capture initialized, {len(devices)} devices found")
            return True
        except ImportError:
            logger.warning("sounddevice library not available")
            return False
        except Exception as e:
            logger.error(f"Audio initialization failed: {e}")
            return False

    async def get_sample(self, duration_seconds: float) -> Optional[AudioSample]:
        """Capture audio from microphone."""
        if not self._available:
            return None

        try:
            import sounddevice as sd

            num_samples = int(duration_seconds * self.sample_rate)
            recording = sd.rec(
                num_samples,
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                device=self.device,
            )
            sd.wait()

            return AudioSample(
                data=recording.flatten(),
                sample_rate=self.sample_rate,
                duration_seconds=duration_seconds,
            )

        except Exception as e:
            logger.error(f"Audio capture failed: {e}")
            return None


# ============================================================================
# Audio Sensor
# ============================================================================


class AudioSensor(Sensor):
    """Audio sensor for environment classification and stress detection."""

    def __init__(
        self,
        sensor_id: str = "audio_primary",
        config: Optional[SensorConfig] = None,
        audio_provider: Optional[AudioProvider] = None,
    ):
        super().__init__(sensor_id, SensorType.AUDIO, config)

        self.audio_provider = audio_provider or SimulatedAudioProvider()
        self.feature_extractor = AudioFeatureExtractor()
        self.environment_classifier = AudioEnvironmentClassifier()
        self.stress_detector = VoiceStressDetector()
        self.threat_detector = ThreatCueDetector()

        self._sample_duration = config.audio_sample_duration_seconds if config else 2.0
        self._stress_detection_enabled = config.stress_detection_enabled if config else True
        self._event_callbacks: list[Callable[[SensorEvent], None]] = []

    def register_event_callback(self, callback: Callable[[SensorEvent], None]) -> None:
        """Register callback for audio events."""
        self._event_callbacks.append(callback)

    async def initialize(self) -> bool:
        """Initialize the audio sensor."""
        self._status = SensorStatus.INITIALIZING

        try:
            # Initialize audio provider if it has an initialize method
            if hasattr(self.audio_provider, "initialize"):
                success = await self.audio_provider.initialize()
                if not success:
                    logger.warning("Audio provider initialization failed, using simulated data")

            self._status = SensorStatus.ACTIVE
            logger.info(f"Audio sensor {self.sensor_id} initialized")
            return True

        except Exception as e:
            self._set_error(f"Initialization failed: {e}")
            return False

    async def read(self) -> SensorReading:
        """Take an audio reading."""
        # Get audio sample
        sample = await self.audio_provider.get_sample(self._sample_duration)

        if sample is None:
            return SensorReading(
                sensor_type=self.sensor_type,
                sensor_id=self.sensor_id,
                data=None,
                confidence=Confidence.VERY_LOW,
                metadata={"error": "No audio sample"},
            )

        # Extract features
        features = self.feature_extractor.extract(sample)

        # Classify environment
        environment, confidence = self.environment_classifier.classify(features)

        # Detect stress (if speech detected and enabled)
        stress_level = StressLevel.UNKNOWN
        if self._stress_detection_enabled and features.has_speech:
            stress_level = self.stress_detector.detect(features)
            self.stress_detector.update_baseline(features)

            # Emit stress event if elevated
            if stress_level in (StressLevel.ELEVATED, StressLevel.HIGH):
                event = SensorEvent(
                    event_type=SensorEventType.STRESS_DETECTED,
                    sensor_type=SensorType.AUDIO,
                    data={"stress_level": stress_level, "features": features},
                )
                self._notify_event(event)

        # Detect threat cues
        threat_cues = self.threat_detector.detect(features)

        # Emit threat event if detected
        if threat_cues:
            event = SensorEvent(
                event_type=SensorEventType.THREAT_DETECTED,
                sensor_type=SensorType.AUDIO,
                data={"threat_cues": threat_cues, "features": features},
            )
            self._notify_event(event)

        # Create analysis
        analysis = AudioAnalysis(
            environment=environment,
            confidence=confidence,
            features=features,
            stress_indicators=stress_level,
            threat_cues=threat_cues,
            speech_detected=features.has_speech,
        )

        # Create audio reading
        audio_reading = AudioReading(
            environment=environment,
            confidence=confidence,
            noise_level_db=features.noise_level_db,
            speech_detected=features.has_speech,
            stress_indicators=stress_level,
            threat_cues=threat_cues,
        )

        reading = SensorReading(
            sensor_type=self.sensor_type,
            sensor_id=self.sensor_id,
            data=analysis,
            confidence=confidence,
            metadata={"audio_reading": audio_reading.to_dict()},
        )

        self._set_reading(reading)
        return reading

    def _notify_event(self, event: SensorEvent) -> None:
        """Notify event callbacks."""
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    async def start(self) -> None:
        """Start continuous audio monitoring."""
        self._running = True
        self._status = SensorStatus.ACTIVE

        while self._running:
            try:
                await self.read()
                await asyncio.sleep(self.config.audio_update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Audio sensor error: {e}")
                await asyncio.sleep(1.0)

    async def stop(self) -> None:
        """Stop audio monitoring."""
        self._running = False
        self._status = SensorStatus.INACTIVE
        logger.info(f"Audio sensor {self.sensor_id} stopped")

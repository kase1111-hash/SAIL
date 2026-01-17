"""
SAIL Voice Identification System

Provides speaker identification capabilities using voice embeddings
for automatic user recognition in the multi-user family system.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from src.users.base import (
    IdentificationMethod,
    UserEvent,
    UserEventType,
    UserManagerConfig,
    VoiceProfile,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Voice Embedding Provider
# ============================================================================


class VoiceEmbeddingProvider:
    """
    Base class for voice embedding providers.

    Generates speaker embeddings from audio samples for
    voice identification.
    """

    def __init__(self, model_name: str = "default"):
        self.model_name = model_name
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the embedding model."""
        raise NotImplementedError

    async def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> list[float]:
        """
        Generate embedding for audio sample.

        Args:
            audio: Audio samples as numpy array
            sample_rate: Audio sample rate

        Returns:
            Embedding vector as list of floats
        """
        raise NotImplementedError

    def get_embedding_dim(self) -> int:
        """Get dimension of embeddings."""
        raise NotImplementedError


class LocalVoiceEmbeddingProvider(VoiceEmbeddingProvider):
    """
    Local voice embedding provider.

    Uses resemblyzer or falls back to simple audio features
    if resemblyzer is not available.
    """

    def __init__(self, model_name: str = "default"):
        super().__init__(model_name)
        self._use_resemblyzer = False
        self._encoder = None
        self._embedding_dim = 256

    async def initialize(self) -> bool:
        """Initialize the embedding model."""
        try:
            from resemblyzer import VoiceEncoder
            self._encoder = VoiceEncoder()
            self._use_resemblyzer = True
            self._embedding_dim = 256
            self._initialized = True
            logger.info("Initialized resemblyzer voice encoder")
            return True
        except ImportError:
            logger.warning("resemblyzer not available, using simple audio features")
            self._use_resemblyzer = False
            self._embedding_dim = 128
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize voice embedding: {e}")
            return False

    async def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> list[float]:
        """Generate embedding for audio sample."""
        if not self._initialized:
            await self.initialize()

        if self._use_resemblyzer:
            return self._embed_resemblyzer(audio, sample_rate)
        else:
            return self._embed_simple(audio, sample_rate)

    def _embed_resemblyzer(self, audio: np.ndarray, sample_rate: int) -> list[float]:
        """Generate embedding using resemblyzer."""
        try:

            # Preprocess audio
            if sample_rate != 16000:
                # Resample to 16kHz
                import scipy.signal
                audio = scipy.signal.resample(audio, int(len(audio) * 16000 / sample_rate))

            # Normalize
            audio = audio.astype(np.float32)
            if audio.max() > 1.0:
                audio = audio / 32768.0

            # Generate embedding
            embedding = self._encoder.embed_utterance(audio)
            return embedding.tolist()

        except Exception as e:
            logger.error(f"Resemblyzer embedding failed: {e}")
            return self._embed_simple(audio, sample_rate)

    def _embed_simple(self, audio: np.ndarray, sample_rate: int) -> list[float]:
        """
        Generate simple audio feature embedding.

        This is a fallback when resemblyzer is not available.
        Uses basic audio features like MFCCs approximation.
        """
        try:
            # Ensure float32
            audio = audio.astype(np.float32)
            if audio.max() > 1.0:
                audio = audio / 32768.0

            # Frame the audio
            frame_length = int(0.025 * sample_rate)  # 25ms frames
            hop_length = int(0.010 * sample_rate)    # 10ms hop
            n_frames = max(1, (len(audio) - frame_length) // hop_length + 1)

            features = []

            for i in range(min(n_frames, 100)):  # Max 100 frames
                start = i * hop_length
                end = start + frame_length
                frame = audio[start:end]

                if len(frame) < frame_length:
                    frame = np.pad(frame, (0, frame_length - len(frame)))

                # Simple features
                energy = np.sum(frame ** 2)
                zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))

                # FFT-based features
                fft = np.abs(np.fft.rfft(frame))
                spectral_centroid = np.sum(np.arange(len(fft)) * fft) / (np.sum(fft) + 1e-8)
                spectral_rolloff = np.searchsorted(np.cumsum(fft), 0.85 * np.sum(fft))

                features.extend([energy, zcr, spectral_centroid / 1000, spectral_rolloff / 100])

            # Pad or truncate to fixed dimension
            features = features[:self._embedding_dim]
            while len(features) < self._embedding_dim:
                features.append(0.0)

            # Normalize
            features = np.array(features)
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm

            return features.tolist()

        except Exception as e:
            logger.error(f"Simple embedding failed: {e}")
            # Return hash-based fallback
            return self._hash_based_embedding(audio)

    def _hash_based_embedding(self, audio: np.ndarray) -> list[float]:
        """Generate hash-based pseudo-embedding as last resort."""
        audio_bytes = audio.tobytes()
        hash_val = hashlib.sha256(audio_bytes).hexdigest()

        # Convert hash to embedding
        embedding = []
        for i in range(0, min(len(hash_val), self._embedding_dim * 2), 2):
            val = int(hash_val[i:i+2], 16) / 255.0 - 0.5
            embedding.append(val)

        while len(embedding) < self._embedding_dim:
            embedding.append(0.0)

        return embedding[:self._embedding_dim]

    def get_embedding_dim(self) -> int:
        """Get dimension of embeddings."""
        return self._embedding_dim


# ============================================================================
# Voice Identification Result
# ============================================================================


@dataclass
class VoiceIdentificationResult:
    """Result from voice identification."""

    user_id: str | None = None
    user_name: str | None = None
    confidence: float = 0.0
    method: IdentificationMethod = IdentificationMethod.DEFAULT
    is_identified: bool = False

    # Match details
    matched_profile_id: str | None = None
    similarity_scores: dict[str, float] = field(default_factory=dict)

    # Timing
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "confidence": self.confidence,
            "method": self.method.value,
            "is_identified": self.is_identified,
            "matched_profile_id": self.matched_profile_id,
            "similarity_scores": self.similarity_scores,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# Voice Identification System
# ============================================================================


class VoiceIdentificationSystem:
    """
    System for identifying users by their voice.

    Manages voice profiles, enrollment, and real-time identification.
    """

    def __init__(
        self,
        config: UserManagerConfig | None = None,
        embedding_provider: VoiceEmbeddingProvider | None = None,
    ):
        self.config = config or UserManagerConfig()
        self._embedding_provider = embedding_provider or LocalVoiceEmbeddingProvider()

        # Voice profiles indexed by user_id
        self._profiles: dict[str, VoiceProfile] = {}

        # User name mapping
        self._user_names: dict[str, str] = {}

        # State
        self._initialized = False

        # Event callbacks
        self._event_callbacks: list[Callable[[UserEvent], None]] = []

        # Statistics
        self._identification_count = 0
        self._successful_identifications = 0

    async def initialize(self) -> bool:
        """Initialize the voice identification system."""
        try:
            success = await self._embedding_provider.initialize()
            if success:
                await self._load_profiles()
                self._initialized = True
                logger.info(f"Voice ID system initialized with {len(self._profiles)} profiles")
            return success
        except Exception as e:
            logger.error(f"Failed to initialize voice ID system: {e}")
            return False

    async def _load_profiles(self) -> None:
        """Load voice profiles from disk."""
        profiles_path = self.config.data_path / "voice_profiles"
        if not profiles_path.exists():
            return

        for profile_file in profiles_path.glob("*.json"):
            try:
                with open(profile_file) as f:
                    data = json.load(f)
                profile = VoiceProfile.from_dict(data)
                self._profiles[profile.user_id] = profile
            except Exception as e:
                logger.warning(f"Failed to load profile {profile_file}: {e}")

    async def _save_profile(self, profile: VoiceProfile) -> None:
        """Save a voice profile to disk."""
        profiles_path = self.config.data_path / "voice_profiles"
        profiles_path.mkdir(parents=True, exist_ok=True)

        profile_file = profiles_path / f"{profile.user_id}.json"
        with open(profile_file, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)

    def register_user(self, user_id: str, user_name: str) -> VoiceProfile:
        """
        Register a user for voice identification.

        Args:
            user_id: Unique user identifier
            user_name: User's display name

        Returns:
            VoiceProfile for enrollment
        """
        if user_id in self._profiles:
            return self._profiles[user_id]

        profile = VoiceProfile(
            user_id=user_id,
            required_samples=self.config.voice_enrollment_samples,
            min_confidence_threshold=self.config.voice_confidence_threshold,
        )
        self._profiles[user_id] = profile
        self._user_names[user_id] = user_name

        logger.info(f"Registered user {user_name} for voice identification")
        return profile

    async def enroll_sample(
        self,
        user_id: str,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> tuple[bool, str]:
        """
        Add an enrollment sample for a user.

        Args:
            user_id: User to enroll
            audio: Audio sample
            sample_rate: Audio sample rate

        Returns:
            Tuple of (success, message)
        """
        if user_id not in self._profiles:
            return False, "User not registered for voice identification"

        profile = self._profiles[user_id]

        if profile.enrollment_complete:
            return True, "Enrollment already complete"

        try:
            # Generate embedding
            embedding = await self._embedding_provider.embed(audio, sample_rate)

            # Add to profile
            profile.add_embedding(embedding)

            # Save profile
            await self._save_profile(profile)

            samples_remaining = profile.required_samples - profile.sample_count

            if profile.enrollment_complete:
                self._emit_event(UserEvent(
                    event_type=UserEventType.VOICE_ENROLLED,
                    user_id=user_id,
                    data={"sample_count": profile.sample_count},
                ))
                return True, "Voice enrollment complete!"
            else:
                return True, f"Sample recorded. {samples_remaining} more samples needed."

        except Exception as e:
            logger.error(f"Enrollment failed: {e}")
            return False, f"Failed to process audio sample: {e}"

    async def identify(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> VoiceIdentificationResult:
        """
        Identify a speaker from audio.

        Args:
            audio: Audio sample to identify
            sample_rate: Audio sample rate

        Returns:
            VoiceIdentificationResult
        """
        start_time = time.time()
        self._identification_count += 1

        result = VoiceIdentificationResult()

        if not self._initialized:
            await self.initialize()

        # Check if we have any enrolled profiles
        enrolled_profiles = {
            uid: p for uid, p in self._profiles.items()
            if p.is_enrolled and p.embeddings
        }

        if not enrolled_profiles:
            result.processing_time_ms = (time.time() - start_time) * 1000
            return result

        try:
            # Generate embedding for input audio
            query_embedding = await self._embedding_provider.embed(audio, sample_rate)
            query_vec = np.array(query_embedding)

            # Compare against all enrolled profiles
            best_match_id = None
            best_score = 0.0
            similarity_scores = {}

            for user_id, profile in enrolled_profiles.items():
                centroid = profile.get_centroid()
                if centroid is None:
                    continue

                # Calculate cosine similarity
                profile_vec = np.array(centroid)
                similarity = self._cosine_similarity(query_vec, profile_vec)
                similarity_scores[user_id] = similarity

                if similarity > best_score:
                    best_score = similarity
                    best_match_id = user_id

            result.similarity_scores = similarity_scores

            # Check if best match exceeds threshold
            if best_match_id and best_score >= self.config.voice_confidence_threshold:
                result.user_id = best_match_id
                result.user_name = self._user_names.get(best_match_id, "Unknown")
                result.confidence = best_score
                result.matched_profile_id = self._profiles[best_match_id].profile_id
                result.is_identified = True
                result.method = IdentificationMethod.VOICE

                self._successful_identifications += 1

                self._emit_event(UserEvent(
                    event_type=UserEventType.USER_IDENTIFIED,
                    user_id=best_match_id,
                    data={
                        "method": "voice",
                        "confidence": best_score,
                    },
                ))

        except Exception as e:
            logger.error(f"Identification failed: {e}")

        result.processing_time_ms = (time.time() - start_time) * 1000
        return result

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def get_profile(self, user_id: str) -> VoiceProfile | None:
        """Get voice profile for a user."""
        return self._profiles.get(user_id)

    def is_enrolled(self, user_id: str) -> bool:
        """Check if a user is enrolled for voice identification."""
        profile = self._profiles.get(user_id)
        return profile is not None and profile.is_enrolled

    def get_enrollment_status(self, user_id: str) -> dict[str, Any]:
        """Get enrollment status for a user."""
        profile = self._profiles.get(user_id)
        if profile is None:
            return {
                "registered": False,
                "enrolled": False,
                "sample_count": 0,
                "required_samples": self.config.voice_enrollment_samples,
            }

        return {
            "registered": True,
            "enrolled": profile.is_enrolled,
            "sample_count": profile.sample_count,
            "required_samples": profile.required_samples,
            "enrollment_complete": profile.enrollment_complete,
        }

    async def delete_profile(self, user_id: str) -> bool:
        """Delete a user's voice profile."""
        if user_id not in self._profiles:
            return False

        del self._profiles[user_id]
        if user_id in self._user_names:
            del self._user_names[user_id]

        # Delete from disk
        profile_file = self.config.data_path / "voice_profiles" / f"{user_id}.json"
        if profile_file.exists():
            profile_file.unlink()

        return True

    def register_event_callback(self, callback: Callable[[UserEvent], None]) -> None:
        """Register callback for voice ID events."""
        self._event_callbacks.append(callback)

    def _emit_event(self, event: UserEvent) -> None:
        """Emit a user event."""
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get voice identification statistics."""
        enrolled_count = sum(1 for p in self._profiles.values() if p.is_enrolled)

        return {
            "initialized": self._initialized,
            "total_profiles": len(self._profiles),
            "enrolled_profiles": enrolled_count,
            "identification_attempts": self._identification_count,
            "successful_identifications": self._successful_identifications,
            "success_rate": (
                self._successful_identifications / self._identification_count
                if self._identification_count > 0 else 0.0
            ),
        }


# ============================================================================
# Factory Functions
# ============================================================================


def create_voice_id_system(
    config: UserManagerConfig | None = None,
) -> VoiceIdentificationSystem:
    """Create a voice identification system."""
    return VoiceIdentificationSystem(config)


async def create_initialized_voice_id_system(
    config: UserManagerConfig | None = None,
) -> VoiceIdentificationSystem:
    """Create and initialize a voice identification system."""
    system = VoiceIdentificationSystem(config)
    await system.initialize()
    return system

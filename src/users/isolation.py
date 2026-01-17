"""
SAIL Per-User Data Isolation Layer

Provides data isolation between users for privacy protection.
Ensures each user's context, history, and preferences are
kept separate from other users.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from src.users.base import UserManagerConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Encryption Helpers
# ============================================================================


def derive_key(user_id: str, salt: bytes) -> bytes:
    """Derive an encryption key from user ID."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(user_id.encode()))
    return key


class UserDataEncryptor:
    """
    Handles encryption/decryption of per-user data.
    """

    def __init__(self, user_id: str, data_path: Path):
        self.user_id = user_id
        self.data_path = data_path
        self._fernet: Fernet | None = None
        self._salt: bytes | None = None

    def initialize(self) -> bool:
        """Initialize encryption for user."""
        try:
            salt_file = self.data_path / f"{self.user_id}.salt"

            if salt_file.exists():
                with open(salt_file, "rb") as f:
                    self._salt = f.read()
            else:
                self._salt = os.urandom(16)
                salt_file.parent.mkdir(parents=True, exist_ok=True)
                with open(salt_file, "wb") as f:
                    f.write(self._salt)

            key = derive_key(self.user_id, self._salt)
            self._fernet = Fernet(key)
            return True

        except Exception as e:
            logger.error(f"Failed to initialize encryption for {self.user_id}: {e}")
            return False

    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data."""
        if self._fernet is None:
            self.initialize()
        return self._fernet.encrypt(data)

    def decrypt(self, data: bytes) -> bytes:
        """Decrypt data."""
        if self._fernet is None:
            self.initialize()
        return self._fernet.decrypt(data)

    def encrypt_json(self, data: dict[str, Any]) -> bytes:
        """Encrypt JSON data."""
        json_bytes = json.dumps(data).encode()
        return self.encrypt(json_bytes)

    def decrypt_json(self, data: bytes) -> dict[str, Any]:
        """Decrypt JSON data."""
        json_bytes = self.decrypt(data)
        return json.loads(json_bytes.decode())


# ============================================================================
# User Data Store
# ============================================================================


@dataclass
class UserDataStore:
    """
    Isolated data store for a single user.

    Manages all data belonging to a specific user with
    optional encryption.
    """

    user_id: str
    data_path: Path
    encryption_enabled: bool = True

    # Encryptor (lazy initialized)
    _encryptor: UserDataEncryptor | None = field(default=None, repr=False)

    # Cached data
    _context_buffer: dict[str, Any] = field(default_factory=dict)
    _preferences: dict[str, Any] = field(default_factory=dict)
    _history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize the data store."""
        self.user_path = self.data_path / self.user_id
        self.user_path.mkdir(parents=True, exist_ok=True)

        if self.encryption_enabled:
            self._encryptor = UserDataEncryptor(self.user_id, self.data_path / "keys")
            self._encryptor.initialize()

    def save_context(self, context_id: str, data: dict[str, Any]) -> bool:
        """
        Save context data for the user.

        Args:
            context_id: Identifier for the context (e.g., "immediate", "session")
            data: Context data to save

        Returns:
            True if successful
        """
        try:
            context_file = self.user_path / f"context_{context_id}.dat"

            if self.encryption_enabled and self._encryptor:
                encrypted = self._encryptor.encrypt_json(data)
                with open(context_file, "wb") as f:
                    f.write(encrypted)
            else:
                with open(context_file, "w") as f:
                    json.dump(data, f)

            self._context_buffer[context_id] = data
            return True

        except Exception as e:
            logger.error(f"Failed to save context {context_id} for {self.user_id}: {e}")
            return False

    def load_context(self, context_id: str) -> dict[str, Any] | None:
        """
        Load context data for the user.

        Args:
            context_id: Identifier for the context

        Returns:
            Context data or None
        """
        # Check cache
        if context_id in self._context_buffer:
            return self._context_buffer[context_id]

        try:
            context_file = self.user_path / f"context_{context_id}.dat"

            if not context_file.exists():
                return None

            if self.encryption_enabled and self._encryptor:
                with open(context_file, "rb") as f:
                    encrypted = f.read()
                data = self._encryptor.decrypt_json(encrypted)
            else:
                with open(context_file) as f:
                    data = json.load(f)

            self._context_buffer[context_id] = data
            return data

        except Exception as e:
            logger.error(f"Failed to load context {context_id} for {self.user_id}: {e}")
            return None

    def save_preferences(self, preferences: dict[str, Any]) -> bool:
        """Save user preferences."""
        try:
            prefs_file = self.user_path / "preferences.dat"

            if self.encryption_enabled and self._encryptor:
                encrypted = self._encryptor.encrypt_json(preferences)
                with open(prefs_file, "wb") as f:
                    f.write(encrypted)
            else:
                with open(prefs_file, "w") as f:
                    json.dump(preferences, f)

            self._preferences = preferences
            return True

        except Exception as e:
            logger.error(f"Failed to save preferences for {self.user_id}: {e}")
            return False

    def load_preferences(self) -> dict[str, Any]:
        """Load user preferences."""
        if self._preferences:
            return self._preferences

        try:
            prefs_file = self.user_path / "preferences.dat"

            if not prefs_file.exists():
                return {}

            if self.encryption_enabled and self._encryptor:
                with open(prefs_file, "rb") as f:
                    encrypted = f.read()
                self._preferences = self._encryptor.decrypt_json(encrypted)
            else:
                with open(prefs_file) as f:
                    self._preferences = json.load(f)

            return self._preferences

        except Exception as e:
            logger.error(f"Failed to load preferences for {self.user_id}: {e}")
            return {}

    def append_history(self, entry: dict[str, Any]) -> bool:
        """Append an entry to user's history."""
        try:
            entry["timestamp"] = datetime.now().isoformat()
            self._history.append(entry)

            # Keep history bounded
            if len(self._history) > 1000:
                self._history = self._history[-500:]

            # Save periodically (every 10 entries)
            if len(self._history) % 10 == 0:
                self._save_history()

            return True

        except Exception as e:
            logger.error(f"Failed to append history for {self.user_id}: {e}")
            return False

    def _save_history(self) -> None:
        """Save history to disk."""
        try:
            history_file = self.user_path / "history.dat"

            data = {"entries": self._history}

            if self.encryption_enabled and self._encryptor:
                encrypted = self._encryptor.encrypt_json(data)
                with open(history_file, "wb") as f:
                    f.write(encrypted)
            else:
                with open(history_file, "w") as f:
                    json.dump(data, f)

        except Exception as e:
            logger.error(f"Failed to save history for {self.user_id}: {e}")

    def load_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Load user history."""
        if self._history:
            return self._history[-limit:]

        try:
            history_file = self.user_path / "history.dat"

            if not history_file.exists():
                return []

            if self.encryption_enabled and self._encryptor:
                with open(history_file, "rb") as f:
                    encrypted = f.read()
                data = self._encryptor.decrypt_json(encrypted)
            else:
                with open(history_file) as f:
                    data = json.load(f)

            self._history = data.get("entries", [])
            return self._history[-limit:]

        except Exception as e:
            logger.error(f"Failed to load history for {self.user_id}: {e}")
            return []

    def clear_all_data(self) -> bool:
        """Clear all data for the user."""
        try:
            import shutil
            if self.user_path.exists():
                shutil.rmtree(self.user_path)
            self._context_buffer.clear()
            self._preferences.clear()
            self._history.clear()
            return True
        except Exception as e:
            logger.error(f"Failed to clear data for {self.user_id}: {e}")
            return False


# ============================================================================
# Data Isolation Manager
# ============================================================================


class DataIsolationManager:
    """
    Manages isolated data stores for all users.
    """

    def __init__(self, config: UserManagerConfig | None = None):
        self.config = config or UserManagerConfig()
        self._stores: dict[str, UserDataStore] = {}
        self._current_store: UserDataStore | None = None

    def get_store(self, user_id: str) -> UserDataStore:
        """
        Get or create a data store for a user.

        Args:
            user_id: User ID

        Returns:
            UserDataStore for the user
        """
        if user_id not in self._stores:
            self._stores[user_id] = UserDataStore(
                user_id=user_id,
                data_path=self.config.data_path / "user_data",
                encryption_enabled=self.config.encryption_enabled,
            )
        return self._stores[user_id]

    def set_current_user(self, user_id: str) -> UserDataStore:
        """
        Set the current active user's store.

        Args:
            user_id: User ID

        Returns:
            UserDataStore for the user
        """
        self._current_store = self.get_store(user_id)
        return self._current_store

    def get_current_store(self) -> UserDataStore | None:
        """Get the current user's data store."""
        return self._current_store

    def save_context(self, context_id: str, data: dict[str, Any]) -> bool:
        """Save context for current user."""
        if self._current_store is None:
            return False
        return self._current_store.save_context(context_id, data)

    def load_context(self, context_id: str) -> dict[str, Any] | None:
        """Load context for current user."""
        if self._current_store is None:
            return None
        return self._current_store.load_context(context_id)

    def delete_user_data(self, user_id: str) -> bool:
        """Delete all data for a user."""
        store = self._stores.get(user_id)
        if store:
            success = store.clear_all_data()
            if success:
                del self._stores[user_id]
            return success
        return True

    def get_stats(self) -> dict[str, Any]:
        """Get isolation manager statistics."""
        return {
            "active_stores": len(self._stores),
            "current_user": self._current_store.user_id if self._current_store else None,
            "encryption_enabled": self.config.encryption_enabled,
        }


# ============================================================================
# Factory Functions
# ============================================================================


def create_isolation_manager(
    config: UserManagerConfig | None = None,
) -> DataIsolationManager:
    """Create a data isolation manager."""
    return DataIsolationManager(config)

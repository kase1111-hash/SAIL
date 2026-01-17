"""
SAIL Context Persistence

Provides SQLite-based persistence for context entries with optional encryption.
Supports session recovery and background knowledge storage.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import sqlite3
import threading
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from src.context.base import (
    ContextEntry,
    ContextSummary,
    ContextTier,
    EntryPriority,
    EntryType,
)

logger = logging.getLogger(__name__)


class EncryptionError(Exception):
    """Exception raised for encryption-related errors."""


class ContextEncryption:
    """
    Simple encryption for context data using Fernet-like approach.

    Uses AES encryption with PBKDF2 key derivation for stored context.
    Falls back to no encryption if cryptography is not available.
    """

    def __init__(self, key: bytes | None = None, password: str | None = None) -> None:
        """
        Initialize encryption.

        Args:
            key: Direct encryption key (32 bytes)
            password: Password to derive key from
        """
        self._fernet = None
        self._key: bytes | None = None

        if key:
            self._key = key
        elif password:
            self._key = self._derive_key(password)

        if self._key:
            self._init_cipher()

    def _derive_key(self, password: str, salt: bytes | None = None) -> bytes:
        """Derive encryption key from password using PBKDF2."""
        if salt is None:
            salt = b"SAIL_CONTEXT_SALT_v1"  # Static salt for deterministic key

        # Use PBKDF2 with SHA256
        key = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations=100000,
            dklen=32,
        )
        return key

    def _init_cipher(self) -> None:
        """Initialize the cipher with the key."""
        if self._key is None:
            return

        try:
            from cryptography.fernet import Fernet

            # Fernet requires base64-encoded 32-byte key
            fernet_key = base64.urlsafe_b64encode(self._key[:32])
            self._fernet = Fernet(fernet_key)
        except ImportError:
            logger.warning(
                "cryptography package not installed. "
                "Context will be stored without encryption."
            )
            self._fernet = None

    def encrypt(self, data: str) -> str:
        """
        Encrypt string data.

        Args:
            data: String to encrypt

        Returns:
            Encrypted string (base64 encoded)
        """
        if self._fernet is None:
            return data

        try:
            encrypted = self._fernet.encrypt(data.encode("utf-8"))
            return base64.urlsafe_b64encode(encrypted).decode("ascii")
        except Exception as e:
            raise EncryptionError(f"Encryption failed: {e}") from e

    def decrypt(self, data: str) -> str:
        """
        Decrypt string data.

        Args:
            data: Encrypted string (base64 encoded)

        Returns:
            Decrypted string
        """
        if self._fernet is None:
            return data

        try:
            encrypted = base64.urlsafe_b64decode(data.encode("ascii"))
            decrypted = self._fernet.decrypt(encrypted)
            return decrypted.decode("utf-8")
        except Exception as e:
            raise EncryptionError(f"Decryption failed: {e}") from e

    @property
    def is_enabled(self) -> bool:
        """Check if encryption is enabled."""
        return self._fernet is not None

    @staticmethod
    def generate_key() -> bytes:
        """Generate a random encryption key."""
        return os.urandom(32)


class ContextStore:
    """
    SQLite-based storage for context entries.

    Provides persistent storage with optional encryption for context
    entries, summaries, and session data.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        storage_path: Path | str,
        encryption: ContextEncryption | None = None,
    ) -> None:
        """
        Initialize context store.

        Args:
            storage_path: Path to storage directory
            encryption: Optional encryption handler
        """
        self.storage_path = Path(storage_path)
        self.encryption = encryption

        # Ensure directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Database path
        self._db_path = self.storage_path / "context.db"
        self._lock = threading.RLock()

        # Initialize database
        self._init_database()

    def _init_database(self) -> None:
        """Initialize the SQLite database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Create entries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entries (
                    entry_id TEXT PRIMARY KEY,
                    entry_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    metadata TEXT,
                    parent_id TEXT,
                    thread_id TEXT,
                    is_pinned INTEGER DEFAULT 0,
                    is_compressed INTEGER DEFAULT 0,
                    is_encrypted INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create summaries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS summaries (
                    summary_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    source_entry_ids TEXT,
                    time_range_start TEXT,
                    time_range_end TEXT,
                    entry_count INTEGER DEFAULT 0,
                    topics TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    entry_count INTEGER DEFAULT 0,
                    metadata TEXT
                )
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_entries_timestamp
                ON entries(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_entries_session
                ON entries(session_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_entries_tier
                ON entries(tier)
            """)

            # Schema version table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """)
            cursor.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
                (self.SCHEMA_VERSION,),
            )

            conn.commit()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with proper handling."""
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def save_entry(self, entry: ContextEntry) -> None:
        """
        Save a context entry to the database.

        Args:
            entry: Entry to save
        """
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()

            # Encrypt content if enabled
            content = entry.content
            is_encrypted = False
            if self.encryption and self.encryption.is_enabled:
                content = self.encryption.encrypt(content)
                is_encrypted = True

            # Serialize metadata
            metadata_json = json.dumps(entry.metadata) if entry.metadata else None

            cursor.execute("""
                INSERT OR REPLACE INTO entries (
                    entry_id, entry_type, content, timestamp, tier, priority,
                    user_id, session_id, metadata, parent_id, thread_id,
                    is_pinned, is_compressed, is_encrypted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.entry_id,
                entry.entry_type.value,
                content,
                entry.timestamp.isoformat(),
                entry.tier.value,
                entry.priority.value,
                entry.user_id,
                entry.session_id,
                metadata_json,
                entry.parent_id,
                entry.thread_id,
                1 if entry.is_pinned else 0,
                1 if entry.is_compressed else 0,
                1 if is_encrypted else 0,
            ))

            conn.commit()

    def save_entries(self, entries: list[ContextEntry]) -> None:
        """
        Save multiple entries in a batch.

        Args:
            entries: Entries to save
        """
        for entry in entries:
            self.save_entry(entry)

    def load_entry(self, entry_id: str) -> ContextEntry | None:
        """
        Load an entry by ID.

        Args:
            entry_id: Entry ID to load

        Returns:
            Entry or None
        """
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM entries WHERE entry_id = ?", (entry_id,))
            row = cursor.fetchone()

            if row:
                return self._row_to_entry(dict(row))
            return None

    def load_entries(
        self,
        session_id: str | None = None,
        tier: ContextTier | None = None,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[ContextEntry]:
        """
        Load entries matching criteria.

        Args:
            session_id: Filter by session
            tier: Filter by tier
            since: Only entries after this time
            limit: Maximum entries to return

        Returns:
            List of entries
        """
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM entries WHERE 1=1"
            params = []

            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)

            if tier:
                query += " AND tier = ?"
                params.append(tier.value)

            if since:
                query += " AND timestamp > ?"
                params.append(since.isoformat())

            query += " ORDER BY timestamp DESC"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [self._row_to_entry(dict(row)) for row in rows]

    def _row_to_entry(self, row: dict) -> ContextEntry:
        """Convert a database row to a ContextEntry."""
        content = row["content"]

        # Decrypt if needed
        if row.get("is_encrypted") and self.encryption:
            try:
                content = self.encryption.decrypt(content)
            except EncryptionError:
                logger.warning(f"Failed to decrypt entry {row['entry_id']}")

        # Parse metadata
        metadata = {}
        if row.get("metadata"):
            try:
                metadata = json.loads(row["metadata"])
            except json.JSONDecodeError:
                pass

        return ContextEntry(
            entry_id=row["entry_id"],
            entry_type=EntryType(row["entry_type"]),
            content=content,
            timestamp=datetime.fromisoformat(row["timestamp"]),
            tier=ContextTier(row["tier"]),
            priority=EntryPriority(row["priority"]),
            user_id=row.get("user_id"),
            session_id=row.get("session_id"),
            metadata=metadata,
            parent_id=row.get("parent_id"),
            thread_id=row.get("thread_id"),
            is_pinned=bool(row.get("is_pinned")),
            is_compressed=bool(row.get("is_compressed")),
            is_encrypted=bool(row.get("is_encrypted")),
        )

    def delete_entry(self, entry_id: str) -> bool:
        """
        Delete an entry.

        Args:
            entry_id: Entry to delete

        Returns:
            True if deleted
        """
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM entries WHERE entry_id = ?", (entry_id,))
            conn.commit()
            return cursor.rowcount > 0

    def delete_old_entries(self, before: datetime, tier: ContextTier | None = None) -> int:
        """
        Delete entries older than a given time.

        Args:
            before: Delete entries before this time
            tier: Optional tier filter

        Returns:
            Number of entries deleted
        """
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()

            query = "DELETE FROM entries WHERE timestamp < ? AND is_pinned = 0"
            params = [before.isoformat()]

            if tier:
                query += " AND tier = ?"
                params.append(tier.value)

            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount

    def save_summary(self, summary: ContextSummary) -> None:
        """
        Save a context summary.

        Args:
            summary: Summary to save
        """
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()

            content = summary.content
            if self.encryption and self.encryption.is_enabled:
                content = self.encryption.encrypt(content)

            cursor.execute("""
                INSERT OR REPLACE INTO summaries (
                    summary_id, content, timestamp, source_entry_ids,
                    time_range_start, time_range_end, entry_count, topics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                summary.summary_id,
                content,
                summary.timestamp.isoformat(),
                json.dumps(summary.source_entry_ids),
                summary.time_range_start.isoformat() if summary.time_range_start else None,
                summary.time_range_end.isoformat() if summary.time_range_end else None,
                summary.entry_count,
                json.dumps(summary.topics),
            ))

            conn.commit()

    def load_summaries(self, limit: int | None = None) -> list[ContextSummary]:
        """
        Load summaries.

        Args:
            limit: Maximum summaries to return

        Returns:
            List of summaries
        """
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM summaries ORDER BY timestamp DESC"
            if limit:
                query += f" LIMIT {limit}"

            cursor.execute(query)
            rows = cursor.fetchall()

            summaries = []
            for row in rows:
                row_dict = dict(row)
                content = row_dict["content"]

                if self.encryption:
                    try:
                        content = self.encryption.decrypt(content)
                    except EncryptionError:
                        pass

                summaries.append(ContextSummary(
                    summary_id=row_dict["summary_id"],
                    content=content,
                    timestamp=datetime.fromisoformat(row_dict["timestamp"]),
                    source_entry_ids=json.loads(row_dict.get("source_entry_ids") or "[]"),
                    time_range_start=datetime.fromisoformat(row_dict["time_range_start"])
                    if row_dict.get("time_range_start") else None,
                    time_range_end=datetime.fromisoformat(row_dict["time_range_end"])
                    if row_dict.get("time_range_end") else None,
                    entry_count=row_dict.get("entry_count", 0),
                    topics=json.loads(row_dict.get("topics") or "[]"),
                ))

            return summaries

    def start_session(self, session_id: str, user_id: str | None = None) -> None:
        """
        Record a new session start.

        Args:
            session_id: Session identifier
            user_id: Optional user identifier
        """
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO sessions (session_id, user_id, started_at)
                VALUES (?, ?, ?)
            """, (session_id, user_id, datetime.now().isoformat()))
            conn.commit()

    def end_session(self, session_id: str) -> None:
        """
        Record session end.

        Args:
            session_id: Session to end
        """
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()

            # Count entries
            cursor.execute(
                "SELECT COUNT(*) FROM entries WHERE session_id = ?",
                (session_id,),
            )
            count = cursor.fetchone()[0]

            # Update session
            cursor.execute("""
                UPDATE sessions
                SET ended_at = ?, entry_count = ?
                WHERE session_id = ?
            """, (datetime.now().isoformat(), count, session_id))
            conn.commit()

    def get_last_session_id(self) -> str | None:
        """Get the most recent session ID."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT session_id FROM sessions
                ORDER BY started_at DESC LIMIT 1
            """)
            row = cursor.fetchone()
            return row["session_id"] if row else None

    def get_session_entries(self, session_id: str) -> Iterator[ContextEntry]:
        """
        Get all entries for a session.

        Args:
            session_id: Session to retrieve

        Yields:
            Context entries
        """
        entries = self.load_entries(session_id=session_id)
        yield from entries

    def vacuum(self) -> None:
        """Optimize the database."""
        with self._lock, self._get_connection() as conn:
            conn.execute("VACUUM")

    def get_stats(self) -> dict:
        """Get storage statistics."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()

            stats = {
                "db_size_bytes": self._db_path.stat().st_size if self._db_path.exists() else 0,
            }

            # Entry counts by tier
            cursor.execute("""
                SELECT tier, COUNT(*) as count FROM entries GROUP BY tier
            """)
            stats["entries_by_tier"] = {row["tier"]: row["count"] for row in cursor.fetchall()}

            # Total entries
            cursor.execute("SELECT COUNT(*) FROM entries")
            stats["total_entries"] = cursor.fetchone()[0]

            # Summary count
            cursor.execute("SELECT COUNT(*) FROM summaries")
            stats["total_summaries"] = cursor.fetchone()[0]

            # Session count
            cursor.execute("SELECT COUNT(*) FROM sessions")
            stats["total_sessions"] = cursor.fetchone()[0]

            return stats

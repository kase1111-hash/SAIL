"""
SAIL Context Buffer System Tests

Comprehensive unit tests for the context buffer system components.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.context.base import (
    ContextConfig,
    ContextDepthMode,
    ContextEntry,
    ContextSummary,
    ContextTier,
    DepthModeConfig,
    EntryPriority,
    EntryType,
    get_depth_mode_config,
)
from src.context.buffer import (
    MultiTierBuffer,
    TierBuffer,
    TierStats,
)
from src.context.persistence import (
    ContextEncryption,
    ContextStore,
)
from src.context.manager import (
    ContextBufferManager,
    ContextManagerEvent,
    SessionInfo,
)


# ============================================================================
# Context Entry Tests
# ============================================================================


class TestContextEntry:
    """Tests for context entry dataclass."""

    def test_entry_creation(self):
        """Test creating a context entry."""
        entry = ContextEntry(
            entry_type=EntryType.USER_INPUT,
            content="Hello, world!",
        )

        assert entry.entry_type == EntryType.USER_INPUT
        assert entry.content == "Hello, world!"
        assert entry.tier == ContextTier.IMMEDIATE
        assert entry.priority == EntryPriority.NORMAL
        assert entry.entry_id is not None
        assert entry.is_pinned is False

    def test_entry_with_metadata(self):
        """Test entry with metadata."""
        entry = ContextEntry(
            entry_type=EntryType.SENSOR_DATA,
            content="Location: 40.7128, -74.0060",
            metadata={"lat": 40.7128, "lon": -74.0060},
        )

        assert entry.metadata["lat"] == 40.7128
        assert entry.metadata["lon"] == -74.0060

    def test_entry_to_dict(self):
        """Test converting entry to dictionary."""
        entry = ContextEntry(
            entry_type=EntryType.USER_INPUT,
            content="Test content",
            user_id="user123",
        )

        data = entry.to_dict()
        assert data["entry_type"] == "user_input"
        assert data["content"] == "Test content"
        assert data["user_id"] == "user123"

    def test_entry_from_dict(self):
        """Test creating entry from dictionary."""
        data = {
            "entry_id": "test-123",
            "entry_type": "assistant_response",
            "content": "Hello!",
            "timestamp": datetime.now().isoformat(),
            "tier": "immediate",
            "priority": 50,
        }

        entry = ContextEntry.from_dict(data)
        assert entry.entry_id == "test-123"
        assert entry.entry_type == EntryType.ASSISTANT_RESPONSE
        assert entry.content == "Hello!"

    def test_entry_age(self):
        """Test entry age calculation."""
        entry = ContextEntry(
            entry_type=EntryType.USER_INPUT,
            content="Test",
            timestamp=datetime.now() - timedelta(seconds=60),
        )

        assert 59 < entry.age_seconds < 61


class TestContextSummary:
    """Tests for context summary dataclass."""

    def test_summary_creation(self):
        """Test creating a context summary."""
        summary = ContextSummary(
            content="User discussed weather and travel plans",
            source_entry_ids=["id1", "id2", "id3"],
            entry_count=3,
            topics=["weather", "travel"],
        )

        assert summary.content == "User discussed weather and travel plans"
        assert len(summary.source_entry_ids) == 3
        assert summary.entry_count == 3
        assert "weather" in summary.topics

    def test_summary_to_dict(self):
        """Test converting summary to dictionary."""
        summary = ContextSummary(
            content="Test summary",
            topics=["topic1", "topic2"],
        )

        data = summary.to_dict()
        assert data["content"] == "Test summary"
        assert data["topics"] == ["topic1", "topic2"]


# ============================================================================
# Context Config Tests
# ============================================================================


class TestContextConfig:
    """Tests for context configuration."""

    def test_default_config(self):
        """Test default context configuration."""
        config = ContextConfig()

        assert config.immediate_duration_seconds == 300
        assert config.short_term_duration_seconds == 3600
        assert config.immediate_max_entries == 100
        assert config.encryption_enabled is True

    def test_custom_config(self):
        """Test custom context configuration."""
        config = ContextConfig(
            immediate_duration_seconds=600,
            short_term_duration_seconds=7200,
            immediate_max_entries=200,
        )

        assert config.immediate_duration_seconds == 600
        assert config.short_term_duration_seconds == 7200
        assert config.immediate_max_entries == 200


# ============================================================================
# Entry Types and Priorities Tests
# ============================================================================


class TestEntryType:
    """Tests for entry type enumeration."""

    def test_all_types_defined(self):
        """Test all entry types are defined."""
        expected_types = [
            "USER_INPUT", "ASSISTANT_RESPONSE", "SYSTEM_EVENT",
            "SENSOR_DATA", "INTERVENTION", "SUMMARY",
        ]
        for entry_type in expected_types:
            assert hasattr(EntryType, entry_type)

    def test_type_values(self):
        """Test entry type string values."""
        assert EntryType.USER_INPUT.value == "user_input"
        assert EntryType.ASSISTANT_RESPONSE.value == "assistant_response"
        assert EntryType.SYSTEM_EVENT.value == "system_event"


class TestEntryPriority:
    """Tests for entry priority enumeration."""

    def test_priority_ordering(self):
        """Test priority ordering is correct."""
        assert EntryPriority.CRITICAL > EntryPriority.HIGH
        assert EntryPriority.HIGH > EntryPriority.NORMAL
        assert EntryPriority.NORMAL > EntryPriority.LOW

    def test_priority_values(self):
        """Test priority numeric values."""
        assert EntryPriority.CRITICAL.value == 100
        assert EntryPriority.HIGH.value == 75
        assert EntryPriority.NORMAL.value == 50
        assert EntryPriority.LOW.value == 25


class TestContextTier:
    """Tests for context tier enumeration."""

    def test_all_tiers_defined(self):
        """Test all context tiers are defined."""
        expected_tiers = ["IMMEDIATE", "SHORT_TERM", "SESSION", "BACKGROUND"]
        for tier in expected_tiers:
            assert hasattr(ContextTier, tier)

    def test_tier_values(self):
        """Test tier string values."""
        assert ContextTier.IMMEDIATE.value == "immediate"
        assert ContextTier.SHORT_TERM.value == "short_term"
        assert ContextTier.SESSION.value == "session"


# ============================================================================
# Depth Mode Tests
# ============================================================================


class TestContextDepthMode:
    """Tests for context depth mode enumeration."""

    def test_all_modes_defined(self):
        """Test all depth modes are defined."""
        expected_modes = ["MINIMAL", "DRIVING", "CONVERSATION", "CRISIS"]
        for mode in expected_modes:
            assert hasattr(ContextDepthMode, mode)

    def test_mode_values(self):
        """Test depth mode string values."""
        assert ContextDepthMode.MINIMAL.value == "minimal"
        assert ContextDepthMode.DRIVING.value == "driving"
        assert ContextDepthMode.CONVERSATION.value == "conversation"
        assert ContextDepthMode.CRISIS.value == "crisis"


class TestDepthModeConfig:
    """Tests for depth mode configuration."""

    def test_get_depth_mode_config(self):
        """Test getting depth mode configuration."""
        config = get_depth_mode_config(ContextDepthMode.CONVERSATION)

        assert config.mode == ContextDepthMode.CONVERSATION
        assert config.immediate_retention_factor == 1.0
        assert config.max_context_tokens == 4000

    def test_crisis_mode_config(self):
        """Test crisis mode has higher retention."""
        config = get_depth_mode_config(ContextDepthMode.CRISIS)

        assert config.mode == ContextDepthMode.CRISIS
        assert config.immediate_retention_factor > 1.0  # Higher retention
        assert config.max_context_tokens > 4000  # More context

    def test_minimal_mode_config(self):
        """Test minimal mode has lower retention."""
        config = get_depth_mode_config(ContextDepthMode.MINIMAL)

        assert config.mode == ContextDepthMode.MINIMAL
        assert config.immediate_retention_factor < 1.0
        assert config.max_context_tokens < 4000


# ============================================================================
# Tier Buffer Tests
# ============================================================================


class TestTierBuffer:
    """Tests for single tier buffer."""

    def test_buffer_creation(self):
        """Test creating a tier buffer."""
        buffer = TierBuffer(
            tier=ContextTier.IMMEDIATE,
            max_age_seconds=300,
            max_entries=100,
        )

        assert buffer.tier == ContextTier.IMMEDIATE
        assert len(buffer) == 0

    def test_add_and_get_entry(self):
        """Test adding and retrieving entries."""
        buffer = TierBuffer(
            tier=ContextTier.IMMEDIATE,
            max_age_seconds=300,
            max_entries=100,
        )

        entry = ContextEntry(
            entry_type=EntryType.USER_INPUT,
            content="Test",
        )
        buffer.add(entry)

        assert len(buffer) == 1

        retrieved = buffer.get(entry.entry_id)
        assert retrieved is not None
        assert retrieved.content == "Test"

    def test_remove_entry(self):
        """Test removing an entry."""
        buffer = TierBuffer(
            tier=ContextTier.IMMEDIATE,
            max_age_seconds=300,
            max_entries=100,
        )

        entry = ContextEntry(content="Test")
        buffer.add(entry)
        assert len(buffer) == 1

        removed = buffer.remove(entry.entry_id)
        assert removed is not None
        assert len(buffer) == 0

    def test_pin_entry(self):
        """Test pinning an entry."""
        buffer = TierBuffer(
            tier=ContextTier.IMMEDIATE,
            max_age_seconds=300,
            max_entries=100,
        )

        entry = ContextEntry(content="Important")
        buffer.add(entry)

        assert buffer.pin_entry(entry.entry_id) is True

        retrieved = buffer.get(entry.entry_id)
        assert retrieved.is_pinned is True

    def test_get_entries_with_filter(self):
        """Test getting entries with filters."""
        buffer = TierBuffer(
            tier=ContextTier.IMMEDIATE,
            max_age_seconds=300,
            max_entries=100,
        )

        # Add entries with different priorities
        for i, priority in enumerate([EntryPriority.LOW, EntryPriority.NORMAL, EntryPriority.HIGH]):
            entry = ContextEntry(
                content=f"Entry {i}",
                priority=priority,
            )
            buffer.add(entry)

        # Filter by minimum priority
        high_priority = buffer.get_entries(min_priority=EntryPriority.HIGH)
        assert len(high_priority) == 1

        normal_and_above = buffer.get_entries(min_priority=EntryPriority.NORMAL)
        assert len(normal_and_above) == 2

    def test_clear_buffer(self):
        """Test clearing all entries."""
        buffer = TierBuffer(
            tier=ContextTier.IMMEDIATE,
            max_age_seconds=300,
            max_entries=100,
        )

        for i in range(5):
            buffer.add(ContextEntry(content=f"Entry {i}"))

        assert len(buffer) == 5

        cleared = buffer.clear()
        assert len(cleared) == 5
        assert len(buffer) == 0

    def test_get_stats(self):
        """Test getting buffer statistics."""
        buffer = TierBuffer(
            tier=ContextTier.IMMEDIATE,
            max_age_seconds=300,
            max_entries=100,
        )

        buffer.add(ContextEntry(content="Test 1"))
        buffer.add(ContextEntry(content="Test 2", is_pinned=True))

        stats = buffer.get_stats()
        assert stats.tier == ContextTier.IMMEDIATE
        assert stats.entry_count == 2
        assert stats.pinned_count == 1


# ============================================================================
# Multi-Tier Buffer Tests
# ============================================================================


class TestMultiTierBuffer:
    """Tests for multi-tier buffer."""

    def test_buffer_creation(self):
        """Test creating a multi-tier buffer."""
        buffer = MultiTierBuffer()

        assert buffer.depth_mode == ContextDepthMode.CONVERSATION
        assert buffer.total_entries == 0
        assert buffer.is_crisis_mode is False

    def test_add_entry(self):
        """Test adding entries to buffer."""
        buffer = MultiTierBuffer()

        entry = buffer.add_entry(
            content="Hello",
            entry_type=EntryType.USER_INPUT,
        )

        assert entry.content == "Hello"
        assert entry.tier == ContextTier.IMMEDIATE
        assert buffer.total_entries == 1

    def test_add_user_input(self):
        """Test convenience method for user input."""
        buffer = MultiTierBuffer()

        entry = buffer.add_user_input("Test input")
        assert entry.entry_type == EntryType.USER_INPUT

    def test_add_assistant_response(self):
        """Test convenience method for assistant response."""
        buffer = MultiTierBuffer()

        entry = buffer.add_assistant_response("Test response")
        assert entry.entry_type == EntryType.ASSISTANT_RESPONSE

    def test_get_context(self):
        """Test getting context entries."""
        buffer = MultiTierBuffer()

        # Add some entries
        buffer.add_user_input("User message 1")
        buffer.add_assistant_response("Assistant response 1")
        buffer.add_user_input("User message 2")

        entries = buffer.get_context(max_entries=10)
        assert len(entries) == 3

    def test_get_conversation_context(self):
        """Test getting conversation context."""
        buffer = MultiTierBuffer()

        buffer.add_user_input("Hello")
        buffer.add_assistant_response("Hi there!")
        buffer.add_user_input("How are you?")
        buffer.add_assistant_response("I'm doing well!")

        # Get last 2 turns (4 entries)
        conversation = buffer.get_conversation_context(max_turns=2)

        # Should be oldest first for chat format
        assert len(conversation) >= 2

    def test_set_depth_mode(self):
        """Test changing depth mode."""
        buffer = MultiTierBuffer()

        assert buffer.depth_mode == ContextDepthMode.CONVERSATION

        buffer.set_depth_mode(ContextDepthMode.DRIVING)
        assert buffer.depth_mode == ContextDepthMode.DRIVING

    def test_crisis_mode(self):
        """Test entering and exiting crisis mode."""
        buffer = MultiTierBuffer()

        # Add entry before crisis
        buffer.add_user_input("Before crisis")

        # Enter crisis mode
        buffer.enter_crisis_mode()
        assert buffer.is_crisis_mode is True
        assert buffer.depth_mode == ContextDepthMode.CRISIS

        # Entry added during crisis should be pinned
        entry = buffer.add_user_input("During crisis")
        assert entry.is_pinned is True

        # Exit crisis mode
        buffer.exit_crisis_mode()
        assert buffer.is_crisis_mode is False

    def test_get_stats(self):
        """Test getting buffer statistics."""
        buffer = MultiTierBuffer()

        buffer.add_user_input("Test 1")
        buffer.add_user_input("Test 2")

        stats = buffer.get_stats()
        assert "immediate" in stats
        assert stats["immediate"].entry_count == 2


# ============================================================================
# Persistence Tests
# ============================================================================


class TestContextEncryption:
    """Tests for context encryption."""

    def test_encryption_with_password(self):
        """Test encryption with password."""
        # Skip if cffi backend isn't available (required for cryptography)
        pytest.importorskip("_cffi_backend")

        encryption = ContextEncryption(password="test_password")

        # Only test if cryptography is available
        if encryption.is_enabled:
            plaintext = "Secret message"
            encrypted = encryption.encrypt(plaintext)
            decrypted = encryption.decrypt(encrypted)

            assert encrypted != plaintext
            assert decrypted == plaintext

    def test_encryption_disabled(self):
        """Test encryption when disabled."""
        encryption = ContextEncryption()  # No key or password

        # Should pass through without encryption
        text = "Plain text"
        result = encryption.encrypt(text)
        assert result == text

    def test_generate_key(self):
        """Test key generation."""
        key = ContextEncryption.generate_key()
        assert len(key) == 32


class TestContextStore:
    """Tests for context persistence store."""

    def test_store_creation(self):
        """Test creating a context store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ContextStore(storage_path=tmpdir)
            assert store.storage_path == Path(tmpdir)

    def test_save_and_load_entry(self):
        """Test saving and loading an entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ContextStore(storage_path=tmpdir)

            entry = ContextEntry(
                entry_type=EntryType.USER_INPUT,
                content="Test content",
                user_id="user123",
            )

            store.save_entry(entry)
            loaded = store.load_entry(entry.entry_id)

            assert loaded is not None
            assert loaded.content == "Test content"
            assert loaded.user_id == "user123"

    def test_load_entries_by_session(self):
        """Test loading entries by session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ContextStore(storage_path=tmpdir)

            # Save entries with different sessions
            entry1 = ContextEntry(content="Session A", session_id="session-a")
            entry2 = ContextEntry(content="Session B", session_id="session-b")

            store.save_entry(entry1)
            store.save_entry(entry2)

            # Load session A only
            entries = store.load_entries(session_id="session-a")
            assert len(entries) == 1
            assert entries[0].content == "Session A"

    def test_delete_entry(self):
        """Test deleting an entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ContextStore(storage_path=tmpdir)

            entry = ContextEntry(content="To delete")
            store.save_entry(entry)

            assert store.load_entry(entry.entry_id) is not None

            deleted = store.delete_entry(entry.entry_id)
            assert deleted is True
            assert store.load_entry(entry.entry_id) is None

    def test_save_and_load_summary(self):
        """Test saving and loading summaries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ContextStore(storage_path=tmpdir)

            summary = ContextSummary(
                content="Test summary content",
                topics=["topic1", "topic2"],
            )

            store.save_summary(summary)
            summaries = store.load_summaries()

            assert len(summaries) == 1
            assert summaries[0].content == "Test summary content"
            assert "topic1" in summaries[0].topics

    def test_session_management(self):
        """Test session start and end."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ContextStore(storage_path=tmpdir)

            store.start_session("session-123", user_id="user-456")
            store.end_session("session-123")

            last_session = store.get_last_session_id()
            assert last_session == "session-123"

    def test_get_stats(self):
        """Test getting store statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ContextStore(storage_path=tmpdir)

            store.save_entry(ContextEntry(content="Test"))

            stats = store.get_stats()
            assert stats["total_entries"] == 1
            assert "db_size_bytes" in stats


# ============================================================================
# Context Manager Tests
# ============================================================================


class TestContextManagerEvent:
    """Tests for context manager events."""

    def test_event_creation(self):
        """Test creating a context manager event."""
        event = ContextManagerEvent(
            event_type="entry_added",
            data={"entry_id": "test-123"},
        )

        assert event.event_type == "entry_added"
        assert event.data["entry_id"] == "test-123"


class TestSessionInfo:
    """Tests for session info dataclass."""

    def test_session_creation(self):
        """Test creating session info."""
        session = SessionInfo(
            session_id="session-123",
            user_id="user-456",
        )

        assert session.session_id == "session-123"
        assert session.user_id == "user-456"
        assert session.entry_count == 0
        assert session.is_crisis is False


class TestContextBufferManager:
    """Tests for context buffer manager."""

    def test_manager_creation(self):
        """Test creating a context buffer manager."""
        manager = ContextBufferManager()

        assert manager.is_running is False
        assert manager.session_id is None

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        """Test starting and stopping the manager."""
        manager = ContextBufferManager()

        # Create a config without persistence for testing
        manager._config.persist_sessions = False

        session_id = await manager.start(user_id="test-user")
        assert manager.is_running is True
        assert session_id is not None
        assert manager.session_id == session_id

        await manager.stop()
        assert manager.is_running is False

    @pytest.mark.asyncio
    async def test_add_entries(self):
        """Test adding entries through manager."""
        manager = ContextBufferManager()
        manager._config.persist_sessions = False

        await manager.start()

        entry = manager.add_user_input("Hello")
        assert entry.entry_type == EntryType.USER_INPUT
        assert entry.content == "Hello"

        response = manager.add_assistant_response("Hi there!")
        assert response.entry_type == EntryType.ASSISTANT_RESPONSE

        await manager.stop()

    @pytest.mark.asyncio
    async def test_get_context(self):
        """Test getting context from manager."""
        manager = ContextBufferManager()
        manager._config.persist_sessions = False

        await manager.start()

        manager.add_user_input("Message 1")
        manager.add_assistant_response("Response 1")
        manager.add_user_input("Message 2")

        context = manager.get_context(max_entries=10)
        assert len(context) == 3

        await manager.stop()

    @pytest.mark.asyncio
    async def test_format_for_llm(self):
        """Test formatting context for LLM."""
        manager = ContextBufferManager()
        manager._config.persist_sessions = False

        await manager.start()

        manager.add_user_input("Hello")
        manager.add_assistant_response("Hi there!")

        formatted = manager.format_for_llm()
        assert "User:" in formatted
        assert "Assistant:" in formatted

        await manager.stop()

    @pytest.mark.asyncio
    async def test_depth_mode_change(self):
        """Test changing depth mode."""
        manager = ContextBufferManager()
        manager._config.persist_sessions = False

        await manager.start()

        assert manager.depth_mode == ContextDepthMode.CONVERSATION

        manager.set_depth_mode(ContextDepthMode.DRIVING)
        assert manager.depth_mode == ContextDepthMode.DRIVING

        await manager.stop()

    @pytest.mark.asyncio
    async def test_crisis_mode(self):
        """Test crisis mode through manager."""
        manager = ContextBufferManager()
        manager._config.persist_sessions = False

        await manager.start()

        assert manager.is_crisis_mode is False

        manager.enter_crisis_mode()
        assert manager.is_crisis_mode is True
        assert manager.depth_mode == ContextDepthMode.CRISIS

        manager.exit_crisis_mode()
        assert manager.is_crisis_mode is False

        await manager.stop()

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting manager statistics."""
        manager = ContextBufferManager()
        manager._config.persist_sessions = False

        await manager.start()

        manager.add_user_input("Test")

        stats = manager.get_stats()
        assert "buffer" in stats
        assert "session" in stats
        assert stats["session"]["entry_count"] == 1

        await manager.stop()

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test using manager as async context manager."""
        manager = ContextBufferManager()
        manager._config.persist_sessions = False

        async with manager:
            assert manager.is_running is True
            manager.add_user_input("Test")

        assert manager.is_running is False
